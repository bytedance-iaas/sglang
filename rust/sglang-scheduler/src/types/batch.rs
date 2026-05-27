// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Scheduler-side batch types.
//!
//! `ScheduleBatch` is the mutable in-scheduler batch; `ModelWorkerBatchView`
//! is the read-only snapshot we serialise to the wire (`ForwardBatchGenerationReq`).
//!
//! ## Status
//!
//! The shapes here are **deliberately minimal** — the Python
//! `ScheduleBatch` has ~80 fields covering every feature SGLang has ever
//! supported (encoder-decoder, multimodal, mamba, hisparse, dllm,
//! grammar, …).  Porting them faithfully is a separate effort; the
//! point of this stub is to lock down the *core* shape so the rest of
//! the scheduler can reference `ScheduleBatch` while we add fields
//! one feature at a time.
//!
//! See `python/sglang/srt/managers/schedule_batch.py` for the full
//! Python class.

use std::sync::Arc;

use crate::types::{forward_mode::ForwardMode, req::Req};
use crate::wire::{
    ModelWorkerBatchPayload, SamplingBatchInfoPayload, TensorIPC, forward_mode_wire,
};

/// Mutable in-scheduler batch.  Tracks the running set of requests and
/// the shape of the next forward pass.
#[derive(Debug)]
pub struct ScheduleBatch {
    /// All requests currently in this batch.  Indices into this Vec line
    /// up with `req_pool_indices[i]` etc.
    pub reqs: Vec<Arc<std::sync::RwLock<Req>>>,

    /// `ReqToTokenPool` slot for each req — index into the scheduler's
    /// `req_to_token` table.
    pub req_pool_indices: Vec<u32>,

    /// Per-req current sequence length (number of tokens whose KV is
    /// allocated on the GPU).  Stays in lock-step with each req's
    /// `kv_allocated_len`.
    pub seq_lens: Vec<i32>,

    /// Per-req prefill input length at admission.  Used by the radix
    /// cache for prefix accounting.
    pub orig_seq_lens: Vec<i32>,

    /// Token ids the model should consume on the NEXT forward.  For
    /// decode this is `[last sampled token per req]`.  For extend this
    /// is the prefill tokens concatenated.
    pub input_ids: Vec<i32>,

    /// Forward-pass mode (set by `prepare_for_decode` / `prepare_for_extend`).
    pub forward_mode: ForwardMode,

    /// Slot indices the radix cache or finished-req path freed since
    /// the last forward; shipped as `ModelWorkerBatch.indices_to_free`
    /// on the next send.  Drained on every batch build.
    pub indices_to_free: Vec<i64>,

    // ──────────────────────────────────────────────────────────────────
    // Feature flags / aggregates derived from `reqs`.  Mirror of
    // `ScheduleBatch.has_*` fields — recomputed when the composition
    // changes.
    // ──────────────────────────────────────────────────────────────────
    pub return_logprob: bool,
    pub has_grammar: bool,
    pub has_stream: bool,
}

impl ScheduleBatch {
    pub fn new(forward_mode: ForwardMode) -> Self {
        Self {
            reqs: Vec::new(),
            req_pool_indices: Vec::new(),
            seq_lens: Vec::new(),
            orig_seq_lens: Vec::new(),
            input_ids: Vec::new(),
            forward_mode,
            indices_to_free: Vec::new(),
            return_logprob: false,
            has_grammar: false,
            has_stream: false,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.reqs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.reqs.is_empty()
    }

    /// Remove all reqs whose `finished()` returns true.  Mirror of
    /// `ScheduleBatch.filter_batch` in `schedule_batch.py`, simplified
    /// to the common case (no `chunked_req_to_exclude`).
    pub fn filter_finished(&mut self) {
        let keep: Vec<bool> = self
            .reqs
            .iter()
            .map(|r| !r.read().unwrap().finished())
            .collect();
        if keep.iter().all(|b| *b) {
            return;
        }
        Self::retain_by_mask(&mut self.reqs, &keep);
        Self::retain_indexed(&mut self.req_pool_indices, &keep);
        Self::retain_indexed(&mut self.seq_lens, &keep);
        Self::retain_indexed(&mut self.orig_seq_lens, &keep);
        // input_ids may not be 1-per-req in extend mode — leave it to
        // the next `prepare_for_*` call to rebuild.
        if self.forward_mode.is_decode() {
            Self::retain_indexed(&mut self.input_ids, &keep);
        }
    }

    fn retain_by_mask<T>(v: &mut Vec<T>, keep: &[bool]) {
        let mut idx = 0;
        v.retain(|_| {
            let k = keep[idx];
            idx += 1;
            k
        });
    }

    fn retain_indexed<T: Copy>(v: &mut Vec<T>, keep: &[bool]) {
        let mut idx = 0;
        v.retain(|_| {
            let k = keep[idx];
            idx += 1;
            k
        });
    }

    /// Build the wire-shaped `ModelWorkerBatchPayload` the worker
    /// expects under `ForwardBatchGenerationReq.batch`.
    ///
    /// `vocab_size` is required for the sampling-info payload (it
    /// defaults from the worker handshake's model config — wire it
    /// through from the event loop).  `req_pool` is consulted when
    /// any req has a non-zero `prefix_len_from_cache`: the slot row
    /// (size `max_context_len`) is snapshot into `req_to_token_cpu`
    /// so the worker can write the cached prefix's KV positions back
    /// into `req_to_token_gpu` before its forward pass.  Pass `None`
    /// when the caller is sure no req has a cached prefix (tests).
    pub fn to_model_worker_batch_payload(
        &self,
        vocab_size: i32,
        device: &str,
        req_pool: Option<&crate::memory::ReqToTokenPool>,
    ) -> ModelWorkerBatchPayload {
        let forward_mode = match self.forward_mode {
            ForwardMode::Extend => forward_mode_wire::EXTEND,
            ForwardMode::Decode => forward_mode_wire::DECODE,
            ForwardMode::Mixed => forward_mode_wire::MIXED,
            ForwardMode::Idle => forward_mode_wire::IDLE,
            ForwardMode::Prebuilt => forward_mode_wire::PREBUILT,
            ForwardMode::SplitPrefill => forward_mode_wire::SPLIT_PREFILL,
            ForwardMode::TargetVerify => forward_mode_wire::TARGET_VERIFY,
            ForwardMode::DraftExtend => forward_mode_wire::DRAFT_EXTEND,
        };
        let bs = self.batch_size();
        let req_pool_indices_i32: Vec<i32> =
            self.req_pool_indices.iter().map(|&v| v as i32).collect();
        let seq_lens_sum: i64 = self.seq_lens.iter().map(|&v| v as i64).sum();

        // Extend fields — populated only when forward_mode is extend/mixed.
        let extend_num_tokens = if self.forward_mode.is_extend() {
            Some(self.input_ids.len() as i64)
        } else {
            None
        };
        // Per-req cached prefix length — set by the batch builder when
        // the radix cache scores the admission.
        let prefix_lens_per_req: Vec<i32> = self
            .reqs
            .iter()
            .map(|r| r.read().unwrap().prefix_len_from_cache as i32)
            .collect();

        // `extend_seq_lens` on the wire is MISLEADINGLY named — Python
        // populates it from `self.extend_lens` (per-req NEW token count),
        // NOT `self.seq_lens` (full prompt length).  See
        // `schedule_batch.py:2515` (`extend_seq_lens = self.extend_lens`).
        //
        // The worker's `compute_position_kernel` writes `extend_seq_lens[i]`
        // positions per req into a `positions` tensor sized
        // `extend_num_tokens = sum(seq_lens - prefix_lens)`.  Sending the
        // full `seq_lens` here when any req has a cached prefix overruns
        // the `positions` tensor → corrupt position indices → attention
        // attends at out-of-range positions → CUDA `IndexKernel.cu:111
        // index out of bounds`.  The first request always works because
        // an empty cache makes `seq_lens == extend_lens` for every req;
        // any subsequent request that shares even a single token with
        // a cached entry (BOS, system prompt, chat template) trips this.
        let extend_seq_lens = if self.forward_mode.is_extend() {
            Some(
                self.seq_lens
                    .iter()
                    .zip(prefix_lens_per_req.iter())
                    .map(|(&s, &p)| s - p)
                    .collect::<Vec<i32>>(),
            )
        } else {
            None
        };
        let extend_prefix_lens = if self.forward_mode.is_extend() {
            Some(prefix_lens_per_req.clone())
        } else {
            None
        };

        let multimodal_inputs: Option<Vec<Option<rmpv::Value>>> = Some(vec![None; bs]);
        let encoder_cached: Option<Vec<bool>> = if bs > 0 { Some(vec![true; bs]) } else { None };
        let lora_ids: Option<Vec<Option<String>>> = Some(vec![None; bs]);

        let rids: Vec<String> = self
            .reqs
            .iter()
            .map(|r| r.read().unwrap().rid.clone())
            .collect();

        ModelWorkerBatchPayload {
            forward_mode,
            // Python's `ModelWorkerBatch.input_ids` is `torch.int64`
            // (see `schedule_batch.py:1596` —
            // `torch.tensor(input_ids, dtype=torch.int64, …)`).
            // `nn.Embedding(input_ids)` strictly requires Long; sending
            // int32 silently produces garbage embeddings, which then
            // feed the model and cause CUDA index OOB further downstream.
            input_ids: TensorIPC::from_i64(
                &self.input_ids.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            ),
            req_pool_indices: TensorIPC::from_i64(
                &req_pool_indices_i32
                    .iter()
                    .map(|&v| v as i64)
                    .collect::<Vec<_>>(),
            ),
            seq_lens: TensorIPC::from_i64(
                &self.seq_lens.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            ),
            out_cache_loc: None, // deferred to GPU
            seq_lens_cpu: Some(TensorIPC::from_i64(
                &self.seq_lens.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            )),
            seq_lens_sum,
            return_logprob: self.return_logprob,
            top_logprobs_nums: None,
            token_ids_logprobs: None,
            global_num_tokens: {
                // Single-DP fallback — see scheduler::dp_attn for the
                // multi-DP path (not ported yet).
                let dp = crate::scheduler::dp_attn::prepare_local_dp_attention_sync(self);
                Some(dp.global_num_tokens)
            },
            global_num_tokens_for_logprob: {
                let dp = crate::scheduler::dp_attn::prepare_local_dp_attention_sync(self);
                Some(dp.global_num_tokens_for_logprob)
            },
            is_extend_in_batch: self.forward_mode.is_extend(),
            all_extend_in_batch: self.forward_mode == ForwardMode::Extend,
            can_run_dp_cuda_graph: false,
            tbo_split_seq_index: None,
            global_forward_mode: Some({
                let dp = crate::scheduler::dp_attn::prepare_local_dp_attention_sync(self);
                dp.global_forward_mode
            }),
            extend_num_tokens,
            extend_seq_lens,
            extend_prefix_lens,
            extend_logprob_start_lens: None,
            extend_input_logprob_token_ids: None,
            sampling_info: {
                let views: Vec<crate::wire::model_worker_batch::SamplingParamsView> = self
                    .reqs
                    .iter()
                    .map(|r| {
                        let r = r.read().unwrap();
                        crate::wire::model_worker_batch::SamplingParamsView::from_sampling_params(
                            &r.sampling_params,
                        )
                    })
                    .collect();
                SamplingBatchInfoPayload::from_sampling_params(
                    views.into_iter(),
                    vocab_size,
                    device,
                )
            },
            multimodal_inputs,
            encoder_cached,
            encoder_lens: None,
            encoder_lens_cpu: None,
            encoder_out_cache_loc: None,
            lora_ids,
            orig_seq_lens: if self.orig_seq_lens.is_empty() {
                None
            } else {
                // Python expects `torch.int32` (see schedule_batch.py:1709).
                // Sending int64 would mismatch the worker's dtype check and
                // break downstream indexing kernels.
                Some(TensorIPC::from_i32(&self.orig_seq_lens))
            },
            input_embeds: None,
            replace_embeds: None,
            replace_positions: None,
            ne_token_table: None,
            token_type_ids: None,
            spec_algorithm: None,
            spec_info: None,
            // Mirror of `schedule_batch.py:2607` — Python's logic is:
            //   FULL if any req wants hidden states
            //   else (spec_info.capture_hidden_mode if spec_info else NULL)
            // The spec-info branch is a TODO until speculative decoding
            // is ported; for now any spec_info implies NULL.
            capture_hidden_mode: {
                use crate::wire::capture_hidden_mode_wire as chm;
                let any_return_hidden = self
                    .reqs
                    .iter()
                    .any(|r| r.read().unwrap().return_hidden_states);
                Some(if any_return_hidden {
                    chm::FULL
                } else {
                    chm::NULL
                })
            },
            hicache_consumer_index: -1,
            dimensions: None,
            return_pooled_hidden_states: false,
            is_prefill_only: false,
            multi_item_delimiter_indices: None,
            dllm_block_offsets: None,
            dllm_config: None,
            reqs: None,
            has_grammar: self.has_grammar,
            rids: Some(rids),
            return_hidden_states_before_norm: false,
            mamba_track_indices: None,
            mamba_track_mask: None,
            mamba_track_seqlens: None,
            req_to_token_cpu: build_req_to_token_cpu(self, req_pool),
            indices_to_free: if self.indices_to_free.is_empty() {
                None
            } else {
                Some(TensorIPC::from_i64(&self.indices_to_free))
            },
        }
    }
}

/// Build the `req_to_token_cpu` snapshot the worker writes back into
/// `req_to_token_gpu` at the start of every forward pass.
///
/// Source: `schedule_batch.py:2557` — Python computes
/// `self.req_to_token_pool.req_to_token[req_pool_indices].clone()` and
/// ships it on the wire so the worker's `_prepare_batch` can plant
/// the cached prefix slots into the GPU's req_to_token table.  Skipped
/// when no req has a cached prefix (saves a few KB per iter).
///
/// Shape on the wire is `[bs, max_context_len]` int32, exactly
/// matching the GPU side's `req_to_token_pool.req_to_token`.
fn build_req_to_token_cpu(
    batch: &ScheduleBatch,
    req_pool: Option<&crate::memory::ReqToTokenPool>,
) -> Option<TensorIPC> {
    if !batch.forward_mode.is_extend() || batch.is_empty() {
        return None;
    }
    let pool = req_pool?;

    // Skip the snapshot when no req has a cached prefix — mirrors the
    // Python `need_snapshot` gate.  Per-req cost dominates the wire
    // size of this field at large `max_context_len`.
    let any_prefix = batch
        .reqs
        .iter()
        .any(|r| r.read().unwrap().prefix_len_from_cache > 0);
    if !any_prefix {
        return None;
    }

    let bs = batch.batch_size();
    let max_ctx = pool.max_context_len() as usize;
    let mut buf: Vec<i32> = Vec::with_capacity(bs * max_ctx);
    for &slot in &batch.req_pool_indices {
        match pool.read(slot, pool.max_context_len()) {
            Ok(row) => buf.extend_from_slice(row),
            Err(err) => {
                log::warn!("req_to_token_cpu snapshot: failed to read slot {slot}: {err}");
                // Zero-fill on read failure so the tensor shape stays
                // consistent with `req_pool_indices.len()`.
                buf.extend(std::iter::repeat_n(0i32, max_ctx));
            }
        }
    }
    Some(TensorIPC::from_i32_2d(&buf, bs, max_ctx))
}

/// Read-only snapshot the scheduler sends on the wire.  Maps to the
/// `ModelWorkerBatch` dataclass in Python.
///
/// TODO(rust-port): serialise this into the
/// `ForwardBatchGenerationReq.batch` field — the Python receiver
/// (`TpWorkerServer._prepare_batch`) decodes a `ModelWorkerBatch` so
/// we need to produce a msgpack map with the right field names.
/// `schedule_batch.py:ModelWorkerBatch` has 30+ fields; pick them up
/// in waves as features are ported.
#[derive(Debug)]
pub struct ModelWorkerBatchView<'a> {
    pub forward_mode: ForwardMode,
    pub input_ids: &'a [i32],
    pub req_pool_indices: &'a [u32],
    pub seq_lens: &'a [i32],
    pub orig_seq_lens: &'a [i32],
    pub indices_to_free: &'a [i64],
    pub seq_lens_sum: i64,
    pub return_logprob: bool,
}

impl<'a> ModelWorkerBatchView<'a> {
    pub fn from_schedule_batch(batch: &'a ScheduleBatch) -> Self {
        let seq_lens_sum = batch.seq_lens.iter().map(|&v| v as i64).sum();
        Self {
            forward_mode: batch.forward_mode,
            input_ids: &batch.input_ids,
            req_pool_indices: &batch.req_pool_indices,
            seq_lens: &batch.seq_lens,
            orig_seq_lens: &batch.orig_seq_lens,
            indices_to_free: &batch.indices_to_free,
            seq_lens_sum,
            return_logprob: batch.return_logprob,
        }
    }
}
