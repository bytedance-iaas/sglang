// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Per-step output processing — what to do with a
//! `DecodeForwardSlimOutput` once the worker hands it back.
//!
//! Source: `scheduler_output_processor_mixin.py` (~1300 lines on the
//! Python side).  The Rust port today covers:
//!   * Appending the sampled token to each req's `output_ids`.
//!   * `check_finished` for length / stop-token (Req method).
//!   * Updating the page tracker from
//!     `deferred_alloc.free_pages_remaining`.
//!   * Releasing the KV-slot footprint of just-finished reqs via the
//!     `ReqToTokenPool` (slot ids → page-tracker `free`).
//!   * Writing the worker's `out_cache_loc` into the per-req `req_to_token`
//!     rows for extend-mode replies (mirror of
//!     `CpuScheduler.update_cache_from_scheduler`).
//!
//! What's deferred to follow-ups (Python is authoritative until then):
//!   * Logprobs streaming (`process_batch_result_decode`'s `return_logprob`
//!     branch).
//!   * Grammar / structured output integration.
//!   * Hidden-states return.
//!   * Routed-experts (MoE) telemetry.
//!   * Mamba bookkeeping.
//!   * Speculative-decoding accept counting.

use rmpv::Value;

use crate::memory::{CpuPageTracker, ReqToTokenPool};
use crate::radix_cache::RadixCache;
use crate::types::{FinishReason, ForwardMode, ScheduleBatch};
use crate::wire::{BatchTokenIDOutput, DecodeForwardSlimOutput, DeferredAllocIPC};

#[derive(Debug, Default)]
pub struct StepStats {
    pub num_generated_tokens: u64,
    pub num_finished_reqs: u64,
    pub can_run_cuda_graph: bool,
    /// Slot ids the output processor returned to the
    /// `ReqToTokenPool`.  Mostly diagnostic; the pool tracks its own
    /// free list.
    pub freed_req_pool_slots: u32,
    /// Per-iteration scheduler → detokenizer frame.  `None` when the
    /// batch was empty (idle iteration); otherwise the caller pushes
    /// it through the detokenizer PUSH socket.
    pub detokenizer_output: Option<BatchTokenIDOutput>,
}

/// Apply a worker reply to the scheduler-side state.
pub fn process_batch_result(
    batch: &mut ScheduleBatch,
    reply: &DecodeForwardSlimOutput,
    page_tracker: &mut CpuPageTracker,
    req_pool: &mut ReqToTokenPool,
) -> StepStats {
    process_batch_result_with_cache(batch, reply, page_tracker, req_pool, None)
}

/// Variant that also touches the radix cache: finished reqs insert
/// their post-prefix slot indices into the tree (so future reqs can
/// reuse them) and dec_lock_ref the node they pinned at admission.
pub fn process_batch_result_with_cache(
    batch: &mut ScheduleBatch,
    reply: &DecodeForwardSlimOutput,
    page_tracker: &mut CpuPageTracker,
    req_pool: &mut ReqToTokenPool,
    mut radix_cache: Option<&mut RadixCache>,
) -> StepStats {
    let mut stats = StepStats {
        can_run_cuda_graph: reply.can_run_cuda_graph,
        ..Default::default()
    };

    // 1. Worker's authoritative free-page count.  Layered with our
    //    pending-free count by `CpuPageTracker::available_size`.
    if let Some(da) = &reply.deferred_alloc {
        page_tracker.update_free_count(da.free_pages_remaining);

        // The worker reports newly-allocated KV slot ids for every
        // iteration that produced KV writes (extend + decode).  We
        // must mirror those writes into the CPU `req_to_token` table,
        // otherwise the cache.insert at finish-time grafts STALE slot
        // ids (from a previous tenant of the same `req_pool` row),
        // and the next req that hits the cache attends against
        // garbage KV → CUDA index OOB.
        //
        // Mirror of `scheduler_cpu.py:update_cache_from_scheduler`
        // (extend branch at line 608, decode branch at line 647).
        if matches!(batch.forward_mode, ForwardMode::Extend | ForwardMode::Mixed) {
            apply_extend_deferred_alloc(batch, da, req_pool);
        } else if matches!(batch.forward_mode, ForwardMode::Decode) {
            apply_decode_deferred_alloc(batch, da, req_pool);
        }
    }

    // 2. Sampled tokens — one per req for decode + the first sampled
    //    token per req for extend.  Same shape (1-per-req) either way.
    let Some(next_token_ids) = reply.next_token_ids.as_i64() else {
        log::warn!(
            "next_token_ids dtype was {} (expected int64); leaving batch unchanged",
            reply.next_token_ids.dtype()
        );
        return stats;
    };
    if next_token_ids.len() != batch.reqs.len() {
        log::warn!(
            "next_token_ids length {} != batch.reqs.len {} — composition desync?",
            next_token_ids.len(),
            batch.reqs.len()
        );
    }

    // Finished-req bookkeeping captured here, applied after the batch
    // loop drops its locks.
    struct FinishedReq {
        slot: u32,
        kv_len: u32,
        cached_node: Option<crate::radix_cache::NodeId>,
        // tokens (prompt + outputs) that should be grafted into the
        // radix cache, in token order.
        full_tokens: Vec<i32>,
        prefix_len_from_cache: usize,
    }
    let mut finished: Vec<FinishedReq> = Vec::new();

    // Build the per-iteration output frame for the detokenizer
    // alongside the bookkeeping loop.  Skipped when the batch is empty
    // (idle iteration).
    let mut out = if batch.reqs.is_empty() {
        None
    } else {
        Some(BatchTokenIDOutput::with_capacity(batch.reqs.len()))
    };

    for (i, &tok) in next_token_ids.iter().take(batch.reqs.len()).enumerate() {
        let mut req = batch.reqs[i].write().unwrap();
        req.output_ids.push(tok as i32);
        let finish_reason_opt = req.check_finished().cloned();
        if let Some(reason) = &finish_reason_opt {
            stats.num_finished_reqs += 1;
            if !matches!(reason, FinishReason::Retracted) {
                let cached_node = req.cached_node.take();
                let prefix_len_from_cache = req.prefix_len_from_cache as usize;
                let mut full_tokens =
                    Vec::with_capacity(req.origin_input_ids.len() + req.output_ids.len());
                full_tokens.extend_from_slice(&req.origin_input_ids);
                full_tokens.extend_from_slice(&req.output_ids);
                if let Some(idx) = req.req_pool_idx.take() {
                    let len = req.kv_allocated_len;
                    finished.push(FinishedReq {
                        slot: idx,
                        kv_len: len,
                        cached_node,
                        full_tokens,
                        prefix_len_from_cache,
                    });
                }
                req.kv_committed_freed = true;
                req.kv_overallocated_freed = true;
            }
        }

        // Emit one entry into the detokenizer output frame for this req.
        if let Some(o) = out.as_mut() {
            o.rids.as_mut().unwrap().push(req.rid.clone());
            o.http_worker_ipcs.as_mut().unwrap().push(None);
            o.finished_reasons
                .push(finish_reason_opt.as_ref().map(finish_reason_to_value));
            o.decoded_texts.push(String::new());
            // One newly-sampled token per req per step.
            let new_tok = tok;
            o.decode_ids.push(vec![new_tok]);
            o.read_offsets.push(0);
            o.skip_special_tokens
                .push(req.sampling_params.skip_special_tokens);
            o.spaces_between_special_tokens
                .push(req.sampling_params.spaces_between_special_tokens);
            o.no_stop_trim.push(req.sampling_params.no_stop_trim);
            o.prompt_tokens.push(req.origin_input_ids.len() as i64);
            o.reasoning_tokens.push(0);
            o.completion_tokens.push(req.output_ids.len() as i64);
            o.cached_tokens.push(req.prefix_len_from_cache as i64);
            // skip_tokenizer_init path: include the raw output id so
            // the detokenizer manager can stream without re-decoding.
            o.output_ids.as_mut().unwrap().push(vec![new_tok]);
        }

        stats.num_generated_tokens += 1;
    }

    stats.detokenizer_output = out;

    // 3. For each finished req:
    //    a. Release the lock the admission step took on its cached prefix.
    //    b. Read the full slot row and graft (tokens, slots) into the
    //       radix cache so future reqs can reuse the suffix as a prefix.
    //    c. The slot indices that ended up duplicated by `insert` go
    //       back to the page tracker; the rest are owned by the cache now.
    //    d. Free the `ReqToTokenPool` slot.
    for done in finished {
        if done.kv_len == 0 {
            // No KV ever allocated (admitted but never prefilled?)
            if let Some(node) = done.cached_node
                && let Some(cache) = radix_cache.as_deref_mut()
            {
                cache.dec_lock_ref(node);
            }
            if let Err(err) = req_pool.free(done.slot) {
                log::warn!("req_pool free failed for slot {}: {err}", done.slot);
            }
            stats.freed_req_pool_slots += 1;
            continue;
        }
        let slot_indices: Vec<i32> = match req_pool.read(done.slot, done.kv_len) {
            Ok(indices) => indices.to_vec(),
            Err(err) => {
                log::warn!(
                    "could not read KV slot indices for finished req \
                     (slot={}, len={}): {err}",
                    done.slot,
                    done.kv_len
                );
                if let Some(node) = done.cached_node
                    && let Some(cache) = radix_cache.as_deref_mut()
                {
                    cache.dec_lock_ref(node);
                }
                if let Err(err) = req_pool.free(done.slot) {
                    log::warn!("req_pool free failed for slot {}: {err}", done.slot);
                }
                stats.freed_req_pool_slots += 1;
                continue;
            }
        };

        // Release the admission-time lock first so the cache sees the
        // prefix as evictable while we insert the suffix.
        if let Some(node) = done.cached_node
            && let Some(cache) = radix_cache.as_deref_mut()
        {
            cache.dec_lock_ref(node);
        }

        // Graft the full (tokens, slots) span into the cache.  The
        // returned `duplicates` list contains every slot that matched
        // an existing tree node (i.e. `slots[0..new_prefix_len]`).
        let tokens_len = done.full_tokens.len().min(slot_indices.len());
        let inserted_tokens = &done.full_tokens[..tokens_len];
        let inserted_slots = &slot_indices[..tokens_len];
        let duplicates = if let Some(cache) = radix_cache.as_deref_mut() {
            cache.insert(inserted_tokens, inserted_slots)
        } else {
            Vec::new()
        };

        // Three-region split, mirroring Python's
        // `radix_cache.cache_finished_req`:
        //   * `[0..cache_protected_len)` — slots planted at admission.
        //     The cache ALREADY OWNS these (`inc_lock_ref` pinned them
        //     and the in-tree value array still references them).
        //     Freeing would tell the worker to release a slot the cache
        //     considers live → next match returns stale slots → CUDA
        //     index OOB on the next req that hits this prefix.
        //   * `[cache_protected_len..new_prefix_len)` — slots the req
        //     freshly allocated but the insert walk merged into an
        //     existing branch (another req finished with the same
        //     suffix while we ran).  These are GENUINELY orphaned and
        //     must be freed.
        //   * `[new_prefix_len..)` — slots now grafted into the cache
        //     as a new leaf.  Cache OWNS these too.  Not freed.
        if radix_cache.is_some() {
            let new_prefix_len = duplicates.len();
            let protected = done.prefix_len_from_cache.min(new_prefix_len);
            if new_prefix_len > protected {
                page_tracker.free_i32(&slot_indices[protected..new_prefix_len]);
            }
            // Trailing slots beyond `tokens_len` weren't passed to
            // `insert` at all — they're surplus from `req_pool.read`'s
            // over-reporting (rare).  Reclaim them unconditionally.
            if slot_indices.len() > tokens_len {
                page_tracker.free_i32(&slot_indices[tokens_len..]);
            }
        } else {
            // No cache wired → all slots back to the allocator.
            page_tracker.free_i32(&slot_indices);
        }

        if let Err(err) = req_pool.free(done.slot) {
            log::warn!("req_pool free failed for slot {}: {err}", done.slot);
        }
        stats.freed_req_pool_slots += 1;
    }

    stats
}

/// Plant the worker-allocated slot indices into `req_to_token` for an
/// extend-mode reply.  Mirrors the extend branch of
/// `CpuScheduler.update_cache_from_scheduler`.
fn apply_extend_deferred_alloc(
    batch: &ScheduleBatch,
    da: &DeferredAllocIPC,
    req_pool: &mut ReqToTokenPool,
) {
    if da.mode != "extend" {
        return;
    }
    let Some(req_pool_indices) = da.req_pool_indices.as_i64() else {
        log::warn!(
            "deferred_alloc.req_pool_indices dtype {} != int64; skipping writeback",
            da.req_pool_indices.dtype()
        );
        return;
    };
    // Worker emits `out_cache_loc` as int64 (GPU-side dtype — see
    // `tp_worker_server.py:_deferred_alloc`); the local `.to(torch.int32)`
    // calls only apply to the worker-side `req_to_token_gpu` writeback,
    // not the IPC payload.  Accept either dtype and narrow into a
    // local `Vec<i32>` for `req_pool.write`.
    let out_cache_loc_owned: Vec<i32> = if let Some(slice) = da.out_cache_loc.as_i32() {
        slice.to_vec()
    } else if let Some(slice) = da.out_cache_loc.as_i64() {
        slice.iter().map(|&v| v as i32).collect()
    } else {
        log::warn!(
            "deferred_alloc.out_cache_loc dtype {} not in {{int32, int64}}; skipping writeback",
            da.out_cache_loc.dtype()
        );
        return;
    };
    let out_cache_loc: &[i32] = &out_cache_loc_owned;
    let Some(prefix_lens) = da.prefix_lens.as_ref().and_then(|t| t.as_i64()) else {
        log::warn!("deferred_alloc.prefix_lens missing for extend mode; skipping writeback");
        return;
    };
    let Some(extend_lens) = da.extend_lens.as_ref().and_then(|t| t.as_i64()) else {
        log::warn!("deferred_alloc.extend_lens missing for extend mode; skipping writeback");
        return;
    };

    let mut pt = 0usize;
    for (i, &slot) in req_pool_indices.iter().enumerate() {
        let pre = prefix_lens[i] as u32;
        let ext = extend_lens[i] as usize;
        let slot_u32 = slot as u32;
        if pt + ext > out_cache_loc.len() {
            log::warn!(
                "out_cache_loc shorter than declared extend_lens (i={i}, pt={pt}, ext={ext})"
            );
            return;
        }
        if let Err(err) = req_pool.write(slot_u32, pre, &out_cache_loc[pt..pt + ext]) {
            log::warn!("req_pool write failed for slot {slot_u32}: {err}");
            // Also update each req's `kv_allocated_len` to reflect the
            // freshly-prefilled span.
        } else if let Some(req_arc) = batch.reqs.get(i) {
            let mut req = req_arc.write().unwrap();
            req.kv_allocated_len = pre + ext as u32;
            req.kv_committed_len = pre + ext as u32;
        }
        pt += ext;
    }
}

/// Plant the worker-allocated KV slot id for a decode step into the
/// CPU `req_to_token` table.  Mirrors the `decode` branch of
/// `scheduler_cpu.py:update_cache_from_scheduler` (line 647).
///
/// For each running req `i`, the worker reports:
///   * `req_pool_indices[i]` — the req's row in `req_to_token`.
///   * `seq_lens_minus1[i]`  — the column to write (the position of
///     the just-decoded token, == `seq_len - 1` after the step).
///   * `out_cache_loc[i]`    — the KV slot id allocated for that token.
///
/// Skipping this writeback (as the Rust port did before this fix) leaves
/// the CPU table stale: when the req finishes, `req_pool.read(...)`
/// returns whatever slot ids the *previous* tenant of this row had,
/// and `cache.insert` grafts the stale slot ids into the radix tree.
/// The next req that matches the prefix attends against garbage KV
/// (the GPU has long since reused those slot ids), producing the
/// classic "CUDA index out of bounds" failure.
fn apply_decode_deferred_alloc(
    batch: &ScheduleBatch,
    da: &DeferredAllocIPC,
    req_pool: &mut ReqToTokenPool,
) {
    if da.mode != "decode" {
        return;
    }
    let Some(req_pool_indices) = da.req_pool_indices.as_i64() else {
        log::warn!(
            "deferred_alloc.req_pool_indices dtype {} != int64; skipping decode writeback",
            da.req_pool_indices.dtype()
        );
        return;
    };
    // Worker emits `out_cache_loc` as int64 (same as extend); narrow
    // to i32 for the CPU pool.  Accept both dtypes for safety.
    let out_cache_loc_owned: Vec<i32> = if let Some(slice) = da.out_cache_loc.as_i32() {
        slice.to_vec()
    } else if let Some(slice) = da.out_cache_loc.as_i64() {
        slice.iter().map(|&v| v as i32).collect()
    } else {
        log::warn!(
            "deferred_alloc.out_cache_loc dtype {} not in {{int32, int64}}; skipping decode writeback",
            da.out_cache_loc.dtype()
        );
        return;
    };
    let Some(seq_lens_minus1) = da.seq_lens_minus1.as_ref().and_then(|t| t.as_i64()) else {
        log::warn!("deferred_alloc.seq_lens_minus1 missing for decode mode; skipping writeback");
        return;
    };

    if req_pool_indices.len() != out_cache_loc_owned.len()
        || req_pool_indices.len() != seq_lens_minus1.len()
    {
        log::warn!(
            "decode deferred_alloc shape mismatch: req_pool_indices={}, out_cache_loc={}, seq_lens_minus1={}",
            req_pool_indices.len(),
            out_cache_loc_owned.len(),
            seq_lens_minus1.len()
        );
        return;
    }

    // `prepare_for_decode` (the previous iter) already bumped
    // `kv_allocated_len` / `kv_committed_len` by 1, so this writeback
    // only needs to plant the slot id at the column the worker chose.
    let _ = batch;
    for (i, (&slot, &col)) in req_pool_indices
        .iter()
        .zip(seq_lens_minus1.iter())
        .enumerate()
    {
        let slot_u32 = slot as u32;
        let col_u32 = col as u32;
        if let Err(err) = req_pool.write(slot_u32, col_u32, &out_cache_loc_owned[i..=i]) {
            log::warn!("decode writeback failed for slot {slot_u32} col {col_u32}: {err}");
        }
    }
}

/// Convert a Rust `FinishReason` into the JSON-shaped dict Python's
/// `BaseFinishReason.to_json()` produces.  The detokenizer manager
/// reads `finished_reasons[i]["type"]` to route per-req cleanup.
fn finish_reason_to_value(reason: &FinishReason) -> Value {
    let entries: Vec<(Value, Value)> = match reason {
        FinishReason::Length { length } => vec![
            ("type".into(), Value::from("length")),
            ("length".into(), Value::from(*length)),
        ],
        FinishReason::MatchedToken { token_id } => vec![
            ("type".into(), Value::from("stop")),
            ("matched".into(), Value::from(*token_id)),
        ],
        FinishReason::MatchedStr { stop } => vec![
            ("type".into(), Value::from("stop")),
            ("matched".into(), Value::from(stop.clone())),
        ],
        FinishReason::Aborted { message } => vec![
            ("type".into(), Value::from("abort")),
            ("message".into(), Value::from(message.clone())),
        ],
        FinishReason::Retracted => {
            vec![
                ("type".into(), Value::from("abort")),
                ("message".into(), Value::from("retracted")),
            ]
        }
        FinishReason::Internal { message } => vec![
            ("type".into(), Value::from("abort")),
            ("message".into(), Value::from(message.clone())),
        ],
    };
    Value::Map(entries)
}
