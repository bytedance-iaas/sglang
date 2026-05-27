// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! On-the-wire mirror of Python's `ModelWorkerBatch` dataclass and the
//! `SamplingBatchInfo` it carries.
//!
//! Wire shape: msgspec encodes a plain Python dataclass (no `tag=True`)
//! as a fixmap of `field_name: value`.  The `ForwardBatchGenerationReq`
//! tagged struct nests this map directly under its `batch` field.
//!
//! Source: `schedule_batch.py:ModelWorkerBatch` (~40 fields) and
//! `sampling_batch_info.py:SamplingBatchInfo` (~15 fields).
//!
//! ## Scope
//!
//! This mirror covers every field msgspec will try to decode — leaving
//! one out causes the worker's decoder to reject the whole struct.
//! Fields the Rust scheduler can't populate yet (penalizer state,
//! grammars, encoder-decoder, multimodal, spec-decoding, dLLM, …) are
//! shipped as the same default values the Python class assigns when no
//! one sets them (`None` / `False` / `[]`).  Once any of those features
//! gets a real Rust port, swap the `None` for the typed value.

use rmpv::Value;
use serde::Serialize;

use crate::wire::TensorIPC;

/// Python `ForwardMode` is an `IntEnum`; msgspec encodes it as the int
/// value.  `IntEnum.auto()` starts at 1, so the discriminants below
/// must stay in sync with the order in
/// `model_executor/forward_batch_info.py`.
pub mod forward_mode_wire {
    pub const EXTEND: i32 = 1;
    pub const DECODE: i32 = 2;
    pub const MIXED: i32 = 3;
    pub const IDLE: i32 = 4;
    pub const TARGET_VERIFY: i32 = 5;
    pub const DRAFT_EXTEND: i32 = 6;
    pub const DRAFT_EXTEND_V2: i32 = 7;
    pub const PREBUILT: i32 = 8;
    pub const SPLIT_PREFILL: i32 = 9;
    pub const DLLM_EXTEND: i32 = 10;
}

/// Mirror of Python `CaptureHiddenMode(IntEnum)` from
/// `model_executor/forward_batch_info.py:195`.
pub mod capture_hidden_mode_wire {
    /// Don't capture anything.
    pub const NULL: i32 = 0;
    /// Capture hidden state of the last token (used by speculative decoding).
    pub const LAST: i32 = 1;
    /// Capture hidden states of every token (used when a req sets
    /// `return_hidden_states=True`).
    pub const FULL: i32 = 2;
}

/// Mirror of `SamplingBatchInfo` (sampling/sampling_batch_info.py).
///
/// The four required tensor fields below carry one element per request.
/// Everything else defaults to `None` / `False` and is filled in as the
/// scheduler gains the corresponding feature.
#[derive(Debug, Clone, Serialize)]
pub struct SamplingBatchInfoPayload {
    /// Shape `[batch, 1]`, dtype `float32`.  Wire schema mirrors the
    /// `.view(-1, 1)` call in `from_schedule_batch`.
    pub temperatures: TensorIPC,
    /// Shape `[batch]`, dtype `float32`.
    pub top_ps: TensorIPC,
    /// Shape `[batch]`, dtype `int32`.
    pub top_ks: TensorIPC,
    /// Shape `[batch]`, dtype `float32`.
    pub min_ps: TensorIPC,

    pub is_all_greedy: bool,
    pub need_top_p_sampling: bool,
    pub need_top_k_sampling: bool,
    pub need_min_p_sampling: bool,

    pub vocab_size: i32,

    // Grammar / structured-output state — `None` until the grammar
    // port lands.  `grammars` is a `List[Optional[GrammarObject]]` in
    // Python; we ship `None` (msgspec interprets a missing-or-None
    // field for a defaulted member as the default).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammars: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab_mask: Option<TensorIPC>,
    // ``apply_mask_func: Optional[Callable]`` — Callables are not
    // serialisable, msgspec skips it on encode.  Omit entirely.

    // Penalizer state — `None` until the penalizer port lands.  msgspec
    // would otherwise try to encode a custom `BatchedPenalizerOrchestratorIPC`
    // via the type's `enc_hook` (see `msgpack_struct.py:1356`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub penalizer_orchestrator: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acc_additive_penalties: Option<TensorIPC>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acc_scaling_penalties: Option<TensorIPC>,

    pub has_custom_logit_processor: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_params: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_logit_processor: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_seed: Option<TensorIPC>,

    pub device: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<TensorIPC>,
}

impl SamplingBatchInfoPayload {
    /// Build with greedy defaults — temperature 1.0, top_p 1.0, top_k -1,
    /// min_p 0.0 per req — suitable for the simplest decode path.
    pub fn from_uniform(batch_size: usize, vocab_size: i32, device: &str) -> Self {
        let temperatures: Vec<f32> = vec![1.0; batch_size];
        let top_ps: Vec<f32> = vec![1.0; batch_size];
        let top_ks: Vec<i32> = vec![-1; batch_size];
        let min_ps: Vec<f32> = vec![0.0; batch_size];

        Self {
            temperatures: TensorIPC::from_f32_2d(&temperatures, batch_size, 1),
            top_ps: TensorIPC::from_f32(&top_ps),
            top_ks: TensorIPC::from_i32(&top_ks),
            min_ps: TensorIPC::from_f32(&min_ps),
            is_all_greedy: false,
            need_top_p_sampling: false,
            need_top_k_sampling: false,
            need_min_p_sampling: false,
            vocab_size,
            grammars: None,
            vocab_mask: None,
            penalizer_orchestrator: None,
            acc_additive_penalties: None,
            acc_scaling_penalties: None,
            has_custom_logit_processor: false,
            custom_params: None,
            custom_logit_processor: None,
            sampling_seed: None,
            device: device.to_string(),
            logit_bias: None,
        }
    }

    /// Build the wire payload from per-request sampling params.  Mirror
    /// of Python's `SamplingBatchInfo.from_schedule_batch` minus the
    /// penalizer orchestrator (which is currently a no-op upstream —
    /// see `BatchedPenalizerOrchestrator.__init__` and the rainj-me
    /// TODO note disabling all penalizers).
    ///
    /// `sampling_seed` only fires when at least one req asks for it —
    /// mirrors `enable_deterministic_inference`.
    pub fn from_sampling_params(
        params_iter: impl ExactSizeIterator<Item = SamplingParamsView>,
        vocab_size: i32,
        device: &str,
    ) -> Self {
        let batch_size = params_iter.len();
        let mut temperatures = Vec::with_capacity(batch_size);
        let mut top_ps = Vec::with_capacity(batch_size);
        let mut top_ks = Vec::with_capacity(batch_size);
        let mut min_ps = Vec::with_capacity(batch_size);
        let mut seeds = Vec::with_capacity(batch_size);
        let mut any_seed_set = false;
        let mut is_all_greedy = true;
        let mut need_top_p = false;
        let mut need_top_k = false;
        let mut need_min_p = false;
        let top_k_all = crate::types::sampling::TOP_K_ALL;

        for p in params_iter {
            temperatures.push(p.temperature);
            top_ps.push(p.top_p);
            top_ks.push(p.top_k);
            min_ps.push(p.min_p);
            if let Some(s) = p.seed {
                seeds.push(s);
                any_seed_set = true;
            } else {
                seeds.push(42);
            }
            if p.top_k > 1 {
                is_all_greedy = false;
            }
            if p.top_p != 1.0 {
                need_top_p = true;
            }
            if p.top_k != top_k_all {
                need_top_k = true;
            }
            if p.min_p > 0.0 {
                need_min_p = true;
            }
        }

        let sampling_seed = if any_seed_set {
            Some(TensorIPC::from_i64(&seeds))
        } else {
            None
        };

        Self {
            temperatures: TensorIPC::from_f32_2d(&temperatures, batch_size, 1),
            top_ps: TensorIPC::from_f32(&top_ps),
            top_ks: TensorIPC::from_i32(&top_ks),
            min_ps: TensorIPC::from_f32(&min_ps),
            is_all_greedy,
            need_top_p_sampling: need_top_p,
            need_top_k_sampling: need_top_k,
            need_min_p_sampling: need_min_p,
            vocab_size,
            grammars: None,
            vocab_mask: None,
            // Penalizers are intentionally disabled — see the rainj-me
            // TODO in `BatchedPenalizerOrchestrator.__init__`.
            penalizer_orchestrator: None,
            acc_additive_penalties: None,
            acc_scaling_penalties: None,
            has_custom_logit_processor: false,
            custom_params: None,
            custom_logit_processor: None,
            sampling_seed,
            device: device.to_string(),
            logit_bias: None,
        }
    }
}

/// Minimal view over the fields the wire payload reads.  Pulled out as
/// a struct so callers can build it from either `SamplingParams` (the
/// scheduler-side type) or directly from `SamplingParamsIPC` (the
/// tokenizer wire type) without forcing a conversion through one or
/// the other.
#[derive(Debug, Clone, Copy)]
pub struct SamplingParamsView {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub seed: Option<i64>,
}

impl SamplingParamsView {
    pub fn from_sampling_params(p: &crate::types::SamplingParams) -> Self {
        Self {
            temperature: p.temperature as f32,
            top_p: p.top_p as f32,
            top_k: p.top_k,
            min_p: p.min_p as f32,
            seed: p.seed,
        }
    }
}

/// Mirror of `schedule_batch.py:ModelWorkerBatch`.  Field order matches
/// the Python source — msgspec decodes by name so order on the wire
/// doesn't matter, but keeping them aligned makes diffs against the
/// Python class trivial.
#[derive(Debug, Clone, Serialize)]
pub struct ModelWorkerBatchPayload {
    pub forward_mode: i32, // ForwardMode IntEnum value
    pub input_ids: TensorIPC,
    pub req_pool_indices: TensorIPC,
    pub seq_lens: TensorIPC,
    pub out_cache_loc: Option<TensorIPC>,
    pub seq_lens_cpu: Option<TensorIPC>,
    pub seq_lens_sum: i64,

    // Logprob streaming.
    pub return_logprob: bool,
    pub top_logprobs_nums: Option<Vec<i32>>,
    pub token_ids_logprobs: Option<Vec<Vec<i32>>>,

    // DP-attention.
    pub global_num_tokens: Option<Vec<i64>>,
    pub global_num_tokens_for_logprob: Option<Vec<i64>>,
    pub is_extend_in_batch: bool,
    pub all_extend_in_batch: bool,
    pub can_run_dp_cuda_graph: bool,
    pub tbo_split_seq_index: Option<i64>,
    pub global_forward_mode: Option<i32>,

    // Extend (prefill) bookkeeping.
    pub extend_num_tokens: Option<i64>,
    pub extend_seq_lens: Option<Vec<i32>>,
    pub extend_prefix_lens: Option<Vec<i32>>,
    pub extend_logprob_start_lens: Option<Vec<i32>>,
    pub extend_input_logprob_token_ids: Option<TensorIPC>,

    pub sampling_info: SamplingBatchInfoPayload,

    // Multimodal — per-req optional payload.  The Python class made the
    // inner type `Optional[MultimodalInputs]` so a mixed list decodes
    // cleanly; we ship `None` per req until the multimodal port lands.
    pub multimodal_inputs: Option<Vec<Option<Value>>>,

    // Encoder-decoder.
    pub encoder_cached: Option<Vec<bool>>,
    pub encoder_lens: Option<TensorIPC>,
    pub encoder_lens_cpu: Option<Vec<i32>>,
    pub encoder_out_cache_loc: Option<TensorIPC>,

    // LoRA.
    pub lora_ids: Option<Vec<Option<String>>>,

    pub orig_seq_lens: Option<TensorIPC>,

    // Input embeddings (Qwen, etc.).
    pub input_embeds: Option<TensorIPC>,
    pub replace_embeds: Option<TensorIPC>,
    pub replace_positions: Option<TensorIPC>,

    pub ne_token_table: Option<TensorIPC>,
    pub token_type_ids: Option<TensorIPC>,

    // Speculative decoding.  `SpeculativeAlgorithm` is a Python `Enum`
    // (not `IntEnum`) — msgspec's `enc_hook` converts it via
    // `SpeculativeAlgorithm(obj)` from an int discriminant, so we just
    // ship the int.  `None` is encoded as msgpack nil.
    pub spec_algorithm: Option<i32>,
    pub spec_info: Option<Value>,

    pub capture_hidden_mode: Option<i32>, // CaptureHiddenMode IntEnum
    pub hicache_consumer_index: i32,

    pub dimensions: Option<Vec<i64>>,
    pub return_pooled_hidden_states: bool,
    pub is_prefill_only: bool,

    pub multi_item_delimiter_indices: Option<Vec<TensorIPC>>,

    pub dllm_block_offsets: Option<Vec<i32>>,
    pub dllm_config: Option<Value>,

    // ``reqs`` is multi-MB Req objects on the Python side; we ship
    // ``None`` and rely on ``rids`` for the worker-side tracing path.
    pub reqs: Option<Value>,
    pub has_grammar: bool,

    pub rids: Option<Vec<String>>,

    pub return_hidden_states_before_norm: bool,

    pub mamba_track_indices: Option<TensorIPC>,
    pub mamba_track_mask: Option<TensorIPC>,
    pub mamba_track_seqlens: Option<TensorIPC>,

    pub req_to_token_cpu: Option<TensorIPC>,
    pub indices_to_free: Option<TensorIPC>,
}

impl ModelWorkerBatchPayload {
    /// Encode `self` as an `rmpv::Value` for inclusion as the `batch`
    /// field of a `ForwardBatchGenerationReq` (which is then serialised
    /// as part of the wire enum).  Round-tripping via `rmpv::Value`
    /// avoids the field-flattening question that direct nesting raises
    /// — see also `sglang-frontend/src/ipc/wire.rs` for the same trick.
    pub fn to_msgpack_value(&self) -> Result<Value, rmp_serde::encode::Error> {
        let bytes = rmp_serde::to_vec_named(self)?;
        // Parse it back as a generic value so the caller can drop it
        // into another struct's field.  rmpv is the same lib msgspec's
        // wire is compatible with.
        let value: Value = rmp_serde::from_slice(&bytes)
            .map_err(|e| rmp_serde::encode::Error::Syntax(format!("rmpv reparse: {e}")))?;
        Ok(value)
    }
}
