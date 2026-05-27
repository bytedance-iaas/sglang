// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Scheduler → detokenizer wire types.
//!
//! Source: `BatchTokenIDOutput` / `BatchEmbeddingOutput` in
//! `python/sglang/srt/managers/io_struct/msgpack_struct.py`.
//!
//! The Rust scheduler today emits a minimal `BatchTokenIDOutput`
//! covering the fields the Python detokenizer reads on the common
//! `skip_tokenizer_init=True` path: per-req new token ids, prompt /
//! completion / cached counts, finish reasons, and detokenization
//! flags.  Everything else (logprobs, hidden states, routed experts,
//! placeholder tokens, …) rides out as the Python defaults (`None`)
//! since the Rust scheduler doesn't compute those yet.

use rmpv::Value;
use serde::{Deserialize, Serialize};

/// Mirror of Python `BatchTokenIDOutput`.  All `Option<…>` fields
/// default to `None` on serialize so the wire stays compact and old
/// detokenizers still decode (msgspec tolerates missing optionals).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTokenIDOutput {
    // ── Speculative decoding metrics (always emitted, even when off) ──
    pub spec_verify_ct: Vec<i64>,
    pub spec_accepted_drafts: Vec<i64>,
    pub spec_acceptance_histogram: Vec<Vec<i64>>,

    // ── Per-req progress ───────────────────────────────────────────────
    /// Per-req finish reason as a JSON-shaped dict, or `None` for
    /// still-running requests.  Shape mirrors Python
    /// `BaseFinishReason.to_json()`:
    ///   * `{"type": "length", "length": N}`
    ///   * `{"type": "stop", "matched": <int|str>}`
    ///   * `{"type": "abort", "message": "…"}`
    pub finished_reasons: Vec<Option<Value>>,

    /// Per-req decoded text.  Empty strings on the skip-tokenizer-init
    /// path — the detokenizer manager fills them in from `output_ids`.
    pub decoded_texts: Vec<String>,

    /// Per-req new token ids since the last frame.  A `Vec<i64>` per
    /// request; usually one token per decode step (one extend frame can
    /// emit a longer list on prefill replies).
    pub decode_ids: Vec<Vec<i64>>,

    /// Read-offset into the decoded text the detokenizer has streamed
    /// so far.  Zero on the skip-tokenizer-init path.
    pub read_offsets: Vec<i64>,

    pub skip_special_tokens: Vec<bool>,
    pub spaces_between_special_tokens: Vec<bool>,
    pub no_stop_trim: Vec<bool>,

    pub prompt_tokens: Vec<i64>,
    pub reasoning_tokens: Vec<i64>,
    pub completion_tokens: Vec<i64>,
    pub cached_tokens: Vec<i64>,

    // ── Optional fields the Python detokenizer tolerates as `None` ────
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_ids: Option<Vec<Vec<i64>>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_token_logprobs_val: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_token_logprobs_idx: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_logprobs_val: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_logprobs_idx: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_top_logprobs_val: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_top_logprobs_idx: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_top_logprobs_val: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_top_logprobs_idx: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_token_ids_logprobs_val: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_token_ids_logprobs_idx: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_ids_logprobs_val: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_ids_logprobs_idx: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_entropy_val: Option<Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_hidden_states: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routed_experts: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placeholder_tokens_idx: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placeholder_tokens_val: Option<Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retraction_counts: Option<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_steps: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub customized_info: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens_details: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_ranks: Option<Vec<Option<i32>>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_stats: Option<Value>,

    // From BaseBatchReq.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rids: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_worker_ipcs: Option<Vec<Option<String>>>,
}

impl BatchTokenIDOutput {
    /// Build an empty payload sized to `batch_size`.  Caller fills the
    /// per-req vectors; defaults match what the Python detokenizer
    /// expects when speculative-decoding metrics are off.
    pub fn with_capacity(batch_size: usize) -> Self {
        Self {
            spec_verify_ct: vec![0; batch_size],
            spec_accepted_drafts: vec![0; batch_size],
            spec_acceptance_histogram: vec![Vec::new(); batch_size],
            finished_reasons: Vec::with_capacity(batch_size),
            decoded_texts: Vec::with_capacity(batch_size),
            decode_ids: Vec::with_capacity(batch_size),
            read_offsets: Vec::with_capacity(batch_size),
            skip_special_tokens: Vec::with_capacity(batch_size),
            spaces_between_special_tokens: Vec::with_capacity(batch_size),
            no_stop_trim: Vec::with_capacity(batch_size),
            prompt_tokens: Vec::with_capacity(batch_size),
            reasoning_tokens: Vec::with_capacity(batch_size),
            completion_tokens: Vec::with_capacity(batch_size),
            cached_tokens: Vec::with_capacity(batch_size),
            output_ids: Some(Vec::with_capacity(batch_size)),
            input_token_logprobs_val: None,
            input_token_logprobs_idx: None,
            output_token_logprobs_val: None,
            output_token_logprobs_idx: None,
            input_top_logprobs_val: None,
            input_top_logprobs_idx: None,
            output_top_logprobs_val: None,
            output_top_logprobs_idx: None,
            input_token_ids_logprobs_val: None,
            input_token_ids_logprobs_idx: None,
            output_token_ids_logprobs_val: None,
            output_token_ids_logprobs_idx: None,
            output_token_entropy_val: None,
            output_hidden_states: None,
            routed_experts: None,
            placeholder_tokens_idx: None,
            placeholder_tokens_val: None,
            retraction_counts: None,
            token_steps: None,
            load: None,
            customized_info: None,
            cached_tokens_details: None,
            dp_ranks: None,
            time_stats: None,
            rids: Some(Vec::with_capacity(batch_size)),
            http_worker_ipcs: Some(Vec::with_capacity(batch_size)),
        }
    }
}
