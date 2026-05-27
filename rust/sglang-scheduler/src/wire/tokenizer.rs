// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Tokenizer → scheduler requests.
//!
//! These travel on a separate ZMQ socket from the worker IPC (the
//! `recv_from_tokenizer` PULL socket in `scheduler.py`).  The Python
//! source-of-truth lives in
//! `python/sglang/srt/managers/io_struct/msgpack_struct.py`.
//!
//! Only the subset of fields the Rust scheduler actually reads is
//! modelled with strong types; the long tail of optional fields rides
//! out as `Option<rmpv::Value>` so we accept anything the tokenizer
//! sends without forcing a schema lock.

use rmpv::Value;
use serde::{Deserialize, Serialize};

/// Convenience alias for the "we don't care, just preserve it" field
/// pattern.  msgspec rides arbitrary nested structures (dicts, lists,
/// nested struct payloads) and we don't need to reach into them.
type OptValue = Option<Value>;

/// `SamplingParamsIPC` — see `msgpack_struct.py`.
///
/// Mirrors only the fields the scheduler reaches for during admission
/// (`max_new_tokens`, `stop_token_ids`, `ignore_eos`, …).  Everything
/// else is preserved as `extra` so the wire payload can round-trip
/// through the Rust scheduler unmodified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParamsIPC {
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: u32,

    #[serde(default)]
    pub stop_token_ids: Option<Vec<i64>>,

    #[serde(default = "default_temperature")]
    pub temperature: f32,

    #[serde(default = "default_top_p")]
    pub top_p: f32,

    #[serde(default = "default_top_k")]
    pub top_k: i32,

    #[serde(default)]
    pub min_p: f32,

    #[serde(default)]
    pub frequency_penalty: f32,

    #[serde(default)]
    pub presence_penalty: f32,

    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    #[serde(default)]
    pub min_new_tokens: u32,

    #[serde(default = "default_n")]
    pub n: u32,

    #[serde(default)]
    pub ignore_eos: bool,

    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,

    #[serde(default = "default_true")]
    pub spaces_between_special_tokens: bool,

    #[serde(default)]
    pub no_stop_trim: bool,

    #[serde(default)]
    pub sampling_seed: Option<i64>,

    // Everything else (stop strings, regex, structural tag, custom
    // params, logit_bias, …) survives unparsed.  See README for the
    // policy on these.
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub stop_regex: Option<Value>,
    #[serde(default)]
    pub json_schema: Option<String>,
    #[serde(default)]
    pub regex: Option<String>,
    #[serde(default)]
    pub ebnf: Option<String>,
    #[serde(default)]
    pub structural_tag: Option<String>,
    #[serde(default)]
    pub custom_params: Option<Value>,
    #[serde(default)]
    pub stream_interval: Option<i64>,
    #[serde(default)]
    pub logit_bias: Option<Value>,
    #[serde(default)]
    pub stop_str_max_len: u32,
    #[serde(default)]
    pub stop_regex_max_len: u32,
}

fn default_max_new_tokens() -> u32 {
    128
}
fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    1.0
}
fn default_top_k() -> i32 {
    -1
}
fn default_repetition_penalty() -> f32 {
    1.0
}
fn default_n() -> u32 {
    1
}
fn default_true() -> bool {
    true
}

impl Default for SamplingParamsIPC {
    fn default() -> Self {
        Self {
            max_new_tokens: default_max_new_tokens(),
            stop_token_ids: None,
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: default_top_k(),
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: default_repetition_penalty(),
            min_new_tokens: 0,
            n: default_n(),
            ignore_eos: false,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            no_stop_trim: false,
            sampling_seed: None,
            stop: None,
            stop_regex: None,
            json_schema: None,
            regex: None,
            ebnf: None,
            structural_tag: None,
            custom_params: None,
            stream_interval: None,
            logit_bias: None,
            stop_str_max_len: 0,
            stop_regex_max_len: 0,
        }
    }
}

/// `TokenizedGenerateReqInput` — see `msgpack_struct.py`.
///
/// The scheduler admits these into the waiting queue.  Long tail of
/// optional fields lives in `extras` as opaque msgpack `Value` so a
/// drift between the Python tokenizer and the Rust scheduler doesn't
/// break decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizedGenerateReqInput {
    pub input_ids: Vec<i64>,
    pub sampling_params: SamplingParamsIPC,
    pub return_logprob: bool,
    pub logprob_start_len: i64,
    pub top_logprobs_num: u32,
    pub stream: bool,

    #[serde(default)]
    pub input_text: Option<String>,
    #[serde(default)]
    pub token_ids_logprob: Option<Vec<i64>>,
    #[serde(default)]
    pub mm_inputs: OptValue,
    #[serde(default)]
    pub return_hidden_states: bool,
    #[serde(default)]
    pub return_routed_experts: bool,
    #[serde(default)]
    pub routed_experts_start_len: u32,
    #[serde(default)]
    pub input_embeds: OptValue,
    #[serde(default)]
    pub positional_embed_overrides: OptValue,
    #[serde(default)]
    pub session_params: OptValue,
    #[serde(default)]
    pub lora_id: Option<String>,
    #[serde(default)]
    pub custom_logit_processor: Option<String>,
    #[serde(default)]
    pub bootstrap_host: Option<String>,
    #[serde(default)]
    pub bootstrap_port: Option<i64>,
    #[serde(default)]
    pub bootstrap_room: Option<i64>,
    #[serde(default)]
    pub bootstrap_pair_key: Option<String>,
    #[serde(default)]
    pub decode_tp_size: Option<i32>,
    #[serde(default)]
    pub require_reasoning: bool,
    #[serde(default)]
    pub routed_dp_rank: Option<i32>,
    #[serde(default)]
    pub disagg_prefill_dp_rank: Option<i32>,
    #[serde(default)]
    pub priority: Option<i64>,
    #[serde(default)]
    pub extra_key: Option<String>,
    #[serde(default)]
    pub routing_key: Option<String>,
    #[serde(default)]
    pub no_logs: bool,
    #[serde(default)]
    pub return_bytes: bool,
    #[serde(default)]
    pub return_entropy: bool,
    #[serde(default)]
    pub token_type_ids: Option<Vec<i64>>,
    #[serde(default)]
    pub need_wait_for_mm_inputs: Option<bool>,
    #[serde(default)]
    pub num_items_assigned: OptValue,
    #[serde(default)]
    pub multi_item_delimiter_indices: Option<Vec<i64>>,
    #[serde(default)]
    pub time_stats: OptValue,

    // From BaseReq
    #[serde(default)]
    pub rid: Option<String>,
    #[serde(default)]
    pub http_worker_ipc: Option<String>,
}

/// `BatchTokenizedGenerateReqInput` — a list of `TokenizedGenerateReqInput`s
/// sent together.  The scheduler explodes this into per-request admissions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTokenizedGenerateReqInput {
    pub reqs: Vec<TokenizedGenerateReqInput>,
    #[serde(default)]
    pub rids: Option<Vec<String>>,
    #[serde(default)]
    pub http_worker_ipcs: Option<Vec<String>>,
}

/// `TokenizedEmbeddingReqInput` — embedding mode.  Not driven by the
/// Rust scheduler yet (we exit cleanly if one arrives), but the type
/// has to decode so `Wire` round-trips don't fail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizedEmbeddingReqInput {
    pub input_ids: Vec<i64>,
    pub sampling_params: SamplingParamsIPC,

    #[serde(default)]
    pub input_text: Option<String>,
    #[serde(default)]
    pub image_inputs: OptValue,
    #[serde(default)]
    pub token_type_ids: Option<Vec<i64>>,
    #[serde(default)]
    pub positional_embed_overrides: OptValue,
    #[serde(default)]
    pub routed_dp_rank: Option<i32>,
    #[serde(default)]
    pub priority: Option<i64>,
    #[serde(default)]
    pub dimensions: Option<i64>,
    #[serde(default)]
    pub lora_id: Option<String>,
    #[serde(default)]
    pub multi_item_delimiter_indices: Option<Vec<i64>>,
    #[serde(default)]
    pub time_stats: OptValue,
    #[serde(default)]
    pub return_pooled_hidden_states: bool,

    #[serde(default)]
    pub rid: Option<String>,
    #[serde(default)]
    pub http_worker_ipc: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTokenizedEmbeddingReqInput {
    pub reqs: Vec<TokenizedEmbeddingReqInput>,
    #[serde(default)]
    pub rids: Option<Vec<String>>,
    #[serde(default)]
    pub http_worker_ipcs: Option<Vec<String>>,
}

/// `AbortReq` — instructs the scheduler to abort a running or queued
/// request (or all of them when `abort_all` is set).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AbortReq {
    #[serde(default)]
    pub abort_all: bool,
    #[serde(default)]
    pub finished_reason: OptValue,
    #[serde(default)]
    pub abort_message: Option<String>,
    #[serde(default)]
    pub rid: Option<String>,
    #[serde(default)]
    pub http_worker_ipc: Option<String>,
}
