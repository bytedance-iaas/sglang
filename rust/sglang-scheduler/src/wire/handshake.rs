// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! GPU worker handshake — first round-trip after socket connect.
//!
//! Source: `python/sglang/srt/managers/io_struct/msgpack_struct.py`
//! classes `GPUWorkerHandshakeReqInput` and `GPUWorkerHandshakeReqOutput`.

use serde::{Deserialize, Serialize};

use crate::wire::tensor_ipc::TensorIPC;

/// Empty input — the worker only needs to know the scheduler is alive.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GPUWorkerHandshakeReqInput {}

/// Worker → scheduler init payload.
///
/// The Python `_init_from_handshake` consumes these fields one-by-one;
/// see `tp_worker_client.py:_init_from_handshake`.  Optional fields use
/// `#[serde(default)]` so older workers without the field still decode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUWorkerHandshakeReqOutput {
    // --- worker_info tuple (must stay in the same order as Python's
    //     ``TpWorkerClient._worker_info``) -----------------------------------
    pub max_total_num_tokens: i64,
    pub max_prefill_tokens: i64,
    pub max_running_requests: i64,
    pub max_req_len: i64,
    pub max_req_input_len: i64,
    pub random_seed: i64,
    pub device: String,
    pub req_to_token_pool_size: i64,
    pub req_to_token_pool_max_context_len: i64,
    pub token_to_kv_pool_size: i64,

    #[serde(default)]
    pub max_queued_requests: Option<i64>,

    // --- hybrid SWA ----------------------------------------------------------
    #[serde(default)]
    pub is_hybrid_swa: bool,
    #[serde(default)]
    pub sliding_window_size: Option<i64>,

    // --- tokens-per-layer info ----------------------------------------------
    #[serde(default)]
    pub full_max_total_num_tokens: i64,
    #[serde(default)]
    pub swa_max_total_num_tokens: i64,

    // --- misc model flags ---------------------------------------------------
    #[serde(default)]
    pub is_dllm: bool,

    // --- model-runner attributes that the scheduler may inspect.  These
    //     are opaque to the Rust scheduler (they exist for parity with the
    //     Python ``ModelRunnerProxy``); we just pass them around. ------------
    #[serde(default)]
    pub lora_manager: Option<serde_json::Value>,
    #[serde(default)]
    pub token_table: Option<TensorIPC>,
    #[serde(default)]
    pub linear_attn_model_spec: Option<serde_json::Value>,
    #[serde(default)]
    pub hybrid_gdn_config: Option<serde_json::Value>,
    #[serde(default)]
    pub mamba2_config: Option<serde_json::Value>,

    // --- ModelConfig scalars piggybacked off ``model_runner.model_config``
    //     so the Rust scheduler doesn't need HuggingFace ``transformers``
    //     itself.  Defaulted so an older worker connecting to a newer
    //     scheduler still decodes (the scheduler then falls back to the
    //     previous behaviour: ``vocab_size = 0`` etc.). ---------------------
    #[serde(default)]
    pub vocab_size: i64,
    #[serde(default)]
    pub context_len: i64,
    #[serde(default = "default_true")]
    pub is_generation: bool,
    /// Some models declare multiple EOS ids; we keep the wire shape as
    /// a list and the scheduler treats any of them as a stop signal.
    #[serde(default)]
    pub hf_eos_token_ids: Option<Vec<i64>>,
    /// Reasoning-model end-of-think marker.  `None` for non-reasoning
    /// models.
    #[serde(default)]
    pub think_end_id: Option<i64>,
}

fn default_true() -> bool {
    true
}
