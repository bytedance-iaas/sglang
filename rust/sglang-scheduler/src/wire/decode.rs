// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Decode-step hot path (msgpack-typed, Rust-decodable).
//!
//! Source: `python/sglang/srt/managers/io_struct/msgpack_struct.py`
//! classes `DecodeStepControlReq`, `DeferredAllocIPC`, `DecodeForwardSlimOutput`.
//!
//! These are the only wire types on the critical per-step path; getting
//! their shape right is what makes the rest of the scheduler portable.

use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;

use crate::wire::tensor_ipc::TensorIPC;

/// Delta-only control message for an in-cache decode step.  Stable batch
/// fields (req_pool_indices, lora_ids, forward_mode, sampling info) live
/// on the worker's cached batch and are not re-sent.
///
/// FutureMap fields (`input_slot` / `output_slot`) drive the 2-ahead
/// pipeline contract; leave both `None` for the standard 1-ahead path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeStepControlReq {
    pub seq_lens: TensorIPC,
    pub seq_lens_cpu: TensorIPC,
    pub orig_seq_lens: TensorIPC,
    pub seq_lens_sum: i64,

    /// `None` when `input_slot` is set — worker resolves input ids from
    /// its GPU FutureMap slot instead of the payload.
    #[serde(default)]
    pub input_ids: Option<TensorIPC>,

    #[serde(default)]
    pub indices_to_free: Option<TensorIPC>,

    // Mamba state — `None` for non-Mamba models (zero wire overhead).
    #[serde(default)]
    pub mamba_track_indices: Option<TensorIPC>,
    #[serde(default)]
    pub mamba_track_mask: Option<TensorIPC>,
    #[serde(default)]
    pub mamba_track_seqlens: Option<TensorIPC>,

    // DP-attention token counts (`None` for non-DP).
    #[serde(default)]
    pub global_num_tokens: Option<Vec<i64>>,
    #[serde(default)]
    pub global_num_tokens_for_logprob: Option<Vec<i64>>,

    // FutureMap pipeline contract.  See `TpWorkerServer._future_tokens`.
    #[serde(default)]
    pub input_slot: Option<i64>,
    #[serde(default)]
    pub output_slot: Option<i64>,
}

/// Worker → scheduler reply describing the KV slot indices allocated for
/// the current step.  The Rust scheduler uses these to update its
/// CPU-side req_to_token table and `CpuPageTracker`.
///
/// Field shape mirrors `DeferredAllocIPC` in `msgpack_struct.py`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeferredAllocIPC {
    /// `"decode"` or `"extend"`.
    pub mode: String,
    pub req_pool_indices: TensorIPC,
    pub out_cache_loc: TensorIPC,

    // Decode-only.
    #[serde(default)]
    pub seq_lens_minus1: Option<TensorIPC>,

    // Extend-only.
    #[serde(default)]
    pub prefix_lens: Option<TensorIPC>,
    #[serde(default)]
    pub extend_lens: Option<TensorIPC>,

    pub free_pages_remaining: i64,
}

/// Slim per-step reply on the hot path — the worker's view of what just
/// happened on the GPU.  Rich-Python-object fields (`logits_output`,
/// `routed_experts_output`, `expert_distribution_metrics`, `next_draft_input`)
/// are shipped as opaque pickle blobs because they only matter for rare
/// paths (logprob streaming, MoE telemetry) the Rust scheduler ignores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeForwardSlimOutput {
    pub next_token_ids: TensorIPC,

    #[serde(default)]
    pub deferred_alloc: Option<DeferredAllocIPC>,
    #[serde(default)]
    pub accept_lens: Option<TensorIPC>,

    #[serde(default)]
    pub can_run_cuda_graph: bool,
    #[serde(default)]
    pub num_accepted_drafts: i64,
    #[serde(default)]
    pub num_accepted_drafts_per_req_cpu: Option<Vec<i64>>,

    // Opaque pickle blobs for rare paths.
    #[serde(default)]
    pub logits_output_pickle: Option<ByteBuf>,
    #[serde(default)]
    pub routed_experts_output_pickle: Option<ByteBuf>,
    #[serde(default)]
    pub expert_distribution_metrics_pickle: Option<ByteBuf>,
    #[serde(default)]
    pub next_draft_input_pickle: Option<ByteBuf>,
}
