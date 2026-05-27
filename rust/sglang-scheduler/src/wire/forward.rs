// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Full forward-pass request types.  Sent when the worker has no cached
//! decode batch (first step, after composition change, prefill, …).
//!
//! Source: msgspec mirrors of the dataclasses in
//! `python/sglang/srt/managers/io_struct/pickle_struct.py` —
//! `ForwardBatchGenerationReq`, `ForwardBatchEmbeddingReq`,
//! `ForwardBatchSplitPrefillReq`.
//!
//! NOTE: the Python wire currently still carries a full `ModelWorkerBatch`
//! inside these structs (lots of fields).  Until that's fully ported,
//! the Rust mirror keeps it as a generic msgpack value so the scheduler
//! can ferry it through without inspecting the fields.

use rmpv::Value;
use serde::{Deserialize, Serialize};

/// Full forward pass for generation (prefill / decode / mixed).
///
/// The `batch` field is intentionally opaque (`rmpv::Value`) — a full
/// `ModelWorkerBatch` mirror is its own port, with ~30 typed fields and
/// nested sampling/penalizer state.  The scheduler builds it once in
/// Python today and passes it through msgpack without inspection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardBatchGenerationReq {
    pub batch: Value,
    #[serde(default)]
    pub pp_proxy_tensors: Option<Value>,
    #[serde(default)]
    pub is_verify: bool,
    #[serde(default)]
    pub skip_attn_backend_init: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardBatchEmbeddingReq {
    pub batch: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardBatchSplitPrefillReq {
    pub batch: Value,
}
