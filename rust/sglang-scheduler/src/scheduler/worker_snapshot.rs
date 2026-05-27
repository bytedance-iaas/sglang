// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Cached view of the worker handshake reply.
//!
//! The handshake (`GPUWorkerHandshakeReqOutput`) carries a mix of:
//!   * sizing fields (`max_total_num_tokens`, `req_to_token_pool_size`, …)
//!     — consumed once at boot to size the page tracker / req pool.
//!   * model-config scalars (`vocab_size`, `context_len`, EOS, …) —
//!     consumed throughout the loop: `vocab_size` enters every
//!     `SamplingBatchInfo` payload, EOS defaults onto every admitted
//!     `Req` that didn't ship its own stop tokens.
//!
//! `WorkerSnapshot` is the thin readonly handle the scheduler keeps
//! around for the latter category.  Built once at handshake time and
//! threaded through admission + payload construction.

use crate::wire::GPUWorkerHandshakeReqOutput;

#[derive(Debug, Clone)]
pub struct WorkerSnapshot {
    pub device: String,
    pub vocab_size: i32,
    pub context_len: u32,
    pub is_generation: bool,
    /// Default stop ids for reqs that arrive without their own.  Empty
    /// `Vec` means "no default" (model didn't ship one).
    pub default_stop_token_ids: Vec<i32>,
    pub think_end_id: Option<i32>,
}

impl WorkerSnapshot {
    pub fn from_handshake(hs: &GPUWorkerHandshakeReqOutput) -> Self {
        // Older workers without the new fields decode them as the
        // defaults (`0` / `true` / `None`) — fall back to keep the same
        // pre-Option-B behaviour: `vocab_size = 0` rides through to the
        // sampling info, and no EOS default is injected.
        let vocab_size = if hs.vocab_size > 0 && hs.vocab_size <= i32::MAX as i64 {
            hs.vocab_size as i32
        } else {
            0
        };
        let context_len = if hs.context_len > 0 && hs.context_len <= u32::MAX as i64 {
            hs.context_len as u32
        } else {
            0
        };
        let default_stop_token_ids = hs
            .hf_eos_token_ids
            .as_ref()
            .map(|v| v.iter().map(|&t| t as i32).collect())
            .unwrap_or_default();
        let think_end_id = hs.think_end_id.map(|v| v as i32);
        Self {
            device: hs.device.clone(),
            vocab_size,
            context_len,
            is_generation: hs.is_generation,
            default_stop_token_ids,
            think_end_id,
        }
    }
}
