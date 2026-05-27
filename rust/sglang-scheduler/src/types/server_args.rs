// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Minimal subset of `ServerArgs` that the scheduler actually reads.
//!
//! See `python/sglang/srt/server_args.py` for the full set (~200 fields).
//! Most affect either tokenizer or model-runner configuration, neither
//! of which lives in this crate.  The fields below are the ones the
//! Python `Scheduler.__init__` reads while building the queue, policy,
//! and admission flow.

#[derive(Debug, Clone)]
pub struct ServerArgs {
    pub model_path: String,

    /// KV pool page size in tokens.  Must match the value the GPU worker
    /// reports in its handshake (`req_to_token_pool_max_context_len`
    /// rounds to a multiple of this).
    pub page_size: u32,

    /// Max simultaneous running reqs in a decode batch.
    pub max_running_requests: u32,

    /// Max tokens per prefill iter.
    pub max_prefill_tokens: u32,

    /// Per-req max output length (cap on `sampling_params.max_new_tokens`).
    pub max_total_tokens: Option<u32>,

    /// Scheduling policy name — see `policies/` module.  Defaults to FCFS.
    pub schedule_policy: String,

    /// Reserved-token headroom for retraction safety.  Mirrors
    /// `--schedule-conservativeness` / the related env vars.
    pub schedule_conservativeness: f32,

    /// How often the scheduler emits a `Decode batch ...` log line.
    /// Mirrors `--decode-log-interval` on the Python side; the Python
    /// default is `40`.  `0` disables periodic decode-stats logging
    /// (every-iter prefill stats still fire).
    pub decode_log_interval: u32,

    /// Disable the radix prefix cache and fall back to the
    /// `ChunkedCache` semantics (no prefix matching, no cross-req
    /// KV sharing, every req frees all its KV slots on finish).
    /// Mirrors Python's `--disable-radix-cache` flag (see
    /// `chunk_cache.py`).  Useful when the radix cache is suspected
    /// of causing cross-request contamination or correctness issues.
    pub disable_radix_cache: bool,
}

impl Default for ServerArgs {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            page_size: 16,
            max_running_requests: 256,
            max_prefill_tokens: 8192,
            max_total_tokens: None,
            schedule_policy: "fcfs".into(),
            schedule_conservativeness: 1.0,
            decode_log_interval: 40,
            disable_radix_cache: false,
        }
    }
}
