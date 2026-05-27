// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! DP-attention sync — Rust mirror of
//! `python/sglang/srt/managers/scheduler_dp_attn_mixin.py`.
//!
//! In Python this mixin coordinates multiple DP ranks via a torch
//! `ProcessGroup` `all_gather` so that every rank sees the same
//! `global_num_tokens` / `global_forward_mode` before sending its
//! `ModelWorkerBatch`.  The Rust scheduler today runs a single DP rank
//! per process (one scheduler drives all TP ranks), so the only thing
//! the wire payload needs is the locally-computed equivalent: the DP
//! group is size 1, so `global_num_tokens == [local_num_tokens]`.
//!
//! When/if a multi-DP Rust scheduler lands, the all-gather will need a
//! real cross-process group (likely gloo or NCCL) — until then this
//! module just computes the local fallback.

use crate::types::{ForwardMode, ScheduleBatch};

/// Output of `prepare_local_dp_attention_sync` — values the caller
/// stamps into the `ModelWorkerBatchPayload`.
#[derive(Debug, Clone)]
pub struct DpSync {
    pub global_num_tokens: Vec<i64>,
    pub global_num_tokens_for_logprob: Vec<i64>,
    pub global_forward_mode: i32,
    pub can_run_dp_cuda_graph: bool,
    pub tbo_split_seq_index: Option<i64>,
}

/// Local (single-DP) fallback that matches the Python equivalent's
/// behaviour when `dp_size == 1`.  Tokens for logprob are conservatively
/// estimated as the same as `num_tokens` (Python overrides this when
/// `return_logprob` is true on any req — the worker tolerates over-
/// estimation).
pub fn prepare_local_dp_attention_sync(batch: &ScheduleBatch) -> DpSync {
    let local_num_tokens = batch.input_ids.len() as i64;
    let forward_mode = forward_mode_int(batch.forward_mode);
    DpSync {
        global_num_tokens: vec![local_num_tokens],
        global_num_tokens_for_logprob: vec![local_num_tokens],
        global_forward_mode: forward_mode,
        can_run_dp_cuda_graph: false,
        tbo_split_seq_index: None,
    }
}

fn forward_mode_int(fm: ForwardMode) -> i32 {
    use crate::wire::forward_mode_wire as fmw;
    match fm {
        ForwardMode::Extend => fmw::EXTEND,
        ForwardMode::Decode => fmw::DECODE,
        ForwardMode::Mixed => fmw::MIXED,
        ForwardMode::Idle => fmw::IDLE,
        ForwardMode::Prebuilt => fmw::PREBUILT,
        ForwardMode::SplitPrefill => fmw::SPLIT_PREFILL,
        ForwardMode::TargetVerify => fmw::TARGET_VERIFY,
        ForwardMode::DraftExtend => fmw::DRAFT_EXTEND,
    }
}
