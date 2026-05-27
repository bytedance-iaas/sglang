// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! `SamplingParams` — sampling configuration carried with each request.
//!
//! Mirrors the subset of fields the scheduler actually inspects from
//! `python/sglang/srt/sampling/sampling_params.py`.  Penalizer state
//! (`BatchedPenalizerOrchestrator`) is intentionally left out — that's a
//! whole subsystem of its own.  When penalties land, add them as a
//! separate `Penalizers` struct on `SamplingBatchInfo`, not here.

use serde::{Deserialize, Serialize};

/// Sentinel for "consider the full vocabulary" — matches the constant
/// of the same name in the Python frontend (`sglang-frontend/src/ipc/wire.rs`).
pub const TOP_K_ALL: i32 = 1 << 30;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub max_new_tokens: u32,
    pub temperature: f64,
    pub top_p: f64,
    /// `-1` means "consider the full vocab" (normalized to `TOP_K_ALL`).
    pub top_k: i32,
    pub min_p: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
    pub repetition_penalty: f64,
    pub min_new_tokens: u32,
    pub n: u32,
    pub stop: Option<Vec<String>>,
    pub stop_token_ids: Option<Vec<i32>>,
    pub ignore_eos: bool,
    pub skip_special_tokens: bool,
    pub spaces_between_special_tokens: bool,
    pub no_stop_trim: bool,
    pub seed: Option<i64>,
    pub return_logprob: bool,
    pub logprob_start_len: i64,
    pub top_logprobs_num: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            max_new_tokens: 128,
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            min_new_tokens: 0,
            n: 1,
            stop: None,
            stop_token_ids: None,
            ignore_eos: false,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            no_stop_trim: false,
            seed: None,
            return_logprob: false,
            logprob_start_len: -1,
            top_logprobs_num: 0,
        }
    }
}

impl SamplingParams {
    /// `top_k = -1` → `TOP_K_ALL`, identical to `SamplingParams.normalize()`
    /// in `sampling_params.py`.
    pub fn normalize(&mut self) {
        if self.top_k < 0 {
            self.top_k = TOP_K_ALL;
        }
    }
}
