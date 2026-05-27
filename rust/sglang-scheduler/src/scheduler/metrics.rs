// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Throughput logging for the scheduler event loop.
//!
//! Source: `python/sglang/srt/observability/scheduler_metrics_mixin.py`
//! — specifically the `report_prefill_stats` / `report_decode_stats`
//! human-readable log lines.  The full mixin is ~1k LOC covering
//! Prometheus / KV-events / MFU / DP cooperation / PD disaggregation /
//! spec decoding / LoRA / HiCache / routing keys / streaming sessions /
//! device timer.  All of those are **not** ported here; this module is
//! the focused minimum to surface decode throughput, the metric the
//! Rust scheduler was missing on single-req workloads.
//!
//! Each iteration:
//!   * Prefill iter → emit `Prefill batch, #new-seq: …, input throughput: …`.
//!   * Decode iter → bump counters; every `decode_log_interval` steps
//!     emit `Decode batch, #running-req: …, gen throughput (token/s): …`.

use std::time::Instant;

use crate::memory::CpuPageTracker;
use crate::types::{ForwardMode, ScheduleBatch};

#[derive(Debug)]
pub struct SchedulerMetrics {
    /// Tokens generated since the last decode-stats emit.  Numerator
    /// for `gen throughput`.
    pub num_generated_tokens: u64,
    /// Cumulative decode-step count.  We emit every
    /// `decode_log_interval` increments.
    pub forward_ct_decode: u64,

    last_decode_tic: Instant,
    last_prefill_tic: Instant,
}

impl Default for SchedulerMetrics {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            num_generated_tokens: 0,
            forward_ct_decode: 0,
            last_decode_tic: now,
            last_prefill_tic: now,
        }
    }
}

impl SchedulerMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the human-readable log line for one prefill iter.  Always
    /// returns `Some(...)` (prefill stats fire on every iter, matching
    /// Python).
    pub fn report_prefill_step(
        &mut self,
        batch: &ScheduleBatch,
        new_input_tokens: u64,
        cached_tokens: u64,
        page_tracker: &CpuPageTracker,
        waiting_queue_len: usize,
    ) -> Option<String> {
        if !batch.forward_mode.is_extend() {
            return None;
        }
        let now = Instant::now();
        let gap = now.duration_since(self.last_prefill_tic).as_secs_f64();
        self.last_prefill_tic = now;
        let input_tps = if gap > 0.0 {
            new_input_tokens as f64 / gap
        } else {
            0.0
        };
        let usage = pool_usage_ratio(page_tracker);
        let msg = format!(
            "Prefill batch, #new-seq: {}, #new-token: {}, #cached-token: {}, \
             token usage: {:.2}, #running-req: {}, #queue-req: {}, \
             input throughput (token/s): {:.2}",
            batch.batch_size(),
            new_input_tokens,
            cached_tokens,
            usage,
            batch.batch_size(),
            waiting_queue_len,
            input_tps,
        );
        Some(msg)
    }

    /// Bump per-iter counters and, every `decode_log_interval` steps,
    /// return a formatted `Decode batch, …` log line.
    ///
    /// `tokens_generated_this_step` is the number of newly sampled
    /// tokens this iteration (= batch size on the standard non-spec
    /// path; spec decoding adds accepted drafts).
    pub fn report_decode_step(
        &mut self,
        batch: &ScheduleBatch,
        tokens_generated_this_step: u64,
        can_run_cuda_graph: bool,
        page_tracker: &CpuPageTracker,
        waiting_queue_len: usize,
        decode_log_interval: u32,
    ) -> Option<String> {
        if !matches!(batch.forward_mode, ForwardMode::Decode) {
            return None;
        }
        self.num_generated_tokens += tokens_generated_this_step;
        self.forward_ct_decode += 1;

        if decode_log_interval == 0
            || !self
                .forward_ct_decode
                .is_multiple_of(decode_log_interval as u64)
        {
            return None;
        }

        let now = Instant::now();
        let gap = now.duration_since(self.last_decode_tic).as_secs_f64();
        self.last_decode_tic = now;
        let gen_tps = if gap > 0.0 {
            self.num_generated_tokens as f64 / gap
        } else {
            0.0
        };
        let usage = pool_usage_ratio(page_tracker);
        let num_running = batch.batch_size();
        self.num_generated_tokens = 0;

        let msg = format!(
            "Decode batch, #running-req: {}, token usage: {:.2}, \
             cuda graph: {}, gen throughput (token/s): {:.2}, #queue-req: {}",
            num_running, usage, can_run_cuda_graph, gen_tps, waiting_queue_len,
        );
        Some(msg)
    }
}

fn pool_usage_ratio(tracker: &CpuPageTracker) -> f64 {
    let total = (tracker.total_pages() * tracker.page_size()) as f64;
    if total <= 0.0 {
        return 0.0;
    }
    let avail = tracker.available_size() as f64;
    let used = (total - avail).max(0.0);
    used / total
}
