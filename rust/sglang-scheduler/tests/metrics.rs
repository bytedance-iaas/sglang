// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Tests for `SchedulerMetrics` throughput logger.  No worker / no
//! sockets — pure formatting + interval logic.

use std::sync::{Arc, RwLock};

use sglang_scheduler::memory::CpuPageTracker;
use sglang_scheduler::scheduler::SchedulerMetrics;
use sglang_scheduler::types::{ForwardMode, Req, SamplingParams, ScheduleBatch};

const PAGE: i64 = 16;

fn make_req(rid: &str) -> Arc<RwLock<Req>> {
    Arc::new(RwLock::new(Req::new(
        rid.into(),
        vec![1, 2, 3, 4],
        SamplingParams::default(),
    )))
}

fn decode_batch(num_reqs: usize) -> ScheduleBatch {
    let mut batch = ScheduleBatch::new(ForwardMode::Decode);
    for i in 0..num_reqs {
        batch.reqs.push(make_req(&format!("r{i}")));
        batch.req_pool_indices.push(i as u32);
        batch.seq_lens.push(5);
        batch.orig_seq_lens.push(4);
        batch.input_ids.push(99);
    }
    batch
}

#[test]
fn decode_step_emits_log_only_at_interval() {
    let mut m = SchedulerMetrics::new();
    let tracker = CpuPageTracker::new(8, PAGE);
    let batch = decode_batch(1);

    // Interval = 4: steps 1, 2, 3 are silent; step 4 emits.
    let interval = 4u32;
    for _ in 0..(interval - 1) {
        let out = m.report_decode_step(&batch, 1, true, &tracker, 0, interval);
        assert!(out.is_none(), "should not emit before hitting the interval");
    }
    let out = m.report_decode_step(&batch, 1, true, &tracker, 0, interval);
    let msg = out.expect("4th decode step must emit a log line");
    assert!(msg.contains("Decode batch"));
    assert!(msg.contains("#running-req: 1"));
    assert!(msg.contains("gen throughput (token/s)"));
}

#[test]
fn decode_step_zero_interval_disables_logging() {
    let mut m = SchedulerMetrics::new();
    let tracker = CpuPageTracker::new(8, PAGE);
    let batch = decode_batch(2);
    for _ in 0..50 {
        assert!(m
            .report_decode_step(&batch, 2, false, &tracker, 0, 0)
            .is_none());
    }
}

#[test]
fn decode_step_counts_tokens_across_interval() {
    // Bursty TPS sanity: feed 8 steps with 3 tokens each → 24 tokens.
    // Interval 4 emits twice; the first emit covers 12 tokens.
    let mut m = SchedulerMetrics::new();
    let tracker = CpuPageTracker::new(8, PAGE);
    let batch = decode_batch(3);
    let interval = 4u32;
    let mut emits: Vec<String> = Vec::new();
    for _ in 0..8 {
        if let Some(s) = m.report_decode_step(&batch, 3, true, &tracker, 0, interval) {
            emits.push(s);
        }
    }
    assert_eq!(emits.len(), 2, "two emits expected for 8 steps @ interval 4");

    // The throughput-counter resets after each emit, so num_generated_tokens
    // must be 0 going into the next interval window.
    assert_eq!(m.num_generated_tokens, 0);
}

#[test]
fn prefill_step_always_emits_with_input_throughput() {
    let mut m = SchedulerMetrics::new();
    let tracker = CpuPageTracker::new(8, PAGE);
    let mut batch = ScheduleBatch::new(ForwardMode::Extend);
    batch.reqs.push(make_req("p0"));
    batch.req_pool_indices.push(0);
    batch.seq_lens.push(64);
    batch.orig_seq_lens.push(64);
    batch.input_ids.extend(std::iter::repeat_n(7, 64));

    let msg = m
        .report_prefill_step(&batch, 64, 0, &tracker, 0)
        .expect("prefill iter emits every time");
    assert!(msg.contains("Prefill batch"));
    assert!(msg.contains("#new-seq: 1"));
    assert!(msg.contains("#new-token: 64"));
    assert!(msg.contains("input throughput (token/s)"));
}

#[test]
fn decode_step_ignored_on_extend_batches() {
    let mut m = SchedulerMetrics::new();
    let tracker = CpuPageTracker::new(8, PAGE);
    let mut batch = ScheduleBatch::new(ForwardMode::Extend);
    batch.reqs.push(make_req("x"));
    batch.req_pool_indices.push(0);
    batch.seq_lens.push(8);
    batch.orig_seq_lens.push(8);
    batch.input_ids.extend(std::iter::repeat_n(1, 8));

    // report_decode_step should be a no-op when forward_mode is Extend.
    assert!(m
        .report_decode_step(&batch, 8, true, &tracker, 0, 1)
        .is_none());
    assert_eq!(m.forward_ct_decode, 0);
}
