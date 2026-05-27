// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Integration tests for the tokenizer → scheduler request source.
//!
//! We bind a PULL socket on a private inproc endpoint, push a few
//! wire frames at it, and verify the request source drains them into
//! a `WaitingQueue` in the expected shape.  This exercises the same
//! framing the Python `recv_from_tokenizer` socket speaks.

use std::collections::HashSet;

use sglang_scheduler::queue::WaitingQueue;
use sglang_scheduler::scheduler::{WorkerSnapshot, drain_into};
use sglang_scheduler::transport::{PullSource, PushSink};
use sglang_scheduler::wire::{
    AbortReq, GPUWorkerHandshakeReqOutput, SamplingParamsIPC, TokenizedGenerateReqInput, Wire,
};

fn snapshot_with(eos: Option<Vec<i64>>, context_len: i64) -> WorkerSnapshot {
    let hs = GPUWorkerHandshakeReqOutput {
        max_total_num_tokens: 0,
        max_prefill_tokens: 0,
        max_running_requests: 0,
        max_req_len: 0,
        max_req_input_len: 0,
        random_seed: 0,
        device: "cpu".into(),
        req_to_token_pool_size: 0,
        req_to_token_pool_max_context_len: 0,
        token_to_kv_pool_size: 0,
        max_queued_requests: None,
        is_hybrid_swa: false,
        sliding_window_size: None,
        full_max_total_num_tokens: 0,
        swa_max_total_num_tokens: 0,
        is_dllm: false,
        lora_manager: None,
        token_table: None,
        linear_attn_model_spec: None,
        hybrid_gdn_config: None,
        mamba2_config: None,
        vocab_size: 32000,
        context_len,
        is_generation: true,
        hf_eos_token_ids: eos,
        think_end_id: None,
    };
    WorkerSnapshot::from_handshake(&hs)
}

fn default_snapshot() -> WorkerSnapshot {
    snapshot_with(None, 0)
}

fn generate_req(rid: &str, ids: Vec<i64>, max_new_tokens: u32) -> TokenizedGenerateReqInput {
    let mut sp = SamplingParamsIPC::default();
    sp.max_new_tokens = max_new_tokens;
    TokenizedGenerateReqInput {
        input_ids: ids,
        sampling_params: sp,
        return_logprob: false,
        logprob_start_len: -1,
        top_logprobs_num: 0,
        stream: false,
        input_text: None,
        token_ids_logprob: None,
        mm_inputs: None,
        return_hidden_states: false,
        return_routed_experts: false,
        routed_experts_start_len: 0,
        input_embeds: None,
        positional_embed_overrides: None,
        session_params: None,
        lora_id: None,
        custom_logit_processor: None,
        bootstrap_host: None,
        bootstrap_port: None,
        bootstrap_room: None,
        bootstrap_pair_key: None,
        decode_tp_size: None,
        require_reasoning: false,
        routed_dp_rank: None,
        disagg_prefill_dp_rank: None,
        priority: None,
        extra_key: None,
        routing_key: None,
        no_logs: false,
        return_bytes: false,
        return_entropy: false,
        token_type_ids: None,
        need_wait_for_mm_inputs: None,
        num_items_assigned: None,
        multi_item_delimiter_indices: None,
        time_stats: None,
        rid: Some(rid.into()),
        http_worker_ipc: None,
    }
}

#[test]
fn drains_three_generate_reqs_into_waiting_queue() {
    let ctx = zmq::Context::new();
    let endpoint = "inproc://tok-source-drain";
    let sink = PushSink::bind(&ctx, endpoint).unwrap();
    let source = PullSource::connect(&ctx, endpoint).unwrap();

    for i in 0..3 {
        let req = generate_req(&format!("req-{i}"), vec![1, 2, 3, i as i64], 16);
        sink.send(&Wire::TokenizedGenerateReqInput(req)).unwrap();
    }

    // Give zmq a moment to flush inproc.
    std::thread::sleep(std::time::Duration::from_millis(10));

    let mut waiting = WaitingQueue::new();
    let running: HashSet<String> = HashSet::new();
    let snapshot = default_snapshot();
    let outcome = drain_into(&source, &mut waiting, &running, &snapshot).unwrap();

    assert_eq!(outcome.admitted, 3);
    assert_eq!(outcome.aborted, 0);
    assert_eq!(outcome.dropped_unsupported, 0);
    assert_eq!(waiting.len(), 3);

    let snapshot = waiting.as_slice();
    let rids: Vec<String> = snapshot
        .iter()
        .map(|r| r.read().unwrap().rid.clone())
        .collect();
    assert_eq!(rids, vec!["req-0", "req-1", "req-2"]);
}

#[test]
fn abort_by_rid_drops_from_waiting_queue() {
    let ctx = zmq::Context::new();
    let endpoint = "inproc://tok-source-abort";
    let sink = PushSink::bind(&ctx, endpoint).unwrap();
    let source = PullSource::connect(&ctx, endpoint).unwrap();

    sink.send(&Wire::TokenizedGenerateReqInput(generate_req(
        "alpha",
        vec![1, 2, 3],
        4,
    )))
    .unwrap();
    sink.send(&Wire::TokenizedGenerateReqInput(generate_req(
        "beta",
        vec![4, 5, 6],
        4,
    )))
    .unwrap();
    sink.send(&Wire::AbortReq(AbortReq {
        rid: Some("alpha".into()),
        ..Default::default()
    }))
    .unwrap();

    std::thread::sleep(std::time::Duration::from_millis(10));

    let mut waiting = WaitingQueue::new();
    let running: HashSet<String> = HashSet::new();
    let snapshot = default_snapshot();
    let outcome = drain_into(&source, &mut waiting, &running, &snapshot).unwrap();

    assert_eq!(outcome.admitted, 2);
    assert_eq!(outcome.aborted, 1);
    assert_eq!(waiting.len(), 1);
    let only = &waiting.as_slice()[0];
    assert_eq!(only.read().unwrap().rid, "beta");
}

#[test]
fn abort_all_clears_waiting_queue() {
    let ctx = zmq::Context::new();
    let endpoint = "inproc://tok-source-abort-all";
    let sink = PushSink::bind(&ctx, endpoint).unwrap();
    let source = PullSource::connect(&ctx, endpoint).unwrap();

    for i in 0..4 {
        sink.send(&Wire::TokenizedGenerateReqInput(generate_req(
            &format!("r{i}"),
            vec![i as i64],
            4,
        )))
        .unwrap();
    }
    sink.send(&Wire::AbortReq(AbortReq {
        abort_all: true,
        ..Default::default()
    }))
    .unwrap();

    std::thread::sleep(std::time::Duration::from_millis(10));

    let mut waiting = WaitingQueue::new();
    let running: HashSet<String> = HashSet::new();
    let snapshot = default_snapshot();
    let outcome = drain_into(&source, &mut waiting, &running, &snapshot).unwrap();

    assert_eq!(outcome.admitted, 4);
    assert_eq!(outcome.aborted, 1);
    assert_eq!(waiting.len(), 0);
}

#[test]
fn eos_fallback_from_snapshot_populates_stop_token_ids() {
    let ctx = zmq::Context::new();
    let endpoint = "inproc://tok-source-eos";
    let sink = PushSink::bind(&ctx, endpoint).unwrap();
    let source = PullSource::connect(&ctx, endpoint).unwrap();

    // Request without its own stop_token_ids.
    let req = generate_req("naked", vec![1, 2, 3], 8);
    assert!(req.sampling_params.stop_token_ids.is_none());
    sink.send(&Wire::TokenizedGenerateReqInput(req)).unwrap();

    std::thread::sleep(std::time::Duration::from_millis(10));

    let mut waiting = WaitingQueue::new();
    let running: HashSet<String> = HashSet::new();
    // Worker reports two EOS ids — both should land on the admitted req.
    let snapshot = snapshot_with(Some(vec![2, 7]), 0);
    let outcome = drain_into(&source, &mut waiting, &running, &snapshot).unwrap();

    assert_eq!(outcome.admitted, 1);
    let only = &waiting.as_slice()[0];
    let stops = only
        .read()
        .unwrap()
        .sampling_params
        .stop_token_ids
        .clone()
        .expect("EOS fallback should have populated stop_token_ids");
    assert_eq!(stops, vec![2, 7]);
}

#[test]
fn eos_fallback_respects_existing_stop_tokens() {
    let ctx = zmq::Context::new();
    let endpoint = "inproc://tok-source-eos-existing";
    let sink = PushSink::bind(&ctx, endpoint).unwrap();
    let source = PullSource::connect(&ctx, endpoint).unwrap();

    let mut req = generate_req("explicit", vec![1, 2, 3], 8);
    req.sampling_params.stop_token_ids = Some(vec![999]);
    sink.send(&Wire::TokenizedGenerateReqInput(req)).unwrap();

    std::thread::sleep(std::time::Duration::from_millis(10));

    let mut waiting = WaitingQueue::new();
    let running: HashSet<String> = HashSet::new();
    let snapshot = snapshot_with(Some(vec![2, 7]), 0);
    drain_into(&source, &mut waiting, &running, &snapshot).unwrap();

    let only = &waiting.as_slice()[0];
    let stops = only
        .read()
        .unwrap()
        .sampling_params
        .stop_token_ids
        .clone()
        .expect("explicit stop_token_ids must survive");
    // Client's explicit choice wins over the model default.
    assert_eq!(stops, vec![999]);
}

#[test]
fn context_len_truncates_oversized_prompt() {
    let ctx = zmq::Context::new();
    let endpoint = "inproc://tok-source-ctx-cap";
    let sink = PushSink::bind(&ctx, endpoint).unwrap();
    let source = PullSource::connect(&ctx, endpoint).unwrap();

    // Prompt 100 tokens, max_new 50.  With context_len = 64 the snapshot
    // truncates the prompt to 64 - 1 - 50 = 13 tokens.
    let req = generate_req("oversized", (0..100).collect(), 50);
    sink.send(&Wire::TokenizedGenerateReqInput(req)).unwrap();

    std::thread::sleep(std::time::Duration::from_millis(10));

    let mut waiting = WaitingQueue::new();
    let running: HashSet<String> = HashSet::new();
    let snapshot = snapshot_with(None, 64);
    drain_into(&source, &mut waiting, &running, &snapshot).unwrap();

    let only = &waiting.as_slice()[0];
    assert_eq!(only.read().unwrap().origin_input_ids.len(), 13);
}
