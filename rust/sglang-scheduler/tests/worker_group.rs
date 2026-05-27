// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Integration tests for `WorkerClientGroup` fan-out semantics.
//!
//! The tests spin up N PAIR sockets in-process to play the role of TP
//! workers.  Each "fake worker" reads requests and responds with a
//! canned typed reply.  The group is then asked to do various RPCs and
//! we verify:
//!   * broadcast_leader_only fans out to every rank but returns the
//!     leader's reply only;
//!   * broadcast_all_confirm collects every reply;
//!   * handshake's broadcast_leader_only contract holds (every rank
//!     sees the input, leader's reply shapes the snapshot);
//!   * TP=1 still works (single-element group);
//!   * non-leader stragglers are drained — a follow-up RPC after a
//!     broadcast_leader_only doesn't get a stale reply on its socket.

use std::sync::{Arc, Barrier};
use std::thread;

use sglang_scheduler::scheduler::WorkerClientGroup;
use sglang_scheduler::wire::{
    DecodeForwardSlimOutput, GPUWorkerHandshakeReqOutput, GetMemUsageReqOutput, LoRAUpdateOutput,
    TensorIPC, UnloadLoRAAdapterReqInput, Wire,
};

/// Fake worker: bind a PAIR endpoint and respond to each request with a
/// canned reply from `replies` (one reply per incoming request, in order).
fn spawn_fake_worker(
    ctx: zmq::Context,
    endpoint: String,
    replies: Vec<Wire>,
    ready: Arc<Barrier>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let socket = ctx.socket(zmq::PAIR).expect("socket");
        socket.bind(&endpoint).expect("bind");
        ready.wait();
        for reply in replies {
            // Read the incoming request (we don't inspect the body —
            // just count, by reading both frames).
            let _magic = socket.recv_bytes(0).expect("recv magic");
            let _body = socket.recv_bytes(0).expect("recv body");
            // Send the canned reply with the same framing.
            let bytes = rmp_serde::to_vec_named(&reply).expect("encode");
            socket
                .send_multipart([b"0xSG02".as_ref(), bytes.as_slice()], 0)
                .expect("send");
        }
    })
}

fn handshake_reply(rank: i64, vocab: i64) -> Wire {
    Wire::GPUWorkerHandshakeReqOutput(GPUWorkerHandshakeReqOutput {
        max_total_num_tokens: 100,
        max_prefill_tokens: 50,
        max_running_requests: 4,
        max_req_len: 32,
        max_req_input_len: 32,
        random_seed: rank,
        device: format!("cuda:{rank}"),
        req_to_token_pool_size: 4,
        req_to_token_pool_max_context_len: 32,
        token_to_kv_pool_size: 64,
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
        vocab_size: vocab,
        context_len: 32,
        is_generation: true,
        hf_eos_token_ids: Some(vec![2]),
        think_end_id: None,
    })
}

fn mem_usage_reply(weight_load: f64) -> Wire {
    Wire::GetMemUsageReqOutput(GetMemUsageReqOutput {
        weight_load_mem_usage: weight_load,
        graph_mem_usage: 1024.0,
    })
}

fn unload_lora_reply(success: bool) -> Wire {
    Wire::LoRAUpdateOutput(LoRAUpdateOutput {
        success,
        error_message: None,
        loaded_adapters: None,
    })
}

#[test]
fn tp1_group_works_end_to_end() {
    let ctx = zmq::Context::new();
    let endpoint = "inproc://wg-tp1".to_string();
    let ready = Arc::new(Barrier::new(2));

    // Worker will serve: handshake → get_mem_usage.
    let worker = spawn_fake_worker(
        ctx.clone(),
        endpoint.clone(),
        vec![handshake_reply(0, 32000), mem_usage_reply(512.0)],
        ready.clone(),
    );
    ready.wait();

    let group = WorkerClientGroup::connect_all(&ctx, &[endpoint]).expect("connect");
    assert_eq!(group.len(), 1);
    assert!(group.is_tp1());
    assert_eq!(group.handshake().vocab_size, 32000);
    assert_eq!(group.handshake().device, "cuda:0");

    let mem = group.get_mem_usage().expect("mem");
    assert_eq!(mem.weight_load_mem_usage, 512.0);
    assert_eq!(mem.graph_mem_usage, 1024.0);

    worker.join().expect("worker join");
}

#[test]
fn tp_group_handshake_broadcasts_and_picks_leader() {
    let ctx = zmq::Context::new();
    let endpoints: Vec<String> = (0..3)
        .map(|i| format!("inproc://wg-hsbroadcast-{i}"))
        .collect();

    let ready = Arc::new(Barrier::new(4)); // 3 workers + main thread
    let mut handles = Vec::new();
    for (i, ep) in endpoints.iter().enumerate() {
        // Each rank reports a different vocab_size so we can verify the
        // group picks rank 0's value and warns about the mismatch.
        let replies = vec![handshake_reply(i as i64, 32000 + i as i64)];
        handles.push(spawn_fake_worker(
            ctx.clone(),
            ep.clone(),
            replies,
            ready.clone(),
        ));
    }
    ready.wait();

    let group = WorkerClientGroup::connect_all(&ctx, &endpoints).expect("connect");
    assert_eq!(group.len(), 3);
    assert!(!group.is_tp1());
    // Leader is rank 0: vocab_size 32000, device cuda:0.
    assert_eq!(group.handshake().vocab_size, 32000);
    assert_eq!(group.handshake().device, "cuda:0");

    for h in handles {
        h.join().expect("worker join");
    }
}

#[test]
fn broadcast_leader_only_returns_leader_reply_and_drains_stragglers() {
    let ctx = zmq::Context::new();
    let endpoints: Vec<String> = (0..3)
        .map(|i| format!("inproc://wg-broadcast-leader-{i}"))
        .collect();

    let ready = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();
    // Each worker handles handshake then get_mem_usage.  Workers report
    // *different* weight-load values so the test asserts the leader's
    // value is returned (not e.g. rank 2's).
    for (i, ep) in endpoints.iter().enumerate() {
        let replies = vec![
            handshake_reply(i as i64, 32000),
            mem_usage_reply(1000.0 + i as f64),
        ];
        handles.push(spawn_fake_worker(
            ctx.clone(),
            ep.clone(),
            replies,
            ready.clone(),
        ));
    }
    ready.wait();

    let group = WorkerClientGroup::connect_all(&ctx, &endpoints).expect("connect");
    let mem = group.get_mem_usage().expect("mem");
    // Leader is rank 0 → weight_load_mem_usage == 1000.0.
    assert_eq!(mem.weight_load_mem_usage, 1000.0);

    for h in handles {
        h.join().expect("worker join");
    }
}

#[test]
fn broadcast_all_confirm_collects_every_reply() {
    let ctx = zmq::Context::new();
    let endpoints: Vec<String> = (0..3)
        .map(|i| format!("inproc://wg-broadcast-all-{i}"))
        .collect();

    let ready = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();
    // handshake + unload_lora.  All three workers report success.
    for (i, ep) in endpoints.iter().enumerate() {
        let replies = vec![handshake_reply(i as i64, 32000), unload_lora_reply(true)];
        handles.push(spawn_fake_worker(
            ctx.clone(),
            ep.clone(),
            replies,
            ready.clone(),
        ));
    }
    ready.wait();

    let group = WorkerClientGroup::connect_all(&ctx, &endpoints).expect("connect");
    let req = UnloadLoRAAdapterReqInput {
        lora_name: "adapter1".into(),
    };
    // The group returns the leader's reply; if any other rank had
    // failed, the macro would warn but still return leader.
    let out = group.unload_lora_adapter(req).expect("unload");
    assert!(out.success);

    for h in handles {
        h.join().expect("worker join");
    }
}

#[test]
fn consecutive_broadcast_leader_only_calls_stay_lockstep() {
    // Regression test for the drain semantics: if straggler replies
    // aren't drained after RPC #1, RPC #2's leader-side recv would pick
    // up RPC #1's stale reply from a non-leader socket buffer.  We
    // run two consecutive RPCs and verify both return the correct
    // leader values.
    let ctx = zmq::Context::new();
    let endpoints: Vec<String> = (0..3)
        .map(|i| format!("inproc://wg-lockstep-{i}"))
        .collect();

    let ready = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();
    for (i, ep) in endpoints.iter().enumerate() {
        let replies = vec![
            handshake_reply(i as i64, 32000),
            mem_usage_reply(2000.0 + i as f64),
            // Second RPC: emit a different value per rank again.
            Wire::DecodeForwardSlimOutput(DecodeForwardSlimOutput {
                next_token_ids: TensorIPC::from_i64(&[10 + i as i64]),
                deferred_alloc: None,
                accept_lens: None,
                can_run_cuda_graph: i == 0,
                num_accepted_drafts: 0,
                num_accepted_drafts_per_req_cpu: None,
                logits_output_pickle: None,
                routed_experts_output_pickle: None,
                expert_distribution_metrics_pickle: None,
                next_draft_input_pickle: None,
            }),
        ];
        handles.push(spawn_fake_worker(
            ctx.clone(),
            ep.clone(),
            replies,
            ready.clone(),
        ));
    }
    ready.wait();

    let group = WorkerClientGroup::connect_all(&ctx, &endpoints).expect("connect");
    let mem = group.get_mem_usage().expect("rpc1");
    assert_eq!(
        mem.weight_load_mem_usage, 2000.0,
        "RPC1 must read rank-0's mem reply"
    );

    let decode = group
        .decode_step(sglang_scheduler::wire::DecodeStepControlReq {
            seq_lens: TensorIPC::from_i64(&[]),
            seq_lens_cpu: TensorIPC::from_i64(&[]),
            orig_seq_lens: TensorIPC::from_i64(&[]),
            seq_lens_sum: 0,
            input_ids: None,
            indices_to_free: None,
            mamba_track_indices: None,
            mamba_track_mask: None,
            mamba_track_seqlens: None,
            global_num_tokens: None,
            global_num_tokens_for_logprob: None,
            input_slot: None,
            output_slot: None,
        })
        .expect("rpc2");
    // Rank 0's reply only — `can_run_cuda_graph: true` is the leader value.
    assert!(
        decode.can_run_cuda_graph,
        "RPC2 must read rank-0's decode reply (not rank-1 or rank-2)"
    );
    assert_eq!(
        decode.next_token_ids.as_i64().expect("int64"),
        &[10i64][..],
        "leader's next_token_ids only"
    );

    for h in handles {
        h.join().expect("worker join");
    }
}
