// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Wire-format round-trip sanity checks.  These don't exercise the
//! Python side directly — that needs a running TpWorkerServer — but
//! they confirm `rmp-serde` produces the expected tagged-fixmap shape
//! and that the dec round-trip via the unified `Wire` enum picks the
//! right variant on tag.

use serde_bytes::ByteBuf;
use sglang_scheduler::wire::{
    DecodeForwardSlimOutput, DecodeStepControlReq, DeferredAllocIPC,
    GPUWorkerHandshakeReqOutput, TensorIPC, Wire,
};

fn roundtrip(w: Wire) -> Wire {
    let bytes = rmp_serde::to_vec_named(&w).expect("encode");
    rmp_serde::from_slice::<Wire>(&bytes).expect("decode")
}

#[test]
fn handshake_output_roundtrips() {
    let original = Wire::GPUWorkerHandshakeReqOutput(GPUWorkerHandshakeReqOutput {
        max_total_num_tokens: 388544,
        max_prefill_tokens: 8192,
        max_running_requests: 256,
        max_req_len: 32768,
        max_req_input_len: 32768,
        random_seed: 42,
        device: "cuda:0".into(),
        req_to_token_pool_size: 4096,
        req_to_token_pool_max_context_len: 32768,
        token_to_kv_pool_size: 388544,
        max_queued_requests: Some(1024),
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
        context_len: 32768,
        is_generation: true,
        hf_eos_token_ids: Some(vec![2, 32007]),
        think_end_id: Some(151668),
    });
    let restored = roundtrip(original);
    match restored {
        Wire::GPUWorkerHandshakeReqOutput(out) => {
            assert_eq!(out.max_total_num_tokens, 388544);
            assert_eq!(out.device, "cuda:0");
            assert_eq!(out.max_queued_requests, Some(1024));
            assert_eq!(out.vocab_size, 32000);
            assert_eq!(out.context_len, 32768);
            assert!(out.is_generation);
            assert_eq!(out.hf_eos_token_ids, Some(vec![2, 32007]));
            assert_eq!(out.think_end_id, Some(151668));
        }
        other => panic!("wrong variant after roundtrip: {:?}", other),
    }
}

#[test]
fn handshake_older_worker_decodes_with_defaults() {
    // Encode a handshake reply *without* the new model_config scalars
    // (mimicking an older Python worker) and confirm the Rust mirror
    // still decodes — using `serde_json::from_value` via a hand-built
    // map, since msgpack maps decoded by rmp-serde need named keys to
    // be present in the input.
    use rmpv::Value;
    let map = vec![
        ("type".into(), Value::from("GPUWorkerHandshakeReqOutput")),
        ("max_total_num_tokens".into(), Value::from(1i64)),
        ("max_prefill_tokens".into(), Value::from(1i64)),
        ("max_running_requests".into(), Value::from(1i64)),
        ("max_req_len".into(), Value::from(1i64)),
        ("max_req_input_len".into(), Value::from(1i64)),
        ("random_seed".into(), Value::from(0i64)),
        ("device".into(), Value::from("cpu")),
        ("req_to_token_pool_size".into(), Value::from(1i64)),
        ("req_to_token_pool_max_context_len".into(), Value::from(1i64)),
        ("token_to_kv_pool_size".into(), Value::from(1i64)),
    ];
    let val = Value::Map(map);
    let bytes = rmp_serde::to_vec_named(&val).expect("encode");
    let w: Wire = rmp_serde::from_slice(&bytes).expect("decode");
    match w {
        Wire::GPUWorkerHandshakeReqOutput(out) => {
            // Older worker → defaults.
            assert_eq!(out.vocab_size, 0);
            assert_eq!(out.context_len, 0);
            assert!(out.is_generation);
            assert!(out.hf_eos_token_ids.is_none());
            assert!(out.think_end_id.is_none());
        }
        other => panic!("wrong variant: {:?}", other),
    }
}

#[test]
fn decode_step_control_roundtrips() {
    let seq_lens = TensorIPC::from_i64(&[10, 12, 14]);
    let seq_lens_cpu = TensorIPC::from_i64(&[10, 12, 14]);
    let orig_seq_lens = TensorIPC::from_i64(&[10, 12, 14]);
    let input_ids = TensorIPC::from_i64(&[5, 7, 9]);

    let original = Wire::DecodeStepControlReq(DecodeStepControlReq {
        seq_lens,
        seq_lens_cpu,
        orig_seq_lens,
        seq_lens_sum: 36,
        input_ids: Some(input_ids),
        indices_to_free: None,
        mamba_track_indices: None,
        mamba_track_mask: None,
        mamba_track_seqlens: None,
        global_num_tokens: None,
        global_num_tokens_for_logprob: None,
        input_slot: None,
        output_slot: Some(7),
    });
    let restored = roundtrip(original);
    match restored {
        Wire::DecodeStepControlReq(c) => {
            assert_eq!(c.seq_lens_sum, 36);
            assert_eq!(c.output_slot, Some(7));
            let ids = c.input_ids.as_ref().expect("input_ids present");
            assert_eq!(ids.as_i64().expect("int64"), &[5, 7, 9][..]);
        }
        other => panic!("wrong variant after roundtrip: {:?}", other),
    }
}

#[test]
fn decode_forward_slim_output_roundtrips() {
    let original = Wire::DecodeForwardSlimOutput(DecodeForwardSlimOutput {
        next_token_ids: TensorIPC::from_i64(&[11, 13]),
        deferred_alloc: Some(DeferredAllocIPC {
            mode: "decode".into(),
            req_pool_indices: TensorIPC::from_i64(&[0, 1]),
            out_cache_loc: TensorIPC::from_i32(&[160, 176]),
            seq_lens_minus1: Some(TensorIPC::from_i64(&[9, 11])),
            prefix_lens: None,
            extend_lens: None,
            free_pages_remaining: 24000,
        }),
        accept_lens: None,
        can_run_cuda_graph: true,
        num_accepted_drafts: 0,
        num_accepted_drafts_per_req_cpu: None,
        logits_output_pickle: None,
        routed_experts_output_pickle: Some(ByteBuf::from(vec![1, 2, 3])),
        expert_distribution_metrics_pickle: None,
        next_draft_input_pickle: None,
    });
    let restored = roundtrip(original);
    match restored {
        Wire::DecodeForwardSlimOutput(o) => {
            assert!(o.can_run_cuda_graph);
            let da = o.deferred_alloc.as_ref().expect("deferred_alloc");
            assert_eq!(da.mode, "decode");
            assert_eq!(da.free_pages_remaining, 24000);
            assert_eq!(
                da.out_cache_loc.as_i32().expect("int32"),
                &[160, 176][..]
            );
        }
        other => panic!("wrong variant after roundtrip: {:?}", other),
    }
}

/// Make sure the discriminator on the wire is the variant name — the
/// scheduler's compatibility with the Python `msgspec.Struct(tag=True)`
/// schema hinges on this.
#[test]
fn wire_carries_type_tag() {
    let w = Wire::GPUWorkerHandshakeReqOutput(GPUWorkerHandshakeReqOutput {
        max_total_num_tokens: 1,
        max_prefill_tokens: 1,
        max_running_requests: 1,
        max_req_len: 1,
        max_req_input_len: 1,
        random_seed: 1,
        device: "cpu".into(),
        req_to_token_pool_size: 1,
        req_to_token_pool_max_context_len: 1,
        token_to_kv_pool_size: 1,
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
        vocab_size: 0,
        context_len: 0,
        is_generation: true,
        hf_eos_token_ids: None,
        think_end_id: None,
    });
    let bytes = rmp_serde::to_vec_named(&w).expect("encode");
    // Walk the msgpack bytes via rmpv to confirm the "type" key exists
    // with the variant name as value.
    let val: rmpv::Value = rmp_serde::from_slice(&bytes).expect("rmpv decode");
    let map = val.as_map().expect("top-level map");
    let (key, value) = map.first().expect("at least one entry");
    assert_eq!(key.as_str(), Some("type"));
    assert_eq!(value.as_str(), Some("GPUWorkerHandshakeReqOutput"));
}
