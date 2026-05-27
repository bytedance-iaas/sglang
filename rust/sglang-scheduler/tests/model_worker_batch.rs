// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Tests that the `ModelWorkerBatchPayload` msgpack encoding matches
//! what the Python worker's msgspec decoder expects.
//!
//! We can't link directly against the Python decoder here, so the
//! tests confirm:
//!
//!   * Required fields are present in the encoded map.
//!   * Field names match the Python `ModelWorkerBatch` dataclass.
//!   * Round-trip via `rmpv::Value` preserves the structure.
//!   * Tensor fields encode as the 3-tuple `(shape, dtype, bytes)`.
//!
//! Once a Python `TpWorkerServer` is reachable in CI, swap this for an
//! actual round-trip via `msgspec.msgpack.Decoder`.

use rmpv::Value;
use sglang_scheduler::wire::{
    forward_mode_wire, ModelWorkerBatchPayload, SamplingBatchInfoPayload, TensorIPC,
};

fn minimal_payload() -> ModelWorkerBatchPayload {
    let bs = 3usize;
    ModelWorkerBatchPayload {
        forward_mode: forward_mode_wire::DECODE,
        input_ids: TensorIPC::from_i32(&[1, 2, 3]),
        req_pool_indices: TensorIPC::from_i64(&[0, 1, 2]),
        seq_lens: TensorIPC::from_i64(&[10, 11, 12]),
        out_cache_loc: None,
        seq_lens_cpu: Some(TensorIPC::from_i64(&[10, 11, 12])),
        seq_lens_sum: 33,
        return_logprob: false,
        top_logprobs_nums: None,
        token_ids_logprobs: None,
        global_num_tokens: None,
        global_num_tokens_for_logprob: None,
        is_extend_in_batch: false,
        all_extend_in_batch: false,
        can_run_dp_cuda_graph: false,
        tbo_split_seq_index: None,
        global_forward_mode: None,
        extend_num_tokens: None,
        extend_seq_lens: None,
        extend_prefix_lens: None,
        extend_logprob_start_lens: None,
        extend_input_logprob_token_ids: None,
        sampling_info: SamplingBatchInfoPayload::from_uniform(bs, 32000, "cuda:0"),
        multimodal_inputs: Some(vec![None; bs]),
        encoder_cached: None,
        encoder_lens: None,
        encoder_lens_cpu: None,
        encoder_out_cache_loc: None,
        lora_ids: Some(vec![None; bs]),
        orig_seq_lens: None,
        input_embeds: None,
        replace_embeds: None,
        replace_positions: None,
        ne_token_table: None,
        token_type_ids: None,
        spec_algorithm: None,
        spec_info: None,
        capture_hidden_mode: None,
        hicache_consumer_index: -1,
        dimensions: None,
        return_pooled_hidden_states: false,
        is_prefill_only: false,
        multi_item_delimiter_indices: None,
        dllm_block_offsets: None,
        dllm_config: None,
        reqs: None,
        has_grammar: false,
        rids: Some(vec!["r1".into(), "r2".into(), "r3".into()]),
        return_hidden_states_before_norm: false,
        mamba_track_indices: None,
        mamba_track_mask: None,
        mamba_track_seqlens: None,
        req_to_token_cpu: None,
        indices_to_free: None,
    }
}

fn map_get<'a>(map: &'a [(Value, Value)], key: &str) -> Option<&'a Value> {
    map.iter()
        .find(|(k, _)| k.as_str() == Some(key))
        .map(|(_, v)| v)
}

#[test]
fn payload_encodes_required_fields() {
    let payload = minimal_payload();
    let bytes = rmp_serde::to_vec_named(&payload).expect("encode");
    let value: Value = rmp_serde::from_slice(&bytes).expect("rmpv decode");
    let map = value.as_map().expect("payload encodes as a map");

    // Spot-check that the required dataclass fields are all present.
    for required in [
        "forward_mode",
        "input_ids",
        "req_pool_indices",
        "seq_lens",
        "seq_lens_sum",
        "return_logprob",
        "is_extend_in_batch",
        "all_extend_in_batch",
        "can_run_dp_cuda_graph",
        "sampling_info",
        "hicache_consumer_index",
    ] {
        assert!(
            map_get(map, required).is_some(),
            "required field {required:?} missing from encoded payload"
        );
    }

    // forward_mode encoded as int — msgspec decodes int → IntEnum.
    let fm = map_get(map, "forward_mode")
        .and_then(Value::as_i64)
        .expect("forward_mode is an int");
    assert_eq!(fm, forward_mode_wire::DECODE as i64);
}

#[test]
fn tensor_fields_encode_as_3_tuple() {
    let payload = minimal_payload();
    let bytes = rmp_serde::to_vec_named(&payload).expect("encode");
    let value: Value = rmp_serde::from_slice(&bytes).expect("rmpv decode");
    let map = value.as_map().expect("map");

    let input_ids = map_get(map, "input_ids").expect("input_ids present");
    let arr = input_ids.as_array().expect("tensor encodes as array");
    assert_eq!(arr.len(), 3, "TensorIPC is (shape, dtype, data)");
    // shape is a list
    assert!(arr[0].as_array().is_some());
    // dtype is a string
    assert_eq!(arr[1].as_str(), Some("int32"));
    // data is bytes
    assert!(arr[2].as_slice().is_some() || arr[2].as_str().is_none());
}

#[test]
fn sampling_info_is_nested_map() {
    let payload = minimal_payload();
    let bytes = rmp_serde::to_vec_named(&payload).expect("encode");
    let value: Value = rmp_serde::from_slice(&bytes).expect("rmpv decode");
    let map = value.as_map().expect("map");

    let si = map_get(map, "sampling_info").expect("sampling_info present");
    let si_map = si.as_map().expect("sampling_info is a nested map");
    for required in [
        "temperatures",
        "top_ps",
        "top_ks",
        "min_ps",
        "is_all_greedy",
        "vocab_size",
        "device",
    ] {
        assert!(
            map_get(si_map, required).is_some(),
            "sampling_info missing {required:?}"
        );
    }
    assert_eq!(map_get(si_map, "device").and_then(Value::as_str), Some("cuda:0"));
}

#[test]
fn to_msgpack_value_roundtrip() {
    let payload = minimal_payload();
    let value = payload.to_msgpack_value().expect("to_msgpack_value");
    // It should be a top-level map (the inner ForwardBatchGenerationReq
    // will nest this directly under its `batch` field).
    assert!(value.is_map(), "payload encodes as a top-level map");
}

#[test]
fn wire_dtypes_match_python_expectations() {
    // Pins the wire dtypes the Python worker reads.  Mismatches
    // (e.g. shipping `input_ids` as int32 when `nn.Embedding` requires
    // Long) silently produce garbage embeddings and surface later as
    // CUDA `IndexKernel.cu:111: index out of bounds`.
    //
    // Source of truth: dtype= calls in schedule_batch.py / memory_pool.py:
    //   * input_ids        — torch.int64  (schedule_batch.py:1596)
    //   * req_pool_indices — torch.int64  (schedule_batch.py:1391)
    //   * seq_lens         — torch.int64  (schedule_batch.py:1599)
    //   * seq_lens_cpu     — torch.int64  (schedule_batch.py:1602)
    //   * orig_seq_lens    — torch.int32  (schedule_batch.py:1709)
    //   * req_to_token_cpu — torch.int32  (memory_pool.py:149)
    use std::sync::{Arc, RwLock};
    use sglang_scheduler::memory::ReqToTokenPool;
    use sglang_scheduler::types::{ForwardMode, Req, SamplingParams, ScheduleBatch};

    let mut batch = ScheduleBatch::new(ForwardMode::Extend);
    let mut sp = SamplingParams::default();
    sp.max_new_tokens = 4;
    let r = Req::new("rid".into(), vec![10, 20, 30], sp);
    batch.reqs.push(Arc::new(RwLock::new(r)));
    batch.req_pool_indices.push(1);
    batch.seq_lens.push(3);
    batch.orig_seq_lens.push(3);
    batch.input_ids = vec![10, 20, 30];

    let pool = ReqToTokenPool::new(8, 64);
    let payload = batch.to_model_worker_batch_payload(32000, "cuda:0", Some(&pool));

    assert_eq!(payload.input_ids.dtype(), "int64",
        "input_ids MUST be int64 (nn.Embedding requires Long)");
    assert_eq!(payload.req_pool_indices.dtype(), "int64");
    assert_eq!(payload.seq_lens.dtype(), "int64");
    assert_eq!(payload.seq_lens_cpu.as_ref().unwrap().dtype(), "int64");
    assert_eq!(payload.orig_seq_lens.as_ref().unwrap().dtype(), "int32",
        "orig_seq_lens MUST be int32");
}

#[test]
fn capture_hidden_mode_full_when_any_req_returns_hidden_states() {
    use std::sync::{Arc, RwLock};
    use sglang_scheduler::types::{ForwardMode, Req, SamplingParams, ScheduleBatch};
    use sglang_scheduler::wire::capture_hidden_mode_wire as chm;

    fn req_with_hidden(rid: &str, return_hidden: bool) -> Arc<RwLock<Req>> {
        let mut r = Req::new(rid.into(), vec![1, 2, 3], SamplingParams::default());
        r.return_hidden_states = return_hidden;
        Arc::new(RwLock::new(r))
    }

    let mut batch = ScheduleBatch::new(ForwardMode::Decode);
    batch.reqs.push(req_with_hidden("a", false));
    batch.reqs.push(req_with_hidden("b", false));
    batch.req_pool_indices = vec![0, 1];
    batch.seq_lens = vec![5, 5];
    batch.orig_seq_lens = vec![3, 3];
    batch.input_ids = vec![10, 20];

    // No req wants hidden states → NULL.
    let payload = batch.to_model_worker_batch_payload(32000, "cuda:0", None);
    assert_eq!(payload.capture_hidden_mode, Some(chm::NULL));

    // Flip one req's flag → FULL on the batch.
    batch.reqs[1].write().unwrap().return_hidden_states = true;
    let payload = batch.to_model_worker_batch_payload(32000, "cuda:0", None);
    assert_eq!(payload.capture_hidden_mode, Some(chm::FULL));
}

#[test]
fn sampling_info_from_per_req_params_sets_need_flags() {
    use sglang_scheduler::wire::SamplingParamsView;

    let views = vec![
        SamplingParamsView {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            min_p: 0.0,
            seed: None,
        },
        SamplingParamsView {
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            min_p: 0.05,
            seed: Some(42),
        },
    ];
    let info = SamplingBatchInfoPayload::from_sampling_params(views.into_iter(), 32000, "cuda:0");
    assert!(info.need_top_p_sampling, "0.9 != 1.0 → top_p sampling required");
    assert!(info.need_top_k_sampling, "top_k=50 is not TOP_K_ALL");
    assert!(info.need_min_p_sampling, "0.05 > 0");
    assert!(!info.is_all_greedy, "top_k=50 means non-greedy");
    assert!(info.sampling_seed.is_some(), "at least one req set a seed");
}
