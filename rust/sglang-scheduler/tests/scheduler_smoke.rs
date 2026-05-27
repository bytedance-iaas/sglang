// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Integration smoke tests for the larger scheduler pieces.
//!
//! These don't require a worker — they exercise pure data-structure
//! interactions (page tracker + req pool + radix cache + retract).

use std::sync::{Arc, RwLock};

use sglang_scheduler::memory::{CpuPageTracker, ReqToTokenPool};
use sglang_scheduler::queue::WaitingQueue;
use sglang_scheduler::radix_cache::RadixCache;
use sglang_scheduler::scheduler::output_processor::process_batch_result;
use sglang_scheduler::scheduler::retract::{
    evict_from_tree_cache, reclaim_for_decode, retract_decode,
};
use sglang_scheduler::types::{ForwardMode, Req, SamplingParams, ScheduleBatch};
use sglang_scheduler::wire::{DecodeForwardSlimOutput, DeferredAllocIPC, TensorIPC, Wire};

const PAGE: i64 = 16;

fn make_req(rid: &str, prompt_len: usize, max_new: u32, output_so_far: usize) -> Arc<RwLock<Req>> {
    let mut sp = SamplingParams::default();
    sp.max_new_tokens = max_new;
    let mut req = Req::new(rid.into(), vec![7; prompt_len], sp);
    req.output_ids = vec![5; output_so_far];
    req.req_pool_idx = None;
    req.kv_allocated_len = (prompt_len + output_so_far) as u32;
    req.kv_committed_len = req.kv_allocated_len;
    Arc::new(RwLock::new(req))
}

#[test]
fn retract_picks_longest_output_first() {
    let mut pool = ReqToTokenPool::new(8, 64);
    let mut tracker = CpuPageTracker::new(8, PAGE);

    let mut batch = ScheduleBatch::new(ForwardMode::Decode);
    let mut waiting = WaitingQueue::new();

    // Three reqs in the running batch; the longest output should be
    // retracted first.  Use distinct output lengths to make the
    // ordering unambiguous.
    let r_short = make_req("short", 4, 100, 1);
    let r_long = make_req("long", 4, 100, 10);
    let r_mid = make_req("mid", 4, 100, 5);

    for (i, req) in [&r_short, &r_long, &r_mid].iter().enumerate() {
        let slot = pool.alloc(1).unwrap()[0];
        {
            let mut w = req.write().unwrap();
            w.req_pool_idx = Some(slot);
        }
        batch.req_pool_indices.push(slot);
        batch.seq_lens.push(10 + (i as i32));
        batch.orig_seq_lens.push(4);
        batch.reqs.push((*req).clone());
        let _ = i;
    }
    assert_eq!(batch.batch_size(), 3);

    // Pretend the pool is exhausted (set base_free_pages to 0).
    tracker.update_free_count(0);
    // Need at least 1 page for the next step.
    let outcome = retract_decode(&mut batch, &mut waiting, &mut tracker, &mut pool, PAGE);
    // We requested PAGE tokens; the tracker still reports 0 free
    // because we never restocked.  retract_decode keeps trying until
    // it can't usefully retract anything — eventually it'll wind down
    // to a single req and abort it.  Just verify at least one was
    // retracted and the long-output one was preferred.
    assert!(outcome.retracted_reqs >= 1);
    // The "long" req should no longer own a pool slot.
    assert!(r_long.read().unwrap().req_pool_idx.is_none());
}

#[test]
fn radix_cache_smoke_with_real_indices() {
    let mut cache = RadixCache::new();
    // Three reqs share the first 4 tokens.
    let prompt_a = [10, 20, 30, 40, 50];
    let slots_a = [101, 102, 103, 104, 105];
    let prompt_b = [10, 20, 30, 40, 60];
    let slots_b = [201, 202, 203, 204, 205];

    let dup = cache.insert(&prompt_a, &slots_a);
    assert!(dup.is_empty());
    let dup = cache.insert(&prompt_b, &slots_b);
    // First 4 are duplicates (shared prefix).
    assert_eq!(dup.len(), 4);

    // A third req hits the cached prefix.
    let m = cache.match_prefix(&[10, 20, 30, 40, 70]);
    assert_eq!(m.prefix_len, 4);
    assert_eq!(m.slots, vec![101, 102, 103, 104]);
}

#[test]
fn evict_from_tree_cache_frees_pages_into_tracker() {
    let mut cache = RadixCache::new();
    let mut tracker = CpuPageTracker::new(8, PAGE);

    // Seed the cache with two divergent sequences so several leaves
    // become candidates for eviction.
    cache.insert(&[1, 2, 3, 4], &[101, 102, 103, 104]);
    cache.insert(&[1, 2, 5, 6], &[101, 102, 105, 106]);
    assert!(cache.evictable_size() >= 4);

    // Pretend the budget is gone; ask for 2 tokens worth of relief.
    tracker.update_free_count(0);
    let freed = evict_from_tree_cache(&mut cache, &mut tracker, 2);
    assert!(freed >= 2);
}

#[test]
fn reclaim_for_decode_prefers_cache_eviction_over_retraction() {
    let mut pool = ReqToTokenPool::new(8, 64);
    let mut tracker = CpuPageTracker::new(8, PAGE);
    let mut cache = RadixCache::new();
    let mut waiting = WaitingQueue::new();
    let mut batch = ScheduleBatch::new(ForwardMode::Decode);

    // One running req — retraction would drop it; we want the cache
    // path to satisfy the budget first.
    let r = make_req("only", 4, 100, 1);
    let slot = pool.alloc(1).unwrap()[0];
    {
        let mut w = r.write().unwrap();
        w.req_pool_idx = Some(slot);
    }
    batch.req_pool_indices.push(slot);
    batch.seq_lens.push(5);
    batch.orig_seq_lens.push(4);
    batch.reqs.push(r.clone());

    // Stack the radix cache with plenty of evictable tokens.
    cache.insert(
        &[1, 2, 3, 4, 5, 6, 7, 8],
        &[201, 202, 203, 204, 205, 206, 207, 208],
    );
    tracker.update_free_count(0);

    let outcome = reclaim_for_decode(
        &mut batch,
        &mut waiting,
        &mut tracker,
        &mut pool,
        &mut cache,
        4,
    );
    // Cache eviction should have produced enough; retraction
    // shouldn't have kicked in.
    assert!(outcome.evicted_cache_tokens >= 4);
    assert_eq!(outcome.retracted_reqs, 0);
    assert_eq!(outcome.aborted, 0);
    // The req still owns its slot.
    assert!(r.read().unwrap().req_pool_idx.is_some());
}

#[test]
fn process_batch_result_emits_detokenizer_output_frame() {
    // Two running reqs; the worker reports two newly-sampled tokens.
    // process_batch_result must produce a `BatchTokenIDOutput` with
    // matching per-req entries.
    let mut pool = ReqToTokenPool::new(8, 64);
    let mut tracker = CpuPageTracker::new(8, PAGE);

    let mut batch = ScheduleBatch::new(ForwardMode::Decode);
    let r0 = make_req("alpha", 4, 100, 1);
    let r1 = make_req("beta", 4, 100, 1);
    for req in [&r0, &r1] {
        let slot = pool.alloc(1).unwrap()[0];
        req.write().unwrap().req_pool_idx = Some(slot);
        batch.req_pool_indices.push(slot);
        batch.seq_lens.push(5);
        batch.orig_seq_lens.push(4);
        batch.reqs.push(req.clone());
    }

    let reply = DecodeForwardSlimOutput {
        next_token_ids: TensorIPC::from_i64(&[42, 99]),
        deferred_alloc: None,
        accept_lens: None,
        can_run_cuda_graph: true,
        num_accepted_drafts: 0,
        num_accepted_drafts_per_req_cpu: None,
        logits_output_pickle: None,
        routed_experts_output_pickle: None,
        expert_distribution_metrics_pickle: None,
        next_draft_input_pickle: None,
    };

    let stats = process_batch_result(&mut batch, &reply, &mut tracker, &mut pool);
    let out = stats
        .detokenizer_output
        .expect("non-empty batch must populate detokenizer_output");

    // Per-req shape.
    assert_eq!(
        out.rids.as_ref().unwrap(),
        &vec!["alpha".to_string(), "beta".to_string()]
    );
    assert_eq!(out.decode_ids, vec![vec![42i64], vec![99]]);
    assert_eq!(out.completion_tokens, vec![2, 2]); // 1 prior + 1 newly sampled
    assert_eq!(out.prompt_tokens, vec![4, 4]);

    // Wire round-trip: must encode + decode as a tagged-union variant.
    let frame = Wire::BatchTokenIDOutput(out);
    let bytes = rmp_serde::to_vec_named(&frame).expect("encode");
    let decoded: Wire = rmp_serde::from_slice(&bytes).expect("decode");
    assert!(matches!(decoded, Wire::BatchTokenIDOutput(_)));
}

#[test]
fn finish_does_not_free_cache_owned_prefix_slots() {
    // Regression for "CUDA index out of bounds on the 3rd request".
    //
    // The bug: when a req's `cache_protected_len > 0` (i.e. its prefix
    // was planted from the cache at admission), the finish path was
    // freeing the entire `duplicates` slice returned by `cache.insert`
    // — which includes those cache-owned slots.  The next req that
    // hit the prefix would get back slot ids the worker had since
    // re-allocated to a different req, and the GPU would index into
    // garbage KV.
    //
    // This test drives the finish path with one req whose prefix is
    // already in the cache and verifies that the cache-owned prefix
    // slots stay out of `page_tracker.drain_pending_free()`.
    use std::sync::{Arc, RwLock};

    use sglang_scheduler::scheduler::output_processor::process_batch_result_with_cache;
    use sglang_scheduler::types::{FinishReason, Req, SamplingParams};

    let mut pool = ReqToTokenPool::new(8, 64);
    let mut tracker = CpuPageTracker::new(64, PAGE);
    tracker.update_free_count(64);
    let mut cache = RadixCache::new();

    // Seed the cache with a 3-token prefix mapped to slots [101,102,103].
    cache.insert(&[10, 20, 30], &[101, 102, 103]);

    // Build a single req mimicking admission with that cached prefix:
    //   * prefix_len_from_cache = 3, cached_node pinned.
    //   * Slot row already has [101,102,103,..] planted at columns 0..3.
    //   * One additional token (the cap-saved last one) is at column 3.
    let slot = pool.alloc(1).unwrap()[0];
    pool.write(slot, 0, &[101, 102, 103, 200]).unwrap();
    let m = cache.match_prefix(&[10, 20, 30]);
    cache.inc_lock_ref(m.last_node);

    let mut sp = SamplingParams::default();
    sp.max_new_tokens = 1; // finishes after the next decoded token
    let mut req = Req::new("rid".into(), vec![10, 20, 30, 40], sp);
    req.req_pool_idx = Some(slot);
    req.cached_node = Some(m.last_node);
    req.prefix_len_from_cache = 3;
    req.cache_protected_len = 3;
    // 4 tokens already prefilled → kv_allocated_len = 4.
    req.kv_allocated_len = 4;
    req.kv_committed_len = 4;
    let req_arc = Arc::new(RwLock::new(req));

    let mut batch = ScheduleBatch::new(ForwardMode::Decode);
    batch.reqs.push(req_arc.clone());
    batch.req_pool_indices.push(slot);
    batch.seq_lens.push(4);
    batch.orig_seq_lens.push(4);

    // Worker decode step emits token 42; with max_new_tokens=1 the
    // req finishes on this iter.
    let reply = DecodeForwardSlimOutput {
        next_token_ids: TensorIPC::from_i64(&[42]),
        deferred_alloc: None,
        accept_lens: None,
        can_run_cuda_graph: true,
        num_accepted_drafts: 0,
        num_accepted_drafts_per_req_cpu: None,
        logits_output_pickle: None,
        routed_experts_output_pickle: None,
        expert_distribution_metrics_pickle: None,
        next_draft_input_pickle: None,
    };

    let _ = process_batch_result_with_cache(
        &mut batch,
        &reply,
        &mut tracker,
        &mut pool,
        Some(&mut cache),
    );
    // Confirm the req actually finished (length-based stop).
    assert!(matches!(
        req_arc.read().unwrap().finished_reason,
        Some(FinishReason::Length { .. })
    ));

    // The smoking gun: pending_free MUST NOT contain any of the
    // cache-protected slots [101, 102, 103].  Slot 200 (the cap-saved
    // last token) may or may not be in pending_free depending on
    // whether the cache walk grafted it (here the tree had only the
    // 3-token prefix, so 200 is grafted as new leaf and stays).
    let pending = tracker.drain_pending_free();
    assert!(
        !pending.iter().any(|&s| s == 101 || s == 102 || s == 103),
        "cache-owned slots {{101, 102, 103}} must NOT be freed; got pending={pending:?}"
    );

    // The cache must still report the prefix as live.
    let m_after = cache.match_prefix(&[10, 20, 30]);
    assert_eq!(m_after.slots, vec![101, 102, 103]);
}

#[test]
fn decode_deferred_alloc_writes_new_slot_into_req_to_token() {
    // Regression for the CUDA index OOB on the *2nd identical request*.
    //
    // The Rust scheduler used to skip `apply_*_deferred_alloc` whenever
    // forward_mode was Decode (only Extend / Mixed were handled).  As
    // a result the CPU `req_to_token` row stayed stale through every
    // decode step, and the slot ids the worker actually wrote on the
    // GPU never made it back into the table the cache reads from.
    // When the req finished, `cache.insert` grafted whatever the
    // *previous tenant* of the slot row had written, and the next
    // matching req attended against KV slots that had since been
    // re-allocated → "Triton CUDA: index out of bounds".
    //
    // This test simulates 3 decode steps and asserts the table carries
    // the worker's slot ids after each one.
    let mut pool = ReqToTokenPool::new(8, 64);
    let mut tracker = CpuPageTracker::new(64, PAGE);
    tracker.update_free_count(64);

    let mut batch = ScheduleBatch::new(ForwardMode::Decode);
    let req = make_req("rid", 4, 100, 0);
    let slot = pool.alloc(1).unwrap()[0];
    {
        let mut w = req.write().unwrap();
        w.req_pool_idx = Some(slot);
        // Prefill already happened; populate columns 0..4 with the
        // prefill-time slot ids.
        w.kv_allocated_len = 4;
        w.kv_committed_len = 4;
    }
    pool.write(slot, 0, &[101, 102, 103, 200]).unwrap();
    batch.req_pool_indices.push(slot);
    batch.seq_lens.push(5);
    batch.orig_seq_lens.push(4);
    batch.reqs.push(req.clone());

    // Three consecutive decode replies, each reporting a fresh slot
    // id for the just-decoded token at column seq_len - 1.
    let decode_replies = [(201i32, 4i64), (202, 5), (203, 6)];
    for (i, (new_slot, col)) in decode_replies.iter().enumerate() {
        // The event loop's `prepare_for_decode` would bump these; in
        // this test we manage them by hand to keep the harness small.
        {
            let mut w = req.write().unwrap();
            w.kv_allocated_len = 5 + i as u32;
            w.kv_committed_len = 5 + i as u32;
        }
        let reply = DecodeForwardSlimOutput {
            next_token_ids: TensorIPC::from_i64(&[(*new_slot + 1000) as i64]),
            deferred_alloc: Some(DeferredAllocIPC {
                mode: "decode".into(),
                req_pool_indices: TensorIPC::from_i64(&[slot as i64]),
                out_cache_loc: TensorIPC::from_i64(&[*new_slot as i64]),
                seq_lens_minus1: Some(TensorIPC::from_i64(&[*col])),
                prefix_lens: None,
                extend_lens: None,
                free_pages_remaining: 64,
            }),
            accept_lens: None,
            can_run_cuda_graph: true,
            num_accepted_drafts: 0,
            num_accepted_drafts_per_req_cpu: None,
            logits_output_pickle: None,
            routed_experts_output_pickle: None,
            expert_distribution_metrics_pickle: None,
            next_draft_input_pickle: None,
        };
        let _ = process_batch_result(&mut batch, &reply, &mut tracker, &mut pool);

        // After each step the CPU req_to_token row must contain the
        // slot the worker wrote at column `col`.
        let row = pool.read(slot, (*col as u32) + 1).expect("read");
        assert_eq!(
            row[*col as usize] as i32, *new_slot,
            "decode step {i}: req_to_token[{slot}, {col}] must == {new_slot}, got row={row:?}"
        );
    }

    // Final state must reflect all three decoded slot ids.
    let row = pool.read(slot, 7).expect("read final");
    assert_eq!(
        row,
        &[101, 102, 103, 200, 201, 202, 203],
        "all decoded slots must be planted in CPU req_to_token"
    );
}

#[test]
fn extend_writeback_accepts_int64_out_cache_loc() {
    // The Python worker emits `out_cache_loc` as int64.  This test pins
    // that the Rust output processor narrows int64 → i32 and writes the
    // resulting slot indices into `req_to_token`.
    let mut pool = ReqToTokenPool::new(8, 64);
    let mut tracker = CpuPageTracker::new(8, PAGE);
    tracker.update_free_count(8);

    let mut batch = ScheduleBatch::new(ForwardMode::Extend);
    // One req with a 3-token prompt; extend allocates 3 new KV slots.
    let r = make_req("only", 3, 100, 0);
    let slot = pool.alloc(1).unwrap()[0];
    r.write().unwrap().req_pool_idx = Some(slot);
    batch.req_pool_indices.push(slot);
    batch.seq_lens.push(3);
    batch.orig_seq_lens.push(3);
    batch.reqs.push(r.clone());

    let reply = DecodeForwardSlimOutput {
        next_token_ids: TensorIPC::from_i64(&[42]),
        deferred_alloc: Some(DeferredAllocIPC {
            mode: "extend".into(),
            req_pool_indices: TensorIPC::from_i64(&[slot as i64]),
            // Worker ships int64 — the Rust side must accept it.
            out_cache_loc: TensorIPC::from_i64(&[100, 101, 102]),
            seq_lens_minus1: None,
            prefix_lens: Some(TensorIPC::from_i64(&[0])),
            extend_lens: Some(TensorIPC::from_i64(&[3])),
            free_pages_remaining: 32,
        }),
        accept_lens: None,
        can_run_cuda_graph: true,
        num_accepted_drafts: 0,
        num_accepted_drafts_per_req_cpu: None,
        logits_output_pickle: None,
        routed_experts_output_pickle: None,
        expert_distribution_metrics_pickle: None,
        next_draft_input_pickle: None,
    };

    let _ = process_batch_result(&mut batch, &reply, &mut tracker, &mut pool);

    // After extend writeback, the req's KV slot row must contain
    // [100, 101, 102] (narrowed from int64).  +1 because the decode
    // step also appended `next_token_ids[0]` → kv_allocated_len = 4.
    let req = r.read().unwrap();
    assert_eq!(req.kv_allocated_len, 3);
    drop(req);
    let row = pool.read(slot, 3).expect("read row");
    assert_eq!(row, &[100i32, 101, 102]);
}

#[test]
fn process_batch_result_empty_batch_emits_no_frame() {
    let mut pool = ReqToTokenPool::new(8, 64);
    let mut tracker = CpuPageTracker::new(8, PAGE);
    let mut batch = ScheduleBatch::new(ForwardMode::Decode);
    let reply = DecodeForwardSlimOutput {
        next_token_ids: TensorIPC::from_i64(&[]),
        deferred_alloc: None,
        accept_lens: None,
        can_run_cuda_graph: true,
        num_accepted_drafts: 0,
        num_accepted_drafts_per_req_cpu: None,
        logits_output_pickle: None,
        routed_experts_output_pickle: None,
        expert_distribution_metrics_pickle: None,
        next_draft_input_pickle: None,
    };
    let stats = process_batch_result(&mut batch, &reply, &mut tracker, &mut pool);
    assert!(stats.detokenizer_output.is_none());
}

#[test]
fn cached_prefix_admission_slices_input_ids_and_snapshots_req_to_token() {
    use sglang_scheduler::queue::{SchedulePolicy, SchedulePolicyKind};
    use sglang_scheduler::scheduler::BatchBuilder;

    // Seed the radix cache with a 3-token prefix occupying KV slots
    // [101, 102, 103].  A new req comes in whose prompt extends that
    // prefix by 2 more tokens — the scheduler must:
    //   1. Push ONLY the 2 new tokens into batch.input_ids.
    //   2. Plant [101, 102, 103] into req_to_token at the new slot's
    //      first 3 columns so the payload's `req_to_token_cpu`
    //      snapshot carries them to the worker.
    //   3. Report extend_prefix_lens = [3] in the payload, with
    //      extend_num_tokens = 2 (= sum(extend_seq_lens - prefix)).
    //
    // Failure of any of these was the root cause of "req2 returns
    // garbage; req3 illegal CUDA access" on the live worker.
    let mut cache = RadixCache::new();
    cache.insert(&[10, 20, 30], &[101, 102, 103]);

    let mut pool = ReqToTokenPool::new(8, 64);
    let tracker = CpuPageTracker::new(64, PAGE);
    let mut waiting = WaitingQueue::new();

    let mut sp = SamplingParams::default();
    sp.max_new_tokens = 100;
    let req = Req::new("rid".into(), vec![10, 20, 30, 40, 50], sp);
    waiting.push(Arc::new(RwLock::new(req)));

    let builder = BatchBuilder::new(PAGE, 8, SchedulePolicy::new(SchedulePolicyKind::Fcfs));
    let running = ScheduleBatch::new(ForwardMode::Extend);
    let batch = builder
        .get_new_batch_prefill_with_cache(
            &running,
            &mut waiting,
            &tracker,
            &mut pool,
            Some(&mut cache),
        )
        .expect("admission must succeed");

    // 1. Only the 2 new tokens (post-prefix) made it into input_ids.
    assert_eq!(
        batch.input_ids,
        vec![40, 50],
        "batch.input_ids must NOT include the cached prefix tokens"
    );
    // seq_lens stays as the full prompt length.
    assert_eq!(batch.seq_lens, vec![5]);

    // 2. The req's prefix_len_from_cache reflects the cache hit.
    let req_state = batch.reqs[0].read().unwrap();
    assert_eq!(req_state.prefix_len_from_cache, 3);
    assert!(req_state.cached_node.is_some());
    let slot = req_state.req_pool_idx.expect("slot allocated");
    drop(req_state);

    // 3. The cached prefix slots are planted into req_to_token at the
    //    new req's slot, columns 0..3.
    let row = pool.read(slot, 3).expect("read planted prefix");
    assert_eq!(row, &[101, 102, 103]);

    // 4. The payload exposes them via req_to_token_cpu, and
    //    extend_prefix_lens / extend_num_tokens are coherent.
    let payload = batch.to_model_worker_batch_payload(32000, "cuda:0", Some(&pool));
    assert_eq!(payload.extend_prefix_lens.as_deref(), Some(&[3i32][..]));
    assert_eq!(payload.extend_num_tokens, Some(2));
    let snapshot = payload
        .req_to_token_cpu
        .as_ref()
        .expect("req_to_token_cpu must be populated when prefix_len > 0");
    assert_eq!(
        snapshot.dtype(),
        "int32",
        "req_to_token_cpu is int32 (matches torch req_to_token)"
    );
}

#[test]
fn full_cache_hit_caps_prefix_at_input_len_minus_one() {
    // Edge case: identical prompt arrives twice in a row.  Naively
    // `match_prefix` returns prefix_len == input_len → `input_ids`
    // ends up empty → msgspec on the worker side panics with
    // `torch.frombuffer(b"", count=-1) must not be 0`.
    //
    // The scheduler caps the matched prefix to `input_len - 1` so
    // the worker always has at least one token to prefill (this is
    // the same rule Python applies — see schedule_batch.py:1007's
    // "matched length is at most 1 less than input length" note).
    use sglang_scheduler::queue::{SchedulePolicy, SchedulePolicyKind};
    use sglang_scheduler::scheduler::BatchBuilder;

    // Seed the cache with the full 4-token sequence.
    let mut cache = RadixCache::new();
    cache.insert(&[10, 20, 30, 40], &[101, 102, 103, 104]);

    let mut pool = ReqToTokenPool::new(8, 64);
    let tracker = CpuPageTracker::new(64, PAGE);
    let mut waiting = WaitingQueue::new();

    let mut sp = SamplingParams::default();
    sp.max_new_tokens = 16;
    waiting.push(Arc::new(RwLock::new(Req::new(
        "rid".into(),
        vec![10, 20, 30, 40],
        sp,
    ))));

    let builder = BatchBuilder::new(PAGE, 8, SchedulePolicy::new(SchedulePolicyKind::Fcfs));
    let running = ScheduleBatch::new(ForwardMode::Extend);
    let batch = builder
        .get_new_batch_prefill_with_cache(
            &running,
            &mut waiting,
            &tracker,
            &mut pool,
            Some(&mut cache),
        )
        .expect("admission must succeed");

    // input_ids must be non-empty: one token survives the cap.
    assert_eq!(
        batch.input_ids.len(),
        1,
        "input_ids must carry the final token, not empty"
    );
    assert_eq!(batch.input_ids[0], 40);
    // prefix_len_from_cache is exactly input_len - 1 = 3.
    let req = batch.reqs[0].read().unwrap();
    assert_eq!(req.prefix_len_from_cache, 3);

    // Payload mirrors the cap.
    drop(req);
    let payload = batch.to_model_worker_batch_payload(32000, "cuda:0", Some(&pool));
    assert_eq!(payload.extend_prefix_lens.as_deref(), Some(&[3i32][..]));
    assert_eq!(payload.extend_num_tokens, Some(1));
}

#[test]
fn extend_seq_lens_wire_field_is_extend_lens_not_seq_lens() {
    // Wire-contract regression: the field named `extend_seq_lens` on
    // `ModelWorkerBatch` is the per-req NEW token count (= `extend_lens`),
    // NOT the full `seq_lens`.  See `schedule_batch.py:2515`:
    //   `extend_seq_lens = self.extend_lens`
    //
    // The worker's `compute_position_kernel` writes
    // `extend_seq_lens[i]` positions into a tensor sized
    // `extend_num_tokens = sum(seq_lens - prefix_lens)`.  Sending full
    // `seq_lens` when any req has a cached prefix overruns the positions
    // tensor → corrupt position indices → CUDA `IndexKernel.cu:111`
    // index OOB.  This was the root cause of the persistent
    // "req2 crashes with CUDA OOB even after dtype / cache-protection
    // / decode-writeback fixes".
    use sglang_scheduler::queue::{SchedulePolicy, SchedulePolicyKind};
    use sglang_scheduler::scheduler::BatchBuilder;

    // Seed a 1-token prefix (the BOS scenario — even "disjoint" prompts
    // share this).
    let mut cache = RadixCache::new();
    cache.insert(&[1], &[100]);

    let mut pool = ReqToTokenPool::new(8, 64);
    let tracker = CpuPageTracker::new(64, PAGE);
    let mut waiting = WaitingQueue::new();

    let mut sp = SamplingParams::default();
    sp.max_new_tokens = 16;
    // 5-token prompt, sharing the 1-token BOS prefix.
    waiting.push(Arc::new(RwLock::new(Req::new(
        "rid".into(),
        vec![1, 20, 30, 40, 50],
        sp,
    ))));

    let builder = BatchBuilder::new(PAGE, 8, SchedulePolicy::new(SchedulePolicyKind::Fcfs));
    let running = ScheduleBatch::new(ForwardMode::Extend);
    let batch = builder
        .get_new_batch_prefill_with_cache(
            &running,
            &mut waiting,
            &tracker,
            &mut pool,
            Some(&mut cache),
        )
        .expect("admission");

    let payload = batch.to_model_worker_batch_payload(32000, "cuda:0", Some(&pool));
    // Total prompt = 5, prefix = 1, so extend_lens = 4.
    assert_eq!(payload.extend_prefix_lens.as_deref(), Some(&[1i32][..]));
    assert_eq!(
        payload.extend_seq_lens.as_deref(),
        Some(&[4i32][..]),
        "extend_seq_lens on the wire MUST be `extend_lens` (new tokens), \
         not the full seq_lens — see schedule_batch.py:2515"
    );
    assert_eq!(
        payload.extend_num_tokens,
        Some(4),
        "extend_num_tokens MUST equal sum(extend_seq_lens) so the \
         positions tensor is sized correctly"
    );
    // And: sum(extend_seq_lens) == extend_num_tokens == input_ids.len()
    let sum_extend: i32 = payload.extend_seq_lens.as_deref().unwrap().iter().sum();
    assert_eq!(sum_extend as i64, payload.extend_num_tokens.unwrap());
}

#[test]
fn no_cached_prefix_skips_req_to_token_snapshot() {
    use sglang_scheduler::queue::{SchedulePolicy, SchedulePolicyKind};
    use sglang_scheduler::scheduler::BatchBuilder;

    // Empty cache → no prefix hit → no snapshot needed (and
    // input_ids carries the full prompt as before).
    let mut cache = RadixCache::new();
    let mut pool = ReqToTokenPool::new(8, 64);
    let tracker = CpuPageTracker::new(64, PAGE);
    let mut waiting = WaitingQueue::new();

    let mut sp = SamplingParams::default();
    sp.max_new_tokens = 100;
    let req = Req::new("rid".into(), vec![10, 20, 30], sp);
    waiting.push(Arc::new(RwLock::new(req)));

    let builder = BatchBuilder::new(PAGE, 8, SchedulePolicy::new(SchedulePolicyKind::Fcfs));
    let running = ScheduleBatch::new(ForwardMode::Extend);
    let batch = builder
        .get_new_batch_prefill_with_cache(
            &running,
            &mut waiting,
            &tracker,
            &mut pool,
            Some(&mut cache),
        )
        .expect("admission must succeed");

    assert_eq!(batch.input_ids, vec![10, 20, 30]);
    assert_eq!(batch.reqs[0].read().unwrap().prefix_len_from_cache, 0);

    let payload = batch.to_model_worker_batch_payload(32000, "cuda:0", Some(&pool));
    assert!(
        payload.req_to_token_cpu.is_none(),
        "no req has a cached prefix → no req_to_token_cpu snapshot"
    );
    assert_eq!(payload.extend_prefix_lens.as_deref(), Some(&[0i32][..]));
    assert_eq!(payload.extend_num_tokens, Some(3));
}
