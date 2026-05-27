// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Retraction under memory pressure.
//!
//! Source: `ScheduleBatch.retract_decode` in `schedule_batch.py`.
//!
//! When the running batch can't fit another decode step in the KV pool,
//! we evict the cheapest-to-retract running reqs (largest output_ids
//! count first — same heuristic the Python uses for the non-spec path).
//! Retracted reqs get their KV freed and go back to the waiting queue
//! for a re-prefill on the next admission.
//!
//! Limits this port:
//!   * No `decode_offload_manager` (disaggregated decode KV offload).
//!   * No HiSparse coordinator handoff.
//!   * No `evict_from_tree_cache` integration (radix cache port pending).

use std::sync::{Arc, RwLock};

use crate::memory::{CpuPageTracker, ReqToTokenPool};
use crate::queue::WaitingQueue;
use crate::radix_cache::RadixCache;
use crate::types::{FinishReason, Req, ScheduleBatch};

#[derive(Debug, Default)]
pub struct RetractionOutcome {
    pub retracted_reqs: u32,
    /// True when even after retracting the rest, the remaining
    /// request(s) still don't fit — the Python equivalent aborts the
    /// last req in that case.
    pub aborted: u32,
    /// Tokens reclaimed by evicting unprotected radix-cache leaves
    /// before falling back to per-req retraction.  Mirrors the
    /// `evict_from_tree_cache` step in `Scheduler.check_decode_mem`.
    pub evicted_cache_tokens: i64,
}

/// Try the radix-cache eviction step before retracting running reqs.
///
/// Mirrors `Scheduler.evict_from_tree_cache` — Python evicts whole
/// pages worth of unprotected leaves until either the budget is met or
/// the cache runs out of evictable nodes.  Returns the number of tokens
/// freed (== the number of slot indices pushed into the page tracker's
/// pending-free list).
pub fn evict_from_tree_cache(
    cache: &mut RadixCache,
    page_tracker: &mut CpuPageTracker,
    needed_tokens: i64,
) -> i64 {
    let mut total_freed = 0i64;
    let mut remaining = needed_tokens.max(0);
    while remaining > 0 && cache.evictable_size() > 0 {
        // Evict up to `remaining` tokens in this pass; the cache may
        // return more if it had to free a whole leaf to clear the
        // request — that's fine, surplus reduces future pressure.
        let freed = cache.evict(remaining);
        if freed.is_empty() {
            break;
        }
        total_freed += freed.len() as i64;
        page_tracker.free_i32(&freed);
        remaining -= freed.len() as i64;
    }
    total_freed
}

/// Retract decode reqs from `batch` until `available_tokens` >=
/// `needed_tokens`.  Mirrors the loop body of `retract_decode`.
///
/// Returns an `Err`-like outcome with `aborted > 0` when even the
/// smallest possible batch can't fit; the caller decides what to do
/// (Python sends back `FINISH_ABORT` to the user).
pub fn retract_decode(
    batch: &mut ScheduleBatch,
    waiting: &mut WaitingQueue,
    page_tracker: &mut CpuPageTracker,
    req_pool: &mut ReqToTokenPool,
    needed_tokens_per_step: i64,
) -> RetractionOutcome {
    retract_decode_with_cache(
        batch,
        waiting,
        page_tracker,
        req_pool,
        None,
        needed_tokens_per_step,
    )
}

/// Variant of `retract_decode` that releases the radix-cache lock
/// the retracted req held.  Called by `reclaim_for_decode` once the
/// cache is wired in.
pub fn retract_decode_with_cache(
    batch: &mut ScheduleBatch,
    waiting: &mut WaitingQueue,
    page_tracker: &mut CpuPageTracker,
    req_pool: &mut ReqToTokenPool,
    mut radix_cache: Option<&mut RadixCache>,
    needed_tokens_per_step: i64,
) -> RetractionOutcome {
    let mut outcome = RetractionOutcome::default();

    if batch.is_empty() {
        return outcome;
    }

    // Sorted indices into batch.reqs — retract the longest-output first
    // (the most expensive to keep going, least painful to drop).  Matches
    // the non-spec branch of Python's `retract_decode`.
    let mut order: Vec<usize> = (0..batch.reqs.len()).collect();
    order.sort_by_key(|&i| {
        // sort_by_key wants a Reverse for descending order, but the
        // Python uses `(len(output_ids), -len(origin_input_ids))`
        // descending so longer outputs come first; with shorter input
        // breaking ties.  Approximate with the same key flipped.
        let r = batch.reqs[i].read().unwrap();
        std::cmp::Reverse((r.output_ids.len(), r.origin_input_ids.len()))
    });

    while !batch.is_empty()
        && (page_tracker.available_size() < needed_tokens_per_step
            || (batch.batch_size() == 1 && page_tracker.available_size() < needed_tokens_per_step))
    {
        let Some(victim_pos) = order.pop() else { break };
        if victim_pos >= batch.reqs.len() {
            continue;
        }

        // Last surviving req that still doesn't fit — Python aborts it
        // rather than retract-to-empty.
        if batch.batch_size() == 1 {
            let req_arc = batch.reqs[victim_pos].clone();
            {
                let mut req = req_arc.write().unwrap();
                req.finished_reason = Some(FinishReason::Internal {
                    message:
                        "Out of memory even after retracting all other requests".into(),
                });
                if let Some(node) = req.cached_node.take() {
                    if let Some(cache) = radix_cache.as_deref_mut() {
                        cache.dec_lock_ref(node);
                    }
                }
                if let Some(slot) = req.req_pool_idx.take() {
                    let len = req.kv_allocated_len;
                    if len > 0 {
                        if let Ok(slot_indices) = req_pool.read(slot, len) {
                            let owned = slot_indices.to_vec();
                            page_tracker.free_i32(&owned);
                        }
                    }
                    let _ = req_pool.free(slot);
                }
            }
            outcome.aborted += 1;
            // Remove from batch — keep the indices consistent.
            batch.reqs.remove(victim_pos);
            batch.req_pool_indices.remove(victim_pos);
            if !batch.seq_lens.is_empty() {
                batch.seq_lens.remove(victim_pos);
            }
            if !batch.orig_seq_lens.is_empty() {
                batch.orig_seq_lens.remove(victim_pos);
            }
            break;
        }

        // Normal retraction: free KV, reset bookkeeping, push back to
        // the waiting queue for re-prefill.
        let req_arc = batch.reqs[victim_pos].clone();
        {
            let mut req = req_arc.write().unwrap();
            // Release the cache lock the admission step took.
            if let Some(node) = req.cached_node.take() {
                if let Some(cache) = radix_cache.as_deref_mut() {
                    cache.dec_lock_ref(node);
                }
            }
            if let Some(slot) = req.req_pool_idx.take() {
                let len = req.kv_allocated_len;
                if len > 0 {
                    if let Ok(slot_indices) = req_pool.read(slot, len) {
                        let owned = slot_indices.to_vec();
                        page_tracker.free_i32(&owned);
                    }
                }
                let _ = req_pool.free(slot);
            }
            req.reset_for_retract();
        }
        waiting.push(req_arc);
        outcome.retracted_reqs += 1;

        batch.reqs.remove(victim_pos);
        batch.req_pool_indices.remove(victim_pos);
        if !batch.seq_lens.is_empty() {
            batch.seq_lens.remove(victim_pos);
        }
        if !batch.orig_seq_lens.is_empty() {
            batch.orig_seq_lens.remove(victim_pos);
        }

        // Rebuild `order` indices since later positions shifted down.
        for o in order.iter_mut() {
            if *o > victim_pos {
                *o -= 1;
            }
        }
    }

    outcome
}

/// Full memory-pressure recovery: try radix-cache eviction first (if a
/// cache is wired), then fall back to retracting running reqs.  Source:
/// `Scheduler.check_decode_mem` in `scheduler.py`.
///
/// `cache = None` selects the `ChunkedCache` path — no eviction, just
/// retract running reqs.  Mirrors Python's `ChunkCache.evict` no-op.
pub fn reclaim_for_decode(
    batch: &mut ScheduleBatch,
    waiting: &mut WaitingQueue,
    page_tracker: &mut CpuPageTracker,
    req_pool: &mut ReqToTokenPool,
    mut cache: Option<&mut RadixCache>,
    needed_tokens_per_step: i64,
) -> RetractionOutcome {
    // 1. Evict unprotected radix-cache leaves (if a cache is wired).
    let evicted = if page_tracker.available_size() < needed_tokens_per_step {
        let want = needed_tokens_per_step - page_tracker.available_size();
        if let Some(c) = cache.as_deref_mut() {
            evict_from_tree_cache(c, page_tracker, want)
        } else {
            0
        }
    } else {
        0
    };

    // 2. If that wasn't enough, retract running reqs.  The cache (if
    //    any) gets its lock_refs released as part of the retraction.
    let mut outcome = if page_tracker.available_size() < needed_tokens_per_step {
        retract_decode_with_cache(
            batch,
            waiting,
            page_tracker,
            req_pool,
            cache.as_deref_mut(),
            needed_tokens_per_step,
        )
    } else {
        RetractionOutcome::default()
    };
    outcome.evicted_cache_tokens = evicted;
    outcome
}

/// Convenience for tests.
#[allow(dead_code)]
pub(crate) fn _push_req(batch: &mut ScheduleBatch, req: Arc<RwLock<Req>>, slot: u32, seq_len: i32) {
    batch.req_pool_indices.push(slot);
    batch.seq_lens.push(seq_len);
    batch.orig_seq_lens.push(seq_len);
    batch.reqs.push(req);
}
