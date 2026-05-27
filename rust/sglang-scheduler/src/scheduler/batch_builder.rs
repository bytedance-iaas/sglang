// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Batch builder — picks the next batch to run.
//!
//! Source: `Scheduler.get_next_batch_to_run`,
//! `Scheduler.update_running_batch`, `Scheduler.get_new_batch_prefill`
//! in `python/sglang/srt/managers/scheduler.py`.
//!
//! Today this is a deliberately minimal stub:
//!   * Decode-only: filter finished, increment seq_lens by 1, set
//!     `input_ids` from each req's last sampled token.
//!   * Extend: if the running batch is empty and the waiting queue has
//!     requests, admit them (no radix cache, no chunked prefill).
//!
//! What's missing — and why each item is genuinely its own port:
//!   * Radix cache prefix match.  (Self-contained data structure.)
//!   * Retraction under memory pressure.  Needs the page tracker plus
//!     a stable retract-by-id contract with the worker.
//!   * Chunked prefill — `policies/chunked_prefill.py`.
//!   * DP-attention sync — `scheduler_dp_attn_mixin.py`.
//!   * Spec-decoding paths.
//!   * Hisparse / mamba / encoder-decoder.

use std::sync::{Arc, RwLock};

use crate::memory::{CpuPageTracker, ReqToTokenPool};
use crate::queue::{SchedulePolicy, WaitingQueue};
use crate::radix_cache::RadixCache;
use crate::types::{ForwardMode, Req, ScheduleBatch};

pub struct BatchBuilder {
    page_size: i64,
    max_running_requests: usize,
    /// Cap on the **new** tokens admitted into a single prefill batch.
    /// Mirrors Python `PrefillAdder(rem_input_tokens=max_prefill_tokens)`
    /// (see `scheduler.py:2751`).  Without this cap, the admission only
    /// bounds by the KV-pool budget — for large KV pools that lets a
    /// single prefill iter ship 100K+ input tokens to the worker, which
    /// blows the MoE kernels' `m_numtopk <= MAX_TOKENS_PER_EXPERT * topk`
    /// assertion (`MAX_TOKENS_PER_EXPERT = 65536`).
    max_prefill_tokens: i64,
    policy: SchedulePolicy,
}

impl BatchBuilder {
    pub fn new(
        page_size: i64,
        max_running_requests: usize,
        max_prefill_tokens: i64,
        policy: SchedulePolicy,
    ) -> Self {
        Self {
            page_size,
            max_running_requests,
            max_prefill_tokens,
            policy,
        }
    }

    /// Update the current decode batch: filter finished reqs, advance
    /// seq lens, set next-step input ids.  Mirrors
    /// `ScheduleBatch.prepare_for_decode` in `schedule_batch.py`.
    pub fn prepare_for_decode(&self, batch: &mut ScheduleBatch) {
        batch.filter_finished();
        if batch.is_empty() {
            return;
        }
        batch.forward_mode = ForwardMode::Decode;
        let bs = batch.reqs.len();
        batch.input_ids.clear();
        batch.input_ids.reserve(bs);
        for (i, req_lock) in batch.reqs.iter().enumerate() {
            let mut req = req_lock.write().unwrap();
            let next_input = req
                .output_ids
                .last()
                .copied()
                .unwrap_or_else(|| *req.origin_input_ids.last().unwrap_or(&0));
            batch.input_ids.push(next_input);
            batch.seq_lens[i] += 1;
            req.kv_allocated_len += 1;
            req.kv_committed_len += 1;
        }
    }

    /// Try to build a prefill batch from the waiting queue.  Returns
    /// `None` when nothing is admissible (queue empty, no req-pool
    /// slots, or no KV budget).
    ///
    /// When `radix_cache` is `Some`, the admitted reqs `inc_lock_ref`
    /// the longest matching prefix and have their `cached_node` /
    /// `prefix_len_from_cache` set so the output processor can release
    /// the lock on finish.
    pub fn get_new_batch_prefill(
        &self,
        running: &ScheduleBatch,
        waiting: &mut WaitingQueue,
        page_tracker: &CpuPageTracker,
        req_pool: &mut ReqToTokenPool,
    ) -> Option<ScheduleBatch> {
        self.get_new_batch_prefill_with_cache(
            running,
            waiting,
            page_tracker,
            req_pool,
            None,
        )
    }

    /// Variant of `get_new_batch_prefill` that consults the radix cache
    /// for prefix reuse.  Pulled out as a separate method so callers
    /// without a cache (tests, the embedding path) don't have to fake
    /// one.
    pub fn get_new_batch_prefill_with_cache(
        &self,
        running: &ScheduleBatch,
        waiting: &mut WaitingQueue,
        page_tracker: &CpuPageTracker,
        req_pool: &mut ReqToTokenPool,
        mut radix_cache: Option<&mut RadixCache>,
    ) -> Option<ScheduleBatch> {
        if waiting.is_empty() {
            return None;
        }
        let remaining_run_slots =
            self.max_running_requests.saturating_sub(running.batch_size());
        let remaining_pool_slots = req_pool.available_size() as usize;
        let admission_cap = remaining_run_slots.min(remaining_pool_slots);
        if admission_cap == 0 {
            return None;
        }

        // Per-iter prefill token budget.  Two independent caps:
        //   1. `free_tokens` — the KV-pool budget (admission can't
        //      allocate more KV slots than the pool has free).
        //   2. `max_prefill_tokens` — the per-iter prefill ceiling
        //      (Python `PrefillAdder.rem_input_tokens`, default 8192).
        //      Without this cap, large KV pools let a single prefill
        //      iter ship 100K+ new tokens, blowing the MoE kernel's
        //      `m_numtopk <= MAX_TOKENS_PER_EXPERT * topk` assertion
        //      (`MAX_TOKENS_PER_EXPERT = 65536`).
        //
        // TODO(rust-port): the Python version also uses `PrefillAdder`
        // to account for post-admission seq-length growth (much larger
        // than the prefill length itself) and to chunk long prompts
        // into `chunked_prefill_size` pieces.  Until that lands, a
        // single oversized req gets dropped with a warning rather than
        // sent and OOM'd at the worker.
        let free_tokens = page_tracker.available_size();
        let token_budget = free_tokens.min(self.max_prefill_tokens);
        let mut admitted: Vec<Arc<RwLock<Req>>> = Vec::new();
        let mut accumulated_tokens: i64 = 0;

        let drained = waiting.drain_with_cache(
            admission_cap,
            &self.policy,
            radix_cache.as_deref().map(|c| &*c),
        );
        for req_arc in drained {
            // Account for prefix reuse: a req with a 100-token prompt
            // and a 60-token cached prefix only needs 40 new KV slots.
            //
            // NOTE: cap the matched length at `input_len - 1` (same
            // rule as Python's `max_prefix_len = input_len - 1`).  The
            // worker must forward at least one token per req to
            // compute logits for sampling — and the wire serialization
            // can't represent an empty `input_ids` tensor (msgspec's
            // `torch.frombuffer(b"", count=-1)` errors out).
            let (total_len, prefix_len) = {
                let r = req_arc.read().unwrap();
                let total = r.origin_input_ids.len() as i64;
                let raw = match radix_cache.as_deref() {
                    Some(cache) => cache.match_prefix(&r.origin_input_ids).prefix_len as i64,
                    None => 0,
                };
                (total, raw.min((total - 1).max(0)))
            };
            let needed = total_len - prefix_len;

            // A req whose lone new-token count exceeds the per-iter
            // budget can't fit even in an empty batch.  Until chunked
            // prefill is ported, log a warning and push back to
            // waiting — the user will hit a timeout rather than a
            // silent OOM/wire-blow-up at the worker.
            if needed > self.max_prefill_tokens && admitted.is_empty() {
                let rid = req_arc.read().unwrap().rid.clone();
                log::warn!(
                    "req {} needs {} new tokens (> max_prefill_tokens={}); \
                     chunked prefill not yet ported — request will stall",
                    rid,
                    needed,
                    self.max_prefill_tokens
                );
                waiting.push(req_arc);
                break;
            }

            if accumulated_tokens + needed > token_budget {
                waiting.push(req_arc);
                break;
            }
            accumulated_tokens += needed;
            admitted.push(req_arc);
        }

        if admitted.is_empty() {
            return None;
        }

        let slots = match req_pool.alloc(admitted.len() as u32) {
            Some(slots) => slots,
            None => {
                for req in admitted {
                    waiting.push(req);
                }
                return None;
            }
        };

        let mut batch = ScheduleBatch::new(ForwardMode::Extend);
        for (i, req_arc) in admitted.into_iter().enumerate() {
            let slot = slots[i];
            {
                let mut req = req_arc.write().unwrap();
                let len = req.origin_input_ids.len() as i32;
                req.req_pool_idx = Some(slot);

                // Pin the matched prefix so eviction can't drop it
                // while this req is running, and capture the slot
                // indices so we can plant them into `req_pool` for the
                // worker's GPU req_to_token to consume during attention.
                //
                // Source of truth: `tp_worker_server.py:558` reads
                // `req_to_token_cpu[req_pool_indices]` and writes it
                // into `req_to_token_gpu`.  Without this, the worker
                // attends against an empty row of slot indices and
                // returns garbage (this was the cause of "req2 returns
                // garbage" in the user's bug report).
                let mut prefix_slots: Vec<i32> = Vec::new();
                if let Some(cache) = radix_cache.as_deref_mut() {
                    let mut m = cache.match_prefix(&req.origin_input_ids);
                    // Same cap as the accounting loop above and the
                    // Python `max_prefix_len = input_len - 1` rule:
                    // every req must have at least one token to
                    // prefill so the worker can compute logits.
                    let cap = req.origin_input_ids.len().saturating_sub(1);
                    if m.prefix_len > cap {
                        m.prefix_len = cap;
                        m.slots.truncate(cap);
                    }
                    if m.prefix_len > 0 {
                        cache.inc_lock_ref(m.last_node);
                        req.cached_node = Some(m.last_node);
                        req.prefix_len_from_cache = m.prefix_len as u32;
                        req.cache_protected_len = m.prefix_len as u32;
                        prefix_slots = m.slots;
                    }
                }

                // CRITICAL: only the post-prefix tokens go into
                // `batch.input_ids`.  The worker computes
                // `extend_lens = seq_lens - prefix_lens` and expects
                // `input_ids.len() == sum(extend_lens)`.  Pushing the
                // full prompt while reporting a non-zero prefix_len
                // walks off the end of out_cache_loc → KV corruption.
                let prefix_len = req.prefix_len_from_cache as usize;
                batch
                    .input_ids
                    .extend_from_slice(&req.origin_input_ids[prefix_len..]);
                batch.seq_lens.push(len);
                batch.orig_seq_lens.push(len);
                batch.req_pool_indices.push(slot);

                // Plant the cached prefix slots into the CPU
                // `req_to_token` at columns `0..prefix_len`.  The
                // `to_model_worker_batch_payload` snapshot copies this
                // row into `req_to_token_cpu` so the worker can write
                // it into `req_to_token_gpu` before its forward pass.
                if !prefix_slots.is_empty() {
                    if let Err(err) = req_pool.write(slot, 0, &prefix_slots) {
                        log::warn!(
                            "failed to plant cached prefix slots at slot {slot}: {err}"
                        );
                    }
                }
            }
            batch.reqs.push(req_arc);
        }
        Some(batch)
    }

    pub fn page_size(&self) -> i64 {
        self.page_size
    }
}
