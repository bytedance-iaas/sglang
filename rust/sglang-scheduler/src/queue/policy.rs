// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Scheduling policies — the ordering applied to the waiting queue
//! when building the next prefill batch.
//!
//! The Python policies live in
//! `python/sglang/srt/managers/policies/` and the active one is
//! selected by `ServerArgs.schedule_policy`.

use std::sync::Arc;
use std::sync::RwLock;

use crate::radix_cache::RadixCache;
use crate::types::Req;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulePolicyKind {
    /// First-come-first-served — admission order is arrival order.
    Fcfs,
    /// Prefer shortest unfinished output (encourages quick turnaround).
    Lof,
    /// Prefer longest radix-cache prefix match (re-use hot prefixes).
    /// Needs the radix cache to score; falls back to FCFS until then.
    LongestPrefix,
    /// User-provided per-request priority.
    Priority,
}

impl SchedulePolicyKind {
    /// Parse `ServerArgs.schedule_policy`.  Unknown names fall back to
    /// FCFS with a warning — same forgiving behaviour as the Python
    /// constructor.
    pub fn parse(name: &str) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "fcfs" => SchedulePolicyKind::Fcfs,
            "lof" => SchedulePolicyKind::Lof,
            "longest-prefix" | "longest_prefix" | "lpm" => SchedulePolicyKind::LongestPrefix,
            "priority" => SchedulePolicyKind::Priority,
            other => {
                log::warn!("unknown schedule_policy={other:?}; falling back to fcfs");
                SchedulePolicyKind::Fcfs
            }
        }
    }
}

/// Policy applied to the waiting queue.  Today this is just a sorting
/// strategy; the Python equivalent also drives chunked-prefill and
/// over-allocation guards (`policies/policy.py`).  Those will land
/// when batch building gets more complete.
#[derive(Debug, Clone, Copy)]
pub struct SchedulePolicy {
    pub kind: SchedulePolicyKind,
}

impl SchedulePolicy {
    pub fn new(kind: SchedulePolicyKind) -> Self {
        Self { kind }
    }

    /// Order the waiting queue.  Returns indices into `reqs` in admission
    /// order — first one returned is the next to admit.
    pub fn order(&self, reqs: &[Arc<RwLock<Req>>]) -> Vec<usize> {
        self.order_with_cache(reqs, None)
    }

    /// Variant of `order` that scores prefix overlap via the radix
    /// cache.  Used by `LongestPrefix`; the other policies ignore the
    /// cache argument.
    pub fn order_with_cache(
        &self,
        reqs: &[Arc<RwLock<Req>>],
        cache: Option<&RadixCache>,
    ) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..reqs.len()).collect();
        match self.kind {
            SchedulePolicyKind::Fcfs => {}
            SchedulePolicyKind::Lof => {
                idx.sort_by_key(|&i| {
                    reqs[i].read().unwrap().sampling_params.max_new_tokens
                });
            }
            SchedulePolicyKind::LongestPrefix => {
                // Sort by prefix match length, descending.  Ties break
                // by arrival order (stable sort).
                if let Some(cache) = cache {
                    let scores: Vec<usize> = reqs
                        .iter()
                        .map(|r| {
                            let r = r.read().unwrap();
                            cache.match_prefix(&r.origin_input_ids).prefix_len
                        })
                        .collect();
                    idx.sort_by(|&a, &b| scores[b].cmp(&scores[a]));
                }
                // Without a cache: keep FCFS ordering.
            }
            SchedulePolicyKind::Priority => {
                // TODO(rust-port): port priority field on Req.  Stub: FCFS.
            }
        }
        idx
    }
}
