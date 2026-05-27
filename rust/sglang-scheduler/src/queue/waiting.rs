// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Waiting queue — requests that have been admitted by the scheduler
//! but have not yet been included in a forward batch.
//!
//! Source: `Scheduler.waiting_queue` (a plain `List[Req]` in Python).
//! The Rust version owns the requests via `Arc<RwLock<Req>>` so the
//! batch builder can hold a reference into the queue without copying.

use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

use crate::queue::policy::SchedulePolicy;
use crate::radix_cache::RadixCache;
use crate::types::Req;

#[derive(Debug, Default)]
pub struct WaitingQueue {
    inner: VecDeque<Arc<RwLock<Req>>>,
}

impl WaitingQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn push(&mut self, req: Arc<RwLock<Req>>) {
        self.inner.push_back(req);
    }

    /// View into the underlying slice — for policy ordering and read-only
    /// inspection.
    pub fn as_slice(&self) -> Vec<Arc<RwLock<Req>>> {
        // VecDeque can't expose a single slice; collect cheap Arcs.
        self.inner.iter().cloned().collect()
    }

    /// Remove and return up to `max_count` requests in policy order.
    /// Mirrors the per-iter admission loop in
    /// `Scheduler.get_new_batch_prefill`.
    pub fn drain(&mut self, max_count: usize, policy: &SchedulePolicy) -> Vec<Arc<RwLock<Req>>> {
        self.drain_with_cache(max_count, policy, None)
    }

    /// Cache-aware drain — required by the `LongestPrefix` policy so
    /// it can score reqs by their radix-cache overlap.
    pub fn drain_with_cache(
        &mut self,
        max_count: usize,
        policy: &SchedulePolicy,
        cache: Option<&RadixCache>,
    ) -> Vec<Arc<RwLock<Req>>> {
        if self.inner.is_empty() || max_count == 0 {
            return Vec::new();
        }
        let snapshot = self.as_slice();
        let take: Vec<usize> = policy
            .order_with_cache(&snapshot, cache)
            .into_iter()
            .take(max_count)
            .collect();

        // Build the admitted vector in policy order.
        let admitted: Vec<_> = take.iter().map(|&i| snapshot[i].clone()).collect();

        // Rebuild the queue with the survivors, preserving their
        // original arrival order.
        let mut to_remove = vec![false; snapshot.len()];
        for &i in &take {
            to_remove[i] = true;
        }
        let drained: Vec<_> = self.inner.drain(..).collect();
        let mut surviving = VecDeque::with_capacity(drained.len() - take.len());
        for (i, req) in drained.into_iter().enumerate() {
            if !to_remove[i] {
                surviving.push_back(req);
            }
        }
        self.inner = surviving;

        admitted
    }

    /// Drop reqs by rid — used by the abort path.  Returns the count
    /// removed.
    pub fn drop_by_rid(&mut self, rid: &str) -> usize {
        let before = self.inner.len();
        self.inner.retain(|r| r.read().unwrap().rid != rid);
        before - self.inner.len()
    }

    /// Drop **all** queued requests — used by `AbortReq { abort_all: true }`.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}
