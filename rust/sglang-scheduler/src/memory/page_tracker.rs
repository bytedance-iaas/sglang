// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Rust port of `CpuPageTracker` from
//! `python/sglang/srt/managers/scheduler_cpu.py`.
//!
//! The CPU scheduler defers all KV slot index computation to the GPU
//! worker; on the scheduler side we only need to know:
//!
//! 1. The current free-page budget (for admission decisions).
//! 2. The set of slot indices that have been freed locally but not yet
//!    propagated to the GPU.  These ship with the next batch's
//!    ``indices_to_free`` field and get processed by the worker before
//!    its next allocation pass.
//!
//! `available_size()` is **always derived** as
//! ``_base_free_pages + unique(_pending_free) ⋅ page_size``.  This avoids
//! the drift-on-overwrite bug the Python version hit before the same
//! restructure (see git blame on `scheduler_cpu.py:update_free_count`):
//! per-call increments on `free()` lose precision when the same page is
//! freed by multiple paths (radix tail-free + adjacent eviction), so we
//! deduplicate on every `available_size()` query.
//!
//! Page 0 is reserved by the GPU allocator (padded dummy-output slot)
//! and must never appear in `indices_to_free`; the filter here mirrors
//! the same defensive strip in the Python version.

use std::collections::HashSet;

#[derive(Debug)]
pub struct CpuPageTracker {
    /// Total number of pages in the GPU allocator (page 0 is reserved).
    total_pages: i64,
    /// Page size in tokens — also the slot-index → page-index divisor.
    page_size: i64,
    /// GPU worker's last authoritative free-page count.  Updated by
    /// `update_free_count`.
    base_free_pages: i64,
    /// Slot indices freed locally since the last `drain_pending_free`.
    /// Concatenated into one tensor when shipped to the worker.
    pending_free: Vec<i64>,
}

impl CpuPageTracker {
    pub fn new(total_pages: i64, page_size: i64) -> Self {
        Self {
            total_pages,
            page_size,
            base_free_pages: total_pages,
            pending_free: Vec::new(),
        }
    }

    pub fn total_pages(&self) -> i64 {
        self.total_pages
    }

    pub fn page_size(&self) -> i64 {
        self.page_size
    }

    /// Free tokens = (GPU's last report + un-propagated pages) ⋅ page_size.
    pub fn available_size(&self) -> i64 {
        (self.base_free_pages + self.pending_unique_pages() as i64) * self.page_size
    }

    /// Accumulate freed slot indices.  Indices in page 0 (slot < page_size)
    /// are stripped — they're always spurious reads from un-written
    /// `req_to_token` entries, never legitimate allocations.
    pub fn free(&mut self, indices: &[i64]) {
        if indices.is_empty() {
            return;
        }
        let threshold = self.page_size;
        for &idx in indices {
            if idx >= threshold {
                self.pending_free.push(idx);
            }
        }
    }

    /// Same as `free` but for `i32` indices (the GPU side's typical
    /// `out_cache_loc` dtype).
    pub fn free_i32(&mut self, indices: &[i32]) {
        if indices.is_empty() {
            return;
        }
        let threshold = self.page_size;
        for &idx in indices {
            let idx = idx as i64;
            if idx >= threshold {
                self.pending_free.push(idx);
            }
        }
    }

    /// Return and clear all accumulated free indices.  The caller ships
    /// them as the next batch's `indices_to_free` — once they reach the
    /// worker, the worker's free count will include them, so we drop our
    /// local copies here.
    pub fn drain_pending_free(&mut self) -> Vec<i64> {
        std::mem::take(&mut self.pending_free)
    }

    /// Sync the GPU worker's authoritative free-page count.  Anything
    /// still in `pending_free` is invisible to the GPU and layered on
    /// top by `available_size`.
    pub fn update_free_count(&mut self, free_pages: i64) {
        self.base_free_pages = free_pages;
    }

    /// Reset to the post-clear state: page 0 reserved, all others free.
    pub fn clear(&mut self) {
        self.base_free_pages = self.total_pages - 1;
        self.pending_free.clear();
    }

    /// Unique pages across all pending tensors.  Matches the GPU
    /// allocator's dedup (`unique(idx // page_size)`) — the GPU only
    /// records each freed page once, so the CPU view must too.
    fn pending_unique_pages(&self) -> usize {
        if self.pending_free.is_empty() {
            return 0;
        }
        // The pool is small enough (a few thousand pages at most) that
        // HashSet is cheap; saves an allocator round-trip vs sort+dedup.
        let mut seen = HashSet::with_capacity(self.pending_free.len());
        for &idx in &self.pending_free {
            seen.insert(idx / self.page_size);
        }
        seen.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: i64 = 16;

    #[test]
    fn basic_free_and_drain() {
        let mut t = CpuPageTracker::new(1000, PAGE);
        assert_eq!(t.available_size(), 1000 * PAGE);

        // Three indices in three distinct pages.
        t.free(&[16, 32, 48]);
        assert_eq!(t.available_size(), (1000 + 3) * PAGE);

        // GPU reports it sees 900 free; we layer pending on top.
        t.update_free_count(900);
        assert_eq!(t.available_size(), (900 + 3) * PAGE);

        // Drain (sent to GPU) — pending count clears.
        let drained = t.drain_pending_free();
        assert_eq!(drained.len(), 3);
        assert_eq!(t.available_size(), 900 * PAGE);

        // GPU now knows; report includes the freed.
        t.update_free_count(903);
        assert_eq!(t.available_size(), 903 * PAGE);
    }

    #[test]
    fn duplicate_pages_dedup() {
        let mut t = CpuPageTracker::new(1000, PAGE);
        t.free(&[16]); // page 1
        t.free(&[17]); // also page 1
        assert_eq!(t.available_size(), (1000 + 1) * PAGE);
        t.update_free_count(900);
        assert_eq!(t.available_size(), (900 + 1) * PAGE);
    }

    #[test]
    fn page_zero_is_stripped() {
        let mut t = CpuPageTracker::new(1000, PAGE);
        // Index in page 0 — never a legit allocation.
        t.free(&[0, 5, 15, 16]);
        // Only page 1 (idx 16) counts.
        assert_eq!(t.available_size(), (1000 + 1) * PAGE);
        let drained = t.drain_pending_free();
        assert_eq!(drained, vec![16]);
    }

    #[test]
    fn multi_tensor_dedup() {
        let mut t = CpuPageTracker::new(1000, PAGE);
        t.free(&[16, 32, 48]); // pages 1, 2, 3
        t.free(&[20, 64]); // page 1 (dup), page 4
        // Unique pages = {1, 2, 3, 4} = 4
        assert_eq!(t.available_size(), (1000 + 4) * PAGE);
    }
}
