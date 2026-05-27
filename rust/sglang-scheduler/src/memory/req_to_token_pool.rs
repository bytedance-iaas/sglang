// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! `ReqToTokenPool` — CPU-side table mapping per-request slots to the
//! KV-cache token indices that hold their KV.
//!
//! Source: `python/sglang/srt/mem_cache/memory_pool.py:ReqToTokenPool`.
//!
//! The Python pool stores a 2-D `int32` tensor of shape
//! `(size + 1, max_context_len)` where row 0 is a padding slot for
//! cuda-graph-padded batches.  The Rust port matches that layout — a
//! flat `Vec<i32>` indexed via `row * max_context_len + col`.
//!
//! Allocation:
//!   * `alloc(reqs)` reserves a row per req without one, in arrival
//!     order, returning the assigned slot ids.  Reqs that already own a
//!     slot (chunked prefill continuing across chunks) keep it.
//!   * `free(req_pool_idx)` returns the slot to the free list.
//!   * `write(row, col_range, values)` plants the KV slot indices the
//!     worker reports back in `DeferredAllocIPC.out_cache_loc`.
//!
//! What's intentionally NOT mirrored here:
//!   * `enable_memory_saver` / `TorchMemorySaverAdapter` — CUDA-side
//!     concern, doesn't apply to the CPU table.
//!   * `MambaPool` — separate Mamba state pool, postponed until the
//!     Mamba schedule path lands.

#[derive(Debug, thiserror::Error)]
pub enum ReqToTokenPoolError {
    #[error("no free req_pool slots (size={size})")]
    Exhausted { size: u32 },
    #[error("free called on slot {slot} which is not currently allocated")]
    DoubleFree { slot: u32 },
    #[error(
        "write out of bounds: row {row} cols {start}..{end} (max_context_len={max_context_len})"
    )]
    WriteOob {
        row: u32,
        start: u32,
        end: u32,
        max_context_len: u32,
    },
}

pub struct ReqToTokenPool {
    size: u32,
    max_context_len: u32,
    /// Row-major `(size + 1) * max_context_len` `int32` table.  Row 0 is
    /// the padding slot; slots 1..=size are user-allocatable.
    req_to_token: Vec<i32>,
    /// Free slot ids, kept as a stack so we hand out the most-recently
    /// freed slot first — matches the Python `list.append` / slice
    /// allocation pattern.
    free_slots: Vec<u32>,
    /// Counter for `available_size` without walking `free_slots.len()`
    /// on every query.
    available: u32,
}

impl ReqToTokenPool {
    pub fn new(size: u32, max_context_len: u32) -> Self {
        let table_rows = (size + 1) as usize;
        let table_cols = max_context_len as usize;
        let req_to_token = vec![0i32; table_rows * table_cols];
        // Free slots start at 1..=size (row 0 reserved for padding).
        // Python uses ascending order; mirror that so the lowest free
        // slot is handed out first.  We pop from the end, so push in
        // reverse.
        let free_slots: Vec<u32> = (1..=size).rev().collect();
        Self {
            size,
            max_context_len,
            req_to_token,
            free_slots,
            available: size,
        }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn max_context_len(&self) -> u32 {
        self.max_context_len
    }

    pub fn available_size(&self) -> u32 {
        self.available
    }

    /// Allocate `count` slot ids, in ascending order.  Returns `None`
    /// when the pool is exhausted (mirrors `Python.alloc` returning
    /// `None` so the caller can refuse the admission).
    pub fn alloc(&mut self, count: u32) -> Option<Vec<u32>> {
        if count > self.available {
            return None;
        }
        let mut out = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let slot = self.free_slots.pop().expect("available accounting");
            out.push(slot);
        }
        self.available -= count;
        // Python returns slots in ascending order (`free_slots[:need_size]`);
        // our reverse-stack hands them out ascending naturally if the pool
        // hasn't been freed-into yet.  After mixed alloc/free the order
        // is no longer guaranteed ascending — the Python side doesn't
        // care either (the caller treats them as opaque ids).
        Some(out)
    }

    /// Return a single slot to the free list.  Errors if the slot is
    /// already free or out of range — the Python `free` just asserts.
    pub fn free(&mut self, slot: u32) -> Result<(), ReqToTokenPoolError> {
        if slot == 0 || slot > self.size {
            return Err(ReqToTokenPoolError::DoubleFree { slot });
        }
        // Don't double-free.  O(n) check, fine for the typical pool
        // size (a few thousand slots); swap for a bitset if profiling
        // shows it matters.
        if self.free_slots.contains(&slot) {
            return Err(ReqToTokenPoolError::DoubleFree { slot });
        }
        self.free_slots.push(slot);
        self.available += 1;
        Ok(())
    }

    /// Write KV slot indices into row `req_pool_idx` columns
    /// `start..start+values.len()`.  The Python equivalent is
    /// `self.req_to_token[req_pool_idx, start:start+len(values)] = values`.
    pub fn write(
        &mut self,
        req_pool_idx: u32,
        start: u32,
        values: &[i32],
    ) -> Result<(), ReqToTokenPoolError> {
        let end = start + values.len() as u32;
        if req_pool_idx > self.size || end > self.max_context_len {
            return Err(ReqToTokenPoolError::WriteOob {
                row: req_pool_idx,
                start,
                end,
                max_context_len: self.max_context_len,
            });
        }
        let row = req_pool_idx as usize;
        let cols = self.max_context_len as usize;
        let dst = &mut self.req_to_token
            [row * cols + start as usize..row * cols + end as usize];
        dst.copy_from_slice(values);
        Ok(())
    }

    /// Read the slot indices at row `req_pool_idx` columns `0..len`.
    /// Used by `release_kv_cache` to discover which KV slots a finished
    /// request owned.
    pub fn read(&self, req_pool_idx: u32, len: u32) -> Result<&[i32], ReqToTokenPoolError> {
        if req_pool_idx > self.size || len > self.max_context_len {
            return Err(ReqToTokenPoolError::WriteOob {
                row: req_pool_idx,
                start: 0,
                end: len,
                max_context_len: self.max_context_len,
            });
        }
        let row = req_pool_idx as usize;
        let cols = self.max_context_len as usize;
        Ok(&self.req_to_token[row * cols..row * cols + len as usize])
    }

    /// Return the row to its zero-initialised state and put the slot
    /// back on the free list — used at shutdown / clear.
    pub fn clear(&mut self) {
        for v in self.req_to_token.iter_mut() {
            *v = 0;
        }
        self.free_slots = (1..=self.size).rev().collect();
        self.available = self.size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_then_free() {
        let mut pool = ReqToTokenPool::new(4, 32);
        assert_eq!(pool.available_size(), 4);
        let slots = pool.alloc(3).expect("alloc 3");
        assert_eq!(slots.len(), 3);
        assert_eq!(pool.available_size(), 1);
        pool.free(slots[1]).unwrap();
        assert_eq!(pool.available_size(), 2);
    }

    #[test]
    fn exhausted_alloc_returns_none() {
        let mut pool = ReqToTokenPool::new(2, 8);
        let _ = pool.alloc(2).unwrap();
        assert!(pool.alloc(1).is_none());
    }

    #[test]
    fn write_and_read_back() {
        let mut pool = ReqToTokenPool::new(4, 8);
        let slots = pool.alloc(1).unwrap();
        pool.write(slots[0], 0, &[100, 101, 102]).unwrap();
        assert_eq!(pool.read(slots[0], 3).unwrap(), &[100, 101, 102]);
    }

    #[test]
    fn write_oob_errors() {
        let mut pool = ReqToTokenPool::new(2, 4);
        let slots = pool.alloc(1).unwrap();
        let err = pool.write(slots[0], 0, &[1, 2, 3, 4, 5]).unwrap_err();
        assert!(matches!(err, ReqToTokenPoolError::WriteOob { .. }));
    }

    #[test]
    fn double_free_errors() {
        let mut pool = ReqToTokenPool::new(2, 4);
        let slots = pool.alloc(1).unwrap();
        pool.free(slots[0]).unwrap();
        assert!(pool.free(slots[0]).is_err());
    }

    #[test]
    fn slot_zero_is_reserved() {
        let mut pool = ReqToTokenPool::new(4, 4);
        // The padding row should never be handed out.
        for _ in 0..4 {
            let s = pool.alloc(1).unwrap()[0];
            assert!(s >= 1 && s <= 4);
        }
    }
}
