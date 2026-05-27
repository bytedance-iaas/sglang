// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Radix-cache port for KV-prefix reuse.
//!
//! Source: `python/sglang/srt/mem_cache/radix_cache.py`.
//!
//! ## Scope of this port
//!
//! Covers the operations the scheduler needs to drive prefix reuse:
//!
//!   1. `match_prefix(tokens) -> matched_slot_indices, last_node`
//!      — walk the tree, return the longest prefix that's cached.
//!   2. `insert(tokens, slot_indices) -> dedup_overlap` — graft the
//!      finished req's KV into the tree, returning the prefix length
//!      that already existed (so the caller can free those slot
//!      indices).
//!   3. `inc_lock_ref` / `dec_lock_ref` — pin / unpin the path from
//!      `node` up to root.  Running reqs hold a lock on the node
//!      returned by `match_prefix` so eviction can't drop their
//!      prefix out from under them.
//!   4. `evict(num_tokens) -> freed_slot_indices` — drop *unlocked*
//!      leaves until the budget is met.
//!
//! What's **not** ported in this cut:
//!   * Page-aligned eviction granularity.  Python evicts whole pages;
//!     this cut treats each KV slot index as a single unit.  For
//!     `page_size = 1` (the default in `ServerArgs`) these are
//!     equivalent.
//!   * `enable_metrics` / detailed timing.
//!   * HiCache (multi-tier KV offload) — separate subsystem.
//!
//! ## Data structure
//!
//! Each node carries a token-id slice (its label edge from the parent)
//! and a slot-index slice (the KV positions for those tokens).  Children
//! are indexed by the first token in their label — same as the Python
//! `TreeNode.children: Dict[int, TreeNode]`.

use std::collections::HashMap;

/// Identifier for a tree node — index into the `nodes` arena.  Using an
/// arena keeps the cache cache-friendly and lets multiple ports refer to
/// nodes by id without owning them.
pub type NodeId = u32;

const ROOT: NodeId = 0;

#[derive(Debug, Clone)]
struct Node {
    /// Edge label from the parent — the token ids that lead here.
    key: Vec<i32>,
    /// KV slot indices the worker assigned for `key`.  Same length as
    /// `key` (one slot per token; when `page_size > 1` the slots align
    /// to pages but we keep the per-token mapping for simplicity).
    value: Vec<i32>,
    parent: Option<NodeId>,
    children: HashMap<i32, NodeId>,
    /// Total bytes of `value` across this node — kept as i64 for the
    /// future page-aligned port; today it's `value.len() as i64`.
    cached_tokens: i64,
    /// Number of running requests whose prefix passes through this
    /// node.  When `> 0` the node (and the slot indices it owns) is
    /// protected from eviction.  Maintained by `inc_lock_ref` /
    /// `dec_lock_ref`, which walk the chain from a leaf up to root.
    lock_ref: u32,
}

impl Node {
    fn root() -> Self {
        Node {
            key: Vec::new(),
            value: Vec::new(),
            parent: None,
            children: HashMap::new(),
            cached_tokens: 0,
            lock_ref: 0,
        }
    }

    fn leaf(parent: NodeId, key: Vec<i32>, value: Vec<i32>) -> Self {
        let cached_tokens = value.len() as i64;
        Node {
            key,
            value,
            parent: Some(parent),
            children: HashMap::new(),
            cached_tokens,
            lock_ref: 0,
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct MatchResult {
    /// KV slot indices for the matched prefix, in token order.
    pub slots: Vec<i32>,
    /// Last node touched by the walk — caller can pass it into a
    /// future `inc_lock_ref` once that's ported.
    pub last_node: NodeId,
    /// Length of the matched prefix in tokens (== `slots.len()`).
    pub prefix_len: usize,
}

#[derive(Debug)]
pub struct RadixCache {
    nodes: Vec<Node>,
    /// Total tokens currently cached (== sum of all node `value.len`).
    /// Maintained incrementally to keep `evictable_size` O(1).
    total_tokens: i64,
    /// Subset of `total_tokens` that lives under a node with `lock_ref > 0`.
    /// Mirrors Python's `protected_size_`.
    protected_tokens: i64,
}

impl Default for RadixCache {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixCache {
    pub fn new() -> Self {
        Self {
            nodes: vec![Node::root()],
            total_tokens: 0,
            protected_tokens: 0,
        }
    }

    /// Tokens currently safe to evict — i.e. cached but not pinned by a
    /// running request.  Mirrors Python's `evictable_size_`.
    pub fn evictable_size(&self) -> i64 {
        self.total_tokens - self.protected_tokens
    }

    /// Tokens pinned by `inc_lock_ref` from at least one running req.
    /// Mirrors Python's `protected_size_`.
    pub fn protected_size(&self) -> i64 {
        self.protected_tokens
    }

    /// Total tokens held (locked + unlocked) — useful for diagnostics.
    pub fn total_size(&self) -> i64 {
        self.total_tokens
    }

    /// Pin the chain from `node` up to root.  Every node along the path
    /// that transitions from `lock_ref == 0` to `lock_ref == 1` moves
    /// its tokens from evictable to protected.
    ///
    /// Mirrors `inc_lock_ref(node)` in `radix_cache.py`.
    pub fn inc_lock_ref(&mut self, node: NodeId) {
        let mut cur = node;
        while cur != ROOT {
            if self.nodes[cur as usize].lock_ref == 0 {
                self.protected_tokens += self.nodes[cur as usize].cached_tokens;
            }
            self.nodes[cur as usize].lock_ref += 1;
            cur = self.nodes[cur as usize]
                .parent
                .expect("only root has no parent");
        }
    }

    /// Release the lock the matching `inc_lock_ref` took.
    ///
    /// Mirrors `dec_lock_ref(node)` in `radix_cache.py`.
    pub fn dec_lock_ref(&mut self, node: NodeId) {
        let mut cur = node;
        while cur != ROOT {
            debug_assert!(
                self.nodes[cur as usize].lock_ref > 0,
                "dec_lock_ref without matching inc_lock_ref"
            );
            self.nodes[cur as usize].lock_ref =
                self.nodes[cur as usize].lock_ref.saturating_sub(1);
            if self.nodes[cur as usize].lock_ref == 0 {
                self.protected_tokens -= self.nodes[cur as usize].cached_tokens;
            }
            cur = self.nodes[cur as usize]
                .parent
                .expect("only root has no parent");
        }
    }

    /// Walk the tree following `tokens`, returning the longest matching
    /// prefix's KV slots.  Mirrors `match_prefix` in radix_cache.py
    /// (page_size == 1 branch).
    pub fn match_prefix(&self, tokens: &[i32]) -> MatchResult {
        let mut slots = Vec::new();
        let mut node = ROOT;
        let mut pos = 0;
        while pos < tokens.len() {
            let first = tokens[pos];
            let Some(&child_id) = self.nodes[node as usize].children.get(&first) else {
                break;
            };
            let child = &self.nodes[child_id as usize];
            // Find longest shared prefix between tokens[pos..] and child.key.
            let mut shared = 0;
            while shared < child.key.len()
                && pos + shared < tokens.len()
                && child.key[shared] == tokens[pos + shared]
            {
                shared += 1;
            }
            slots.extend_from_slice(&child.value[..shared]);
            pos += shared;
            if shared < child.key.len() {
                // Tokens diverge inside this edge — partial match.
                node = child_id;
                break;
            }
            node = child_id;
        }
        MatchResult {
            prefix_len: slots.len(),
            slots,
            last_node: node,
        }
    }

    /// Graft `tokens` + matching `slots` into the tree.  Returns the
    /// list of KV slot indices that were duplicates of the existing
    /// cache (the caller should free those — they're now owned by the
    /// cache via the older nodes).
    ///
    /// Mirrors `insert(token_ids, kv_indices)` in radix_cache.py.
    pub fn insert(&mut self, tokens: &[i32], slots: &[i32]) -> Vec<i32> {
        debug_assert_eq!(tokens.len(), slots.len(), "1 slot per token");
        if tokens.is_empty() {
            return Vec::new();
        }

        let mut duplicates = Vec::new();
        let mut parent = ROOT;
        let mut pos = 0;
        while pos < tokens.len() {
            let first = tokens[pos];
            let child_id = self.nodes[parent as usize].children.get(&first).copied();
            match child_id {
                Some(child_id) => {
                    let child_key_len = self.nodes[child_id as usize].key.len();
                    let mut shared = 0;
                    while shared < child_key_len
                        && pos + shared < tokens.len()
                        && self.nodes[child_id as usize].key[shared] == tokens[pos + shared]
                    {
                        shared += 1;
                    }
                    // The slots for the shared prefix are already in
                    // the tree; the incoming ones are duplicates.
                    duplicates.extend_from_slice(&slots[pos..pos + shared]);
                    if shared == child_key_len {
                        // Consumed this edge entirely; descend.
                        parent = child_id;
                        pos += shared;
                        continue;
                    }
                    // Split the edge at `shared`.
                    let split_id = self.split_edge(child_id, shared);
                    parent = split_id;
                    pos += shared;
                }
                None => {
                    // No matching child — graft a fresh leaf with the
                    // remainder.
                    let key = tokens[pos..].to_vec();
                    let value = slots[pos..].to_vec();
                    self.add_leaf(parent, first, key, value);
                    return duplicates;
                }
            }
        }
        duplicates
    }

    fn add_leaf(&mut self, parent: NodeId, key0: i32, key: Vec<i32>, value: Vec<i32>) -> NodeId {
        let leaf_id = self.nodes.len() as NodeId;
        let cached = value.len() as i64;
        self.nodes.push(Node::leaf(parent, key, value));
        self.nodes[parent as usize].children.insert(key0, leaf_id);
        self.total_tokens += cached;
        leaf_id
    }

    /// Split `node`'s edge at column `at`.  The new mid node takes the
    /// prefix; the existing node keeps the suffix.  Returns the mid
    /// node's id.
    fn split_edge(&mut self, node: NodeId, at: usize) -> NodeId {
        // SAFETY: split is purely re-pointing; no concurrent mutation.
        let (mid_key, mid_value, suffix_key, suffix_value, parent, suffix_first) = {
            let n = &self.nodes[node as usize];
            let mid_key = n.key[..at].to_vec();
            let mid_value = n.value[..at].to_vec();
            let suffix_key = n.key[at..].to_vec();
            let suffix_value = n.value[at..].to_vec();
            let suffix_first = suffix_key[0];
            (
                mid_key,
                mid_value,
                suffix_key,
                suffix_value,
                n.parent.expect("non-root nodes have parents"),
                suffix_first,
            )
        };
        // Promote `node` to be the suffix.
        self.nodes[node as usize].key = suffix_key;
        self.nodes[node as usize].value = suffix_value;
        self.nodes[node as usize].cached_tokens =
            self.nodes[node as usize].value.len() as i64;

        // Build the mid node.  Inherit `lock_ref` from the suffix so
        // any req that locked `node` before the split keeps the same
        // path-to-root locked after.  See Python `_split_node`.
        let mid_id = self.nodes.len() as NodeId;
        let mid_cached = mid_value.len() as i64;
        let mid_first_token = mid_key[0];
        let inherited_lock_ref = self.nodes[node as usize].lock_ref;
        let mut mid = Node::leaf(parent, mid_key, mid_value);
        mid.cached_tokens = mid_cached;
        mid.lock_ref = inherited_lock_ref;
        mid.children.insert(suffix_first, node);
        self.nodes.push(mid);

        // The cached-token range that "belongs" to a locked node moved
        // when we split: if the original node was locked (and thus its
        // tokens were in `protected_tokens`), they still are — the mid
        // node owns part of them now.  No net change in either counter
        // because mid + suffix together still cover the same span.

        // Re-parent `node` under `mid` and rewire `parent → mid`.
        self.nodes[node as usize].parent = Some(mid_id);
        let parent_node = &mut self.nodes[parent as usize];
        parent_node.children.insert(mid_first_token, mid_id);

        mid_id
    }

    /// Evict approximately `num_tokens` worth of leaf KV from
    /// **unlocked** leaves.  Returns the freed slot indices the caller
    /// can hand back to `CpuPageTracker::free`.
    ///
    /// Today this walks leaves in arena order; a real port would pick
    /// the least-recently-used leaf (see Python's `evictable_heap`).
    pub fn evict(&mut self, num_tokens: i64) -> Vec<i32> {
        let mut freed = Vec::new();
        let mut remaining = num_tokens;
        let mut idx = self.nodes.len();
        while remaining > 0 && idx > 0 {
            idx -= 1;
            if idx == ROOT as usize {
                break;
            }
            if !self.nodes[idx].is_leaf() {
                continue;
            }
            // Skip locked leaves — a running req still owns this prefix.
            if self.nodes[idx].lock_ref > 0 {
                continue;
            }
            let leaf_cached = self.nodes[idx].cached_tokens;
            if leaf_cached == 0 {
                continue;
            }
            // Detach the leaf.
            let parent = self
                .nodes[idx]
                .parent
                .expect("leaf has parent (root cannot be a leaf with tokens)");
            let key0 = self.nodes[idx].key[0];
            freed.extend_from_slice(&self.nodes[idx].value);
            self.total_tokens -= leaf_cached;
            remaining -= leaf_cached;

            // Remove from parent's children map; the arena entry stays
            // (NodeIds remain stable) but becomes orphaned.  A
            // production port should compact.
            self.nodes[idx].value.clear();
            self.nodes[idx].key.clear();
            self.nodes[idx].cached_tokens = 0;
            self.nodes[parent as usize].children.remove(&key0);
        }
        freed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_then_insert_then_match() {
        let mut c = RadixCache::new();
        // Empty cache — match returns nothing.
        let m = c.match_prefix(&[1, 2, 3]);
        assert_eq!(m.prefix_len, 0);
        assert!(m.slots.is_empty());

        // Insert sequence [1,2,3] with slots [10,20,30].
        let dup = c.insert(&[1, 2, 3], &[10, 20, 30]);
        assert!(dup.is_empty());
        assert_eq!(c.evictable_size(), 3);

        // Full prefix match.
        let m = c.match_prefix(&[1, 2, 3]);
        assert_eq!(m.slots, vec![10, 20, 30]);

        // Partial match.
        let m = c.match_prefix(&[1, 2]);
        assert_eq!(m.slots, vec![10, 20]);

        // Divergence past the shared prefix.
        let m = c.match_prefix(&[1, 2, 4]);
        assert_eq!(m.slots, vec![10, 20]);
    }

    #[test]
    fn insert_with_shared_prefix_splits_edge() {
        let mut c = RadixCache::new();
        c.insert(&[1, 2, 3, 4], &[10, 20, 30, 40]);
        // Insert [1,2,5,6] — shared prefix [1,2] forces a split.
        let dup = c.insert(&[1, 2, 5, 6], &[10, 20, 50, 60]);
        // The first 2 slots are duplicates of the existing edge.
        assert_eq!(dup, vec![10, 20]);

        let m = c.match_prefix(&[1, 2, 3]);
        assert_eq!(m.slots, vec![10, 20, 30]);
        let m = c.match_prefix(&[1, 2, 5]);
        assert_eq!(m.slots, vec![10, 20, 50]);
    }

    #[test]
    fn evict_frees_some_leaves() {
        let mut c = RadixCache::new();
        c.insert(&[1, 2, 3], &[10, 20, 30]);
        c.insert(&[4, 5], &[40, 50]);
        assert_eq!(c.evictable_size(), 5);
        let freed = c.evict(5);
        assert_eq!(freed.len(), 5);
        // All of the above are in `freed`; the cache is empty.
        assert_eq!(c.evictable_size(), 0);
    }

    #[test]
    fn inc_lock_ref_moves_tokens_to_protected() {
        let mut c = RadixCache::new();
        c.insert(&[1, 2, 3], &[10, 20, 30]);
        let m = c.match_prefix(&[1, 2, 3]);
        assert_eq!(c.evictable_size(), 3);
        assert_eq!(c.protected_size(), 0);

        c.inc_lock_ref(m.last_node);
        assert_eq!(c.evictable_size(), 0);
        assert_eq!(c.protected_size(), 3);

        c.dec_lock_ref(m.last_node);
        assert_eq!(c.evictable_size(), 3);
        assert_eq!(c.protected_size(), 0);
    }

    #[test]
    fn evict_skips_locked_leaves() {
        let mut c = RadixCache::new();
        c.insert(&[1, 2, 3], &[10, 20, 30]);
        c.insert(&[4, 5, 6], &[40, 50, 60]);

        // Lock the first sequence.
        let m = c.match_prefix(&[1, 2, 3]);
        c.inc_lock_ref(m.last_node);

        // Ask for plenty of eviction — only the unlocked leaf should
        // surrender its slots.
        let freed = c.evict(100);
        assert_eq!(freed.len(), 3);
        assert!(freed.iter().all(|&s| s == 40 || s == 50 || s == 60));

        // The locked prefix is still fully resident.
        let m = c.match_prefix(&[1, 2, 3]);
        assert_eq!(m.slots, vec![10, 20, 30]);
        assert_eq!(c.protected_size(), 3);
    }

    #[test]
    fn split_inherits_lock_ref() {
        let mut c = RadixCache::new();
        c.insert(&[1, 2, 3, 4], &[10, 20, 30, 40]);
        // Lock the full sequence first.
        let m = c.match_prefix(&[1, 2, 3, 4]);
        c.inc_lock_ref(m.last_node);
        assert_eq!(c.protected_size(), 4);

        // Cause a split via a divergent insert.
        c.insert(&[1, 2, 5, 6], &[10, 20, 50, 60]);

        // Lock counters should add up correctly — the original 4
        // tokens are still protected (the new branch isn't).
        assert_eq!(c.protected_size(), 4);

        // Dec lock and the protected size drops to zero.
        let m = c.match_prefix(&[1, 2, 3, 4]);
        c.dec_lock_ref(m.last_node);
        assert_eq!(c.protected_size(), 0);
        assert_eq!(c.evictable_size(), 6);
    }
}
