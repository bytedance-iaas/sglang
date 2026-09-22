"""CPU lifecycle coverage for decode-radix plus EAGLE draft KV.

The target and draft models share one numerical slot namespace (the target
allocator and ReqToTokenPool) but own separate physical KV tensors.  These
tests exercise that contract with the real paged allocator and radix tree;
they intentionally do not duplicate the exhaustive lock-ref scenarios in
``test_decode_radix_lock_ref.py``.
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


PAGE_SIZE = 4
NUM_PAGES = 16


class _Req:
    def __init__(self, raw_tokens, req_pool_idx, prefix_len, last_node):
        self.full_untruncated_fill_ids = array("q", raw_tokens)
        self.origin_input_ids = array("q", raw_tokens)
        self.output_ids = array("q")
        self.extend_range = Range(0, len(raw_tokens))
        self.req_pool_idx = req_pool_idx
        self.cache_protected_len = prefix_len
        self.last_node = last_node
        self.extra_key = None
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.priority = 0
        self.kv_committed_len = len(raw_tokens)
        self.kv = SimpleNamespace(kv_allocated_len=len(raw_tokens))

    def get_fill_ids(self):
        return self.full_untruncated_fill_ids[: self.extend_range.end]


def _make_shared_namespace():
    allocator = PagedTokenToKVPoolAllocator(
        size=NUM_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=4,
        max_context_len=64,
        device="cpu",
        enable_memory_saver=False,
    )
    tree = RadixCache.create_simulated(
        mock_allocator=allocator,
        page_size=PAGE_SIZE,
    )
    tree.req_to_token_pool = req_pool
    tree.is_eagle = True

    # Same numerical IDs index distinct target and draft allocations.
    physical_slots = allocator.size + PAGE_SIZE
    target_kv = torch.full((physical_slots,), -1, dtype=torch.int64)
    draft_kv = torch.full((physical_slots,), -1, dtype=torch.int64)
    return allocator, req_pool, tree, target_kv, draft_kv


def _page_ids(indices):
    return set((indices // PAGE_SIZE).tolist())


class TestDecodeRadixSpeculativeLifecycle(unittest.TestCase):
    def test_shared_prefix_isolated_suffix_and_abort_then_full_hit(self):
        allocator, req_pool, tree, target_kv, draft_kv = _make_shared_namespace()

        # Nine raw EAGLE tokens are eight bigram KV entries: exactly two pages.
        prefix_tokens = list(range(100, 109))
        prefix_indices = allocator.alloc(2 * PAGE_SIZE)
        target_kv[prefix_indices] = 1_000 + prefix_indices
        draft_kv[prefix_indices] = 2_000 + prefix_indices
        tree.insert(
            InsertParams(
                key=RadixKey(array("q", prefix_tokens)),
                value=prefix_indices,
            )
        )

        matched = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", prefix_tokens)))
        )
        self.assertEqual(len(matched.device_indices), 2 * PAGE_SIZE)
        torch.testing.assert_close(matched.device_indices, prefix_indices)

        # Two requests lock the same prefix but allocate disjoint suffix pages.
        tree.inc_lock_ref(matched.last_device_node)
        tree.inc_lock_ref(matched.last_device_node)
        raw_a = prefix_tokens + [201, 202, 203, 204]
        raw_b = prefix_tokens + [211, 212, 213, 214]
        suffix_a = allocator.alloc(2 * PAGE_SIZE)
        suffix_b = allocator.alloc(2 * PAGE_SIZE)
        self.assertTrue(_page_ids(prefix_indices).isdisjoint(_page_ids(suffix_a)))
        self.assertTrue(_page_ids(prefix_indices).isdisjoint(_page_ids(suffix_b)))
        self.assertTrue(_page_ids(suffix_a).isdisjoint(_page_ids(suffix_b)))

        req_a = _Req(raw_a, 1, len(prefix_indices), matched.last_device_node)
        req_b = _Req(raw_b, 2, len(prefix_indices), matched.last_device_node)
        # A 13-token raw key has 12 cacheable bigrams plus one physical tail
        # slot. alloc_extend reserves two pages for those five suffix slots.
        row_a = torch.cat((prefix_indices, suffix_a[:5]))
        row_b = torch.cat((prefix_indices, suffix_b[:5]))
        req_pool.write((1, slice(0, len(row_a))), row_a)
        req_pool.write((2, slice(0, len(row_b))), row_b)
        target_kv[suffix_a[:5]] = 3_000 + suffix_a[:5]
        draft_kv[suffix_a[:5]] = 4_000 + suffix_a[:5]
        target_kv[suffix_b[:5]] = 5_000 + suffix_b[:5]
        draft_kv[suffix_b[:5]] = 6_000 + suffix_b[:5]

        prefix_target_before = target_kv[prefix_indices].clone()
        prefix_draft_before = draft_kv[prefix_indices].clone()
        b_target_before = target_kv[suffix_b[:5]].clone()
        b_draft_before = draft_kv[suffix_b[:5]].clone()

        # Aborting A releases only A's suffix. B's lock still protects the
        # shared radix prefix from eviction, and neither B physical buffer moves.
        tree.cache_finished_req(req_a, is_insert=False, kv_len_to_handle=len(raw_a))
        self.assertTrue(_page_ids(suffix_a).issubset(set(allocator.free_pages.tolist())))
        self.assertTrue(_page_ids(suffix_b).isdisjoint(set(allocator.free_pages.tolist())))
        self.assertGreater(tree.protected_size(), 0)
        torch.testing.assert_close(target_kv[prefix_indices], prefix_target_before)
        torch.testing.assert_close(draft_kv[prefix_indices], prefix_draft_before)
        torch.testing.assert_close(target_kv[suffix_b[:5]], b_target_before)
        torch.testing.assert_close(draft_kv[suffix_b[:5]], b_draft_before)

        # Finishing B inserts its full aligned bigram prefix. The last physical
        # tail page is released, while the first B suffix page becomes radix-owned.
        tree.cache_finished_req(req_b, is_insert=True, kv_len_to_handle=len(raw_b))
        full_hit = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", raw_b)))
        )
        expected_full = torch.cat((prefix_indices, suffix_b[:PAGE_SIZE]))
        torch.testing.assert_close(full_hit.device_indices, expected_full)
        self.assertEqual(len(full_hit.device_indices), 3 * PAGE_SIZE)
        self.assertEqual(tree.protected_size(), 0)

        # A subsequent request may reuse released A/tail pages, never a page
        # still referenced by the radix tree. Overwriting both physical pools at
        # those reused IDs therefore leaves a target/draft full hit intact.
        tree.inc_lock_ref(full_hit.last_device_node)
        self.addCleanup(tree.dec_lock_ref, full_hit.last_device_node)
        reused = allocator.alloc(2 * PAGE_SIZE)
        self.assertTrue(_page_ids(reused).isdisjoint(_page_ids(expected_full)))
        target_kv[reused] = 7_000 + reused
        draft_kv[reused] = 8_000 + reused
        torch.testing.assert_close(target_kv[prefix_indices], prefix_target_before)
        torch.testing.assert_close(draft_kv[prefix_indices], prefix_draft_before)
        torch.testing.assert_close(target_kv[suffix_b[:PAGE_SIZE]], b_target_before[:4])
        torch.testing.assert_close(draft_kv[suffix_b[:PAGE_SIZE]], b_draft_before[:4])

    def test_eagle_unaligned_tail_never_becomes_shared_prefix(self):
        allocator, req_pool, tree, _, _ = _make_shared_namespace()
        prefix_tokens = list(range(9))
        prefix_indices = allocator.alloc(2 * PAGE_SIZE)
        tree.insert(
            InsertParams(
                key=RadixKey(array("q", prefix_tokens)),
                value=prefix_indices,
            )
        )
        matched = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", prefix_tokens)))
        )
        tree.inc_lock_ref(matched.last_device_node)

        # Eleven raw tokens mean ten bigrams. Only the first eight are page
        # aligned, so the three physical suffix slots must be reclaimed rather
        # than exposing a partially initialized target/draft page through radix.
        raw = prefix_tokens + [90, 91]
        tail_page = allocator.alloc(PAGE_SIZE)
        req = _Req(raw, 1, len(prefix_indices), matched.last_device_node)
        row = torch.cat((prefix_indices, tail_page[:3]))
        req_pool.write((1, slice(0, len(row))), row)
        tree.cache_finished_req(req, is_insert=True, kv_len_to_handle=len(raw))

        rematch = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", raw)))
        )
        torch.testing.assert_close(rematch.device_indices, prefix_indices)
        self.assertTrue(_page_ids(tail_page).issubset(set(allocator.free_pages.tolist())))


if __name__ == "__main__":
    unittest.main()
