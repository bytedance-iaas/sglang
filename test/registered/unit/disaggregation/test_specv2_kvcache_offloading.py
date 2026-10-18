"""
Unit tests for _release_finished_req in DecodeKVCacheOffloadManager.

Verifies that over-allocated KV cache slots (from speculative decoding v2)
are correctly freed when a request finishes, preventing GPU memory leaks.

Requires: torch, sglang (run in an environment with sglang installed)
"""

import gc
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from weakref import WeakKeyDictionary as WeakKeyDict

import torch

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _make_mock_req(
    req_pool_idx: int,
    kv_committed_len: int,
    kv_allocated_len: int,
    prefix_indices_len: int = 0,
    rid: int = 0,
    origin_len: int = 0,
):
    """Create a mock Req with the KV cache state needed for testing."""
    req = MagicMock()
    req.rid = rid
    req.origin_input_ids = list(range(origin_len))
    req.kv = ReqKvInfo(
        req_pool_idx=req_pool_idx,
        kv_committed_len=kv_committed_len,
        kv_allocated_len=kv_allocated_len,
    )
    req.prefix_indices = list(range(prefix_indices_len))
    req.effective_kv_committed_len = lambda: req.kv.kv_committed_len
    req.finished.return_value = True
    return req


class _RecordingAllocator(BaseTokenToKVPoolAllocator):
    """Single-pool double. Subclassing the base routes free_full / free_segment /
    free_segments into free(), so a new free API cannot slip past the recorder."""

    def __init__(self, page_size: int):
        super().__init__(
            size=1024,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.freed = []

    def clear(self):
        self.freed = []

    def alloc(self, need_size: int):
        raise NotImplementedError

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.clone())


def _make_manager(pool_size: int, page_size: int = 1):
    """Create a DecodeKVCacheOffloadManager with mock pools for testing."""
    # Build a real req_to_token tensor so indexing works
    req_to_token = torch.arange(pool_size, dtype=torch.int64).unsqueeze(0)

    req_to_token_pool = MagicMock()
    req_to_token_pool.req_to_token = req_to_token

    allocator = _RecordingAllocator(page_size)
    freed_indices = allocator.freed

    tree_cache = MagicMock()
    tree_cache.is_chunk_cache.return_value = True
    tree_cache.protected_size_ = 0
    tree_cache.req_to_token_pool = req_to_token_pool
    tree_cache.token_to_kv_pool_allocator = allocator
    tree_cache.free_kv_row = lambda owner, ranges: BasePrefixCache.free_kv_row(
        tree_cache, owner, ranges
    )

    # Bypass __init__ entirely and set attributes directly
    manager = object.__new__(DecodeKVCacheOffloadManager)
    manager.req_to_token_pool = req_to_token_pool
    manager.token_to_kv_pool_allocator = allocator
    manager.page_size = page_size
    manager.tree_cache = tree_cache
    manager.offloaded_state = WeakKeyDict()
    manager.ongoing_offload = {}
    manager.ongoing_backup = {}
    manager.offload_inflight = WeakKeyDict()

    def release_slot(req):
        req.kv.req_pool_idx = None

    req_to_token_pool.free.side_effect = release_slot

    return manager, freed_indices


class _FinishedEvent:
    def synchronize(self):
        pass


class TestReleaseFinishedReq(unittest.TestCase):
    """Tests for _release_finished_req overallocation cleanup."""

    def test_no_overallocation(self):
        """Without spec v2, kv_committed == kv_allocated; no extra free."""
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=20,  # no overallocation
            origin_len=8,
        )

        manager._release_finished_req(req)

        # Prefill [0:8] and committed [8:20]; no overalloc free.
        self.assertEqual(len(freed), 2)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 8, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[1], torch.arange(8, 20, dtype=torch.int64)))
        manager.req_to_token_pool.free.assert_called_once_with(req)

    def test_with_overallocation(self):
        """With spec v2, overallocated slots [committed:allocated] must be freed."""
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=28,  # 8 over-allocated slots
            origin_len=8,
        )

        manager._release_finished_req(req)

        # Prefill [0:8], committed [8:20], overallocated [20:28].
        self.assertEqual(len(freed), 3)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 8, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[1], torch.arange(8, 20, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[2], torch.arange(20, 28, dtype=torch.int64)))
        manager.req_to_token_pool.free.assert_called_once_with(req)

    def test_overallocation_with_page_alignment(self):
        """With page_size > 1, start of overallocated range is ceil-aligned."""
        page_size = 4
        manager, freed = _make_manager(pool_size=32, page_size=page_size)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=10,  # not page-aligned
            kv_allocated_len=28,
            origin_len=4,
        )

        manager._release_finished_req(req)

        # Prefill [0:4], committed [4:10],
        # overallocated: start_p = ceil_align(10, 4) = 12, end_p = 28 => [12:28]
        self.assertEqual(len(freed), 3)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 4, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[1], torch.arange(4, 10, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[2], torch.arange(12, 28, dtype=torch.int64)))

    def test_overallocation_page_aligned_noop(self):
        """When ceil_align(committed, page_size) >= allocated, no overalloc free."""
        page_size = 4
        manager, freed = _make_manager(pool_size=32, page_size=page_size)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=10,  # ceil_align(10, 4) = 12
            kv_allocated_len=12,  # same as aligned start
            origin_len=4,
        )

        manager._release_finished_req(req)

        # Prefill [0:4] and committed [4:10]; no overalloc since start_p == end_p
        self.assertEqual(len(freed), 2)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 4, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[1], torch.arange(4, 10, dtype=torch.int64)))

    def test_chunk_release_does_not_mutate_cache_protection_accounting(self):
        """Prefix observations are not lock ownership; ChunkCache has no tree."""
        manager, _ = _make_manager(pool_size=32)
        manager.tree_cache.protected_size_ = 10
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=20,
            prefix_indices_len=5,
        )

        manager._release_finished_req(req)

        self.assertEqual(manager.tree_cache.protected_size_, 10)

    def test_radix_release_uses_cache_owner_after_last_ack(self):
        manager, freed = _make_manager(pool_size=32)
        manager.tree_cache.is_chunk_cache.return_value = False
        manager.tree_cache.protected_size_ = 8
        req = _make_mock_req(0, 20, 28, prefix_indices_len=8, origin_len=8)
        req.kv.cache_protected_len = 8
        manager.offloaded_state[req] = OffloadedState(inc_len=4)
        manager.offload_inflight[req] = 1

        def owner_release(request, cache, is_insert):
            self.assertIs(request, req)
            self.assertIs(cache, manager.tree_cache)
            self.assertTrue(is_insert)
            self.assertFalse(manager._has_inflight_offload(request))
            manager.req_to_token_pool.free(request)
            request.kv.mark_kv_released()

        with patch(
            "sglang.srt.mem_cache.common.release_kv_cache", side_effect=owner_release
        ) as release:
            manager._release_finished_req(req)
            release.assert_not_called()
            manager._mark_offload_finished(req)
            manager.finalize_release_on_finish(req)
            release.assert_called_once_with(req, manager.tree_cache, is_insert=True)
            manager.finalize_release_on_finish(req)
            self.assertEqual(release.call_count, 1)
        self.assertEqual(freed, [])
        self.assertEqual(manager.tree_cache.protected_size_, 8)
        self.assertNotIn(req, manager.offloaded_state)

    def test_unfinished_request_cannot_release_even_without_inflight_copy(self):
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(0, 20, 20)
        req.finished.return_value = False
        manager._release_finished_req(req)
        self.assertEqual(freed, [])
        manager.req_to_token_pool.free.assert_not_called()

    def test_chunk_release_is_idempotent(self):
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(0, 20, 28, origin_len=8)
        manager._release_finished_req(req)
        released = [part.clone() for part in freed]
        manager._release_finished_req(req)
        self.assertEqual(len(freed), len(released))
        manager.req_to_token_pool.free.assert_called_once_with(req)

    def test_real_radix_release_retains_prefix_unlocks_and_trims_spec_tail(self):
        # Exercise the real cache owner, not a mocked release_kv_cache. Full
        # model/GPU correctness remains a separate target-runtime gate.
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                manager, freed = _make_manager(32, page_size=page_size)
                cache = RadixCache.create_simulated(
                    mock_allocator=manager.token_to_kv_pool_allocator,
                    page_size=page_size,
                )
                cache.req_to_token_pool = manager.req_to_token_pool
                row = cache.req_to_token_pool.req_to_token
                cache.req_to_token_pool.write = (
                    lambda where, values, row=row: row.__setitem__(where, values)
                )
                cache.is_eagle = True
                prefix = array("q", range(9))
                cache.insert(
                    InsertParams(
                        key=RadixKey(prefix, is_bigram=True), value=torch.arange(8)
                    )
                )
                hit = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(prefix, is_bigram=True))
                )
                cache.inc_lock_ref(hit.last_device_node)
                manager.tree_cache = cache
                req = _make_mock_req(0, 20, 28, prefix_indices_len=8, origin_len=8)
                req.origin_input_ids = array("q", range(8))
                req.output_ids = array("q", range(8, 20))
                req.kv.cache_protected_len = 8
                req.last_node = hit.last_device_node
                req.extra_key = req.cache_salt = None
                req.skip_radix_cache_insert = False
                req.priority = 0
                with patch(
                    "sglang.srt.mem_cache.common.get_spec",
                    return_value=SimpleNamespace(speculative_algorithm="EAGLE"),
                ):
                    manager._release_finished_req(req)
                    manager._release_finished_req(req)
                self.assertEqual(cache.protected_size(), 0)
                self.assertEqual(hit.last_device_node.lock_ref, 0)
                manager.req_to_token_pool.free.assert_called_once_with(req)
                freed_slots = torch.cat(freed).tolist()
                self.assertFalse(set(range(8)).intersection(freed_slots))
                self.assertEqual(len(freed_slots), len(set(freed_slots)))
                self.assertEqual(
                    set(freed_slots), set(range((19 // page_size) * page_size, 28))
                )

    def test_release_finished_req_frees_prefill_and_pops_state(self):
        """
        _release_finished_req frees the prefill-aligned slots in addition to
        the committed range; freeing them mid-decode instead races with
        concurrent admission and cross-pollinates KV reads.
        """
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=20,
            rid="req-prefill-present",
            origin_len=8,
        )
        manager.offloaded_state[req] = OffloadedState(inc_len=4)

        manager._release_finished_req(req)

        # Two frees in order: prefill [0:8] then committed [8:20].
        self.assertEqual(len(freed), 2)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 8, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[1], torch.arange(8, 20, dtype=torch.int64)))
        # State entry is removed at the end of _release_finished_req.
        self.assertNotIn(req, manager.offloaded_state)

    def test_release_finished_req_skips_prefill_free_when_prompt_below_page(self):
        """
        When the prompt is shorter than page_size (no prefill chunk was ever
        offloaded), no prefill-aligned free is emitted.
        """
        manager, freed = _make_manager(pool_size=32, page_size=4)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=10,
            kv_allocated_len=10,
            rid="req-prefill-zero",
            origin_len=3,  # 3 // 4 * 4 == 0
        )

        manager._release_finished_req(req)

        # Only the committed range [0:10] is freed.
        self.assertEqual(len(freed), 1)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 10, dtype=torch.int64)))

    def test_finalize_release_frees_prefill_without_prior_state(self):
        """
        finalize_release_on_finish handles the case where no incremental
        offload ever ran: the prefill-aligned slots must still be freed by
        the consolidated free site in _release_finished_req.
        """
        manager, freed = _make_manager(pool_size=32, page_size=4)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=13,
            kv_allocated_len=13,
            rid="req-finalize-no-state",
            origin_len=12,  # prefill_len = 12 // 4 * 4 = 12
        )

        manager.finalize_release_on_finish(req)

        # _release_finished_req frees prefill [0:12] then committed [12:13].
        self.assertEqual(len(freed), 2)
        expected_prefill = torch.arange(0, 12, dtype=torch.int64)
        expected_committed = torch.arange(12, 13, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected_prefill))
        self.assertTrue(torch.equal(freed[1], expected_committed))
        # No state entry is left behind.
        self.assertNotIn(req, manager.offloaded_state)

    def test_unfinished_offload_ack_does_not_free_incremental_slots(self):
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=1
        )
        req.finished.return_value = False
        manager.offloaded_state[req] = OffloadedState(inc_len=4)
        manager.offload_inflight[req] = 1
        manager.ongoing_offload[7] = (
            req,
            torch.arange(4, 8, dtype=torch.int64),
            [10, 11, 12, 13],
            0.0,
        )
        manager.cache_controller = MagicMock()
        manager.cache_controller.ack_write_queue = [
            HiCacheAck(None, _FinishedEvent(), [7])
        ]
        manager._trigger_backup = MagicMock(return_value="last_hash")

        manager._check_offload_progress(1)

        self.assertEqual(freed, [])
        manager.req_to_token_pool.free.assert_not_called()
        self.assertNotIn(req, manager.offload_inflight)

    def test_offload_kv_cache_tracks_inflight_write_until_ack(self):
        manager, freed = _make_manager(pool_size=32, page_size=4)
        manager.cache_controller = MagicMock()
        manager.cache_controller.get_hash_str = MagicMock(return_value="prefill_hash")
        manager.cache_controller.write = MagicMock(
            return_value=torch.arange(4, 8, dtype=torch.int64)
        )
        manager.decode_host_mem_pool = MagicMock()
        manager.request_counter = 0
        manager.offload_stride = 4

        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=5
        )
        req.origin_input_ids = [0, 1, 2, 3]
        req.output_ids = [4, 5, 6, 7, 8]
        req.finished.return_value = False

        did_offload = manager.offload_kv_cache(req)

        self.assertTrue(did_offload)
        self.assertEqual(manager.offload_inflight[req], 1)
        self.assertEqual(manager.offloaded_state[req].inc_len, 4)
        manager.cache_controller.write.assert_called_once()

        manager.cache_controller.ack_write_queue = [
            HiCacheAck(None, _FinishedEvent(), [1])
        ]
        manager._trigger_backup = MagicMock(return_value="last_hash")

        manager._check_offload_progress(1)

        self.assertEqual(freed, [])
        self.assertNotIn(req, manager.offload_inflight)

    def test_offload_never_copies_uncommitted_speculative_rows(self):
        manager, freed = _make_manager(pool_size=32, page_size=4)
        manager.cache_controller = MagicMock()
        manager.cache_controller.write.return_value = torch.arange(4, 8)
        manager.decode_host_mem_pool = MagicMock()
        manager.request_counter = 0
        manager.offload_stride = 4
        req = _make_mock_req(req_pool_idx=0, kv_committed_len=8, kv_allocated_len=16)
        req.origin_input_ids = [0, 1, 2, 3]
        req.output_ids = list(range(4, 13))
        req.finished.return_value = False

        self.assertTrue(manager.offload_kv_cache(req))
        manager.cache_controller.write.assert_called_once()
        torch.testing.assert_close(
            manager.cache_controller.write.call_args.kwargs["device_indices"],
            torch.arange(4, 8),
        )
        self.assertEqual(manager.ongoing_offload[1][2], [4, 5, 6, 7])
        self.assertEqual(manager.offloaded_state[req].inc_len, 4)
        self.assertEqual(freed, [])

    def test_offload_reduced_commit_boundary_does_not_submit_negative_chunk(self):
        manager, _ = _make_manager(pool_size=32, page_size=4)
        manager.cache_controller = MagicMock()
        manager.decode_host_mem_pool = MagicMock()
        manager.request_counter = 0
        manager.offload_stride = 4
        req = _make_mock_req(req_pool_idx=0, kv_committed_len=12, kv_allocated_len=16)
        req.origin_input_ids = list(range(8))
        req.output_ids = list(range(8, 17))
        manager.offloaded_state[req] = OffloadedState(inc_len=4)
        # Models the intentional strip-thinking bound becoming prompt-only.
        req.effective_kv_committed_len = lambda: len(req.origin_input_ids)

        self.assertFalse(manager.offload_kv_cache(req))
        manager.cache_controller.write.assert_not_called()
        self.assertEqual(manager.offloaded_state[req].inc_len, 4)
        self.assertEqual(manager.ongoing_offload, {})

    def test_offload_released_slot_is_not_indexed(self):
        manager, _ = _make_manager(pool_size=32, page_size=4)
        manager.cache_controller = MagicMock()
        manager.decode_host_mem_pool = MagicMock()
        req = _make_mock_req(req_pool_idx=None, kv_committed_len=8, kv_allocated_len=8)
        req.output_ids = [4, 5]
        manager.req_to_token_pool.req_to_token = MagicMock()

        self.assertFalse(manager.offload_kv_cache(req))
        manager.req_to_token_pool.req_to_token.__getitem__.assert_not_called()
        manager.cache_controller.write.assert_not_called()

    def test_retraction_drain_joins_peer_check_before_refusing(self):
        manager, freed = _make_manager(pool_size=32)
        manager.cache_controller = MagicMock(ack_write_queue=[])
        manager.tp_world_size = 2
        manager.tp_group = object()
        manager.check_offload_progress = MagicMock()
        ops = []

        def peer_pending(tensor, op, group):
            self.assertIs(group, manager.tp_group)
            self.assertEqual(tensor.device.type, "cpu")
            self.assertEqual(tensor.item(), 0)
            ops.append(op)
            tensor.fill_(1)

        with patch.object(torch.distributed, "all_reduce", side_effect=peer_pending):
            with self.assertRaisesRegex(RuntimeError, "refusing to retract"):
                manager.drain_before_retraction()

        manager.check_offload_progress.assert_called_once()
        self.assertEqual(ops, [torch.distributed.ReduceOp.MAX])
        self.assertEqual(freed, [])

    def test_retraction_drain_refuses_missing_ack_but_not_host_backup(self):
        manager, _ = _make_manager(pool_size=32)
        manager.cache_controller = MagicMock(ack_write_queue=[])
        manager.tp_world_size = 1
        manager.check_offload_progress = MagicMock()
        req = _make_mock_req(0, 8, 8)
        manager.offload_inflight[req] = 1
        with self.assertRaisesRegex(RuntimeError, "refusing to retract"):
            manager.drain_before_retraction()

        manager.offload_inflight.clear()
        manager.ongoing_backup[1] = (req.rid, torch.arange(4), 0)
        manager.drain_before_retraction()
        self.assertEqual(manager.check_offload_progress.call_count, 2)

    def test_reused_rid_does_not_share_offload_lifecycle(self):
        manager, _ = _make_manager(pool_size=32, page_size=4)
        manager.cache_controller = MagicMock()
        manager.cache_controller.get_hash_str.return_value = "prefill_hash"
        manager.cache_controller.write.return_value = torch.arange(
            4, 8, dtype=torch.int64
        )
        manager.decode_host_mem_pool = MagicMock()
        manager.request_counter = 0
        manager.offload_stride = 4

        old_req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid="reused"
        )
        new_req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid="reused"
        )
        for req in (old_req, new_req):
            req.origin_input_ids = [0, 1, 2, 3]
            req.output_ids = [4, 5, 6, 7, 8]
            req.finished.return_value = False

        self.assertTrue(manager.offload_kv_cache(old_req))
        old_req.finished.return_value = True

        # A completed request leaves the API before its asynchronous D2H copy
        # necessarily finishes, so a caller can reuse the same rid here.
        self.assertTrue(manager.offload_kv_cache(new_req))
        self.assertIsNot(old_req, new_req)
        self.assertIn(old_req, manager.offloaded_state)
        self.assertIn(new_req, manager.offloaded_state)
        self.assertEqual(manager.offload_inflight[old_req], 1)
        self.assertEqual(manager.offload_inflight[new_req], 1)

        manager.cache_controller.ack_write_queue = [
            HiCacheAck(None, _FinishedEvent(), [1])
        ]
        manager._trigger_backup = MagicMock(return_value="old_last_hash")
        manager._check_offload_progress(1)

        self.assertNotIn(old_req, manager.offloaded_state)
        self.assertNotIn(old_req, manager.offload_inflight)
        self.assertIn(new_req, manager.offloaded_state)
        self.assertEqual(manager.offloaded_state[new_req].inc_len, 4)
        self.assertEqual(manager.offload_inflight[new_req], 1)
        self.assertIn(2, manager.ongoing_offload)

    def test_finalize_release_defers_while_offload_is_in_flight(self):
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=2
        )
        manager.offloaded_state[req] = OffloadedState(inc_len=8)
        manager.offload_inflight[req] = 1

        manager.finalize_release_on_finish(req)

        self.assertEqual(freed, [])
        manager.req_to_token_pool.free.assert_not_called()
        self.assertIn(req, manager.offloaded_state)

    def test_finished_offload_ack_waits_for_other_inflight_writes(self):
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=3
        )
        req.finished.return_value = True
        manager.offloaded_state[req] = OffloadedState(inc_len=8)
        manager.offload_inflight[req] = 2
        manager.ongoing_offload[8] = (
            req,
            torch.arange(4, 8, dtype=torch.int64),
            [10, 11, 12, 13],
            0.0,
        )
        manager.cache_controller = MagicMock()
        manager.cache_controller.ack_write_queue = [
            HiCacheAck(None, _FinishedEvent(), [8])
        ]
        manager._trigger_backup = MagicMock(return_value="last_hash")

        manager._check_offload_progress(1)

        self.assertEqual(freed, [])
        manager.req_to_token_pool.free.assert_not_called()
        self.assertEqual(manager.offload_inflight[req], 1)

    def test_finished_request_releases_all_committed_slots_after_last_offload_ack(
        self,
    ):
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=20,
            rid=4,
            origin_len=4,
        )
        req.finished.return_value = True
        manager.offloaded_state[req] = OffloadedState(inc_len=8)
        manager.offload_inflight[req] = 1
        manager.ongoing_offload[9] = (
            req,
            torch.arange(8, 12, dtype=torch.int64),
            [14, 15, 16, 17],
            0.0,
        )
        manager.cache_controller = MagicMock()
        manager.cache_controller.ack_write_queue = [
            HiCacheAck(None, _FinishedEvent(), [9])
        ]
        manager._trigger_backup = MagicMock(return_value="last_hash")

        manager._check_offload_progress(1)

        self.assertEqual(len(freed), 2)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 4, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[1], torch.arange(4, 20, dtype=torch.int64)))
        manager.req_to_token_pool.free.assert_called_once_with(req)
        self.assertNotIn(req, manager.offloaded_state)
        self.assertNotIn(req, manager.offload_inflight)

    def test_dropped_req_does_not_pin_offload_state(self):
        manager, _ = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=6
        )
        manager.offloaded_state[req] = OffloadedState(inc_len=4)
        manager.offload_inflight[req] = 1

        del req
        gc.collect()

        self.assertEqual(len(manager.offloaded_state), 0)
        self.assertEqual(len(manager.offload_inflight), 0)


if __name__ == "__main__":
    unittest.main()
