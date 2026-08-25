"""fix(c) coverage: HiSparse spec-verify grow path degrades gracefully under
KV-retraction pressure instead of SIGQUIT-ing the scheduler.

Root cause of the C=64 decode crash (batch > ~16): on a KV-full decode retract,
Scheduler.update_running_batch retracts one request, then the very next
target-verify step calls HiSparseCoordinator.prepare_verify_slots_spec_v2 ->
get_draft_device_slots -> _ensure_padded_buffer.  That method allocated the
extra graph-stable draft page from the HiSparse *device* pool
(hisparse_attn_allocator) with a raw alloc() and RAISED RuntimeError when it
returned None -- the exception propagated out of run_batch into the scheduler
event loop, which sends SIGQUIT to PID 1 and restarts both decode pods.

The device pool's schedulable_hisparse_available() (which gates PD admission via
hisparse_direct_admission_capacity and KV-full decode retraction) counts
reclaimable resident device pages as available.  The two other device-pool
allocation sites (admit_request_direct, dynamic-decode grow) already honor that
promise by calling demote_until_hisparse_available() before allocating.  fix(c)
makes _ensure_padded_buffer do the same: reclaim resident pages and retry before
raising, and recompute the grow plan against the post-demotion residency state.

These tests pin that behavior with a minimal coordinator stub:
- alloc succeeds first try            -> no reclaim, slots written.
- alloc None then reclaim frees room  -> retry succeeds, no raise, slots written.
- alloc None and reclaim cannot help  -> clear RuntimeError (not a bare grow msg).
- reclaim demotes the batch's request -> grow plan recomputed, no double-write.
"""

import unittest

import torch

from sglang.srt.managers.hisparse_coordinator import (
    HiSparseCoordinator,
    HiSparseResidencyState,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

DEVICE_BUFFER_SIZE = 4
PAGE_SIZE = 2
PADDED = DEVICE_BUFFER_SIZE + PAGE_SIZE  # 6
MAX_REQ = 8


class _FakeHisparseAllocator:
    """Device pool whose alloc() can be scripted to fail then succeed once the
    test's reclaim hook 'frees' capacity."""

    def __init__(self, available: int):
        self._available = available
        self._next_index = 1000
        self.alloc_calls = 0

    def available_size(self) -> int:
        return self._available

    def alloc(self, need: int):
        self.alloc_calls += 1
        if need > self._available:
            return None
        self._available -= need
        chunk = torch.arange(
            self._next_index, self._next_index + need, dtype=torch.int64
        )
        self._next_index += need
        return chunk


class _FakeKVAllocator:
    def __init__(self, hisparse_attn_allocator):
        self.hisparse_attn_allocator = hisparse_attn_allocator


class TestHiSparseGrowReclaim(CustomTestCase):
    def _make_coordinator(self, available: int) -> HiSparseCoordinator:
        coord = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coord.device_buffer_size = DEVICE_BUFFER_SIZE
        coord.page_size = PAGE_SIZE
        coord.padded_buffer_size = PADDED
        coord._residency_states = {}
        coord.req_device_buffer_size = torch.zeros(MAX_REQ, dtype=torch.int64)
        coord.req_to_device_buffer = torch.zeros(
            (MAX_REQ, PADDED), dtype=torch.int64
        )
        coord.req_device_buffer_tokens = torch.full(
            (1, MAX_REQ, PADDED), -1, dtype=torch.int32
        )
        coord.req_device_buffer_token_locs = torch.full(
            (1, MAX_REQ, PADDED), -1, dtype=torch.int32
        )
        coord._device_buffer_arange_i32 = torch.arange(PADDED, dtype=torch.int32)
        allocator = _FakeHisparseAllocator(available)
        coord.token_to_kv_pool_allocator = _FakeKVAllocator(allocator)
        coord._demote_log = []
        return coord

    def test_alloc_success_first_try_no_reclaim(self):
        coord = self._make_coordinator(available=PADDED)
        allocator = coord.token_to_kv_pool_allocator.hisparse_attn_allocator

        def _no_reclaim(_need):  # must not be called
            raise AssertionError("reclaim called even though alloc succeeded")

        coord.demote_until_hisparse_available = _no_reclaim
        coord._ensure_padded_buffer(torch.tensor([0], dtype=torch.int64))

        self.assertEqual(allocator.alloc_calls, 1)
        self.assertEqual(int(coord.req_device_buffer_size[0]), PADDED)

    def test_reclaim_then_retry_succeeds_without_raise(self):
        coord = self._make_coordinator(available=0)
        allocator = coord.token_to_kv_pool_allocator.hisparse_attn_allocator

        def _reclaim(need):
            # Simulate demote_until_hisparse_available freeing resident pages.
            allocator._available = need
            coord._demote_log.append(need)
            return True

        coord.demote_until_hisparse_available = _reclaim
        coord._ensure_padded_buffer(torch.tensor([0], dtype=torch.int64))

        self.assertEqual(coord._demote_log, [PADDED])
        self.assertGreaterEqual(allocator.alloc_calls, 2)
        self.assertEqual(int(coord.req_device_buffer_size[0]), PADDED)
        # Draft page (columns device_buffer_size..padded) must be materialized.
        self.assertTrue(
            torch.all(
                coord.req_to_device_buffer[0, DEVICE_BUFFER_SIZE:PADDED] > 0
            )
        )

    def test_reclaim_insufficient_raises_clear_error(self):
        coord = self._make_coordinator(available=0)

        def _reclaim(_need):
            # Reclaim runs but frees nothing.
            return False

        coord.demote_until_hisparse_available = _reclaim
        with self.assertRaises(RuntimeError) as ctx:
            coord._ensure_padded_buffer(torch.tensor([0], dtype=torch.int64))
        # New message names reclaim, not the bare legacy "failed to grow" text.
        self.assertIn("after", str(ctx.exception).lower())
        self.assertIn("reclaim", str(ctx.exception).lower())

    def test_grow_plan_recomputed_after_reclaim_demotes_batch_request(self):
        """If reclaim demotes a request that was resident (grow=page) into a
        non-resident device-buffered slot, or frees a slot already at padded,
        the post-reclaim recompute must reflect the new state and not
        double-allocate or write a stale plan."""
        coord = self._make_coordinator(available=0)
        allocator = coord.token_to_kv_pool_allocator.hisparse_attn_allocator
        # req 0 starts resident (needs only one extra page = PAGE_SIZE).
        coord._residency_states[0] = HiSparseResidencyState.RESIDENT

        def _reclaim(need):
            # Reclaim satisfies the request AND flips req 0 out of the batch's
            # grow need by bringing it to full padded capacity (as a demotion
            # to DEVICE_BUFFERED with a full buffer would).
            coord.req_device_buffer_size[0] = PADDED
            coord._residency_states[0] = HiSparseResidencyState.DEVICE_BUFFERED
            allocator._available = need
            return True

        coord.demote_until_hisparse_available = _reclaim
        # Should return cleanly: after recompute total_grow == 0.
        coord._ensure_padded_buffer(torch.tensor([0], dtype=torch.int64))
        self.assertEqual(int(coord.req_device_buffer_size[0]), PADDED)


if __name__ == "__main__":
    unittest.main()
