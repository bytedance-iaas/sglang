import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
    _HiSparsePageOwnership,
    _released_page_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _LogicalPageAllocator:
    def __init__(self, page_size=256, num_pages=4):
        self.page_size = page_size
        self.size = page_size * num_pages
        self._next_page = 0
        self.freed_pages = set()

    def available_size(self):
        return self.size - self._next_page * self.page_size

    def alloc_extend(
        self,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
    ):
        page = self._next_page
        self._next_page += 1
        return torch.arange(
            page * self.page_size,
            page * self.page_size + extend_num_tokens,
            dtype=torch.int64,
        )

    def free(self, indices):
        self.freed_pages.update((indices // self.page_size).tolist())


class _PhysicalPageAllocator:
    def __init__(self, page_size=64, num_pages=8):
        self.page_size = page_size
        self.size = page_size * num_pages
        # Slot/page zero is the allocator sentinel, matching the real allocator.
        self.is_not_in_free_group = True
        self.free_pages = list(range(1, num_pages + 1))
        self.used_pages = set()

    def available_size(self):
        return len(self.free_pages) * self.page_size

    def _take_page(self):
        page = self.free_pages.pop(0)
        self.used_pages.add(page)
        return torch.arange(
            page * self.page_size,
            (page + 1) * self.page_size,
            dtype=torch.int64,
        )

    def alloc(self, need_size):
        assert need_size == self.page_size
        return self._take_page()

    def alloc_extend(self, *args, **kwargs):
        return self._take_page()

    def free(self, indices):
        pages = torch.unique(indices // self.page_size).tolist()
        for page in pages:
            self.used_pages.remove(page)
            self.free_pages.append(page)


class _C4Pool:
    page_size = 64

    def __init__(self, mapping):
        self.full_to_hisparse_device_index_mapping = mapping

    @staticmethod
    def translate_loc_from_full_to_compressed(full_indices):
        return full_indices[(full_indices + 1) % 4 == 0] // 4

    def _translate_loc_to_hisparse_device(self, compressed_indices):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]


class TestDeepSeekV4HiSparseAllocator(CustomTestCase):
    def test_dsv4_free_group_owns_req_to_token_views(self):
        """Deferred finish frees must survive req-row reuse under overlap."""
        logical_allocator = MagicMock()
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator.logical_attn_allocator = logical_allocator
        allocator.is_not_in_free_group = True
        allocator.free_group = []

        req_to_token = torch.arange(256, dtype=torch.int64).reshape(2, 128)
        committed = req_to_token[0, :64]
        speculative_tail = req_to_token[0, 64:96]
        expected = torch.cat((committed.clone(), speculative_tail.clone()))

        allocator.free_group_begin()
        allocator.free(committed)
        allocator.free_segment(speculative_tail, start_pos=64)

        # Overlap scheduling can recycle and rewrite the request row before the
        # outer DSV4 allocator drains its batched free transaction.
        req_to_token[0].copy_(req_to_token[1])
        allocator.free_group_end()

        logical_allocator.free.assert_called_once()
        torch.testing.assert_close(logical_allocator.free.call_args.args[0], expected)

    def test_released_page_ids_includes_pd_staging_pages(self):
        allocator = SimpleNamespace(
            free_pages=torch.tensor([1, 2], dtype=torch.int64),
            release_pages=torch.tensor([7, 8], dtype=torch.int64),
        )

        self.assertEqual(
            _released_page_ids(allocator, device=torch.device("cpu")).tolist(),
            [1, 2, 7, 8],
        )

    def test_device_buffer_discards_mapping_to_pd_staged_page(self):
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator.page_size = 4
        allocator.full_to_hisparse_device_index_mapping = torch.tensor(
            [0, 5, 6, 7], dtype=torch.int64
        )
        allocator.hisparse_attn_allocator = SimpleNamespace(
            free_pages=torch.tensor([2], dtype=torch.int64),
            release_pages=torch.tensor([1], dtype=torch.int64),
            alloc=MagicMock(
                return_value=torch.tensor([12, 13, 14, 15], dtype=torch.int64)
            ),
        )

        result = allocator.alloc_device_buffer(torch.tensor([1, 2, 3]), 4)

        self.assertEqual(result.tolist(), [12, 13, 14, 15])
        allocator.hisparse_attn_allocator.alloc.assert_called_once_with(4)
        self.assertEqual(
            allocator.full_to_hisparse_device_index_mapping.tolist(), [0, 0, 0, 0]
        )

    def test_page_ownership_clears_all_owners_before_stable_page_free(self):
        mapping = torch.zeros(8, dtype=torch.int64)
        mapping[torch.tensor([1, 2, 3])] = torch.tensor([9, 5, 11])
        buffer_owner = torch.tensor([4, 7], dtype=torch.int64)
        child_allocator = MagicMock(is_not_in_free_group=True)

        def verify_owner_clear(free_indices):
            self.assertEqual(mapping[[1, 2, 3]].tolist(), [0, 0, 0])
            self.assertEqual(buffer_owner.tolist(), [0, 0])
            # First-seen page order is page 2, then page 1.
            self.assertEqual(
                free_indices.tolist(), list(range(8, 12)) + list(range(4, 8))
            )

        child_allocator.free.side_effect = verify_owner_clear
        ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=child_allocator, page_size=4
        )
        ownership.claim(buffer_owner)

        ownership.release(
            mapping_indices=torch.tensor([1, 2, 3]),
            extra_owned_coordinates=buffer_owner,
            clear_extra_owner=buffer_owner.zero_,
        )

        child_allocator.free.assert_called_once()

    def test_page_ownership_rejects_child_allocator_free_group(self):
        mapping = torch.tensor([0, 4], dtype=torch.int64)
        child_allocator = MagicMock(is_not_in_free_group=True)
        ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=child_allocator, page_size=4
        )
        child_allocator.is_not_in_free_group = False

        with self.assertRaises(AssertionError):
            ownership.release(mapping_indices=torch.tensor([1]))

        self.assertEqual(mapping.tolist(), [0, 4])
        child_allocator.free.assert_not_called()

    def test_page_ownership_releases_shared_page_after_last_mapping_owner(self):
        """Separate release calls must not return a still-referenced C4 page."""
        page_size = 4
        mapping = torch.zeros(8, dtype=torch.int64)
        # Two independently retired logical ranges refer to different slots in
        # the same physical page. This happens when page-aligned allocation
        # outlives the request-visible compressed range.
        mapping[1] = 5
        mapping[6] = 7
        child_allocator = MagicMock(is_not_in_free_group=True)
        ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=child_allocator, page_size=page_size
        )

        ownership.release(mapping_indices=torch.tensor([1]))

        self.assertEqual(mapping[1].item(), 0)
        self.assertEqual(mapping[6].item(), 7)
        child_allocator.free.assert_not_called()

        ownership.release(mapping_indices=torch.tensor([6]))

        child_allocator.free.assert_called_once()
        self.assertEqual(child_allocator.free.call_args.args[0].tolist(), [4, 5, 6, 7])

    def test_page_ownership_pins_side_buffer_across_logical_free_transactions(self):
        """A detached logical alias must not make a live buffer page reusable."""
        page_size = 4
        mapping = torch.zeros(8, dtype=torch.int64)
        mapping[1] = 5
        side_buffer = torch.tensor([4, 5, 6, 7], dtype=torch.int64)
        child_allocator = _PhysicalPageAllocator(page_size=page_size, num_pages=2)
        allocated = child_allocator.alloc(page_size)
        self.assertEqual(allocated.tolist(), side_buffer.tolist())
        ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=child_allocator, page_size=page_size
        )
        ownership.claim(side_buffer)

        # A cache transaction can retire an accepted/rejected logical alias
        # before request_finished drops the coordinator's side-buffer owner.
        ownership.release(mapping_indices=torch.tensor([1]))
        self.assertEqual(mapping[1].item(), 0)
        self.assertEqual(child_allocator.used_pages, {1})
        self.assertEqual(child_allocator.free_pages, [2])

        ownership.release(
            mapping_indices=torch.empty(0, dtype=torch.int64),
            extra_owned_coordinates=side_buffer,
        )
        self.assertEqual(child_allocator.used_pages, set())
        self.assertEqual(child_allocator.free_pages.count(1), 1)

    def test_page_ownership_retires_stale_aliases_of_released_side_page(self):
        """A side-buffer release owns and retires every alias of its full page."""
        page_size = 4
        mapping = torch.zeros(12, dtype=torch.int64)
        side_buffer = torch.tensor([4, 5, 6, 7], dtype=torch.int64)
        child_allocator = _PhysicalPageAllocator(page_size=page_size, num_pages=2)
        allocated = child_allocator.alloc(page_size)
        self.assertEqual(allocated.tolist(), side_buffer.tolist())
        ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=child_allocator, page_size=page_size
        )
        ownership.claim(side_buffer)

        # request_finished can see only the committed req_to_token range.  An
        # EAGLE verify tail may remain outside that range while still aliasing
        # the coordinator-owned full page.  The explicit side-buffer owner is
        # the canonical owner of the page, so dropping it must also retire
        # those stale aliases rather than leaking the physical page forever.
        mapping[1] = 4
        mapping[9] = 7
        mapping[10] = 9
        ownership.release(
            mapping_indices=torch.tensor([1]),
            extra_owned_coordinates=side_buffer,
        )

        self.assertEqual(mapping[[1, 9]].tolist(), [0, 0])
        self.assertEqual(mapping[10].item(), 9)
        self.assertEqual(child_allocator.used_pages, set())
        self.assertEqual(child_allocator.free_pages.count(1), 1)

    def test_consecutive_natural_finish_restores_physical_capacity(self):
        """Reusing one request slot must not leak its side-buffer pages."""
        page_size = 4
        mapping = torch.zeros(16, dtype=torch.int64)
        physical = _PhysicalPageAllocator(page_size=page_size, num_pages=3)
        ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=physical, page_size=page_size
        )
        initial_available = physical.available_size()
        logical_locs = torch.tensor([1, 5, 9], dtype=torch.int64)

        for _ in range(8):
            side_page = physical.alloc(page_size)
            side_page_id = int(side_page[0] // page_size)
            ownership.claim(side_page)
            # Natural EAGLE completion can leave request-visible and verify-tail
            # aliases to the coordinator-owned side page.
            mapping[logical_locs] = side_page[: logical_locs.numel()]
            mapping[13] = side_page[-1]

            before = ownership.debug_snapshot(side_page)
            self.assertEqual(before["request_claimed_pages"], [side_page_id])
            self.assertEqual(before["request_mapping_pages"], [side_page_id])

            ownership.release(
                mapping_indices=logical_locs,
                extra_owned_coordinates=side_page,
            )

            after = ownership.debug_snapshot(side_page)
            self.assertEqual(after["request_claimed_pages"], [])
            self.assertEqual(after["request_mapping_pages"], [])
            self.assertEqual(after["extra_owner_pages"], 0)
            self.assertEqual(after["mapping_slots"], 0)
            self.assertEqual(after["available"], initial_available)
            self.assertEqual(physical.free_pages.count(side_page_id), 1)

    def test_generic_finish_order_survives_earlier_logical_alias_retirement(self):
        """Replay the production cache/coordinator/final-cache ownership order."""
        page_size = 64
        logical = _LogicalPageAllocator(page_size=page_size, num_pages=2)
        physical = _PhysicalPageAllocator(page_size=page_size, num_pages=2)
        mapping = torch.zeros(logical.size + page_size + 1, dtype=torch.int64)
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator.page_size = page_size
        allocator.logical_attn_allocator = logical
        allocator.hisparse_attn_allocator = physical
        allocator.full_to_hisparse_device_index_mapping = mapping
        allocator.is_not_in_free_group = True
        allocator.free_group = []
        allocator._page_ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=physical, page_size=page_size
        )

        side_buffer = physical.alloc(page_size)
        logical_locs = torch.arange(page_size, dtype=torch.int64)
        mapping[logical_locs[:7]] = side_buffer[:7]
        allocator.claim_hisparse_ownership(side_buffer)

        # A speculative/cache transaction retires the temporary aliases while
        # req_to_device_buffer still owns the whole page.  Before this fix the
        # page became reusable here and request finish returned it again.
        allocator.free_hisparse(logical_locs[:7])
        self.assertEqual(physical.used_pages, {1})
        self.assertEqual(physical.available_size(), page_size)

        # Production finish first drops the coordinator owner, then the chunk
        # cache frees the request's logical KV indices.
        allocator.release_hisparse_ownership(
            mapping_indices=logical_locs,
            extra_owned_coordinates=side_buffer,
        )
        allocator.free(logical_locs)
        self.assertEqual(physical.used_pages, set())
        self.assertEqual(physical.available_size(), physical.size)
        self.assertEqual(physical.free_pages.count(1), 1)

    def test_generic_resident_finish_returns_spec_page_once(self):
        """Coordinator finish detaches aliases before returning its side page."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        page_size = 64
        logical = _LogicalPageAllocator(page_size=page_size, num_pages=2)
        physical = _PhysicalPageAllocator(page_size=page_size, num_pages=2)
        mapping = torch.zeros(logical.size + page_size + 1, dtype=torch.int64)
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator.page_size = page_size
        allocator.logical_attn_allocator = logical
        allocator.hisparse_attn_allocator = physical
        allocator.full_to_hisparse_device_index_mapping = mapping
        allocator.is_not_in_free_group = True
        allocator.free_group = []
        allocator._page_ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=physical, page_size=page_size
        )

        side_page = physical.alloc(page_size)
        allocator.claim_hisparse_ownership(side_page)
        logical_locs = torch.arange(1, 8, dtype=torch.int64)
        mapping[logical_locs] = side_page[: logical_locs.numel()]

        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.page_size = page_size
        coordinator.device_buffer_size = 4
        coordinator.padded_buffer_size = coordinator.device_buffer_size + page_size
        coordinator.req_to_device_buffer = torch.zeros(
            (1, coordinator.padded_buffer_size), dtype=torch.int64
        )
        coordinator.req_to_device_buffer[0, coordinator.device_buffer_size :] = (
            side_page
        )
        coordinator.req_device_buffer_size = torch.tensor(
            [coordinator.padded_buffer_size], dtype=torch.int64
        )
        coordinator.req_device_buffer_tokens = torch.zeros(
            (1, 1, coordinator.padded_buffer_size), dtype=torch.int32
        )
        coordinator.req_device_buffer_token_locs = torch.zeros_like(
            coordinator.req_device_buffer_tokens
        )
        coordinator.lru_slots = torch.zeros(
            (1, 1, coordinator.padded_buffer_size), dtype=torch.int16
        )
        coordinator._lru_init = torch.zeros(
            coordinator.padded_buffer_size, dtype=torch.int16
        )
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=logical_locs.unsqueeze(0)
        )
        coordinator.mem_pool_device = SimpleNamespace(
            translate_loc_from_full_to_compressed=lambda locs: locs,
            full_to_hisparse_device_index_mapping=mapping,
        )
        coordinator.token_to_kv_pool_allocator = allocator
        req = SimpleNamespace(
            rid="resident-finish",
            req_pool_idx=0,
            kv=SimpleNamespace(kv_allocated_len=logical_locs.numel()),
        )

        coordinator._free_resident_spec_page(req, free_physical=True)
        self.assertTrue(torch.all(mapping[logical_locs] == 0))
        self.assertEqual(physical.free_pages.count(1), 1)
        self.assertEqual(physical.available_size(), physical.size)

        # release_kv_cache follows coordinator cleanup. It now owns only the
        # logical page and cannot return the physical page a second time.
        allocator.free(logical_locs)
        self.assertEqual(physical.free_pages.count(1), 1)
        self.assertEqual(physical.available_size(), physical.size)

    def test_page_ownership_does_not_release_page_already_staged_for_reuse(self):
        """PD release_pages is part of the physical allocator's free set."""
        mapping = torch.tensor([0, 5], dtype=torch.int64)
        child_allocator = MagicMock(is_not_in_free_group=True)
        child_allocator.free_pages = torch.tensor([2], dtype=torch.int64)
        child_allocator.release_pages = torch.tensor([1], dtype=torch.int64)
        ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=child_allocator, page_size=4
        )

        ownership.release(mapping_indices=torch.tensor([1]))

        self.assertEqual(mapping.tolist(), [0, 0])
        child_allocator.free.assert_not_called()

    def test_page_ownership_never_returns_positive_sentinel_coordinates(self):
        """A non-page-aligned PD tail may map into positive slots of page zero."""
        page_size = 4
        mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        child_allocator = MagicMock(is_not_in_free_group=True)
        child_allocator.free_pages = torch.tensor([1, 2], dtype=torch.int64)
        child_allocator.release_pages = torch.empty(0, dtype=torch.int64)
        ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=child_allocator, page_size=page_size
        )

        ownership.release(mapping_indices=torch.tensor([1, 2, 3]))

        self.assertEqual(mapping.tolist(), [0, 0, 0, 0])
        child_allocator.free.assert_not_called()

    def test_non_page_aligned_pd_finish_keeps_physical_capacity_bounded(self):
        """Coordinator cleanup followed by cache free must not credit page zero."""
        page_size = 4
        logical = _LogicalPageAllocator(page_size=page_size, num_pages=2)
        physical = _PhysicalPageAllocator(page_size=page_size, num_pages=2)
        mapping = torch.tensor([0, 1, 2, 3, 0, 0, 0, 0], dtype=torch.int64)
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator.page_size = page_size
        allocator.logical_attn_allocator = logical
        allocator.hisparse_attn_allocator = physical
        allocator.full_to_hisparse_device_index_mapping = mapping
        allocator.is_not_in_free_group = True
        allocator.free_group = []
        allocator._page_ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=physical, page_size=page_size
        )
        logical_locs = torch.tensor([1, 2, 3], dtype=torch.int64)

        # This is the runtime finish order: the coordinator first retires its
        # request-visible aliases, then ChunkCache releases the logical KV.
        allocator.release_hisparse_ownership(mapping_indices=logical_locs)
        allocator.free(logical_locs)

        self.assertEqual(mapping.tolist(), [0] * mapping.numel())
        self.assertEqual(physical.available_size(), physical.size)
        self.assertEqual(physical.free_pages, [1, 2])

    def test_release_mapped_pages_retires_every_speculative_alias(self):
        """A committed C4 page is returned only after all aliases are clear."""
        page_size = 4
        mapping = torch.tensor([0, 5, 0, 7, 9, 6, 0, 11], dtype=torch.int64)
        child_allocator = _PhysicalPageAllocator(page_size=page_size, num_pages=3)
        child_allocator.alloc(page_size)
        child_allocator.alloc(page_size)
        ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=child_allocator, page_size=page_size
        )

        ownership.release_mapped_pages(torch.tensor([1, 1], dtype=torch.int64))

        self.assertEqual(mapping.tolist(), [0, 0, 0, 0, 9, 0, 0, 11])
        self.assertEqual(child_allocator.used_pages, {2})
        self.assertEqual(child_allocator.free_pages.count(1), 1)

        # A repeated lifecycle cleanup observes the page in the allocator's
        # reusable set and must not return it a second time.
        ownership.release_mapped_pages(torch.tensor([1], dtype=torch.int64))
        self.assertEqual(child_allocator.free_pages.count(1), 1)

    def test_release_mapped_pages_rejects_side_buffer_owner(self):
        page_size = 4
        mapping = torch.tensor([0, 5, 6, 7], dtype=torch.int64)
        child_allocator = _PhysicalPageAllocator(page_size=page_size, num_pages=2)
        side_page = child_allocator.alloc(page_size)
        ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=child_allocator, page_size=page_size
        )
        ownership.claim(side_page)

        with self.assertRaisesRegex(RuntimeError, "coordinator-owned pages"):
            ownership.release_mapped_pages(torch.tensor([1], dtype=torch.int64))

        self.assertEqual(mapping.tolist(), [0, 5, 6, 7])
        self.assertEqual(child_allocator.used_pages, {1})

    def test_dsv4_finish_releases_composite_and_coordinator_c4_pages(self):
        """Finish must release both physical owners before logical free."""
        logical = _LogicalPageAllocator()
        physical = _PhysicalPageAllocator()
        mapping = torch.zeros(4 * logical.size + physical.page_size, dtype=torch.int64)
        c4_pool = _C4Pool(mapping)

        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator.compress_ratio = 4
        allocator.page_size = logical.page_size
        allocator.hisparse_page_size = physical.page_size
        allocator.logical_attn_allocator = logical
        allocator.hisparse_attn_allocator = physical
        allocator.hisparse_kvcache = c4_pool
        allocator.full_to_hisparse_device_index_mapping = mapping
        allocator.is_not_in_free_group = True
        allocator.free_group = []
        allocator._page_ownership = _HiSparsePageOwnership(
            mapping=mapping,
            child_allocator=physical,
            page_size=physical.page_size,
        )

        # The target coordinator canonically owns one fixed C4 side-buffer page.
        coordinator_page = physical.alloc(physical.page_size)
        allocator.claim_hisparse_ownership(coordinator_page)
        coordinator_page_id = int(coordinator_page[0] // physical.page_size)

        logical_locs = allocator.alloc_extend(
            prefix_lens=torch.tensor([0]),
            prefix_lens_cpu=torch.tensor([0]),
            seq_lens=torch.tensor([logical.page_size]),
            seq_lens_cpu=torch.tensor([logical.page_size]),
            last_loc=torch.tensor([-1]),
            extend_num_tokens=logical.page_size,
        )
        self.assertEqual(len(physical.used_pages), 2)

        compressed_locs = c4_pool.translate_loc_from_full_to_compressed(logical_locs)
        device_buffer_owner = coordinator_page.clone()
        allocator.release_hisparse_ownership(
            mapping_indices=compressed_locs,
            extra_owned_coordinates=device_buffer_owner,
            clear_extra_owner=device_buffer_owner.zero_,
        )

        self.assertEqual(physical.used_pages, set())
        self.assertTrue(torch.all(mapping[:-1] == 0))
        self.assertTrue(torch.all(device_buffer_owner == 0))
        self.assertEqual(coordinator_page_id, 1)

        # release_kv_cache runs after coordinator cleanup and owns logical pages.
        allocator.free(logical_locs)
        self.assertEqual(physical.used_pages, set())

    def test_dsv4_finish_releases_mapping_after_side_buffer_is_empty(self):
        """DSV4 mapping ownership outlives an already-retired side buffer."""
        logical = _LogicalPageAllocator(page_size=64, num_pages=2)
        physical = _PhysicalPageAllocator(page_size=16, num_pages=2)
        mapping = torch.zeros(64, dtype=torch.int64)
        c4_pool = _C4Pool(mapping)

        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator.compress_ratio = 4
        allocator.page_size = logical.page_size
        allocator.hisparse_page_size = physical.page_size
        allocator.logical_attn_allocator = logical
        allocator.hisparse_attn_allocator = physical
        allocator.hisparse_kvcache = c4_pool
        allocator.full_to_hisparse_device_index_mapping = mapping
        allocator.is_not_in_free_group = True
        allocator.free_group = []
        allocator._page_ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=physical, page_size=physical.page_size
        )

        page = physical.alloc(physical.page_size)
        logical_locs = torch.arange(3, 64, 4, dtype=torch.int64)
        compressed_locs = c4_pool.translate_loc_from_full_to_compressed(logical_locs)
        mapping[compressed_locs] = page

        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.decode_producer_stream = None
        coordinator.wait_for_pending_backup = lambda: None
        coordinator.clear_pending_draft_extend_backup = lambda: None
        coordinator.is_dsv4_hisparse = True
        coordinator._device_slot_owner = coordinator
        coordinator._host_slot_owner = coordinator
        coordinator._is_resident = lambda req_idx: False
        coordinator.req_device_buffer_size = torch.zeros(1, dtype=torch.int64)
        coordinator.req_to_device_buffer = torch.zeros((1, 16), dtype=torch.int64)
        coordinator.req_device_buffer_tokens = torch.full(
            (1, 1, 16), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_token_locs = torch.full(
            (1, 1, 16), -1, dtype=torch.int32
        )
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=logical_locs.unsqueeze(0)
        )
        coordinator.mem_pool_device = c4_pool
        coordinator.token_to_kv_pool_allocator = allocator
        coordinator.req_to_host_pool = torch.full((1, 16), -1, dtype=torch.int64)
        coordinator.req_to_host_pool_allocated_len = torch.zeros(1, dtype=torch.int64)
        coordinator._debug_validate_host_request_slots = lambda *args, **kwargs: None
        coordinator.lru_slots = torch.zeros((1, 1, 16), dtype=torch.int16)
        coordinator._lru_init = torch.zeros(16, dtype=torch.int16)
        coordinator._skip_first_backup = torch.zeros(1, dtype=torch.bool)
        coordinator._req_c4_retired_len = {0: 0}
        coordinator._req_c4_written_len = {0: 16}
        coordinator.active_hisparse_reqs = {0: object()}
        coordinator._clear_residency_state = lambda req_idx: None
        req = SimpleNamespace(
            rid="empty-side-buffer-finish",
            req_pool_idx=0,
            kv=SimpleNamespace(kv_allocated_len=logical_locs.numel()),
        )

        HiSparseCoordinator.request_finished(coordinator, req)

        self.assertTrue(torch.all(mapping[compressed_locs] == 0))
        self.assertEqual(physical.used_pages, set())
        self.assertEqual(physical.available_size(), physical.size)

    def test_dsv4_consecutive_natural_finish_restores_side_page_capacity(self):
        """The real coordinator finish path must survive request-slot reuse."""
        page_size = 4
        logical = _LogicalPageAllocator(page_size=16, num_pages=2)
        physical = _PhysicalPageAllocator(page_size=page_size, num_pages=3)
        mapping = torch.zeros(32, dtype=torch.int64)
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator.compress_ratio = 4
        allocator.page_size = logical.page_size
        allocator.hisparse_page_size = page_size
        allocator.logical_attn_allocator = logical
        allocator.hisparse_attn_allocator = physical
        allocator.full_to_hisparse_device_index_mapping = mapping
        allocator.is_not_in_free_group = True
        allocator.free_group = []
        allocator._page_ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=physical, page_size=page_size
        )

        logical_locs = torch.tensor([1, 5, 9], dtype=torch.int64)
        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.debug_validate_lifecycle = False
        coordinator.decode_producer_stream = None
        coordinator.wait_for_pending_backup = lambda: None
        coordinator.clear_pending_draft_extend_backup = lambda: None
        coordinator.is_dsv4_hisparse = True
        coordinator._device_slot_owner = coordinator
        coordinator._host_slot_owner = coordinator
        coordinator._is_resident = lambda req_idx: False
        coordinator.req_device_buffer_size = torch.zeros(1, dtype=torch.int64)
        coordinator.req_to_device_buffer = torch.zeros((1, 4), dtype=torch.int64)
        coordinator.req_device_buffer_tokens = torch.full(
            (1, 1, 4), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_token_locs = torch.full(
            (1, 1, 4), -1, dtype=torch.int32
        )
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=logical_locs.unsqueeze(0)
        )
        coordinator.mem_pool_device = SimpleNamespace(
            translate_loc_from_full_to_compressed=lambda locs: locs,
            full_to_hisparse_device_index_mapping=mapping,
        )
        coordinator.token_to_kv_pool_allocator = allocator
        coordinator.req_to_host_pool = torch.full((1, 4), -1, dtype=torch.int64)
        coordinator.req_to_host_pool_allocated_len = torch.zeros(1, dtype=torch.int64)
        coordinator.lru_slots = torch.zeros((1, 1, 4), dtype=torch.int16)
        coordinator._lru_init = torch.zeros(4, dtype=torch.int16)
        coordinator._skip_first_backup = torch.zeros(1, dtype=torch.bool)
        coordinator._req_c4_retired_len = {}
        coordinator._req_c4_written_len = {}
        coordinator._clear_residency_state = lambda req_idx: None
        initial_available = physical.available_size()

        for request_index in range(8):
            side_page = physical.alloc(page_size)
            allocator.claim_hisparse_ownership(side_page)
            coordinator.req_to_device_buffer[0].copy_(side_page)
            coordinator.req_device_buffer_size[0] = page_size
            mapping[logical_locs] = side_page[: logical_locs.numel()]
            coordinator.active_hisparse_reqs = {0: object()}
            req = SimpleNamespace(
                rid=f"natural-finish-{request_index}",
                req_pool_idx=0,
                kv=SimpleNamespace(kv_allocated_len=logical_locs.numel()),
            )

            HiSparseCoordinator.request_finished(coordinator, req)

            self.assertEqual(physical.available_size(), initial_available)
            self.assertEqual(allocator._page_ownership._extra_owner_page_ids, set())
            self.assertTrue(torch.all(mapping == 0))
            self.assertTrue(torch.all(coordinator.req_to_device_buffer == 0))

    def test_dsv4_page_boundary_rehomes_24_temporary_pages_without_leak(self):
        """Every speculative C4 page is released before its mapping is replaced.

        A 1,500-token generation crosses about 24 logical 64-token boundaries.
        At each boundary EAGLE has reserved a complete temporary C4 page, while
        only its first semantic slot is about to be remapped to the stable
        request buffer.  Repeating that transaction must not strand one page per
        boundary.
        """
        c4_page_size = 16
        num_boundaries = 24
        physical = _PhysicalPageAllocator(
            page_size=c4_page_size, num_pages=num_boundaries + 2
        )
        mapping = torch.zeros(
            num_boundaries * c4_page_size + c4_page_size, dtype=torch.int64
        )
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator.compress_ratio = 4
        allocator.page_size = 64
        allocator.hisparse_page_size = c4_page_size
        allocator.hisparse_attn_allocator = physical
        allocator.full_to_hisparse_device_index_mapping = mapping
        allocator._page_ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=physical, page_size=c4_page_size
        )
        allocator.get_last_loc_compressed = lambda locs: locs.to(torch.int64)

        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.device = torch.device("cpu")
        coordinator.token_to_kv_pool_allocator = allocator
        coordinator.mem_pool_device = SimpleNamespace(
            page_size=c4_page_size,
            full_to_hisparse_device_index_mapping=mapping,
        )
        coordinator.is_dsv4_hisparse = True
        coordinator.compress_ratio = 4
        coordinator.device_buffer_size = c4_page_size
        coordinator.padded_buffer_size = 2 * c4_page_size
        coordinator.req_to_device_buffer = torch.zeros(
            (1, coordinator.padded_buffer_size), dtype=torch.int64
        )
        coordinator.req_device_buffer_size = torch.tensor(
            [coordinator.padded_buffer_size], dtype=torch.int64
        )
        coordinator.req_device_buffer_token_locs = torch.zeros(
            (1, 1, coordinator.padded_buffer_size), dtype=torch.int32
        )
        coordinator.req_device_buffer_tokens = torch.full(
            (1, 1, coordinator.padded_buffer_size), -1, dtype=torch.int32
        )
        coordinator._eager_backup_previous_token = lambda *args: None
        coordinator.advance_dynamic_residency = lambda *args: None
        coordinator._is_resident = lambda req_idx: False
        coordinator._grow_device_buffers = (
            lambda *args: coordinator.req_to_device_buffer[
                torch.tensor([0]), torch.tensor([coordinator.device_buffer_size])
            ]
        )

        stable_buffer = torch.cat(
            [physical.alloc(c4_page_size), physical.alloc(c4_page_size)]
        )
        allocator.claim_hisparse_ownership(stable_buffer)
        coordinator.req_to_device_buffer[0].copy_(stable_buffer)
        initial_available = physical.size
        expected_during_request = initial_available - stable_buffer.numel()

        for boundary in range(num_boundaries):
            mapping_start = boundary * c4_page_size
            temporary_page = physical.alloc(c4_page_size)
            mapping[mapping_start : mapping_start + c4_page_size] = temporary_page
            full_seq_len = (mapping_start + 1) * allocator.compress_ratio

            coordinator.map_last_loc_to_buffer(
                seq_lens=torch.tensor([full_seq_len], dtype=torch.int64),
                out_cache_loc=torch.tensor([mapping_start], dtype=torch.int64),
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([full_seq_len], dtype=torch.int64),
                req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
            )

            self.assertEqual(
                physical.available_size(),
                expected_during_request,
                f"temporary C4 page leaked at boundary {boundary}",
            )

        allocator.release_hisparse_ownership(
            mapping_indices=torch.arange(mapping.numel(), dtype=torch.int64),
            extra_owned_coordinates=stable_buffer,
            clear_extra_owner=lambda: coordinator.req_to_device_buffer.zero_(),
        )
        self.assertEqual(physical.available_size(), initial_available)
        self.assertEqual(physical.used_pages, set())
        self.assertEqual(len(physical.free_pages), len(set(physical.free_pages)))
        self.assertTrue(torch.all(mapping == 0))

    def test_dsv4_boundary_rehome_rejects_existing_side_buffer_alias(self):
        """A temporary page cannot simultaneously have two canonical owners."""
        c4_page_size = 16
        physical = _PhysicalPageAllocator(page_size=c4_page_size, num_pages=3)
        mapping = torch.zeros(2 * c4_page_size, dtype=torch.int64)
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator.compress_ratio = 4
        allocator.page_size = 64
        allocator.hisparse_page_size = c4_page_size
        allocator.hisparse_attn_allocator = physical
        allocator.full_to_hisparse_device_index_mapping = mapping
        allocator._page_ownership = _HiSparsePageOwnership(
            mapping=mapping, child_allocator=physical, page_size=c4_page_size
        )
        allocator.get_last_loc_compressed = lambda locs: locs.to(torch.int64)

        temporary_page = physical.alloc(c4_page_size)
        mapping[:c4_page_size] = temporary_page
        stable_page = physical.alloc(c4_page_size)

        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.device = torch.device("cpu")
        coordinator.token_to_kv_pool_allocator = allocator
        coordinator.mem_pool_device = SimpleNamespace(
            page_size=c4_page_size,
            full_to_hisparse_device_index_mapping=mapping,
        )
        coordinator.is_dsv4_hisparse = True
        coordinator.compress_ratio = 4
        coordinator.device_buffer_size = c4_page_size
        coordinator.padded_buffer_size = 2 * c4_page_size
        coordinator.req_to_device_buffer = torch.zeros(
            (1, coordinator.padded_buffer_size), dtype=torch.int64
        )
        coordinator.req_to_device_buffer[0, :c4_page_size] = temporary_page
        coordinator.req_to_device_buffer[0, c4_page_size:] = stable_page
        coordinator.req_device_buffer_size = torch.tensor(
            [coordinator.padded_buffer_size], dtype=torch.int64
        )
        coordinator.req_device_buffer_token_locs = torch.zeros(
            (1, 1, coordinator.padded_buffer_size), dtype=torch.int32
        )
        coordinator.req_device_buffer_tokens = torch.full(
            (1, 1, coordinator.padded_buffer_size), -1, dtype=torch.int32
        )
        coordinator._eager_backup_previous_token = lambda *args: None
        coordinator.advance_dynamic_residency = lambda *args: None

        with self.assertRaisesRegex(
            RuntimeError,
            "temporary pages must not already belong to a device buffer",
        ):
            coordinator.map_last_loc_to_buffer(
                seq_lens=torch.tensor([4], dtype=torch.int64),
                out_cache_loc=torch.tensor([0], dtype=torch.int64),
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
                req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
            )

        self.assertTrue(torch.equal(mapping[:c4_page_size], temporary_page))
        self.assertEqual(physical.used_pages, {1, 2})

    def test_dsv4_extend_allocates_owner_for_direct_pd_partial_page(self):
        """A host-only prompt tail must not continue in sentinel page zero."""
        compress_ratio = 4
        c4_page_size = 16
        logical = MagicMock()
        logical.available_size.return_value = 1024
        logical.alloc_extend.return_value = torch.arange(326, 384, dtype=torch.int64)
        physical = MagicMock()
        physical.is_not_in_free_group = True
        physical.available_size.return_value = 16 * 8
        partial_page = torch.arange(48, 64, dtype=torch.int64)
        physical.alloc.return_value = partial_page
        physical.alloc_extend.return_value = torch.arange(49, 64, dtype=torch.int64)
        mapping = torch.zeros(256, dtype=torch.int64)
        c4_pool = SimpleNamespace(
            translate_loc_from_full_to_compressed=lambda locs: locs[
                (locs + 1) % compress_ratio == 0
            ]
            // compress_ratio,
            # Reproduce a tail already mis-mapped to a positive slot in sentinel
            # page zero. Page ownership, not coordinate sign, decides validity.
            _translate_loc_to_hisparse_device=lambda locs: torch.ones_like(locs),
        )

        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator.compress_ratio = compress_ratio
        allocator.page_size = 64
        allocator.hisparse_page_size = c4_page_size
        allocator.logical_attn_allocator = logical
        allocator.hisparse_attn_allocator = physical
        allocator.hisparse_kvcache = c4_pool
        allocator.full_to_hisparse_device_index_mapping = mapping
        allocator._ensure_hisparse_available = MagicMock(return_value=True)

        result = allocator.alloc_extend(
            prefix_lens=torch.tensor([70], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([70], dtype=torch.int64),
            seq_lens=torch.tensor([128], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([128], dtype=torch.int64),
            last_loc=torch.tensor([325], dtype=torch.int64),
            extend_num_tokens=58,
        )

        torch.testing.assert_close(result, logical.alloc_extend.return_value)
        physical.alloc.assert_called_once_with(c4_page_size)
        self.assertEqual(physical.alloc_extend.call_args.kwargs["num_new_pages"], 0)
        # C4 prefix length is 17, so generated position 17 must continue at
        # offset one of the newly owned page: predecessor = 48 + 1 - 1.
        passed_last_loc = physical.alloc_extend.call_args.args[4]
        self.assertEqual(passed_last_loc.tolist(), [48])
        self.assertNotEqual(passed_last_loc.tolist(), [1])

    def test_dsv4_extend_rolls_back_new_pages_after_c4_failure(self):
        """A composite allocation failure must preserve both old tail pages."""
        compress_ratio = 4
        c4_page_size = 16
        logical = MagicMock()
        logical.available_size.return_value = 1024
        # Prefix token 69 lives in logical page 5. The extension reuses its tail
        # and allocates logical page 6, which is the only page rollback may free.
        logical.alloc_extend.return_value = torch.arange(326, 448, dtype=torch.int64)
        physical = MagicMock()
        physical.is_not_in_free_group = True
        physical.available_size.return_value = c4_page_size * 8
        partial_page = torch.arange(48, 64, dtype=torch.int64)
        physical.alloc.return_value = partial_page
        physical.alloc_extend.return_value = None
        mapping = torch.zeros(256, dtype=torch.int64)
        c4_pool = SimpleNamespace(
            translate_loc_from_full_to_compressed=lambda locs: locs[
                (locs + 1) % compress_ratio == 0
            ]
            // compress_ratio,
            _translate_loc_to_hisparse_device=lambda locs: mapping[locs],
        )

        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator.compress_ratio = compress_ratio
        allocator.page_size = 64
        allocator.hisparse_page_size = c4_page_size
        allocator.logical_attn_allocator = logical
        allocator.hisparse_attn_allocator = physical
        allocator.hisparse_kvcache = c4_pool
        allocator.full_to_hisparse_device_index_mapping = mapping
        allocator._ensure_hisparse_available = MagicMock(return_value=True)

        with self.assertRaisesRegex(RuntimeError, "alloc_extend"):
            allocator.alloc_extend(
                prefix_lens=torch.tensor([70], dtype=torch.int64),
                prefix_lens_cpu=torch.tensor([70], dtype=torch.int64),
                seq_lens=torch.tensor([192], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([192], dtype=torch.int64),
                last_loc=torch.tensor([325], dtype=torch.int64),
                extend_num_tokens=122,
            )

        # The new direct-PD C4 owner is returned. The reused logical page 5 is
        # kept, while only the newly allocated logical page 6 is rolled back.
        torch.testing.assert_close(physical.free.call_args.args[0], partial_page)
        torch.testing.assert_close(
            logical.free.call_args.args[0], torch.arange(384, 448, dtype=torch.int64)
        )
        self.assertTrue(torch.all(mapping == 0))

    def test_dsv4_extend_rolls_back_logical_page_when_partial_owner_fails(self):
        """The direct-PD owner allocation is part of the same transaction."""
        compress_ratio = 4
        c4_page_size = 16
        logical = MagicMock()
        logical.available_size.return_value = 1024
        logical.alloc_extend.return_value = torch.arange(326, 448, dtype=torch.int64)
        physical = MagicMock()
        physical.is_not_in_free_group = True
        physical.available_size.return_value = c4_page_size * 8
        physical.alloc.return_value = None
        mapping = torch.zeros(256, dtype=torch.int64)
        c4_pool = SimpleNamespace(
            translate_loc_from_full_to_compressed=lambda locs: locs[
                (locs + 1) % compress_ratio == 0
            ]
            // compress_ratio,
            _translate_loc_to_hisparse_device=lambda locs: mapping[locs],
        )

        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator.compress_ratio = compress_ratio
        allocator.page_size = 64
        allocator.hisparse_page_size = c4_page_size
        allocator.logical_attn_allocator = logical
        allocator.hisparse_attn_allocator = physical
        allocator.hisparse_kvcache = c4_pool
        allocator.full_to_hisparse_device_index_mapping = mapping
        allocator._ensure_hisparse_available = MagicMock(return_value=True)

        with self.assertRaisesRegex(RuntimeError, "partial-page owners"):
            allocator.alloc_extend(
                prefix_lens=torch.tensor([70], dtype=torch.int64),
                prefix_lens_cpu=torch.tensor([70], dtype=torch.int64),
                seq_lens=torch.tensor([192], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([192], dtype=torch.int64),
                last_loc=torch.tensor([325], dtype=torch.int64),
                extend_num_tokens=122,
            )

        physical.alloc_extend.assert_not_called()
        physical.free.assert_not_called()
        torch.testing.assert_close(
            logical.free.call_args.args[0], torch.arange(384, 448, dtype=torch.int64)
        )
        self.assertTrue(torch.all(mapping == 0))

    def test_dsv4_shared_allocator_releases_mirrored_page_once(self):
        """Target and draft aliases on one allocator release a page once."""
        page_size = 64
        physical = _PhysicalPageAllocator(page_size=page_size)
        shared_mapping = torch.zeros(page_size + 1, dtype=torch.int64)
        ownership = _HiSparsePageOwnership(
            mapping=shared_mapping,
            child_allocator=physical,
            page_size=page_size,
        )
        shared_page = physical.alloc(page_size)
        shared_mapping[:page_size] = shared_page
        target_buffer_owner = shared_page.clone()
        draft_buffer_alias = shared_page.clone()
        ownership.claim(target_buffer_owner)

        # The draft coordinator is only a mirror: detach its local alias first.
        # It shares both the allocator and mapping with target, so it must not
        # clear the canonical mapping or return physical pages.
        draft_buffer_alias.zero_()

        self.assertEqual(physical.used_pages, {1})
        self.assertTrue(torch.all(shared_mapping[:page_size] == shared_page))
        self.assertTrue(torch.all(draft_buffer_alias == 0))

        # The target coordinator clears the shared mapping and its local owner,
        # then returns the complete physical page exactly once.
        ownership.release(
            mapping_indices=torch.arange(page_size),
            extra_owned_coordinates=target_buffer_owner,
            clear_extra_owner=target_buffer_owner.zero_,
        )
        self.assertEqual(physical.used_pages, set())
        self.assertTrue(torch.all(shared_mapping == 0))
        self.assertTrue(torch.all(target_buffer_owner == 0))

    def test_forwards_swa_tail_allocation_to_logical_allocator(self):
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        logical_allocator = MagicMock(spec=["alloc_extend_swa_tail"])
        allocator.logical_attn_allocator = logical_allocator

        expected = torch.tensor([8, 9, 10], dtype=torch.int64)
        logical_allocator.alloc_extend_swa_tail.return_value = expected

        prefix_lens = torch.tensor([0], dtype=torch.int64)
        prefix_lens_cpu = torch.tensor([0], dtype=torch.int64)
        seq_lens = torch.tensor([512], dtype=torch.int64)
        seq_lens_cpu = torch.tensor([512], dtype=torch.int64)
        last_loc = torch.tensor([-1], dtype=torch.int64)

        result = allocator.alloc_extend_swa_tail(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=512,
            swa_tail_len=128,
        )

        self.assertIs(result, expected)
        logical_allocator.alloc_extend_swa_tail.assert_called_once()
        _, kwargs = logical_allocator.alloc_extend_swa_tail.call_args
        self.assertIs(kwargs["prefix_lens"], prefix_lens)
        self.assertIs(kwargs["prefix_lens_cpu"], prefix_lens_cpu)
        self.assertIs(kwargs["seq_lens"], seq_lens)
        self.assertIs(kwargs["seq_lens_cpu"], seq_lens_cpu)
        self.assertIs(kwargs["last_loc"], last_loc)
        self.assertEqual(kwargs["extend_num_tokens"], 512)
        self.assertEqual(kwargs["swa_tail_len"], 128)

    def test_hisparse_budget_uses_full_logical_capacity_for_swa_tail(self):
        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        logical_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=32),
            full_available_size=MagicMock(return_value=512),
        )
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            logical_attn_allocator=logical_allocator
        )
        queue.scheduler = SimpleNamespace(enable_hisparse=True, last_batch=None)
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._need_space_for_single_req = MagicMock(return_value=0)
        queue._active_reserved_tokens = MagicMock(return_value=0)

        budget = queue._allocatable_token_budgets()

        self.assertEqual(budget, 512)
        logical_allocator.full_available_size.assert_called_once_with()
        logical_allocator.available_size.assert_not_called()

    def test_hisparse_prealloc_uses_swa_tail_for_direct_host_path(self):
        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        fill_len = 512
        swa_tail_len = 128
        kv_loc = torch.arange(512, 512 + fill_len, dtype=torch.int64)
        host_indices = torch.arange(1000, 1128, dtype=torch.int64)

        req = SimpleNamespace(
            rid="req-0",
            origin_input_ids=list(range(fill_len)),
            output_ids=[],
            kv=None,
        )

        def set_extend_range(start, end):
            req.extend_range = SimpleNamespace(start=start, end=end, length=end - start)

        req.set_extend_range = set_extend_range

        class ReqToTokenPool:
            def __init__(self):
                self.writes = []

            def alloc(self, reqs):
                for item in reqs:
                    item.req_pool_idx = 0
                return torch.tensor([0], dtype=torch.int64)

            def write(self, indices, values):
                self.writes.append((indices, values))

        req_to_token_pool = ReqToTokenPool()
        allocator = SimpleNamespace(
            device=torch.device("cpu"),
            page_size=256,
            available_size=MagicMock(return_value=fill_len),
            alloc_extend_swa_tail=MagicMock(return_value=kv_loc),
            alloc_logical_only=MagicMock(return_value=kv_loc),
        )
        regular_host_alloc = MagicMock(return_value=host_indices)
        req_to_host_pool = torch.full((1, len(host_indices)), -1, dtype=torch.int64)
        req_to_host_pool_allocated_len = torch.zeros(1, dtype=torch.int64)
        coordinator = SimpleNamespace(
            mem_pool_host=SimpleNamespace(alloc_paged_token_slots=regular_host_alloc),
            req_to_host_pool=req_to_host_pool,
            req_to_host_pool_allocated_len=req_to_host_pool_allocated_len,
            host_token_len=MagicMock(side_effect=lambda token_len: token_len // 4),
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.req_to_token_pool = req_to_token_pool
        queue.token_to_kv_pool_allocator = allocator
        queue.tree_cache = SimpleNamespace(
            evictable_size=MagicMock(return_value=0),
            protected_size=MagicMock(return_value=0),
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            hisparse_coordinator=coordinator,
            draft_hisparse_coordinator=None,
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
        )
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_tail_len = MagicMock(return_value=swa_tail_len)

        result = queue._pre_alloc(req)

        self.assertTrue(torch.equal(result, host_indices))
        allocator.alloc_extend_swa_tail.assert_called_once()
        allocator.alloc_logical_only.assert_not_called()
        _, kwargs = allocator.alloc_extend_swa_tail.call_args
        self.assertEqual(kwargs["extend_num_tokens"], fill_len)
        self.assertEqual(kwargs["swa_tail_len"], swa_tail_len)
        self.assertEqual(req.kv.swa_evicted_seqlen, fill_len - swa_tail_len)
        self.assertEqual(req.kv.kv_allocated_len, fill_len)
        self.assertEqual(req.kv_committed_len, fill_len)
        self.assertEqual(req.extend_range.length, fill_len)
        self.assertEqual(len(req_to_token_pool.writes), 1)
        self.assertEqual(
            coordinator.host_token_len.call_args_list,
            [unittest.mock.call(fill_len), unittest.mock.call(fill_len)],
        )
        regular_host_alloc.assert_called_once_with(
            coordinator.req_to_host_pool,
            coordinator.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            0,
            len(host_indices),
        )
        self.assertTrue(torch.equal(req_to_token_pool.writes[0][1], kv_loc))

        # C4 indexer/C128 use the logical allocator's full-page IDs. They do not
        # use either the independently allocated host pages or C4 sparse slots.
        np.testing.assert_array_equal(
            np.unique(kv_loc.numpy() // allocator.page_size),
            np.array([2, 3]),
        )

    def test_mooncake_uses_separate_host_and_device_page_indices(self):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        manager = object.__new__(MooncakeKVManager)
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_custom_mem_pool = False
        manager._transfer_data = MagicMock(return_value=0)

        with ThreadPoolExecutor(max_workers=1) as executor:
            ret = manager._send_kvcache_generic(
                mooncake_session_id="session",
                src_data_ptrs=[1000, 2000, 3000],
                dst_data_ptrs=[10000, 20000, 30000],
                item_lens=[100, 100, 100],
                prefill_data_indices=np.array([1, 2], dtype=np.int32),
                dst_data_indices=np.array([7, 8], dtype=np.int32),
                executor=executor,
                dst_device_data_indices=np.array([21, 22], dtype=np.int32),
                dst_device_data_ptrs={20000, 30000},
            )

        self.assertEqual(ret, 0)
        manager._transfer_data.assert_called_once_with(
            "session",
            [
                (1100, 10700, 200),
                (2100, 22100, 200),
                (3100, 32100, 200),
            ],
        )

    def test_mooncake_derives_device_buffers_from_local_pp_layout(self):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[1000, 2000, 3000],
            kv_item_lens=[100, 100, 100],
            kv_layer_ids=[],
            mla_compression_ratios=[4, 128, 4, 128],
            prefill_start_layer=0,
            prefill_end_layer=2,
        )
        manager._send_kvcache_generic = MagicMock(return_value=0)
        executor = MagicMock()

        manager.send_kvcache(
            "session",
            np.array([1], dtype=np.int32),
            [10000, 20000, 30000],
            np.array([7], dtype=np.int32),
            executor,
            dst_device_kv_indices=np.array([21], dtype=np.int32),
        )

        kwargs = manager._send_kvcache_generic.call_args.kwargs
        self.assertEqual(kwargs["dst_device_data_ptrs"], {20000, 30000})

    def test_mooncake_transfer_metadata_carries_device_page_indices(self):
        from sglang.srt.disaggregation.mooncake.conn import TransferInfo

        host_pages = np.array([7, 8], dtype=np.int32)
        device_pages = np.array([21, 22], dtype=np.int32)
        info = TransferInfo.from_zmq(
            [
                b"9",
                b"127.0.0.1",
                b"12345",
                b"session",
                host_pages.tobytes(),
                b"0",
                b"",
                b"1",
                b"0",
                device_pages.tobytes(),
            ]
        )

        np.testing.assert_array_equal(info.dst_kv_indices, host_pages)
        np.testing.assert_array_equal(info.dst_device_kv_indices, device_pages)


if __name__ == "__main__":
    unittest.main()
