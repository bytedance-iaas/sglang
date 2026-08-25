import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    _HiSparsePageOwnership,
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
