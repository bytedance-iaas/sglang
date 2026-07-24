import sys
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.eic_cache_controller import (
    EICCacheController,
    EICCacheOperation,
)
from sglang.srt.mem_cache.eic_chunk_cache import EICChunkCache
from sglang.srt.mem_cache.eic_hiradix_cache import EICPagedHiRadixCache
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class FakeHostPool:
    def alloc(self, size: int):
        return torch.arange(size, dtype=torch.int32)


class FakeDeviceAllocator:
    def alloc(self, size: int):
        return torch.arange(100, 100 + size, dtype=torch.int32)


class TestEICHiCacheRegression(unittest.TestCase):
    def test_match_from_remote_skips_zero_committed_tokens(self):
        cache = object.__new__(EICPagedHiRadixCache)
        cache.match_req_set = []
        cache.cache_controller = mock.Mock()
        cache._match_for_remote_fetch = mock.Mock(
            side_effect=AssertionError("empty remote key should be skipped")
        )

        req = mock.Mock(
            rid="health-rid",
            origin_input_ids=[0],
            output_ids=[],
            extra_key=None,
        )

        cache.match_from_remote([req])

        cache._match_for_remote_fetch.assert_not_called()
        cache.cache_controller.batch_find_longest_prefix_in_eic.assert_not_called()
        self.assertEqual(cache.match_req_set, [])

    def test_eic_controller_write_uses_queue_api(self):
        controller = object.__new__(EICCacheController)
        controller.mem_pool_host = FakeHostPool()
        controller.write_queue = Queue()

        device_indices = torch.tensor([7, 8, 9], dtype=torch.int32)
        host_indices = controller.write(device_indices, priority=-2, node_id=11)

        self.assertEqual(host_indices.tolist(), [0, 1, 2])
        operation = controller.write_queue.get_nowait()
        self.assertIsInstance(operation, EICCacheOperation)
        self.assertEqual(operation.host_indices.tolist(), [0, 1, 2])
        self.assertIs(operation.device_indices, device_indices)
        self.assertEqual(operation.node_id, 11)
        self.assertEqual(operation.node_ids, [11])
        self.assertIsNone(operation.content_hash)
        self.assertEqual(operation.priority, -2)

    def test_eic_controller_load_uses_queue_api(self):
        controller = object.__new__(EICCacheController)
        controller.mem_pool_device_allocator = FakeDeviceAllocator()
        controller.load_queue = Queue()

        host_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
        with mock.patch(
            "sglang.srt.managers.eic_cache_controller.torch.cuda.current_stream"
        ) as current_stream:
            stream = mock.Mock()
            current_stream.return_value = stream
            device_indices = controller.load(host_indices, priority=-3, node_id=12)

        stream.synchronize.assert_called_once()
        self.assertEqual(device_indices.tolist(), [100, 101, 102])
        operation = controller.load_queue.get_nowait()
        self.assertIsInstance(operation, EICCacheOperation)
        self.assertIs(operation.host_indices, host_indices)
        self.assertEqual(operation.device_indices.tolist(), [100, 101, 102])
        self.assertEqual(operation.node_id, 12)
        self.assertEqual(operation.node_ids, [12])
        self.assertIsNone(operation.content_hash)
        self.assertEqual(operation.priority, -3)

    def test_unified_tree_node_exposes_storage_hash_helpers(self):
        root = UnifiedTreeNode((ComponentType.FULL, ComponentType.SWA))
        child = UnifiedTreeNode((ComponentType.FULL, ComponentType.SWA))
        child.parent = root
        root.hash_value = ["root-hash"]
        child.hash_value = ["child-hash-0", "child-hash-1"]

        self.assertEqual(root.get_last_hash_value(), "root-hash")
        self.assertEqual(child.get_last_hash_value(), "child-hash-1")
        self.assertEqual(child.get_prefix_hash_values(child.parent), ["root-hash"])
        self.assertFalse(
            hasattr(root, "hicache_storage_pass_prefix_keys"),
            "storage pass-prefix config belongs to the cache, not tree nodes",
        )

    def test_batch_exists_impl_failed_batch_keeps_cardinality(self):
        # A failed top-level mexist must yield exactly one False per input key.
        # The EIC outcome object may still carry per-object status codes; if the
        # loop below falls through to them we would return 2*N results, which
        # later overflows component_keys indexing in _batch_io_v2. See the
        # `continue` guard in EICStorage._batch_exists_impl.
        fake_eic = SimpleNamespace(
            StatusCode=SimpleNamespace(SUCCESS=0, FAILED=1),
            StringVector=list,
            ExistOption=SimpleNamespace,
        )
        with mock.patch.dict(sys.modules, {"eic": fake_eic}):
            from sglang.srt.mem_cache.storage.eic.eic_storage import EICStorage

            storage = object.__new__(EICStorage)
            storage.eic_namespace = "poc"
            storage._get_eic_key = lambda keys: list(keys)

            outcome = SimpleNamespace(status_codes=[fake_eic.StatusCode.SUCCESS] * 3)
            storage.connection = mock.Mock()
            storage.connection.mexist.return_value = (
                fake_eic.StatusCode.FAILED,
                outcome,
            )

            keys = [f"k{i}" for i in range(3)]
            result = storage._batch_exists_impl(keys)

        self.assertEqual(result, [False, False, False])
        self.assertEqual(len(result), len(keys))

    def test_eic_chunk_cache_passes_tp_group_to_controller(self):
        cache = object.__new__(EICChunkCache)
        cache.page_size = 16
        cache.load_cache_event = object()
        cache.token_to_kv_pool_host = object()
        params = SimpleNamespace(
            token_to_kv_pool_allocator=object(), tp_cache_group=object()
        )
        server_args = SimpleNamespace()

        with mock.patch(
            "sglang.srt.mem_cache.eic_chunk_cache.EICCacheController"
        ) as controller_cls:
            EICChunkCache._init_cache_controller(cache, params, server_args)

        controller_cls.assert_called_once_with(
            params.token_to_kv_pool_allocator,
            cache.token_to_kv_pool_host,
            cache.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=cache.load_cache_event,
            write_policy="write_through",
            server_args=server_args,
        )

    def test_free_swa_skips_reserved_page_and_dedups_aliases(self):
        # free_swa must drop reserved-page sentinels + dedup aliases under EIC.
        from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

        alloc = object.__new__(SWATokenToKVPoolAllocator)
        alloc.page_size = 256
        alloc.swa_attn_allocator = mock.Mock()
        alloc.full_to_swa_index_mapping = torch.tensor(
            [6, 8, 10, 21, 512, 512, 512, 768], dtype=torch.int64
        )
        alloc._expand_to_full_pages = lambda idx: idx

        with mock.patch(
            "sglang.srt.mem_cache.swa_memory_pool.get_global_server_args",
            return_value=SimpleNamespace(enable_eic_cache=True),
        ):
            SWATokenToKVPoolAllocator.free_swa(alloc, torch.arange(8))

        freed = alloc.swa_attn_allocator.free.call_args[0][0]
        self.assertEqual(sorted(freed.tolist()), [512, 768])


if __name__ == "__main__":
    unittest.main()
