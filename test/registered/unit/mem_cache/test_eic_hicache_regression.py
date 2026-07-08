import unittest
from queue import Queue
from unittest import mock

import torch

from sglang.srt.managers.eic_cache_controller import (
    EICCacheController,
    EICCacheOperation,
)
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


if __name__ == "__main__":
    unittest.main()
