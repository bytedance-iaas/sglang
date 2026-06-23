import unittest
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _RecordingAllocator:
    def __init__(self, size: int):
        self.size = size
        self.freed = []

    def free(self, indices: torch.Tensor):
        self.freed.append(indices.clone())

    def available_size(self):
        return self.size


def _make_swa_allocator_stub(mapping_size: int = 16):
    stub = SimpleNamespace(
        page_size=1,
        is_not_in_free_group=True,
        free_group=[],
        full_attn_allocator=_RecordingAllocator(mapping_size),
        swa_attn_allocator=_RecordingAllocator(mapping_size),
        full_to_swa_index_mapping=torch.zeros(mapping_size, dtype=torch.int64),
    )
    stub.free = MethodType(SWATokenToKVPoolAllocator.free, stub)
    stub.free_swa = MethodType(SWATokenToKVPoolAllocator.free_swa, stub)
    return stub


class TestSWAFreeDedup(CustomTestCase):
    def test_group_free_deduplicates_full_indices(self):
        allocator = _make_swa_allocator_stub()
        allocator.full_to_swa_index_mapping[3] = 7

        BaseTokenToKVPoolAllocator.free_group_begin(allocator)
        allocator.free(torch.tensor([3], dtype=torch.int64))
        allocator.free(torch.tensor([3], dtype=torch.int64))
        BaseTokenToKVPoolAllocator.free_group_end(allocator)

        self.assertEqual(len(allocator.full_attn_allocator.freed), 1)
        self.assertTrue(
            torch.equal(
                allocator.full_attn_allocator.freed[0],
                torch.tensor([3], dtype=torch.int64),
            )
        )
        self.assertEqual(len(allocator.swa_attn_allocator.freed), 1)
        self.assertTrue(
            torch.equal(
                allocator.swa_attn_allocator.freed[0],
                torch.tensor([7], dtype=torch.int64),
            )
        )
        self.assertEqual(int(allocator.full_to_swa_index_mapping[3]), 0)

    def test_free_swa_deduplicates_aliased_swa_indices(self):
        allocator = _make_swa_allocator_stub()
        allocator.full_to_swa_index_mapping[1] = 5
        allocator.full_to_swa_index_mapping[2] = 5

        allocator.free_swa(torch.tensor([1, 2], dtype=torch.int64))

        self.assertEqual(len(allocator.swa_attn_allocator.freed), 1)
        self.assertTrue(
            torch.equal(
                allocator.swa_attn_allocator.freed[0],
                torch.tensor([5], dtype=torch.int64),
            )
        )
        self.assertEqual(int(allocator.full_to_swa_index_mapping[1]), 0)
        self.assertEqual(int(allocator.full_to_swa_index_mapping[2]), 0)


if __name__ == "__main__":
    unittest.main()
