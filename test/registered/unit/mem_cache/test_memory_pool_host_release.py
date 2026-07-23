"""Unit tests for lazy host-pool free-list release bookkeeping."""

import threading
import unittest

import torch

from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    LogicalHostPool,
    MHATokenToKVPoolHost,
    MambaPoolHost,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestLazyHostPoolRelease(CustomTestCase):
    @staticmethod
    def _make_mha_pool():
        pool = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        pool.size = 8
        pool.page_size = 1
        pool.device = "cpu"
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    @staticmethod
    def _make_mamba_pool():
        pool = MambaPoolHost.__new__(MambaPoolHost)
        pool.size = 8
        pool.page_size = 1
        pool.device = "cpu"
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    @staticmethod
    def _make_deepseek_v4_pool():
        pool = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
        pool.size = 8
        pool.slot_page_size = 2
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    def _assert_lazy_release(self, pool):
        self.assertEqual(pool.free(torch.empty(0, dtype=torch.int64)), 0)
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])

        allocated = pool.alloc(6)
        free_slots_before = pool.free_slots
        pool.free(allocated[:2])

        # free() records a released chunk without rebuilding the primary list.
        self.assertIs(pool.free_slots, free_slots_before)
        self.assertEqual(pool.num_release_slots, 2)
        self.assertEqual(len(pool.release_slots), 1)
        self.assertEqual(pool.available_size(), 4)

        # Existing slots are consumed before pending releases are merged.
        self.assertTrue(torch.equal(pool.alloc(2), torch.tensor([6, 7])))
        self.assertEqual(pool.num_release_slots, 2)

        # Exhausting the primary list merges and reuses pending slots.
        self.assertTrue(torch.equal(pool.alloc(2), torch.tensor([0, 1])))
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])
        self.assertEqual(pool.available_size(), 0)

        pool.clear()
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])
        self.assertEqual(pool.available_size(), 8)

        # Exercise the general merge path with multiple released chunks.
        allocated = pool.alloc(8)
        pool.free(allocated[:2])
        pool.free(allocated[2:4])
        self.assertEqual(len(pool.release_slots), 2)
        self.assertTrue(torch.equal(pool.alloc(4), torch.tensor([0, 1, 2, 3])))
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])

    def test_mha_pool_lazy_release(self):
        self._assert_lazy_release(self._make_mha_pool())

    def test_mamba_pool_lazy_release(self):
        self._assert_lazy_release(self._make_mamba_pool())

    def test_logical_pool_lazy_release(self):
        self._assert_lazy_release(LogicalHostPool(size=8, page_size=1))

    def test_deepseek_v4_pool_lazy_release(self):
        pool = self._make_deepseek_v4_pool()
        self._assert_lazy_release(pool)

        # Preserve the V4 pool's page-aligned allocation behavior.
        pool.clear()
        self.assertEqual(len(pool.alloc(1)), 2)


if __name__ == "__main__":
    unittest.main()
