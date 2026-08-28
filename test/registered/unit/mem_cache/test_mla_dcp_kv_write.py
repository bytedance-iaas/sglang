import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMLADCPKVWrite(CustomTestCase):
    @staticmethod
    def _mla_pool():
        pool = MLATokenToKVPool.__new__(MLATokenToKVPool)
        pool.size = 16
        pool.page_size = 1
        pool.dtype = torch.float32
        pool.store_dtype = torch.float32
        pool.start_layer = 0
        pool.dsa_kv_cache_store_fp8 = False
        pool.kv_buffer = [torch.zeros(17, 1, 1)]
        return pool

    @classmethod
    def _hybrid_pool(cls):
        hybrid_pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
        hybrid_pool.use_mla = True
        hybrid_pool.full_attention_layer_id_mapping = {0: 0}
        hybrid_pool.full_kv_pool = cls._mla_pool()
        return hybrid_pool

    def test_uses_logical_owner_mask_with_physical_locations(self):
        hybrid_pool = self._hybrid_pool()
        logical_loc = torch.arange(16, dtype=torch.int64)
        physical_loc = logical_loc // 8
        dcp_kv_mask = logical_loc % 8 == 3
        cache_k = torch.arange(16, dtype=torch.float32).reshape(16, 1, 1)

        hybrid_pool.set_kv_buffer(
            SimpleNamespace(layer_id=0),
            physical_loc,
            cache_k,
            torch.zeros_like(cache_k),
            dcp_kv_mask=dcp_kv_mask,
        )

        kv_buffer = hybrid_pool.full_kv_pool.kv_buffer[0]
        torch.testing.assert_close(kv_buffer[:2, 0, 0], torch.tensor([3.0, 11.0]))
        torch.testing.assert_close(kv_buffer[2:, 0, 0], torch.zeros(15))

    def test_fp4_pool_applies_explicit_dcp_mask(self):
        pool = MLATokenToKVPoolFP4.__new__(MLATokenToKVPoolFP4)
        pool.size = 16
        pool.page_size = 1
        pool.dtype = torch.float32
        pool.store_dtype = torch.float32
        pool.start_layer = 0
        pool.dsa_kv_cache_store_fp8 = False
        pool.kv_buffer = [torch.zeros(17, 1, 1)]
        logical_loc = torch.arange(16, dtype=torch.int64)
        cache_k = torch.arange(16, dtype=torch.float32).reshape(16, 1, 1)

        pool.set_kv_buffer(
            SimpleNamespace(layer_id=0),
            logical_loc // 8,
            cache_k,
            torch.zeros_like(cache_k),
            dcp_kv_mask=logical_loc % 8 == 3,
        )

        torch.testing.assert_close(
            pool.kv_buffer[0][:2, 0, 0], torch.tensor([3.0, 11.0])
        )
        torch.testing.assert_close(pool.kv_buffer[0][2:, 0, 0], torch.zeros(15))


if __name__ == "__main__":
    unittest.main()
