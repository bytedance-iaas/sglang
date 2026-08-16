import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.sparsity.factory import parse_hisparse_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiSparseConfig(unittest.TestCase):
    def _parse(self, config=None):
        return parse_hisparse_config(
            SimpleNamespace(
                hisparse_config=None if config is None else json.dumps(config)
            )
        )

    def test_dynamic_residency_is_opt_in(self):
        config = self._parse()
        self.assertFalse(config.dynamic_residency)
        self.assertEqual(config.dynamic_residency_mode, "adaptive")
        self.assertEqual(config.dynamic_residency_max_tokens, 32768)
        self.assertEqual(config.dynamic_residency_max_requests, 1)
        self.assertEqual(config.dynamic_residency_min_remaining_tokens, 256)
        self.assertEqual(config.dynamic_residency_promote_watermark, 0.20)
        self.assertEqual(config.dynamic_residency_demote_watermark, 0.10)
        self.assertEqual(config.dynamic_residency_cooldown_steps, 16)
        self.assertEqual(config.dynamic_residency_admission_window_seconds, 1800)

    def test_dynamic_residency_configuration(self):
        config = self._parse(
            {
                "dynamic_residency": True,
                "dynamic_residency_mode": "admission_window",
                "dynamic_residency_max_tokens": 16384,
                "dynamic_residency_max_requests": 2,
                "dynamic_residency_min_remaining_tokens": 128,
                "dynamic_residency_promote_watermark": 0.30,
                "dynamic_residency_demote_watermark": 0.15,
                "dynamic_residency_cooldown_steps": 32,
                "dynamic_residency_admission_window_seconds": 900,
            }
        )
        self.assertTrue(config.dynamic_residency)
        self.assertEqual(config.dynamic_residency_mode, "admission_window")
        self.assertEqual(config.dynamic_residency_max_tokens, 16384)
        self.assertEqual(config.dynamic_residency_max_requests, 2)
        self.assertEqual(config.dynamic_residency_min_remaining_tokens, 128)
        self.assertEqual(config.dynamic_residency_promote_watermark, 0.30)
        self.assertEqual(config.dynamic_residency_demote_watermark, 0.15)
        self.assertEqual(config.dynamic_residency_cooldown_steps, 32)
        self.assertEqual(config.dynamic_residency_admission_window_seconds, 900)

    def test_dynamic_residency_rejects_invalid_values(self):
        invalid_configs = [
            {"dynamic_residency": 1},
            {"dynamic_residency_mode": "oscillate_forever"},
            {"dynamic_residency_max_tokens": 0},
            {"dynamic_residency_max_requests": 0},
            {"dynamic_residency_min_remaining_tokens": -1},
            {"dynamic_residency_cooldown_steps": -1},
            {"dynamic_residency_admission_window_seconds": -1},
            {"dynamic_residency_promote_watermark": 1.1},
            {"dynamic_residency_demote_watermark": -0.1},
            {
                "dynamic_residency_promote_watermark": 0.1,
                "dynamic_residency_demote_watermark": 0.1,
            },
        ]
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self._parse(config)

    def test_draft_pool_reuses_target_hisparse_mapping(self):
        configurator = SimpleNamespace(
            is_draft_worker=True,
            is_hybrid_swa=False,
        )
        draft_pool = object.__new__(HiSparseDSATokenToKVPool)
        target_allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        mapping = torch.tensor([0, 7, -1], dtype=torch.int64)
        target_allocator.full_to_hisparse_device_index_mapping = mapping

        with (
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_memory",
                return_value=SimpleNamespace(enable_hisparse=True),
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="decode"),
            ),
        ):
            result = KVCacheConfigurator._build_token_to_kv_pool_allocator(
                configurator,
                sizes=SimpleNamespace(),
                token_to_kv_pool=draft_pool,
                is_dsv4_model=False,
                req_to_token_pool=SimpleNamespace(),
                token_to_kv_pool_allocator=target_allocator,
            )

        self.assertIs(result, target_allocator)
        self.assertIs(draft_pool.full_to_hisparse_device_index_mapping, mapping)


if __name__ == "__main__":
    unittest.main()
