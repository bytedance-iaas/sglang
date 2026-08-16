import ast
import inspect
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.sparsity.factory import parse_hisparse_config
from sglang.srt.model_executor.model_runner import ModelRunner
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

    def test_model_runner_wires_hisparse_runtime_configuration(self):
        tree = ast.parse(inspect.getsource(ModelRunner.maybe_init_hisparse_coordinator))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "HiSparseCoordinator"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg for keyword in calls[0].keywords}
        self.assertTrue(
            {
                "max_num_steps",
                "mem_pool_device_override",
                "dynamic_residency",
                "dynamic_residency_mode",
                "dynamic_residency_max_tokens",
                "dynamic_residency_max_requests",
                "dynamic_residency_min_remaining_tokens",
                "dynamic_residency_promote_watermark",
                "dynamic_residency_demote_watermark",
                "dynamic_residency_cooldown_steps",
                "dynamic_residency_admission_window_seconds",
            }.issubset(keywords)
        )

    def test_coordinator_uses_owned_kv_container(self):
        tree = ast.parse(inspect.getsource(HiSparseCoordinator))
        legacy_accesses = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "kv_allocated_len"
            and isinstance(node.value, ast.Name)
            and node.value.id in {"req", "r"}
        ]
        self.assertEqual(legacy_accesses, [])

    def test_direct_admission_commits_owner_state_before_draft_mirror(self):
        coordinator = SimpleNamespace(
            debug_validate_lifecycle=False,
            device_buffer_size=8,
            req_device_buffer_tokens=torch.zeros((1, 1, 8), dtype=torch.int32),
            active_hisparse_reqs={},
            _skip_first_backup=[False],
            _try_promote_from_host=Mock(return_value=False),
            _device_buffer_alloc_size=Mock(return_value=8),
            demote_until_hisparse_available=Mock(return_value=True),
            alloc_device_buffer=Mock(),
            host_token_len=Mock(return_value=16),
            _preload_to_device_buffer=Mock(),
            _state=Mock(return_value=SimpleNamespace(value="device_buffered")),
        )
        req = SimpleNamespace(
            rid="direct-admission",
            req_pool_idx=0,
            kv=SimpleNamespace(kv_allocated_len=16),
            hisparse_staging=True,
        )

        HiSparseCoordinator.admit_request_direct(coordinator, req)

        coordinator._try_promote_from_host.assert_called_once_with(
            req, sync_mirrors=False, admission_boundary=True
        )
        coordinator.demote_until_hisparse_available.assert_called_once_with(8)
        coordinator.alloc_device_buffer.assert_called_once_with(req)
        coordinator._preload_to_device_buffer.assert_not_called()
        self.assertTrue(torch.all(coordinator.req_device_buffer_tokens == -1))
        self.assertFalse(req.hisparse_staging)
        self.assertTrue(coordinator._skip_first_backup[0])
        self.assertIs(coordinator.active_hisparse_reqs[0], req)


if __name__ == "__main__":
    unittest.main()
