import json
import unittest
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
