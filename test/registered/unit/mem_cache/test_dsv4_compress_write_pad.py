import unittest

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    get_compress_state_ring_size,
    get_compress_state_write_pad,
)
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCompressStateWritePad(CustomTestCase):
    """Pin the compression ring capacity used by speculative verification."""

    def test_pad_is_zero_without_speculation(self):
        for compress_ratio in (4, 128):
            ring_size = get_compress_state_ring_size(compress_ratio, False)
            with self.subTest(cr=compress_ratio, ring=ring_size):
                self.assertEqual(
                    get_compress_state_write_pad(compress_ratio, ring_size), 0
                )

    def test_pad_matches_speculative_ring_capacity(self):
        for compress_ratio, expected in ((4, 10), (128, 130)):
            ring_size = get_compress_state_ring_size(compress_ratio, True)
            with self.subTest(cr=compress_ratio, ring=ring_size):
                self.assertEqual(
                    get_compress_state_write_pad(compress_ratio, ring_size), expected
                )

    def test_pad_is_zero_for_rings_below_one_window(self):
        self.assertEqual(get_compress_state_write_pad(128, 1), 0)

    def test_startup_capacity_accepts_default_gamma_five(self):
        configurator = DSV4PoolConfigurator.__new__(DSV4PoolConfigurator)
        configurator.c4_ring_size = 16
        configurator.c128_ring_size = 256
        configurator.num_layers_ca4 = 1
        configurator.num_layers_ca128 = 1

        # gamma=5 produces a six-token verify window and fits both rings.
        configurator._assert_ring_serves_draft_tokens(6)

    def test_startup_capacity_rejects_more_than_c4_ring_supports(self):
        configurator = DSV4PoolConfigurator.__new__(DSV4PoolConfigurator)
        configurator.c4_ring_size = 16
        configurator.c128_ring_size = 256
        configurator.num_layers_ca4 = 1
        configurator.num_layers_ca128 = 1

        with self.assertRaisesRegex(AssertionError, "serves at most 10"):
            configurator._assert_ring_serves_draft_tokens(11)


if __name__ == "__main__":
    unittest.main()
