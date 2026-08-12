"""CPU regression tests for request-scoped DSV4 DSpark draft SWA rings."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDsv4DsparkRadixPolicy(CustomTestCase):
    def test_request_scoped_ring_reprefills_tail_without_disabling_radix(self):
        """A request-scoped draft ring rebuilds its tail on a prefix hit."""
        disable, reprefill_tail = kv_cache_builder._resolve_radix_cache_config(
            disable_radix_cache=False,
            disable_for_multimodal_transformers=False,
            uses_request_scoped_draft_swa_ring=True,
            sliding_window_size=128,
        )

        self.assertFalse(disable)
        self.assertEqual(reprefill_tail, 128)

    def test_explicit_disable_still_wins(self):
        disable, reprefill_tail = kv_cache_builder._resolve_radix_cache_config(
            disable_radix_cache=True,
            disable_for_multimodal_transformers=False,
            uses_request_scoped_draft_swa_ring=True,
            sliding_window_size=128,
        )

        self.assertTrue(disable)
        self.assertEqual(reprefill_tail, 128)


class TestUnifiedRadixRequestScopedSwaTail(CustomTestCase):
    @staticmethod
    def _make_cache(*, request_scoped_tail, hicache_tail=0):
        cache = object.__new__(UnifiedRadixCache)
        cache._request_scoped_swa_reprefill_tail_tokens = request_scoped_tail
        cache.cache_controller = object() if hicache_tail else None
        cache.tree_core = SimpleNamespace(has_swa_host_pool=not hicache_tail)
        cache.components = (
            {ComponentType.SWA: SimpleNamespace(sliding_window_size=hicache_tail)}
            if hicache_tail
            else {}
        )
        return cache

    def test_request_scoped_tail_is_exposed_to_prefix_matching(self):
        cache = self._make_cache(request_scoped_tail=128)

        self.assertEqual(cache.swa_reprefill_tail_tokens(), 128)

    def test_larger_hicache_tail_wins(self):
        cache = self._make_cache(request_scoped_tail=128, hicache_tail=256)

        self.assertEqual(cache.swa_reprefill_tail_tokens(), 256)


if __name__ == "__main__":
    unittest.main()
