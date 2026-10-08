# SPDX-License-Identifier: Apache-2.0
"""Capability routing for pooled DSA index tails."""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.dsa.dsa_backend_kpool import (
    DeepseekSparseAttnBackendKPoolMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _backend(*, sm=9, fp8=True, rope=0, kpool=4):
    return SimpleNamespace(
        device_sm_major=sm,
        dsa_kv_cache_store_fp8=fp8,
        qk_rope_head_dim=rope,
        dsa_index_kpool=kpool,
    )


class TestKPoolTailBackendResolution(CustomTestCase):
    def _resolve(self, backend, impl, topk=object()):
        return DeepseekSparseAttnBackendKPoolMixin._resolve_kpool_tail_backend(
            backend, topk, impl
        )

    def test_hopper_nope_fp8_flashmla_kv_uses_tilelang_tail_reader(self):
        self.assertEqual(self._resolve(_backend(), "flashmla_kv"), "tilelang")

    def test_flashmla_sparse_keeps_existing_arch_fallbacks(self):
        self.assertEqual(self._resolve(_backend(sm=9), "flashmla_sparse"), "fa3")
        self.assertEqual(self._resolve(_backend(sm=10), "flashmla_sparse"), "trtllm")

    def test_blackwell_flashmla_kv_keeps_trtllm_fallback(self):
        self.assertEqual(self._resolve(_backend(sm=10), "flashmla_kv"), "trtllm")

    def test_hopper_fallback_requires_exact_nope_fp8_layout(self):
        for backend in (_backend(fp8=False), _backend(rope=64)):
            with self.subTest(backend=backend):
                self.assertEqual(self._resolve(backend, "flashmla_kv"), "flashmla_kv")

    def test_no_tail_keeps_flashmla_kv(self):
        self.assertEqual(self._resolve(_backend(), "flashmla_kv", topk=None), "flashmla_kv")
        self.assertEqual(self._resolve(_backend(kpool=1), "flashmla_kv"), "flashmla_kv")

    def test_unhandled_backend_still_fails_closed(self):
        with self.assertRaisesRegex(NotImplementedError, "index_kpool > 1"):
            DeepseekSparseAttnBackendKPoolMixin._check_kpool_tail_backend(
                _backend(), object(), "triton", "decode"
            )


if __name__ == "__main__":
    unittest.main()
