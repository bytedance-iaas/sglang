# SPDX-License-Identifier: Apache-2.0
"""Capability routing for pooled DSA index tails."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKPoolTailBackendResolution(CustomTestCase):
    def setUp(self):
        from sglang.srt.layers.attention.dsa.dsa_backend_kpool import (
            DeepseekSparseAttnBackendKPoolMixin,
        )

        self.backend = DeepseekSparseAttnBackendKPoolMixin()
        self.backend.dsa_index_kpool = 64

    def _resolve(self, impl: str, sm_major: int = 9):
        self.backend.device_sm_major = sm_major
        return self.backend._resolve_kpool_tail_backend(
            topk_indices=object(), dsa_impl=impl
        )

    def test_flashmla_backends_fall_back_to_sparse_tail_kernels(self):
        for impl in ("flashmla_sparse", "flashmla_kv"):
            with self.subTest(impl=impl):
                self.assertEqual(self._resolve(impl, sm_major=9), "fa3")
                self.assertEqual(self._resolve(impl, sm_major=10), "trtllm")

    def test_non_flashmla_backend_is_unchanged(self):
        for impl in ("fa3", "tilelang", "trtllm", "triton"):
            with self.subTest(impl=impl):
                self.assertEqual(self._resolve(impl), impl)

    def test_no_tail_keeps_flashmla_kv(self):
        self.backend.device_sm_major = 9
        self.assertEqual(
            self.backend._resolve_kpool_tail_backend(None, "flashmla_kv"),
            "flashmla_kv",
        )
        self.backend.dsa_index_kpool = 1
        self.assertEqual(self._resolve("flashmla_kv"), "flashmla_kv")

    def test_unhandled_backend_still_fails_closed(self):
        with self.assertRaisesRegex(NotImplementedError, "index_kpool > 1"):
            self.backend._check_kpool_tail_backend(
                topk_indices=object(), dsa_impl="triton", phase="decode"
            )


if __name__ == "__main__":
    unittest.main()
