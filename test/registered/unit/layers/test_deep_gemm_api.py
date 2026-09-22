import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.layers.deep_gemm_wrapper.api import get_masked_fp8_gemm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMaskedGemmCompatibility(unittest.TestCase):
    def test_legacy_preserves_overlap_contract(self):
        legacy, modern = Mock(), Mock()
        module = SimpleNamespace(
            fp8_m_grouped_gemm_nt_masked=legacy,
            m_grouped_fp8_gemm_nt_masked=modern,
        )
        self.assertIs(get_masked_fp8_gemm(module), legacy)
        self.assertIs(get_masked_fp8_gemm(module, require_overlap=True), legacy)

    def test_upstream_ordinary_call_preserves_arguments(self):
        modern = Mock()
        module = SimpleNamespace(m_grouped_fp8_gemm_nt_masked=modern)
        get_masked_fp8_gemm(module)(1, 2, 3, masked_m=4, expected_m=5)
        modern.assert_called_once_with(1, 2, 3, masked_m=4, expected_m=5)

    def test_upstream_cannot_silently_drop_overlap_signal(self):
        modern = Mock()
        module = SimpleNamespace(m_grouped_fp8_gemm_nt_masked=modern)
        with self.assertRaisesRegex(RuntimeError, "signal API"):
            get_masked_fp8_gemm(module, require_overlap=True)
        modern.assert_not_called()


if __name__ == "__main__":
    unittest.main()
