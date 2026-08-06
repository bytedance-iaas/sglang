"""CPU coverage for DeepSeek MLA Breakable CUDA Graph routing."""

import unittest
from unittest.mock import Mock, patch

import sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla as mla_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepseekMlaBreakableCudaGraph(CustomTestCase):
    def test_deepgemm_q_b_proj_uses_noncollective_bcg_break(self):
        attention = Mock(_use_min_latency_q_b_gemm=True)
        q_lora = object()
        expected = object()

        with (
            patch.object(mla_module, "is_in_breakable_cuda_graph", return_value=True),
            patch.object(
                mla_module,
                "bcg_deepgemm_q_b_proj_forward",
                return_value=expected,
            ) as bcg_forward,
        ):
            actual = mla_module._q_b_proj_forward(attention, q_lora)

        self.assertIs(actual, expected)
        bcg_forward.assert_called_once_with(attention, q_lora)
        attention.q_b_proj_forward.assert_not_called()

    def test_plain_q_b_proj_path_is_unchanged(self):
        expected = object()
        attention = Mock(_use_min_latency_q_b_gemm=False)
        attention.q_b_proj_forward.return_value = expected
        q_lora = object()

        with (
            patch.object(mla_module, "is_in_breakable_cuda_graph", return_value=True),
            patch.object(mla_module, "bcg_deepgemm_q_b_proj_forward") as bcg_forward,
        ):
            actual = mla_module._q_b_proj_forward(attention, q_lora)

        self.assertIs(actual, expected)
        attention.q_b_proj_forward.assert_called_once_with(q_lora)
        bcg_forward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
