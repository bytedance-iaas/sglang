import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
    _preprocess_humming_mxfp4_by_expert,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMxfp4HummingPreprocess(unittest.TestCase):
    def test_preprocesses_one_expert_at_a_time_in_place(self):
        calls = []

        def preprocess(weight, scale):
            calls.append(weight.shape[0])
            expert_id = len(calls)
            return (
                (weight + expert_id).view(1, 1, 2, 2),
                (scale + expert_id).view(1, 1, 1, 2, 1),
                torch.tensor([expert_id]),
            )

        package = types.ModuleType("flashinfer")
        package.__path__ = []
        fused_moe = types.ModuleType("flashinfer.fused_moe")
        fused_moe.preprocess_moe_weights_for_sm90_mixed_gemm_humming = preprocess
        package.fused_moe = fused_moe

        weight = torch.zeros((3, 2, 2), dtype=torch.uint8)
        scale = torch.zeros((3, 2, 1), dtype=torch.uint8)
        weight_ptr = weight.data_ptr()
        scale_ptr = scale.data_ptr()
        with patch.dict(
            sys.modules,
            {"flashinfer": package, "flashinfer.fused_moe": fused_moe},
        ):
            weight_out, scale_out, residual = _preprocess_humming_mxfp4_by_expert(
                weight, scale
            )

        self.assertEqual(calls, [1, 1, 1])
        self.assertEqual(weight_out.data_ptr(), weight_ptr)
        self.assertEqual(scale_out.data_ptr(), scale_ptr)
        self.assertEqual(tuple(weight_out.shape), (3, 1, 2, 2))
        self.assertEqual(tuple(scale_out.shape), (3, 1, 1, 2, 1))
        expected = torch.tensor([1, 2, 3], dtype=torch.uint8)
        torch.testing.assert_close(weight_out[:, 0, 0, 0], expected)
        torch.testing.assert_close(scale_out[:, 0, 0, 0, 0], expected)
        torch.testing.assert_close(residual, torch.tensor([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
