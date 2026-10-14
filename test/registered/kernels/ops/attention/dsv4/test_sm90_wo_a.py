"""H20 O-A projection: FP32 reference, graph updates, and model dispatch."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.sm90_wo_a import wo_a_bf16_sm90
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "requires an SM90 GPU",
)
class TestSm90WoA(CustomTestCase):
    def test_numerics_and_graph(self):
        torch.manual_seed(13)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for rows in (96, 120, 127, 144, 159, 160, 180, 192):
                x = torch.randn(rows, 8, 4096, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(8, 1024, 4096, device="cuda", dtype=torch.bfloat16) / 64
                for _ in range(3):
                    wo_a_bf16_sm90(x, w)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = wo_a_bf16_sm90(x, w)
                for scale in (1.0, 0.001, 64.0):
                    x.normal_().mul_(scale)
                    w.normal_().div_(64)
                    graph.replay()
                    expected = torch.einsum("tgd,grd->tgr", x.float(), w.float())
                    torch.testing.assert_close(
                        actual.float(), expected, atol=2e-5 * scale, rtol=0.004
                    )
                    self.assertTrue(actual.is_contiguous())
                graph.reset()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    def test_model_dispatch(self):
        from sglang.srt.models.deepseek_v4 import _apply_wo_a_bf16_matmul

        x = torch.randn(193, 8, 4096, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(8, 1024, 4096, device="cuda", dtype=torch.bfloat16) / 64
        for enabled, verify, fast, rows, expected_calls in (
            (True, True, True, 96, 1),
            (True, True, True, 120, 1),
            (True, True, True, 159, 1),
            (True, True, True, 160, 1),
            (True, True, True, 180, 1),
            (True, True, True, 192, 1),
            (True, True, True, 95, 0),
            (True, True, True, 193, 0),
            (False, True, True, 120, 0),
            (True, False, True, 120, 0),
            (True, True, False, 120, 0),
            (False, True, True, 180, 0),
            (True, False, True, 180, 0),
            (True, True, False, 180, 0),
            (True, True, True, 6, 0),
        ):
            with (
                envs.SGLANG_OPT_DSV41_SM90_WO_A.override(enabled),
                patch(
                    "sglang.kernels.ops.attention.dsv4.sm90_wo_a.wo_a_bf16_sm90",
                    wraps=wo_a_bf16_sm90,
                ) as kernel,
            ):
                actual = _apply_wo_a_bf16_matmul(
                    x[:rows],
                    w,
                    is_decode=False,
                    is_target_verify=verify,
                    fast_path=fast,
                )
                self.assertEqual(kernel.call_count, expected_calls)
                expected = torch.einsum("tgd,grd->tgr", x[:rows], w)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0.008)


if __name__ == "__main__":
    unittest.main()
