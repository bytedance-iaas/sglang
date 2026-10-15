"""SM90 launch tuning must preserve the complete mHC result bit for bit."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.layernorm import mhc
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "SM90 CUDA required",
)
class TestMhcSm90Launch(CustomTestCase):
    def test_graph_and_batch_invariance(self):
        torch.manual_seed(1431)
        for rows in (8, 95, 96, 100, 108, 114, 120, 128, 132, 160, 180, 181, 192, 256):
            for strided in (False, True):
                with self.subTest(rows=rows, strided=strided):
                    self.check_case(rows, strided)

    def check_case(self, rows, strided):
        pitch = 20480 + (16 if strided else 0)
        x = torch.randn(rows, pitch, device="cuda", dtype=torch.bfloat16)[:, :20480]
        w = torch.randn(24, pitch, device="cuda")[:, :20480]
        scale = torch.tensor([0.7, 1.1, 0.3], device="cuda")
        base = torch.randn(24, device="cuda")

        def run(value=x, optimized=True):
            return mhc.hc_mix_stats_sinkhorn(
                value,
                w,
                scale,
                base,
                4,
                20,
                1e-6,
                1e-6,
                sm90_decode=optimized,
            )

        # The reference uses the original four-warp, one-stage launch.
        with patch.object(mhc, "_num_stages_for", return_value=1):
            reference = run(optimized=False)
            actual = run()
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run()
            for magnitude in (0.0, 0.01, 1.0, 10.0):
                x.copy_(torch.randn_like(x) * magnitude)
                w.copy_(torch.randn_like(w) / 20480**0.5)
                base.normal_()
                graph.replay()
                reference = run(optimized=False)
                alone = run(x[:1], optimized=False)
                for a, b, c in zip(captured, reference, alone):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                    torch.testing.assert_close(a[:1], c, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
