import unittest

import torch

from sglang.kernels.ops.attention.dsa.sentinel_pad import pad_kpool_indices
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=3, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestKPoolSentinelPad(unittest.TestCase):
    def test_exact_fresh_graph_rows_and_tail(self):
        for rows, width in [(96, 2051), (16, 2048), (1, 2049), (0, 2051)]:
            with self.subTest(rows=rows, width=width):
                indices = torch.randint(
                    -1, 200000, (rows, width), dtype=torch.int32, device="cuda"
                )
                padded = (width + 63) // 64 * 64
                output = torch.empty((rows, padded), dtype=torch.int32, device="cuda")
                pad_kpool_indices(indices, output)
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(
                        output,
                        torch.nn.functional.pad(indices, (0, padded - width), value=-1),
                    )
                )
                if not rows:
                    continue
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    pad_kpool_indices(indices, output)
                for tail in range(4):
                    indices.random_(-1, 200000)
                    indices[0].fill_(-1)
                    if width > 2048:
                        indices[:, 2048:].fill_(-1)
                        if tail:
                            indices[:, 2048 : 2048 + min(tail, width - 2048)] = 37
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(
                        torch.equal(
                            output,
                            torch.nn.functional.pad(
                                indices, (0, padded - width), value=-1
                            ),
                        )
                    )


if __name__ == "__main__":
    unittest.main()
