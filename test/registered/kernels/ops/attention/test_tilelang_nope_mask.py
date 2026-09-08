"""NoPE mask lowering must zero invalid gathers and retain live KPool tails."""

import pytest
import torch

from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not is_sm90_supported(), reason="requires SM90")


@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("topk", [64, 2112])
def test_bf16_nope_masks_invalid_rows_before_gemm(heads, topk):
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

    generator = torch.Generator(device="cuda").manual_seed(809 + heads + topk)
    q = torch.randn(4, heads, 512, generator=generator, device="cuda").bfloat16()
    kv = torch.randn(257, 1, 512, generator=generator, device="cuda").bfloat16()
    kv[0] = float("nan")
    indices = torch.randint(
        1, 257, (4, 1, topk), generator=generator, device="cuda", dtype=torch.int32
    )
    indices[0] = -1
    indices[1, :, 3:] = -1
    indices[2, :, -3:] = 257
    if topk > 2048:
        indices[3, :, 30:2048] = -1
        indices[3, :, 2051:] = -1

    def check(actual):
        ids = indices[:, 0].long()
        valid = (ids >= 0) & (ids < kv.shape[0])
        gathered = kv.float()[ids.clamp(1, kv.shape[0] - 1), 0]
        gathered = torch.where(valid[:, :, None], gathered, 0)
        scores = torch.einsum("qhd,qkd->qhk", q.float(), gathered) * 0.08
        probabilities = (
            scores.masked_fill(~valid[:, None], -torch.inf).softmax(-1).nan_to_num(0)
        )
        expected = torch.einsum("qhk,qkd->qhd", probabilities, gathered)
        assert actual.isfinite().all()
        assert (actual[0] == 0).all()
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)

    def run():
        return tilelang_sparse_fwd(q, kv, indices, 0.08)[0]

    check(run())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = run()
    indices[1] = 256
    kv[256].fill_(0.25)
    graph.replay()
    check(replayed)
