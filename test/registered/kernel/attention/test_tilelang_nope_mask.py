"""BF16 NoPE attention must ignore invalid indices and zero empty rows."""

import pytest
import torch

from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not is_sm90_supported(), reason="requires SM90")


@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("topk", [64, 2112])
@pytest.mark.parametrize("invalid_indices", [False, True], ids=["valid", "invalid"])
def test_bf16_nope_mask(heads, topk, invalid_indices):
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

    generator = torch.Generator(device="cuda").manual_seed(809 + heads + topk)
    q = torch.randn(6, heads, 512, generator=generator, device="cuda").bfloat16()
    kv = torch.randn(257, 1, 512, generator=generator, device="cuda").bfloat16()
    indices = torch.randint(
        1, 257, (6, 1, topk), generator=generator, device="cuda", dtype=torch.int32
    )
    if invalid_indices:
        # Invalid gathers must be zero-filled, even when reserved row 0 is NaN.
        kv[0] = float("nan")
        indices[0] = -1
        indices[1, :, 3:] = -1
        indices[2] = kv.shape[0]
        indices[3, :, ::3] = -2
        indices[3, :, 1::3] = kv.shape[0] + 17
        if topk > 64:
            # An empty block must not discard valid entries in later blocks.
            indices[5, :, :2048] = -1
            indices[5, :, 2051:] = -1
    else:
        # Both ends of the valid index domain remain usable.
        indices[0] = 0
        indices[1] = kv.shape[0] - 1

    def check(actual):
        ids = indices[:, 0].long()
        valid = (ids >= 0) & (ids < kv.shape[0])
        gathered = kv.float()[ids.clamp(0, kv.shape[0] - 1), 0]
        gathered = torch.where(valid[:, :, None], gathered, 0)
        scores = torch.einsum("qhd,qkd->qhk", q.float(), gathered) * 0.08
        scores = scores.masked_fill(~valid[:, None], -torch.inf)
        # Avoid an all -inf softmax in the reference for empty rows.
        empty = ~valid.any(dim=-1)
        scores[empty] = 0
        probabilities = scores.softmax(-1).masked_fill(~valid[:, None], 0)
        expected = torch.einsum("qhk,qkd->qhd", probabilities, gathered)
        assert actual.isfinite().all()
        assert (actual[empty] == 0).all()
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

    # Replay must consume the updated indices and KV contents.
    indices[0] = kv.shape[0] - 1
    kv[-1].fill_(0.25)
    if invalid_indices:
        indices[4] = -1
    graph.replay()
    check(replayed)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
