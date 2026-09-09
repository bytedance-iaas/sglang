"""Native H16 sparse prefill: masking, KPool tails, lengths, sinks and replay."""

import pytest
import torch

from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not is_sm90_supported(), reason="requires SM90")


def _case(topk):
    generator = torch.Generator(device="cuda").manual_seed(38430 + topk)
    q = (torch.randn(7, 16, 512, generator=generator, device="cuda") * 0.5).bfloat16()
    kv = torch.randn(257, 512, generator=generator, device="cuda").bfloat16()
    kv[0] = float("nan")
    indices = torch.randint(
        1, 257, (7, topk), generator=generator, device="cuda", dtype=torch.int32
    )
    indices[0] = -1
    indices[1, 1:] = -1
    indices[2, 3] = 257
    indices[2, 5] = -10
    if topk >= 2112:
        indices[:, 2048:] = -1
        indices[3, 2050] = 256  # Tail after holes remains live.
    lengths = torch.tensor(
        [topk, topk, topk, topk, 0, 33, topk - 1], device="cuda", dtype=torch.int32
    )
    return q, kv, indices, lengths


def _reference(q, kv, indices, lengths, sink, scale):
    ids = indices.long()
    valid = (ids >= 0) & (ids < kv.shape[0])
    valid &= torch.arange(ids.shape[-1], device=q.device)[None] < lengths[:, None]
    gathered = kv.float()[ids.clamp(1, kv.shape[0] - 1)]
    gathered = torch.where(valid[:, :, None], gathered, 0)
    scores = torch.einsum("qhd,qkd->qhk", q.float(), gathered) * scale
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    sink_scores = sink[None, :, None].expand(q.shape[0], -1, -1)
    probs = torch.cat((scores, sink_scores), -1).softmax(-1)[:, :, :-1]
    return torch.einsum("qhk,qkd->qhd", probs, gathered)


@pytest.mark.parametrize("topk", [64, 128, 2112, 2176])
@pytest.mark.parametrize("with_sink", [False, True])
def test_native_h16_sparse_boundaries(topk, with_sink):
    from sglang.kernels.ops.attention.sparse_mla_h16_sm90 import sparse_mla_h16_fwd

    q, kv, indices, lengths = _case(topk)
    sink = (
        torch.linspace(-1, 1, 16, device="cuda")
        if with_sink
        else torch.full((16,), -1e30, device="cuda")
    )
    actual = sparse_mla_h16_fwd(q, kv, indices, lengths, sink, 0.08)
    expected = _reference(q, kv, indices, lengths, sink, 0.08)
    assert actual.isfinite().all()
    assert (actual[[0, 4]] == 0).all()
    torch.testing.assert_close(actual.float(), expected, atol=0.012, rtol=0.02)
    assert (actual.float() - expected).norm() / expected.norm() < 0.006


def test_native_h16_graph_replay():
    from sglang.kernels.ops.attention.sparse_mla_h16_sm90 import sparse_mla_h16_fwd

    q, kv, indices, lengths = _case(2112)
    sink = torch.full((16,), -1e30, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            sparse_mla_h16_fwd(q, kv, indices, lengths, sink, 0.08)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = sparse_mla_h16_fwd(q, kv, indices, lengths, sink, 0.08)
    indices[0, 2050] = 256
    lengths[5] = 2112
    kv[256].fill_(0.25)
    graph.replay()
    expected = _reference(q, kv, indices, lengths, sink, 0.08)
    torch.testing.assert_close(actual.float(), expected, atol=0.012, rtol=0.02)
    assert (actual[4] == 0).all()


def test_native_h16_rejects_real_h64_and_handles_empty_batch():
    from sglang.kernels.ops.attention.sparse_mla_h16_sm90 import sparse_mla_h16_fwd

    q, kv, indices, lengths = _case(64)
    sink = torch.full((16,), -1e30, device="cuda")
    with pytest.raises(ValueError, match="16, 512"):
        sparse_mla_h16_fwd(q.repeat(1, 4, 1), kv, indices, lengths, sink, 0.08)
    out = sparse_mla_h16_fwd(q[:0], kv, indices[:0], lengths[:0], sink, 0.08)
    assert out.shape == (0, 16, 512)
