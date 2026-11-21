import pytest
import torch

from sglang.kernels.ops.attention.dsa.litetopk_prefill import (
    fp8_indexer_topk_recovery,
    is_fp8_recovery_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-b200")

TOPK = 512


def make_inputs(rows: int, keys: int, *, heads: int = 32, seed: int = 7):
    torch.manual_seed(seed)
    q = torch.randn(rows, heads, 128, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(keys, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.ones(keys, device="cuda")
    weights = torch.randn(rows, heads, device="cuda") / heads
    return q, k, scales, weights


def reference_logits(q, k, scales, weights, starts, ends):
    logits = torch.zeros(q.shape[0], k.shape[0], device=q.device)
    kf = k.float()
    for head in range(q.shape[1]):
        logits += weights[:, head, None] * torch.relu(q[:, head].float() @ kf.T)
    logits *= scales[None]
    key = torch.arange(k.shape[0], device=q.device)
    invalid = (key[None] < starts[:, None]) | (key[None] >= ends[:, None])
    return logits.masked_fill(invalid, -torch.inf)


def check_output(out, q, k, scales, weights, starts, ends):
    visible = (ends - starts).clamp(min=0)
    valid = out >= 0
    assert torch.equal(valid.sum(1), visible.clamp(max=out.shape[1]))
    absolute = out.long() + starts[:, None].long()
    assert (
        (~valid) | ((absolute >= starts[:, None]) & (absolute < ends[:, None]))
    ).all()
    sorted_ids = absolute.masked_fill(~valid, -1).sort().values
    assert ((sorted_ids[:, 1:] < 0) | (sorted_ids.diff(dim=1) > 0)).all()

    logits = reference_logits(q, k, scales, weights, starts, ends)
    selected = logits.gather(1, absolute.clamp(min=0)).masked_fill(~valid, -torch.inf)
    expected = logits.topk(out.shape[1], dim=1).values
    torch.testing.assert_close(
        selected.sort(descending=True).values,
        expected,
        rtol=2e-4,
        atol=1e-4,
    )


@pytest.mark.skipif(not is_fp8_recovery_supported(), reason="SM100 only")
@pytest.mark.parametrize("heads", [32, 64])
def test_packed_causal_rows_match_reference(heads):
    lens = [5000, 7000]
    q_lens = [16, 16]
    q, k, scales, weights = make_inputs(sum(q_lens), sum(lens), heads=heads)
    starts, ends, base = [], [], 0
    for length, rows in zip(lens, q_lens):
        starts.extend([base] * rows)
        ends.extend(range(base + length - rows + 1, base + length + 1))
        base += length
    starts = torch.tensor(starts, dtype=torch.int32, device="cuda")
    ends = torch.tensor(ends, dtype=torch.int32, device="cuda")
    out = fp8_indexer_topk_recovery(q, weights, k, scales, starts, ends, TOPK)
    check_output(out, q, k, scales, weights, starts, ends)


@pytest.mark.skipif(not is_fp8_recovery_supported(), reason="SM100 only")
def test_candidate_overflow_and_wide_ties_recover_exactly():
    rows, keys = 8, 16384
    q, k, scales, weights = make_inputs(rows, keys)
    weights.zero_()
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), keys, dtype=torch.int32, device="cuda")
    out, state = fp8_indexer_topk_recovery(
        q,
        weights,
        k,
        scales,
        starts,
        ends,
        TOPK,
        return_state=True,
    )
    torch.cuda.synchronize()
    counts, _, overflow = state
    assert (counts > 12288).all()
    assert (overflow != 0).all()
    expected = torch.arange(TOPK, dtype=torch.int32, device="cuda").expand_as(out)
    assert torch.equal(out, expected)


@pytest.mark.skipif(not is_fp8_recovery_supported(), reason="SM100 only")
def test_cuda_graph_replay():
    rows, keys = 8, 8192
    q, k, scales, weights = make_inputs(rows, keys)
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), keys, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, TOPK), dtype=torch.int32, device="cuda")

    def run():
        fp8_indexer_topk_recovery(q, weights, k, scales, starts, ends, TOPK, out=out)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    out.fill_(-7)
    graph.replay()
    torch.cuda.synchronize()
    check_output(out, q, k, scales, weights, starts, ends)


@pytest.mark.skipif(not is_fp8_recovery_supported(), reason="SM100 only")
def test_independent_stream_workspaces():
    rows, keys = 8, 8192
    q1, k1, scales1, weights1 = make_inputs(rows, keys, seed=11)
    q2, k2, scales2, weights2 = make_inputs(rows, keys, seed=13)
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), keys, dtype=torch.int32, device="cuda")
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    stream1.wait_stream(torch.cuda.current_stream())
    stream2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream1):
        out1 = fp8_indexer_topk_recovery(q1, weights1, k1, scales1, starts, ends, TOPK)
    with torch.cuda.stream(stream2):
        out2 = fp8_indexer_topk_recovery(q2, weights2, k2, scales2, starts, ends, TOPK)
    torch.cuda.current_stream().wait_stream(stream1)
    torch.cuda.current_stream().wait_stream(stream2)
    check_output(out1, q1, k1, scales1, weights1, starts, ends)
    check_output(out2, q2, k2, scales2, weights2, starts, ends)
