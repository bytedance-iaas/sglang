"""Correctness and graph coverage for the fused KPool top-k v2 route.

The KPool mode selects pooled history groups, expands every selected group to
``pool_size`` token indices, appends the open-pool tail, and applies either a
page-table or additive-offset transform.  The selected group order and the
choice among equal-score boundary ties are unspecified; the group membership,
expansion, tail order, mapping, padding, and graph refresh are not.
"""

from __future__ import annotations

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_kpool_ragged_v2,
    topk_transform_kpool_v2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

POOL_SIZE = 4
TOKEN_TOPK = 2048
GROUP_TOPK = TOKEN_TOPK // POOL_SIZE


def _plan(pool_lens: torch.Tensor) -> torch.Tensor:
    plan = plan_topk_v2(pool_lens)
    # Production builds the plan during metadata preparation, before the model
    # writes scores.  Keep the same launch separation for PDL visibility.
    torch.cuda.synchronize()
    return plan


def _selected_groups(
    scores: torch.Tensor, *, start: int, length: int, topk: int
) -> torch.Tensor:
    if length <= topk:
        return torch.arange(length, dtype=torch.int64)
    return torch.topk(scores[start : start + length].cpu(), topk).indices.to(
        torch.int64
    )


def _assert_row(
    *,
    output: torch.Tensor,
    score_row: torch.Tensor,
    pool_len: int,
    token_seq_len: int,
    group_topk: int,
    row_start: int = 0,
    page_row: torch.Tensor | None = None,
    offset: int = 0,
) -> None:
    history_groups = min(pool_len, group_topk)
    history_cols = history_groups * POOL_SIZE
    tail_count = token_seq_len % POOL_SIZE
    logical = output.clone().to(torch.int64)

    valid = logical >= 0
    if page_row is not None:
        page_row = page_row.cpu().long()
        positions = torch.searchsorted(page_row, logical[valid])
        assert bool((positions < page_row.numel()).all())
        torch.testing.assert_close(page_row[positions], logical[valid], rtol=0, atol=0)
        logical[valid] = positions
    else:
        logical[valid] -= offset

    history = logical[:history_cols]
    if history_cols:
        expanded = history.view(history_groups, POOL_SIZE)
        groups = torch.div(expanded[:, 0], POOL_SIZE, rounding_mode="floor")
        torch.testing.assert_close(
            expanded,
            groups[:, None] * POOL_SIZE + torch.arange(POOL_SIZE),
            rtol=0,
            atol=0,
        )
        assert groups.unique().numel() == history_groups
        expected_groups = _selected_groups(
            score_row, start=row_start, length=pool_len, topk=group_topk
        )
        expected_values = score_row.cpu()[row_start + expected_groups].sort().values
        actual_values = score_row.cpu()[row_start + groups].sort().values
        torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)

    expected_tail = torch.arange(
        pool_len * POOL_SIZE,
        pool_len * POOL_SIZE + tail_count,
        dtype=torch.int64,
    )
    torch.testing.assert_close(
        logical[history_cols : history_cols + tail_count],
        expected_tail,
        rtol=0,
        atol=0,
    )
    assert bool((logical[history_cols + tail_count :] == -1).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
@torch.inference_mode()
def test_kpool_topk_v2_actual_b16_t6_shape_page_mapping() -> None:
    """GLM-5.3-Flash Decode shape: 96 rows and a 65K token history."""
    rows, width = 16 * 6, 65536 // POOL_SIZE
    base = torch.arange(width, dtype=torch.float32, device="cuda")
    scores = base.expand(rows, -1).contiguous()
    scores[1::2].copy_(scores[1::2].flip(1))
    pattern = torch.tensor(
        [0, 1, 511, 512, 513, 8192, width - 1, width],
        dtype=torch.int32,
        device="cuda",
    )
    pool_lens = pattern.repeat(rows // pattern.numel())
    token_seq_lens = pool_lens * POOL_SIZE + torch.arange(
        rows, dtype=torch.int32, device="cuda"
    ).remainder(POOL_SIZE)

    token_width = width * POOL_SIZE + POOL_SIZE - 1
    row_base = torch.arange(rows, dtype=torch.int64, device="cuda") * 100_000
    logical = torch.arange(token_width, dtype=torch.int64, device="cuda")
    page_table = (row_base[:, None] + logical[None, :] * 3).to(torch.int32)
    out = torch.full(
        (rows, TOKEN_TOPK + POOL_SIZE - 1),
        -2,
        dtype=torch.int32,
        device="cuda",
    )

    topk_transform_kpool_v2(
        scores=scores,
        pool_lens=pool_lens,
        out=out,
        pool_size=POOL_SIZE,
        metadata=_plan(pool_lens),
        token_seq_lens=token_seq_lens,
        page_table=page_table,
    )
    torch.cuda.synchronize()

    out_cpu, scores_cpu = out.cpu(), scores.cpu()
    page_cpu = page_table.cpu()
    for row in range(rows):
        _assert_row(
            output=out_cpu[row],
            score_row=scores_cpu[row],
            pool_len=int(pool_lens[row]),
            token_seq_len=int(token_seq_lens[row]),
            group_topk=GROUP_TOPK,
            page_row=page_cpu[row],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
@torch.inference_mode()
def test_kpool_topk_v2_ragged_windows_and_request_page_rows() -> None:
    rows = 8
    starts_host = [0, 1, 2, 3, 64, 67, 4096, 4099]
    lengths_host = [0, 1, 127, 128, 129, 513, 8192, 9000]
    width = (max(s + n for s, n in zip(starts_host, lengths_host)) + 3) & ~3
    scores = torch.full((rows, width), 1e6, dtype=torch.float32, device="cuda")
    for row, (start, length) in enumerate(zip(starts_host, lengths_host)):
        scores[row, start : start + length] = torch.arange(
            length, dtype=torch.float32, device="cuda"
        )
    before = scores.clone()
    starts = torch.tensor(starts_host, dtype=torch.int32, device="cuda")
    pool_lens = torch.tensor(lengths_host, dtype=torch.int32, device="cuda")
    token_seq_lens = pool_lens * POOL_SIZE + torch.arange(
        rows, dtype=torch.int32, device="cuda"
    ).remainder(POOL_SIZE)
    table_rows = 5
    token_width = max(lengths_host) * POOL_SIZE + POOL_SIZE - 1
    table_base = torch.arange(table_rows, dtype=torch.int64, device="cuda") * 100_000
    logical = torch.arange(token_width, dtype=torch.int64, device="cuda")
    page_table = (table_base[:, None] + logical[None, :] * 3).to(torch.int32)
    page_row_index = torch.tensor(
        [4, 0, 3, 1, 4, 2, 0, 3], dtype=torch.int32, device="cuda"
    )
    out = torch.full(
        (rows, TOKEN_TOPK + POOL_SIZE - 1),
        -2,
        dtype=torch.int32,
        device="cuda",
    )

    topk_transform_kpool_ragged_v2(
        scores=scores,
        pool_lens=pool_lens,
        out=out,
        pool_size=POOL_SIZE,
        row_starts=starts,
        token_seq_lens=token_seq_lens,
        page_table=page_table,
        page_table_row_index=page_row_index,
    )
    torch.cuda.synchronize()

    out_cpu, before_cpu = out.cpu(), before.cpu()
    page_cpu, page_rows_cpu = page_table.cpu(), page_row_index.cpu()
    for row in range(rows):
        _assert_row(
            output=out_cpu[row],
            score_row=before_cpu[row],
            row_start=starts_host[row],
            pool_len=lengths_host[row],
            token_seq_len=int(token_seq_lens[row]),
            group_topk=GROUP_TOPK,
            page_row=page_cpu[int(page_rows_cpu[row])],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
@torch.inference_mode()
def test_kpool_topk_v2_ties_and_cuda_graph_dynamic_refresh() -> None:
    rows, width, group_topk = 96, 1024, 128
    token_topk = group_topk * POOL_SIZE
    scores = torch.ones((rows, width), dtype=torch.float32, device="cuda")
    scores[:, width // 2 :] = 2
    pool_lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    token_seq_lens = pool_lens * POOL_SIZE + torch.arange(
        rows, dtype=torch.int32, device="cuda"
    ).remainder(POOL_SIZE)
    offsets = torch.arange(rows, dtype=torch.int32, device="cuda") * 10_000
    out = torch.full(
        (rows, token_topk + POOL_SIZE - 1),
        -2,
        dtype=torch.int32,
        device="cuda",
    )
    plan = _plan(pool_lens)
    pointers = tuple(
        t.data_ptr() for t in (scores, pool_lens, token_seq_lens, out, plan)
    )

    def run() -> None:
        topk_transform_kpool_v2(
            scores=scores,
            pool_lens=pool_lens,
            out=out,
            pool_size=POOL_SIZE,
            metadata=plan,
            token_seq_lens=token_seq_lens,
            out_offsets=offsets,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for reverse in (False, True):
        if reverse:
            scores.copy_(scores.flip(1))
            pool_lens.copy_(
                torch.arange(rows, dtype=torch.int32, device="cuda").remainder(
                    width + 1
                )
            )
            token_seq_lens.copy_(
                pool_lens * POOL_SIZE
                + torch.arange(rows, dtype=torch.int32, device="cuda").remainder(
                    POOL_SIZE
                )
            )
        # Production refreshes metadata before the score producer launches.
        # Refresh this exact owned buffer outside the transform graph, then
        # prove the capture consumes new contents without changing an address.
        plan_topk_v2(pool_lens, out=plan)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert pointers == tuple(
            t.data_ptr() for t in (scores, pool_lens, token_seq_lens, out, plan)
        )
        out_cpu, scores_cpu = out.cpu(), scores.cpu()
        for row in range(rows):
            _assert_row(
                output=out_cpu[row],
                score_row=scores_cpu[row],
                pool_len=int(pool_lens[row]),
                token_seq_len=int(token_seq_lens[row]),
                group_topk=group_topk,
                offset=int(offsets[row]),
            )
