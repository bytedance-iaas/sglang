"""Exact scoring and overflow regression tests for the opt-in SM90 indexer."""

import pytest
import torch
from test_sm90_fp4_grouped_indexer import TestSm90Fp4GroupedIndexer as GroupedFixture

from sglang.kernels.ops.attention.dsv4.sm90_length_aware_indexer import (
    candidate_blocks,
    prefix_logits,
    select_prefix_topk,
)
from sglang.kernels.ops.attention.dsv4.sm90_paired_indexer import paired_prefix_logits
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires SM90",
)


@pytest.mark.parametrize("width,wide", [(131, False), (32771, True), (65536, False)])
def test_paired_scores_and_per_query_candidates(width, wide):
    x = GroupedFixture().make_inputs(rows=12, width=width, heads=32)
    if wide:
        x.table[:, x.page * 64 :] = torch.randint(
            105, 148, x.table[:, x.page * 64 :].shape, device="cuda", dtype=torch.uint8
        )
    lens = x.lens.int()
    lens[0] = 0
    lens[1] = min(257, width)
    args = (x.q, x.weights, x.mapping, x.req, lens, x.table, x.page, x.ratio, width)
    reference = prefix_logits(*args)
    actual = paired_prefix_logits(*args, group_size=6)
    valid = torch.arange(reference.shape[1], device="cuda")[None, :] < lens[:, None]
    torch.testing.assert_close(actual[valid], reference[valid], atol=0, rtol=0)
    candidates = candidate_blocks(
        reference, lens, width, min(2048, (width + 7) // 8), 8
    )
    compact_lens = torch.minimum(lens, candidates.lengths)
    compact_width = min(width, candidates.blocks.shape[1] * 8)
    args = (
        x.q,
        x.weights,
        x.mapping,
        x.req,
        compact_lens,
        x.table,
        x.page,
        x.ratio,
        compact_width,
    )
    reference = prefix_logits(*args, candidates=candidates, visible=lens)
    actual = paired_prefix_logits(
        *args, candidates=candidates, visible=lens, group_size=6
    )
    valid = (
        torch.arange(reference.shape[1], device="cuda")[None, :] < compact_lens[:, None]
    )
    torch.testing.assert_close(actual[valid], reference[valid], atol=0, rtol=0)


@pytest.mark.parametrize("constant", [False, True])
def test_zero_cache_and_mixed_generated_pages(constant):
    x = GroupedFixture().make_inputs(rows=12, width=8192, heads=32)
    original = x.table.clone()
    x.table.zero_()
    if not constant:
        pages = (x.slots[:, -1000:] // x.page).unique()
        x.table[pages] = original[pages]
    args = (x.q, x.weights, x.mapping, x.req, x.lens, x.table, x.page, x.ratio, x.width)
    expected = prefix_logits(*args)
    actual = paired_prefix_logits(*args, group_size=6)
    valid = torch.arange(x.width, device="cuda")[None, :] < x.lens[:, None]
    torch.testing.assert_close(actual[valid], expected[valid], atol=0, rtol=0)


@pytest.mark.parametrize("width", [8192, 16384, 32768])
@pytest.mark.parametrize("kind", ["finite", "large", "constant"])
def test_topk_coarse_bin_overflow_remains_exact(width, kind):
    if kind == "finite":
        values = (
            1
            + torch.arange(width, device="cuda")
            .div(width // 4, rounding_mode="floor")
            .float()
            / 128
        )
    elif kind == "large":
        values = torch.linspace(1e6, 1e9, width, device="cuda")
    else:
        values = torch.zeros(width, device="cuda")
    scores = values[None, :].clone()
    lens = torch.tensor([width], device="cuda", dtype=torch.int32)
    indices = select_prefix_topk(scores, lens, 512)
    actual = values[indices[0].long()].sort().values
    expected = values.topk(512).values.sort().values
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
