"""Regression tests for speculative DSV4 compression-ring write planning."""

from __future__ import annotations

import pytest
import torch

from sglang.jit_kernel.benchmark.bench_activation import register_cuda_ci
from sglang.jit_kernel.tests.deepseek_v4.common import (
    make_paged_context,
    to_seq_extend,
)

register_cuda_ci(est_time=20, suite="base-b-kernel-unit-1-gpu-large")

C4_RING_SIZE = 16
C128_RING_SIZE = 256
C4_RING_SIZE_NO_SPEC = 8
C128_RING_SIZE_NO_SPEC = 128


def _window_size(compress_ratio: int) -> int:
    return compress_ratio * (2 if compress_ratio == 4 else 1)


def _max_draft_tokens(compress_ratio: int, ring_size: int) -> int:
    window_size = _window_size(compress_ratio)
    return ring_size - window_size + 2 if ring_size > window_size else 0


def _written_positions(plan_w: torch.Tensor, prefix_len: int) -> set[int]:
    words = plan_w.cpu().view(torch.uint32).view(-1, 2)
    ragged_ids = words[:, 0]
    valid = ragged_ids != 0xFFFFFFFF
    return {prefix_len + int(r) for r in ragged_ids[valid]}


def _make_plan_positions(
    *,
    compress_ratio: int,
    ring_size: int,
    prefix_len: int,
    num_draft_tokens: int,
    on_gpu: bool = False,
) -> set[int]:
    ctx = make_paged_context(
        bs=1, compress_ratio=compress_ratio, ring_size=ring_size
    )
    seq_lens, extend_lens, num_q = to_seq_extend(
        [(prefix_len + num_draft_tokens, num_draft_tokens)]
    )
    if on_gpu:
        seq_lens = seq_lens.to(ctx.req_to_token.device)
        extend_lens = extend_lens.to(ctx.req_to_token.device)
    plan = ctx.make_prefill_plan(seq_lens, extend_lens, num_q)
    return _written_positions(plan.plan_w, prefix_len)


@pytest.mark.parametrize(
    "compress_ratio,ring_size", ((4, C4_RING_SIZE), (128, C128_RING_SIZE))
)
def test_speculative_ring_keeps_every_supported_draft_token(
    compress_ratio: int, ring_size: int
):
    max_draft_tokens = _max_draft_tokens(compress_ratio, ring_size)
    draft_counts = sorted(
        {1, 2, 3, 4, 5, max_draft_tokens - 1, max_draft_tokens}
        & set(range(1, max_draft_tokens + 1))
    )
    for num_draft_tokens in draft_counts:
        for prefix_len in (512, 513, 515):
            written = _make_plan_positions(
                compress_ratio=compress_ratio,
                ring_size=ring_size,
                prefix_len=prefix_len,
                num_draft_tokens=num_draft_tokens,
            )
            expected = set(range(prefix_len, prefix_len + num_draft_tokens))
            assert expected <= written


@pytest.mark.parametrize(
    "compress_ratio,ring_size", ((4, C4_RING_SIZE), (128, C128_RING_SIZE))
)
def test_cpu_and_gpu_planners_have_identical_write_sets(
    compress_ratio: int, ring_size: int
):
    max_draft_tokens = _max_draft_tokens(compress_ratio, ring_size)
    for num_draft_tokens in (1, 4, max_draft_tokens):
        for prefix_len in (512, 513, 515):
            kwargs = dict(
                compress_ratio=compress_ratio,
                ring_size=ring_size,
                prefix_len=prefix_len,
                num_draft_tokens=num_draft_tokens,
            )
            assert _make_plan_positions(**kwargs, on_gpu=False) == (
                _make_plan_positions(**kwargs, on_gpu=True)
            )


@pytest.mark.parametrize(
    "compress_ratio,ring_size",
    ((4, C4_RING_SIZE_NO_SPEC), (128, C128_RING_SIZE_NO_SPEC)),
)
def test_plain_prefill_write_set_is_unchanged(
    compress_ratio: int, ring_size: int
):
    assert _max_draft_tokens(compress_ratio, ring_size) == 0
    is_overlap = compress_ratio == 4
    for seq_len in (512, 600, 777):
        ctx = make_paged_context(
            bs=1, compress_ratio=compress_ratio, ring_size=ring_size
        )
        seq_lens, extend_lens, num_q = to_seq_extend([(seq_len, seq_len)])
        plan = ctx.make_prefill_plan(seq_lens, extend_lens, num_q)
        written = _written_positions(plan.plan_w, 0)

        last_c_pos = seq_len // compress_ratio * compress_ratio
        first_w_pos = last_c_pos - (compress_ratio if is_overlap else 0)
        expected = {
            pos
            for pos in range(seq_len)
            if pos >= first_w_pos
            or (
                is_overlap
                and pos % ctx.swa_page_size
                >= ctx.swa_page_size - compress_ratio
            )
        }
        assert written == expected


@pytest.mark.parametrize(
    "compress_ratio,ring_size", ((4, C4_RING_SIZE), (128, C128_RING_SIZE))
)
def test_over_capacity_is_detectable_by_startup_bound(
    compress_ratio: int, ring_size: int
):
    too_many = _max_draft_tokens(compress_ratio, ring_size) + compress_ratio + 1
    prefix_len = 512
    written = _make_plan_positions(
        compress_ratio=compress_ratio,
        ring_size=ring_size,
        prefix_len=prefix_len,
        num_draft_tokens=too_many,
    )
    expected = set(range(prefix_len, prefix_len + too_many))
    assert expected - written
