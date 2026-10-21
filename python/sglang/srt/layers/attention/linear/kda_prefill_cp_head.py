# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");

"""Head-owned KDA prefill helpers for interleave context parallelism.

The existing correctness bridge gathers every CP token shard and runs the same
head-sharded recurrence on every CP rank.  For CP2, this module exchanges half
of the local heads so each CP rank instead runs one H/2 shard over the full
global token order.  The inverse exchange restores local interleaved rows and
the original H-head layout for the output projection.

Only pure layout and state-pool helpers live here.  Runtime selection remains a
default-off, model-specific decision in ``KDAAttnBackend``.
"""

from typing import NamedTuple, Optional, Sequence

import torch
from sglang.srt.runtime_context import get_parallel


class KdaPrefillCpHeadPlan(NamedTuple):
    cp_size: int
    cp_rank: int
    local_rows: int
    global_rows: int
    local_heads: int
    owned_heads: int
    head_start: int
    head_end: int


def plan_kda_prefill_cp_head(
    *,
    local_rows: int,
    local_heads: int,
    logical_rank_rows: Sequence[int],
    physical_rank_rows: Sequence[int],
    cp_size: int,
    cp_rank: int,
) -> Optional[KdaPrefillCpHeadPlan]:
    """Return the narrow CP2 plan or ``None`` for the common bridge fallback.

    Equal logical and physical row counts are required so every rank makes the
    same decision without a device synchronization and all-to-all split sizes
    stay identical.  Uneven, padded, empty, or non-pair batches keep the
    existing all-gather bridge.
    """

    if cp_size != 2 or local_rows <= 0 or local_heads <= 0:
        return None
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"CP rank {cp_rank} outside [0, {cp_size})")
    if local_heads % cp_size:
        return None
    if len(logical_rank_rows) != cp_size or len(physical_rank_rows) != cp_size:
        return None
    logical = tuple(int(value) for value in logical_rank_rows)
    physical = tuple(int(value) for value in physical_rank_rows)
    if any(value != local_rows for value in logical):
        return None
    if physical != logical:
        return None
    owned_heads = local_heads // cp_size
    head_start = cp_rank * owned_heads
    return KdaPrefillCpHeadPlan(
        cp_size=cp_size,
        cp_rank=cp_rank,
        local_rows=local_rows,
        global_rows=local_rows * cp_size,
        local_heads=local_heads,
        owned_heads=owned_heads,
        head_start=head_start,
        head_end=head_start + owned_heads,
    )


def pack_local_rows_for_head_owners(
    x: torch.Tensor, plan: KdaPrefillCpHeadPlan
) -> torch.Tensor:
    """Pack ``[local_rows, H, ...]`` as ``[dst_cp, local_rows, H/2, ...]``."""

    if x.ndim < 2 or x.shape[:2] != (plan.local_rows, plan.local_heads):
        raise ValueError(
            f"expected [local_rows, local_heads, ...] = "
            f"[{plan.local_rows}, {plan.local_heads}, ...], got {tuple(x.shape)}"
        )
    return (
        x.reshape(plan.local_rows, plan.cp_size, plan.owned_heads, *x.shape[2:])
        .movedim(1, 0)
        .contiguous()
    )


def merge_token_shards_for_head_owner(
    received: torch.Tensor, plan: KdaPrefillCpHeadPlan
) -> torch.Tensor:
    """Restore global interleave order from received CP token shards."""

    if received.ndim < 3 or received.shape[:3] != (
        plan.cp_size,
        plan.local_rows,
        plan.owned_heads,
    ):
        raise ValueError(
            "expected received [cp, local_rows, owned_heads, ...], got "
            f"{tuple(received.shape)}"
        )
    return received.transpose(0, 1).reshape(
        plan.global_rows, plan.owned_heads, *received.shape[3:]
    )


def pack_global_rows_for_token_owners(
    x: torch.Tensor, plan: KdaPrefillCpHeadPlan
) -> torch.Tensor:
    """Pack global interleave rows as ``[dst_cp, local_rows, H/2, ...]``."""

    if x.ndim < 2 or x.shape[:2] != (plan.global_rows, plan.owned_heads):
        raise ValueError(
            f"expected [global_rows, owned_heads, ...] = "
            f"[{plan.global_rows}, {plan.owned_heads}, ...], got {tuple(x.shape)}"
        )
    return (
        x.reshape(plan.local_rows, plan.cp_size, plan.owned_heads, *x.shape[2:])
        .movedim(1, 0)
        .contiguous()
    )


def merge_head_shards_for_token_owner(
    received: torch.Tensor, plan: KdaPrefillCpHeadPlan
) -> torch.Tensor:
    """Concatenate source CP ranks back into the original local head order."""

    if received.ndim < 3 or received.shape[:3] != (
        plan.cp_size,
        plan.local_rows,
        plan.owned_heads,
    ):
        raise ValueError(
            "expected received [cp, local_rows, owned_heads, ...], got "
            f"{tuple(received.shape)}"
        )
    return received.transpose(0, 1).reshape(
        plan.local_rows, plan.local_heads, *received.shape[3:]
    )


def exchange_local_rows_for_owned_heads(
    x: torch.Tensor, plan: KdaPrefillCpHeadPlan
) -> torch.Tensor:
    send = pack_local_rows_for_head_owners(x, plan)
    received = torch.empty_like(send)
    get_parallel().attn_cp_group.all_to_all_single(received, send)
    return merge_token_shards_for_head_owner(received, plan)


def restore_local_rows_from_owned_heads(
    x: torch.Tensor, plan: KdaPrefillCpHeadPlan
) -> torch.Tensor:
    send = pack_global_rows_for_token_owners(x, plan)
    received = torch.empty_like(send)
    get_parallel().attn_cp_group.all_to_all_single(received, send)
    return merge_head_shards_for_token_owner(received, plan)


def select_flat_head_range(
    x: torch.Tensor,
    plan: KdaPrefillCpHeadPlan,
    *,
    head_dim: int,
    head_axis: int = -1,
) -> torch.Tensor:
    """Select the owned head range from one flattened ``H * head_dim`` axis."""

    axis = head_axis % x.ndim
    if x.shape[axis] != plan.local_heads * head_dim:
        raise ValueError(
            f"flattened head axis has {x.shape[axis]} elements, expected "
            f"{plan.local_heads * head_dim}"
        )
    shape = list(x.shape)
    shape[axis : axis + 1] = [plan.local_heads, head_dim]
    expanded = x.reshape(shape)
    index = [slice(None)] * expanded.ndim
    index[axis] = slice(plan.head_start, plan.head_end)
    selected = expanded[tuple(index)]
    out_shape = list(x.shape)
    out_shape[axis] = plan.owned_heads * head_dim
    return selected.reshape(out_shape).contiguous()


def select_packed_qkv_head_range(
    x: torch.Tensor,
    plan: KdaPrefillCpHeadPlan,
    *,
    head_dim: int,
    packed_axis: int = -1,
) -> torch.Tensor:
    """Select heads from a flattened ``[Q heads, K heads, V heads]`` axis."""

    axis = packed_axis % x.ndim
    expected = 3 * plan.local_heads * head_dim
    if x.shape[axis] != expected:
        raise ValueError(
            f"packed QKV axis has {x.shape[axis]} elements, expected {expected}"
        )
    shape = list(x.shape)
    shape[axis : axis + 1] = [3, plan.local_heads, head_dim]
    expanded = x.reshape(shape)
    index = [slice(None)] * expanded.ndim
    index[axis + 1] = slice(plan.head_start, plan.head_end)
    selected = expanded[tuple(index)]
    out_shape = list(x.shape)
    out_shape[axis] = 3 * plan.owned_heads * head_dim
    return selected.reshape(out_shape).contiguous()


def write_flat_head_range(
    destination: torch.Tensor,
    source: torch.Tensor,
    plan: KdaPrefillCpHeadPlan,
    *,
    head_axis: int,
) -> None:
    """Write an explicit H/2 axis into a full-H tensor."""

    axis = head_axis % destination.ndim
    expected = list(destination.shape)
    expected[axis] = plan.owned_heads
    if list(source.shape) != expected:
        raise ValueError(f"source shape {tuple(source.shape)} != {tuple(expected)}")
    index = [slice(None)] * destination.ndim
    index[axis] = slice(plan.head_start, plan.head_end)
    destination[tuple(index)] = source


def write_packed_qkv_head_range(
    destination: torch.Tensor,
    source: torch.Tensor,
    plan: KdaPrefillCpHeadPlan,
    *,
    head_dim: int,
    packed_axis: int = -1,
) -> None:
    """Write owned heads into a flattened full Q/K/V channel axis."""

    axis = packed_axis % destination.ndim
    expected_full = 3 * plan.local_heads * head_dim
    expected_owned = 3 * plan.owned_heads * head_dim
    if destination.shape[axis] != expected_full or source.shape[axis] != expected_owned:
        raise ValueError(
            f"packed QKV shapes do not match full/owned widths: "
            f"{destination.shape[axis]}/{source.shape[axis]} vs "
            f"{expected_full}/{expected_owned}"
        )
    full_shape = list(destination.shape)
    full_shape[axis : axis + 1] = [3, plan.local_heads, head_dim]
    owned_shape = list(source.shape)
    owned_shape[axis : axis + 1] = [3, plan.owned_heads, head_dim]
    full = destination.reshape(full_shape)
    owned = source.reshape(owned_shape)
    index = [slice(None)] * full.ndim
    index[axis + 1] = slice(plan.head_start, plan.head_end)
    full[tuple(index)] = owned


def gather_full_head_axis(
    owned: torch.Tensor, plan: KdaPrefillCpHeadPlan, *, head_axis: int
) -> torch.Tensor:
    """Replicate rank-ordered H/2 state shards as a full-H tensor."""

    axis = head_axis % owned.ndim
    if owned.shape[axis] != plan.owned_heads:
        raise ValueError(
            f"owned head axis has {owned.shape[axis]}, expected {plan.owned_heads}"
        )
    return get_parallel().attn_cp_group.all_gather(owned.contiguous(), dim=axis)
