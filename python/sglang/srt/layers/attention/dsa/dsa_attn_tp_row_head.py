# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Attention-TP row/head exchange for sparse MLA prefill.

Each attention-TP rank normally owns a head shard for every local query row.
For kernels that run materially better with the full real head count, exchange
equal row slices so each rank instead owns every head for its query rows.  The
inverse exchange restores the original token-major, head-sharded layout before
the existing output projection.

The first production consumer is deliberately narrow: GLM-5.3 Flash CP2 with
attention-TP2 and native FlashMLA V32.  The pure layout helpers stay independent
of that model gate so the collective ownership contract is CPU-testable.
"""

from typing import NamedTuple, Optional

import torch

from sglang.srt.runtime_context import get_parallel


class DsaAttnTpRowHeadPlan(NamedTuple):
    num_rows: int
    rows_per_rank: int
    tp_size: int
    tp_rank: int
    local_start: int
    local_end: int


def plan_dsa_attn_tp_row_head(
    num_rows: int, tp_size: int, tp_rank: int
) -> Optional[DsaAttnTpRowHeadPlan]:
    """Return an equal-row plan or ``None`` for the common fallback.

    The current H20 candidate is an attention-TP pair.  Empty or uneven rows
    keep the existing head-padded kernel path on every rank; padding them here
    would require new per-request sparse metadata semantics.
    """

    if tp_size != 2 or num_rows <= 0 or num_rows % tp_size:
        return None
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"attention TP rank {tp_rank} outside [0, {tp_size})")
    rows_per_rank = num_rows // tp_size
    local_start = tp_rank * rows_per_rank
    return DsaAttnTpRowHeadPlan(
        num_rows=num_rows,
        rows_per_rank=rows_per_rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        local_start=local_start,
        local_end=local_start + rows_per_rank,
    )


def pack_query_rows(x: torch.Tensor, plan: DsaAttnTpRowHeadPlan) -> torch.Tensor:
    """Pack destination-owned row chunks for all-to-all."""

    if x.ndim != 3 or x.shape[0] != plan.num_rows:
        raise ValueError(
            f"expected query [rows, heads, dim] with {plan.num_rows} rows, "
            f"got {tuple(x.shape)}"
        )
    return x.reshape(
        plan.tp_size, plan.rows_per_rank, x.shape[1], x.shape[2]
    ).contiguous()


def merge_query_heads(
    received: torch.Tensor, plan: DsaAttnTpRowHeadPlan
) -> torch.Tensor:
    """Merge source-rank head shards into global head order."""

    if received.ndim != 4 or received.shape[:2] != (
        plan.tp_size,
        plan.rows_per_rank,
    ):
        raise ValueError(
            "expected received query [tp, rows_per_rank, heads, dim], got "
            f"{tuple(received.shape)}"
        )
    return received.permute(1, 0, 2, 3).reshape(
        plan.rows_per_rank,
        plan.tp_size * received.shape[2],
        received.shape[3],
    )


def pack_output_heads(
    x: torch.Tensor, plan: DsaAttnTpRowHeadPlan
) -> torch.Tensor:
    """Pack each original head owner's output for the inverse all-to-all."""

    if (
        x.ndim != 3
        or x.shape[0] != plan.rows_per_rank
        or x.shape[1] % plan.tp_size
    ):
        raise ValueError(
            "expected output [rows_per_rank, tp * heads, dim], got "
            f"{tuple(x.shape)}"
        )
    local_heads = x.shape[1] // plan.tp_size
    return x.reshape(
        plan.rows_per_rank, plan.tp_size, local_heads, x.shape[2]
    ).permute(1, 0, 2, 3).contiguous()


def merge_output_rows(
    received: torch.Tensor, plan: DsaAttnTpRowHeadPlan
) -> torch.Tensor:
    """Restore token-major rows for this rank's original head shard."""

    if received.ndim != 4 or received.shape[:2] != (
        plan.tp_size,
        plan.rows_per_rank,
    ):
        raise ValueError(
            "expected received output [tp, rows_per_rank, heads, dim], got "
            f"{tuple(received.shape)}"
        )
    return received.reshape(plan.num_rows, received.shape[2], received.shape[3])


def redistribute_query_heads(
    x: torch.Tensor, plan: DsaAttnTpRowHeadPlan
) -> torch.Tensor:
    send = pack_query_rows(x, plan)
    received = torch.empty_like(send)
    get_parallel().attn_tp_group.all_to_all_single(received, send)
    return merge_query_heads(received, plan)


def restore_output_rows(
    x: torch.Tensor, plan: DsaAttnTpRowHeadPlan
) -> torch.Tensor:
    send = pack_output_heads(x, plan)
    received = torch.empty_like(send)
    get_parallel().attn_tp_group.all_to_all_single(received, send)
    return merge_output_rows(received, plan)


def local_row_slice(x: torch.Tensor, plan: DsaAttnTpRowHeadPlan) -> torch.Tensor:
    if x.shape[0] != plan.num_rows:
        raise ValueError(
            f"row-aligned metadata has {x.shape[0]} rows, expected {plan.num_rows}"
        )
    return x[plan.local_start : plan.local_end]
