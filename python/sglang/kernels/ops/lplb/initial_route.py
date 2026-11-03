"""Initial locality-aware routing from the LIVE runtime physical mapping.

For the eager Block-LPLB migration path. The subsequent Block-LPLB planner
computes exact global residual quotas. This replaces the provisional LP solve;
it does not freeze EPLB layout or bypass final quota/mapping/weight checks.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _initial(
    ID, MAP, VALID, OUT, N, WIDTH: tl.constexpr, TOPK: tl.constexpr, RANK: tl.constexpr
):
    j = tl.program_id(0) * 256 + tl.arange(0, 256)
    e = tl.load(ID + j, j < N, -1)
    n = tl.load(VALID + e, (j < N) & (e >= 0), 1)
    # Metadata collapses any local replica to column 0. Remote copies are spread
    # deterministically by token, without a histogram, atomics, or CPU mirror.
    col = ((j // TOPK) + RANK * 131) % tl.maximum(n, 1)
    p = tl.load(MAP + e * WIDTH + col, (j < N) & (e >= 0), -1)
    tl.store(OUT + j, p, j < N)


def initial_route(ids, info):
    assert info.num_physical_experts > 0 and not info.rank_invariant
    out = torch.empty_like(ids)
    if ids.numel():
        _initial[(tr.cdiv(ids.numel(), 256),)](
            ids,
            info.partial_logical_to_all_physical_map,
            info.partial_logical_to_all_physical_map_num_valid,
            out,
            ids.numel(),
            info.partial_logical_to_all_physical_map.shape[1],
            ids.shape[-1],
            info.source_rank_offset,
        )
    return out
