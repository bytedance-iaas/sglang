"""Global integer Block-LPLB planner. Bsum is never an objective.

Bmax means sum(ceil(physical_count / block_size)) on the busiest rank.
It is an explicit row-block proxy, not a claim about heterogeneous MegaMoE tiles.
Every accepted pair transfer conserves logical counts and cannot increase either
Bmax or Tmax. Global optimality or strictly improved tails are not guaranteed.
"""

import itertools
import os

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args


@cache_once
def _module(p, r, m, iterations, variant="reference"):
    args = make_cpp_args(p, r, m, iterations)
    route = make_cpp_args(p, r)
    plan = (
        f"block_plan<{args}>"
        if variant == "reference"
        else f"block_plan_warp<{args},{1 if variant == 'boundary' else 0}>"
    )
    return load_jit(
        "lplb_block_lplb_" + variant,
        *args,
        cuda_files=["lplb/block_lplb.cuh"],
        cuda_wrappers=[
            ("plan", plan),
            ("route", f"block_route<{route}>"),
            ("hist", f"block_hist<{p}>"),
        ],
    )


from sglang.srt.eplb.balance_config import BLOCK_SIZE


class BlockLPLB:
    def __init__(
        self, phy2log, ranks, block_size=BLOCK_SIZE, iterations=16, values=None
    ):
        if ranks <= 0 or ranks > 256 or block_size <= 0 or iterations <= 0:
            raise ValueError("Positive block size/iterations and 1..256 ranks required")
        values = phy2log.detach().cpu().tolist() if values is None else list(values)
        if not values:
            raise ValueError("Empty expert mapping")
        if len(values) % ranks or min(values) < 0:
            raise ValueError("Expected valid equal-size physical expert shards")
        self.p, self.r = len(values), ranks
        self.logical = max(values) + 1
        links = list(range(self.p))
        slots = {}
        for p, logical in enumerate(values):
            slots.setdefault(logical, []).append(p)
        pairs = []
        for copies in slots.values():
            for j, p in enumerate(copies):
                links[p] = copies[(j + 1) % len(copies)]
            pairs.extend(
                (a, b)
                for a, b in itertools.combinations(copies, 2)
                if a // (self.p // ranks) != b // (self.p // ranks)
            )
        pairs.sort()  # Preserve native planner pair visit order exactly.
        self.phy = torch.tensor(
            values + links, dtype=torch.int32, device=phy2log.device
        )
        self.pairs = torch.tensor(
            pairs, dtype=torch.int32, device=phy2log.device
        ).reshape(-1, 2)
        variant = os.getenv("SGLANG_LPLB_BLOCK_PLANNER_VARIANT", "reference")
        if variant not in ("reference", "warp", "boundary"):
            raise ValueError(variant)
        iterations = int(
            os.getenv("SGLANG_LPLB_BLOCK_PLANNER_ITERATIONS", str(iterations))
        )
        self.mod = _module(self.p, ranks, block_size, iterations, variant)
        self.block_size = block_size

    def plan(self, counts, quota=None):
        assert (
            counts.shape == (self.r, self.p)
            and counts.dtype == torch.int32
            and counts.is_contiguous()
        )
        if quota is None:
            quota = torch.empty((self.p,), dtype=torch.int32, device=counts.device)
        self.mod.plan(counts, self.pairs, quota)
        return quota

    def route(self, ids, counts, quota, rank, out=None, cursor=None):
        assert 0 <= rank < self.r
        flat = ids.to(torch.int32).contiguous().view(-1)
        if out is None:
            out = torch.empty_like(flat)
        if cursor is None:
            cursor = torch.empty((self.logical,), dtype=torch.int32, device=ids.device)
        cursor.zero_()
        self.mod.route(flat, out, self.phy, counts, quota, cursor, rank)
        return out.reshape(ids.shape)
