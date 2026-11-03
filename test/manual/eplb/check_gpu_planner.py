"""Compare the actual GPU K1 planner against the independent CPU oracle."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))

import torch

from sglang.kernels.ops.lplb.block_planner import BlockLPLB
from sglang.srt.eplb.moe_shape import MoeShape
from sglang.srt.eplb.multi_swap import MultiPlanner
from sglang.srt.eplb.multi_swap_cpu import plan


def main():
    rows = []
    for world, logical, hidden, topk in (
        (world, logical, hidden, topk)
        for world in (8, 16, 32)
        for logical, hidden, topk in ((256, 4096, 6), (384, 5120, 6), (288, 4096, 8))
    ):
        name = f"ep{world}_e{logical}"
        shape = MoeShape(logical, logical + world, world, hidden, topk)
        mapping = list(range(shape.logical)) + list(range(world))
        m = torch.tensor(mapping, device="cuda", dtype=torch.int64)
        planner = MultiPlanner("cuda", 1, shape)
        q = torch.empty(shape.physical, device="cuda", dtype=torch.int32)
        for seed in range(6):
            generator = torch.Generator().manual_seed(seed)
            cpu = (
                torch.randint(0, 1500, (shape.physical,), generator=generator).tolist()
                if seed
                else [0] * shape.physical
            )
            q.copy_(torch.tensor(cpu, device="cuda", dtype=torch.int32))
            planner.plan(m, q)
            torch.cuda.synchronize()
            expected = plan(mapping, cpu, 1, shape)
            assert planner.mapping.cpu().tolist() == expected["mapping"], (
                name,
                seed,
                "mapping",
            )
            assert planner.quota.cpu().tolist() == expected["quota"], (
                name,
                seed,
                "quota",
            )
            assert planner.counter.cpu().tolist() == expected["counts"], (
                name,
                seed,
                "budget",
            )
            rows.append(dict(profile=name, seed=seed, copies=len(expected["copies"])))
        # Exercise the extracted Tail planner and physical-ID routing wrapper.
        tail = BlockLPLB(m, world, 64)
        counts = torch.ones((world, shape.physical), device="cuda", dtype=torch.int32)
        quota = tail.plan(counts)
        assert int(quota.sum()) == world * shape.physical
        ids = torch.arange(shape.physical, device="cuda", dtype=torch.int32)
        realized = torch.zeros(shape.physical, device="cuda", dtype=torch.int32)
        for rank in range(world):
            routed = tail.route(ids, counts, quota, rank)
            assert torch.equal(m[routed.long()], m[ids.long()])
            realized += torch.bincount(routed.long(), minlength=shape.physical).to(
                torch.int32
            )
        assert torch.equal(realized, quota)
    print(json.dumps(dict(status="passed", planner_cases=rows, block_route_profiles=9)))


if __name__ == "__main__":
    main()
