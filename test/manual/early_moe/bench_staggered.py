"""Eight-GPU staggered gate arrival with the real in-kernel peer stop protocol.

Synthetic transformed weights and routed IDs only; no model or HTTP service.
Ready epochs are polled by the fused FFN itself. Delay generation is outside
the measured operator, on an independent benchmark IPC allocation.
"""

import argparse
import importlib.util
import json
import socket
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "python"))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from profiles import PROFILES

from sglang.kernels.ops.moe.early_moe import (
    STAT_NAMES,
    create_early_moe_context,
    early_moe,
)
from sglang.kernels.ops.moe.early_moe.runtime import PeerEpochs


def worker(rank, args, port):
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    path = ROOT / "operator_smoke.py"
    spec = importlib.util.spec_from_file_location("smoke_helpers", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.configure(args.profile)
    h, i, k, e, slots = PROFILES[args.profile].dimensions
    weights = helper.weights_for_layer(rank, 0)
    x, ids, routing = helper.inputs(rank, args.tokens, 101)
    generator = torch.Generator(device="cuda").manual_seed(1000 + rank)
    local = (
        torch.rand(ids.shape, generator=generator, device="cuda") < args.local_fraction
    )
    ids = torch.where(local, ids, (ids + slots) % e).contiguous()
    local_map = torch.full((e,), -1, device="cuda", dtype=torch.int32)
    local_map[rank * slots : (rank + 1) * slots] = torch.arange(
        slots, device="cuda", dtype=torch.int32
    )
    reverse = torch.arange(
        rank * slots, (rank + 1) * slots, device="cuda", dtype=torch.int32
    )
    context = create_early_moe_context(
        dist.group.WORLD,
        hidden=h,
        intermediate=i,
        topk=k,
        logical_experts=e,
        local_slots=slots,
        max_tokens=args.tokens,
        max_inflight_blocks=16,
        stages=4,
        tile_n=256,
        bulk_prepare=True,
        diagnostic=False,
        audit_capacity=0,
    )
    context.bind(weights, local_map, reverse, 1)
    gate = PeerEpochs(dist.group.WORLD, context.lib)
    begin, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    reports = []
    for epoch in range(1, args.iterations + 3):
        gate.delay(epoch, args.delays_us[rank])
        begin.record()
        result = early_moe(
            x,
            ids,
            routing,
            weights,
            local_map,
            1,
            epoch,
            context=context,
            activation_clamp=helper.CLAMP,
        )
        end.record()
        end.synchronize()
        row = dict(zip(STAT_NAMES, result.stats[0].cpu().tolist()))
        assert (
            row["status"] == 0 and row["admitted_tasks"] == row["completed_m64_tasks"]
        )
        assert row["completed_assignments"] <= row["eligible_local_assignments"]
        done = result.early_done.bool()
        assert torch.equal(result.remaining_physical_ids, torch.where(done, -1, ids))
        assert int(result.early_count_by_expert.sum()) == int(done.sum())
        assert (
            int(result.remaining_count_by_expert.sum()) + int(done.sum())
            == args.tokens * k
        )
        if epoch > 2:
            row.update(
                rank=rank,
                iteration=epoch - 3,
                delay_us=args.delays_us[rank],
                cuda_event_ms=begin.elapsed_time(end),
                early_fraction=row["completed_assignments"] / max(1, args.tokens * k),
                local_completion_fraction=row["completed_assignments"]
                / max(1, row["eligible_local_assignments"]),
                profile=args.profile,
                tokens=args.tokens,
            )
            reports.append(row)
    (Path(args.output) / f"rank{rank}.json").write_text(json.dumps(reports, indent=2))
    gate.close()
    context.close()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="dsv4")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--local-fraction", type=float, default=0.125)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--delays-us",
        type=float,
        nargs=8,
        default=[0, 100, 200, 300, 400, 500, 600, 700],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.tokens <= 8192 or not 0 <= args.local_fraction <= 1:
        parser.error("tokens must be 1..8192 and local fraction 0..1")
    if args.iterations < 1 or min(args.delays_us) < 0:
        parser.error("iterations must be positive and delays nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(args, port), nprocs=8, join=True)
    rows = [
        row
        for rank in range(8)
        for row in json.loads((args.output / f"rank{rank}.json").read_text())
    ]
    (args.output / "results.json").write_text(json.dumps(rows, indent=2))
    print(
        json.dumps(
            {
                "status": "passed",
                "rank_iterations": len(rows),
                "completed_assignments": sum(r["completed_assignments"] for r in rows),
                "eligible_assignments": sum(
                    r["eligible_local_assignments"] for r in rows
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
