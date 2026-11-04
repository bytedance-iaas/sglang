"""Eight-GPU exact-output regression for native slot-ordered early combine.

Uses real checkpoint weights when --model is supplied. Tests poisoned unread
slots, interleaved completed slots, all/none completed, an empty rank, uneven
partial blocks, and the actual fused early operator with a timed stop signal.
"""

import argparse
import json
import socket
from datetime import timedelta
from pathlib import Path

import operator_smoke as fixture
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from checkpoint_weights import actual_weights

from sglang.kernels.ops.moe.early_moe import (
    STAT_NAMES,
    create_early_moe_context,
    early_moe,
)
from sglang.kernels.ops.moe.early_moe.runtime import PeerEpochs


def worker(rank, args, port):
    import iaas_kernels as impl

    fixture.configure("dsv4")
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=args.world,
        timeout=timedelta(seconds=300),
    )
    h, i, k, slots = fixture.H, fixture.I, fixture.K, fixture.S
    experts = args.world * slots
    weights = (
        actual_weights(Path(args.model), 21, rank)
        if args.model
        else fixture.weights_for_layer(rank, 21)
    )
    n = 0 if rank == 0 else args.tokens + rank
    x, ids, routing = fixture.inputs(rank, n, 72311)
    for slot in range(1, k, 2):
        ids[:, slot] = ((rank + 1) % args.world) * slots + fixture.SLOTS[slot]
    buf = impl.get_symm_buffer_for_sm90_mega_moe(
        dist.group.WORLD,
        experts,
        args.tokens + args.world,
        k,
        h,
        i,
    )

    def run(route, values=None, done=None):
        impl.mega_moe_pre_dispatch_sm90(
            x,
            route,
            routing,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=n,
            routed_scaling_factor=1.0,
        )
        y = torch.empty((n, h), dtype=torch.bfloat16, device="cuda")
        impl.fp8_mega_moe(
            y,
            (weights.l1, weights.l1_scale),
            (weights.l2, weights.l2_scale),
            buf,
            activation_clamp=fixture.CLAMP,
            early_values=values,
            early_done=done,
        )
        torch.cuda.synchronize()
        return y

    full = run(ids)
    assert torch.equal(full, run(ids)), "native repeat changed output"
    contributions = []
    for slot in range(k):
        one = torch.full_like(ids, -1)
        one[:, slot] = ids[:, slot]
        contributions.append(run(one))
    values = torch.stack(contributions, dim=1).contiguous()
    index = torch.arange(n * k, device="cuda").reshape(n, k)
    reports = []
    for name, mask in [
        ("none", torch.zeros_like(ids, dtype=torch.bool)),
        ("interleaved", index % 3 == 0),
        ("last_slot", index % k == k - 1),
        ("all", torch.ones_like(ids, dtype=torch.bool)),
        ("fresh_none_after_all", torch.zeros_like(ids, dtype=torch.bool)),
    ]:
        done = mask.to(torch.uint8)
        poisoned = values.masked_fill(~mask[..., None], float("nan"))
        actual = run(torch.where(mask, -1, ids), poisoned, done)
        different = int((actual != full).sum())
        reports.append(dict(case=name, different=different, completed=int(done.sum())))
        assert different == 0, (rank, name, different)

    local_map = torch.full((experts,), -1, dtype=torch.int32, device="cuda")
    local_map[rank * slots : (rank + 1) * slots] = torch.arange(slots, device="cuda")
    reverse = torch.arange(
        rank * slots, (rank + 1) * slots, dtype=torch.int32, device="cuda"
    )
    stop = torch.zeros(1, dtype=torch.uint64, device="cuda")
    context = create_early_moe_context(
        dist.group.WORLD,
        max_tokens=args.tokens + args.world,
        hidden=h,
        intermediate=i,
        topk=k,
        logical_experts=experts,
        local_slots=slots,
        max_inflight_blocks=16,
        stages=4,
        tile_n=256,
        bulk_prepare=True,
        stop_signal=stop,
        external_ready=True,
        diagnostic=True,
        audit_capacity=128,
    )
    context.bind(weights, local_map, reverse, 1)
    timer = PeerEpochs(dist.group.WORLD, context.lib)
    signal_stream = torch.cuda.Stream()
    for epoch, delay in enumerate([20000, 100, 20000], 1):
        torch.cuda.synchronize()
        dist.barrier()
        timer.delay_stop(epoch, delay, context.peer.flags, stop, signal_stream)
        result = early_moe(
            x,
            ids,
            routing,
            weights,
            local_map,
            1,
            epoch,
            context=context,
            activation_clamp=fixture.CLAMP,
        )
        actual = run(
            result.remaining_physical_ids,
            result.early_values.view(-1, k, h),
            result.early_done,
        )
        stats = dict(zip(STAT_NAMES, result.stats[0].cpu().tolist()))
        different = int((actual != full).sum())
        reports.append(dict(case=f"fused_epoch_{epoch}", different=different, **stats))
        assert stats["status"] == 0 and different == 0, (rank, epoch, different, stats)
        assert torch.equal(
            result.remaining_physical_ids,
            torch.where(result.early_done.bool(), -1, ids),
        )
    signal_stream.synchronize()
    timer.close()
    context.close()
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"rank{rank}.json").write_text(json.dumps(reports, indent=2))
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--tokens", type=int, default=65)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(args, port), nprocs=args.world, join=True)
    rows = [
        dict(rank=r, **row)
        for r in range(args.world)
        for row in json.loads((args.output / f"rank{r}.json").read_text())
    ]
    summary = dict(status="passed", ranks=args.world, cases=len(rows), rows=rows)
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
