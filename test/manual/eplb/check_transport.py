"""Eight-rank byte-exact migration checks for IPC and NCCL transports."""

import json
import socket
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.eplb.expert_transport import ExpertTransport
from sglang.srt.eplb.moe_shape import MoeShape
from sglang.srt.eplb.multi_swap import MultiPlanner


def worker(rank, port, output):
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    cpu = dist.new_group(backend="gloo")
    group = SimpleNamespace(
        cpu_group=cpu, device_group=dist.group.WORLD, pynccl_comm=None
    )
    shape = MoeShape(256, 264, 8, 4096, 6)
    m = torch.tensor(
        list(range(256)) + list(range(8)), device="cuda", dtype=torch.int64
    )
    planner = MultiPlanner("cuda", 1, shape)
    gen = torch.Generator().manual_seed(4)
    q = torch.randint(0, 1500, (264,), generator=gen).to("cuda", dtype=torch.int32)
    planner.plan(m, q)
    copies = planner.copies[: int(planner.info[0])].cpu().tolist()
    assert copies, "Fixture must exercise real transfers"
    rows = torch.arange(
        rank * shape.slots, (rank + 1) * shape.slots, device="cuda", dtype=torch.int64
    )
    weights = [
        rows[:, None].expand(-1, 256).clone(),
        (rows[:, None].expand(-1, 32) + 1000).float().clone(),
    ]
    originals = [t.clone() for t in weights]
    passed = []
    for kind in ("ipc", "nccl"):
        for target, original in zip(weights, originals):
            target.copy_(original)
        transport = ExpertTransport(weights, group, rank, 0, shape, kind)
        ok = torch.ones(1, device="cuda", dtype=torch.int32)
        transport.read(planner, ok, 1)
        transport.commit(planner, ok, 1, True)
        expected = rows.clone()
        for src, dst in copies:
            if dst // shape.slots == rank:
                expected[dst % shape.slots] = src
        assert torch.equal(weights[0], expected[:, None].expand_as(weights[0]))
        assert torch.equal(
            weights[1], (expected + 1000)[:, None].expand_as(weights[1]).float()
        )
        assert int(ok[0]) == 1
        passed.append(kind)
        dist.barrier()
    Path(output, f"rank{rank}.json").write_text(
        json.dumps(dict(rank=rank, transports=passed, copies=copies))
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port, args.output), nprocs=8, join=True)
    print("PASS: IPC and NCCL expert bytes, 8 ranks")
