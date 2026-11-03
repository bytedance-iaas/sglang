"""Standalone 8-GPU DeepSeek physical-route fused early-MoE correctness smoke.

Run directly, not through torchrun. Requires this directory's compiled
megamoe_early package, CUDA PyTorch and eight free peer-accessible SM90 GPUs.
No model weights, SGLang or original acceptance-result files are required.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "python"))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from profiles import PROFILES

from sglang.kernels.ops.moe.early_moe import (
    ActiveWeights,
    create_early_moe_context,
    early_moe,
)
from sglang.kernels.ops.moe.early_moe.runtime import PeerEpochs


def configure(profile):
    global H, I, K, E, S, GEOMETRY, SLOTS
    H, I, K, E, S = PROFILES[profile].dimensions
    GEOMETRY = PROFILES[profile].abi
    SLOTS = tuple(range(K - 3)) + (S - 3, S - 2, S - 1)


# Exercise both low and final slots, including slots beyond a single warp.
configure("dsv4")
CLAMP = 10.0


def quantize_weight(x):
    n, k = x.shape
    block = x.float().reshape(n // 128, 128, k // 128, 128)
    scale = block.abs().amax((1, 3)).clamp_min(1e-10) * (1 / 448.0)
    value = (block / scale[:, None, :, None]).to(torch.float8_e4m3fn).reshape(n, k)
    return value, scale.contiguous()


def quantize_rows(x, group, native_a1=False):
    grouped = x.float().reshape(x.shape[0], -1, group)
    amax = grouped.abs().amax(-1).clamp_min(1e-10)
    scale = (
        (amax / torch.full((), 448.0, device=x.device))
        if native_a1
        else amax * (1 / 448.0)
    )
    # Match production reciprocal/multiply FP8 rounding, not direct division.
    value = (grouped * (1.0 / scale)[..., None]).to(torch.float8_e4m3fn)
    return (value.float() * scale[..., None]).reshape_as(x)


def weights_for_layer(rank, layer):
    # Unused slots are allocated but deliberately never routed/read. Each used
    # physical slot has a different deterministic matrix, including the last three slots.
    kw = dict(device="cuda")
    w1 = torch.empty((S, 2 * I, H), dtype=torch.float8_e4m3fn, **kw)
    w2 = torch.empty((S, H, I), dtype=torch.float8_e4m3fn, **kw)
    sf1 = torch.empty((S, 2 * I // 128, H // 128), dtype=torch.float32, **kw)
    sf2 = torch.empty((S, H // 128, I // 128), dtype=torch.float32, **kw)
    for slot in SLOTS:
        g = torch.Generator(device="cuda").manual_seed(
            74000 + layer * 4096 + rank * S + slot
        )
        q1, f1 = quantize_weight(
            torch.randn((2 * I, H), generator=g, device="cuda") / math.sqrt(H)
        )
        q2, f2 = quantize_weight(
            torch.randn((H, I), generator=g, device="cuda") / math.sqrt(I)
        )
        w1[slot].copy_(
            q1.reshape(2, I // 8, 8, H).permute(1, 0, 2, 3).reshape(2 * I, H)
        )
        w2[slot].copy_(q2)
        sf1[slot].copy_(f1)
        sf2[slot].copy_(f2)
    return ActiveWeights(w1, sf1, w2, sf2)


def inputs(rank, n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed + rank)
    # High-enough amplitudes exercise DeepSeek's clamp10, rather than only its
    # unclamped region. Routing scale1.5 is already included exactly once.
    hidden = torch.randn((n, H), generator=g, device="cuda", dtype=torch.bfloat16) * 8
    ids = torch.tensor(
        [rank * S + s for s in SLOTS], device="cuda", dtype=torch.int32
    ).repeat(n, 1)
    routing = (
        torch.softmax(torch.randn((n, K), generator=g, device="cuda"), dim=-1).float()
        * 1.5
    )
    return hidden.contiguous(), ids.contiguous(), routing.contiguous()


def snapshot(result, n):
    # This is the required same-stream consumption before the next layer reuses
    # its arena. No CPU synchronization is inserted between layer launches.
    return SimpleNamespace(
        remaining=result.remaining_logical_ids.clone(),
        done=result.early_done.clone(),
        values=result.early_values[: n * K].clone(),
        early_count=result.early_count_by_expert.clone(),
        remaining_count=result.remaining_count_by_expert.clone(),
        stats=result.stats[0].clone(),
        audit=result.audit.clone(),
        status=result.status.clone(),
    )


def oracle(x, ids, routing, weights, rank, selected):
    out = torch.empty((selected.numel(), H), device="cuda", dtype=torch.bfloat16)
    selected_ids = ids.flatten()[selected]
    for physical in selected_ids.unique().cpu().tolist():
        slot = physical - rank * S
        assert slot in SLOTS
        which = (selected_ids == physical).nonzero().flatten()
        p = selected[which]
        xx = quantize_rows(x[p // K], 128, native_a1=True)
        unpack = (
            weights.l1[slot]
            .reshape(I // 8, 2, 8, H)
            .permute(1, 0, 2, 3)
            .reshape(2 * I, H)
        )
        w1 = (
            unpack.float().reshape(2 * I // 128, 128, H // 128, 128)
            * weights.l1_scale[slot][:, None, :, None]
        ).reshape(2 * I, H)
        gu = xx @ w1.T
        gate, up = gu[:, :I].clamp_max(CLAMP), gu[:, I:].clamp(-CLAMP, CLAMP)
        act = torch.nn.functional.silu(gate) * up * routing.flatten()[p, None]
        act = quantize_rows(act, 64)
        w2 = (
            weights.l2[slot].float().reshape(H // 128, 128, I // 128, 128)
            * weights.l2_scale[slot][:, None, :, None]
        ).reshape(H, I)
        out[which] = (act @ w2.T).to(torch.bfloat16)
    return out


def check(rank, name, inp, weights, out, full):
    x, ids, routing = inp
    n = len(x)
    stats = out.stats.cpu().tolist()
    done = out.done.bool()
    valid = ids >= 0
    assert int(out.status.item()) == stats[20] == 0, (name, rank, stats)
    assert torch.equal(done ^ (out.remaining >= 0), valid)
    assert torch.equal(out.remaining[~done], ids[~done])
    assert torch.all(out.remaining[done] == -1)
    assert torch.equal(
        torch.bincount(ids[done].long(), minlength=E).int(), out.early_count
    )
    assert torch.equal(
        torch.bincount(ids[(~done) & valid].long(), minlength=E).int(),
        out.remaining_count,
    )
    assert int(done.sum()) == stats[7]
    assert int(done.any(-1).sum()) == stats[19]
    assert stats[13] == stats[8], ("uncommitted admission", stats)
    assert stats[12] <= stats[25] and stats[26] <= stats[25]
    assert stats[14] == 0, ("unexpected audit overflow", stats)
    if full:
        assert stats[7] == stats[6] == n * K, (
            "long stop window failed to exhaust candidates",
            name,
            rank,
            stats,
        )
        assert stats[8] == K * math.ceil(n / 64)
        assert stats[9] == K * int(n % 64 != 0)
    else:
        assert stats[7] == stats[8] == stats[13] == 0
    audit = out.audit[: stats[13]].cpu().tolist()
    assert sum(a[2] for a in audit) == stats[7]
    for a in audit:
        assert a[1] in [rank * S + s for s in SLOTS]
        assert a[3] <= stats[2]
        assert a[3] <= a[4] <= a[5] <= a[6] <= a[7]
    selected = done.flatten().nonzero().flatten()
    rel, max_abs, reference_max, merged_rel = None, None, None, None
    if selected.numel():
        ref = oracle(x, ids, routing, weights, rank, selected)
        got = out.values[selected]
        diff = got.float() - ref.float()
        rel = float(diff.norm() / ref.float().norm().clamp_min(1e-20))
        max_abs = float(diff.abs().max())
        reference_max = float(ref.float().abs().max())
        assert rel <= 0.008 and max_abs <= 0.02 * reference_max + 1e-4, (
            name,
            rank,
            rel,
            max_abs,
            reference_max,
        )
        # Full-candidate case also validates the per-token weighted reduction.
        combined = torch.zeros((n * K, H), device="cuda", dtype=torch.bfloat16)
        expected = torch.zeros_like(combined)
        combined[selected], expected[selected] = got, ref
        a, b = (
            combined.reshape(n, K, H).float().sum(1),
            expected.reshape(n, K, H).float().sum(1),
        )
        merged_rel = float((a - b).norm() / b.norm().clamp_min(1e-20))
        assert merged_rel <= 0.008, ("token merge error", merged_rel)
    return dict(
        case=name,
        rank=rank,
        tokens=n,
        selected_physical_slots=[rank * S + s for s in SLOTS],
        stats=stats,
        audit=audit,
        rel_l2=rel,
        max_abs=max_abs,
        reference_max=reference_max,
        token_merge_rel_l2=merged_rel,
        input_routed_scale=1.5,
        activation_clamp=CLAMP,
    )


def worker(rank, args, port):
    configure(args.profile)
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=300),
    )
    local = torch.full((E,), -1, device="cuda", dtype=torch.int32)
    local[rank * S : (rank + 1) * S] = torch.arange(S, device="cuda", dtype=torch.int32)
    reverse = torch.arange(rank * S, (rank + 1) * S, device="cuda", dtype=torch.int32)
    weights = [weights_for_layer(rank, layer) for layer in range(2)]
    stops = [torch.zeros(1, device="cuda", dtype=torch.uint64) for _ in range(2)]
    contexts = []
    for layer in range(2):
        ctx = create_early_moe_context(
            dist.group.WORLD,
            hidden=H,
            intermediate=I,
            topk=K,
            logical_experts=E,
            local_slots=S,
            max_tokens=128,
            max_inflight_blocks=args.inflight,
            tile_n=256,
            stages=4,
            bulk_prepare=True,
            stop_signal=stops[layer],
            diagnostic=True,
            audit_capacity=128,
            timeout_s=60,
            shared_scratch=None if not contexts else contexts[0].scratch,
        )
        ctx.bind(weights[layer], local, reverse, 1)
        contexts.append(ctx)
    assert contexts[0].scratch is contexts[1].scratch
    assert (
        contexts[0].result.early_values.data_ptr()
        == contexts[1].result.early_values.data_ptr()
    )
    assert contexts[0].peer.flags.data_ptr() != contexts[1].peer.flags.data_ptr()
    gate = PeerEpochs(dist.group.WORLD, contexts[0].lib)
    signal_stream = torch.cuda.Stream()
    data = [
        inputs(rank, 0 if rank == 0 else 63 + rank, 76100),
        inputs(rank, 65 + rank, 76200),
    ]
    torch.cuda.synchronize()
    dist.barrier()
    pending = []
    for layer, ctx in enumerate(contexts):
        epoch = layer + 1
        gate.delay_stop(
            epoch, args.stop_us, ctx.peer.flags, stops[layer], signal_stream
        )
        result = early_moe(
            *data[layer],
            weights[layer],
            local,
            1,
            epoch,
            context=ctx,
            activation_clamp=CLAMP,
        )
        pending.append(snapshot(result, len(data[layer][0])))
    torch.cuda.synchronize()
    reports = [
        check(
            rank,
            f"layer{layer}_shared_arena",
            data[layer],
            weights[layer],
            pending[layer],
            True,
        )
        for layer in range(2)
    ]
    # All results were consumed/copied before second-layer arena reuse. Even
    # after completion a foreign stream must not silently acquire the arena.
    rejected = False
    with torch.cuda.stream(signal_stream):
        try:
            early_moe(
                *data[0],
                weights[0],
                local,
                1,
                3,
                context=contexts[0],
                activation_clamp=CLAMP,
            )
        except RuntimeError as error:
            assert "another CUDA stream" in str(error), str(error)
            rejected = True
    assert rejected
    # Fresh epoch on the correct stream with an already-published stop must
    # produce no stale done bits or inherited counts from either earlier layer.
    ctx = contexts[0]
    ctx.signal_stop(3)
    result = early_moe(
        *data[0], weights[0], local, 1, 3, context=ctx, activation_clamp=CLAMP
    )
    last = snapshot(result, len(data[0][0]))
    torch.cuda.synchronize()
    reports.append(
        check(rank, "presignaled_fresh_epoch", data[0], weights[0], last, False)
    )
    output = Path(args.output)
    output.mkdir(exist_ok=True, parents=True)
    (output / f"rank{rank}.json").write_text(
        json.dumps(
            dict(
                status="passed",
                rank=rank,
                geometry=GEOMETRY,
                shared_arena=True,
                cross_stream_rejected=rejected,
                stop_source="benchmark_external_stop",
                stop_delay_us=args.stop_us,
                launch=contexts[0].launch_info,
                cases=reports,
            ),
            indent=2,
        )
    )
    dist.barrier()
    gate.close()
    for ctx in contexts:
        ctx.close()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(ROOT / "operator_smoke_results"))
    p.add_argument("--stop-us", type=int, default=20000)
    p.add_argument("--inflight", type=int, default=16)
    p.add_argument("--profile", choices=sorted(PROFILES), default="dsv4")
    args = p.parse_args()
    configure(args.profile)
    if Path(args.output).exists() and any(Path(args.output).iterdir()):
        raise RuntimeError(
            "choose a new output directory; refusing to overwrite prior evidence"
        )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(args, port), nprocs=8, join=True)
    rows = [
        json.loads((Path(args.output) / f"rank{rank}.json").read_text())
        for rank in range(8)
    ]
    summary = dict(
        status="passed",
        ranks=8,
        cases=24,
        geometry=GEOMETRY,
        max_assignment_rel_l2=max(
            c["rel_l2"] or 0 for row in rows for c in row["cases"]
        ),
        max_token_merge_rel_l2=max(
            c["token_merge_rel_l2"] or 0 for row in rows for c in row["cases"]
        ),
        exercised_last_physical=(E - 1)
        in rows[7]["cases"][1]["selected_physical_slots"],
        shared_arena_all=all(
            row["shared_arena"] and row["cross_stream_rejected"] for row in rows
        ),
    )
    (Path(args.output) / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
