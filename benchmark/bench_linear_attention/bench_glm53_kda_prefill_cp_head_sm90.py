#!/usr/bin/env python3
"""Two-H20 complete component A/B for GLM-5.3 KDA CP head ownership.

Control reproduces the current CP bridge: three CP all-gathers, H32 convolution
and recurrence on both ranks, then local interleave sharding.  Candidate calls
the production P2 method: one packed input A2A, H16 full-sequence convolution
and recurrence, replicated final/named conv+FP32 state restoration, and output
A2A.  The benchmark is a component gate, not a service TTFT claim.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.layers.attention.linear.kda_prefill_cp_head import (
    plan_kda_prefill_cp_head,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import publish_build_topology

LOCAL_ROWS = 4096
GLOBAL_ROWS = 8192
HEADS = 32
OWNED_HEADS = 16
HEAD_DIM = 128
CONV_HISTORY = 3
STATE_ROWS = 2
ROUNDS = 7
ITERATIONS = 5
RTOL = 0.02
ATOL = 0.03


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values),
    }


def all_gather_interleave(x: torch.Tensor) -> torch.Tensor:
    group = get_parallel().attn_cp_group
    gathered = group.all_gather(x.contiguous(), dim=0)
    return (
        gathered.reshape(2, LOCAL_ROWS, *x.shape[1:])
        .transpose(0, 1)
        .reshape(GLOBAL_ROWS, *x.shape[1:])
    )


def make_inputs(rank: int):
    generator = torch.Generator(device="cuda").manual_seed(20261021)
    qkv_global = (
        torch.randn(
            GLOBAL_ROWS,
            3 * HEADS * HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.05
    )
    gate_global = (
        torch.randn(
            1,
            GLOBAL_ROWS,
            HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    beta_global = (
        torch.randn(
            1,
            GLOBAL_ROWS,
            HEADS,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    conv_weight = (
        torch.randn(
            3 * HEADS * HEAD_DIM,
            CONV_HISTORY + 1,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.02
    )
    conv_bias = (
        torch.randn(
            3 * HEADS * HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    )
    a_log = (
        torch.randn(
            1, 1, HEADS, 1, device="cuda", dtype=torch.float32, generator=generator
        )
        * 0.1
        - 1.5
    )
    dt_bias = (
        torch.randn(
            HEADS * HEAD_DIM,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.05
    )
    conv_initial = (
        torch.randn(
            STATE_ROWS,
            CONV_HISTORY,
            3 * HEADS * HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.02
    )
    state_initial = (
        torch.randn(
            STATE_ROWS,
            HEADS,
            HEAD_DIM,
            HEAD_DIM,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.001
    )
    return {
        "qkv_local": qkv_global[rank::2].contiguous(),
        "gate_local": gate_global[:, rank::2].contiguous(),
        "beta_local": beta_global[:, rank::2].contiguous(),
        "conv_weight": conv_weight,
        "conv_bias": conv_bias,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "conv_initial": conv_initial,
        "state_initial": state_initial,
    }


def make_layer(inputs):
    return SimpleNamespace(
        layer_id=0,
        num_q_heads=HEADS,
        num_k_heads=HEADS,
        num_v_heads=HEADS,
        head_q_dim=HEAD_DIM,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        q_dim=HEADS * HEAD_DIM,
        k_dim=HEADS * HEAD_DIM,
        v_dim=HEADS * HEAD_DIM,
        conv_weights=inputs["conv_weight"],
        bias=inputs["conv_bias"],
        A_log=inputs["a_log"],
        dt_bias=inputs["dt_bias"],
        lower_bound=None,
    )


def make_backend(inputs):
    backend = object.__new__(KDAAttnBackend)
    cache = SimpleNamespace(
        conv=[inputs["conv_initial"].clone()],
        temporal=inputs["state_initial"].clone(),
    )
    backend.req_to_token_pool = SimpleNamespace(
        mamba2_layer_cache=lambda _layer_id: cache
    )
    backend.kernel_dispatcher = TritonKDAKernel()
    backend.forward_metadata = SimpleNamespace(
        query_start_loc=torch.tensor(
            [0, GLOBAL_ROWS], device="cuda", dtype=torch.int32
        ),
        mamba_cache_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
        has_mamba_track_mask=True,
        conv_states_mask_indices=torch.tensor([1], device="cuda", dtype=torch.long),
        track_conv_indices=torch.tensor(
            [[GLOBAL_ROWS - 3, GLOBAL_ROWS - 2, GLOBAL_ROWS - 1]],
            device="cuda",
            dtype=torch.long,
        ),
        track_chunk_idx=None,
        track_ssm_h_src=torch.empty(0, device="cuda", dtype=torch.long),
        track_ssm_h_dst=torch.empty(0, device="cuda", dtype=torch.long),
        track_ssm_h_batch_src=torch.empty(0, device="cuda", dtype=torch.long),
        track_ssm_final_src=torch.tensor([0], device="cuda", dtype=torch.long),
        track_ssm_final_dst=torch.tensor([1], device="cuda", dtype=torch.long),
        track_ssm_recompute_dst=None,
    )
    return backend, cache


def reset_cache(cache, inputs):
    cache.conv[0].copy_(inputs["conv_initial"])
    cache.temporal.copy_(inputs["state_initial"])


def run_control(inputs, layer, cache):
    qkv = all_gather_interleave(inputs["qkv_local"])
    gate = all_gather_interleave(inputs["gate_local"].squeeze(0)).unsqueeze(0)
    beta = all_gather_interleave(inputs["beta_local"].squeeze(0)).unsqueeze(0)
    cache.conv[0][1] = qkv[-3:]
    qkv = causal_conv1d_fn(
        qkv.transpose(0, 1),
        layer.conv_weights,
        layer.bias,
        activation="silu",
        conv_states=cache.conv[0].transpose(-1, -2),
        has_initial_state=torch.tensor([True], device="cuda"),
        cache_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
        query_start_loc=torch.tensor(
            [0, GLOBAL_ROWS], device="cuda", dtype=torch.int32
        ),
        seq_lens_cpu=[GLOBAL_ROWS],
    ).transpose(0, 1)
    q, k, v = qkv.split([HEADS * HEAD_DIM] * 3, dim=-1)
    output = TritonKDAKernel().extend(
        q=q.unflatten(-1, (-1, HEAD_DIM)).unsqueeze(0),
        k=k.unflatten(-1, (-1, HEAD_DIM)).unsqueeze(0),
        v=v.unflatten(-1, (-1, HEAD_DIM)).unsqueeze(0),
        g=gate,
        beta=beta,
        ssm_states=cache.temporal,
        cache_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
        query_start_loc=torch.tensor(
            [0, GLOBAL_ROWS], device="cuda", dtype=torch.int32
        ),
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        beta_is_raw=False,
        return_intermediate_states=False,
    )
    cache.temporal[1].copy_(cache.temporal[0])
    return output[:, get_parallel().attn_cp_rank :: 2].contiguous()


def time_path(run, reset) -> list[float]:
    values = []
    for _ in range(ITERATIONS):
        reset()
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    dist.barrier()
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise SystemExit(f"requires exactly two ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(world_rank=rank, tp_size=world_size)
    initialize_model_parallel()

    try:
        if torch.cuda.get_device_capability(local_rank) != (9, 0):
            raise RuntimeError(
                f"requires SM90, got {torch.cuda.get_device_capability(local_rank)}"
            )
        inputs = make_inputs(rank)
        layer = make_layer(inputs)
        control_backend, control_cache = make_backend(inputs)
        candidate_backend, candidate_cache = make_backend(inputs)
        del control_backend
        plan = plan_kda_prefill_cp_head(
            local_rows=LOCAL_ROWS,
            local_heads=HEADS,
            logical_rank_rows=[LOCAL_ROWS, LOCAL_ROWS],
            physical_rank_rows=[LOCAL_ROWS, LOCAL_ROWS],
            cp_size=2,
            cp_rank=rank,
        )
        assert plan is not None
        forward_batch = SimpleNamespace(
            extend_prefix_lens=torch.tensor([1], device="cuda", dtype=torch.int32),
            extend_seq_lens_cpu=[GLOBAL_ROWS],
        )

        def control():
            return run_control(inputs, layer, control_cache)

        def candidate():
            return KDAAttnBackend._forward_extend_cp_head_exchange(
                candidate_backend,
                layer,
                forward_batch,
                inputs["qkv_local"],
                inputs["gate_local"],
                inputs["beta_local"],
                plan,
            )

        with get_parallel().override(
            attn_cp_group=get_parallel().tp_group,
            attn_cp_size=2,
            attn_cp_rank=rank,
        ):
            for _ in range(3):
                reset_cache(control_cache, inputs)
                control()
                reset_cache(candidate_cache, inputs)
                candidate()
            torch.cuda.synchronize()

            reset_cache(control_cache, inputs)
            control_output = control().float()
            control_state = control_cache.temporal.float().clone()
            control_conv = control_cache.conv[0].float().clone()
            reset_cache(candidate_cache, inputs)
            candidate_output = candidate().float()
            candidate_state = candidate_cache.temporal.float().clone()
            candidate_conv = candidate_cache.conv[0].float().clone()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                candidate_output, control_output, rtol=RTOL, atol=ATOL
            )
            torch.testing.assert_close(
                candidate_state, control_state, rtol=RTOL, atol=ATOL
            )
            torch.testing.assert_close(
                candidate_conv, control_conv, rtol=RTOL, atol=ATOL
            )

            control_ms = []
            candidate_ms = []
            records = []
            for round_index in range(ROUNDS):
                order = (
                    (("control", control), ("candidate", candidate))
                    if round_index % 2 == 0
                    else (("candidate", candidate), ("control", control))
                )
                record = {"round": round_index, "order": [name for name, _ in order]}
                for name, run in order:
                    cache = control_cache if name == "control" else candidate_cache
                    values = time_path(
                        run, lambda cache=cache: reset_cache(cache, inputs)
                    )
                    record[f"{name}_ms"] = statistics.median(values)
                    record[f"{name}_samples_ms"] = values
                    (control_ms if name == "control" else candidate_ms).append(
                        record[f"{name}_ms"]
                    )
                record["candidate_reduction_pct"] = (
                    1 - record["candidate_ms"] / record["control_ms"]
                ) * 100
                records.append(record)

        local = {
            "rank": rank,
            "device": torch.cuda.get_device_name(local_rank),
            "capability": list(torch.cuda.get_device_capability(local_rank)),
            "correctness": {
                "rtol": RTOL,
                "atol": ATOL,
                "output_max_abs": float(
                    (candidate_output - control_output).abs().max()
                ),
                "state_max_abs": float((candidate_state - control_state).abs().max()),
                "conv_max_abs": float((candidate_conv - control_conv).abs().max()),
            },
            "measurement": {
                "control_ms": control_ms,
                "candidate_ms": candidate_ms,
                "control_stats_ms": distribution(control_ms),
                "candidate_stats_ms": distribution(candidate_ms),
                "round_records": records,
            },
        }
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local, gathered, dst=0)
        if rank == 0:
            reductions = [
                100
                * (
                    1
                    - item["measurement"]["candidate_stats_ms"]["median"]
                    / item["measurement"]["control_stats_ms"]["median"]
                )
                for item in gathered
            ]
            result = {
                "schema": "glm53_kda_prefill_cp_head_sm90_component_v1",
                "success": True,
                "timestamp": time.time(),
                "scope": "three CP gathers plus H32 conv/recurrence/localize versus packed input A2A plus H16 conv/recurrence, final+named state restore, and output A2A; not service TTFT",
                "source": {"commit": args.source_commit, "tree": args.source_tree},
                "runtime": {
                    "torch": torch.__version__,
                    "sglang_kernel": importlib.metadata.version("sglang-kernel"),
                    "world_size": world_size,
                },
                "shape": {
                    "local_rows": LOCAL_ROWS,
                    "global_rows": GLOBAL_ROWS,
                    "control_heads_per_rank": HEADS,
                    "candidate_owned_heads_per_rank": OWNED_HEADS,
                    "head_dim": HEAD_DIM,
                    "conv_history": CONV_HISTORY,
                    "persistent_state_rows": 1,
                    "named_checkpoint_rows": 1,
                },
                "rank_results": gathered,
                "rank_median_reduction_pct": reductions,
                "all_ranks_candidate_faster": all(value > 0 for value in reductions),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
