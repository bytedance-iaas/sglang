#!/usr/bin/env python3
"""Two-H20 component A/B for GLM-5.3 DSA row/head exchange.

Control runs native FlashMLA V32 on every local query row with 32 real heads
padded to 64. Candidate includes both attention-TP all-to-all exchanges and
runs only half the rows with all 64 real heads. This measures the complete DSA
consumer boundary, not service TTFT or throughput.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

os.environ.setdefault("SGLANG_EXPERIMENTAL_DSA_KPOOL_FLASHMLA_V32", "1")

import torch
import torch.distributed as dist

from sglang.kernels.ops.attention.dsa.quant_k_cache import (
    quantize_k_cache_separate,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.attention.dsa.dsa_backend_kpool import (
    DeepseekSparseAttnBackendKPoolMixin,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.test_utils import publish_build_topology


ROWS = 4096
LOCAL_HEADS = 32
FULL_HEADS = 64
HEAD_DIM = 512
HISTORY = 2048
SOURCE_WIDTH = 2051
PHYSICAL_PAGE_SIZE = 64
PHYSICAL_TOKENS = 200_000
ROUNDS = 7
ITERATIONS = 10
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


def make_cache() -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(20261021)
    nope = torch.randn(
        PHYSICAL_TOKENS,
        1,
        HEAD_DIM,
        device="cuda",
        generator=generator,
    )
    nope.view(PHYSICAL_TOKENS, 4, 128).mul_(
        torch.tensor([0.01, 0.5, 2.0, 8.0], device="cuda")[None, :, None]
    )
    packed, packed_rope = quantize_k_cache_separate(
        nope.bfloat16(),
        torch.empty(
            PHYSICAL_TOKENS,
            1,
            0,
            device="cuda",
            dtype=torch.bfloat16,
        ),
    )
    cache = torch.cat((packed, packed_rope), -1).view(torch.float8_e4m3fn)
    assert cache.shape == (PHYSICAL_TOKENS, 1, 528), cache.shape
    return cache


def make_inputs(rank: int):
    cache = make_cache()
    q = (
        torch.randn(
            ROWS,
            LOCAL_HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
            generator=torch.Generator(device="cuda").manual_seed(53 + rank),
        )
        * 0.05
    )
    indices = torch.full(
        (ROWS, SOURCE_WIDTH), -1, device="cuda", dtype=torch.int32
    )
    lengths = torch.empty(ROWS, device="cuda", dtype=torch.int32)
    base = torch.arange(SOURCE_WIDTH, device="cuda", dtype=torch.int32)
    offsets = torch.arange(ROWS, device="cuda", dtype=torch.int32)[:, None] * 2112
    candidates = (base[None, :] + offsets) % PHYSICAL_TOKENS
    for row in range(ROWS):
        length = HISTORY + row % 4
        indices[row, :length] = candidates[row, :length]
        lengths[row] = length
    indices[:, 7] = indices[:, 6]
    return q, cache, indices, lengths


def time_path(run, iterations: int) -> float:
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        run()
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end) / iterations
    dist.barrier()
    return elapsed


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

        q, cache, indices, lengths = make_inputs(rank)
        backend = SimpleNamespace(
            dsa_index_kpool=4,
            device_sm_major=9,
            dsa_kv_cache_store_fp8=True,
            qk_rope_head_dim=0,
            flashmla_kv_num_q_heads=FULL_HEADS,
            physical_page_size=PHYSICAL_PAGE_SIZE,
            kv_cache_dim=528,
            dsa_index_topk=HISTORY,
            flashmla_kv_format="V32_NO_ROPE",
        )
        backend._use_kpool_flashmla_v32 = MethodType(
            DeepseekSparseAttnBackendKPoolMixin._use_kpool_flashmla_v32, backend
        )
        control_metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=lengths,
            flashmla_metadata=SimpleNamespace(
                flashmla_metadata=None, num_splits=None
            ),
            kpool_flashmla_sched_meta={},
        )
        candidate_metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=lengths,
            flashmla_metadata=control_metadata.flashmla_metadata,
            kpool_flashmla_sched_meta={},
        )
        layer = SimpleNamespace(tp_q_head_num=LOCAL_HEADS, head_dim=HEAD_DIM)
        scale = HEAD_DIM**-0.5

        def control_path():
            return DeepseekSparseAttnBackend._forward_flashmla_kv(
                backend,
                q_all=q,
                kv_cache=cache,
                v_head_dim=HEAD_DIM,
                sm_scale=scale,
                layer=layer,
                metadata=control_metadata,
                page_table_1=indices,
            )

        def candidate_path():
            return DeepseekSparseAttnBackend._forward_flashmla_kv(
                backend,
                q_all=q,
                kv_cache=cache,
                v_head_dim=HEAD_DIM,
                sm_scale=scale,
                layer=layer,
                metadata=candidate_metadata,
                page_table_1=indices,
                attn_tp_row_head_exchange=True,
            )

        for _ in range(3):
            control_path()
            candidate_path()
        torch.cuda.synchronize()

        control_value = control_path().float()
        candidate_value = candidate_path().float()
        torch.cuda.synchronize()
        abs_error = (candidate_value - control_value).abs()
        torch.testing.assert_close(
            candidate_value, control_value, rtol=RTOL, atol=ATOL
        )

        control_ms: list[float] = []
        candidate_ms: list[float] = []
        records = []
        for round_index in range(ROUNDS):
            order = (
                (("control", control_path), ("candidate", candidate_path))
                if round_index % 2 == 0
                else (("candidate", candidate_path), ("control", control_path))
            )
            record = {"round": round_index, "order": [name for name, _ in order]}
            for name, run in order:
                elapsed = time_path(run, ITERATIONS)
                record[f"{name}_ms"] = elapsed
                (control_ms if name == "control" else candidate_ms).append(elapsed)
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
                "max_abs": float(abs_error.max()),
                "mean_abs": float(abs_error.mean()),
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
            paired_reductions = [
                100
                * (
                    1
                    - item["measurement"]["candidate_stats_ms"]["median"]
                    / item["measurement"]["control_stats_ms"]["median"]
                )
                for item in gathered
            ]
            result = {
                "schema": "glm53_dsa_attn_tp_row_head_sm90_component_v1",
                "success": True,
                "timestamp": time.time(),
                "scope": "native V32 DSA consumer plus two attention-TP A2A exchanges; not service TTFT or throughput",
                "source": {"commit": args.source_commit, "tree": args.source_tree},
                "runtime": {
                    "torch": torch.__version__,
                    "sglang_kernel": importlib.metadata.version("sglang-kernel"),
                    "world_size": world_size,
                },
                "shape": {
                    "rows_control": ROWS,
                    "rows_candidate_per_rank": ROWS // world_size,
                    "local_heads_control": LOCAL_HEADS,
                    "real_heads_candidate": FULL_HEADS,
                    "head_dim": HEAD_DIM,
                    "history": HISTORY,
                    "source_width": SOURCE_WIDTH,
                    "physical_tokens": PHYSICAL_TOKENS,
                    "cache_bytes_per_token": 528,
                },
                "rank_results": gathered,
                "rank_median_reduction_pct": paired_reductions,
                "all_ranks_candidate_faster": all(v > 0 for v in paired_reductions),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
