#!/usr/bin/env python3
"""Multi-cycle SiDP DMA Graph benchmark with controlled BF16 GEMM compute.

This is intentionally separate from ``bench_dma_cycle.py``.  It compares the
production-style fixed compute-order DMA/Event pipeline, slice-major S=2/S=4
variants, the same fixed DMA order with generation flags, and dynamic-owner
conditional DMA/generation flags while a serial chain of configurable GEMMs
runs on the compute stream.

The default graph models Gemma4's six cycles and two-cycle rolling cache:
cycle 0 is resident, cycle 1 is prefetched at graph start, cycle c+2 is issued
after cycle c releases its final remote slot, and next-forward cycle 0 is
refilled from cycle 4 before the final stream join.  The GEMM does not consume
the copied bytes; the Event/flag dependency still gates each remote layer.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import random
import shlex
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

SGLANG_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SGLANG_ROOT / "python"))

DYNAMIC_DMA_MODES = ("dynamic_dma", "dynamic_dma_compute_priority")
DYNAMIC_SM_MODES = ("dynamic_sm", "dynamic_sm_compute_priority")
DYNAMIC_MODES = (*DYNAMIC_DMA_MODES, *DYNAMIC_SM_MODES)
FIXED_SM_MODES = ("compute_sm_flag",)
SM_MODES = (*FIXED_SM_MODES, *DYNAMIC_SM_MODES)
SLICE_MODE_FACTORS = {
    "compute_dma_slice2": 2,
    "compute_dma_slice4": 4,
    "compute_dma_slice4_group2": 4,
    "compute_dma_slice4_group4": 4,
    "compute_dma_slice4_flag": 4,
    "compute_dma_slice4_group2_flag": 4,
    "compute_dma_slice4_group4_flag": 4,
}
GROUP_MODE_COUNTS = {
    "compute_dma_slice4_group2": 2,
    "compute_dma_slice4_group4": 4,
    "compute_dma_slice4_group2_flag": 2,
    "compute_dma_slice4_group4_flag": 4,
}
EVENT_MODES = (
    "compute_dma",
    "compute_dma_slice2",
    "compute_dma_slice4",
    "compute_dma_slice4_group2",
    "compute_dma_slice4_group4",
)
FIXED_FLAG_MODES = (
    "compute_dma_flag",
    "compute_dma_slice4_flag",
    "compute_dma_slice4_group2_flag",
    "compute_dma_slice4_group4_flag",
)
FIXED_MODES = (*EVENT_MODES, *FIXED_FLAG_MODES)
MODES = ("compute_only", "compute_dma", "compute_dma_flag", *DYNAMIC_DMA_MODES)
SLICE_STUDY_MODES = (
    "compute_only",
    "compute_dma",
    "compute_dma_slice2",
    "compute_dma_slice4",
)
GROUP_STUDY_MODES = (
    "compute_dma",
    "compute_dma_slice4",
    "compute_dma_slice4_group2",
    "compute_dma_slice4_group4",
    "compute_dma_flag",
    "compute_dma_slice4_flag",
    "compute_dma_slice4_group2_flag",
    "compute_dma_slice4_group4_flag",
)
FLAG_MODES = (*FIXED_FLAG_MODES, *SM_MODES, *DYNAMIC_DMA_MODES)
ALL_MODES = tuple(
    dict.fromkeys((*MODES, *SM_MODES, *SLICE_STUDY_MODES, *GROUP_STUDY_MODES))
)
CLAIM_ORDER = {
    "dynamic_dma": 0,
    "dynamic_dma_compute_priority": 1,
    "dynamic_sm": 0,
    "dynamic_sm_compute_priority": 1,
}
CACHE_DEPTH = 2
SM_TRACE_FIELDS = 6


def csv_numbers(raw, cast, minimum, option):
    try:
        values = [cast(item.strip()) for item in raw.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{option}: invalid numeric list") from error
    if not values or any(not math.isfinite(v) or v < minimum for v in values):
        raise argparse.ArgumentTypeError(f"{option}: values must be >= {minimum}")
    return values


def is_local_position(position, rank, world_size, k):
    return any(position == (rank + replica) % world_size for replica in range(k))


def candidates(rank, world_size, k):
    """Remote positions in the production compute order."""
    return [
        position
        for position in range(world_size)
        if not is_local_position(position, rank, world_size, k)
    ]


def slot_for(cycle, candidate_index, candidate_count):
    return (cycle % CACHE_DEPTH) * candidate_count + candidate_index


def fill_generation(cycle):
    return cycle // CACHE_DEPTH + 1


def reuse_requirement(cycle):
    return cycle // CACHE_DEPTH


def next_forward_generations(cycles):
    required = cycles // CACHE_DEPTH
    return required, required + 1


def slice_bounds(size, slice_factor, slice_index):
    """Return an exact, gap-free byte range for one component slice."""
    return (
        size * slice_index // slice_factor,
        size * (slice_index + 1) // slice_factor,
    )


def owner_group_ranges(owner_count, group_count):
    """Split ordered owners into up to group_count contiguous, front-heavy groups."""
    width = (owner_count + group_count - 1) // group_count
    return [
        (begin, min(begin + width, owner_count))
        for begin in range(0, owner_count, width)
    ]


def communication_sequence(cycles):
    """Communication operations in comm-stream order for one forward."""
    return [*range(1, cycles), 0]


def operation_name(cycle):
    return "next_c0" if cycle == 0 else f"c{cycle}"


def planned_offsets(scenario, seed, sample, world_size):
    """Return one paired launch-offset vector for every compared mode.

    Random scenarios redraw every rank independently for every sample.  The
    minimum is removed because only relative DP arrival time matters.
    """
    rng = random.Random(seed + scenario["index"] * 1_000_003 + sample)
    offsets = [rng.uniform(0, scenario["max_us"]) for _ in range(world_size)]
    first = min(offsets)
    return [offset - first for offset in offsets]


def planned_continuous_gaps(scenario, seed, replay_count, world_size):
    """Return paired per-rank launch gaps for one continuous epoch.

    Replay 0 uses a relative initial offset. Every later value is an
    independent scheduler gap after that rank's preceding replay completes.
    The complete plan is deterministic and reused by every compared mode.
    """
    initial = planned_offsets(scenario, seed, -2_000_000, world_size)
    gaps = [[initial[rank]] for rank in range(world_size)]
    for replay in range(1, replay_count):
        for rank in range(world_size):
            rng = random.Random(
                seed
                + scenario["index"] * 1_000_003
                + replay * 10_000_019
                + rank * 1_000_000_007
            )
            gaps[rank].append(rng.uniform(0, scenario["max_us"]))
    return gaps


def delay_label(scenario, arrival_model="reset_uniform"):
    if arrival_model == "continuous":
        return f"continuous per-rank gap U[0,{scenario['max_us']}]us/replay"
    return f"independent U[0,{scenario['max_us']}]us/sample"


def continuous_epoch_metrics(rank_replays):
    """Summarize one epoch without synchronizing between replays."""
    replay_count = len(rank_replays[0])
    rank_count = len(rank_replays)
    first_launch = min(rows[0]["launch_begin_ns"] for rows in rank_replays)
    last_done = max(rows[-1]["completion_observed_ns"] for rows in rank_replays)
    wall_ms = (last_done - first_launch) / 1e6
    rank_wall_ms = [
        (rows[-1]["completion_observed_ns"] - rows[0]["launch_begin_ns"]) / 1e6
        for rows in rank_replays
    ]
    rank_rates = [replay_count * 1000 / value for value in rank_wall_ms]
    launch_spans = [
        (
            max(
                rank_replays[rank][replay]["launch_begin_ns"]
                for rank in range(rank_count)
            )
            - min(
                rank_replays[rank][replay]["launch_begin_ns"]
                for rank in range(rank_count)
            )
        )
        / 1e3
        for replay in range(replay_count)
    ]
    scheduler_gaps = [
        row["requested_offset_us"] for rows in rank_replays for row in rows[1:]
    ]
    return {
        "replays_per_rank": replay_count,
        "total_rank_replays": replay_count * rank_count,
        "epoch_wall_ms": wall_ms,
        "aggregate_rank_replays_per_second": (
            replay_count * rank_count * 1000 / wall_ms
        ),
        "rank_replays_per_second": {
            "min": min(rank_rates),
            "median": statistics.median(rank_rates),
            "max": max(rank_rates),
        },
        "launch_span_us": {
            "initial": launch_spans[0],
            "final": launch_spans[-1],
            "p50": statistics.median(launch_spans),
            "p95": percentile(launch_spans, 0.95),
            "max": max(launch_spans),
        },
        "scheduler_gap_us": {
            "p50": statistics.median(scheduler_gaps) if scheduler_gaps else 0.0,
            "p95": percentile(scheduler_gaps, 0.95) if scheduler_gaps else 0.0,
            "max": max(scheduler_gaps) if scheduler_gaps else 0.0,
        },
    }


def continuous_throughput_ratios(epoch_metrics):
    """Report the requested controlled comparisons as throughput ratios."""
    rates = {
        mode: values["aggregate_rank_replays_per_second"]
        for mode, values in epoch_metrics.items()
    }
    pairs = (
        ("compute_dma_slice4", "compute_dma", "event_s4_to_s1"),
        ("compute_dma_slice4_flag", "compute_dma_flag", "flag_s4_to_s1"),
        ("compute_dma_flag", "compute_dma", "s1_flag_to_event"),
        (
            "compute_dma_slice4_flag",
            "compute_dma_slice4",
            "s4_flag_to_event",
        ),
    )
    return {
        name: rates[candidate] / rates[reference]
        for candidate, reference, name in pairs
        if candidate in rates and reference in rates
    }


def _percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize_sm_execution_trace(
    rows, count, overflow, metadata, device_sm_count, capacity
):
    """Compact CTA-level `%smid/globaltimer` rows for JSON transport.

    The raw tensor stays on the worker and is never put in the benchmark JSON.
    Exact physical SM association is available for these instrumented kernels;
    opaque GEMM kernels remain intentionally uninstrumented.
    """
    if overflow:
        raise RuntimeError(
            f"SM execution trace overflowed: count={count}, capacity={capacity}"
        )
    valid_count = min(int(count), int(capacity))
    by_tag = {}
    for row in rows[:valid_count]:
        if len(row) != SM_TRACE_FIELDS:
            continue
        tag, block, entry_smid, exit_smid, begin_ns, end_ns = map(int, row)
        if tag < 0 or begin_ns <= 0 or end_ns < begin_ns:
            continue
        by_tag.setdefault(tag, []).append(
            {
                "block": block,
                "entry_smid": entry_smid,
                "exit_smid": exit_smid,
                "begin_ns": begin_ns,
                "end_ns": end_ns,
            }
        )
    metadata_by_tag = {int(item["tag"]): item for item in metadata}
    launches = []
    role_rows = {}
    for tag, records in sorted(by_tag.items()):
        meta = metadata_by_tag.get(
            tag, {"tag": tag, "role": "unknown", "kernel": "unknown"}
        )
        entry_smids = sorted({item["entry_smid"] for item in records})
        exit_smids = sorted({item["exit_smid"] for item in records})
        durations = [item["end_ns"] - item["begin_ns"] for item in records]
        launch = {
            **meta,
            "cta_records": len(records),
            "entry_smids": entry_smids,
            "exit_smids": exit_smids,
            "smid_migrations": sum(
                item["entry_smid"] != item["exit_smid"] for item in records
            ),
            "first_entry_globaltimer_ns": min(item["begin_ns"] for item in records),
            "last_exit_globaltimer_ns": max(item["end_ns"] for item in records),
            "cta_duration_mean_ns": statistics.mean(durations),
            "cta_duration_p95_ns": _percentile(durations, 0.95),
        }
        launches.append(launch)
        role = str(meta.get("role", "unknown"))
        aggregate = role_rows.setdefault(
            role,
            {
                "role": role,
                "kernel_launches": 0,
                "cta_records": 0,
                "smids": set(),
                "smid_migrations": 0,
            },
        )
        aggregate["kernel_launches"] += 1
        aggregate["cta_records"] += len(records)
        aggregate["smids"].update(entry_smids)
        aggregate["smids"].update(exit_smids)
        aggregate["smid_migrations"] += launch["smid_migrations"]
    roles = []
    for aggregate in role_rows.values():
        smids = sorted(aggregate.pop("smids"))
        roles.append(
            {
                **aggregate,
                "smids": smids,
                "unique_sm_count": len(smids),
                "device_sm_coverage": (
                    len(smids) / device_sm_count if device_sm_count else 0.0
                ),
            }
        )
    return {
        "schema": "tag,block,entry_smid,exit_smid,entry_globaltimer_ns,exit_globaltimer_ns",
        "record_count": valid_count,
        "capacity": int(capacity),
        "overflow": bool(overflow),
        "device_sm_count": int(device_sm_count),
        "roles": sorted(roles, key=lambda item: item["role"]),
        "launches": launches,
        "limitation": (
            "SMIDs are exact for instrumented communication/control CTAs. "
            "They do not reveal which SMs executed the opaque GEMM; correlate "
            "with Nsight Systems overlap and NCU unit rollups."
        ),
    }


def set_cuda_profiler(torch, enabled):
    """Start/stop an external profiler capture from the rank-0 controller."""
    cudart = torch.cuda.cudart()
    if enabled:
        cudart.cudaProfilerStart()
    else:
        cudart.cudaProfilerStop()


def graph_ratios(graph_p50):
    ratios = {}
    if "compute_only" in graph_p50:
        ratios.update(
            {
                f"{mode}_to_compute_only_graph": value
                / graph_p50["compute_only"]
                for mode, value in graph_p50.items()
                if mode != "compute_only"
            }
        )
    if "compute_dma" in graph_p50:
        for mode in SLICE_MODE_FACTORS:
            if mode in graph_p50:
                ratios[f"{mode}_to_compute_dma_graph"] = (
                    graph_p50[mode] / graph_p50["compute_dma"]
                )
    if "compute_dma_flag" in graph_p50 and "compute_dma" in graph_p50:
        ratios["fixed_flag_to_event_graph"] = (
            graph_p50["compute_dma_flag"] / graph_p50["compute_dma"]
        )
    if "dynamic_dma" in graph_p50 and "compute_dma_flag" in graph_p50:
        ratios["dynamic_to_fixed_flag_graph"] = (
            graph_p50["dynamic_dma"] / graph_p50["compute_dma_flag"]
        )
        if "compute_dma" in graph_p50:
            ratios["dynamic_to_event_graph"] = (
                graph_p50["dynamic_dma"] / graph_p50["compute_dma"]
            )
    if (
        "dynamic_dma_compute_priority" in graph_p50
        and "compute_dma_flag" in graph_p50
    ):
        ratios["compute_priority_to_fixed_flag_graph"] = (
            graph_p50["dynamic_dma_compute_priority"]
            / graph_p50["compute_dma_flag"]
        )
        if "dynamic_dma" in graph_p50:
            ratios["compute_priority_to_rotating_graph"] = (
                graph_p50["dynamic_dma_compute_priority"]
                / graph_p50["dynamic_dma"]
            )
    for reference, modes in (
        (
            "compute_dma_slice4",
            ("compute_dma_slice4_group2", "compute_dma_slice4_group4"),
        ),
        (
            "compute_dma_slice4_flag",
            (
                "compute_dma_slice4_group2_flag",
                "compute_dma_slice4_group4_flag",
            ),
        ),
    ):
        if reference in graph_p50:
            for mode in modes:
                if mode in graph_p50:
                    ratios[f"{mode}_to_{reference}_graph"] = (
                        graph_p50[mode] / graph_p50[reference]
                    )
    for event_mode, flag_mode in (
        ("compute_dma", "compute_dma_flag"),
        ("compute_dma_slice4", "compute_dma_slice4_flag"),
        ("compute_dma_slice4_group2", "compute_dma_slice4_group2_flag"),
        ("compute_dma_slice4_group4", "compute_dma_slice4_group4_flag"),
    ):
        if event_mode in graph_p50 and flag_mode in graph_p50:
            ratios[f"{flag_mode}_to_{event_mode}_graph"] = (
                graph_p50[flag_mode] / graph_p50[event_mode]
            )
    # Keep the original JSON keys for consumers of the pre-slicing benchmark.
    legacy_compute_only_names = {
        "compute_dma": "event_to_compute_only_graph",
        "compute_dma_flag": "fixed_flag_to_compute_only_graph",
        "dynamic_dma": "dynamic_to_compute_only_graph",
        "dynamic_dma_compute_priority": "compute_priority_to_compute_only_graph",
    }
    for mode, name in legacy_compute_only_names.items():
        if mode in graph_p50 and "compute_only" in graph_p50:
            ratios[name] = graph_p50[mode] / graph_p50["compute_only"]
    return ratios


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def compute_boundary_metrics(start, compute_end, cycle_events, comm_cycles):
    """Account for every compute-path interval using already-recorded Events."""
    cycle_elapsed = [begin.elapsed_time(end) for begin, end in cycle_events]
    pre_cycle0 = start.elapsed_time(cycle_events[0][0])
    inter_cycle = [
        cycle_events[index][1].elapsed_time(cycle_events[index + 1][0])
        for index in range(len(cycle_events) - 1)
    ]
    post_cycle = cycle_events[-1][1].elapsed_time(compute_end)
    compute_path = start.elapsed_time(compute_end)
    boundary_total = pre_cycle0 + sum(inter_cycle) + post_cycle
    accounted = sum(cycle_elapsed) + boundary_total
    result = {
        "pre_cycle0_gap_ms": float(pre_cycle0),
        "inter_cycle_gap_ms": [float(value) for value in inter_cycle],
        "post_cycle_gap_ms": float(post_cycle),
        "boundary_gap_total_ms": float(boundary_total),
        "cycle_sum_ms": float(sum(cycle_elapsed)),
        "accounted_compute_path_ms": float(accounted),
        "residual_ms": float(compute_path - accounted),
    }
    if 1 in comm_cycles:
        c1_start = comm_cycles[1][0]
        c1_start_from_graph = start.elapsed_time(c1_start)
        c0_start_from_graph = pre_cycle0
        result.update(
            {
                "c1_start_from_graph_ms": float(c1_start_from_graph),
                "c0_start_minus_c1_start_ms": float(
                    c0_start_from_graph - c1_start_from_graph
                ),
            }
        )
    return result


def trial_metrics(ranks):
    first_launch = min(rank["launch_begin_ns"] for rank in ranks)
    last_launch = max(rank["launch_begin_ns"] for rank in ranks)
    last_done = max(rank["completion_observed_ns"] for rank in ranks)
    graph_ms = [rank["graph_ms"] for rank in ranks]
    compute_path_ms = [rank["compute_path_ms"] for rank in ranks]
    tail_join_ms = [rank["tail_join_ms"] for rank in ranks]
    return {
        "max_rank_graph_ms": max(graph_ms),
        "mean_rank_graph_ms": statistics.mean(graph_ms),
        "max_rank_compute_path_ms": max(compute_path_ms),
        "max_rank_tail_join_ms": max(tail_join_ms),
        "actual_host_launch_span_us": (last_launch - first_launch) / 1e3,
        "max_host_launch_lateness_us": max(
            rank["launch_lateness_us"] for rank in ranks
        ),
        "all_rank_observed_wall_ms": (last_done - first_launch) / 1e6,
    }


def summarize(records, world_size):
    result = {}
    for field in records[0]["metrics"]:
        values = [record["metrics"][field] for record in records]
        result[field] = {
            "p50": statistics.median(values),
            "p95": percentile(values, 0.95),
            "min": min(values),
            "max": max(values),
        }
    result["rank_graph_ms_p50"] = [
        statistics.median(record["ranks"][rank]["graph_ms"] for record in records)
        for rank in range(world_size)
    ]
    result["rank_compute_path_ms_p50"] = [
        statistics.median(
            record["ranks"][rank]["compute_path_ms"] for record in records
        )
        for rank in range(world_size)
    ]
    result["cycle_compute_ms_p50"] = [
        statistics.median(
            rank["compute_cycles"][cycle]["elapsed_ms"]
            for record in records
            for rank in record["ranks"]
        )
        for cycle in range(len(records[0]["ranks"][0]["compute_cycles"]))
    ]
    result["cycle_compute_max_rank_ms_p50"] = [
        statistics.median(
            max(rank["compute_cycles"][cycle]["elapsed_ms"] for rank in record["ranks"])
            for record in records
        )
        for cycle in range(len(records[0]["ranks"][0]["compute_cycles"]))
    ]
    boundary_scalars = (
        "pre_cycle0_gap_ms",
        "post_cycle_gap_ms",
        "boundary_gap_total_ms",
        "cycle_sum_ms",
        "accounted_compute_path_ms",
        "residual_ms",
    )
    if "c1_start_from_graph_ms" in records[0]["ranks"][0]["compute_boundaries"]:
        boundary_scalars += (
            "c1_start_from_graph_ms",
            "c0_start_minus_c1_start_ms",
        )
    result["compute_boundary_max_rank_ms_p50"] = {
        field: statistics.median(
            max(rank["compute_boundaries"][field] for rank in record["ranks"])
            for record in records
        )
        for field in boundary_scalars
    }
    inter_count = len(
        records[0]["ranks"][0]["compute_boundaries"]["inter_cycle_gap_ms"]
    )
    result["compute_boundary_max_rank_ms_p50"]["inter_cycle_gap_ms"] = [
        statistics.median(
            max(
                rank["compute_boundaries"]["inter_cycle_gap_ms"][index]
                for rank in record["ranks"]
            )
            for record in records
        )
        for index in range(inter_count)
    ]
    if records[0]["ranks"][0]["comm_cycles"]:
        operation_count = len(records[0]["ranks"][0]["comm_cycles"])
        result["comm_cycle_ms_p50"] = [
            {
                "operation": records[0]["ranks"][0]["comm_cycles"][index]["operation"],
                "elapsed_ms": statistics.median(
                    rank["comm_cycles"][index]["elapsed_ms"]
                    for record in records
                    for rank in record["ranks"]
                ),
                "max_rank_ms_p50": statistics.median(
                    max(
                        rank["comm_cycles"][index]["elapsed_ms"]
                        for rank in record["ranks"]
                    )
                    for record in records
                ),
            }
            for index in range(operation_count)
        ]
    result["samples"] = len(records)
    return result


def write_report(path, payload):
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload.get("arrival_model", "reset_uniform") == "continuous":
        random_delay_note = (
            "- continuous 模式每个 mode/setting 只在 epoch 开头对齐一次；各 rank 随后独立连续 "
            f"replay {payload['continuous_replays']} 次。第一次使用共同起点上的随机 offset，后续每次在本 "
            "rank 上一 replay 完成后等待独立 U[0,max] scheduler gap；中间没有 barrier/all-gather，四个模式复用同一组 gap。"
        )
        sample_note = (
            f"- 每设置、每模式运行 1 个 continuous epoch × {payload['continuous_replays']} replay/rank，"
            f"warmup {payload['warmup']} 组；step trace={payload['trace_steps']}，layer trace={payload['trace_layers']}。"
        )
    else:
        random_delay_note = "- 每个 sample 为各 rank 独立生成 U[0,max] 延迟并减去最小值；同一 sample 的全部模式复用同一个 offset 向量，下一 sample 重新抽样。"
        sample_note = f"- 每设置采样 {payload['iterations']} 组，warmup {payload['warmup']} 组；step trace={payload['trace_steps']}，layer trace={payload['trace_layers']}。"
    lines = [
        "# SiDP DMA + 可控 GEMM 多 cycle benchmark",
        "",
        f"- 状态：{payload['status']}；GPU：{payload['device']} × {payload['num_gpus']}",
        f"- PyTorch/CUDA：{payload.get('torch', '-')}/{payload.get('cuda', '-')}；commit：{payload.get('git_commit', '-')}。",
        f"- cycle/cache：{payload['cycles']}/{CACHE_DEPTH}；K：{payload['k_values']}；每层 component bytes：{payload['component_bytes']}。",
        f"- GEMM：BF16 [{payload['gemm_m']},{payload['gemm_k']}] × [{payload['gemm_k']},{payload['gemm_n']}]；repeat sweep={payload['gemm_repeats_values']}。",
        sample_note,
        f"- 本次模式：{payload['modes']}。compute_dma 是固定顺序整层 DMA + Event；compute_dma_slice2/4 将每个 component 切为 2/4 片并按 slice→owner 排序；slice4_group2/4 先将连续 owner 分为 2/4 组，再逐组完成四片；名称以 _flag 结尾的固定模式将对应 Event RAW/WAR 改为 fill/comp generation flag；dynamic 两组使用动态 claim。",
        random_delay_note,
        "- graph 模拟 cycle0 resident、c1 首发、c+2 rolling refill、next-forward c0 refill和最终 join。",
        "- GEMM 使用独立 A/B，并不读取通信 payload；远端 layer 仍严格等待对应 slot 的 Event/flag，因此本基准测调度关键路径与资源重叠，不验证真实 FFN 数值。",
        "- CUDA graph_ms 不含人为 host 启动延迟；all-rank host wall 包含启动差和 host 观测开销。",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
        "| K | GEMM repeats/layer | 启动延迟场景 | 模式 | graph max-rank p50/p95(ms) | compute-path max-rank p50(ms) | tail join max-rank p50(ms) | all-rank host wall p50(ms) |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        for mode in payload["modes"]:
            summary = case["modes"][mode]
            graph = summary["max_rank_graph_ms"]
            lines.append(
                f"| {case['k']} | {case['gemm_repeats']} | {case['delay_label']} | {mode} | "
                f"{graph['p50']:.4f}/{graph['p95']:.4f} | "
                f"{summary['max_rank_compute_path_ms']['p50']:.4f} | "
                f"{summary['max_rank_tail_join_ms']['p50']:.4f} | "
                f"{summary['all_rank_observed_wall_ms']['p50']:.4f} |"
            )
    if payload.get("arrival_model", "reset_uniform") == "continuous":
        lines += [
            "",
            "## Continuous epoch 吞吐",
            "",
            "| K | GEMM repeats/layer | scheduler gap | 模式 | epoch wall(ms) | aggregate rank-replay/s | per-rank replay/s min/median/max | launch span initial/final/p95(us) |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
        for case in payload["cases"]:
            for mode in payload["modes"]:
                epoch = case["continuous_epochs"][mode]
                rate = epoch["rank_replays_per_second"]
                span = epoch["launch_span_us"]
                lines.append(
                    f"| {case['k']} | {case['gemm_repeats']} | {case['delay_label']} | {mode} | "
                    f"{epoch['epoch_wall_ms']:.4f} | {epoch['aggregate_rank_replays_per_second']:.4f} | "
                    f"{rate['min']:.4f}/{rate['median']:.4f}/{rate['max']:.4f} | "
                    f"{span['initial']:.2f}/{span['final']:.2f}/{span['p95']:.2f} |"
                )
        lines += ["", "### S4/S1 与 flag/event 吞吐差值", ""]
        for case in payload["cases"]:
            ratio_text = "；".join(
                f"{name}={(value - 1) * 100:+.2f}%"
                for name, value in case.get("throughput_ratios", {}).items()
            )
            lines.append(
                f"- K={case['k']}，repeats={case['gemm_repeats']}，delay={case['delay_label']}：{ratio_text}。"
            )
    lines += ["", "## 对照差值", ""]
    for case in payload["cases"]:
        ratios = case["ratios"]
        ratio_text = "；".join(f"{name}={value:.4f}" for name, value in ratios.items())
        compute_text = "；".join(
            f"{mode}={case['modes'][mode]['cycle_compute_ms_p50']}"
            for mode in payload["modes"]
        )
        compute_max_text = "；".join(
            f"{mode}={case['modes'][mode]['cycle_compute_max_rank_ms_p50']}"
            for mode in payload["modes"]
        )
        boundary_text = "；".join(
            f"{mode}={case['modes'][mode]['compute_boundary_max_rank_ms_p50']}"
            for mode in payload["modes"]
        )
        comm_text = "；".join(
            f"{mode}={case['modes'][mode].get('comm_cycle_ms_p50', [])}"
            for mode in payload["modes"]
            if mode != "compute_only"
        )
        lines += [
            f"- K={case['k']}，repeats={case['gemm_repeats']}，delay={case['delay_label']}：{ratio_text}。",
            f"  - compute cycle p50(ms)：{compute_text}。",
            f"  - compute cycle max-rank p50(ms)：{compute_max_text}。",
            f"  - compute boundary max-rank p50(ms)：{boundary_text}。",
            f"  - comm cycle p50(ms)：{comm_text}。",
        ]
    lines += [
        "",
        "## 观测边界",
        "",
        "`compute_path_ms` 从 graph 开始到最后一个 GEMM 完成，包含逐层 RAW 等待；`tail_join_ms` 只表示最后一个 GEMM 后等待 comm stream 的尾巴。",
        "每个 compute-cycle 的 elapsed 包含该 cycle 内 remote slot 的等待；每个 comm-cycle 的 elapsed 包含 WAR、claim/控制以及 DMA，而不是纯 memcpy 时间。",
        "S=2/4 的 step trace 从该 owner 第一片开始计到最后一片结束，中间包含其他 owner 的分片，因而是该 owner 的交错完成窗口，不是纯 memcpy duration。",
        "compute boundary 指标完全复用 graph/cycle Event，不新增节点；pre-c0、inter-cycle、post-cycle 与 cycle sum 应重构 compute_path，residual 应接近零。",
        "打开 `--trace-layers` 后，原始样本增加逐 layer RAW wait 和 GEMM 窗口；打开 `--trace-steps` 后增加逐 copy 的动态选择、claim 或 Event-WAR等待及 transfer 窗口。两者都会插入 Event，应与默认低扰动数据分开运行。",
        "不同 GPU 的 Event 时钟没有做公共时钟对齐；跨 rank 只比较各自 duration 与 host-observed makespan。",
        "这个基准没有 attention、RMSNorm、真实 FFN 两段 GEMM，也不让 GEMM读取拷贝的权重；它用于定位顺序/同步/重叠，不替代 serving profiling。",
        "原始逐次、逐rank数据在同名 `.samples.jsonl`。",
    ]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def timing_event(torch):
    return torch.cuda.Event(enable_timing=True, external=True)


def dump_cuda_graph(graph, path, flags):
    """Dump the retained cudaGraph_t without relying on torch debug mode."""
    import ctypes

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cudart = ctypes.CDLL("libcudart.so")
    debug_dot_print = cudart.cudaGraphDebugDotPrint
    debug_dot_print.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
    debug_dot_print.restype = ctypes.c_int
    raw_graph = graph.raw_cuda_graph()
    error = debug_dot_print(
        ctypes.c_void_p(int(raw_graph)),
        str(path).encode(),
        ctypes.c_uint(flags),
    )
    if error:
        get_error_string = cudart.cudaGetErrorString
        get_error_string.argtypes = [ctypes.c_int]
        get_error_string.restype = ctypes.c_char_p
        message = get_error_string(error)
        raise RuntimeError(
            "cudaGraphDebugDotPrint failed: "
            f"{message.decode() if message else f'CUDA error {error}'}"
        )


def event_handle(event):
    return f"0x{int(event.cuda_event):x}"


class PipelineGraph:
    def __init__(
        self,
        rank,
        k,
        gemm_repeats,
        args,
        copier,
        source_ptrs,
        destinations,
        owner_ptrs,
    ):
        import torch
        from sglang.kernels.ops.sidp import (
            claim_owner,
            claim_owner_traced,
            copy_selected,
            copy_selected_traced,
            load_sidp_sm_copy_module,
            publish_generation,
            publish_generation_traced,
            publish_selected_fill,
            publish_selected_fill_traced,
            record_trace,
            release_owner,
            release_owner_traced,
            reset_cycle_state,
            reset_cycle_state_traced,
            reset_forward_state,
            reset_forward_state_traced,
            reset_sm_trace,
            select_fixed,
            select_fixed_traced,
            wait_generation,
            wait_generation_traced,
        )
        from sglang.kernels.ops.sidp.dma_graph import load_sidp_dma_graph_module

        self.torch = torch
        self.rank = rank
        self.k = k
        self.args = args
        self.gemm_repeats = gemm_repeats
        self.positions = candidates(rank, args.num_gpus, k)
        self.position_to_index = {
            position: index for index, position in enumerate(self.positions)
        }
        self.count = len(self.positions)
        self.slot_count = CACHE_DEPTH * self.count
        self.destinations = destinations
        self.source_ptrs = source_ptrs
        self.copier = copier
        self.owner_ptrs = owner_ptrs
        self.owner_candidates = torch.tensor(
            self.positions, dtype=torch.int32, device=rank
        )
        self.candidate_slots = {
            cycle: torch.tensor(
                [slot_for(cycle, index, self.count) for index in range(self.count)],
                dtype=torch.int32,
                device=rank,
            )
            for cycle in range(args.cycles)
        }
        self.done = {
            cycle: torch.empty(self.count, dtype=torch.uint8, device=rank)
            for cycle in range(args.cycles)
        }
        self.fill = torch.zeros(self.slot_count, dtype=torch.int32, device=rank)
        self.comp = torch.zeros_like(self.fill)
        self.cursor = torch.zeros(1, dtype=torch.int32, device=rank)
        self.selected = torch.empty_like(self.cursor)
        self.error = torch.zeros_like(self.cursor)
        self.spins = torch.zeros(1, dtype=torch.int64, device=rank)
        self.collisions = torch.zeros_like(self.spins)
        self.copies = {
            cycle: torch.tensor(
                [
                    [
                        (
                            source_ptrs[owner][component],
                            destinations[component][
                                slot_for(cycle, index, self.count)
                            ].data_ptr(),
                            size,
                        )
                        for index, owner in enumerate(self.positions)
                    ]
                    for component, size in enumerate(args.component_bytes)
                ],
                dtype=torch.uint64,
                device="cpu",
            )
            for cycle in range(args.cycles)
        }
        self.sm_copy_descriptors = {
            cycle: [
                (
                    torch.tensor(
                        [source_ptrs[owner][component] for owner in self.positions],
                        dtype=torch.uint64,
                        device=rank,
                    ),
                    torch.tensor(
                        [
                            destinations[component][
                                slot_for(cycle, index, self.count)
                            ].data_ptr()
                            for index in range(self.count)
                        ],
                        dtype=torch.uint64,
                        device=rank,
                    ),
                    torch.full(
                        (self.count,),
                        size,
                        dtype=torch.int64,
                        device=rank,
                    ),
                )
                for component, size in enumerate(args.component_bytes)
            ]
            for cycle in range(args.cycles)
        }
        self.ready = [torch.cuda.Event() for _ in range(self.slot_count)]
        self.consumed = [torch.cuda.Event() for _ in range(self.slot_count)]
        self.compute_input = torch.randn(
            (args.gemm_m, args.gemm_k), dtype=torch.bfloat16, device=rank
        ) / math.sqrt(args.gemm_k)
        self.compute_weight = torch.randn(
            (args.gemm_k, args.gemm_n), dtype=torch.bfloat16, device=rank
        ) / math.sqrt(args.gemm_k)
        self.compute_output = torch.empty(
            (args.gemm_m, args.gemm_n), dtype=torch.bfloat16, device=rank
        )
        self.graphs = {}
        self.streams = {}
        self.graph_events = {}
        self.compute_cycle_events = {}
        self.comm_cycle_events = {}
        self.layer_events = {}
        self.step_events = {}
        self.selected_trace = {}
        self.spins_trace = {}
        self.collisions_trace = {}
        self.sm_trace_rows = {}
        self.sm_trace_count = {}
        self.sm_trace_overflow = {}
        self.sm_trace_metadata = {}
        self.dma_module = load_sidp_dma_graph_module()
        self.claim_owner = claim_owner
        self.claim_owner_traced = claim_owner_traced
        self.copy_selected = copy_selected
        self.copy_selected_traced = copy_selected_traced
        self.publish_selected_fill = publish_selected_fill
        self.publish_selected_fill_traced = publish_selected_fill_traced
        self.record_trace = record_trace
        self.release_owner = release_owner
        self.release_owner_traced = release_owner_traced
        self.reset_cycle_state = reset_cycle_state
        self.reset_cycle_state_traced = reset_cycle_state_traced
        self.reset_forward_state = reset_forward_state
        self.reset_forward_state_traced = reset_forward_state_traced
        self.reset_sm_trace = reset_sm_trace
        self.select_fixed = select_fixed
        self.select_fixed_traced = select_fixed_traced
        self.wait_generation = wait_generation
        self.wait_generation_traced = wait_generation_traced
        self.publish_generation = publish_generation
        self.publish_generation_traced = publish_generation_traced
        properties = torch.cuda.get_device_properties(rank)
        self.device_sm_count = int(properties.multi_processor_count)
        self.sm_copy_ctas = args.sm_copy_ctas or self.device_sm_count * 4
        self.timeout_clocks = int(properties.clock_rate * 1000 * args.device_timeout_s)
        load_sidp_sm_copy_module()

        # Warm up cuBLAS outside capture on an otherwise private stream.
        warmup_stream = torch.cuda.Stream()
        with torch.cuda.stream(warmup_stream):
            torch.mm(
                self.compute_input,
                self.compute_weight,
                out=self.compute_output,
            )
        warmup_stream.synchronize()

        # Setup establishes the same next-forward invariant used by serving.
        self._initialize_resident_slots()
        for event in self.consumed:
            event.record(torch.cuda.current_stream())
        torch.cuda.current_stream().synchronize()

        for mode in args.modes:
            self._capture(mode)

    def _nvtx(self, role, **fields):
        """Annotate graph construction without affecting normal benchmark runs.

        Nsight Systems 2026.4 can project these pre-capture ranges onto CUDA
        Graph nodes with ``--cuda-graph-trace=node::nvtx-precapture``.  Keep
        this opt-in because the benchmark's ordinary timing path should remain
        byte-for-byte free of profiler annotations.
        """
        if not self.args.nsight_annotations:
            return contextlib.nullcontext()
        suffix = "|".join(f"{key}={value}" for key, value in fields.items())
        message = f"sidp|role={role}"
        if suffix:
            message += f"|{suffix}"
        return self.torch.cuda.nvtx.range(message)

    def _prepare_sm_trace(self, mode):
        if not self.args.sm_execution_trace:
            return
        self.sm_trace_rows[mode] = self.torch.full(
            (self.args.sm_trace_capacity, SM_TRACE_FIELDS),
            -1,
            dtype=self.torch.int64,
            device=self.rank,
        )
        self.sm_trace_count[mode] = self.torch.zeros(
            1, dtype=self.torch.int32, device=self.rank
        )
        self.sm_trace_overflow[mode] = self.torch.zeros_like(
            self.sm_trace_count[mode]
        )
        self.sm_trace_metadata[mode] = []

    def _new_sm_trace_tag(self, mode, role, kernel, **fields):
        tag = len(self.sm_trace_metadata[mode])
        self.sm_trace_metadata[mode].append(
            {"tag": tag, "role": role, "kernel": kernel, **fields}
        )
        return tag

    def _sm_trace_args(self, mode, tag):
        return (
            self.sm_trace_rows[mode],
            self.sm_trace_count[mode],
            self.sm_trace_overflow[mode],
            tag,
        )

    def _initialize_resident_slots(self):
        stream = self.torch.cuda.current_stream()
        for index, owner in enumerate(self.positions):
            slot = slot_for(0, index, self.count)
            for component, size in enumerate(self.args.component_bytes):
                self.copier.async_copy(
                    self.destinations[component][slot].data_ptr(),
                    self.source_ptrs[owner][component],
                    size,
                    stream.cuda_stream,
                )
        stream.synchronize()

    def _new_step_events(self, stream):
        rows = [[timing_event(self.torch) for _ in range(3)] for _ in range(self.count)]
        for row in rows:
            for event in row:
                event.record(stream)
        stream.synchronize()
        return rows

    def _capture(self, mode):
        torch = self.torch
        self._prepare_sm_trace(mode)
        compute_stream = torch.cuda.Stream()
        comm_stream = torch.cuda.Stream() if mode != "compute_only" else None
        start, compute_end, end = (
            timing_event(torch),
            timing_event(torch),
            timing_event(torch),
        )
        compute_cycles = [
            (timing_event(torch), timing_event(torch)) for _ in range(self.args.cycles)
        ]
        comm_cycles = {}
        if mode != "compute_only":
            comm_cycles = {
                cycle: (timing_event(torch), timing_event(torch))
                for cycle in communication_sequence(self.args.cycles)
            }
        layer_events = (
            [
                [timing_event(torch) for _ in range(3)]
                for _ in range(self.args.cycles * self.args.num_gpus)
            ]
            if self.args.trace_layers
            else []
        )
        step_events = {}
        if mode != "compute_only" and self.args.trace_steps:
            step_events = {
                cycle: self._new_step_events(comm_stream)
                for cycle in communication_sequence(self.args.cycles)
            }

        trace_size = self.args.cycles * self.count
        if mode in DYNAMIC_MODES and self.args.trace_steps:
            self.selected_trace[mode] = torch.empty(
                trace_size, dtype=torch.int32, device=self.rank
            )
            self.spins_trace[mode] = torch.empty(
                trace_size, dtype=torch.int64, device=self.rank
            )
            self.collisions_trace[mode] = torch.empty_like(self.spins_trace[mode])

        graph = torch.cuda.CUDAGraph(keep_graph=True)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=compute_stream):
            start.record(compute_stream)
            if self.args.sm_execution_trace:
                self.reset_sm_trace(
                    self.sm_trace_count[mode], self.sm_trace_overflow[mode]
                )
            if mode in FLAG_MODES:
                with self._nvtx("control", op="reset_forward"):
                    if self.args.sm_execution_trace:
                        tag = self._new_sm_trace_tag(
                            mode,
                            "control",
                            "reset_forward_state_traced_kernel",
                            op="reset_forward",
                        )
                        self.reset_forward_state_traced(
                            self.fill,
                            self.comp,
                            self.count,
                            self.error,
                            *self._sm_trace_args(mode, tag),
                        )
                    else:
                        self.reset_forward_state(
                            self.fill,
                            self.comp,
                            self.count,
                            self.error,
                        )
            if mode != "compute_only":
                comm_stream.wait_stream(compute_stream)
                self._enqueue_communication(
                    mode,
                    1,
                    comm_stream,
                    comm_cycles[1],
                    step_events.get(1),
                )

            for cycle in range(self.args.cycles):
                cycle_start, cycle_end = compute_cycles[cycle]
                cycle_start.record(compute_stream)
                with self._nvtx("compute_cycle", cycle=cycle):
                    for position in range(self.args.num_gpus):
                        layer = cycle * self.args.num_gpus + position
                        row = layer_events[layer] if layer_events else None
                        if row:
                            row[0].record(compute_stream)
                        remote = position in self.position_to_index
                        if mode != "compute_only" and remote and cycle > 0:
                            candidate_index = self.position_to_index[position]
                            slot = slot_for(cycle, candidate_index, self.count)
                            with self._nvtx(
                                "raw_wait", cycle=cycle, layer=layer, slot=slot
                            ):
                                if mode in FLAG_MODES:
                                    if self.args.sm_execution_trace:
                                        tag = self._new_sm_trace_tag(
                                            mode,
                                            "raw_wait",
                                            "wait_generation_traced_kernel",
                                            cycle=cycle,
                                            layer=layer,
                                            slot=slot,
                                        )
                                        self.wait_generation_traced(
                                            self.fill,
                                            slot,
                                            fill_generation(cycle),
                                            self.args.backoff_ns,
                                            self.timeout_clocks,
                                            self.error,
                                            *self._sm_trace_args(mode, tag),
                                        )
                                    else:
                                        self.wait_generation(
                                            self.fill,
                                            slot,
                                            fill_generation(cycle),
                                            self.args.backoff_ns,
                                            self.timeout_clocks,
                                            self.error,
                                        )
                                else:
                                    compute_stream.wait_event(self.ready[slot])
                        if row:
                            row[1].record(compute_stream)
                        with self._nvtx(
                            "compute", cycle=cycle, layer=layer, op="gemm"
                        ):
                            for _ in range(self.gemm_repeats):
                                torch.mm(
                                    self.compute_input,
                                    self.compute_weight,
                                    out=self.compute_output,
                                )
                        if row:
                            row[2].record(compute_stream)

                        if mode != "compute_only" and remote:
                            candidate_index = self.position_to_index[position]
                            slot = slot_for(cycle, candidate_index, self.count)
                            with self._nvtx(
                                "control",
                                cycle=cycle,
                                layer=layer,
                                slot=slot,
                                op="publish_computed",
                            ):
                                if mode in FLAG_MODES:
                                    if self.args.sm_execution_trace:
                                        tag = self._new_sm_trace_tag(
                                            mode,
                                            "control",
                                            "publish_generation_traced_kernel",
                                            op="publish_computed",
                                            cycle=cycle,
                                            layer=layer,
                                            slot=slot,
                                        )
                                        self.publish_generation_traced(
                                            self.comp,
                                            slot,
                                            fill_generation(cycle),
                                            *self._sm_trace_args(mode, tag),
                                        )
                                    else:
                                        self.publish_generation(
                                            self.comp, slot, fill_generation(cycle)
                                        )
                                else:
                                    self.consumed[slot].record(compute_stream)

                            if position == self.positions[-1]:
                                next_cycle = cycle + CACHE_DEPTH
                                if next_cycle < self.args.cycles:
                                    self._enqueue_communication(
                                        mode,
                                        next_cycle,
                                        comm_stream,
                                        comm_cycles[next_cycle],
                                        step_events.get(next_cycle),
                                    )
                                elif cycle == self.args.cycles - CACHE_DEPTH:
                                    self._enqueue_communication(
                                        mode,
                                        0,
                                        comm_stream,
                                        comm_cycles[0],
                                        step_events.get(0),
                                    )
                cycle_end.record(compute_stream)

            compute_end.record(compute_stream)
            if mode != "compute_only":
                with self._nvtx("tail_join"):
                    compute_stream.wait_stream(comm_stream)
            end.record(compute_stream)
        graph.instantiate()
        self._dump_graph(
            mode, graph, start, compute_end, end, compute_cycles, comm_cycles
        )
        self.graphs[mode] = graph
        self.streams[mode] = (compute_stream, comm_stream)
        self.graph_events[mode] = (start, compute_end, end)
        self.compute_cycle_events[mode] = compute_cycles
        self.comm_cycle_events[mode] = comm_cycles
        self.layer_events[mode] = layer_events
        self.step_events[mode] = step_events

    def _dump_graph(
        self, mode, graph, start, compute_end, end, compute_cycles, comm_cycles
    ):
        if (
            self.rank != 0
            or self.args.dump_graph_dir is None
            or mode not in self.args.dump_graph_modes
        ):
            return
        stem = Path(self.args.dump_graph_dir) / (
            f"rank{self.rank}_k{self.k}_repeat{self.gemm_repeats}_{mode}"
        )
        slim_path = Path(f"{stem}.dot")
        verbose_path = Path(f"{stem}.verbose.dot")
        # CUDA 13: cudaGraphDebugDotFlagsVerbose == 1 << 0.
        dump_cuda_graph(graph, slim_path, 0)
        dump_cuda_graph(graph, verbose_path, 1)
        event_map = {
            "graph_start": event_handle(start),
            "compute_end": event_handle(compute_end),
            "graph_end": event_handle(end),
            "compute_cycles": [
                {
                    "cycle": cycle,
                    "start": event_handle(begin),
                    "end": event_handle(finish),
                }
                for cycle, (begin, finish) in enumerate(compute_cycles)
            ],
            "comm_cycles": {
                operation_name(cycle): {
                    "cycle": cycle,
                    "start": event_handle(begin),
                    "end": event_handle(finish),
                }
                for cycle, (begin, finish) in comm_cycles.items()
            },
            "ready_slots": [event_handle(event) for event in self.ready],
            "consumed_slots": [event_handle(event) for event in self.consumed],
        }
        map_path = Path(f"{stem}.events.json")
        map_path.write_text(
            json.dumps(event_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"rank 0 dumped {mode} CUDA Graph: {slim_path}, {verbose_path}, "
            f"{map_path}",
            flush=True,
        )

    def _enqueue_communication(
        self, mode, cycle, comm_stream, comm_cycle_events, step_events
    ):
        # The conditional-DMA builder obtains the current CUDA stream from
        # TVM-FFI.  Merely passing ``comm_stream`` to Event/memcpy calls is not
        # sufficient: make it current while appending the whole cycle.
        with self.torch.cuda.stream(comm_stream), self._nvtx(
            "comm_cycle", cycle=cycle, operation=operation_name(cycle)
        ):
            self._enqueue_communication_on_current_stream(
                mode, cycle, comm_stream, comm_cycle_events, step_events
            )

    def _enqueue_communication_on_current_stream(
        self, mode, cycle, comm_stream, comm_cycle_events, step_events
    ):
        start, end = comm_cycle_events
        start.record(comm_stream)
        required, target = (
            next_forward_generations(self.args.cycles)
            if cycle == 0
            else (reuse_requirement(cycle), fill_generation(cycle))
        )
        if mode in FIXED_MODES:
            slice_factor = SLICE_MODE_FACTORS.get(mode, 1)
            group_count = GROUP_MODE_COUNTS.get(mode, 1)
            # Each contiguous owner group is completed before the next group:
            # group0(slice0 owners..., slice1 owners...), group1(...), ... .
            # G=1 is the original cycle-wide slice-major ordering.
            for group_begin, group_end in owner_group_ranges(
                self.count, group_count
            ):
                for slice_index in range(slice_factor):
                    for index in range(group_begin, group_end):
                        owner = self.positions[index]
                        slot = slot_for(cycle, index, self.count)
                        row = step_events[index] if step_events else None
                        if slice_index == 0:
                            if row:
                                row[0].record(comm_stream)
                            if required:
                                with self._nvtx(
                                    "war_wait", cycle=cycle, owner=owner, slot=slot
                                ):
                                    if mode in FIXED_FLAG_MODES:
                                        if self.args.sm_execution_trace:
                                            tag = self._new_sm_trace_tag(
                                                mode,
                                                "war_wait",
                                                "wait_generation_traced_kernel",
                                                cycle=cycle,
                                                owner=owner,
                                                slot=slot,
                                            )
                                            self.wait_generation_traced(
                                                self.comp,
                                                slot,
                                                required,
                                                self.args.backoff_ns,
                                                self.timeout_clocks,
                                                self.error,
                                                *self._sm_trace_args(mode, tag),
                                            )
                                        else:
                                            self.wait_generation(
                                                self.comp,
                                                slot,
                                                required,
                                                self.args.backoff_ns,
                                                self.timeout_clocks,
                                                self.error,
                                            )
                                    else:
                                        comm_stream.wait_event(self.consumed[slot])
                            if row:
                                row[1].record(comm_stream)
                        for component, size in enumerate(self.args.component_bytes):
                            begin, finish = slice_bounds(
                                size, slice_factor, slice_index
                            )
                            if finish == begin:
                                continue
                            with self._nvtx(
                                "communication",
                                cycle=cycle,
                                owner=owner,
                                slot=slot,
                                component=component,
                                slice=slice_index,
                                op="dma_copy",
                            ):
                                self.copier.async_copy(
                                    self.destinations[component][slot].data_ptr()
                                    + begin,
                                    self.source_ptrs[owner][component] + begin,
                                    finish - begin,
                                    comm_stream.cuda_stream,
                                )
                        if slice_index + 1 == slice_factor:
                            with self._nvtx(
                                "control",
                                cycle=cycle,
                                owner=owner,
                                slot=slot,
                                op="publish_filled",
                            ):
                                if mode in FIXED_FLAG_MODES:
                                    if self.args.sm_execution_trace:
                                        tag = self._new_sm_trace_tag(
                                            mode,
                                            "control",
                                            "publish_generation_traced_kernel",
                                            op="publish_filled",
                                            cycle=cycle,
                                            owner=owner,
                                            slot=slot,
                                        )
                                        self.publish_generation_traced(
                                            self.fill,
                                            slot,
                                            target,
                                            *self._sm_trace_args(mode, tag),
                                        )
                                    else:
                                        self.publish_generation(
                                            self.fill, slot, target
                                        )
                                else:
                                    self.ready[slot].record(comm_stream)
                            if row:
                                row[2].record(comm_stream)
        elif mode in FIXED_SM_MODES:
            for index, owner in enumerate(self.positions):
                slot = slot_for(cycle, index, self.count)
                row = step_events[index] if step_events else None
                if row:
                    row[0].record(comm_stream)
                if required:
                    with self._nvtx(
                        "war_wait", cycle=cycle, owner=owner, slot=slot
                    ):
                        if self.args.sm_execution_trace:
                            tag = self._new_sm_trace_tag(
                                mode,
                                "war_wait",
                                "wait_generation_traced_kernel",
                                cycle=cycle,
                                owner=owner,
                                slot=slot,
                            )
                            self.wait_generation_traced(
                                self.comp,
                                slot,
                                required,
                                self.args.backoff_ns,
                                self.timeout_clocks,
                                self.error,
                                *self._sm_trace_args(mode, tag),
                            )
                        else:
                            self.wait_generation(
                                self.comp,
                                slot,
                                required,
                                self.args.backoff_ns,
                                self.timeout_clocks,
                                self.error,
                            )
                if row:
                    row[1].record(comm_stream)
                if self.args.sm_execution_trace:
                    tag = self._new_sm_trace_tag(
                        mode,
                        "control",
                        "select_fixed_traced_kernel",
                        cycle=cycle,
                        owner=owner,
                        slot=slot,
                    )
                    self.select_fixed_traced(
                        self.selected, index, *self._sm_trace_args(mode, tag)
                    )
                else:
                    self.select_fixed(self.selected, index)
                for component, descriptors in enumerate(
                    self.sm_copy_descriptors[cycle]
                ):
                    with self._nvtx(
                        "communication",
                        cycle=cycle,
                        owner=owner,
                        slot=slot,
                        component=component,
                        op="sm_copy",
                    ):
                        if self.args.sm_execution_trace:
                            tag = self._new_sm_trace_tag(
                                mode,
                                "communication_sm",
                                "copy_selected_traced_kernel",
                                cycle=cycle,
                                owner=owner,
                                slot=slot,
                                component=component,
                            )
                            self.copy_selected_traced(
                                *descriptors,
                                self.selected,
                                self.sm_copy_ctas,
                                self.args.sm_copy_block,
                                self.error,
                                *self._sm_trace_args(mode, tag),
                            )
                        else:
                            self.copy_selected(
                                *descriptors,
                                self.selected,
                                self.sm_copy_ctas,
                                self.args.sm_copy_block,
                                self.error,
                            )
                if self.args.sm_execution_trace:
                    tag = self._new_sm_trace_tag(
                        mode,
                        "control",
                        "publish_selected_fill_traced_kernel",
                        op="publish_filled",
                        cycle=cycle,
                        owner=owner,
                        slot=slot,
                    )
                    self.publish_selected_fill_traced(
                        self.fill,
                        self.candidate_slots[cycle],
                        self.selected,
                        target,
                        self.error,
                        *self._sm_trace_args(mode, tag),
                    )
                else:
                    self.publish_selected_fill(
                        self.fill,
                        self.candidate_slots[cycle],
                        self.selected,
                        target,
                        self.error,
                    )
                if row:
                    row[2].record(comm_stream)
        elif mode in DYNAMIC_DMA_MODES:
            trace_offset = (
                communication_sequence(self.args.cycles).index(cycle) * self.count
            )
            if step_events:
                handles = self.torch.tensor(
                    [[event.cuda_event for event in row] for row in step_events],
                    dtype=self.torch.uint64,
                )
                profile_args = (
                    self.selected_trace[mode],
                    self.spins_trace[mode],
                    self.collisions_trace[mode],
                    handles,
                    trace_offset,
                )
            else:
                profile_args = (
                    self.selected,
                    self.spins,
                    self.collisions,
                    self.copies[cycle],
                    -1,
                )
            self.dma_module.append_cycle_to_capture(
                self.owner_ptrs,
                self.owner_candidates,
                self.candidate_slots[cycle],
                self.done[cycle],
                self.fill,
                self.comp,
                self.cursor,
                self.selected,
                self.spins,
                self.collisions,
                self.error,
                self.copies[cycle],
                required,
                target,
                CLAIM_ORDER[mode],
                self.rank,
                self.args.backoff_ns,
                self.timeout_clocks,
                *profile_args,
                *self._dynamic_dma_sm_trace_args(mode, cycle),
            )
        elif mode in DYNAMIC_SM_MODES:
            self._enqueue_dynamic_sm(
                mode,
                cycle,
                required,
                target,
                comm_stream,
                step_events,
            )
        else:
            raise AssertionError(mode)
        end.record(comm_stream)

    def _dynamic_dma_sm_trace_args(self, mode, cycle):
        if not self.args.sm_execution_trace:
            return (self.copies[cycle], self.selected, self.selected, -1)
        base = self._new_sm_trace_tag(
            mode,
            "control",
            "reset_cycle_state_traced_kernel",
            cycle=cycle,
            op="reset_cycle",
        )
        for step in range(self.count):
            expected = base + 1 + step * 4
            tag = self._new_sm_trace_tag(
                mode,
                "wait_or_claim",
                "claim_owner_traced_kernel",
                cycle=cycle,
                step=step,
            )
            if tag != expected:
                raise AssertionError("dynamic DMA SM trace tags lost contiguity")
            self._new_sm_trace_tag(
                mode,
                "control",
                "set_dma_condition_traced_kernel",
                cycle=cycle,
                step=step,
            )
            self._new_sm_trace_tag(
                mode,
                "control",
                "release_owner_traced_kernel",
                cycle=cycle,
                step=step,
            )
            self._new_sm_trace_tag(
                mode,
                "control",
                "publish_selected_fill_traced_kernel",
                cycle=cycle,
                step=step,
            )
        return (
            self.sm_trace_rows[mode],
            self.sm_trace_count[mode],
            self.sm_trace_overflow[mode],
            base,
        )

    def _enqueue_dynamic_sm(
        self, mode, cycle, required, target, comm_stream, step_events
    ):
        trace_offset = (
            communication_sequence(self.args.cycles).index(cycle) * self.count
        )
        if self.args.sm_execution_trace:
            tag = self._new_sm_trace_tag(
                mode,
                "control",
                "reset_cycle_state_traced_kernel",
                cycle=cycle,
                op="reset_cycle",
            )
            self.reset_cycle_state_traced(
                self.done[cycle],
                self.selected,
                self.spins,
                self.collisions,
                *self._sm_trace_args(mode, tag),
            )
        else:
            self.reset_cycle_state(
                self.done[cycle], self.selected, self.spins, self.collisions
            )
        for step in range(self.count):
            row = step_events[step] if step_events else None
            if row:
                row[0].record(comm_stream)
            if self.args.sm_execution_trace:
                tag = self._new_sm_trace_tag(
                    mode,
                    "wait_or_claim",
                    "claim_owner_traced_kernel",
                    cycle=cycle,
                    step=step,
                )
                self.claim_owner_traced(
                    self.owner_ptrs,
                    self.owner_candidates,
                    self.candidate_slots[cycle],
                    self.done[cycle],
                    self.comp,
                    required,
                    CLAIM_ORDER[mode],
                    self.cursor,
                    self.selected,
                    self.spins,
                    self.collisions,
                    self.rank,
                    self.args.backoff_ns,
                    self.timeout_clocks,
                    self.error,
                    *self._sm_trace_args(mode, tag),
                )
            else:
                self.claim_owner(
                    self.owner_ptrs,
                    self.owner_candidates,
                    self.candidate_slots[cycle],
                    self.done[cycle],
                    self.comp,
                    required,
                    CLAIM_ORDER[mode],
                    self.cursor,
                    self.selected,
                    self.spins,
                    self.collisions,
                    self.rank,
                    self.args.backoff_ns,
                    self.timeout_clocks,
                    self.error,
                )
            if row:
                row[1].record(comm_stream)
            if step_events:
                self.record_trace(
                    self.selected,
                    self.spins,
                    self.collisions,
                    self.selected_trace[mode],
                    self.spins_trace[mode],
                    self.collisions_trace[mode],
                    trace_offset + step,
                )
            for component, descriptors in enumerate(
                self.sm_copy_descriptors[cycle]
            ):
                with self._nvtx(
                    "communication",
                    cycle=cycle,
                    step=step,
                    component=component,
                    op="sm_copy",
                ):
                    if self.args.sm_execution_trace:
                        tag = self._new_sm_trace_tag(
                            mode,
                            "communication_sm",
                            "copy_selected_traced_kernel",
                            cycle=cycle,
                            step=step,
                            component=component,
                        )
                        self.copy_selected_traced(
                            *descriptors,
                            self.selected,
                            self.sm_copy_ctas,
                            self.args.sm_copy_block,
                            self.error,
                            *self._sm_trace_args(mode, tag),
                        )
                    else:
                        self.copy_selected(
                            *descriptors,
                            self.selected,
                            self.sm_copy_ctas,
                            self.args.sm_copy_block,
                            self.error,
                        )
            if self.args.sm_execution_trace:
                tag = self._new_sm_trace_tag(
                    mode,
                    "control",
                    "release_owner_traced_kernel",
                    cycle=cycle,
                    step=step,
                )
                self.release_owner_traced(
                    self.owner_ptrs,
                    self.owner_candidates,
                    self.selected,
                    self.rank,
                    self.error,
                    *self._sm_trace_args(mode, tag),
                )
                tag = self._new_sm_trace_tag(
                    mode,
                    "control",
                    "publish_selected_fill_traced_kernel",
                    cycle=cycle,
                    step=step,
                )
                self.publish_selected_fill_traced(
                    self.fill,
                    self.candidate_slots[cycle],
                    self.selected,
                    target,
                    self.error,
                    *self._sm_trace_args(mode, tag),
                )
            else:
                self.release_owner(
                    self.owner_ptrs,
                    self.owner_candidates,
                    self.selected,
                    self.rank,
                    self.error,
                )
                self.publish_selected_fill(
                    self.fill,
                    self.candidate_slots[cycle],
                    self.selected,
                    target,
                    self.error,
                )
            if row:
                row[2].record(comm_stream)

    def validate(self, mode):
        torch = self.torch
        if not torch.isfinite(self.compute_output).all().item():
            raise RuntimeError(f"rank {self.rank}: non-finite GEMM output")
        if mode == "compute_only":
            return
        if self.error.item() != 0:
            raise RuntimeError(f"rank {self.rank}: device error {self.error.item()}")
        if mode in FLAG_MODES:
            required, target = next_forward_generations(self.args.cycles)
            expected_fill = [
                target if slot < self.count else required
                for slot in range(self.slot_count)
            ]
            expected_comp = [required] * self.slot_count
            if self.fill.tolist() != expected_fill:
                raise RuntimeError(
                    f"rank {self.rank}: fill generations {self.fill.tolist()} != {expected_fill}"
                )
            if self.comp.tolist() != expected_comp:
                raise RuntimeError(
                    f"rank {self.rank}: comp generations {self.comp.tolist()} != {expected_comp}"
                )
        for component, buffers in enumerate(self.destinations):
            for slot in range(self.slot_count):
                candidate_index = slot % self.count
                owner = self.positions[candidate_index]
                expected = (
                    1 + (owner * len(self.args.component_bytes) + component) % 251
                )
                # Full payload validation is deliberately outside measurement.
                if not torch.all(buffers[slot] == expected).item():
                    raise RuntimeError(
                        "copy mismatch: "
                        f"rank={self.rank}, owner={owner}, component={component}, slot={slot}"
                    )
        if mode in DYNAMIC_MODES and self.args.trace_steps:
            selected = self.selected_trace[mode].tolist()
            for op_index in range(self.args.cycles):
                begin = op_index * self.count
                values = selected[begin : begin + self.count]
                if sorted(values) != list(range(self.count)):
                    raise RuntimeError(
                        f"rank {self.rank}: invalid dynamic coverage at op {op_index}: {values}"
                    )

    def run(
        self,
        mode,
        target_ns,
        requested_offset_us,
        profile_sample=None,
        on_profile_graph_complete=None,
    ):
        deadline = target_ns + round(requested_offset_us * 1000)
        remaining = (deadline - time.monotonic_ns()) / 1e9
        if remaining > 0:
            time.sleep(remaining)
        launched = time.monotonic_ns()
        compute_stream, _ = self.streams[mode]
        message = (
            "sidp|role=graph_replay|"
            f"mode={mode}|rank={self.rank}|k={self.k}|"
            f"gemm_repeats={self.gemm_repeats}|sample={profile_sample}"
        )
        replay_range = (
            self.torch.cuda.nvtx.range(message)
            if profile_sample is not None
            else contextlib.nullcontext()
        )
        with replay_range:
            with self.torch.cuda.stream(compute_stream):
                self.graphs[mode].replay()
            submitted = time.monotonic_ns()
            start, compute_end, end = self.graph_events[mode]
            end.synchronize()
            completed = time.monotonic_ns()
        if on_profile_graph_complete is not None:
            on_profile_graph_complete()
        compute_boundaries = compute_boundary_metrics(
            start,
            compute_end,
            self.compute_cycle_events[mode],
            self.comm_cycle_events[mode],
        )
        result = {
            "rank": self.rank,
            "requested_offset_us": requested_offset_us,
            "launch_begin_ns": launched,
            "launch_submitted_ns": submitted,
            "completion_observed_ns": completed,
            "launch_lateness_us": (launched - deadline) / 1000,
            "graph_ms": float(start.elapsed_time(end)),
            "compute_path_ms": float(start.elapsed_time(compute_end)),
            "tail_join_ms": float(compute_end.elapsed_time(end)),
            "compute_boundaries": compute_boundaries,
            "compute_cycles": [
                {
                    "cycle": cycle,
                    "elapsed_ms": float(begin.elapsed_time(finish)),
                }
                for cycle, (begin, finish) in enumerate(self.compute_cycle_events[mode])
            ],
            "comm_cycles": [],
        }
        if self.args.sm_execution_trace:
            count = int(self.sm_trace_count[mode].item())
            overflow = int(self.sm_trace_overflow[mode].item())
            rows = self.sm_trace_rows[mode][
                : min(count, self.args.sm_trace_capacity)
            ].cpu().tolist()
            result["sm_execution_trace"] = summarize_sm_execution_trace(
                rows,
                count,
                overflow,
                self.sm_trace_metadata[mode],
                self.device_sm_count,
                self.args.sm_trace_capacity,
            )
        for cycle in communication_sequence(self.args.cycles):
            if mode == "compute_only":
                break
            begin, finish = self.comm_cycle_events[mode][cycle]
            result["comm_cycles"].append(
                {
                    "operation": operation_name(cycle),
                    "cycle": cycle,
                    "elapsed_ms": float(begin.elapsed_time(finish)),
                }
            )
        if self.args.trace_layers:
            result["layers"] = []
            for layer, row in enumerate(self.layer_events[mode]):
                position = layer % self.args.num_gpus
                result["layers"].append(
                    {
                        "layer": layer,
                        "cycle": layer // self.args.num_gpus,
                        "position": position,
                        "remote": position in self.position_to_index,
                        "raw_wait_ms": float(row[0].elapsed_time(row[1])),
                        "gemm_window_ms": float(row[1].elapsed_time(row[2])),
                    }
                )
        if self.args.trace_steps and mode != "compute_only":
            if mode in DYNAMIC_MODES:
                selections = self.selected_trace[mode].tolist()
                spins = self.spins_trace[mode].tolist()
                collisions = self.collisions_trace[mode].tolist()
            else:
                selections = list(range(self.count)) * self.args.cycles
                spins = collisions = [0] * (self.count * self.args.cycles)
            result["comm_steps"] = []
            for op_index, cycle in enumerate(communication_sequence(self.args.cycles)):
                pending = set(range(self.count))
                cycle_start = self.comm_cycle_events[mode][cycle][0]
                for step, row in enumerate(self.step_events[mode][cycle]):
                    trace_index = op_index * self.count + step
                    selected = selections[trace_index]
                    spin_value = spins[trace_index]
                    collision_value = collisions[trace_index]
                    front = min(pending)
                    priority_distance = selected - front
                    pending.remove(selected)
                    result["comm_steps"].append(
                        {
                            "operation": operation_name(cycle),
                            "step": step,
                            "selected_index": selected,
                            "owner": self.positions[selected],
                            "slot": slot_for(cycle, selected, self.count),
                            "claim_or_war_wait_ms": float(row[0].elapsed_time(row[1])),
                            "transfer_window_ms": float(row[1].elapsed_time(row[2])),
                            "start_from_cycle_ms": float(
                                cycle_start.elapsed_time(row[0])
                            ),
                            "end_from_cycle_ms": float(
                                cycle_start.elapsed_time(row[2])
                            ),
                            "priority_front_index": front,
                            "priority_distance": priority_distance,
                            "claim_spins": spin_value,
                            "claim_collisions": collision_value,
                            "slice_factor": SLICE_MODE_FACTORS.get(mode, 1),
                            "group_count": GROUP_MODE_COUNTS.get(mode, 1),
                        }
                    )
        return result


def worker(rank, args, port):
    import torch
    import torch.distributed as dist
    from sglang.kernels.ops.sidp import (
        load_sidp_sm_copy_module,
        native_peer_atomic_supported,
    )
    from sglang.srt.layers.sidp.cuda_memcpy import SidpCudaMemcpy

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.manual_seed(args.seed + rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=args.num_gpus,
        timeout=timedelta(seconds=300),
    )
    copier = SidpCudaMemcpy()
    sources = [
        torch.full(
            (size,),
            1 + (rank * len(args.component_bytes) + component) % 251,
            dtype=torch.uint8,
            device=rank,
        )
        for component, size in enumerate(args.component_bytes)
    ]
    owner = torch.full((1,), -1, dtype=torch.int32, device=rank)
    max_remote = args.num_gpus - min(args.k_values)
    destinations = [
        [torch.empty_like(source) for _ in range(CACHE_DEPTH * max_remote)]
        for source in sources
    ]
    torch.cuda.synchronize()
    descriptor = {
        "components": [
            copier.export_ipc_pointer(source.data_ptr(), source.nbytes)
            for source in sources
        ],
        "owner": copier.export_ipc_pointer(owner.data_ptr(), owner.nbytes),
    }
    descriptors = [None] * args.num_gpus
    dist.all_gather_object(descriptors, descriptor)
    mappings = {}

    def pointer(desc):
        if desc["handle"] not in mappings:
            mappings[desc["handle"]] = copier.open_ipc_allocation(desc["handle"])
        return mappings[desc["handle"]] + int(desc["offset"])

    source_ptrs, owner_ptr_values, atomic_support = [], [], []
    for peer in range(args.num_gpus):
        supported = peer == rank or native_peer_atomic_supported(rank, peer)
        atomic_support.append(supported)
        if not supported:
            raise RuntimeError(f"native peer atomic unavailable: {rank} -> {peer}")
        if peer == rank:
            source_ptrs.append([source.data_ptr() for source in sources])
            owner_ptr_values.append(owner.data_ptr())
        else:
            copier.enable_peer_access(peer)
            source_ptrs.append(
                [pointer(component) for component in descriptors[peer]["components"]]
            )
            owner_ptr_values.append(pointer(descriptors[peer]["owner"]))
    owner_ptrs = torch.tensor(owner_ptr_values, dtype=torch.uint64, device=rank)
    load_sidp_sm_copy_module()
    support_by_rank = [None] * args.num_gpus
    dist.all_gather_object(support_by_rank, atomic_support)

    payload = {
        **vars(args),
        "status": "running",
        "timestamp": datetime.now().astimezone().isoformat(),
        "device": torch.cuda.get_device_name(rank),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "native_peer_atomics": support_by_rank,
        "cases": [],
    }
    raw_file = None
    if rank == 0:
        raw_file = (
            Path(args.output).with_suffix(".samples.jsonl").open("x", encoding="utf-8")
        )

    def launch_round(graph, mode, offsets, profile_sample=None):
        deadline = [
            time.monotonic_ns() + round(args.lead_ms * 1e6) if rank == 0 else None
        ]
        dist.broadcast_object_list(deadline, src=0)
        if profile_sample is not None:
            # One process controls the global Nsight Systems capture.  Every
            # rank is already past graph construction/validation and reaches
            # this barrier before rank 0 opens the range.
            dist.barrier()
            if rank == 0:
                set_cuda_profiler(torch, True)
            dist.barrier()
        def finish_profile_range():
            dist.barrier()
            if rank == 0:
                set_cuda_profiler(torch, False)
            dist.barrier()

        local = graph.run(
            mode,
            deadline[0],
            offsets[rank],
            profile_sample,
            finish_profile_range if profile_sample is not None else None,
        )
        results = [None] * args.num_gpus
        dist.all_gather_object(results, local)
        return results

    def launch_continuous_epoch(graph, mode, gap_plan):
        """Run locally without inter-replay collectives, then gather once."""
        # Accumulating the benchmark's nested per-replay result dictionaries can
        # otherwise trigger a 200-400 ms Python cyclic-GC pause around replay
        # 70. Collect while ranks are aligned and keep this instrumentation-only
        # source of jitter out of the measured epoch.
        dist.barrier()
        gc.collect()
        dist.barrier()
        deadline = [
            time.monotonic_ns() + round(args.lead_ms * 1e6)
            if rank == 0
            else None
        ]
        dist.broadcast_object_list(deadline, src=0)
        local_replays = []
        target_ns = deadline[0]
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for replay in range(args.continuous_replays):
                local = graph.run(mode, target_ns, gap_plan[rank][replay])
                local["continuous_replay"] = replay
                local_replays.append(local)
                # Model completion and the per-rank scheduler gap both feed the
                # next launch. There is intentionally no cross-rank rendezvous.
                target_ns = local["completion_observed_ns"]
            all_rank_replays = [None] * args.num_gpus
            dist.all_gather_object(all_rank_replays, local_replays)
        finally:
            if gc_was_enabled:
                gc.enable()
        return all_rank_replays

    for k in args.k_values:
        for repeats in args.gemm_repeats_values:
            graph = PipelineGraph(
                rank,
                k,
                repeats,
                args,
                copier,
                source_ptrs,
                destinations,
                owner_ptrs,
            )
            torch.cuda.synchronize()
            dist.barrier()
            for scenario in args.delay_scenarios:
                validation_offsets = planned_offsets(
                    scenario,
                    args.seed,
                    -1_000_000,
                    args.num_gpus,
                )
                for mode in args.modes:
                    dist.barrier()
                    launch_round(graph, mode, validation_offsets)
                    graph.validate(mode)
                    dist.barrier()
                for warmup in range(args.warmup):
                    offsets = planned_offsets(
                        scenario,
                        args.seed,
                        -warmup - 1,
                        args.num_gpus,
                    )
                    for mode in args.modes:
                        launch_round(graph, mode, offsets)
                records = {mode: [] for mode in args.modes}
                continuous_epochs = {}
                if args.arrival_model == "continuous":
                    gap_plan = planned_continuous_gaps(
                        scenario,
                        args.seed,
                        args.continuous_replays,
                        args.num_gpus,
                    )
                    for mode in args.modes:
                        all_rank_replays = launch_continuous_epoch(
                            graph, mode, gap_plan
                        )
                        for replay in range(args.continuous_replays):
                            ranks = [
                                all_rank_replays[peer][replay]
                                for peer in range(args.num_gpus)
                            ]
                            record = {
                                "k": k,
                                "gemm_repeats": repeats,
                                "mode": mode,
                                "sample": replay,
                                "arrival_model": args.arrival_model,
                                "delay_scenario": scenario,
                                "delay_label": delay_label(
                                    scenario, args.arrival_model
                                ),
                                "base_offsets_us": scenario.get("offsets_us"),
                                "requested_offsets_us": [
                                    gap_plan[peer][replay]
                                    for peer in range(args.num_gpus)
                                ],
                                "ranks": ranks,
                                "metrics": trial_metrics(ranks),
                            }
                            records[mode].append(record)
                            if raw_file is not None:
                                raw_file.write(json.dumps(record) + "\n")
                        if raw_file is not None:
                            raw_file.flush()
                        continuous_epochs[mode] = continuous_epoch_metrics(
                            all_rank_replays
                        )
                        graph.validate(mode)
                        dist.barrier()
                else:
                    for sample in range(args.iterations):
                        offsets = planned_offsets(
                            scenario,
                            args.seed,
                            sample,
                            args.num_gpus,
                        )
                        order = args.modes if sample % 2 == 0 else args.modes[::-1]
                        for mode in order:
                            profile_sample = (
                                sample
                                if sample == args.nsight_profile_sample
                                else None
                            )
                            ranks = launch_round(graph, mode, offsets, profile_sample)
                            record = {
                                "k": k,
                                "gemm_repeats": repeats,
                                "mode": mode,
                                "sample": sample,
                                "arrival_model": args.arrival_model,
                                "delay_scenario": scenario,
                                "delay_label": delay_label(
                                    scenario, args.arrival_model
                                ),
                                "base_offsets_us": scenario.get("offsets_us"),
                                "requested_offsets_us": offsets,
                                "ranks": ranks,
                                "metrics": trial_metrics(ranks),
                            }
                            records[mode].append(record)
                            if raw_file is not None:
                                raw_file.write(json.dumps(record) + "\n")
                                raw_file.flush()
                dist.barrier()
                if owner.item() != -1 or graph.error.item() != 0:
                    raise RuntimeError(
                        f"rank {rank}: owner not FREE / device error after scenario"
                    )
                dist.barrier()
                if rank == 0:
                    summaries = {
                        mode: summarize(records[mode], args.num_gpus)
                        for mode in args.modes
                    }
                    graph_p50 = {
                        mode: summaries[mode]["max_rank_graph_ms"]["p50"]
                        for mode in args.modes
                    }
                    case = {
                        "k": k,
                        "gemm_repeats": repeats,
                        "delay_scenario": scenario,
                        "delay_label": delay_label(scenario, args.arrival_model),
                        "base_offsets_us": scenario.get("offsets_us"),
                        "modes": summaries,
                        "ratios": graph_ratios(graph_p50),
                        "validation": "payload bytes matched; generation state valid; owner FREE",
                    }
                    if args.arrival_model == "continuous":
                        case["continuous_epochs"] = continuous_epochs
                        case["throughput_ratios"] = continuous_throughput_ratios(
                            continuous_epochs
                        )
                    payload["cases"].append(case)
                    write_report(args.output, payload)
                    if args.arrival_model == "continuous":
                        mode_text = ", ".join(
                            f"{mode}={continuous_epochs[mode]['aggregate_rank_replays_per_second']:.4f} rank-replay/s"
                            for mode in args.modes
                        )
                        comparison_text = ", ".join(
                            f"{name}={(value - 1) * 100:+.2f}%"
                            for name, value in case["throughput_ratios"].items()
                        )
                        mode_text = f"{mode_text}; {comparison_text}"
                    else:
                        mode_text = ", ".join(
                            f"{mode}={value:.4f} ms"
                            for mode, value in graph_p50.items()
                        )
                    print(
                        f"K={k} repeats={repeats} "
                        f"delay={delay_label(scenario, args.arrival_model)}: "
                        f"{mode_text}",
                        flush=True,
                    )
                dist.barrier()
            del graph
            torch.cuda.synchronize()
            dist.barrier()
    if rank == 0:
        payload["status"] = "complete"
        write_report(args.output, payload)
        raw_file.close()
        print(f"Saved {args.output} and .md / .samples.jsonl", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--k-values", default="1,4")
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument(
        "--component-bytes",
        default="235929600,117964800",
        help="Comma-separated communication bytes per layer component",
    )
    parser.add_argument("--gemm-m", type=int, default=512)
    parser.add_argument("--gemm-n", type=int, default=4096)
    parser.add_argument("--gemm-k", type=int, default=4096)
    parser.add_argument(
        "--gemm-repeats-values",
        default="1",
        help="Comma-separated GEMM repeats per simulated layer",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--modes",
        help=(
            "Comma-separated explicit mode subset. Profiler tooling uses this "
            "to capture one case without tracing the full comparison matrix."
        ),
    )
    study = parser.add_mutually_exclusive_group()
    study.add_argument(
        "--slice-study",
        action="store_true",
        help=(
            "Run only compute_only and fixed DMA/Event S=1/2/4 modes; "
            "defaults to random-delay maxima 1000,5000,10000us"
        ),
    )
    study.add_argument(
        "--group-study",
        action="store_true",
        help=(
            "Run fixed DMA unsliced baseline plus S=4 with G=1/2/4 "
            "contiguous owner groups, each with Event and generation-flag "
            "synchronization; defaults to random-delay maxima "
            "1000,5000,10000us"
        ),
    )
    parser.add_argument(
        "--random-delay-max-us",
        default="1000,5000,10000",
        help=(
            "Comma-separated uniform-delay maxima. reset_uniform redraws rank "
            "offsets per sample; continuous draws a per-rank scheduler gap "
            "between consecutive replays. Both pair the random input across modes"
        ),
    )
    parser.add_argument(
        "--arrival-model",
        choices=("reset_uniform", "continuous"),
        default="reset_uniform",
        help=(
            "reset_uniform aligns ranks before every replay; continuous aligns "
            "only once, then lets per-rank completion time and scheduler gaps drift"
        ),
    )
    parser.add_argument(
        "--continuous-replays",
        type=int,
        default=256,
        help="Replays per rank and mode in one continuous epoch",
    )
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--lead-ms", type=float, default=20)
    parser.add_argument("--trace-steps", action="store_true")
    parser.add_argument("--trace-layers", action="store_true")
    parser.add_argument(
        "--nsight-annotations",
        action="store_true",
        help=(
            "Add stable SiDP NVTX roles during graph construction and replay; "
            "disabled on the ordinary benchmark path"
        ),
    )
    parser.add_argument(
        "--nsight-profile-sample",
        type=int,
        default=-1,
        help=(
            "Measured sample index controlled by rank 0 with cudaProfilerApi; "
            "requires exactly one K/repeat/delay/mode case"
        ),
    )
    parser.add_argument(
        "--sm-execution-trace",
        action="store_true",
        help=(
            "Use profiler-only communication/control kernels that record "
            "CTA entry/exit %smid and %globaltimer; requires one Nsight sample"
        ),
    )
    parser.add_argument(
        "--sm-trace-capacity",
        type=int,
        default=131072,
        help="Maximum CTA trace rows per rank; overflow is a hard profiling error",
    )
    parser.add_argument(
        "--sm-copy-ctas",
        type=int,
        default=0,
        help="Explicit SM-copy CTA count; 0 means 4x device SM count",
    )
    parser.add_argument(
        "--sm-copy-block",
        type=int,
        choices=(128, 256, 512),
        default=512,
        help="SM-copy threads per CTA for explicit compute_sm/dynamic_sm modes",
    )
    parser.add_argument(
        "--dump-graph-dir",
        help=(
            "Dump rank-0 retained CUDA Graphs as slim/verbose DOT plus an "
            "Event-handle map; disabled by default"
        ),
    )
    parser.add_argument(
        "--dump-graph-modes",
        help="Comma-separated modes to dump; defaults to all selected modes",
    )
    parser.add_argument("--backoff-ns", type=int, default=500)
    parser.add_argument("--device-timeout-s", type=float, default=30)
    parser.add_argument(
        "--output", help="New JSON path; also writes .md and .samples.jsonl"
    )
    args = parser.parse_args(argv)
    try:
        args.k_values = list(
            dict.fromkeys(csv_numbers(args.k_values, int, 1, "--k-values"))
        )
        args.component_bytes = csv_numbers(
            args.component_bytes, int, 1, "--component-bytes"
        )
        args.gemm_repeats_values = list(
            dict.fromkeys(
                csv_numbers(
                    args.gemm_repeats_values,
                    int,
                    1,
                    "--gemm-repeats-values",
                )
            )
        )
        explicit_modes = args.modes
        if explicit_modes is not None and (args.group_study or args.slice_study):
            raise argparse.ArgumentTypeError(
                "--modes cannot be combined with --slice-study/--group-study"
            )
        if explicit_modes is not None:
            args.modes = list(
                dict.fromkeys(
                    item.strip() for item in explicit_modes.split(",") if item.strip()
                )
            )
            invalid_modes = set(args.modes) - set(ALL_MODES)
            if not args.modes or invalid_modes:
                raise argparse.ArgumentTypeError(
                    "--modes must be a non-empty subset of supported modes; "
                    f"invalid={sorted(invalid_modes)}"
                )
        elif args.group_study:
            args.modes = list(GROUP_STUDY_MODES)
        elif args.slice_study:
            args.modes = list(SLICE_STUDY_MODES)
        else:
            args.modes = list(MODES)
        if args.dump_graph_modes is None:
            args.dump_graph_modes = list(args.modes)
        else:
            args.dump_graph_modes = list(
                dict.fromkeys(
                    item.strip()
                    for item in args.dump_graph_modes.split(",")
                    if item.strip()
                )
            )
            invalid_dump_modes = set(args.dump_graph_modes) - set(args.modes)
            if not args.dump_graph_modes or invalid_dump_modes:
                raise argparse.ArgumentTypeError(
                    "--dump-graph-modes must be a non-empty subset of selected "
                    f"modes; invalid={sorted(invalid_dump_modes)}"
                )
        maxima = csv_numbers(
            args.random_delay_max_us,
            float,
            0,
            "--random-delay-max-us",
        )
        args.delay_scenarios = [
            {"kind": "random_uniform", "max_us": maximum, "index": index}
            for index, maximum in enumerate(maxima)
        ]
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.num_gpus < 2 or any(k >= args.num_gpus for k in args.k_values):
        parser.error("require num-gpus >= 2 and 1 <= K < num-gpus")
    if args.cycles < CACHE_DEPTH or args.cycles % CACHE_DEPTH:
        parser.error("--cycles must be even and at least 2")
    if min(args.gemm_m, args.gemm_n, args.gemm_k) < 1:
        parser.error("GEMM dimensions must be positive")
    if (
        args.iterations < 1
        or args.continuous_replays < 1
        or args.warmup < 0
        or args.backoff_ns < 0
    ):
        parser.error(
            "iterations/continuous-replays must be positive; "
            "warmup/backoff must be non-negative"
        )
    if args.sm_trace_capacity < 1 or args.sm_copy_ctas < 0:
        parser.error("SM trace capacity must be positive; SM-copy CTAs must be non-negative")
    if args.nsight_profile_sample < -1 or args.nsight_profile_sample >= args.iterations:
        parser.error("--nsight-profile-sample must be -1 or a measured sample index")
    if args.arrival_model == "continuous" and args.nsight_profile_sample >= 0:
        parser.error(
            "--arrival-model continuous cannot use --nsight-profile-sample; "
            "profiling collectives would destroy continuous arrival semantics"
        )
    if args.nsight_profile_sample >= 0:
        profiler_dimensions = (
            len(args.k_values),
            len(args.gemm_repeats_values),
            len(args.delay_scenarios),
            len(args.modes),
        )
        if profiler_dimensions != (1, 1, 1, 1):
            parser.error(
                "--nsight-profile-sample requires exactly one K, repeat, delay, "
                "and mode so one cudaProfilerApi range identifies one case"
            )
        if not args.nsight_annotations:
            parser.error("--nsight-profile-sample requires --nsight-annotations")
    if args.sm_execution_trace and args.nsight_profile_sample < 0:
        parser.error("--sm-execution-trace requires --nsight-profile-sample")
    if (
        not all(
            math.isfinite(value)
            for value in (args.lead_ms, args.device_timeout_s)
        )
        or args.lead_ms <= 0
        or args.device_timeout_s <= 0
    ):
        parser.error("invalid lead time or device timeout")
    if args.output is None:
        args.output = (
            "check_logs/sidp_dma_compute_pipeline_"
            f"{datetime.now().astimezone():%Y%m%d_%H%M%S}.json"
        )
    args.command = shlex.join([sys.executable, *sys.argv])
    return args


def main():
    args = parse_args()
    import torch

    if args.num_gpus > torch.cuda.device_count():
        raise RuntimeError("Not enough visible CUDA GPUs")
    try:
        args.git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=SGLANG_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        args.git_commit = "unknown"
    args.source_sha256 = {
        str(path.relative_to(SGLANG_ROOT)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in (
            Path(__file__).resolve(),
            SGLANG_ROOT / "python/sglang/kernels/jit/csrc/sidp/dma_graph.cuh",
            SGLANG_ROOT / "python/sglang/kernels/jit/csrc/sidp/sm_copy.cuh",
            SGLANG_ROOT / "python/sglang/srt/layers/sidp/cuda_memcpy.py",
        )
    }
    output = Path(args.output)
    for path in (
        output,
        output.with_suffix(".md"),
        output.with_suffix(".samples.jsonl"),
    ):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite benchmark evidence: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    torch.multiprocessing.spawn(
        worker, args=(args, port), nprocs=args.num_gpus, join=True
    )


if __name__ == "__main__":
    main()
