"""Parse an Nsight Systems SQLite export into SiDP pipeline metrics."""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .interval_metrics import (
    Interval,
    bubbles_with_neighbors,
    describe_ns,
    intersect_intervals,
    interval_total,
    merge_intervals,
)


KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
MEMCPY_TABLE = "CUPTI_ACTIVITY_KIND_MEMCPY"


def _table_columns(connection, table):
    return [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]


def _pick(columns, *names, default="NULL"):
    for name in names:
        if name in columns:
            return f'"{name}"'
    return default


def _strings(connection):
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if "StringIds" not in tables:
        return {}
    columns = _table_columns(connection, "StringIds")
    id_col = _pick(columns, "id")
    value_col = _pick(columns, "value", "str", "string")
    if "NULL" in (id_col, value_col):
        return {}
    return {
        int(row[0]): str(row[1])
        for row in connection.execute(
            f"SELECT {id_col}, {value_col} FROM StringIds"
        )
    }


def classify_kernel(name):
    lower = name.lower()
    if any(
        token in lower for token in ("gemm", "matmul", "cutlass", "mma", "nvjet")
    ):
        return "compute"
    if any(
        token in lower
        for token in (
            "wait_generation",
            "wait_flag",
            "wait_kernel",
            "claim_owner",
        )
    ):
        return "wait_or_claim"
    if any(
        token in lower
        for token in (
            "release_owner",
            "publish_generation",
            "publish_selected_fill",
            "reset_forward",
            "reset_cycle",
            "reset_sm_trace",
            "select_candidate",
            "select_fixed",
            "set_dma_condition",
            "memset",
        )
    ):
        return "control"
    if any(token in lower for token in ("sm_copy", "copy_selected")):
        return "communication_sm"
    return "other_kernel"


def _load_kernels(connection, strings):
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if KERNEL_TABLE not in tables:
        return [], f"missing {KERNEL_TABLE}"
    columns = _table_columns(connection, KERNEL_TABLE)
    fields = {
        "start": _pick(columns, "start"),
        "end": _pick(columns, "end"),
        "device": _pick(columns, "deviceId", default="-1"),
        "stream": _pick(columns, "streamId", default="-1"),
        "pid": _pick(columns, "globalPid", "processId", default="-1"),
        "name": _pick(columns, "demangledName", "shortName", "name", default="-1"),
        "grid_x": _pick(columns, "gridX", default="-1"),
        "grid_y": _pick(columns, "gridY", default="-1"),
        "grid_z": _pick(columns, "gridZ", default="-1"),
        "block_x": _pick(columns, "blockX", default="-1"),
        "block_y": _pick(columns, "blockY", default="-1"),
        "block_z": _pick(columns, "blockZ", default="-1"),
        "shared": _pick(
            columns, "dynamicSharedMemory", "sharedMemoryExecuted", default="-1"
        ),
    }
    if fields["start"] == "NULL" or fields["end"] == "NULL":
        return [], f"{KERNEL_TABLE} lacks start/end"
    query = "SELECT " + ",".join(
        f"{expression} AS {name}" for name, expression in fields.items()
    ) + f' FROM "{KERNEL_TABLE}"'
    result = []
    for row in connection.execute(query):
        values = dict(zip(fields, row))
        raw_name = values["name"]
        name = strings.get(raw_name, str(raw_name))
        start, end = int(values["start"]), int(values["end"])
        if end <= start:
            continue
        result.append(
            {
                **values,
                "name": name,
                "start": start,
                "end": end,
                "duration_ns": end - start,
                "device": int(values["device"]),
                "stream": int(values["stream"]),
                "pid": int(values["pid"]),
                "role": classify_kernel(name),
            }
        )
    return result, None


def _enum_values(connection, pattern):
    result = {}
    tables = [
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        if pattern in row[0].upper()
    ]
    for table in tables:
        columns = _table_columns(connection, table)
        id_col = _pick(columns, "id", "value", "key")
        label_col = _pick(columns, "label", "name", "description")
        if "NULL" in (id_col, label_col) or id_col == label_col:
            continue
        try:
            for key, value in connection.execute(
                f'SELECT {id_col}, {label_col} FROM "{table}"'
            ):
                result[int(key)] = str(value)
        except (sqlite3.DatabaseError, TypeError, ValueError):
            continue
    return result


def _load_memcpy(connection):
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if MEMCPY_TABLE not in tables:
        return [], f"missing {MEMCPY_TABLE}"
    columns = _table_columns(connection, MEMCPY_TABLE)
    fields = {
        "start": _pick(columns, "start"),
        "end": _pick(columns, "end"),
        "device": _pick(columns, "deviceId", default="-1"),
        "stream": _pick(columns, "streamId", default="-1"),
        "pid": _pick(columns, "globalPid", "processId", default="-1"),
        "bytes": _pick(columns, "bytes", "size", default="0"),
        "copy_kind": _pick(columns, "copyKind", "kind", default="-1"),
        "src_kind": _pick(columns, "srcKind", default="-1"),
        "dst_kind": _pick(columns, "dstKind", default="-1"),
    }
    if fields["start"] == "NULL" or fields["end"] == "NULL":
        return [], f"{MEMCPY_TABLE} lacks start/end"
    kind_values = _enum_values(connection, "MEMCPY")
    query = "SELECT " + ",".join(
        f"{expression} AS {name}" for name, expression in fields.items()
    ) + f' FROM "{MEMCPY_TABLE}"'
    result = []
    for row in connection.execute(query):
        values = dict(zip(fields, row))
        start, end = int(values["start"]), int(values["end"])
        if end <= start:
            continue
        copy_kind = int(values["copy_kind"])
        name = kind_values.get(copy_kind, f"memcpy_kind_{copy_kind}")
        result.append(
            {
                **values,
                "name": name,
                "start": start,
                "end": end,
                "duration_ns": end - start,
                "device": int(values["device"]),
                "stream": int(values["stream"]),
                "pid": int(values["pid"]),
                "bytes": int(values["bytes"]),
                "role": "communication_dma",
            }
        )
    return result, None


def _operator_key(kernel):
    signature = (
        kernel["grid_x"],
        kernel["grid_y"],
        kernel["grid_z"],
        kernel["block_x"],
        kernel["block_y"],
        kernel["block_z"],
        kernel["shared"],
    )
    return json.dumps([kernel["role"], kernel["name"], *signature], separators=(",", ":"))


def aggregate_operators(kernels, windows):
    groups = defaultdict(list)
    exemplars = {}
    for kernel in kernels:
        key = _operator_key(kernel)
        groups[key].append(kernel["duration_ns"])
        exemplars[key] = kernel
    result = []
    total_window = sum(end - start for start, end in windows.values()) or 1
    for key, durations in groups.items():
        exemplar = exemplars[key]
        stats = describe_ns(durations)
        result.append(
            {
                "operator_id": key,
                "role": exemplar["role"],
                "name": exemplar["name"],
                "grid": [
                    exemplar["grid_x"],
                    exemplar["grid_y"],
                    exemplar["grid_z"],
                ],
                "block": [
                    exemplar["block_x"],
                    exemplar["block_y"],
                    exemplar["block_z"],
                ],
                "dynamic_shared_bytes": exemplar["shared"],
                "devices": sorted(
                    {kernel["device"] for kernel in kernels if _operator_key(kernel) == key}
                ),
                "critical_window_ratio": stats["total_ms"] * 1e6 / total_window,
                **stats,
            }
        )
    return sorted(result, key=lambda item: item["total_ms"], reverse=True)


def _device_metrics(device, activities):
    activities = sorted(activities, key=lambda item: (item["start"], item["end"]))
    start = min(item["start"] for item in activities)
    end = max(item["end"] for item in activities)
    by_role = defaultdict(list)
    intervals = []
    for item in activities:
        interval = Interval(
            item["start"],
            item["end"],
            item["role"],
            item["name"],
            device,
            item["stream"],
            item["pid"],
        )
        by_role[item["role"]].append(interval)
        intervals.append(interval)
    compute = by_role["compute"]
    communication = [
        *by_role["communication_dma"],
        *by_role["communication_sm"],
    ]
    compute_ns = interval_total(compute)
    comm_ns = interval_total(communication)
    overlap_ns = interval_total(intersect_intervals(compute, communication))
    all_active_ns = interval_total(intervals)
    useful_ns = interval_total([*compute, *communication])
    role_overlap_ns = {
        role: interval_total(intersect_intervals(compute, role_intervals))
        for role, role_intervals in by_role.items()
        if role != "compute"
    }
    # These roles execute CUDA kernels on SMs.  Take their interval union before
    # intersecting with compute: simply summing per-role overlap can double-count
    # time when control/wait/SM-copy kernels overlap one another.
    sm_side = [
        *by_role["communication_sm"],
        *by_role["wait_or_claim"],
        *by_role["control"],
    ]
    compute_sm_side_overlap_ns = interval_total(
        intersect_intervals(compute, sm_side)
    )
    bubbles = bubbles_with_neighbors(intervals, start, end)
    effective_gaps = bubbles_with_neighbors([*compute, *communication], start, end)
    return {
        "device": device,
        "process_ids": sorted({item["pid"] for item in activities}),
        "streams": sorted({item["stream"] for item in activities}),
        "window_start_ns": start,
        "window_end_ns": end,
        "window_ms": (end - start) / 1e6,
        "compute_union_ms": compute_ns / 1e6,
        "communication_union_ms": comm_ns / 1e6,
        "compute_communication_overlap_ms": overlap_ns / 1e6,
        "compute_overlap_ratio": overlap_ns / compute_ns if compute_ns else 0.0,
        "comm_hidden_ratio": overlap_ns / comm_ns if comm_ns else 0.0,
        "exposed_communication_ms": (comm_ns - overlap_ns) / 1e6,
        "gpu_active_ratio": all_active_ns / (end - start) if end > start else 0.0,
        "useful_compute_or_comm_ratio": useful_ns / (end - start) if end > start else 0.0,
        "wait_or_claim_union_ms": interval_total(by_role["wait_or_claim"]) / 1e6,
        "control_union_ms": interval_total(by_role["control"]) / 1e6,
        "other_kernel_union_ms": interval_total(by_role["other_kernel"]) / 1e6,
        "compute_overlap_by_role_ms": {
            role: value / 1e6 for role, value in sorted(role_overlap_ns.items())
        },
        "compute_overlap_by_role_ratio": {
            role: value / compute_ns if compute_ns else 0.0
            for role, value in sorted(role_overlap_ns.items())
        },
        "compute_sm_side_overlap_ms": compute_sm_side_overlap_ns / 1e6,
        "compute_sm_side_overlap_ratio": (
            compute_sm_side_overlap_ns / compute_ns if compute_ns else 0.0
        ),
        "idle_bubbles": {
            **describe_ns(round(item["duration_ms"] * 1e6) for item in bubbles),
            "largest": sorted(bubbles, key=lambda item: item["duration_ms"], reverse=True)[:10],
        },
        "non_compute_comm_gaps": {
            **describe_ns(round(item["duration_ms"] * 1e6) for item in effective_gaps),
            "largest": sorted(
                effective_gaps, key=lambda item: item["duration_ms"], reverse=True
            )[:10],
        },
        "role_counts": dict(Counter(item["role"] for item in activities)),
    }


def _aggregate_dma(memcopies, windows):
    groups = defaultdict(list)
    for item in memcopies:
        groups[(item["device"], item["name"])].append(item)
    result = []
    for (device, kind), copies in groups.items():
        durations = [item["duration_ns"] for item in copies]
        byte_count = sum(item["bytes"] for item in copies)
        duration = sum(durations)
        result.append(
            {
                "device": device,
                "copy_kind": kind,
                "bytes": byte_count,
                "gib": byte_count / 2**30,
                "service_time_ms": duration / 1e6,
                "effective_gbps": byte_count / duration if duration else 0.0,
                **describe_ns(durations),
            }
        )
    total_bytes = sum(item["bytes"] for item in memcopies)
    global_start = min((start for start, _ in windows.values()), default=0)
    global_end = max((end for _, end in windows.values()), default=0)
    return {
        "groups": sorted(result, key=lambda item: (item["device"], item["copy_kind"])),
        "total_bytes": total_bytes,
        "total_gib": total_bytes / 2**30,
        "aggregate_payload_gbps_over_global_window": (
            total_bytes / (global_end - global_start) if global_end > global_start else 0.0
        ),
    }


def _benchmark_semantics(path):
    if not path or not Path(path).exists():
        return None
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        return None
    record = records[-1]
    ranks = record["ranks"]
    mode = record.get("mode")
    raw_waits = (
        [
            layer["raw_wait_ms"]
            for rank in ranks
            for layer in rank.get("layers", [])
            if layer.get("remote")
        ]
        if mode != "compute_only"
        else []
    )
    gemm_windows = [
        layer["gemm_window_ms"]
        for rank in ranks
        for layer in rank.get("layers", [])
    ]
    comm_cycles = [
        cycle["elapsed_ms"] for rank in ranks for cycle in rank.get("comm_cycles", [])
    ]
    boundaries = [rank["compute_boundaries"] for rank in ranks]
    cycle_ids = sorted(
        {
            cycle["cycle"]
            for rank in ranks
            for cycle in rank.get("compute_cycles", [])
        }
    )
    per_cycle = []
    for cycle_id in cycle_ids:
        compute_values = [
            cycle["elapsed_ms"]
            for rank in ranks
            for cycle in rank.get("compute_cycles", [])
            if cycle["cycle"] == cycle_id
        ]
        comm_values = [
            cycle["elapsed_ms"]
            for rank in ranks
            for cycle in rank.get("comm_cycles", [])
            if cycle["cycle"] == cycle_id
        ]
        layer_rows = [
            layer
            for rank in ranks
            for layer in rank.get("layers", [])
            if layer["cycle"] == cycle_id
        ]
        raw_values = (
            [layer["raw_wait_ms"] for layer in layer_rows if layer.get("remote")]
            if mode != "compute_only"
            else []
        )
        step_rows = [
            step
            for rank in ranks
            for step in rank.get("comm_steps", [])
            if step.get("operation") == ("next_c0" if cycle_id == 0 else f"c{cycle_id}")
        ]
        per_cycle.append(
            {
                "cycle": cycle_id,
                "communication_operation": (
                    "next_c0" if cycle_id == 0 and comm_values else f"c{cycle_id}"
                ),
                "compute_ms": describe_ns(round(value * 1e6) for value in compute_values),
                "comm_ms": describe_ns(round(value * 1e6) for value in comm_values),
                "raw_wait_ms": describe_ns(round(value * 1e6) for value in raw_values),
                "gemm_window_ms": describe_ns(
                    round(layer["gemm_window_ms"] * 1e6) for layer in layer_rows
                ),
                "war_or_claim_wait_ms": describe_ns(
                    round(step["claim_or_war_wait_ms"] * 1e6) for step in step_rows
                ),
                "transfer_window_ms": describe_ns(
                    round(step["transfer_window_ms"] * 1e6) for step in step_rows
                ),
            }
        )
    sm_trace_by_rank = []
    role_values = defaultdict(
        lambda: {
            "unique_sm_count": [],
            "device_sm_coverage": [],
            "cta_records": 0,
            "kernel_launches": 0,
            "smid_migrations": 0,
        }
    )
    for rank in ranks:
        trace = rank.get("sm_execution_trace")
        if not trace:
            continue
        sm_trace_by_rank.append(
            {
                "rank": rank.get("rank"),
                "record_count": trace.get("record_count", 0),
                "capacity": trace.get("capacity", 0),
                "overflow": trace.get("overflow", False),
                "roles": trace.get("roles", []),
            }
        )
        for role in trace.get("roles", []):
            values = role_values[role["role"]]
            values["unique_sm_count"].append(role.get("unique_sm_count", 0))
            values["device_sm_coverage"].append(
                role.get("device_sm_coverage", 0.0)
            )
            for field in ("cta_records", "kernel_launches", "smid_migrations"):
                values[field] += role.get(field, 0)
    sm_trace_roles = []
    for role, values in sorted(role_values.items()):
        sm_trace_roles.append(
            {
                "role": role,
                "rank_count": len(values["unique_sm_count"]),
                "unique_sm_count_median": statistics.median(
                    values["unique_sm_count"]
                ),
                "unique_sm_count_max": max(values["unique_sm_count"], default=0),
                "device_sm_coverage_median": statistics.median(
                    values["device_sm_coverage"]
                ),
                "device_sm_coverage_max": max(
                    values["device_sm_coverage"], default=0.0
                ),
                "cta_records": values["cta_records"],
                "kernel_launches": values["kernel_launches"],
                "smid_migrations": values["smid_migrations"],
            }
        )
    return {
        "sample": record.get("sample"),
        "mode": mode,
        "k": record.get("k"),
        "requested_offsets_us": record.get("requested_offsets_us"),
        "trial_metrics": record.get("metrics"),
        "raw_wait_ms": describe_ns(round(value * 1e6) for value in raw_waits),
        "gemm_window_ms": describe_ns(round(value * 1e6) for value in gemm_windows),
        "comm_cycle_ms": describe_ns(round(value * 1e6) for value in comm_cycles),
        "rank_compute_boundaries": boundaries,
        "cycles": per_cycle,
        "sm_execution_trace": {
            "enabled": bool(sm_trace_by_rank),
            "ranks": sm_trace_by_rank,
            "roles": sm_trace_roles,
            "limitation": (
                "Exact SMIDs cover instrumented communication/control CTAs only; "
                "opaque GEMM placement is not observed."
                if sm_trace_by_rank
                else "SM execution tracing was not enabled for this sample."
            ),
        },
        "max_rank_pre_cycle0_gap_ms": max(
            (item["pre_cycle0_gap_ms"] for item in boundaries), default=0.0
        ),
        "max_rank_tail_join_ms": max(
            (rank["tail_join_ms"] for rank in ranks), default=0.0
        ),
    }


def analyze_sqlite(sqlite_path, benchmark_samples=None):
    sqlite_path = Path(sqlite_path)
    with sqlite3.connect(sqlite_path) as connection:
        tables = sorted(
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        )
        schemas = {table: _table_columns(connection, table) for table in tables}
        strings = _strings(connection)
        kernels, kernel_limitation = _load_kernels(connection, strings)
        memcopies, memcpy_limitation = _load_memcpy(connection)
    activities = [*kernels, *memcopies]
    by_device = defaultdict(list)
    for item in activities:
        if item["device"] >= 0:
            by_device[item["device"]].append(item)
    windows = {
        device: (
            min(item["start"] for item in items),
            max(item["end"] for item in items),
        )
        for device, items in by_device.items()
    }
    devices = [_device_metrics(device, items) for device, items in sorted(by_device.items())]
    starts = [item["window_start_ns"] for item in devices]
    durations = [item["window_ms"] for item in devices]
    benchmark_semantics = _benchmark_semantics(benchmark_samples)
    mode = (benchmark_semantics or {}).get("mode")
    memcpy_expected = mode not in {
        "compute_only",
        "compute_sm_flag",
        "dynamic_sm",
        "dynamic_sm_compute_priority",
    }
    limitations = [
        value
        for value in (
            kernel_limitation,
            memcpy_limitation if memcpy_expected else None,
        )
        if value is not None
    ]
    if not activities:
        limitations.append("no CUDA kernel or memcpy activities were exported")
    return {
        "sqlite_path": str(sqlite_path),
        "sqlite_schema": schemas,
        "limitations": limitations,
        "kernel_count": len(kernels),
        "memcpy_count": len(memcopies),
        "devices": devices,
        "cross_rank": {
            "device_count": len(devices),
            "launch_skew_ms": (max(starts) - min(starts)) / 1e6 if starts else 0.0,
            "window_duration_min_ms": min(durations) if durations else 0.0,
            "window_duration_max_ms": max(durations) if durations else 0.0,
            "window_duration_mean_ms": statistics.mean(durations) if durations else 0.0,
            "window_duration_cv": (
                statistics.pstdev(durations) / statistics.mean(durations)
                if durations and statistics.mean(durations)
                else 0.0
            ),
            "slowest_device": (
                max(devices, key=lambda item: item["window_ms"])["device"]
                if devices
                else None
            ),
        },
        "dma": _aggregate_dma(memcopies, windows),
        "operators": aggregate_operators(kernels, windows),
        "benchmark_semantics": benchmark_semantics,
    }


def match_case_operators(baseline, variant):
    baseline_map = {item["operator_id"]: item for item in baseline["operators"]}
    rows = []
    for current in variant["operators"]:
        previous = baseline_map.get(current["operator_id"])
        if previous is None:
            rows.append(
                {
                    "operator_id": current["operator_id"],
                    "role": current["role"],
                    "name": current["name"],
                    "status": "new",
                    "variant_total_ms": current["total_ms"],
                    "variant_mean_ms": current["mean_ms"],
                }
            )
            continue
        delta_mean = current["mean_ms"] - previous["mean_ms"]
        rows.append(
            {
                "operator_id": current["operator_id"],
                "role": current["role"],
                "name": current["name"],
                "status": "matched",
                "baseline_mean_ms": previous["mean_ms"],
                "variant_mean_ms": current["mean_ms"],
                "mean_ratio": (
                    current["mean_ms"] / previous["mean_ms"]
                    if previous["mean_ms"]
                    else math.inf
                ),
                "matched_count": min(current["count"], previous["count"]),
                "count_times_delta_mean_ms": min(
                    current["count"], previous["count"]
                )
                * delta_mean,
            }
        )
    return sorted(
        rows,
        key=lambda item: item.get(
            "count_times_delta_mean_ms", item.get("variant_total_ms", 0.0)
        ),
        reverse=True,
    )


def select_compute_hotspots(case_analysis, ratio, minimum, maximum, comparison=None):
    candidates = [
        item for item in case_analysis["operators"] if item["role"] == "compute"
    ]
    selected = [item for item in candidates if item["critical_window_ratio"] >= ratio]
    if comparison:
        regressed = {
            item["operator_id"]
            for item in comparison
            if item.get("count_times_delta_mean_ms", 0) > 0
        }
        selected.extend(item for item in candidates if item["operator_id"] in regressed)
    unique = {item["operator_id"]: item for item in selected}
    for item in candidates:
        if len(unique) >= minimum:
            break
        unique[item["operator_id"]] = item
    return sorted(unique.values(), key=lambda item: item["total_ms"], reverse=True)[:maximum]
