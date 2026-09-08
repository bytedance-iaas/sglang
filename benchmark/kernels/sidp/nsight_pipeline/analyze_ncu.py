"""Parse Nsight Compute CSV and produce evidence-bounded regression hints."""

from __future__ import annotations

import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


CORE_METRIC_TOKENS = (
    "time_duration",
    "throughput",
    "occupancy",
    "register",
    "shared_mem",
    "shared memory",
    "waves_per_sm",
    "waves_per_multiprocessor",
    "cycles_active",
    "pipe_lsu",
    "dram__",
    "lts__",
    "l1tex__",
    "tensor",
    "pipe_tensor",
    "ipc",
    "issue",
    "stall",
    "warps_active",
    "smsp__issue",
    "roofline",
    "arithmetic_intensity",
    "flop",
    "grid_size",
    "block_size",
)


def _number(value):
    value = str(value).strip().replace(",", "")
    if not value or value.lower() in {"n/a", "nan", "inf", "-inf"}:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _find_header(lines):
    for index, line in enumerate(lines):
        lowered = line.lower()
        if "metric name" in lowered and "metric value" in lowered:
            return index, "long"
        if "kernel name" in lowered and "gpu__time_duration" in lowered:
            return index, "wide"
    return None


def parse_ncu_csv(paths):
    launches = defaultdict(lambda: {"metrics": {}, "identity": {}})
    limitations = []
    for path in [Path(path) for path in paths]:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        header_info = _find_header(lines)
        if header_info is None:
            limitations.append(f"no NCU metric header in {path}")
            continue
        header, layout = header_info
        if layout == "long":
            reader = csv.DictReader(lines[header:])
            for row in reader:
                normalized = {str(key).strip('" '): value for key, value in row.items()}
                metric_name = normalized.get("Metric Name", "").strip()
                if not metric_name:
                    continue
                kernel_name = normalized.get("Kernel Name", "unknown").strip()
                launch_id = (
                    normalized.get("Process ID", "-1"),
                    normalized.get("Device", "unknown"),
                    normalized.get("Context", "-1"),
                    normalized.get("Stream", "-1"),
                    normalized.get("ID", "-1"),
                    kernel_name,
                )
                launches[launch_id]["identity"] = {
                    "process_id": launch_id[0],
                    "device": launch_id[1],
                    "context": launch_id[2],
                    "stream": launch_id[3],
                    "launch_id": launch_id[4],
                    "kernel_name": kernel_name,
                    "grid_size": normalized.get("Grid Size"),
                    "block_size": normalized.get("Block Size"),
                }
                value = _number(normalized.get("Metric Value"))
                if value is not None:
                    launches[launch_id]["metrics"][metric_name] = {
                        "value": value,
                        "unit": normalized.get("Metric Unit", ""),
                    }
        else:
            rows = list(csv.reader(lines[header:]))
            names = rows[0]
            units = rows[1] if len(rows) > 1 and _number(rows[1][0]) is None else []
            indexes = {name: index for index, name in enumerate(names)}
            required = ("ID", "Process ID", "Kernel Name", "Device", "Context", "Stream")
            if any(name not in indexes for name in required):
                limitations.append(f"unsupported NCU wide CSV identity columns in {path}")
                continue
            for row in rows[1:]:
                if len(row) != len(names) or _number(row[indexes["ID"]]) is None:
                    continue
                kernel_name = row[indexes["Kernel Name"]].strip()
                launch_id = (
                    row[indexes["Process ID"]],
                    row[indexes["Device"]],
                    row[indexes["Context"]],
                    row[indexes["Stream"]],
                    row[indexes["ID"]],
                    kernel_name,
                )
                launches[launch_id]["identity"] = {
                    "process_id": launch_id[0],
                    "device": launch_id[1],
                    "context": launch_id[2],
                    "stream": launch_id[3],
                    "launch_id": launch_id[4],
                    "kernel_name": kernel_name,
                    "grid_size": row[indexes.get("Grid Size", 0)],
                    "block_size": row[indexes.get("Block Size", 0)],
                }
                for index, metric_name in enumerate(names):
                    if "__" not in metric_name and not metric_name.startswith(
                        ("launch__", "derived__")
                    ):
                        continue
                    value = _number(row[index])
                    if value is not None:
                        launches[launch_id]["metrics"][metric_name] = {
                            "value": value,
                            "unit": units[index] if len(units) == len(names) else "",
                        }
    launch_rows = list(launches.values())
    metric_values = defaultdict(list)
    metric_units = {}
    for launch in launch_rows:
        for name, metric in launch["metrics"].items():
            if any(token in name.lower() for token in CORE_METRIC_TOKENS):
                metric_values[name].append(metric["value"])
                metric_units[name] = metric["unit"]
    metrics = {
        name: {
            "unit": metric_units.get(name, ""),
            "count": len(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "cv": (
                statistics.pstdev(values) / statistics.mean(values)
                if len(values) > 1 and statistics.mean(values)
                else 0.0
            ),
        }
        for name, values in sorted(metric_values.items())
    }
    if not launch_rows:
        limitations.append("NCU produced no parsed kernel launch")
    return {
        "csv_paths": [str(path) for path in paths],
        "launch_count": len(launch_rows),
        "kernels": sorted(
            {launch["identity"]["kernel_name"] for launch in launch_rows}
        ),
        "launch_identities": [launch["identity"] for launch in launch_rows],
        "metrics": metrics,
        "limitations": limitations,
    }


def compare_ncu_metrics(baseline, variant):
    result = []
    for name, current in variant.get("metrics", {}).items():
        previous = baseline.get("metrics", {}).get(name)
        if previous is None or previous["unit"] != current["unit"]:
            continue
        old, new = previous["mean"], current["mean"]
        result.append(
            {
                "metric": name,
                "unit": current["unit"],
                "baseline": old,
                "variant": new,
                "delta": new - old,
                "ratio": new / old if old else None,
            }
        )
    return sorted(result, key=lambda item: item["metric"])


def unit_rollup_dispersion(profile):
    """Summarize hardware-unit rollups without calling them per-SM values."""
    grouped = defaultdict(dict)
    for name, metric in profile.get("metrics", {}).items():
        if not name.startswith(("sm__", "smsp__", "l1tex__", "lts__", "dram__")):
            continue
        match = re.match(r"^(.*)\.(avg|min|max|sum)(\..*)?$", name)
        if not match:
            continue
        prefix, rollup, suffix = match.groups()
        grouped[f"{prefix}{suffix or ''}"][rollup] = {
            "metric": name,
            "value": metric["mean"],
            "unit": metric["unit"],
        }
    rows = []
    for base, values in grouped.items():
        if not {"avg", "min", "max"}.issubset(values):
            continue
        units = {values[key]["unit"] for key in ("avg", "min", "max")}
        if len(units) != 1:
            continue
        average = values["avg"]["value"]
        minimum = values["min"]["value"]
        maximum = values["max"]["value"]
        unit = next(iter(units))
        rows.append(
            {
                "metric_family": base,
                "unit": unit,
                "avg": average,
                "min": minimum,
                "max": maximum,
                "max_to_avg": maximum / average if average else None,
                "range_over_avg": (maximum - minimum) / average if average else None,
                # Tiny percentage counters can have spectacular ratios while being
                # operationally irrelevant.  Keep them available in analysis.json,
                # but rank materially active units first in the human report.
                "material": not (unit == "%" and abs(average) < 0.1),
                "source_metrics": {
                    key: values[key]["metric"] for key in ("avg", "min", "max")
                },
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            item["material"],
            item["range_over_avg"]
            if item["range_over_avg"] is not None
            else -math.inf,
        ),
        reverse=True,
    )


def overlap_contention_hypotheses(
    *,
    systems_mean_ratio,
    compute_overlap_ratio,
    overlap_by_role_ratio,
    compute_sm_side_overlap_ratio,
    isolated_node_baseline,
    isolated_node_variant,
    workload_baseline,
    workload_variant,
    sm_trace_roles,
):
    """Combine orthogonal evidence while preserving attribution boundaries."""
    findings = []
    node_base_duration = _metric(
        isolated_node_baseline.get("metrics", {}), "gpu__time_duration"
    )
    node_variant_duration = _metric(
        isolated_node_variant.get("metrics", {}), "gpu__time_duration"
    )
    node_ratio = (
        node_variant_duration / node_base_duration
        if node_base_duration and node_variant_duration
        else None
    )
    findings.append(
        {
            "kind": "runtime_overlap",
            "confidence": "fact",
            "evidence": (
                f"Systems compute mean ratio={systems_mean_ratio:.3f}, "
                f"compute-overlap ratio={compute_overlap_ratio:.3f}"
                if systems_mean_ratio is not None
                else f"Systems compute-overlap ratio={compute_overlap_ratio:.3f}"
            ),
        }
    )
    if node_ratio is not None:
        findings.append(
            {
                "kind": "isolated_node_repeatability",
                "confidence": "fact",
                "evidence": f"NCU node-replay GEMM duration ratio={node_ratio:.3f}",
                "interpretation": (
                    "Node replay isolates the kernel and does not preserve the actual "
                    "compute/communication overlap."
                ),
            }
        )
    if (
        systems_mean_ratio is not None
        and systems_mean_ratio > 1.03
        and node_ratio is not None
        and node_ratio < 1.03
        and compute_overlap_ratio > 0.1
    ):
        findings.append(
            {
                "kind": "concurrency_specific_slowdown",
                "confidence": "supported",
                "evidence": (
                    "GEMM slows down in the real Systems timeline but the slowdown "
                    "largely disappears under isolated NCU node replay."
                ),
                "interpretation": (
                    "This supports overlap-time resource contention or scheduling, "
                    "rather than a changed GEMM binary or launch shape."
                ),
            }
        )

    sm_role_map = {item["role"]: item for item in sm_trace_roles}
    sm_copy_coverage = sm_role_map.get("communication_sm", {}).get(
        "device_sm_coverage_median", 0.0
    )
    active_sm_overlap = compute_sm_side_overlap_ratio
    if active_sm_overlap > 0:
        findings.append(
            {
                "kind": "communication_kernel_residency",
                "confidence": "fact",
                "evidence": (
                    f"Systems compute overlap with the union of SM-side "
                    f"communication/control kernels="
                    f"{active_sm_overlap:.3f}; traced SM-copy median SM coverage="
                    f"{sm_copy_coverage:.3f}."
                ),
                "interpretation": (
                    "Exact SMIDs apply only to instrumented communication CTAs; "
                    "opaque GEMM SM placement remains unknown."
                ),
            }
        )
    if (
        systems_mean_ratio is not None
        and systems_mean_ratio > 1.03
        and overlap_by_role_ratio.get("communication_sm", 0.0) > 0.05
        and sm_copy_coverage > 0.1
    ):
        findings.append(
            {
                "kind": "sm_execution_contention",
                "confidence": "plausible",
                "evidence": (
                    "Runtime GEMM slowdown coincides with SM-copy execution and "
                    "SM-copy occupies a non-trivial fraction of device SMIDs."
                ),
                "interpretation": (
                    "SM scheduler/residency, LSU or cache contention is plausible, "
                    "but same-SM GEMM placement is not directly observed."
                ),
            }
        )

    workload_deltas = compare_ncu_metrics(workload_baseline, workload_variant)
    pressure = []
    for row in workload_deltas:
        name = row["metric"].lower()
        if not any(token in name for token in ("dram__throughput", "lts__throughput")):
            continue
        if "pct" not in name or row["delta"] <= 10:
            continue
        pressure.append(row)
    for row in pressure[:4]:
        findings.append(
            {
                "kind": "workload_memory_pressure",
                "confidence": "plausible",
                "evidence": (
                    f"Whole-Graph {row['metric']} changed "
                    f"{row['baseline']:.2f}->{row['variant']:.2f} {row['unit']}."
                ),
                "interpretation": (
                    "The counter covers compute plus communication. It supports "
                    "higher shared-memory-hierarchy pressure, but cannot attribute "
                    "all added traffic to GEMM degradation."
                ),
            }
        )
    if not findings:
        findings.append(
            {
                "kind": "insufficient_evidence",
                "confidence": "unknown",
                "evidence": "No comparable Systems/NCU/SM-trace evidence was available.",
            }
        )
    return findings


def derive_gemm_metrics(profile, settings):
    """Add algorithmic GEMM context without pretending cache traffic is known."""
    metrics = profile.get("metrics", {})
    duration_pair = next(
        (
            (name, value)
            for name, value in metrics.items()
            if "gpu__time_duration.avg" in name
        ),
        None,
    ) or next(
        (
            (name, value)
            for name, value in metrics.items()
            if "gpu__time_duration.sum" in name
        ),
        None,
    )
    duration_entry = duration_pair[1] if duration_pair else None
    duration = duration_entry["mean"] if duration_entry else None
    seconds_per_unit = {
        "ns": 1e-9,
        "nsecond": 1e-9,
        "us": 1e-6,
        "usecond": 1e-6,
        "ms": 1e-3,
        "msecond": 1e-3,
        "s": 1.0,
        "second": 1.0,
    }
    duration_s = None
    if duration is not None and duration_entry is not None:
        duration_s = duration * seconds_per_unit.get(duration_entry["unit"], 0.0)
    flops = 2 * int(settings["gemm_m"]) * int(settings["gemm_n"]) * int(
        settings["gemm_k"]
    )
    dram_bytes = None
    dram_bandwidth_gbps = None
    byte_entry = next(
        (
            value
            for name, value in metrics.items()
            if name.endswith("dram__bytes.sum")
            and "/" not in str(value.get("unit", ""))
        ),
        None,
    )
    if byte_entry is not None:
        scale = {
            "byte": 1.0,
            "Kbyte": 1e3,
            "Mbyte": 1e6,
            "Gbyte": 1e9,
        }.get(byte_entry["unit"])
        if scale:
            dram_bytes = byte_entry["mean"] * scale
    rate_entry = next(
        (
            value
            for name, value in metrics.items()
            if name == "dram__bytes.sum.per_second"
        ),
        None,
    )
    if rate_entry is not None:
        rate_scale = {
            "byte/s": 1e-9,
            "Kbyte/s": 1e-6,
            "Mbyte/s": 1e-3,
            "Gbyte/s": 1.0,
            "Tbyte/s": 1e3,
        }.get(rate_entry["unit"])
        if rate_scale is not None:
            dram_bandwidth_gbps = rate_entry["mean"] * rate_scale
            if dram_bytes is None and duration_s:
                dram_bytes = dram_bandwidth_gbps * 1e9 * duration_s
    return {
        "algorithmic_flops": flops,
        "achieved_tflops": flops / duration_s / 1e12 if duration_s else None,
        "ncu_dram_bytes": dram_bytes,
        "ncu_dram_bandwidth_gbps": dram_bandwidth_gbps,
        "algorithmic_flops_per_ncu_dram_byte": (
            flops / dram_bytes if dram_bytes else None
        ),
        "note": (
            "FLOPs use 2*M*N*K for this benchmark GEMM. When NCU exports only "
            "dram__bytes.sum.per_second, bytes are reconstructed as rate*duration. "
            "They reflect measured cache behavior, not tensor-allocation bytes."
        ),
    }


def _metric(metrics, *tokens):
    candidates = [
        (name, value)
        for name, value in metrics.items()
        if all(token.lower() in name.lower() for token in tokens)
    ]
    if not candidates:
        return None
    # Prefer the shortest exact-looking metric rather than an expanded submetric.
    return min(candidates, key=lambda item: len(item[0]))[1]["mean"]


def regression_hypotheses(baseline, variant):
    """Return observations, not unconditional root-cause claims."""
    base = baseline.get("metrics", {})
    current = variant.get("metrics", {})
    duration_base = _metric(base, "gpu__time_duration")
    duration_current = _metric(current, "gpu__time_duration")
    observations = []
    if duration_base and duration_current:
        observations.append(
            {
                "kind": "duration",
                "evidence": f"NCU kernel duration ratio={duration_current / duration_base:.3f}",
            }
        )
    achieved_base = _metric(base, "achieved_occupancy") or _metric(
        base, "warps_active", "pct"
    )
    achieved_current = _metric(current, "achieved_occupancy") or _metric(
        current, "warps_active", "pct"
    )
    if achieved_base is not None and achieved_current is not None:
        delta = achieved_current - achieved_base
        if delta < -5:
            observations.append(
                {
                    "kind": "occupancy_drop",
                    "evidence": f"achieved occupancy changed by {delta:.2f} percentage points",
                    "interpretation": "Lower resident warp capacity may contribute; check register/shared-memory launch limits.",
                }
            )
    for kind, tokens, label in (
        ("dram_pressure", ("dram__throughput", "pct"), "DRAM throughput"),
        ("l2_pressure", ("lts__throughput", "pct"), "L2 throughput"),
        ("sm_pressure", ("sm__throughput", "pct"), "SM throughput"),
    ):
        old = _metric(base, *tokens)
        new = _metric(current, *tokens)
        if old is not None and new is not None and new - old > 10:
            observations.append(
                {
                    "kind": kind,
                    "evidence": f"{label} changed from {old:.2f}% to {new:.2f}%",
                    "interpretation": "The higher utilization is consistent with added contention, but must be correlated with duration and stalls.",
                }
            )
    stall_deltas = []
    for name, metric in current.items():
        if "stall" not in name.lower() or name not in base:
            continue
        delta = metric["mean"] - base[name]["mean"]
        if delta > 0:
            stall_deltas.append((delta, name, base[name]["mean"], metric["mean"]))
    for delta, name, old, new in sorted(stall_deltas, reverse=True)[:5]:
        observations.append(
            {
                "kind": "warp_stall_increase",
                "evidence": f"{name}: {old:.4g} -> {new:.4g} ({delta:+.4g})",
            }
        )
    if not observations:
        observations.append(
            {
                "kind": "insufficient_evidence",
                "evidence": "Selected NCU metrics do not isolate a dominant microarchitectural regression.",
            }
        )
    return observations
