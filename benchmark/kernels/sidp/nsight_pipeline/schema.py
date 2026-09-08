"""Configuration validation and command construction."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from . import SCHEMA_VERSION


DEFAULT_COMMON = {
    "num_gpus": 8,
    "cycles": 6,
    "component_bytes": [235929600, 117964800],
    "gemm_m": 512,
    "gemm_n": 4096,
    "gemm_k": 4096,
    "gemm_repeats": 1,
    "random_delay_max_us": 1000,
    "warmup": 1,
    "iterations": 1,
    "seed": 20260904,
    "lead_ms": 20,
    "sm_copy_ctas": 0,
    "sm_copy_block": 512,
    "sm_trace_capacity": 131072,
}

DEFAULT_COLLECTION = {
    "nsys": {
        "enabled": True,
        "path": "/usr/local/bin/nsys",
        "timeout_s": 900,
        "trace": "cuda,nvtx,osrt",
        "cuda_graph_trace": "node:host-only:nvtx-precapture",
    },
    "ncu": {
        "enabled": False,
        "required": False,
        "path": "/usr/local/cuda/bin/ncu",
        "timeout_s": 1800,
        "hotspot_time_ratio": 0.05,
        "min_kernel_num": 1,
        "max_kernel_num": 3,
        "launch_count": 1,
        "node_enabled": True,
        "workload_enabled": True,
        "sections": [
            "SpeedOfLight",
            "SpeedOfLight_RooflineChart",
            "SpeedOfLight_HierarchicalTensorRooflineChart",
            "ComputeWorkloadAnalysis",
            "MemoryWorkloadAnalysis",
            "LaunchStats",
            "Occupancy",
            "SchedulerStats",
        ],
    },
}

MODE_SLICE_SETTINGS = {
    "compute_dma_slice2": (2, 1),
    "compute_dma_slice4": (4, 1),
    "compute_dma_slice4_group2": (4, 2),
    "compute_dma_slice4_group4": (4, 4),
    "compute_dma_slice4_flag": (4, 1),
    "compute_dma_slice4_group2_flag": (4, 2),
    "compute_dma_slice4_group4_flag": (4, 4),
}


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    if not value:
        raise ValueError("case name becomes empty after sanitization")
    return value


def load_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version={raw.get('schema_version')}; "
            f"expected {SCHEMA_VERSION}"
        )
    resolved = copy.deepcopy(raw)
    resolved["schema_version"] = SCHEMA_VERSION
    resolved["working_directory"] = str(
        Path(resolved.get("working_directory", ".")).resolve()
    )
    benchmark = resolved.get(
        "benchmark", "benchmark/kernels/sidp/bench_dma_compute_pipeline.py"
    )
    benchmark = Path(benchmark)
    if not benchmark.is_absolute():
        benchmark = Path(resolved["working_directory"]) / benchmark
    resolved["benchmark"] = str(benchmark.resolve())
    common = {**DEFAULT_COMMON, **resolved.get("common", {})}
    resolved["common"] = common
    collection = copy.deepcopy(DEFAULT_COLLECTION)
    for tool, values in resolved.get("collection", {}).items():
        if tool not in collection:
            raise ValueError(f"unknown collection tool: {tool}")
        collection[tool].update(values)
    resolved["collection"] = collection
    cases = resolved.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("config requires a non-empty cases list")
    seen = set()
    for case in cases:
        for field in ("name", "mode", "k"):
            if field not in case:
                raise ValueError(f"case is missing {field}: {case}")
        case["name"] = safe_name(str(case["name"]))
        if case["name"] in seen:
            raise ValueError(f"duplicate case name: {case['name']}")
        seen.add(case["name"])
        case["k"] = int(case["k"])
        if case["k"] < 1 or case["k"] >= int(common["num_gpus"]):
            raise ValueError(f"case {case['name']} has invalid K={case['k']}")
    ncu = collection["ncu"]
    if not (0 <= float(ncu["hotspot_time_ratio"]) <= 1):
        raise ValueError("ncu.hotspot_time_ratio must be in [0,1]")
    if not (0 <= int(ncu["min_kernel_num"]) <= int(ncu["max_kernel_num"])):
        raise ValueError("require 0 <= min_kernel_num <= max_kernel_num")
    output = resolved.get("output_dir")
    if not output:
        raise ValueError("config requires output_dir under check_logs")
    output = Path(output)
    if not output.is_absolute():
        output = Path(resolved["working_directory"]) / output
    resolved["output_dir"] = str(output.resolve())
    artifact = Path(resolved.get("artifact_dir", f"/tmp/sidp_nsight_{path.stem}"))
    if not artifact.is_absolute():
        artifact = Path(resolved["working_directory"]) / artifact
    resolved["artifact_dir"] = str(artifact.resolve())
    resolved["config_source"] = str(path)
    return resolved


def case_settings(config, case):
    settings = {**config["common"], **case}
    slices, groups = MODE_SLICE_SETTINGS.get(settings["mode"], (1, 1))
    settings["dma_slices"] = slices
    settings["dma_slice_groups"] = groups
    return settings


def benchmark_command(
    config, case, output_path, *, profile_sample=True, sm_execution_trace=False
):
    settings = case_settings(config, case)
    command = [
        settings.get("python", "python3"),
        config["benchmark"],
        "--num-gpus",
        str(settings["num_gpus"]),
        "--k-values",
        str(settings["k"]),
        "--cycles",
        str(settings["cycles"]),
        "--component-bytes",
        ",".join(str(value) for value in settings["component_bytes"]),
        "--gemm-m",
        str(settings["gemm_m"]),
        "--gemm-n",
        str(settings["gemm_n"]),
        "--gemm-k",
        str(settings["gemm_k"]),
        "--gemm-repeats-values",
        str(settings["gemm_repeats"]),
        "--iterations",
        str(settings["iterations"]),
        "--warmup",
        str(settings["warmup"]),
        "--modes",
        settings["mode"],
        "--random-delay-max-us",
        str(settings["random_delay_max_us"]),
        "--seed",
        str(settings["seed"]),
        "--lead-ms",
        str(settings["lead_ms"]),
        "--sm-copy-ctas",
        str(settings["sm_copy_ctas"]),
        "--sm-copy-block",
        str(settings["sm_copy_block"]),
        "--sm-trace-capacity",
        str(settings["sm_trace_capacity"]),
        "--trace-layers",
        "--trace-steps",
        "--nsight-annotations",
        "--output",
        str(output_path),
    ]
    if profile_sample:
        command.extend(["--nsight-profile-sample", "0"])
    if sm_execution_trace:
        command.append("--sm-execution-trace")
    for value in settings.get("extra_args", []):
        command.append(str(value))
    return command
