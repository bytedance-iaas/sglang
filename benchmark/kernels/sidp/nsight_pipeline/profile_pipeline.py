#!/usr/bin/env python3
"""Collect and analyze one or more SiDP pipeline cases without a GUI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from nsight_pipeline import SCHEMA_VERSION  # noqa: E402
from nsight_pipeline.analyze_ncu import (  # noqa: E402
    compare_ncu_metrics,
    derive_gemm_metrics,
    overlap_contention_hypotheses,
    parse_ncu_csv,
    regression_hypotheses,
    unit_rollup_dispersion,
)
from nsight_pipeline.analyze_nsys import (  # noqa: E402
    analyze_sqlite,
    match_case_operators,
    select_compute_hotspots,
)
from nsight_pipeline.report import render_report  # noqa: E402
from nsight_pipeline.schema import (  # noqa: E402
    benchmark_command,
    case_settings,
    load_config,
)


def _run(command, *, cwd, timeout, log_path, manifest, check=True):
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env={**os.environ, **manifest.get("environment", {})},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        captured = error.stdout or ""
        if isinstance(captured, bytes):
            captured = captured.decode(errors="replace")
        completed = subprocess.CompletedProcess(command, 124, captured + "\nTIMEOUT\n")
    output = completed.stdout or ""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(output, encoding="utf-8")
    record = {
        "command": shlex.join(str(value) for value in command),
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "elapsed_s": time.monotonic() - started,
        "log_path": str(log_path),
        "output_tail": output[-8000:],
    }
    manifest["commands"].append(record)
    if check and completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {record['command']}\n"
            f"see {log_path}\n{record['output_tail']}"
        )
    return completed


def _probe(command, timeout=90):
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return result.returncode, (result.stdout or "")
    except (OSError, subprocess.TimeoutExpired) as error:
        return -1, str(error)


def _first_line(text):
    return next((line.strip() for line in text.splitlines() if line.strip()), "unavailable")


def _version_line(text):
    return next(
        (line.strip() for line in text.splitlines() if "version" in line.lower()),
        _first_line(text),
    )


def _tool_manifest(config):
    nsys_path = config["collection"]["nsys"]["path"]
    ncu_path = config["collection"]["ncu"]["path"]
    nsys_status, nsys_version = _probe([nsys_path, "--version"])
    _, nsys_help = _probe([nsys_path, "profile", "--help"])
    ncu_status, ncu_version = _probe([ncu_path, "--version"])
    _, ncu_sections = _probe([ncu_path, "--list-sections"])
    return {
        "nsys": {
            "path": nsys_path,
            "available": nsys_status == 0,
            "version": _version_line(nsys_version),
            "capabilities": {
                "cuda_graph_trace": "--cuda-graph-trace" in nsys_help,
                "nvtx_precapture": "nvtx-precapture" in nsys_help,
                "cuda_profiler_capture": "cudaProfilerApi" in nsys_help,
            },
        },
        "ncu": {
            "path": ncu_path,
            "available": ncu_status == 0,
            "version": _version_line(ncu_version),
            "sections": [
                section
                for section in config["collection"]["ncu"]["sections"]
                if re.search(rf"(?m)^{re.escape(section)}\s", ncu_sections)
            ],
        },
    }


def _system_manifest(config):
    cwd = config["working_directory"]
    git_status, git_commit = _probe(["git", "-C", cwd, "rev-parse", "HEAD"])
    gpu_status, gpu_text = _probe(
        [
            "nvidia-smi",
            "--query-gpu=index,name,pci.bus_id,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    benchmark = Path(config["benchmark"])
    torch_status, torch_text = _probe(
        [
            sys.executable,
            "-c",
            (
                "import json,torch; print(json.dumps({"
                "'torch':torch.__version__,'cuda':torch.version.cuda,"
                "'cuda_available':torch.cuda.is_available()}))"
            ),
        ]
    )
    try:
        torch_info = json.loads(torch_text.strip()) if torch_status == 0 else {}
    except json.JSONDecodeError:
        torch_info = {"probe_error": torch_text[-2000:]}
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "hostname": os.uname().nodename,
        "python": sys.version,
        "pytorch_cuda": torch_info,
        "git_commit": git_commit.strip() if git_status == 0 else "unavailable",
        "benchmark": str(benchmark),
        "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
        "gpus": gpu_text.splitlines() if gpu_status == 0 else [gpu_text],
    }


def _nsys_command(config, benchmark, trace_prefix, limitations):
    tool = config["collection"]["nsys"]
    capability = config["_tool_manifest"]["nsys"]["capabilities"]
    command = [
        tool["path"],
        "profile",
        "--trace",
        tool["trace"],
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        "--output",
        str(trace_prefix),
    ]
    graph_option = tool.get("cuda_graph_trace")
    if graph_option and capability["cuda_graph_trace"]:
        if "nvtx-precapture" in graph_option and not capability["nvtx_precapture"]:
            graph_option = "node:host-only"
            limitations.append(
                "Nsight Systems lacks nvtx-precapture; graph nodes are classified by kernel/memcpy identity only."
            )
        command.extend(["--cuda-graph-trace", graph_option])
    elif graph_option:
        limitations.append(
            "Nsight Systems lacks --cuda-graph-trace; CUDA Graph node visibility is degraded."
        )
    if not capability["cuda_profiler_capture"]:
        command = [
            value
            for value in command
            if value
            not in ("--capture-range=cudaProfilerApi", "--capture-range-end=stop")
        ]
        limitations.append(
            "Nsight Systems lacks cudaProfilerApi capture range; setup activity may enter the trace."
        )
    return [*command, *benchmark]


def _collect_nsys_case(config, case, artifact_case, manifest, limitations):
    nsys = config["collection"]["nsys"]
    benchmark_output = artifact_case / "benchmark.json"
    benchmark = benchmark_command(
        config,
        case,
        benchmark_output,
        sm_execution_trace=case["mode"] != "compute_only",
    )
    trace_prefix = artifact_case / "trace"
    command = _nsys_command(config, benchmark, trace_prefix, limitations)
    _run(
        command,
        cwd=config["working_directory"],
        timeout=nsys["timeout_s"],
        log_path=artifact_case / "nsys_profile.log",
        manifest=manifest,
    )
    reports = sorted(artifact_case.glob("trace*.nsys-rep"))
    if len(reports) != 1:
        raise RuntimeError(f"expected one .nsys-rep under {artifact_case}, got {reports}")
    report = reports[0]
    sqlite_path = artifact_case / "trace.sqlite"
    _run(
        [
            nsys["path"],
            "export",
            "--type=sqlite",
            f"--output={sqlite_path}",
            "--force-overwrite=true",
            "--quiet=true",
            str(report),
        ],
        cwd=config["working_directory"],
        timeout=nsys["timeout_s"],
        log_path=artifact_case / "nsys_export.log",
        manifest=manifest,
    )
    stats = _run(
        [
            nsys["path"],
            "stats",
            "--report",
            "cuda_gpu_kern_sum",
            "--report",
            "cuda_gpu_mem_time_sum",
            "--report",
            "nvtx_gpu_proj_sum",
            str(report),
        ],
        cwd=config["working_directory"],
        timeout=nsys["timeout_s"],
        log_path=artifact_case / "nsys_builtin_stats.txt",
        manifest=manifest,
        check=False,
    )
    if stats.returncode:
        limitations.append(
            f"case {case['name']}: one or more nsys built-in stats reports failed; custom SQLite analysis was still attempted"
        )
    samples = benchmark_output.with_suffix(".samples.jsonl")
    analysis = analyze_sqlite(sqlite_path, samples)
    if case["mode"] == "compute_only":
        analysis["limitations"] = [
            item
            for item in analysis["limitations"]
            if item != "missing CUPTI_ACTIVITY_KIND_MEMCPY"
        ]
    analysis["raw_artifacts"] = {
        "nsys_rep": str(report),
        "sqlite": str(sqlite_path),
        "benchmark_json": str(benchmark_output),
        "benchmark_samples": str(samples),
        "builtin_stats": str(artifact_case / "nsys_builtin_stats.txt"),
    }
    analysis["builtin_stats_excerpt"] = (stats.stdout or "")[-12000:]
    return analysis


def _ncu_filter(kernel_name):
    return "regex:^" + re.escape(kernel_name) + "$"


def _collect_ncu_kernel(config, case, operator, artifact_path, manifest):
    ncu = config["collection"]["ncu"]
    benchmark_output = artifact_path / "benchmark.json"
    benchmark = benchmark_command(config, case, benchmark_output, profile_sample=True)
    prefix = artifact_path / "profile_%p"
    command = [
        ncu["path"],
        "--target-processes",
        "all",
        "--graph-profiling",
        "node",
        "--profile-from-start",
        "off",
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        _ncu_filter(operator["name"]),
        "--launch-count",
        str(ncu["launch_count"]),
        "--force-overwrite",
        "--export",
        str(prefix),
    ]
    for section in config["_tool_manifest"]["ncu"]["sections"]:
        command.extend(["--section", section])
    command.extend(benchmark)
    completed = _run(
        command,
        cwd=config["working_directory"],
        timeout=ncu["timeout_s"],
        log_path=artifact_path / "ncu_profile.log",
        manifest=manifest,
        check=False,
    )
    reports = sorted(artifact_path.glob("profile_*.ncu-rep"))
    csv_paths = []
    for index, report in enumerate(reports):
        imported = _run(
            [
                ncu["path"],
                "--import",
                str(report),
                "--csv",
                "--page",
                "raw",
            ],
            cwd=config["working_directory"],
            timeout=ncu["timeout_s"],
            log_path=artifact_path / f"ncu_import_{index}.csv",
            manifest=manifest,
            check=False,
        )
        csv_paths.append(artifact_path / f"ncu_import_{index}.csv")
        if imported.returncode:
            break
    parsed = parse_ncu_csv(csv_paths)
    if completed.returncode:
        parsed["limitations"].append(
            f"ncu collection exited with {completed.returncode}; see {artifact_path / 'ncu_profile.log'}"
        )
    if not reports:
        parsed["limitations"].append(
            "ncu produced no .ncu-rep; kernel filter mismatch, counter permission, or Graph replay support may be the blocker"
        )
    parsed["raw_artifacts"] = {
        "directory": str(artifact_path),
        "reports": [str(path) for path in reports],
    }
    return parsed


def _collect_ncu_workload(config, case, artifact_path, manifest):
    """Profile the whole CUDA Graph as one concurrent workload.

    Unlike node profiling, graph profiling keeps the graph's internal stream
    topology for each replay pass.  Metrics are therefore aggregate workload
    counters, not counters attributable to the GEMM alone.
    """
    ncu = config["collection"]["ncu"]
    benchmark_output = artifact_path / "benchmark.json"
    benchmark = benchmark_command(
        config, case, benchmark_output, profile_sample=True, sm_execution_trace=False
    )
    prefix = artifact_path / "profile_%p"
    command = [
        ncu["path"],
        "--target-processes",
        "all",
        "--graph-profiling",
        "graph",
        "--replay-mode",
        "kernel",
        "--profile-from-start",
        "off",
        "--launch-count",
        str(ncu["launch_count"]),
        "--force-overwrite",
        "--export",
        str(prefix),
    ]
    for section in config["_tool_manifest"]["ncu"]["sections"]:
        command.extend(["--section", section])
    command.extend(benchmark)
    completed = _run(
        command,
        cwd=config["working_directory"],
        timeout=ncu["timeout_s"],
        log_path=artifact_path / "ncu_profile.log",
        manifest=manifest,
        check=False,
    )
    reports = sorted(artifact_path.glob("profile_*.ncu-rep"))
    csv_paths = []
    for index, report in enumerate(reports):
        imported = _run(
            [
                ncu["path"],
                "--import",
                str(report),
                "--csv",
                "--page",
                "raw",
            ],
            cwd=config["working_directory"],
            timeout=ncu["timeout_s"],
            log_path=artifact_path / f"ncu_import_{index}.csv",
            manifest=manifest,
            check=False,
        )
        csv_paths.append(artifact_path / f"ncu_import_{index}.csv")
        if imported.returncode:
            break
    parsed = parse_ncu_csv(csv_paths)
    parsed["profile_scope"] = "whole_cuda_graph_workload"
    parsed["attribution"] = (
        "Counters cover compute plus communication/control nodes while the graph "
        "retains its internal concurrency; they are not GEMM-only metrics."
    )
    if completed.returncode:
        parsed["limitations"].append(
            f"NCU graph collection exited with {completed.returncode}; see {artifact_path / 'ncu_profile.log'}"
        )
    if not reports:
        parsed["limitations"].append(
            "ncu produced no graph-level .ncu-rep; target NCU may lack graph workload profiling"
        )
    parsed["raw_artifacts"] = {
        "directory": str(artifact_path),
        "reports": [str(path) for path in reports],
    }
    return parsed


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_run_directories(config, output_dir, artifact_dir):
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path(config["working_directory"]).resolve(),
        Path(config["working_directory"]).resolve().parent,
    }
    for label, path in (("output_dir", output_dir), ("artifact_dir", artifact_dir)):
        path = path.resolve()
        if path in protected or len(path.parts) < 3:
            raise ValueError(f"refusing broad {label}: {path}")
    if output_dir.resolve() == artifact_dir.resolve():
        raise ValueError("output_dir and artifact_dir must be different")
    if output_dir.resolve() in artifact_dir.resolve().parents:
        raise ValueError("artifact_dir cannot contain output_dir")
    if artifact_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("output_dir cannot contain artifact_dir")


def run(config, *, force=False, disable_ncu=False):
    output_dir = Path(config["output_dir"])
    artifact_dir = Path(config["artifact_dir"])
    _validate_run_directories(config, output_dir, artifact_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if artifact_dir.exists() and any(artifact_dir.iterdir()) and not force:
        raise FileExistsError(f"artifact directory is not empty: {artifact_dir}")
    if force:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tools = _tool_manifest(config)
    config["_tool_manifest"] = tools
    if config["collection"]["nsys"]["enabled"] and not tools["nsys"]["available"]:
        raise RuntimeError(f"nsys unavailable: {tools['nsys']}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "system": _system_manifest(config),
        "tools": tools,
        "environment": {
            str(key): str(value) for key, value in config.get("environment", {}).items()
        },
        "commands": [],
    }
    resolved = {key: value for key, value in config.items() if key != "_tool_manifest"}
    _write_json(output_dir / "resolved_config.json", resolved)
    limitations = []
    cases = {}
    for case in config["cases"]:
        artifact_case = artifact_dir / case["name"] / "nsys"
        artifact_case.mkdir(parents=True)
        print(f"[SiDP Nsight] Systems case {case['name']} ...", flush=True)
        nsys_analysis = _collect_nsys_case(
            config, case, artifact_case, manifest, limitations
        )
        cases[case["name"]] = {
            "settings": case_settings(config, case),
            "nsys": nsys_analysis,
        }

    baseline_name = config.get("baseline_case", config["cases"][0]["name"])
    if baseline_name not in cases:
        raise ValueError(f"baseline_case does not exist: {baseline_name}")
    comparisons = []
    ncu_config = config["collection"]["ncu"]
    for case in config["cases"]:
        if case["name"] == baseline_name:
            continue
        deltas = match_case_operators(cases[baseline_name]["nsys"], cases[case["name"]]["nsys"])
        hotspots = select_compute_hotspots(
            cases[case["name"]]["nsys"],
            float(ncu_config["hotspot_time_ratio"]),
            int(ncu_config["min_kernel_num"]),
            int(ncu_config["max_kernel_num"]),
            deltas,
        )
        comparisons.append(
            {
                "baseline": baseline_name,
                "variant": case["name"],
                "operator_deltas": deltas,
                "hotspots": hotspots,
            }
        )

    ncu_runs = []
    ncu_enabled = ncu_config["enabled"] and not disable_ncu
    if ncu_enabled and not tools["ncu"]["available"]:
        message = "Nsight Compute requested but unavailable"
        if ncu_config["required"]:
            raise RuntimeError(message)
        limitations.append(message)
        ncu_enabled = False
    case_config = {case["name"]: case for case in config["cases"]}
    ncu_cache = {}
    if ncu_enabled and ncu_config.get("node_enabled", True):
        for comparison in comparisons:
            for operator_index, operator in enumerate(comparison["hotspots"]):
                pair = {}
                for case_name in (comparison["baseline"], comparison["variant"]):
                    cache_key = (case_name, operator["operator_id"])
                    if cache_key not in ncu_cache:
                        artifact_path = (
                            artifact_dir
                            / case_name
                            / "ncu"
                            / f"hotspot_{operator_index}"
                        )
                        artifact_path.mkdir(parents=True, exist_ok=True)
                        print(
                            f"[SiDP Nsight] Compute case {case_name}, hotspot {operator_index} ...",
                            flush=True,
                        )
                        ncu_cache[cache_key] = _collect_ncu_kernel(
                            config,
                            case_config[case_name],
                            operator,
                            artifact_path,
                            manifest,
                        )
                    pair[case_name] = ncu_cache[cache_key]
                base, current = pair[comparison["baseline"]], pair[comparison["variant"]]
                ncu_runs.append(
                    {
                        "baseline": comparison["baseline"],
                        "variant": comparison["variant"],
                        "operator_id": operator["operator_id"],
                        "kernel_name": operator["name"],
                        "systems_mean_ratio": next(
                            (
                                item.get("mean_ratio")
                                for item in comparison["operator_deltas"]
                                if item["operator_id"] == operator["operator_id"]
                            ),
                            None,
                        ),
                        "baseline_launch_count": base["launch_count"],
                        "variant_launch_count": current["launch_count"],
                        "baseline_metrics": base["metrics"],
                        "variant_metrics": current["metrics"],
                        "baseline_derived": derive_gemm_metrics(
                            base,
                            cases[comparison["baseline"]]["settings"],
                        ),
                        "variant_derived": derive_gemm_metrics(
                            current,
                            cases[comparison["variant"]]["settings"],
                        ),
                        "metric_deltas": compare_ncu_metrics(base, current),
                        "hypotheses": regression_hypotheses(base, current),
                        "limitations": [*base["limitations"], *current["limitations"]],
                        "raw_artifacts": {
                            "baseline": base["raw_artifacts"],
                            "variant": current["raw_artifacts"],
                        },
                    }
                )

    workload_runs = []
    workload_cache = {}
    if ncu_enabled and ncu_config.get("workload_enabled", True):
        for comparison in comparisons:
            pair = {}
            for case_name in (comparison["baseline"], comparison["variant"]):
                if case_name not in workload_cache:
                    artifact_path = artifact_dir / case_name / "ncu" / "whole_graph"
                    artifact_path.mkdir(parents=True, exist_ok=True)
                    print(
                        f"[SiDP Nsight] Compute whole-Graph case {case_name} ...",
                        flush=True,
                    )
                    workload_cache[case_name] = _collect_ncu_workload(
                        config,
                        case_config[case_name],
                        artifact_path,
                        manifest,
                    )
                pair[case_name] = workload_cache[case_name]
            base = pair[comparison["baseline"]]
            current = pair[comparison["variant"]]
            workload_runs.append(
                {
                    "baseline": comparison["baseline"],
                    "variant": comparison["variant"],
                    "baseline_launch_count": base["launch_count"],
                    "variant_launch_count": current["launch_count"],
                    "baseline_metrics": base["metrics"],
                    "variant_metrics": current["metrics"],
                    "metric_deltas": compare_ncu_metrics(base, current),
                    "baseline_unit_rollups": unit_rollup_dispersion(base),
                    "variant_unit_rollups": unit_rollup_dispersion(current),
                    "limitations": [*base["limitations"], *current["limitations"]],
                    "attribution": current.get("attribution"),
                    "raw_artifacts": {
                        "baseline": base["raw_artifacts"],
                        "variant": current["raw_artifacts"],
                    },
                }
            )

    node_by_comparison = {}
    for node_run in ncu_runs:
        node_by_comparison.setdefault(
            (node_run["baseline"], node_run["variant"]), node_run
        )
    workload_by_comparison = {
        (run["baseline"], run["variant"]): run for run in workload_runs
    }
    contention_analyses = []
    for comparison in comparisons:
        key = (comparison["baseline"], comparison["variant"])
        node = node_by_comparison.get(key)
        workload = workload_by_comparison.get(key)
        if node is None or workload is None:
            continue
        variant_nsys = cases[comparison["variant"]]["nsys"]
        devices = variant_nsys.get("devices", [])
        overlap_ratio = (
            statistics.median(
                item.get("compute_overlap_ratio", 0.0) for item in devices
            )
            if devices
            else 0.0
        )
        overlap_roles = {}
        role_names = {
            role
            for item in devices
            for role in item.get("compute_overlap_by_role_ratio", {})
        }
        for role in role_names:
            overlap_roles[role] = statistics.median(
                item.get("compute_overlap_by_role_ratio", {}).get(role, 0.0)
                for item in devices
            )
        sm_side_overlap_ratio = (
            statistics.median(
                item.get("compute_sm_side_overlap_ratio", 0.0)
                for item in devices
            )
            if devices
            else 0.0
        )
        semantics = variant_nsys.get("benchmark_semantics") or {}
        sm_trace = semantics.get("sm_execution_trace") or {}
        contention_analyses.append(
            {
                "baseline": comparison["baseline"],
                "variant": comparison["variant"],
                "kernel_name": node["kernel_name"],
                "findings": overlap_contention_hypotheses(
                    systems_mean_ratio=node.get("systems_mean_ratio"),
                    compute_overlap_ratio=overlap_ratio,
                    overlap_by_role_ratio=overlap_roles,
                    compute_sm_side_overlap_ratio=sm_side_overlap_ratio,
                    isolated_node_baseline={"metrics": node["baseline_metrics"]},
                    isolated_node_variant={"metrics": node["variant_metrics"]},
                    workload_baseline={"metrics": workload["baseline_metrics"]},
                    workload_variant={"metrics": workload["variant_metrics"]},
                    sm_trace_roles=sm_trace.get("roles", []),
                ),
                "evidence": {
                    "systems_compute_overlap_ratio_median": overlap_ratio,
                    "systems_compute_overlap_by_role_ratio_median": overlap_roles,
                    "systems_compute_sm_side_overlap_ratio_median": (
                        sm_side_overlap_ratio
                    ),
                    "sm_execution_trace": sm_trace,
                },
            }
        )

    manifest["environment"] = sorted(manifest["environment"])
    _write_json(output_dir / "manifest.json", manifest)
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "output_dir": str(output_dir),
        "artifact_dir": str(artifact_dir),
        "manifest": manifest,
        "cases": cases,
        "comparisons": comparisons,
        "ncu": {
            "enabled": ncu_enabled,
            "node_runs": ncu_runs,
            # Backward-compatible alias for the first version of the schema.
            "runs": ncu_runs,
            "workload_runs": workload_runs,
            "contention_analyses": contention_analyses,
        },
        "limitations": limitations,
    }
    _write_json(output_dir / "analysis.json", analysis)
    render_report(analysis, output_dir / "report.md")
    shutil.copyfile(Path(__file__).with_name("how_to_analyse.md"), output_dir / "how_to_analyse.md")
    print(f"[SiDP Nsight] report: {output_dir / 'report.md'}", flush=True)
    print(f"[SiDP Nsight] facts:  {output_dir / 'analysis.json'}", flush=True)
    print(f"[SiDP Nsight] raw profiler artifacts remain remote: {artifact_dir}", flush=True)
    return analysis


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Pipeline profiler JSON configuration")
    parser.add_argument("--output-dir", help="Override config output_dir")
    parser.add_argument("--artifact-dir", help="Override remote raw artifact_dir")
    parser.add_argument("--force", action="store_true", help="Replace these exact run directories")
    parser.add_argument("--no-ncu", action="store_true", help="Collect Systems only")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = str(Path(args.output_dir).resolve())
    if args.artifact_dir:
        config["artifact_dir"] = str(Path(args.artifact_dir).resolve())
    run(config, force=args.force, disable_ncu=args.no_ncu)


if __name__ == "__main__":
    main()
