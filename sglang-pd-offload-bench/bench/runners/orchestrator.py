from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from bench.analysis.validity import classify_experiment_validity
from bench.backends.registry import create_backend
from bench.collectors.metrics import request_dataframe, summarize_request_metrics
from bench.replay.registry import create_replay
from bench.types import BackendMetrics, CacheMetrics


def _mock_cache_metrics(backend_name: str, mode: str) -> CacheMetrics:
    if backend_name == "gpu_only":
        return CacheMetrics(gpu_hit_count=100, gpu_hit_ratio=0.995, eviction_count=0)
    if mode == "prepopulated_remote_cache":
        return CacheMetrics(
            gpu_hit_count=65,
            gpu_hit_ratio=0.65,
            host_hit_count=10,
            host_hit_ratio=0.1,
            remote_hit_count=25,
            remote_hit_ratio=0.25,
            eviction_count=50,
            remote_load_bytes=500_000_000,
            remote_writeback_bytes=100_000_000,
            remote_load_latency_p50_ms=2.1,
            remote_load_latency_p95_ms=6.2,
            remote_load_latency_p99_ms=11.5,
        )
    return CacheMetrics(
        gpu_hit_count=78,
        gpu_hit_ratio=0.78,
        host_hit_count=8,
        host_hit_ratio=0.08,
        remote_hit_count=14,
        remote_hit_ratio=0.14,
        eviction_count=20,
        remote_load_bytes=220_000_000,
        remote_writeback_bytes=70_000_000,
    )


def _mock_backend_metrics(backend_name: str) -> BackendMetrics:
    if backend_name == "gpu_only":
        return BackendMetrics()
    if backend_name == "hicache_mooncake":
        return BackendMetrics(transfer_latency_p50_ms=1.9, transfer_latency_p95_ms=4.7, transfer_throughput_mb_s=820)
    return BackendMetrics(transfer_latency_p50_ms=1.5, transfer_latency_p95_ms=4.2, transfer_throughput_mb_s=910)


def run_one(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    backend_name = config["backend"]
    replay_name = config["replay"]
    cache_mode = config.get("cache_mode", "online_built_cache")

    backend = create_backend(backend_name)
    replay = create_replay(replay_name)

    backend.start(config)
    assert backend.healthcheck(), "backend healthcheck failed"
    replay.prepare(config)

    records = replay.replay(backend_name=backend_name)
    df = request_dataframe(records)
    request_summary = summarize_request_metrics(df)

    cache = _mock_cache_metrics(backend_name, cache_mode)
    backend_metrics = _mock_backend_metrics(backend_name)
    validity = classify_experiment_validity(cache)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "request_level.csv", index=False)

    summary = {
        "backend": backend_name,
        "replay": replay_name,
        "cache_mode": cache_mode,
        "request_metrics": request_summary,
        "cache_metrics": asdict(cache),
        "backend_metrics": asdict(backend_metrics),
        "validity": validity,
        "manifest": replay.export_manifest(),
        "runtime": backend.describe_runtime(),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    backend.stop()
    return summary


def aggregate_runs(run_dirs: list[Path], output_file: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "run": run_dir.name,
                "backend": summary["backend"],
                "replay": summary["replay"],
                "cache_mode": summary["cache_mode"],
                "validity": summary["validity"],
                "ttft_p95_ms": summary["request_metrics"]["ttft_p95_ms"],
                "success_rate": summary["request_metrics"]["success_rate"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_file, index=False)
    return out
