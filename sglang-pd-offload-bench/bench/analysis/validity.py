from __future__ import annotations

from bench.types import CacheMetrics


def classify_experiment_validity(cache: CacheMetrics, ttft_p95_gap_ms: float = 0.0) -> str:
    if cache.remote_hit_count > 0 or cache.remote_load_bytes > 0 or cache.remote_writeback_bytes > 0:
        if cache.eviction_count > 0:
            return "VALID_OFFLOAD_TEST"
        return "NO_REMOTE_PARTICIPATION_OBSERVED"

    gpu_fit = cache.gpu_hit_ratio >= 0.98 and cache.eviction_count == 0 and cache.remote_load_bytes == 0
    if gpu_fit and ttft_p95_gap_ms <= 5.0:
        return "LIKELY_GPU_FIT"

    return "INSUFFICIENT_REUSE_PRESSURE"
