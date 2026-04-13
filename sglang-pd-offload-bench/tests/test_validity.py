from bench.analysis.validity import classify_experiment_validity
from bench.types import CacheMetrics


def test_gpu_fit_label() -> None:
    c = CacheMetrics(gpu_hit_ratio=0.99, eviction_count=0, remote_load_bytes=0)
    assert classify_experiment_validity(c) == "LIKELY_GPU_FIT"


def test_valid_offload_label() -> None:
    c = CacheMetrics(remote_hit_count=3, eviction_count=1, remote_load_bytes=1024)
    assert classify_experiment_validity(c) == "VALID_OFFLOAD_TEST"
