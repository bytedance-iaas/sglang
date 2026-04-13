from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestRecord:
    request_id: str
    ttft_ms: float
    success: bool
    status: str
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class CacheMetrics:
    gpu_hit_count: int = 0
    gpu_hit_ratio: float = 0.0
    host_hit_count: int = 0
    host_hit_ratio: float = 0.0
    remote_hit_count: int = 0
    remote_hit_ratio: float = 0.0
    file_or_disk_hit_count: int = 0
    eviction_count: int = 0
    remote_load_bytes: int = 0
    remote_writeback_bytes: int = 0
    remote_load_latency_p50_ms: float = 0.0
    remote_load_latency_p95_ms: float = 0.0
    remote_load_latency_p99_ms: float = 0.0
    writeback_latency_p50_ms: float = 0.0
    writeback_latency_p95_ms: float = 0.0
    writeback_latency_p99_ms: float = 0.0


@dataclass
class BackendMetrics:
    transfer_latency_p50_ms: float = 0.0
    transfer_latency_p95_ms: float = 0.0
    transfer_latency_p99_ms: float = 0.0
    transfer_throughput_mb_s: float = 0.0
    connection_errors: int = 0
    retries: int = 0
    transport_failures: int = 0


@dataclass
class RunArtifacts:
    requests: list[RequestRecord] = field(default_factory=list)
    cache: CacheMetrics = field(default_factory=CacheMetrics)
    backend: BackendMetrics = field(default_factory=BackendMetrics)
    metadata: dict[str, Any] = field(default_factory=dict)
