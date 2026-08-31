from dataclasses import dataclass
from enum import Enum


class SidpPrefetchPolicy(str, Enum):
    """How one SiDP cycle orders its remote layer fills."""

    COMPUTE = "compute"
    STATIC_PEAK = "static_peak"
    DYNAMIC_OWNER = "dynamic_owner"


class SidpCopyBackend(str, Enum):
    """Data-movement implementation used by the cycle pipeline."""

    DMA = "dma"
    SM = "sm"


@dataclass
class SidpConfig:
    dp_size: int
    dp_rank: int
    k: int = 1
    cache_cycles: int = 2
    rdzv_port: int = 0
    rdzv_host: str = "127.0.0.1"
    num_layers: int = 0
    transfer_dtype: str = "same"
    enable_cycle_overlap: bool = False
    prefetch_policy: str = SidpPrefetchPolicy.COMPUTE.value
    copy_backend: str = SidpCopyBackend.DMA.value
    # Compatibility input for out-of-tree callers. Runtime code uses the
    # resolved ``prefetch_policy`` above.
    enable_peak_shifting: bool = False
    enable_debug_logging: bool = False
    enable_graph_profiling: bool = False
    profile_dummy_compute: bool = False
    peak_sync_strategy: str = "none"
    peak_sync_min_raw_bs: int = 64
    peak_sync_max_replays: int = 0
    peak_sync_timeout_s: float = 30.0
    profile_sample_interval: int = 20
    profile_warmup_replays: int = 20
    profile_output_dir: str = "/tmp/sidp_profile"
