from dataclasses import dataclass


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
    enable_peak_shifting: bool = False
    enable_debug_logging: bool = False
    enable_graph_profiling: bool = False
    profile_dummy_compute: bool = False
    peak_sync_strategy: str = "force_sync"
    peak_sync_min_raw_bs: int = 64
    peak_sync_max_replays: int = 0
    peak_sync_timeout_s: float = 30.0
    profile_sample_interval: int = 20
    profile_warmup_replays: int = 20
    profile_output_dir: str = "/tmp/sidp_profile"
