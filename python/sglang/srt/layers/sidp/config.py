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
