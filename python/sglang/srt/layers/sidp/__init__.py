"""SiDP (Shared-weight Intra-node Data Parallelism): Dense-FFN weight sharing across DP ranks via CUDA IPC + NVLink prefetch."""

from sglang.srt.layers.sidp.config import SidpConfig
from sglang.srt.layers.sidp.sidp_manager import SidpManager

_GLOBAL_SIDP_MANAGER = None


def get_global_sidp_manager():
    return _GLOBAL_SIDP_MANAGER


def set_global_sidp_manager(manager):
    global _GLOBAL_SIDP_MANAGER
    _GLOBAL_SIDP_MANAGER = manager


__all__ = [
    "SidpManager",
    "SidpConfig",
    "get_global_sidp_manager",
    "set_global_sidp_manager",
]
