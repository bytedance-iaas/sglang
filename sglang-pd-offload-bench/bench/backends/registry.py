from bench.backends.base import BackendAdapter
from bench.backends.mock_backends import EICOffloadBackend, GPUOnlyBackend, HiCacheMooncakeBackend


def create_backend(name: str) -> BackendAdapter:
    mapping = {
        "gpu_only": GPUOnlyBackend,
        "eic_offload": EICOffloadBackend,
        "hicache_mooncake": HiCacheMooncakeBackend,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported backend: {name}")
    return mapping[name]()
