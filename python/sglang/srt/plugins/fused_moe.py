"""Extension point for backends that own dispatch, expert compute and combine.

Factories run after checkpoint parameters are allocated. Providers validate the
layer, prepare its raw weights before quantization-specific repacking, and return
fully combined local token rows. Registration must not initialize CUDA.
"""

from typing import Any, Callable, Protocol

FUSED_MOE_API_VERSION = 1


class FusedMoEBackend(Protocol):
    def prepare_weights(self, layer: Any) -> None: ...

    def forward(self, layer: Any, hidden_states: Any, topk_output: Any) -> Any: ...


_factories: dict[tuple[str, tuple[int, int]], Callable] = {}
_worker_initializers: dict[str, Callable] = {}


def register_fused_moe_worker_initializer(name: str, initializer):
    previous = _worker_initializers.get(name)
    if previous is not None and previous is not initializer:
        raise ValueError(f"Fused MoE worker initializer already registered for {name}")
    _worker_initializers[name] = initializer


def prepare_fused_moe_worker(server_args):
    """Run before distributed communicators allocate symmetric memory."""
    from sglang.srt.plugins import load_plugins

    load_plugins()
    for initializer in _worker_initializers.values():
        initializer(server_args)


def register_fused_moe_backend(name: str, capability: tuple[int, int], factory):
    key = (name, capability)
    previous = _factories.get(key)
    if previous is not None and previous is not factory:
        raise ValueError(f"Fused MoE backend already registered for {key}")
    _factories[key] = factory


def create_fused_moe_backend(name: str, capability: tuple[int, int], layer):
    from sglang.srt.plugins import load_plugins

    load_plugins()
    factory = _factories.get((name, capability))
    if factory is not None:
        return factory(layer)
    if name == "megamoe" and capability == (9, 0):
        raise RuntimeError(
            "SM90 MegaMoE requires the iaas-kernels plugin. Install its wheel "
            "and include iaas_kernels in SGLANG_PLUGINS if a whitelist is set."
        )
    return None
