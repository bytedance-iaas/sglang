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
    from sglang.srt.arg_groups.overrides import resolved_view

    cfg = resolved_view(server_args)
    if cfg.ep_dispatch_algorithm == "block":
        local_world = cfg.tp_size * cfg.pp_size // cfg.nnodes
        if cfg.ep_size > local_world:
            if cfg.block_moe_backend != "iaas":
                raise ValueError(
                    "Multi-node Block MoE requires --block-moe-backend iaas"
                )
            # Select NVSHMEM before any communicator allocates symmetric memory.
            from iaas_kernels.rdma._buffer import prepare_allocator

            prepare_allocator()
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
    from sglang.srt.runtime_context import get_exec

    if name == "megamoe" and get_exec().moe.ep_dispatch_algorithm == "block":
        # The opt-in block adapter manages mutable physical slots and uses the
        # native FP8 preparation path. Provider-owned packing is incompatible.
        if capability != (9, 0):
            raise ValueError("Block MoE execution currently requires SM90 FP8")
        return None
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
