"""Fused early-MoE API. Configuration import does not initialize CUDA."""

from .config import Geometry

_RUNTIME = {
    "ActiveWeights",
    "EarlyMoEContext",
    "EarlyMoEResult",
    "SharedScratch",
    "create_early_moe_context",
    "early_moe",
    "STAT_NAMES",
}
__all__ = ["Geometry", *sorted(_RUNTIME)]


def __getattr__(name):
    if name in _RUNTIME:
        from . import runtime

        return getattr(runtime, name)
    raise AttributeError(name)
