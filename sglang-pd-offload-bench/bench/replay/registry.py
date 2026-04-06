from bench.replay.adapters import BurstTraceReplay, MultiTurnReplay, SharedContextReplay
from bench.replay.base import ReplayAdapter


def create_replay(name: str) -> ReplayAdapter:
    mapping = {
        "shared_context_replay": SharedContextReplay,
        "multiturn_replay": MultiTurnReplay,
        "burst_trace_replay": BurstTraceReplay,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported replay adapter: {name}")
    return mapping[name]()
