"""Extensible cross-rank CUDA Graph launch synchronization for SiDP peak-shifting."""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Protocol

import torch


class PeakSyncStrategy(Protocol):
    """Align a decode Graph launch and return JSON-serializable diagnostics."""

    name: str

    def before_launch(self, *, raw_batch_size: int, graph_batch_size: int) -> dict:
        ...


def _unsynchronized_result(
    *, raw_batch_size: int, graph_batch_size: int
) -> dict:
    now_ns = time.monotonic_ns()
    return {
        "arrival_ns": now_ns,
        "launch_ns": now_ns,
        "synchronized": False,
        "sync_index": None,
        "barrier_wait_ms": 0.0,
        "raw_batch_size": raw_batch_size,
        "graph_batch_size": graph_batch_size,
    }


class NoSyncStrategy:
    name = "none"

    def before_launch(self, *, raw_batch_size: int, graph_batch_size: int) -> dict:
        return _unsynchronized_result(
            raw_batch_size=raw_batch_size,
            graph_batch_size=graph_batch_size,
        )


class ForceSyncStrategy:
    """Experimental all-rank synchronization reference; not for production.

    The strategy establishes the common wave phase required by static
    peak-shifting, but it blindly couples every DP rank and is only validated
    for fixed, balanced mechanism experiments. A single-rank warmup or a
    dynamically idle rank can make its barrier time out. Keep this path as a
    reference until coordinated scheduling or dynamic shifting is implemented.
    """

    name = "force_sync"

    def __init__(
        self,
        *,
        store: torch.distributed.Store,
        dp_rank: int,
        dp_size: int,
        min_raw_bs: int,
        max_replays: int,
        timeout_s: float,
    ) -> None:
        self.store = store
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.min_raw_bs = min_raw_bs
        self.max_replays = max_replays
        self.timeout = timedelta(seconds=timeout_s)
        self.ready = False
        self.index = 0

    def _wait(self, keys: list[str], phase: str) -> None:
        try:
            self.store.wait(keys, self.timeout)
        except RuntimeError as exc:
            raise RuntimeError(
                f"SiDP force_sync timed out during {phase} at rank "
                f"{self.dp_rank}; the current strategy requires every DP rank "
                "to keep launching decode Graphs. force_sync is an experimental "
                "reference and must not be used in production"
            ) from exc

    def before_launch(self, *, raw_batch_size: int, graph_batch_size: int) -> dict:
        if self.max_replays > 0 and self.index >= self.max_replays:
            return _unsynchronized_result(
                raw_batch_size=raw_batch_size,
                graph_batch_size=graph_batch_size,
            )
        if not self.ready and raw_batch_size < self.min_raw_bs:
            return _unsynchronized_result(
                raw_batch_size=raw_batch_size,
                graph_batch_size=graph_batch_size,
            )

        # Align a GPU-ready boundary, not just CPU enqueue calls. The previous
        # Graph's tail join makes current-stream completion cover SiDP comm work.
        torch.cuda.current_stream().synchronize()
        arrival_ns = time.monotonic_ns()

        if not self.ready:
            ready_keys = [
                f"sidp/peak_sync/ready/{rank}" for rank in range(self.dp_size)
            ]
            self.store.set(ready_keys[self.dp_rank], "1")
            self._wait(ready_keys, "bulk readiness")
            self.ready = True

        sync_index = self.index
        step_keys = [
            f"sidp/peak_sync/step/{sync_index}/{rank}"
            for rank in range(self.dp_size)
        ]
        barrier_start_ns = time.monotonic_ns()
        self.store.set(step_keys[self.dp_rank], "1")
        self._wait(step_keys, f"launch {sync_index}")
        launch_ns = time.monotonic_ns()
        self.index += 1

        return {
            "arrival_ns": arrival_ns,
            "launch_ns": launch_ns,
            "synchronized": True,
            "sync_index": sync_index,
            "barrier_wait_ms": (launch_ns - barrier_start_ns) / 1_000_000,
            "raw_batch_size": raw_batch_size,
            "graph_batch_size": graph_batch_size,
        }


def build_peak_sync_strategy(
    name: str,
    *,
    enabled: bool,
    store: torch.distributed.Store,
    dp_rank: int,
    dp_size: int,
    min_raw_bs: int,
    max_replays: int,
    timeout_s: float,
) -> PeakSyncStrategy:
    if not enabled or name == "none":
        return NoSyncStrategy()
    if name == "force_sync":
        return ForceSyncStrategy(
            store=store,
            dp_rank=dp_rank,
            dp_size=dp_size,
            min_raw_bs=min_raw_bs,
            max_replays=max_replays,
            timeout_s=timeout_s,
        )
    raise ValueError(f"Unknown SiDP peak sync strategy: {name}")
