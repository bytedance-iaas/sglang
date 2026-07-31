from __future__ import annotations

import dataclasses
import logging
import threading
import time
from enum import IntEnum
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


class CompletionKind(IntEnum):
    WRITE = 0
    LOAD = 1


@dataclasses.dataclass(frozen=True)
class CompletionTargets:
    write_prepare: int = 0
    load_prepare: int = 0
    write_commit: int = 0
    load_commit: int = 0


@dataclasses.dataclass
class _KindState:
    observed: int = 0
    ready: int = 0
    prepared: int = 0
    committed: int = 0
    prepared_digest: int = 0


class _StateField(IntEnum):
    PROTOCOL_VERSION = 0
    EPOCH = 1
    HEARTBEAT = 2
    FATAL_ERROR = 3
    CLOSE_REQUESTED = 4
    WRITE_OBSERVED = 5
    WRITE_READY = 6
    WRITE_PREPARED = 7
    WRITE_COMMITTED = 8
    WRITE_PREPARED_DIGEST = 9
    LOAD_OBSERVED = 10
    LOAD_READY = 11
    LOAD_PREPARED = 12
    LOAD_COMMITTED = 13
    LOAD_PREPARED_DIGEST = 14


_PROTOCOL_VERSION = 1
_STATE_WIDTH = len(_StateField)


class PPHiCacheCompletionCoordinator:
    """Exchange HiCache completion frontiers outside the scheduler hot path.

    The scheduler thread remains the only owner of CUDA event queries, ACK
    queues, radix nodes, and cache locks.  This coordinator only exchanges a
    fixed-size CPU snapshot on a dedicated Gloo process group.
    """

    def __init__(
        self,
        *,
        process_group: torch.distributed.ProcessGroup,
        interval_ms: float,
        stall_timeout_s: float,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        gather_fn: Optional[Callable[[list[torch.Tensor], torch.Tensor], None]] = None,
    ) -> None:
        if interval_ms <= 0:
            raise ValueError(
                f"HiCache PP progress interval must be positive, got {interval_ms}"
            )
        if stall_timeout_s <= 0:
            raise ValueError(
                f"HiCache PP stall timeout must be positive, got {stall_timeout_s}"
            )

        self.process_group = process_group
        self.interval_s = interval_ms / 1000.0
        self.stall_timeout_s = stall_timeout_s
        self.rank = (
            torch.distributed.get_rank(group=process_group) if rank is None else rank
        )
        self.world_size = (
            torch.distributed.get_world_size(group=process_group)
            if world_size is None
            else world_size
        )
        self._gather_fn = gather_fn or self._distributed_gather

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._close_requested = False
        self._epoch = 0
        self._heartbeat = 0
        self._fatal_error: Optional[str] = None
        self._fatal_code = 0
        self._states = {
            CompletionKind.WRITE: _KindState(),
            CompletionKind.LOAD: _KindState(),
        }
        self._targets = CompletionTargets()
        self._last_progress_time = time.monotonic()
        self._last_global_state = (0,) * 8
        self._epoch_mismatch_since: Optional[float] = None
        self._progress_rounds = 0
        self._progress_time_s = 0.0
        self._local_snapshot = torch.empty(
            (_STATE_WIDTH,), dtype=torch.int64, device="cpu"
        )
        self._gathered_snapshots = [
            torch.empty((_STATE_WIDTH,), dtype=torch.int64, device="cpu")
            for _ in range(self.world_size)
        ]

    def _distributed_gather(
        self, outputs: list[torch.Tensor], local: torch.Tensor
    ) -> None:
        torch.distributed.all_gather(outputs, local, group=self.process_group)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="hicache-pp-completion",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def close(self, timeout_s: Optional[float] = None) -> None:
        with self._lock:
            self._close_requested = True
            thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout_s or self.stall_timeout_s)
        if thread.is_alive():
            self.report_scheduler_fatal(
                "Timed out stopping HiCache PP completion coordinator: "
                f"rank={self.rank}"
            )
            raise RuntimeError(
                "Timed out stopping HiCache PP completion coordinator; every "
                "rank in the dedicated completion group must stop together"
            )
        with self._lock:
            self._thread = None

    def is_alive(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def publish_local(
        self,
        kind: CompletionKind,
        *,
        observed: int,
        ready: int,
        prepared: int,
        committed: int,
        prepared_digest: int,
    ) -> None:
        if not 0 <= committed <= prepared <= ready <= observed:
            raise RuntimeError(
                "Invalid HiCache PP completion frontier: "
                f"rank={self.rank}, kind={kind.name.lower()}, "
                f"observed={observed}, ready={ready}, prepared={prepared}, "
                f"committed={committed}"
            )
        with self._lock:
            state = self._states[kind]
            if (
                observed < state.observed
                or ready < state.ready
                or prepared < state.prepared
                or committed < state.committed
            ):
                raise RuntimeError(
                    "HiCache PP completion frontier regressed: "
                    f"rank={self.rank}, kind={kind.name.lower()}, old={state}, "
                    f"new=({observed}, {ready}, {prepared}, {committed})"
                )
            state.observed = observed
            state.ready = ready
            state.prepared = prepared
            state.committed = committed
            state.prepared_digest = prepared_digest

    def targets(self) -> CompletionTargets:
        with self._lock:
            if self._fatal_error is not None:
                raise RuntimeError(self._fatal_error)
            return self._targets

    def report_scheduler_fatal(self, message: str) -> None:
        self._set_fatal(message, code=8)

    def reset_epoch(self) -> None:
        with self._lock:
            for kind, state in self._states.items():
                if not (
                    state.observed
                    == state.ready
                    == state.prepared
                    == state.committed
                ):
                    raise RuntimeError(
                        "Cannot reset HiCache PP completion epoch with "
                        f"outstanding {kind.name.lower()} state on rank "
                        f"{self.rank}: {state}"
                    )
            self._epoch += 1
            self._heartbeat = 0
            self._states = {
                CompletionKind.WRITE: _KindState(),
                CompletionKind.LOAD: _KindState(),
            }
            self._targets = CompletionTargets()
            self._last_global_state = (0,) * 8
            self._last_progress_time = time.monotonic()
            self._epoch_mismatch_since = None

    def metrics_snapshot(self) -> dict[str, int | float]:
        with self._lock:
            write = self._states[CompletionKind.WRITE]
            load = self._states[CompletionKind.LOAD]
            return {
                "progress_rounds": self._progress_rounds,
                "progress_time_s": self._progress_time_s,
                "fatal_error": int(self._fatal_error is not None),
                "write_observed": write.observed,
                "write_ready": write.ready,
                "write_prepared": write.prepared,
                "write_committed": write.committed,
                "load_observed": load.observed,
                "load_ready": load.ready,
                "load_prepared": load.prepared,
                "load_committed": load.committed,
            }

    def _snapshot_tensor(self) -> torch.Tensor:
        with self._lock:
            self._heartbeat += 1
            write = self._states[CompletionKind.WRITE]
            load = self._states[CompletionKind.LOAD]
            values = (
                _PROTOCOL_VERSION,
                self._epoch,
                self._heartbeat,
                self._fatal_code,
                int(self._close_requested),
                write.observed,
                write.ready,
                write.prepared,
                write.committed,
                write.prepared_digest,
                load.observed,
                load.ready,
                load.prepared,
                load.committed,
                load.prepared_digest,
            )
            for index, value in enumerate(values):
                self._local_snapshot[index] = value
            return self._local_snapshot

    def _set_fatal(self, message: str, *, code: int) -> None:
        with self._lock:
            if self._fatal_error is None:
                self._fatal_error = message
                self._fatal_code = code
        logger.error(message)

    def _validate_protocol(self, rows: list[list[int]]) -> bool:
        versions = {row[_StateField.PROTOCOL_VERSION] for row in rows}
        if versions != {_PROTOCOL_VERSION}:
            self._set_fatal(
                "HiCache PP completion protocol version mismatch: "
                f"rank={self.rank}, versions={sorted(versions)}",
                code=2,
            )
            return False

        epochs = {row[_StateField.EPOCH] for row in rows}
        with self._lock:
            local_epoch = self._epoch
        if len(epochs) != 1 or epochs != {local_epoch}:
            now = time.monotonic()
            if self._epoch_mismatch_since is None:
                self._epoch_mismatch_since = now
            elif now - self._epoch_mismatch_since >= self.stall_timeout_s:
                self._set_fatal(
                    "HiCache PP completion epoch mismatch: "
                    f"rank={self.rank}, local_epoch={local_epoch}, "
                    f"epochs={sorted(epochs)}",
                    code=3,
                )
            return False
        self._epoch_mismatch_since = None

        fatal_ranks = [
            rank
            for rank, row in enumerate(rows)
            if row[_StateField.FATAL_ERROR] != 0
        ]
        if fatal_ranks:
            self._set_fatal(
                "A peer reported a fatal HiCache PP completion error: "
                f"rank={self.rank}, peers={fatal_ranks}",
                code=4,
            )
            return False
        return True

    @staticmethod
    def _all_close_requested(rows: list[list[int]]) -> bool:
        return all(row[_StateField.CLOSE_REQUESTED] != 0 for row in rows)

    def _process_rows(self, rows: list[list[int]]) -> None:
        if not self._validate_protocol(rows):
            return

        write_observed = min(row[_StateField.WRITE_OBSERVED] for row in rows)
        load_observed = min(row[_StateField.LOAD_OBSERVED] for row in rows)
        write_ready = min(row[_StateField.WRITE_READY] for row in rows)
        load_ready = min(row[_StateField.LOAD_READY] for row in rows)
        write_prepared = [row[_StateField.WRITE_PREPARED] for row in rows]
        load_prepared = [row[_StateField.LOAD_PREPARED] for row in rows]
        write_committed = min(row[_StateField.WRITE_COMMITTED] for row in rows)
        load_committed = min(row[_StateField.LOAD_COMMITTED] for row in rows)

        def commit_target(
            prepared: list[int],
            *,
            digest_field: _StateField,
            current: int,
            kind: CompletionKind,
        ) -> int:
            if len(set(prepared)) != 1 or prepared[0] <= current:
                return current
            digests = {row[digest_field] for row in rows}
            if len(digests) != 1:
                self._set_fatal(
                    "HiCache PP completion fingerprint diverged: "
                    f"rank={self.rank}, kind={kind.name.lower()}, "
                    f"frontier={prepared[0]}, digests={sorted(digests)}",
                    code=5,
                )
                return current
            return prepared[0]

        with self._lock:
            old = self._targets
            self._targets = CompletionTargets(
                write_prepare=max(old.write_prepare, write_ready),
                load_prepare=max(old.load_prepare, load_ready),
                write_commit=commit_target(
                    write_prepared,
                    digest_field=_StateField.WRITE_PREPARED_DIGEST,
                    current=old.write_commit,
                    kind=CompletionKind.WRITE,
                ),
                load_commit=commit_target(
                    load_prepared,
                    digest_field=_StateField.LOAD_PREPARED_DIGEST,
                    current=old.load_commit,
                    kind=CompletionKind.LOAD,
                ),
            )

        global_state = (
            write_observed,
            write_ready,
            min(write_prepared),
            write_committed,
            load_observed,
            load_ready,
            min(load_prepared),
            load_committed,
        )
        now = time.monotonic()
        if global_state != self._last_global_state:
            self._last_global_state = global_state
            self._last_progress_time = now

        outstanding = (
            max(row[_StateField.WRITE_OBSERVED] for row in rows) > write_committed
            or max(row[_StateField.LOAD_OBSERVED] for row in rows) > load_committed
        )
        if outstanding and now - self._last_progress_time >= self.stall_timeout_s:
            self._set_fatal(
                "HiCache PP completion frontier stalled: "
                f"rank={self.rank}, write_observed={write_observed}, "
                f"write_ready={write_ready}, "
                f"write_prepared={write_prepared}, "
                f"write_committed={write_committed}, load_ready={load_ready}, "
                f"load_observed={load_observed}, load_prepared={load_prepared}, "
                f"load_committed={load_committed}",
                code=6,
            )

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                started = time.perf_counter()
                local = self._snapshot_tensor()
                self._gather_fn(self._gathered_snapshots, local)
                rows = [tensor.tolist() for tensor in self._gathered_snapshots]
                self._process_rows(rows)
                elapsed = time.perf_counter() - started
                with self._lock:
                    self._progress_rounds += 1
                    self._progress_time_s += elapsed
                if self._all_close_requested(rows):
                    self._stop_event.set()
                    break
                self._stop_event.wait(self.interval_s)
        except Exception as exc:
            self._set_fatal(
                "HiCache PP completion progress thread failed: "
                f"rank={self.rank}, error={exc!r}",
                code=7,
            )
            self._stop_event.set()
