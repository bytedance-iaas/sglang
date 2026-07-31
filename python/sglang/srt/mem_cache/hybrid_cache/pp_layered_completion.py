from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import torch

from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_WRITE = 0
_LOAD = 1
_NUM_KINDS = 2


class _BackgroundFrontier:
    """Exchange ready/prepared ACK frontiers outside the scheduler thread."""

    def __init__(
        self,
        *,
        process_group: torch.distributed.ProcessGroup,
        interval_ms: float,
    ) -> None:
        if interval_ms <= 0:
            raise ValueError(
                f"HiCache PP progress interval must be positive, got {interval_ms}"
            )

        self.process_group = process_group
        self.interval_s = interval_ms / 1000.0
        self.world_size = torch.distributed.get_world_size(group=process_group)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._local = torch.zeros(6, dtype=torch.int64, device="cpu")
        self._gathered = [torch.empty_like(self._local) for _ in range(self.world_size)]
        self._prepare = [0, 0]
        self._commit = [0, 0]
        self._thread = threading.Thread(
            target=self._run,
            name="hicache-pp-flags",
            daemon=True,
        )
        self._thread.start()

    def publish(
        self,
        ready: list[int],
        prepared: list[int],
        committed: list[int],
    ) -> None:
        if any(
            committed[kind] > prepared[kind] or prepared[kind] > ready[kind]
            for kind in range(_NUM_KINDS)
        ):
            raise RuntimeError(
                "Invalid HiCache PP completion frontier: "
                f"ready={ready}, prepared={prepared}, committed={committed}"
            )
        with self._lock:
            self._raise_if_failed()
            self._local[0] = ready[_WRITE]
            self._local[1] = ready[_LOAD]
            self._local[2] = prepared[_WRITE]
            self._local[3] = prepared[_LOAD]
            self._local[4] = committed[_WRITE]
            self._local[5] = committed[_LOAD]

    def targets(self) -> tuple[tuple[int, int], tuple[int, int]]:
        with self._lock:
            self._raise_if_failed()
            return tuple(self._prepare), tuple(self._commit)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("HiCache PP completion thread failed") from self._error

    def _apply_rows(self, rows: list[list[int]]) -> None:
        with self._lock:
            for kind in range(_NUM_KINDS):
                prepared = [row[2 + kind] for row in rows]
                committed = [row[4 + kind] for row in rows]
                if len(set(committed)) == 1 and prepared == committed:
                    self._prepare[kind] = max(
                        self._prepare[kind],
                        min(row[kind] for row in rows),
                    )
                if len(set(prepared)) == 1 and prepared[0] > min(committed):
                    self._commit[kind] = max(self._commit[kind], prepared[0])

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                with self._lock:
                    local = self._local.clone()
                torch.distributed.all_gather(
                    self._gathered,
                    local,
                    group=self.process_group,
                )
                self._apply_rows([tensor.tolist() for tensor in self._gathered])
                self._stop.wait(self.interval_s)
        except BaseException as exc:
            with self._lock:
                self._error = exc
            logger.exception("HiCache PP completion thread failed")


class PPHiCacheLayeredCompletion:
    """Completion-only layered flags for DSV4 HiCache L2."""

    def __init__(
        self,
        *,
        cache: UnifiedRadixCache,
        server_args: ServerArgs,
    ) -> None:
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        kvcache = cache.token_to_kv_pool_allocator.get_kvcache()
        unsupported: list[str] = []
        if not isinstance(kvcache, DeepSeekV4TokenToKVPool):
            unsupported.append(f"cache={type(kvcache).__name__}, expected DSV4")
        if cache.pp_size <= 1:
            unsupported.append(f"pp_size={cache.pp_size}, expected > 1")
        if server_args.hicache_storage_backend is not None:
            unsupported.append("L3 storage is enabled")
        if server_args.hicache_write_policy != "write_through":
            unsupported.append(
                f"write_policy={server_args.hicache_write_policy!r}, "
                "expected 'write_through'"
            )
        if server_args.enable_eic_cache:
            unsupported.append("EIC is enabled")
        if server_args.enable_dp_attention:
            unsupported.append("DP attention is enabled")
        if unsupported:
            raise ValueError(
                "SGLANG_HICACHE_PP_SYNC_MODE=layered_flags only supports "
                "DeepSeek-V4 UnifiedRadixCache PP L2 write-through: "
                + "; ".join(unsupported)
            )
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "layered_flags requires an initialized distributed process group"
            )

        world_ranks = list(get_world_group().ranks)
        expected_world_size = server_args.tp_size * server_args.pp_size
        if len(world_ranks) != expected_world_size:
            raise RuntimeError(
                "Cannot create HiCache PP completion group: "
                f"world_size={len(world_ranks)}, expected={expected_world_size}"
            )

        self.cache = cache
        self.ready = [0, 0]
        self.prepared = [0, 0]
        self.committed = [0, 0]
        process_group = torch.distributed.new_group(
            ranks=world_ranks,
            backend="gloo",
        )
        self.frontier = _BackgroundFrontier(
            process_group=process_group,
            interval_ms=envs.SGLANG_HICACHE_PP_PROGRESS_INTERVAL_MS.get(),
        )
        logger.info(
            "Enabled minimal layered-flags HiCache PP sync: ranks=%s, "
            "interval_ms=%s",
            world_ranks,
            envs.SGLANG_HICACHE_PP_PROGRESS_INTERVAL_MS.get(),
        )

    def _queue(self, kind: int):
        controller = self.cache.cache_controller
        return (
            controller.ack_write_queue
            if kind == _WRITE
            else controller.ack_load_queue
        )

    def _ongoing(self, kind: int):
        return (
            self.cache.ongoing_write_through
            if kind == _WRITE
            else self.cache.ongoing_load_back
        )

    def _poll_ready(self, kind: int) -> None:
        ready = self.committed[kind]
        for _, finish_event, _ in self._queue(kind):
            if not finish_event.query():
                break
            ready += 1
        if ready < self.ready[kind]:
            raise RuntimeError(
                "HiCache PP ready frontier regressed: "
                f"kind={kind}, old={self.ready[kind]}, new={ready}"
            )
        self.ready[kind] = ready

    def _prepare(self, kind: int, target: int) -> None:
        if target <= self.prepared[kind]:
            return
        pending = target - self.committed[kind]
        queue = self._queue(kind)
        if target > self.ready[kind] or pending > len(queue):
            raise RuntimeError(
                "HiCache PP prepare frontier exceeds local ACKs: "
                f"kind={kind}, target={target}, ready={self.ready[kind]}, "
                f"committed={self.committed[kind]}, queue={len(queue)}"
            )
        if any(not ack[1].query() for ack in queue[:pending]):
            raise RuntimeError(
                f"HiCache PP prepare frontier includes unfinished ACK: kind={kind}"
            )
        self.prepared[kind] = target

    def _consume(self, kind: int, target: int) -> None:
        if target <= self.committed[kind]:
            return
        if target > self.prepared[kind]:
            raise RuntimeError(
                "HiCache PP commit frontier exceeds prepared frontier: "
                f"kind={kind}, target={target}, prepared={self.prepared[kind]}"
            )

        queue = self._queue(kind)
        while self.committed[kind] < target:
            if not queue:
                raise RuntimeError(
                    f"HiCache PP ACK queue exhausted: kind={kind}, target={target}"
                )
            _, finish_event, ack_ids = queue[0]
            if not finish_event.query():
                raise RuntimeError(
                    f"HiCache PP commit reached unfinished ACK: kind={kind}"
                )
            queue.pop(0)
            ongoing = self._ongoing(kind)
            for ack_id in ack_ids:
                if ack_id not in ongoing:
                    raise RuntimeError(
                        "HiCache PP ACK is absent from ongoing operations: "
                        f"kind={kind}, ack_id={ack_id}"
                    )
                entry = ongoing.pop(ack_id)
                if kind == _WRITE:
                    node, lock_params = entry
                    self.cache.dec_lock_ref(node, lock_params)
                else:
                    node, lock_params, host_lock_params = entry
                    self.cache.dec_lock_ref(node, lock_params)
                    self.cache.dec_host_lock_ref(node, host_lock_params)
            self.committed[kind] += 1

    def _check(self, kinds: tuple[int, ...]) -> None:
        for kind in kinds:
            self._poll_ready(kind)
        self.frontier.publish(self.ready, self.prepared, self.committed)

        prepare, _ = self.frontier.targets()
        for kind in kinds:
            self._prepare(kind, prepare[kind])
        self.frontier.publish(self.ready, self.prepared, self.committed)

        _, commit = self.frontier.targets()
        for kind in kinds:
            self._consume(kind, commit[kind])
        self.frontier.publish(self.ready, self.prepared, self.committed)

    def check_write(self) -> None:
        self._check((_WRITE,))

    def check_load(self) -> None:
        self._check((_LOAD,))

    def check_all(self) -> None:
        self._check((_WRITE, _LOAD))
