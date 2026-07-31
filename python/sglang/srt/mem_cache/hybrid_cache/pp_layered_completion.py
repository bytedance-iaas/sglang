from __future__ import annotations

import atexit
import hashlib
import logging
import time
from datetime import timedelta
from typing import TYPE_CHECKING

import torch

from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.environ import envs
from sglang.srt.mem_cache.hybrid_cache.pp_completion_coordinator import (
    CompletionKind,
    PPHiCacheCompletionCoordinator,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class PPHiCacheLayeredCompletion:
    """HiCache-only owner of the layered PP completion protocol."""

    def __init__(
        self, *, cache: UnifiedRadixCache, server_args: ServerArgs
    ) -> None:
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        self.cache = cache
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
        if unsupported:
            raise ValueError(
                "SGLANG_HICACHE_PP_SYNC_MODE=layered_flags only replaces "
                "DeepSeek-V4 UnifiedRadixCache PP completion synchronization "
                "for L2-only write-through: " + "; ".join(unsupported)
            )
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "layered_flags requires an initialized distributed process group"
            )

        self.observed = {kind: 0 for kind in CompletionKind}
        self.ready = {kind: 0 for kind in CompletionKind}
        self.prepared = {kind: 0 for kind in CompletionKind}
        self.committed = {kind: 0 for kind in CompletionKind}
        self.prepared_digest = {kind: 0 for kind in CompletionKind}
        self.committed_digest = {kind: 0 for kind in CompletionKind}
        self._shutdown = False

        stall_timeout_s = envs.SGLANG_HICACHE_PP_STALL_TIMEOUT_S.get()
        self.process_group = self._create_process_group(
            server_args=server_args,
            timeout_s=stall_timeout_s,
        )
        self.coordinator = PPHiCacheCompletionCoordinator(
            process_group=self.process_group,
            interval_ms=envs.SGLANG_HICACHE_PP_PROGRESS_INTERVAL_MS.get(),
            stall_timeout_s=stall_timeout_s,
        )
        self.coordinator.start()
        atexit.register(self.shutdown)
        logger.info(
            "Enabled completion-only layered-flags HiCache PP sync: "
            "rank=%s, group_size=%s, interval_ms=%s",
            torch.distributed.get_rank(group=self.process_group),
            torch.distributed.get_world_size(group=self.process_group),
            envs.SGLANG_HICACHE_PP_PROGRESS_INTERVAL_MS.get(),
        )

    @staticmethod
    def _create_process_group(
        *, server_args: ServerArgs, timeout_s: float
    ) -> torch.distributed.ProcessGroup:
        """Create one dedicated Gloo completion group per attention-DP replica."""

        world_ranks = list(get_world_group().ranks)
        expected_world_size = server_args.tp_size * server_args.pp_size
        if len(world_ranks) != expected_world_size:
            raise RuntimeError(
                "Cannot derive HiCache completion rank layout: "
                f"world_size={len(world_ranks)}, tp_size={server_args.tp_size}, "
                f"pp_size={server_args.pp_size}"
            )
        attn_dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
        if server_args.tp_size % attn_dp_size != 0:
            raise RuntimeError(
                "HiCache completion group requires tp_size divisible by "
                f"attention dp size: tp_size={server_args.tp_size}, "
                f"attn_dp_size={attn_dp_size}"
            )

        replica_width = server_args.tp_size // attn_dp_size
        current_rank = torch.distributed.get_rank()
        selected_group = None
        selected_ranks = None
        for attn_dp_rank in range(attn_dp_size):
            positions = [
                pp_rank * server_args.tp_size + attn_dp_rank * replica_width + lane
                for pp_rank in range(server_args.pp_size)
                for lane in range(replica_width)
            ]
            ranks = [world_ranks[position] for position in positions]
            group = torch.distributed.new_group(
                ranks=ranks,
                backend="gloo",
                timeout=timedelta(seconds=timeout_s),
            )
            if current_rank in ranks:
                selected_group = group
                selected_ranks = ranks
        if selected_group is None:
            raise RuntimeError(
                "Current rank is absent from every HiCache completion group: "
                f"rank={current_rank}, world_ranks={world_ranks}"
            )
        logger.info(
            "Created HiCache completion group: rank=%s, ranks=%s",
            current_rank,
            selected_ranks,
        )
        return selected_group

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self.coordinator.close(
            timeout_s=envs.SGLANG_HICACHE_PP_STALL_TIMEOUT_S.get()
        )
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group(self.process_group)
        self._shutdown = True

    def before_reset(self) -> None:
        self.quiesce()

    def after_reset(self) -> None:
        for kind in CompletionKind:
            self.observed[kind] = 0
            self.ready[kind] = 0
            self.prepared[kind] = 0
            self.committed[kind] = 0
            self.prepared_digest[kind] = 0
            self.committed_digest[kind] = 0
        self.coordinator.reset_epoch()

    @staticmethod
    def _extend_digest(
        digest: int, sequence: int, operation_fingerprint: int
    ) -> int:
        hasher = hashlib.blake2b(digest_size=8)
        hasher.update(int(digest).to_bytes(8, byteorder="little", signed=False))
        hasher.update(int(sequence).to_bytes(8, byteorder="little", signed=False))
        hasher.update(
            int(operation_fingerprint).to_bytes(
                8, byteorder="little", signed=False
            )
        )
        return int.from_bytes(hasher.digest(), byteorder="little") & ((1 << 63) - 1)

    @staticmethod
    def _node_fingerprint(node: UnifiedTreeNode) -> int:
        hasher = hashlib.blake2b(digest_size=8)
        key = node.key
        if key is not None:
            hasher.update(len(key).to_bytes(8, byteorder="little", signed=False))
            hasher.update(str(key.extra_key).encode("utf-8"))
            hasher.update(bytes([int(key.is_bigram)]))
        hashes = node.hash_value or []
        if hashes:
            encoded = hashes[-1].encode("utf-8")
            hasher.update(len(encoded).to_bytes(4, byteorder="little"))
            hasher.update(encoded)
        elif key is not None:
            for token_id in key.token_ids:
                hasher.update(int(token_id).to_bytes(8, "little", signed=True))
        return int.from_bytes(hasher.digest(), byteorder="little") & ((1 << 63) - 1)

    def _queue(self, kind: CompletionKind):
        controller = self.cache.cache_controller
        if kind == CompletionKind.WRITE:
            return controller.ack_write_queue
        return controller.ack_load_queue

    def _ongoing(self, kind: CompletionKind):
        if kind == CompletionKind.WRITE:
            return self.cache.ongoing_write_through
        return self.cache.ongoing_load_back

    def _fatal(self, kind: CompletionKind, message: str) -> None:
        detail = (
            "HiCache layered completion divergence: "
            f"kind={kind.name.lower()}, pp_rank={self.cache.pp_rank}, "
            f"queue={len(self._queue(kind))}, "
            f"ongoing={len(self._ongoing(kind))}, "
            f"observed={self.observed[kind]}, ready={self.ready[kind]}, "
            f"prepared={self.prepared[kind]}, "
            f"committed={self.committed[kind]}: {message}"
        )
        self.coordinator.report_scheduler_fatal(detail)
        raise RuntimeError(detail)

    def _extend_ack_digest(
        self,
        kind: CompletionKind,
        *,
        digest: int,
        frontier: int,
        ack_ids: list[int],
    ) -> tuple[int, int]:
        ongoing = self._ongoing(kind)
        if not ack_ids:
            self._fatal(kind, "encountered an ACK without operations")
        for ack_id in ack_ids:
            entry = ongoing.get(ack_id)
            if entry is None:
                self._fatal(
                    kind,
                    f"ack_id={ack_id} is absent from the ongoing operation map",
                )
            frontier += 1
            digest = self._extend_digest(
                digest,
                frontier,
                self._node_fingerprint(entry[0]),
            )
        return frontier, digest

    def _poll_ready(self, kind: CompletionKind) -> None:
        frontier = self.committed[kind]
        digest = self.committed_digest[kind]
        observed = frontier
        ready = True
        for _, finish_event, ack_ids in self._queue(kind):
            observed += len(ack_ids)
            if not ready:
                continue
            if not finish_event.query():
                ready = False
                continue
            frontier, digest = self._extend_ack_digest(
                kind,
                digest=digest,
                frontier=frontier,
                ack_ids=ack_ids,
            )
        if observed < self.observed[kind]:
            self._fatal(kind, f"local observed frontier regressed to {observed}")
        if frontier < self.ready[kind]:
            self._fatal(kind, f"local ready frontier regressed to {frontier}")
        self.observed[kind] = observed
        self.ready[kind] = frontier

    def _prepare(self, kind: CompletionKind, target: int) -> None:
        if target <= self.prepared[kind]:
            return
        if target > self.ready[kind]:
            self._fatal(
                kind,
                f"prepare target {target} exceeds local ready {self.ready[kind]}",
            )

        frontier = self.committed[kind]
        digest = self.committed_digest[kind]
        for _, finish_event, ack_ids in self._queue(kind):
            ack_end = frontier + len(ack_ids)
            if ack_end > target:
                break
            if not finish_event.query():
                self._fatal(
                    kind,
                    f"prepare target {target} includes an unfinished ACK",
                )
            frontier, digest = self._extend_ack_digest(
                kind,
                digest=digest,
                frontier=frontier,
                ack_ids=ack_ids,
            )
        if frontier > self.prepared[kind]:
            self.prepared[kind] = frontier
            self.prepared_digest[kind] = digest

    def _consume(self, kind: CompletionKind, target: int) -> None:
        if target <= self.committed[kind]:
            return
        if target > self.prepared[kind]:
            self._fatal(
                kind,
                f"commit target {target} exceeds local prepared "
                f"{self.prepared[kind]}",
            )

        queue = self._queue(kind)
        frontier = self.committed[kind]
        digest = self.committed_digest[kind]
        while frontier < target:
            if not queue:
                self._fatal(kind, f"queue exhausted before commit target {target}")
            _, finish_event, ack_ids = queue[0]
            if frontier + len(ack_ids) > target:
                self._fatal(
                    kind,
                    f"commit target {target} splits a local ACK ending at "
                    f"{frontier + len(ack_ids)}",
                )
            if not finish_event.query():
                self._fatal(
                    kind,
                    f"commit target {target} reached an unfinished ACK",
                )
            queue.pop(0)
            frontier, digest = self._extend_ack_digest(
                kind,
                digest=digest,
                frontier=frontier,
                ack_ids=ack_ids,
            )
            ongoing = self._ongoing(kind)
            for ack_id in ack_ids:
                entry = ongoing.pop(ack_id, None)
                if entry is None:
                    self._fatal(
                        kind,
                        f"ack_id={ack_id} disappeared while committing",
                    )
                if kind == CompletionKind.WRITE:
                    node, lock_params = entry
                    self.cache.dec_lock_ref(node, lock_params)
                    if self.cache.enable_storage:
                        self.cache.write_backup_storage(node)
                else:
                    node, lock_params, host_lock_params = entry
                    self.cache.dec_lock_ref(node, lock_params)
                    self.cache.dec_host_lock_ref(node, host_lock_params)
        self.committed[kind] = frontier
        self.committed_digest[kind] = digest

    def _publish(self, kind: CompletionKind) -> None:
        self.coordinator.publish_local(
            kind,
            observed=self.observed[kind],
            ready=self.ready[kind],
            prepared=self.prepared[kind],
            committed=self.committed[kind],
            prepared_digest=self.prepared_digest[kind],
        )

    def _check(self, kinds: tuple[CompletionKind, ...]) -> None:
        self.coordinator.targets()
        for kind in kinds:
            self._poll_ready(kind)
            self._publish(kind)

        targets = self.coordinator.targets()
        prepare_targets = {
            CompletionKind.WRITE: targets.write_prepare,
            CompletionKind.LOAD: targets.load_prepare,
        }
        for kind in kinds:
            self._prepare(kind, prepare_targets[kind])
            self._publish(kind)

        targets = self.coordinator.targets()
        commit_targets = {
            CompletionKind.WRITE: targets.write_commit,
            CompletionKind.LOAD: targets.load_commit,
        }
        for kind in kinds:
            self._consume(kind, commit_targets[kind])
            self._publish(kind)

    def check_write(self) -> None:
        self._check((CompletionKind.WRITE,))

    def check_load(self) -> None:
        self._check((CompletionKind.LOAD,))

    def check_all(self) -> None:
        self._check((CompletionKind.WRITE, CompletionKind.LOAD))

    def quiesce(self) -> None:
        deadline = time.monotonic() + envs.SGLANG_HICACHE_PP_STALL_TIMEOUT_S.get()
        while True:
            self.check_all()
            local_done = all(
                self.observed[kind]
                == self.ready[kind]
                == self.prepared[kind]
                == self.committed[kind]
                for kind in CompletionKind
            )
            controller = self.cache.cache_controller
            queues_empty = (
                not controller.ack_write_queue
                and not controller.ack_load_queue
                and not self.cache.ongoing_write_through
                and not self.cache.ongoing_load_back
            )
            if local_done and queues_empty:
                return
            if time.monotonic() >= deadline:
                self.coordinator.report_scheduler_fatal(
                    "Timed out quiescing HiCache layered completion state: "
                    f"pp_rank={self.cache.pp_rank}, write="
                    f"({self.observed[CompletionKind.WRITE]}, "
                    f"{self.ready[CompletionKind.WRITE]}, "
                    f"{self.prepared[CompletionKind.WRITE]}, "
                    f"{self.committed[CompletionKind.WRITE]}), load="
                    f"({self.observed[CompletionKind.LOAD]}, "
                    f"{self.ready[CompletionKind.LOAD]}, "
                    f"{self.prepared[CompletionKind.LOAD]}, "
                    f"{self.committed[CompletionKind.LOAD]})"
                )
                raise RuntimeError(
                    "Timed out quiescing HiCache layered completion state"
                )
            time.sleep(
                envs.SGLANG_HICACHE_PP_PROGRESS_INTERVAL_MS.get() / 1000.0
            )
