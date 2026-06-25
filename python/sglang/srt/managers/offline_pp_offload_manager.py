from __future__ import annotations

"""Offline PP state offload manager.

Implements prefill-side asynchronous state offload and decode-side asynchronous
prefetch for throughput-oriented offline Pipeline-Parallel serving.

Design (see plan §3.2/3.3/3.3.1):
  - After a wave finishes prefill, its state (KV cache + mamba state for hybrid
    models) is offloaded to host (CPU) and its GPU slots are freed, so prefill
    can keep advancing and accumulate decode waves.
  - During decode, the next wave is prefetched back wave-by-wave ("drip"): as
    soon as free slots appear (released by the finishing DECODING wave) they are
    used to bring back part of the next wave's state, until the whole wave is
    resident and re-mapped, at which point it becomes DECODE_READY.
  - Prefetch is serialized (at most one PREFETCHING wave), while any number of
    fully prefetched waves may wait in DECODE_READY if device memory allows it.
  - Deadlock avoidance: admission budget check (+ wave splitting), no-progress
    timeout rollback (host keeps the authoritative copy, so dropping the partial
    device copy is lossless), and aging / exclusive-mode fallback.

This module deliberately reuses the pool-level ``get_cpu_copy`` /
``load_cpu_copy`` primitives as the device<->host transfer engine and runs them
on a dedicated stream, rather than coupling to the radix/HiCache prefix-reuse
machinery.
"""

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.utils.common import get_device_module

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Per-wave admission divisor. Runtime residency is limited by allocator
# availability; only one wave may be PREFETCHING at a time.
RESIDENT_WAVES = 2

# Fraction of the total allocatable slots a single wave may use, leaving margin
# for fragmentation. Combined with the /RESIDENT_WAVES split for admission.
ADMISSION_ALPHA = 0.95


class WaveState(enum.Enum):
    PREFILLING = "prefilling"
    OFFLOADED = "offloaded"
    PREFETCHING = "prefetching"
    DECODE_READY = "decode_ready"
    DECODING = "decoding"


class EpochState(enum.Enum):
    FILLING = "filling"
    DRAINING = "draining"


@dataclass
class PrefetchStepResult:
    consumed_slots: int = 0
    progressed: bool = False
    blocked_reason: Optional[str] = None
    single_req_too_large: bool = False


@dataclass
class PrefillBudget:
    max_prefill_tokens: int
    prefill_max_requests: Optional[int]
    limited_by: str
    kv_decode_slots: int
    kv_prefetch_slots: int
    mamba_decode_slots: int
    mamba_prefetch_slots: int
    host_bytes_est: int


class StateCodec:
    """Hook for compressing/decompressing state on the offload<->prefetch path.

    v1 uses identity (no compression). A future codec (e.g. FP8/INT4 quant) can
    subclass and reduce transfer volume when ``T_decode_wave < T_prefetch_wave``.
    """

    def encode_on_offload(self, cpu_state):
        return cpu_state

    def decode_on_prefetch(self, cpu_state):
        return cpu_state


@dataclass
class _ReqStateEntry:
    """Per-request host-resident state handle within a wave."""

    rid: str
    token_len: int
    # Number of committed KV tokens that actually exist and must be moved
    # (== prompt length at prefill completion). Distinct from token_len, which
    # is the prompt+gen demand used only for the admission budget.
    committed_len: int = 0
    # Host (CPU) copy of this request's state, as returned by pool.get_cpu_copy.
    cpu_state: object = None
    host_bytes: int = 0
    # Device KV slot indices allocated during prefetch (for exact rollback free).
    # With paged allocators this can be longer than committed_len because the
    # allocator works in whole pages. Only the first committed_len entries are
    # written into req_to_token / loaded from the CPU snapshot.
    device_kv_indices: object = None
    # GPU index tensors captured by async D2H/H2D transfer kernels. They must
    # stay alive until the corresponding transfer event completes; otherwise the
    # caching allocator may reuse their storage while an async index kernel still
    # reads it.
    transfer_kv_indices: object = None
    transfer_mamba_indices: object = None
    # Whether this request's state has been brought back to device (drip unit).
    prefetched: bool = False


@dataclass
class WaveStateHandle:
    wave_id: int
    epoch_id: int
    state: WaveState
    reqs: List["Req"]
    entries: Dict[str, _ReqStateEntry] = field(default_factory=dict)

    # Estimated total device slot demand for decode (KV tokens). Used by the
    # admission budget check.
    d_wave_slots: int = 0
    host_bytes_est: int = 0
    host_bytes_exact: int = 0
    limited_by: str = "unknown"

    # Offload bookkeeping (layer-wise): number of layers offloaded so far.
    offload_done_layers: int = 0
    offload_ready: bool = False
    offload_done_event: object = None

    # Prefetch (drip) bookkeeping: number of requests brought back so far.
    prefetch_done_units: int = 0
    prefetch_ready: bool = False
    prefetch_done_event: object = None

    # Deadlock-avoidance bookkeeping.
    last_progress_tick: int = 0
    retry_count: int = 0
    exclusive_mode: bool = False
    blocked_reason: Optional[str] = None
    blocked_since_tick: Optional[int] = None
    blocked_since_time: Optional[float] = None


class OfflinePPStateOffloadManager:
    """Manage offline-PP state offload (prefill) and prefetch (decode)."""

    def __init__(
        self,
        req_to_token_pool: "ReqToTokenPool",
        token_to_kv_pool_allocator: "BaseTokenToKVPoolAllocator",
        server_args: "ServerArgs",
        codec: Optional[StateCodec] = None,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        self.server_args = server_args
        self.codec = codec or StateCodec()

        self.stall_ticks = server_args.offline_pp_prefetch_stall_ticks
        self.prefetch_hard_timeout_sec = (
            server_args.offline_pp_prefetch_hard_timeout_sec
        )
        self.gpu_resident_factor = max(
            float(server_args.offline_pp_gpu_resident_factor), 2.0
        )
        self.min_prefill_waves = (
            server_args.offline_pp_min_prefill_waves or server_args.pp_size
        )
        self.max_prefill_waves = server_args.offline_pp_max_prefill_waves
        max_host_gb = server_args.offline_pp_max_host_memory_gb
        self.host_limit_bytes = (
            int(max_host_gb * (1024**3)) if max_host_gb is not None else None
        )

        # Whether this model is hybrid (KV + mamba). HybridReqToTokenPool exposes
        # get_mamba_indices; the kv_cache get_cpu_copy then accepts mamba_indices.
        self.is_hybrid = hasattr(self.req_to_token_pool, "get_mamba_indices")

        device_module = get_device_module()
        # Dedicated transfer streams so offload/prefetch overlap with the main
        # compute stream (full-duplex PCIe: offload=D2H, prefetch=H2D).
        self.offload_stream = device_module.Stream()
        self.prefetch_stream = device_module.Stream()
        self.device_module = device_module

        self.layer_num = self._infer_layer_num()

        # Wave registry.
        self._next_wave_id = 0
        self.waves: Dict[int, WaveStateHandle] = {}
        # Queue of OFFLOADED waves waiting for their decode turn (wave_ids).
        self.offloaded_queue: List[int] = []

        self._tick = 0
        self.epoch_state = EpochState.FILLING
        self.current_epoch_id = 0
        self.fill_stop_requested = False
        self.fill_stop_reason: Optional[str] = None
        self.inflight_prefill_mbs: set[int] = set()
        self.host_pinned_bytes = 0
        self._host_bytes_per_committed_token: Optional[float] = None
        self._host_bytes_samples = 0

        logger.info(
            "OfflinePPStateOffloadManager enabled (hybrid=%s, layers=%d, "
            "resident_factor=%.2f, min_prefill_waves=%d, "
            "max_prefill_waves=%s, host_limit_bytes=%s).",
            self.is_hybrid,
            self.layer_num,
            self.gpu_resident_factor,
            self.min_prefill_waves,
            self.max_prefill_waves,
            self.host_limit_bytes,
        )
        if self.host_limit_bytes is None:
            logger.warning(
                "OfflinePP host memory budget is unset; pinned CPU offload state "
                "is effectively unlimited. Set --offline-pp-max-host-memory-gb "
                "for production runs."
            )
        self._log_epoch("EPOCH_START", reason="init")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _infer_layer_num(self) -> int:
        """Best-effort layer count for logging and future layer-wise hooks.

        Most KV pools expose ``layer_num`` directly. Hybrid linear pools compose
        full-attention KV layers with mamba state layers and expose the counts on
        their sub-pools instead.
        """
        layer_num = getattr(self.kv_cache, "layer_num", None)
        if layer_num is not None:
            return layer_num

        transfer_layer_num = getattr(self.kv_cache, "transfer_layer_num", None)
        if transfer_layer_num is not None:
            return transfer_layer_num

        full_layer_num = getattr(self.kv_cache, "full_layer_nums", None)
        mamba_pool = getattr(self.kv_cache, "mamba_pool", None)
        mamba_layer_num = getattr(mamba_pool, "num_mamba_layers", None)
        if full_layer_num is not None or mamba_layer_num is not None:
            return (full_layer_num or 0) + (mamba_layer_num or 0)

        full_kv_pool = getattr(self.kv_cache, "full_kv_pool", None)
        layer_num = getattr(full_kv_pool, "layer_num", None)
        if layer_num is not None:
            return layer_num

        logger.warning(
            "Unable to infer KV cache layer count for offline PP offload "
            "(kv_cache=%s); continuing with layer_num=0.",
            type(self.kv_cache).__name__,
        )
        return 0

    def tick(self) -> None:
        """Advance the logical scheduler clock (called once per scheduler loop)."""
        self._tick += 1

    def _event_done(self, event) -> bool:
        return event is None or event.query()

    def _log_epoch(self, action: str, **kwargs) -> None:
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(
            "OfflinePP %s epoch=%d state=%s waves=%d host_bytes=%d "
            "host_limit=%s%s%s",
            action,
            self.current_epoch_id,
            self.epoch_state.value,
            self.active_epoch_wave_count(),
            self.host_pinned_bytes,
            self.host_limit_bytes,
            " " if extra else "",
            extra,
        )

    def _object_nbytes(self, obj) -> int:
        if obj is None:
            return 0
        if isinstance(obj, torch.Tensor):
            return int(obj.numel() * obj.element_size())
        if isinstance(obj, dict):
            return sum(self._object_nbytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(self._object_nbytes(v) for v in obj)
        if hasattr(obj, "__dict__"):
            return self._object_nbytes(vars(obj))
        return 0

    def _observe_host_bytes(self, committed_len: int, host_bytes: int) -> None:
        if committed_len <= 0 or host_bytes <= 0:
            return
        sample = host_bytes / committed_len
        if self._host_bytes_per_committed_token is None:
            self._host_bytes_per_committed_token = sample
        else:
            n = self._host_bytes_samples
            self._host_bytes_per_committed_token = (
                self._host_bytes_per_committed_token * n + sample
            ) / (n + 1)
        self._host_bytes_samples += 1

    def _set_entry_host_state(
        self, wave: WaveStateHandle, entry: _ReqStateEntry, cpu_state
    ) -> None:
        old = entry.host_bytes
        entry.cpu_state = cpu_state
        entry.host_bytes = self._object_nbytes(cpu_state)
        delta = entry.host_bytes - old
        wave.host_bytes_exact += delta
        self.host_pinned_bytes += delta
        self._observe_host_bytes(entry.committed_len, entry.host_bytes)

    def _release_wave_cpu_state(self, wave: WaveStateHandle) -> None:
        released = 0
        for entry in wave.entries.values():
            if entry.cpu_state is not None or entry.host_bytes:
                released += entry.host_bytes
                entry.cpu_state = None
                entry.host_bytes = 0
        if released:
            wave.host_bytes_exact = max(0, wave.host_bytes_exact - released)
            self.host_pinned_bytes = max(0, self.host_pinned_bytes - released)
            self._log_wave("HOST_STATE_RELEASE", wave, released_bytes=released)

    def _clear_transfer_indices(self, wave: WaveStateHandle) -> None:
        for entry in wave.entries.values():
            entry.transfer_kv_indices = None
            entry.transfer_mamba_indices = None

    def _log_wave(self, action: str, wave: WaveStateHandle, **kwargs) -> None:
        device_slots = 0
        for entry in wave.entries.values():
            if entry.device_kv_indices is not None:
                device_slots += int(entry.device_kv_indices.numel())
        mamba_slots = sum(
            1 for req in wave.reqs if getattr(req, "mamba_pool_idx", None) is not None
        )
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(
            "OfflinePP %s epoch=%d wave=%d state=%s bs=%d prefetched=%d/%d "
            "committed_slots=%d device_slots=%d mamba_slots=%d%s%s",
            action,
            wave.epoch_id,
            wave.wave_id,
            wave.state.value,
            len(wave.reqs),
            wave.prefetch_done_units,
            len(wave.reqs),
            sum(entry.committed_len for entry in wave.entries.values()),
            device_slots,
            mamba_slots,
            " " if extra else "",
            extra,
        )

    def has_decoding_wave(self) -> bool:
        return any(wave.state == WaveState.DECODING for wave in self.waves.values())

    def active_decoding_wave_id(self) -> Optional[int]:
        for wave in self.waves.values():
            if wave.state == WaveState.DECODING:
                return wave.wave_id
        return None

    def active_epoch_wave_count(self, epoch_id: Optional[int] = None) -> int:
        epoch_id = self.current_epoch_id if epoch_id is None else epoch_id
        return sum(1 for wave in self.waves.values() if wave.epoch_id == epoch_id)

    def has_active_epoch_waves(self, epoch_id: Optional[int] = None) -> bool:
        return self.active_epoch_wave_count(epoch_id) > 0

    def is_filling(self) -> bool:
        return self.epoch_state == EpochState.FILLING

    def is_draining(self) -> bool:
        return self.epoch_state == EpochState.DRAINING

    @property
    def inflight_prefill_count(self) -> int:
        return len(self.inflight_prefill_mbs)

    def can_dispatch_prefill(self) -> bool:
        return self.is_filling() and not self.fill_stop_requested

    def has_offline_work(self) -> bool:
        return (
            bool(self.waves)
            or self.fill_stop_requested
            or self.inflight_prefill_count > 0
        )

    def _normalize_mb_id(self, mb_id: Optional[int]) -> int:
        return int(mb_id) if mb_id is not None else -1

    def _log_prefill_inflight(self, action: str, mb_id: Optional[int]) -> None:
        logger.info(
            "OfflinePP %s epoch=%d state=%s mb_id=%s inflight=%d waves=%d",
            action,
            self.current_epoch_id,
            self.epoch_state.value,
            mb_id,
            self.inflight_prefill_count,
            self.active_epoch_wave_count(),
        )

    def note_prefill_dispatched(self, mb_id: Optional[int]) -> None:
        if not self.is_filling():
            logger.warning(
                "OfflinePP PREFILL_INFLIGHT_ADD ignored outside FILLING "
                "epoch=%d state=%s mb_id=%s",
                self.current_epoch_id,
                self.epoch_state.value,
                mb_id,
            )
            return
        key = self._normalize_mb_id(mb_id)
        if key not in self.inflight_prefill_mbs:
            self.inflight_prefill_mbs.add(key)
        self._log_prefill_inflight("PREFILL_INFLIGHT_ADD", mb_id)

    def note_prefill_offloaded(self, mb_id: Optional[int]) -> None:
        key = self._normalize_mb_id(mb_id)
        if key in self.inflight_prefill_mbs:
            self.inflight_prefill_mbs.remove(key)
        elif self.is_filling() or self.fill_stop_requested:
            logger.warning(
                "OfflinePP PREFILL_INFLIGHT_DONE for unknown mb_id "
                "epoch=%d state=%s mb_id=%s inflight=%d",
                self.current_epoch_id,
                self.epoch_state.value,
                mb_id,
                self.inflight_prefill_count,
            )
        self._log_prefill_inflight("PREFILL_INFLIGHT_DONE", mb_id)
        self.maybe_enter_draining()

    def local_fill_stop_reason(self, waiting_queue_empty: bool) -> Optional[str]:
        wave_count = self.active_epoch_wave_count()
        if wave_count == 0:
            return None
        if (
            self.host_limit_bytes is not None
            and self.host_pinned_bytes >= self.host_limit_bytes
        ):
            return "host"
        if (
            self.max_prefill_waves is not None
            and wave_count >= self.max_prefill_waves
        ):
            return "max_waves"
        if waiting_queue_empty:
            return "waiting_empty"
        if wave_count < self.min_prefill_waves:
            return None
        return None

    def enter_draining(self, reason: str) -> None:
        if self.epoch_state == EpochState.DRAINING:
            return
        self._log_epoch("EPOCH_FILL_STOP", reason=reason)
        self.epoch_state = EpochState.DRAINING
        self._log_epoch("EPOCH_DRAIN_START", reason=reason)
        if not self.has_active_epoch_waves():
            self._maybe_finish_epoch_after_retire(self.current_epoch_id)

    def request_draining(self, reason: str) -> None:
        if self.epoch_state != EpochState.FILLING:
            return
        if not self.fill_stop_requested:
            self.fill_stop_requested = True
            self.fill_stop_reason = reason
            self._log_epoch(
                "EPOCH_FILL_STOP_REQUEST",
                reason=reason,
                inflight_prefill=self.inflight_prefill_count,
            )
        self.maybe_enter_draining()

    def maybe_enter_draining(self) -> bool:
        if self.epoch_state != EpochState.FILLING:
            return False
        if not self.fill_stop_requested:
            return False
        if self.inflight_prefill_count > 0:
            return False
        self.enter_draining(self.fill_stop_reason or "unknown")
        return True

    def _maybe_finish_epoch_after_retire(self, retired_epoch_id: int) -> None:
        if self.epoch_state != EpochState.DRAINING:
            return
        if self.has_active_epoch_waves(retired_epoch_id):
            return
        self._log_epoch("EPOCH_DRAIN_DONE", retired_epoch=retired_epoch_id)
        self.current_epoch_id += 1
        self.epoch_state = EpochState.FILLING
        self.fill_stop_requested = False
        self.fill_stop_reason = None
        self.inflight_prefill_mbs.clear()
        self._log_epoch("EPOCH_START", reason="previous_epoch_drained")

    def _mamba_pool_size(self) -> Optional[int]:
        if not self.is_hybrid:
            return None
        mamba_pool = getattr(self.req_to_token_pool, "mamba_pool", None)
        if mamba_pool is not None and getattr(mamba_pool, "size", None) is not None:
            return int(mamba_pool.size)
        allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        if allocator is not None and getattr(allocator, "size", None) is not None:
            return int(allocator.size)
        return None

    def _mamba_available_size(self) -> Optional[int]:
        if not self.is_hybrid:
            return None
        allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        if allocator is None or not hasattr(allocator, "available_size"):
            return None
        return int(allocator.available_size())

    def _req_mamba_slot_demand(self, req: "Req") -> int:
        return 1 if self.is_hybrid else 0

    def _wave_mamba_slots(self, reqs: List["Req"]) -> int:
        return sum(self._req_mamba_slot_demand(req) for req in reqs)

    def _req_committed_slot_demand(self, req: "Req") -> int:
        return self._align_slot_count(self._req_committed_len(req))

    def _estimate_req_host_bytes(self, req: "Req") -> int:
        if self._host_bytes_per_committed_token is None:
            return 0
        return int(self._host_bytes_per_committed_token * self._req_committed_len(req))

    def _gpu_kv_wave_budget(self) -> int:
        return max(
            1,
            int(
                self.token_to_kv_pool_allocator.size
                * ADMISSION_ALPHA
                / self.gpu_resident_factor
            ),
        )

    def _gpu_mamba_wave_budget(self) -> Optional[int]:
        size = self._mamba_pool_size()
        if size is None:
            return None
        return max(1, int(size * ADMISSION_ALPHA / self.gpu_resident_factor))

    def _host_wave_budget(self) -> Optional[int]:
        if self.host_limit_bytes is None:
            return None
        target_waves = self.min_prefill_waves
        if self.max_prefill_waves is not None:
            target_waves = min(target_waves, self.max_prefill_waves)
        per_wave_budget = self.host_limit_bytes // max(1, target_waves)
        remaining_budget = self.host_limit_bytes - self.host_pinned_bytes
        if remaining_budget <= 0:
            return 0
        return max(1, min(per_wave_budget, remaining_budget))

    def prefill_budget_for_waiting_queue(
        self,
        waiting_queue: List["Req"],
        base_max_prefill_tokens: int,
        base_prefill_max_requests: Optional[int],
    ) -> PrefillBudget:
        kv_decode_budget = self._gpu_kv_wave_budget()
        kv_prefetch_budget = kv_decode_budget
        mamba_budget = self._gpu_mamba_wave_budget()
        host_budget = self._host_wave_budget()

        req_cap = (
            base_prefill_max_requests
            if base_prefill_max_requests is not None
            else len(waiting_queue)
        )
        req_cap = max(1, req_cap)

        kv_decode = 0
        kv_prefetch = 0
        mamba_slots = 0
        host_est = 0
        prompt_tokens = 0
        limited_by = "user_cap" if base_prefill_max_requests is not None else "none"
        accepted = 0

        for req in waiting_queue:
            if accepted >= req_cap:
                limited_by = "user_cap"
                break
            req_kv_decode = self._req_slot_demand(req)
            req_kv_prefetch = self._req_committed_slot_demand(req)
            req_mamba = self._req_mamba_slot_demand(req)
            req_host = self._estimate_req_host_bytes(req)
            would_decode = kv_decode + req_kv_decode
            would_prefetch = kv_prefetch + req_kv_prefetch
            would_mamba = mamba_slots + req_mamba
            would_host = host_est + req_host

            if accepted > 0 and would_decode > kv_decode_budget:
                limited_by = "gpu_double_buffer"
                break
            if accepted > 0 and would_prefetch > kv_prefetch_budget:
                limited_by = "gpu_double_buffer"
                break
            if (
                accepted > 0
                and mamba_budget is not None
                and would_mamba > mamba_budget
            ):
                limited_by = "mamba_slots"
                break
            if (
                accepted > 0
                and host_budget is not None
                and req_host > 0
                and would_host > host_budget
            ):
                limited_by = "host"
                break

            kv_decode = would_decode
            kv_prefetch = would_prefetch
            mamba_slots = would_mamba
            host_est = would_host
            prompt_tokens += self._req_committed_len(req)
            accepted += 1

            if accepted >= req_cap:
                limited_by = "user_cap"
                break

        if accepted == 0 and waiting_queue:
            req = waiting_queue[0]
            accepted = 1
            kv_decode = self._req_slot_demand(req)
            kv_prefetch = self._req_committed_slot_demand(req)
            mamba_slots = self._req_mamba_slot_demand(req)
            host_est = self._estimate_req_host_bytes(req)
            prompt_tokens = self._req_committed_len(req)
            limited_by = "single_req_too_large"

        max_prefill_tokens = max(
            1,
            min(base_max_prefill_tokens, prompt_tokens or base_max_prefill_tokens),
        )
        if base_prefill_max_requests is None:
            prefill_max_requests = accepted if waiting_queue else None
        else:
            prefill_max_requests = max(1, min(base_prefill_max_requests, accepted))

        budget = PrefillBudget(
            max_prefill_tokens=max_prefill_tokens,
            prefill_max_requests=prefill_max_requests,
            limited_by=limited_by,
            kv_decode_slots=kv_decode,
            kv_prefetch_slots=kv_prefetch,
            mamba_decode_slots=mamba_slots,
            mamba_prefetch_slots=mamba_slots,
            host_bytes_est=host_est,
        )
        logger.info(
            "OfflinePP WAVE_BUDGET epoch=%d max_prefill_tokens=%d "
            "prefill_max_requests=%s kv_decode_slots=%d kv_prefetch_slots=%d "
            "mamba_decode_slots=%d mamba_prefetch_slots=%d host_bytes_est=%d "
            "host_wave_budget=%s limited_by=%s",
            self.current_epoch_id,
            budget.max_prefill_tokens,
            budget.prefill_max_requests,
            budget.kv_decode_slots,
            budget.kv_prefetch_slots,
            budget.mamba_decode_slots,
            budget.mamba_prefetch_slots,
            budget.host_bytes_est,
            host_budget,
            budget.limited_by,
        )
        return budget

    def new_wave(self, reqs: List["Req"]) -> WaveStateHandle:
        wave_id = self._next_wave_id
        self._next_wave_id += 1
        d_wave = sum(self._req_slot_demand(r) for r in reqs)
        host_est = sum(self._estimate_req_host_bytes(r) for r in reqs)
        handle = WaveStateHandle(
            wave_id=wave_id,
            epoch_id=self.current_epoch_id,
            state=WaveState.PREFILLING,
            reqs=list(reqs),
            d_wave_slots=d_wave,
            host_bytes_est=host_est,
            last_progress_tick=self._tick,
        )
        for r in reqs:
            handle.entries[r.rid] = _ReqStateEntry(
                rid=r.rid,
                token_len=self._req_slot_demand(r),
                committed_len=self._req_committed_len(r),
            )
        self.waves[wave_id] = handle
        self._log_wave("OFFLOAD_ENQUEUE", handle)
        return handle

    def _req_slot_demand(self, req: "Req") -> int:
        """Estimate decode-time device KV slot demand (in tokens) for a req."""
        prompt = len(req.origin_input_ids)
        # max_new_tokens is the upper bound on generated tokens for this req.
        gen = getattr(req.sampling_params, "max_new_tokens", 0) or 0
        return self._align_slot_count(prompt + gen)

    def _align_slot_count(self, token_count: int) -> int:
        page_size = getattr(self.token_to_kv_pool_allocator, "page_size", 1) or 1
        if page_size <= 1 or token_count == 0:
            return token_count
        return ((token_count + page_size - 1) // page_size) * page_size

    def _req_committed_len(self, req: "Req") -> int:
        """Number of KV tokens that actually exist for this req right now.

        At prefill completion this is the committed prompt length; this is what
        must be physically moved device<->host (not the prompt+gen demand).
        """
        committed = getattr(req, "kv_committed_len", 0) or 0
        if committed > 0:
            return committed
        return len(req.origin_input_ids)

    def _req_device_indices(self, req: "Req", length: int) -> torch.Tensor:
        """First ``length`` KV slot indices currently mapped to this request."""
        idx = self.req_to_token_pool.req_to_token[
            self._checked_req_pool_idx(req, "read KV mapping")
        ]
        return idx[:length]

    def _req_mamba_indices(self, req: "Req") -> Optional[torch.Tensor]:
        if not self.is_hybrid:
            return None
        req_pool_idx = self._checked_req_pool_idx(req, "read mamba mapping")
        return self.req_to_token_pool.get_mamba_indices(
            torch.tensor(
                [req_pool_idx],
                dtype=torch.int64,
                device=self.req_to_token_pool.device,
            )
        )

    def _checked_req_pool_idx(self, req: "Req", context: str) -> int:
        req_pool_idx = getattr(req, "req_pool_idx", None)
        if req_pool_idx is None:
            raise RuntimeError(
                "Offline-PP request has no req_pool_idx while trying to "
                f"{context}: rid={getattr(req, 'rid', None)}."
            )
        req_pool_idx = int(req_pool_idx)
        alloc_size = getattr(
            self.req_to_token_pool,
            "_alloc_size",
            getattr(self.req_to_token_pool, "size", 0) + 1,
        )
        if req_pool_idx <= 0 or req_pool_idx >= alloc_size:
            raise RuntimeError(
                "Offline-PP request has invalid req_pool_idx while trying to "
                f"{context}: rid={getattr(req, 'rid', None)} "
                f"req_pool_idx={req_pool_idx} valid=[1,{alloc_size - 1}]."
            )
        return req_pool_idx

    # ------------------------------------------------------------------ #
    # Prefill-side: layer-wise offload
    # ------------------------------------------------------------------ #
    def offload_wave_layer(
        self,
        wave: WaveStateHandle,
        layer_id: int,
        layer_mask: Optional[set] = None,
    ) -> None:
        """Asynchronously offload one prefill layer's state for the wave.

        ``layer_mask`` defaults to all layers; reserved for the future
        "keep some layers resident on GPU" balancing extension.
        """
        if layer_mask is not None and layer_id not in layer_mask:
            return
        # The pool-level get_cpu_copy operates on whole requests rather than a
        # single layer slice, so we snapshot per-request state once the final
        # layer is reached. Earlier layers only advance the progress counter so
        # that prefill tail latency is hidden behind ongoing layer compute.
        wave.offload_done_layers = max(wave.offload_done_layers, layer_id + 1)

    def finalize_offload(
        self,
        wave: WaveStateHandle,
        layer_mask: Optional[set] = None,
        source_stream=None,
    ) -> None:
        """Snapshot the wave's full state to host on the offload stream."""
        if source_stream is not None:
            self.offload_stream.wait_stream(source_stream)
        self.offload_stream.wait_stream(self.device_module.current_stream())
        with self.device_module.stream(self.offload_stream):
            for req in wave.reqs:
                entry = wave.entries[req.rid]
                kv_indices = self._req_device_indices(
                    req, entry.committed_len
                ).clone()
                mamba_indices = self._req_mamba_indices(req)
                if mamba_indices is not None:
                    mamba_indices = mamba_indices.clone()
                entry.transfer_kv_indices = kv_indices
                entry.transfer_mamba_indices = mamba_indices
                if mamba_indices is not None:
                    cpu_state = self.kv_cache.get_cpu_copy(
                        kv_indices,
                        mamba_indices,
                        async_copy=True,
                        pin_memory=True,
                    )
                else:
                    cpu_state = self.kv_cache.get_cpu_copy(
                        kv_indices,
                        async_copy=True,
                        pin_memory=True,
                    )
                self._set_entry_host_state(
                    wave, entry, self.codec.encode_on_offload(cpu_state)
                )
            wave.offload_done_event = self.device_module.Event()
            wave.offload_done_event.record(self.offload_stream)
        wave.offload_ready = False
        self._log_wave(
            "WAVE_ADMIT",
            wave,
            kv_decode_slots=wave.d_wave_slots,
            kv_prefetch_slots=sum(entry.committed_len for entry in wave.entries.values()),
            mamba_decode_slots=self._wave_mamba_slots(wave.reqs),
            mamba_prefetch_slots=self._wave_mamba_slots(wave.reqs),
            host_bytes_est=wave.host_bytes_est,
            host_bytes_exact=wave.host_bytes_exact,
            limited_by=wave.limited_by,
            total_host_bytes=self.host_pinned_bytes,
        )

    def is_wave_offload_ready(self, wave: WaveStateHandle) -> bool:
        if wave.offload_ready:
            return True
        if wave.offload_done_event is None:
            return False
        if self._event_done(wave.offload_done_event):
            wave.offload_ready = True
            self._clear_transfer_indices(wave)
            self._log_wave("OFFLOAD_READY", wave)
        return wave.offload_ready

    def wait_wave_offload_ready(self, wave: WaveStateHandle) -> bool:
        if wave.offload_ready:
            return True
        if wave.offload_done_event is None:
            return False
        self._log_wave("OFFLOAD_WAIT", wave)
        wave.offload_done_event.synchronize()
        wave.offload_ready = True
        self._clear_transfer_indices(wave)
        self._log_wave("OFFLOAD_READY", wave)
        return True

    def mark_wave_offloaded(self, wave: WaveStateHandle) -> bool:
        """Free the wave's device slots and move it to the OFFLOADED queue.

        Must be called only after ``offload_ready`` (state is safely on host).
        """
        if not self.is_wave_offload_ready(wave):
            return False
        for req in wave.reqs:
            self._free_req_device_slots(req, wave.entries[req.rid].committed_len)
        wave.state = WaveState.OFFLOADED
        wave.last_progress_tick = self._tick
        self.offloaded_queue.append(wave.wave_id)
        self._log_wave(
            "OFFLOAD_TO_HOST",
            wave,
            queue_len=len(self.offloaded_queue),
        )
        return True

    def _free_req_device_slots(self, req: "Req", length: int) -> None:
        kv_indices = self._req_device_indices(req, length)
        self.token_to_kv_pool_allocator.free(kv_indices)
        if self.is_hybrid and hasattr(self.req_to_token_pool, "free_mamba_cache"):
            # Free the mamba slot; it will be re-allocated on prefetch.
            self.req_to_token_pool.free_mamba_cache(req)
        # Release the req_to_token row; re-allocated when prefetched back.
        self.req_to_token_pool.free(req)

    # ------------------------------------------------------------------ #
    # Decode-side: admission + drip prefetch
    # ------------------------------------------------------------------ #
    def admit_wave(self, wave: WaveStateHandle) -> bool:
        """Budget check before a wave enters PREFETCHING.

        Returns True if admitted. If the wave's demand exceeds the per-wave
            budget it is split in place (reqs trimmed, remainder re-queued as a new
        wave). The kept half stays schedulable in this tick.
        """
        kv_budget = self._gpu_kv_wave_budget()
        mamba_budget = self._gpu_mamba_wave_budget()
        wave_mamba_slots = self._wave_mamba_slots(wave.reqs)
        if wave.d_wave_slots <= kv_budget and (
            mamba_budget is None or wave_mamba_slots <= mamba_budget
        ):
            return True

        # Split: keep as many leading reqs as fit, re-queue the rest.
        kept: List["Req"] = []
        kv_acc = 0
        mamba_acc = 0
        for req in wave.reqs:
            d = wave.entries[req.rid].token_len
            m = self._req_mamba_slot_demand(req)
            exceeds_kv = kv_acc + d > kv_budget
            exceeds_mamba = mamba_budget is not None and mamba_acc + m > mamba_budget
            if exceeds_kv or exceeds_mamba:
                if not kept:
                    raise ValueError(
                        f"Single request demand kv={d}, mamba={m} exceeds "
                        f"per-wave budget kv={kv_budget}, mamba={mamba_budget}; "
                        "cannot schedule under offline-pp-offload. "
                        "Reduce sequence length or pp memory."
                    )
                break
            kv_acc += d
            mamba_acc += m
            kept.append(req)

        original_entries = wave.entries
        remainder = [r for r in wave.reqs if r not in kept]
        wave.reqs = kept
        wave.entries = {r.rid: original_entries[r.rid] for r in kept}
        wave.d_wave_slots = kv_acc
        wave.host_bytes_exact = sum(entry.host_bytes for entry in wave.entries.values())
        wave.host_bytes_est = sum(self._estimate_req_host_bytes(r) for r in kept)
        if remainder:
            self._requeue_remainder(remainder, original_entries)
        self._log_wave(
            "WAVE_ADMIT",
            wave,
            kv_decode_slots=wave.d_wave_slots,
            kv_prefetch_slots=sum(entry.committed_len for entry in wave.entries.values()),
            mamba_decode_slots=self._wave_mamba_slots(wave.reqs),
            mamba_prefetch_slots=self._wave_mamba_slots(wave.reqs),
            host_bytes_est=wave.host_bytes_est,
            host_bytes_exact=wave.host_bytes_exact,
            limited_by="admission_split",
        )
        return True

    def _requeue_remainder(
        self, reqs: List["Req"], source_entries: Dict[str, _ReqStateEntry]
    ) -> None:
        rem = self.new_wave(reqs)
        # The remainder's state is still on host from the original offload;
        # carry the cpu_state handles over so it does not need re-prefill.
        rem.state = WaveState.OFFLOADED
        rem.offload_ready = True
        for r in reqs:
            entry = source_entries[r.rid]
            entry.device_kv_indices = None
            entry.prefetched = False
            rem.entries[r.rid] = entry
        rem.d_wave_slots = sum(rem.entries[r.rid].token_len for r in reqs)
        rem.host_bytes_exact = sum(entry.host_bytes for entry in rem.entries.values())
        rem.host_bytes_est = sum(self._estimate_req_host_bytes(r) for r in reqs)
        self.offloaded_queue.append(rem.wave_id)
        self._log_wave(
            "OFFLOAD_SPLIT_REQUEUE", rem, queue_len=len(self.offloaded_queue)
        )

    def start_prefetching(self, wave: WaveStateHandle) -> None:
        wave.state = WaveState.PREFETCHING
        wave.last_progress_tick = self._tick
        wave.prefetch_done_event = None
        wave.blocked_reason = None
        wave.blocked_since_tick = None
        wave.blocked_since_time = None
        self._log_wave("PREFETCH_START", wave, queue_len=len(self.offloaded_queue))

    def prefetch_step(self, wave: WaveStateHandle, free_slots: int) -> PrefetchStepResult:
        """Drip-prefetch: bring back as many of the wave's requests as the
        currently available ``free_slots`` budget allows.

        Returns progress and the first resource that blocked the step. Partial
        progress is kept resident; rollback is reserved for hard deadlock
        protection, not normal resource backpressure.
        """
        consumed = 0
        progressed = False
        blocked_reason = None
        single_req_too_large = False
        metadata_stream = self.device_module.current_stream()
        for req in wave.reqs:
            entry = wave.entries[req.rid]
            if entry.prefetched:
                continue
            # Only the committed prefix exists and needs slots now; with a
            # paged allocator the actual allocation must still be page
            # aligned, while req_to_token maps only committed tokens.
            committed = entry.committed_len
            alloc_len = self._align_slot_count(committed)
            if alloc_len > free_slots - consumed:
                # Not enough budget yet for this request; drip continues
                # next tick as more slots are freed by the DECODING wave.
                blocked_reason = "kv_slots"
                if alloc_len > self.token_to_kv_pool_allocator.size:
                    single_req_too_large = True
                break
            if entry.cpu_state is None:
                raise RuntimeError(
                    "Offline-PP prefetch missing host state for "
                    f"wave={wave.wave_id}, req={req.rid}."
                )
            mamba_available = self._mamba_available_size()
            if (
                mamba_available is not None
                and mamba_available < self._req_mamba_slot_demand(req)
            ):
                blocked_reason = "mamba_slots"
                if (
                    self._mamba_pool_size() is not None
                    and self._req_mamba_slot_demand(req) > self._mamba_pool_size()
                ):
                    single_req_too_large = True
                break
            req_available = (
                self.req_to_token_pool.available_size()
                if hasattr(self.req_to_token_pool, "available_size")
                else None
            )
            if req_available is not None and req_available <= 0:
                blocked_reason = "req_slots"
                break

            # Allocator and mapping tensors are shared scheduler metadata. Keep
            # those updates on the current stream, then let the transfer stream
            # wait before issuing the large H2D copies.
            kv_indices = self.token_to_kv_pool_allocator.alloc(alloc_len)
            if kv_indices is None:
                blocked_reason = "kv_slots"
                break

            req.req_pool_idx = None
            if self.is_hybrid:
                req.mamba_pool_idx = None
            entry.device_kv_indices = kv_indices
            try:
                req_pool_indices = self.req_to_token_pool.alloc([req])
                if req_pool_indices is None:
                    self.token_to_kv_pool_allocator.free(kv_indices)
                    entry.device_kv_indices = None
                    blocked_reason = "req_slots"
                    break

                restore_indices = kv_indices[:committed].clone()
                self.req_to_token_pool.write(
                    (req.req_pool_idx, slice(0, committed)), restore_indices
                )
                cpu_state = self.codec.decode_on_prefetch(entry.cpu_state)
                mamba_indices = self._req_mamba_indices(req)
                if mamba_indices is not None:
                    mamba_indices = mamba_indices.clone()
                entry.transfer_kv_indices = restore_indices
                entry.transfer_mamba_indices = mamba_indices

                with self.device_module.stream(self.prefetch_stream):
                    self.prefetch_stream.wait_stream(metadata_stream)
                    if mamba_indices is not None:
                        self.kv_cache.load_cpu_copy(
                            cpu_state,
                            restore_indices,
                            mamba_indices,
                            async_copy=True,
                        )
                    else:
                        self.kv_cache.load_cpu_copy(
                            cpu_state,
                            restore_indices,
                            async_copy=True,
                        )
            except Exception:
                self.prefetch_stream.synchronize()
                self._release_prefetched_req(req, entry)
                raise
            if self.is_hybrid:
                # Hybrid alloc marks a fresh mamba slot as needing a clear.
                # Offline prefetch immediately restores the authoritative host
                # state into that slot, so a later deferred clear would be wrong.
                req.mamba_needs_clear = False
            entry.prefetched = True
            wave.prefetch_done_units += 1
            consumed += alloc_len
            progressed = True

        if progressed:
            with self.device_module.stream(self.prefetch_stream):
                wave.prefetch_done_event = self.device_module.Event()
                wave.prefetch_done_event.record(self.prefetch_stream)
        if progressed:
            wave.last_progress_tick = self._tick
            self._log_wave(
                "PREFETCH_PROGRESS",
                wave,
                consumed_slots=consumed,
                free_slots=free_slots,
            )
        if progressed and wave.blocked_reason is not None:
            self._log_wave(
                "PREFETCH_RESUME",
                wave,
                previous_reason=wave.blocked_reason,
            )
            wave.blocked_reason = None
            wave.blocked_since_tick = None
            wave.blocked_since_time = None
        if blocked_reason is not None:
            self.mark_prefetch_blocked(
                wave,
                blocked_reason,
                single_req_too_large=single_req_too_large,
            )
        return PrefetchStepResult(
            consumed_slots=consumed,
            progressed=progressed,
            blocked_reason=blocked_reason,
            single_req_too_large=single_req_too_large,
        )

    def is_wave_decode_ready(self, wave: WaveStateHandle) -> bool:
        ready = all(e.prefetched for e in wave.entries.values()) and self._event_done(
            wave.prefetch_done_event
        )
        wave.prefetch_ready = ready
        if ready:
            self._clear_transfer_indices(wave)
            self._release_wave_cpu_state(wave)
            if wave.state != WaveState.DECODE_READY:
                self._log_wave("DECODE_READY", wave)
            wave.state = WaveState.DECODE_READY
        return ready

    def wait_prefetch_for_decode(self, wave: WaveStateHandle) -> None:
        if wave.prefetch_done_event is not None:
            self.device_module.current_stream().wait_event(wave.prefetch_done_event)

    def ensure_decode_ready_for_schedule(self, free_slots: int) -> None:
        """Blocking catch-up at the decode scheduling boundary.

        PP ranks must make the same batch decision. Async D2H/H2D event queries
        can complete in different scheduler ticks on different ranks, so before
        starting a decode wave we block only the boundary rank-local laggards
        until their earliest pending wave reaches DECODE_READY.
        """
        if self.is_filling():
            return
        if any(wave.state == WaveState.DECODE_READY for wave in self.waves.values()):
            return

        for wave in list(self.waves.values()):
            if (
                wave.state == WaveState.PREFILLING
                and wave.offload_done_event is not None
            ):
                if self.wait_wave_offload_ready(wave):
                    self.mark_wave_offloaded(wave)

        prefetching = [
            w for w in self.waves.values() if w.state == WaveState.PREFETCHING
        ]
        if not prefetching:
            wave = self.pop_next_offloaded()
            if wave is not None and self.admit_wave(wave):
                self.start_prefetching(wave)
                prefetching = [wave]

        for wave in prefetching:
            if free_slots > 0:
                self.prefetch_step(wave, free_slots)
            if all(e.prefetched for e in wave.entries.values()):
                if wave.prefetch_done_event is not None:
                    self._log_wave("PREFETCH_WAIT", wave)
                    wave.prefetch_done_event.synchronize()
                self.is_wave_decode_ready(wave)

    def _release_prefetched_req(self, req: "Req", entry: _ReqStateEntry) -> None:
        """Release a partially/fully prefetched request back to host-only state."""
        entry.transfer_kv_indices = None
        entry.transfer_mamba_indices = None
        if entry.device_kv_indices is not None:
            self.token_to_kv_pool_allocator.free(entry.device_kv_indices)
            entry.device_kv_indices = None
        if req.req_pool_idx is not None:
            if (
                self.is_hybrid
                and hasattr(self.req_to_token_pool, "free_mamba_cache")
                and req.mamba_pool_idx is not None
            ):
                self.req_to_token_pool.free_mamba_cache(req)
            self.req_to_token_pool.free(req)
        if self.is_hybrid:
            req.mamba_pool_idx = None
        entry.prefetched = False

    # ------------------------------------------------------------------ #
    # Deadlock avoidance
    # ------------------------------------------------------------------ #
    def mark_prefetch_blocked(
        self,
        wave: WaveStateHandle,
        reason: str,
        *,
        single_req_too_large: bool = False,
    ) -> None:
        now = time.monotonic()
        first_block = wave.blocked_reason is None
        changed = wave.blocked_reason != reason
        if first_block:
            wave.blocked_since_tick = self._tick
            wave.blocked_since_time = now
        wave.blocked_reason = reason
        if first_block or changed:
            self._log_wave(
                "PREFETCH_BLOCKED",
                wave,
                reason=reason,
                single_req_too_large=single_req_too_large,
                blocked_ticks=0,
            )

    def has_earlier_resident_wave_to_release(self, wave: WaveStateHandle) -> bool:
        return any(
            other.wave_id < wave.wave_id
            and other.state in (WaveState.DECODING, WaveState.DECODE_READY)
            for other in self.waves.values()
        )

    def is_stalled(self, wave: WaveStateHandle) -> bool:
        if all(e.prefetched for e in wave.entries.values()) and not self._event_done(
            wave.prefetch_done_event
        ):
            return False
        return (self._tick - wave.last_progress_tick) >= self.stall_ticks

    def is_hard_blocked(
        self, wave: WaveStateHandle, result: Optional[PrefetchStepResult] = None
    ) -> bool:
        if result is not None and result.single_req_too_large:
            return True
        if wave.blocked_reason is None:
            return False
        if not self.has_earlier_resident_wave_to_release(wave):
            return True
        if wave.blocked_since_time is None:
            return False
        return (
            time.monotonic() - wave.blocked_since_time
        ) >= self.prefetch_hard_timeout_sec

    def hard_block_reason(
        self, wave: WaveStateHandle, result: Optional[PrefetchStepResult] = None
    ) -> str:
        if result is not None and result.single_req_too_large:
            return "single_req_too_large"
        if not self.has_earlier_resident_wave_to_release(wave):
            return "no_earlier_resident_wave"
        return "hard_timeout"

    def handle_hard_prefetch_block(
        self, wave: WaveStateHandle, result: Optional[PrefetchStepResult] = None
    ) -> None:
        reason = self.hard_block_reason(wave, result)
        self._log_wave(
            "PREFETCH_HARD_TIMEOUT",
            wave,
            reason=reason,
            blocked_reason=wave.blocked_reason,
        )
        if reason == "single_req_too_large":
            raise RuntimeError(
                "Offline-PP prefetch cannot fit a single request back on device: "
                f"wave={wave.wave_id}, blocked_reason={wave.blocked_reason}. "
                "Reduce sequence length or increase available KV/mamba capacity."
            )
        self.rollback_wave(wave)

    def rollback_wave(self, wave: WaveStateHandle) -> None:
        """Drop the partially prefetched device copy and return wave to queue.

        Safe because the authoritative state copy remains on host; only the
        in-progress device replica (and its slots) is discarded.
        """
        self.prefetch_stream.synchronize()
        for req in wave.reqs:
            entry = wave.entries[req.rid]
            if (
                entry.prefetched
                or entry.device_kv_indices is not None
                or req.req_pool_idx is not None
            ):
                self._release_prefetched_req(req, entry)
            req.req_pool_idx = None
            if self.is_hybrid:
                req.mamba_pool_idx = None
        wave.prefetch_done_units = 0
        wave.prefetch_ready = False
        wave.prefetch_done_event = None
        wave.retry_count += 1
        wave.last_progress_tick = self._tick
        wave.blocked_reason = None
        wave.blocked_since_tick = None
        wave.blocked_since_time = None
        wave.state = WaveState.OFFLOADED
        # Aging: re-queue at the front so it is retried before newer waves.
        self.offloaded_queue.insert(0, wave.wave_id)
        # Escalate to exclusive mode after repeated stalls to guarantee
        # forward progress (run alone once the GPU is fully drained).
        if wave.retry_count >= RESIDENT_WAVES:
            wave.exclusive_mode = True
        self._log_wave(
            "ROLLBACK",
            wave,
            retry=wave.retry_count,
            exclusive=wave.exclusive_mode,
        )

    def pop_next_offloaded(self) -> Optional[WaveStateHandle]:
        """Return the next wave to start prefetching, or None if queue empty."""
        if not self.offloaded_queue:
            return None
        wave_id = self.offloaded_queue.pop(0)
        return self.waves.get(wave_id)

    def retire_wave(self, wave: WaveStateHandle) -> None:
        """Drop a fully-decoded wave's bookkeeping."""
        self._log_wave("DECODE_RETIRE", wave)
        retired_epoch_id = wave.epoch_id
        self._release_wave_cpu_state(wave)
        self.waves.pop(wave.wave_id, None)
        self.offloaded_queue = [
            wid for wid in self.offloaded_queue if wid != wave.wave_id
        ]
        self._maybe_finish_epoch_after_retire(retired_epoch_id)

    def retire_wave_by_id(self, wave_id: int) -> None:
        wave = self.waves.get(wave_id)
        if wave is not None:
            self.retire_wave(wave)

    def has_active_waves(self) -> bool:
        return bool(self.waves)

    # ------------------------------------------------------------------ #
    # High-level hooks used by batch formation
    # ------------------------------------------------------------------ #
    def offload_prefilled_wave(
        self, reqs: List["Req"], source_stream=None
    ) -> WaveStateHandle:
        """Register a just-prefilled batch as a wave, snapshot its state to
        host, and free its device slots once the async offload event completes.

        v1 snapshots the full state here (finalize_offload);
        layer-wise overlap is a future refinement (offload_wave_layer hook).
        """
        wave = self.new_wave(reqs)
        self.finalize_offload(wave, source_stream=source_stream)
        return wave

    def take_decode_ready_wave(
        self, mb_id: Optional[int] = None
    ) -> Optional[WaveStateHandle]:
        """Return a DECODE_READY wave and transition it to DECODING, or None."""
        for wave in sorted(self.waves.values(), key=lambda w: w.wave_id):
            if wave.state == WaveState.DECODE_READY:
                wave.state = WaveState.DECODING
                self._log_wave("DECODE_START", wave, mb_id=mb_id)
                return wave
        return None
