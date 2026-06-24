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
    # Device KV slot indices allocated during prefetch (for exact rollback free).
    # With paged allocators this can be longer than committed_len because the
    # allocator works in whole pages. Only the first committed_len entries are
    # written into req_to_token / loaded from the CPU snapshot.
    device_kv_indices: object = None
    # Whether this request's state has been brought back to device (drip unit).
    prefetched: bool = False


@dataclass
class WaveStateHandle:
    wave_id: int
    state: WaveState
    reqs: List["Req"]
    entries: Dict[str, _ReqStateEntry] = field(default_factory=dict)

    # Estimated total device slot demand for decode (KV tokens). Used by the
    # admission budget check.
    d_wave_slots: int = 0

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

        logger.info(
            "OfflinePPStateOffloadManager enabled (hybrid=%s, layers=%d, "
            "resident_waves=%d).",
            self.is_hybrid,
            self.layer_num,
            RESIDENT_WAVES,
        )

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
            "OfflinePP %s wave=%d state=%s bs=%d prefetched=%d/%d "
            "committed_slots=%d device_slots=%d mamba_slots=%d%s%s",
            action,
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

    def new_wave(self, reqs: List["Req"]) -> WaveStateHandle:
        wave_id = self._next_wave_id
        self._next_wave_id += 1
        d_wave = sum(self._req_slot_demand(r) for r in reqs)
        handle = WaveStateHandle(
            wave_id=wave_id,
            state=WaveState.PREFILLING,
            reqs=list(reqs),
            d_wave_slots=d_wave,
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
                kv_indices = self._req_device_indices(req, entry.committed_len)
                mamba_indices = self._req_mamba_indices(req)
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
                entry.cpu_state = self.codec.encode_on_offload(cpu_state)
            wave.offload_done_event = self.device_module.Event()
            wave.offload_done_event.record(self.offload_stream)
        wave.offload_ready = False

    def is_wave_offload_ready(self, wave: WaveStateHandle) -> bool:
        if wave.offload_ready:
            return True
        if wave.offload_done_event is None:
            return False
        if self._event_done(wave.offload_done_event):
            wave.offload_ready = True
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
        total = self.token_to_kv_pool_allocator.size
        budget = int(total * ADMISSION_ALPHA / RESIDENT_WAVES)
        if wave.d_wave_slots <= budget:
            return True

        # Split: keep as many leading reqs as fit, re-queue the rest.
        kept: List["Req"] = []
        acc = 0
        for req in wave.reqs:
            d = wave.entries[req.rid].token_len
            if acc + d > budget:
                if not kept:
                    raise ValueError(
                        f"Single request demand {d} exceeds per-wave budget "
                        f"{budget}; cannot schedule under offline-pp-offload. "
                        "Reduce sequence length or pp memory."
                    )
                break
            acc += d
            kept.append(req)

        original_entries = wave.entries
        remainder = [r for r in wave.reqs if r not in kept]
        wave.reqs = kept
        wave.entries = {r.rid: original_entries[r.rid] for r in kept}
        wave.d_wave_slots = acc
        if remainder:
            self._requeue_remainder(remainder, original_entries)
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
        self.offloaded_queue.append(rem.wave_id)
        self._log_wave(
            "OFFLOAD_SPLIT_REQUEUE", rem, queue_len=len(self.offloaded_queue)
        )

    def start_prefetching(self, wave: WaveStateHandle) -> None:
        wave.state = WaveState.PREFETCHING
        wave.last_progress_tick = self._tick
        wave.prefetch_done_event = None
        self._log_wave("PREFETCH_START", wave, queue_len=len(self.offloaded_queue))

    def prefetch_step(self, wave: WaveStateHandle, free_slots: int) -> int:
        """Drip-prefetch: bring back as many of the wave's requests as the
        currently available ``free_slots`` budget allows.

        Returns the number of slots consumed this step. Reallocates device
        slots, loads host state back, and rebuilds the req_to_token mapping for
        each brought-back request. The wave only becomes DECODE_READY once all
        requests are resident (hard gate in ``is_wave_decode_ready``).
        """
        consumed = 0
        progressed = False
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
                break
            if entry.cpu_state is None:
                raise RuntimeError(
                    "Offline-PP prefetch missing host state for "
                    f"wave={wave.wave_id}, req={req.rid}."
                )

            # Allocator and mapping tensors are shared scheduler metadata. Keep
            # those updates on the current stream, then let the transfer stream
            # wait before issuing the large H2D copies.
            kv_indices = self.token_to_kv_pool_allocator.alloc(alloc_len)
            if kv_indices is None:
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
                    break

                restore_indices = kv_indices[:committed]
                self.req_to_token_pool.write(
                    (req.req_pool_idx, slice(0, committed)), restore_indices
                )
                cpu_state = self.codec.decode_on_prefetch(entry.cpu_state)
                mamba_indices = self._req_mamba_indices(req)

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
        return consumed

    def is_wave_decode_ready(self, wave: WaveStateHandle) -> bool:
        ready = all(e.prefetched for e in wave.entries.values()) and self._event_done(
            wave.prefetch_done_event
        )
        wave.prefetch_ready = ready
        if ready:
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
    def is_stalled(self, wave: WaveStateHandle) -> bool:
        if all(e.prefetched for e in wave.entries.values()) and not self._event_done(
            wave.prefetch_done_event
        ):
            return False
        return (self._tick - wave.last_progress_tick) >= self.stall_ticks

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
        self.waves.pop(wave.wave_id, None)
        self.offloaded_queue = [
            wid for wid in self.offloaded_queue if wid != wave.wave_id
        ]

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
