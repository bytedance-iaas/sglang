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
  - GPU holds at most two waves at a time (double-buffer). This is a structural
    constant, not a tunable.
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

# Number of resident waves on GPU. Structural constant of the double-buffer
# scheme (current DECODING wave + next PREFETCHING/DECODE_READY wave), NOT a knob.
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

    # Prefetch (drip) bookkeeping: number of requests brought back so far.
    prefetch_done_units: int = 0
    prefetch_ready: bool = False

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

        self.layer_num = self.kv_cache.layer_num

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
    def tick(self) -> None:
        """Advance the logical scheduler clock (called once per scheduler loop)."""
        self._tick += 1

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
        return handle

    def _req_slot_demand(self, req: "Req") -> int:
        """Estimate decode-time device KV slot demand (in tokens) for a req."""
        prompt = len(req.origin_input_ids)
        # max_new_tokens is the upper bound on generated tokens for this req.
        gen = getattr(req.sampling_params, "max_new_tokens", 0) or 0
        return prompt + gen

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
        idx = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        return idx[:length]

    def _req_mamba_indices(self, req: "Req") -> Optional[torch.Tensor]:
        if not self.is_hybrid:
            return None
        return self.req_to_token_pool.get_mamba_indices(
            torch.tensor([req.req_pool_idx], device=self.req_to_token_pool.device)
        )

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
        self, wave: WaveStateHandle, layer_mask: Optional[set] = None
    ) -> None:
        """Snapshot the wave's full state to host on the offload stream."""
        with self.device_module.stream(self.offload_stream):
            for req in wave.reqs:
                entry = wave.entries[req.rid]
                kv_indices = self._req_device_indices(req, entry.committed_len)
                mamba_indices = self._req_mamba_indices(req)
                if mamba_indices is not None:
                    cpu_state = self.kv_cache.get_cpu_copy(kv_indices, mamba_indices)
                else:
                    cpu_state = self.kv_cache.get_cpu_copy(kv_indices)
                entry.cpu_state = self.codec.encode_on_offload(cpu_state)
        self.offload_stream.synchronize()
        wave.offload_ready = True

    def mark_wave_offloaded(self, wave: WaveStateHandle) -> None:
        """Free the wave's device slots and move it to the OFFLOADED queue.

        Must be called only after ``offload_ready`` (state is safely on host).
        """
        assert wave.offload_ready, "mark_wave_offloaded before offload completed"
        for req in wave.reqs:
            self._free_req_device_slots(req, wave.entries[req.rid].committed_len)
        wave.state = WaveState.OFFLOADED
        wave.last_progress_tick = self._tick
        self.offloaded_queue.append(wave.wave_id)

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
        wave) and we return False for this tick so the caller retries.
        """
        total = self.token_to_kv_pool_allocator.size
        budget = int(total * ADMISSION_ALPHA / RESIDENT_WAVES)
        if wave.d_wave_slots <= budget:
            return True

        # Split: keep as many leading reqs as fit, re-queue the rest.
        kept: List["Req"] = []
        acc = 0
        for req in wave.reqs:
            d = self._req_slot_demand(req)
            if acc + d > budget and kept:
                break
            acc += d
            kept.append(req)

        if not kept:
            raise ValueError(
                f"Single request demand {self._req_slot_demand(wave.reqs[0])} "
                f"exceeds per-wave budget {budget}; cannot schedule under "
                f"offline-pp-offload. Reduce sequence length or pp memory."
            )

        remainder = [r for r in wave.reqs if r not in kept]
        wave.reqs = kept
        wave.entries = {r.rid: wave.entries[r.rid] for r in kept}
        wave.d_wave_slots = acc
        if remainder:
            self._requeue_remainder(remainder)
        return False

    def _requeue_remainder(self, reqs: List["Req"]) -> None:
        rem = self.new_wave(reqs)
        # The remainder's state is still on host from the original offload;
        # carry the cpu_state handles over so it does not need re-prefill.
        rem.state = WaveState.OFFLOADED
        rem.offload_ready = True
        for r in reqs:
            rem.entries[r.rid].cpu_state = getattr(
                r, "_offline_pp_cpu_state", None
            )
        self.offloaded_queue.append(rem.wave_id)

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
        with self.device_module.stream(self.prefetch_stream):
            for req in wave.reqs:
                entry = wave.entries[req.rid]
                if entry.prefetched:
                    continue
                # Only the committed prefix exists and needs slots now; the
                # decode path extends per-step as usual once the wave runs.
                committed = entry.committed_len
                if committed > free_slots - consumed:
                    # Not enough budget yet for this request; drip continues
                    # next tick as more slots are freed by the DECODING wave.
                    break
                # Re-allocate device slots for this request and restore mapping.
                req.req_pool_idx = None
                if self.is_hybrid:
                    req.mamba_pool_idx = None
                alloc_idx = self.req_to_token_pool.alloc([req])
                if alloc_idx is None:
                    break
                kv_indices = self.token_to_kv_pool_allocator.alloc(committed)
                if kv_indices is None:
                    break
                entry.device_kv_indices = kv_indices
                self.req_to_token_pool.write(
                    (req.req_pool_idx, slice(0, committed)), kv_indices
                )
                cpu_state = self.codec.decode_on_prefetch(entry.cpu_state)
                mamba_indices = self._req_mamba_indices(req)
                if mamba_indices is not None:
                    self.kv_cache.load_cpu_copy(cpu_state, kv_indices, mamba_indices)
                else:
                    self.kv_cache.load_cpu_copy(cpu_state, kv_indices)
                entry.prefetched = True
                wave.prefetch_done_units += 1
                consumed += committed
                progressed = True
        if progressed:
            wave.last_progress_tick = self._tick
        return consumed

    def is_wave_decode_ready(self, wave: WaveStateHandle) -> bool:
        ready = all(e.prefetched for e in wave.entries.values())
        wave.prefetch_ready = ready
        if ready:
            wave.state = WaveState.DECODE_READY
        return ready

    # ------------------------------------------------------------------ #
    # Deadlock avoidance
    # ------------------------------------------------------------------ #
    def is_stalled(self, wave: WaveStateHandle) -> bool:
        return (self._tick - wave.last_progress_tick) >= self.stall_ticks

    def rollback_wave(self, wave: WaveStateHandle) -> None:
        """Drop the partially prefetched device copy and return wave to queue.

        Safe because the authoritative state copy remains on host; only the
        in-progress device replica (and its slots) is discarded.
        """
        self.prefetch_stream.synchronize()
        for req in wave.reqs:
            entry = wave.entries[req.rid]
            if entry.prefetched and entry.device_kv_indices is not None:
                self.token_to_kv_pool_allocator.free(entry.device_kv_indices)
                if self.is_hybrid and hasattr(
                    self.req_to_token_pool, "free_mamba_cache"
                ):
                    self.req_to_token_pool.free_mamba_cache(req)
                self.req_to_token_pool.free(req)
                entry.device_kv_indices = None
                entry.prefetched = False
            req.req_pool_idx = None
            if self.is_hybrid:
                req.mamba_pool_idx = None
        wave.prefetch_done_units = 0
        wave.prefetch_ready = False
        wave.retry_count += 1
        wave.last_progress_tick = self._tick
        wave.state = WaveState.OFFLOADED
        # Aging: re-queue at the front so it is retried before newer waves.
        self.offloaded_queue.insert(0, wave.wave_id)
        # Escalate to exclusive mode after repeated stalls to guarantee
        # forward progress (run alone once the GPU is fully drained).
        if wave.retry_count >= RESIDENT_WAVES:
            wave.exclusive_mode = True
        logger.warning(
            "Offline-PP prefetch wave %d stalled, rolled back "
            "(retry=%d, exclusive=%s).",
            wave.wave_id,
            wave.retry_count,
            wave.exclusive_mode,
        )

    def pop_next_offloaded(self) -> Optional[WaveStateHandle]:
        """Return the next wave to start prefetching, or None if queue empty."""
        if not self.offloaded_queue:
            return None
        wave_id = self.offloaded_queue.pop(0)
        return self.waves.get(wave_id)

    def retire_wave(self, wave: WaveStateHandle) -> None:
        """Drop a fully-decoded wave's bookkeeping."""
        self.waves.pop(wave.wave_id, None)

    # ------------------------------------------------------------------ #
    # High-level hooks used by batch formation
    # ------------------------------------------------------------------ #
    def offload_prefilled_wave(self, reqs: List["Req"]) -> WaveStateHandle:
        """Register a just-prefilled batch as a wave, snapshot its state to
        host, and free its device slots. Returns the OFFLOADED wave handle.

        v1 performs the host snapshot synchronously here (finalize_offload);
        layer-wise overlap is a future refinement (offload_wave_layer hook).
        """
        wave = self.new_wave(reqs)
        self.finalize_offload(wave)
        self.mark_wave_offloaded(wave)
        return wave

    def take_decode_ready_wave(self) -> Optional[WaveStateHandle]:
        """Return a DECODE_READY wave and transition it to DECODING, or None."""
        for wave in self.waves.values():
            if wave.state == WaveState.DECODE_READY:
                wave.state = WaveState.DECODING
                return wave
        return None
