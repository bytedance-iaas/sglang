# to be combined with the sparse coordinator class and sparse algorithm family

import hashlib
import logging
import os
import time
from enum import Enum
from typing import List, Literal, NamedTuple, Optional, Union

import torch

from sglang.kernels.ops.kvcache.hisparse import (
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseDSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import get_device_module, is_hip

device_module = get_device_module()

_is_hip = is_hip()

logger = logging.getLogger(__name__)


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseMigrationAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    migrated_bytes: int


class HiSparseTokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float
    resident_requests: int
    device_buffered_requests: int
    promotions: int
    demotions: int
    promotion_failures: int
    promotion_migrated_bytes: int
    promotion_migration_seconds: float
    demotion_reclaimed_bytes: int
    demotion_transition_seconds: float
    repromotion_suppressed: int
    projected_resident_tokens: int
    resident_time_ratio: float


class HiSparseResidencyState(str, Enum):
    INACTIVE = "inactive"
    DEVICE_BUFFERED = "device_buffered"
    RESIDENT = "resident"
    PROMOTING = "promoting"
    DEMOTING = "demoting"


class HiSparseCoordinator:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Union[
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        host_to_device_ratio: int = 2,
        swap_in_block_size: int = 960,
        max_num_steps: int = 1,
        mem_pool_device_override: Optional[HiSparseDSATokenToKVPool] = None,
        dynamic_residency: bool = False,
        dynamic_residency_mode: Literal[
            "adaptive",
            "forced_resident",
            "forced_host_backed",
            "admission_once",
            "admission_window",
        ] = "adaptive",
        dynamic_residency_max_tokens: int = 32768,
        dynamic_residency_max_requests: int = 1,
        dynamic_residency_min_remaining_tokens: int = 256,
        dynamic_residency_promote_watermark: float = 0.20,
        dynamic_residency_demote_watermark: float = 0.10,
        dynamic_residency_cooldown_steps: int = 16,
        dynamic_residency_admission_window_seconds: int = 1800,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device
        self.swap_in_block_size = swap_in_block_size
        self.max_num_steps = max(max_num_steps, 1)
        self.dynamic_residency = dynamic_residency
        self.dynamic_residency_mode = dynamic_residency_mode
        self.dynamic_residency_max_tokens = dynamic_residency_max_tokens
        self.dynamic_residency_max_requests = dynamic_residency_max_requests
        self.dynamic_residency_min_remaining_tokens = (
            dynamic_residency_min_remaining_tokens
        )
        self.dynamic_residency_promote_watermark = dynamic_residency_promote_watermark
        self.dynamic_residency_demote_watermark = dynamic_residency_demote_watermark
        self.dynamic_residency_cooldown_steps = dynamic_residency_cooldown_steps
        self.dynamic_residency_admission_window_seconds = (
            dynamic_residency_admission_window_seconds
        )
        self.debug_validate_swap_in = (
            os.environ.get("SGLANG_HISPARSE_DEBUG_VALIDATE_SWAP_IN", "0") == "1"
        )
        self.debug_validate_generated_kv = (
            os.environ.get("SGLANG_HISPARSE_DEBUG_GENERATED_KV", "0") == "1"
        )
        self.debug_validate_lifecycle = (
            os.environ.get("SGLANG_HISPARSE_DEBUG_LIFECYCLE", "0") == "1"
        )
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        if self.is_dsv4_hisparse:
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
            page_size = self.mem_pool_device.page_size
            num_host_pages = (
                self.token_to_kv_pool_allocator.size_full // self.compress_ratio
                + page_size
                - 1
            ) // page_size
            self.mem_pool_host = DeepSeekV4PagedHostPool(
                pool_name="dsv4_hisparse_c4",
                device_buffers=self.mem_pool_device.kv_buffer,
                item_bytes=self.mem_pool_device.bytes_per_page_padded,
                num_host_pages=num_host_pages,
                slot_page_size=page_size,
                layout="layer_first",
            )
            self.item_size_bytes = (
                self.mem_pool_device.kv_cache_total_dim
                * self.mem_pool_device.store_dtype.itemsize
            )
        else:
            assert isinstance(
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            # Target and draft workers share the logical allocator/mapping, but
            # own different physical KV tensors. The allocator points at the
            # target pool, so a draft coordinator must bind to the draft pool.
            # Otherwise PD admission copies draft KV into the target pool and
            # later draft lookups silently read the wrong tensor.
            self.mem_pool_device: HiSparseDSATokenToKVPool = (
                mem_pool_device_override
                if mem_pool_device_override is not None
                else self.token_to_kv_pool_allocator.get_kvcache()
            )
            assert isinstance(self.mem_pool_device, HiSparseDSATokenToKVPool)
            assert (
                self.mem_pool_device.page_size
                == self.token_to_kv_pool_allocator.page_size
            )
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=self.mem_pool_device.page_size,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size
        self.page_size = self.mem_pool_device.page_size

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        max_compressed_context_len = (
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_req_slots, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        self.req_device_buffer_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_compressed_context_len + self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.req_to_host_pool_allocated_len = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.active_hisparse_reqs = {}
        # Residency is physical-pool state and therefore belongs to a
        # coordinator, not to Req. Target and MTP draft coordinators share the
        # same Req object but own different KV tensors.
        self._residency_states = {}
        self._last_residency_transition_step = {}
        # admission_once is a per-request, one-way policy. Once a request has
        # reached RESIDENT and is later demoted for pressure, it must remain
        # host-backed until the request lifecycle ends. These sets are local to
        # the canonical physical-slot owner and are cleared with the request.
        self._ever_resident_requests = set()
        self._pressure_demoted_requests = set()
        self._repromotion_suppression_reported = set()
        # admission_window is deliberately worker-scoped rather than tied to a
        # request slot.  A resident finishing or being demoted must not make a
        # different request eligible for runtime promotion while EAGLE may
        # still own speculative logical tail slots.  The next lease can only
        # be acquired by a new request at direct admission after the window.
        self._admission_window_next_allowed_at = 0.0
        self._admission_window_owner_req_idx: Optional[int] = None
        self._decode_step = 0
        self._promotion_count = 0
        self._demotion_count = 0
        self._promotion_failure_count = 0
        self._promotion_migration_acts: List[HiSparseMigrationAct] = []
        self._promotion_migrated_bytes = 0
        self._promotion_migration_seconds = 0.0
        self._demotion_reclaimed_bytes = 0
        self._demotion_transition_seconds = 0.0
        self._repromotion_suppressed_count = 0
        self._resident_request_steps = 0
        self._active_request_steps = 0
        self._device_slot_mirrors: List[HiSparseCoordinator] = []
        self.token_to_kv_pool_allocator.set_demote_until_hisparse_available(
            self.demote_until_hisparse_available
        )
        self.token_to_kv_pool_allocator.set_schedulable_hisparse_available(
            self.schedulable_hisparse_available
        )

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        self._device_buffer_arange_i32 = torch.arange(
            self.padded_buffer_size, dtype=torch.int32, device=device
        )

        # Pre-allocated output buffer for swap_in_selected_pages (CUDA-graph safe)
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots * self.max_num_steps, self.top_k),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        self._skip_first_backup = [False] * max_num_req_slots
        self._pending_draft_extend_backup = None
        self._debug_pending_draft_extend = None
        self._debug_last_generated_kv_bucket = -1
        # Request-local C4 position up to which complete physical pages have
        # been made durable in host memory and returned to the allocator.  PD
        # reserves the whole host row up front, so its allocated length cannot
        # serve as this written/retired watermark.
        self._req_c4_retired_len = {}
        self._req_c4_written_len = {}
        # In PD+speculative decode, target and draft buffers are registered
        # against one page-index vector. The target coordinator owns that
        # logical slot namespace; the draft coordinator mirrors request maps
        # and must not independently allocate or free those slot ids.
        self._host_slot_owner = self
        # Target and draft physical pools intentionally use the same numerical
        # device-slot namespace. Only the target coordinator allocates/frees
        # those numbers; draft coordinators mirror them into their own pools.
        self._device_slot_owner = self

        logger.info(
            "HiSparse dynamic residency: enabled=%s mode=%s max_tokens=%d "
            "max_requests=%d min_remaining_tokens=%d promote_watermark=%.2f "
            "demote_watermark=%.2f cooldown_steps=%d admission_window_seconds=%d",
            self.dynamic_residency,
            self.dynamic_residency_mode,
            self.dynamic_residency_max_tokens,
            self.dynamic_residency_max_requests,
            self.dynamic_residency_min_remaining_tokens,
            self.dynamic_residency_promote_watermark,
            self.dynamic_residency_demote_watermark,
            self.dynamic_residency_cooldown_steps,
            self.dynamic_residency_admission_window_seconds,
        )

    def register_device_slot_mirror(self, mirror: "HiSparseCoordinator") -> None:
        if mirror is self or mirror in self._device_slot_mirrors:
            return
        if mirror.page_size != self.page_size:
            raise RuntimeError(
                "Cannot register HiSparse mirror with different page size: "
                f"owner={self.page_size}, mirror={mirror.page_size}"
            )
        self._device_slot_mirrors.append(mirror)
        mirror._device_slot_owner = self
        # Target and draft may share one allocator. Draft construction runs
        # later and otherwise overwrites its weak callbacks with a coordinator
        # that is not allowed to initiate shared-slot demotion.
        self.token_to_kv_pool_allocator.set_demote_until_hisparse_available(
            self.demote_until_hisparse_available
        )
        self.token_to_kv_pool_allocator.set_schedulable_hisparse_available(
            self.schedulable_hisparse_available
        )

    def _state(self, req_pool_idx: int) -> HiSparseResidencyState:
        return self._residency_states.get(req_pool_idx, HiSparseResidencyState.INACTIVE)

    def _is_resident(self, req_pool_idx: int) -> bool:
        return self._state(req_pool_idx) == HiSparseResidencyState.RESIDENT

    def _set_residency_state(
        self,
        req_pool_idx: int,
        state: HiSparseResidencyState,
        *,
        count_transition: bool = True,
    ) -> None:
        previous = self._state(req_pool_idx)
        self._residency_states[req_pool_idx] = state
        if state == HiSparseResidencyState.RESIDENT:
            self._ever_resident_requests.add(req_pool_idx)
        if state in (
            HiSparseResidencyState.RESIDENT,
            HiSparseResidencyState.DEVICE_BUFFERED,
        ):
            self._last_residency_transition_step[req_pool_idx] = self._decode_step
        if not count_transition or previous == state:
            return
        if state == HiSparseResidencyState.RESIDENT:
            self._promotion_count += 1
        elif state == HiSparseResidencyState.DEVICE_BUFFERED and previous in (
            HiSparseResidencyState.RESIDENT,
            HiSparseResidencyState.DEMOTING,
        ):
            self._demotion_count += 1

    def _clear_residency_state(self, req_pool_idx: int) -> None:
        self._residency_states.pop(req_pool_idx, None)
        self._last_residency_transition_step.pop(req_pool_idx, None)
        self._ever_resident_requests.discard(req_pool_idx)
        self._pressure_demoted_requests.discard(req_pool_idx)
        self._repromotion_suppression_reported.discard(req_pool_idx)

    def _debug_validate_host_request_slots(
        self, req: Req, host_indices: torch.Tensor, *, stage: str
    ) -> None:
        if not self.debug_validate_lifecycle or host_indices.numel() == 0:
            return
        unique_host_indices = torch.unique(host_indices)
        if unique_host_indices.numel() != host_indices.numel():
            raise RuntimeError(
                f"HiSparse request has duplicate host slots at {stage}: "
                f"req={req.rid} req_pool_idx={req.req_pool_idx} "
                f"allocated={host_indices.numel()} "
                f"unique={unique_host_indices.numel()}"
            )

    def _debug_validate_host_allocator_after_free(
        self, req: Req, released_host_slots: int, *, stage: str
    ) -> None:
        if not self.debug_validate_lifecycle:
            return
        free_slots = self.mem_pool_host.free_slots
        release_slots = getattr(self.mem_pool_host, "release_slots", [])
        released_chunks = [
            chunk.to(dtype=torch.int64, device=free_slots.device).flatten()
            for chunk in release_slots
            if chunk.numel() > 0
        ]
        all_free_slots = (
            torch.cat([free_slots, *released_chunks]) if released_chunks else free_slots
        )
        unique_free_slots = torch.unique(all_free_slots)
        if unique_free_slots.numel() != all_free_slots.numel():
            raise RuntimeError(
                "HiSparse host allocator free-list contains duplicates: "
                f"stage={stage} req={req.rid} free={free_slots.numel()} "
                f"deferred={all_free_slots.numel() - free_slots.numel()} "
                f"unique={unique_free_slots.numel()}"
            )
        used_slots = int(self.mem_pool_host.slot_used.sum().item())
        if all_free_slots.numel() + used_slots != self.mem_pool_host.size:
            raise RuntimeError(
                "HiSparse host allocator accounting mismatch: "
                f"stage={stage} req={req.rid} free={free_slots.numel()} "
                f"deferred={all_free_slots.numel() - free_slots.numel()} "
                f"used={used_slots} size={self.mem_pool_host.size}"
            )
        logger.warning(
            "HISPARSE_LIFECYCLE_TRACE stage=%s req=%s req_pool_idx=%s "
            "released_host_slots=%d host_available=%d",
            stage,
            req.rid,
            req.req_pool_idx,
            released_host_slots,
            self.mem_pool_host.available_size(),
        )

    def _debug_device_lifecycle_snapshot(
        self, req: Req, buffer_locs: torch.Tensor, *, stage: str
    ) -> None:
        if (
            not getattr(self, "debug_validate_lifecycle", False)
            or self._device_slot_owner is not self
        ):
            return
        snapshot = self.token_to_kv_pool_allocator.debug_hisparse_ownership(
            buffer_locs
        )
        logger.warning(
            "HISPARSE_DEVICE_LIFECYCLE stage=%s req=%s req_pool_idx=%d "
            "available=%d capacity=%d free_pages=%d release_pages=%d "
            "mapping_slots=%d mapping_pages=%d extra_owner_pages=%d "
            "request_pages=%s request_claimed_pages=%s "
            "request_mapping_pages=%s",
            stage,
            req.rid,
            req.req_pool_idx,
            snapshot["available"],
            snapshot["capacity"],
            snapshot["free_pages"],
            snapshot["release_pages"],
            snapshot["mapping_slots"],
            snapshot["mapping_pages"],
            snapshot["extra_owner_pages"],
            snapshot["request_pages"],
            snapshot["request_claimed_pages"],
            snapshot["request_mapping_pages"],
        )

    def _resident_request_count(self) -> int:
        return sum(
            state == HiSparseResidencyState.RESIDENT
            for state in self._residency_states.values()
        )

    def _device_kv_payload_bytes(self, num_tokens: int) -> int:
        """Return target-plus-mirrors KV payload bytes for logical pool slots."""
        coordinators = [self, *self._device_slot_mirrors]
        return num_tokens * sum(
            coordinator.item_size_bytes * coordinator.mem_pool_device.layer_num
            for coordinator in coordinators
        )

    def _drain_completed_promotion_migrations(self) -> None:
        """Publish completed CUDA timings without synchronizing the decode path."""
        while self._promotion_migration_acts:
            act = self._promotion_migration_acts[0]
            if not act.finish_event.query():
                break
            self._promotion_migration_acts.pop(0)
            self._promotion_migrated_bytes += act.migrated_bytes
            self._promotion_migration_seconds += (
                act.start_event.elapsed_time(act.finish_event) / 1000.0
            )

    def _remaining_output_tokens(self, req: Req) -> int:
        max_new_tokens = getattr(req.sampling_params, "max_new_tokens", 0)
        return max(0, max_new_tokens - len(req.output_ids))

    def _residency_cooldown_complete(self, req_pool_idx: int) -> bool:
        last_step = self._last_residency_transition_step.get(
            req_pool_idx, -self.dynamic_residency_cooldown_steps
        )
        return self._decode_step - last_step >= self.dynamic_residency_cooldown_steps

    def _admission_window_available(self) -> bool:
        return time.monotonic() >= self._admission_window_next_allowed_at

    def _acquire_admission_window(self, req_pool_idx: int) -> None:
        if self.dynamic_residency_mode != "admission_window":
            return
        self._admission_window_owner_req_idx = req_pool_idx
        self._admission_window_next_allowed_at = time.monotonic() + float(
            self.dynamic_residency_admission_window_seconds
        )
        logger.info(
            "HiSparse admission residency window acquired: req_pool_idx=%d "
            "duration_seconds=%d",
            req_pool_idx,
            self.dynamic_residency_admission_window_seconds,
        )

    def _tp_all_true(self, local_value: bool) -> bool:
        agreed = torch.tensor(int(local_value), dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                agreed,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        return bool(agreed.item())

    def mirror_host_slots_from(
        self, owner: "HiSparseCoordinator", req_pool_idx: int
    ) -> None:
        """Mirror one request's canonical host slots from ``owner``.

        The pools hold different layer payloads but the PD wire protocol indexes
        all registered buffers with the same vector. Mirroring the mapping (as
        opposed to allocating from the draft free list) remains correct even
        when the two physical pools expose different token capacities.
        """
        if owner is self:
            return
        if self.page_size != owner.page_size:
            raise RuntimeError(
                "Cannot mirror HiSparse host slots with different page sizes: "
                f"owner={owner.page_size}, mirror={self.page_size}"
            )

        allocated_len = int(owner.req_to_host_pool_allocated_len[req_pool_idx])
        if allocated_len > self.req_to_host_pool.shape[1]:
            raise RuntimeError(
                "Draft HiSparse request map is smaller than the target mapping: "
                f"required={allocated_len}, capacity={self.req_to_host_pool.shape[1]}"
            )

        owner_indices = owner.req_to_host_pool[req_pool_idx, :allocated_len]
        if owner_indices.numel() > 0:
            max_slot = int(owner_indices.max().item())
            if max_slot >= self.mem_pool_host.size:
                raise RuntimeError(
                    "Draft HiSparse host buffer cannot address target slot: "
                    f"slot={max_slot}, capacity={self.mem_pool_host.size}"
                )

        self._host_slot_owner = owner
        self.req_to_host_pool[req_pool_idx, :] = -1
        self.req_to_host_pool[req_pool_idx, :allocated_len].copy_(owner_indices)
        self.req_to_host_pool_allocated_len[req_pool_idx] = allocated_len

    def _ensure_host_slots_for_positions(
        self,
        req_pool_indices: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return canonical host slots, extending only missing row tails.

        PD target and draft coordinators share numerical host slot ids, while
        only the target owns the allocator namespace.  Speculative acceptance
        can straddle the page-aligned preallocation boundary, so a batch can
        contain both existing and missing positions.  Extending from the
        canonical owner's allocated tail preserves existing mappings and then
        mirrors the resulting row into draft state.
        """
        if req_pool_indices.numel() != token_positions.numel():
            raise ValueError(
                "HiSparse host-slot position mismatch: "
                f"requests={req_pool_indices.numel()} "
                f"positions={token_positions.numel()}"
            )
        if token_positions.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)

        owner = self._host_slot_owner
        if owner._host_slot_owner is not owner:
            raise RuntimeError("HiSparse canonical host-slot owner is not self-owned")

        req_indices_cpu = req_pool_indices.to(dtype=torch.int64, device="cpu")
        positions_cpu = token_positions.to(dtype=torch.int64, device="cpu")
        for req_idx in torch.unique(req_indices_cpu).tolist():
            row_positions = positions_cpu[req_indices_cpu == req_idx]
            min_position = int(row_positions.min().item())
            max_position = int(row_positions.max().item())
            if min_position < 0 or max_position >= owner.req_to_host_pool.shape[1]:
                raise RuntimeError(
                    "HiSparse accepted position exceeds host request row: "
                    f"req_pool_idx={req_idx} positions=({min_position}, "
                    f"{max_position}) capacity={owner.req_to_host_pool.shape[1]}"
                )

            allocated_len = int(owner.req_to_host_pool_allocated_len[req_idx])
            existing_positions = row_positions[row_positions < allocated_len]
            if existing_positions.numel() > 0:
                existing_locs = owner.req_to_host_pool[
                    int(req_idx), existing_positions.to(owner.device)
                ]
                if torch.any(existing_locs < 0):
                    raise RuntimeError(
                        "HiSparse host request row has a hole before its "
                        f"allocated tail: req_pool_idx={req_idx} "
                        f"allocated_len={allocated_len}"
                    )

            required_len = max_position + 1
            if required_len > allocated_len:
                owner.mem_pool_host.alloc_paged_token_slots(
                    owner.req_to_host_pool,
                    owner.req_to_host_pool_allocated_len,
                    int(req_idx),
                    allocated_len,
                    required_len - allocated_len,
                )
            if self is not owner:
                self.mirror_host_slots_from(owner, int(req_idx))

        host_locs = self.req_to_host_pool[req_pool_indices, token_positions]
        if torch.any(host_locs < 0):
            raise RuntimeError("HiSparse accepted host mapping is missing after growth")
        return host_locs.to(device=self.device)

    def mirror_device_slots_from(
        self, owner: "HiSparseCoordinator", req_pool_idx: int
    ) -> None:
        """Mirror one request's numerical device slots from ``owner``.

        Target and draft KV tensors are distinct, but the speculative allocator
        exposes one shared full-to-device mapping. Both tensors must therefore
        use identical numerical slots. Allocating a second draft buffer would
        make that single mapping refer to incompatible physical locations.
        """
        if owner is self:
            return
        if self.page_size != owner.page_size:
            raise RuntimeError(
                "Cannot mirror HiSparse device slots with different page sizes: "
                f"owner={owner.page_size}, mirror={self.page_size}"
            )
        if self.padded_buffer_size != owner.padded_buffer_size:
            raise RuntimeError(
                "Cannot mirror HiSparse device slots with different buffer sizes: "
                f"owner={owner.padded_buffer_size}, mirror={self.padded_buffer_size}"
            )

        allocated_len = int(owner.req_device_buffer_size[req_pool_idx])
        if allocated_len > self.req_to_device_buffer.shape[1]:
            raise RuntimeError(
                "Draft HiSparse device map is smaller than target mapping: "
                f"required={allocated_len}, capacity={self.req_to_device_buffer.shape[1]}"
            )
        owner_indices = owner.req_to_device_buffer[req_pool_idx, :allocated_len]
        if owner_indices.numel() > 0:
            max_slot = int(owner_indices.max().item())
            addressable_slots = self._device_pool_addressable_slots()
            if max_slot >= addressable_slots:
                raise RuntimeError(
                    "Draft HiSparse device pool cannot address target slot: "
                    f"slot={max_slot}, capacity={addressable_slots}"
                )

        current_len = int(self.req_device_buffer_size[req_pool_idx])
        if current_len > allocated_len:
            raise RuntimeError(
                "Draft HiSparse device map is larger than its target owner: "
                f"draft={current_len}, target={allocated_len}"
            )
        if current_len > 0 and not torch.equal(
            self.req_to_device_buffer[req_pool_idx, :current_len],
            owner_indices[:current_len],
        ):
            raise RuntimeError(
                "Target and draft HiSparse device-slot namespaces already diverged"
            )

        self._device_slot_owner = owner
        self.req_to_device_buffer[req_pool_idx, current_len:allocated_len].copy_(
            owner_indices[current_len:allocated_len]
        )
        self.req_device_buffer_size[req_pool_idx] = allocated_len
        if owner._is_resident(req_pool_idx):
            # A resident request only needs the graph-stable speculative page;
            # its hot-buffer columns deliberately remain invalid so the swap
            # kernel selects the full-resident mapping fast path.
            self.req_device_buffer_tokens[:, req_pool_idx, :] = -1
            self.req_device_buffer_token_locs[:, req_pool_idx, :] = -1
            if allocated_len > self.device_buffer_size:
                self.req_device_buffer_token_locs[
                    :, req_pool_idx, self.device_buffer_size : allocated_len
                ] = owner_indices[self.device_buffer_size : allocated_len]
            self.lru_slots[:, req_pool_idx, :].copy_(self._lru_init)
            self._skip_first_backup[req_pool_idx] = True
            return
        # Token/LRU metadata is model-specific: the target has all target
        # layers while an EAGLE draft coordinator normally has one layer.
        # Initialize the mirrored numerical slots exactly like a local
        # alloc_device_buffer() instead of copying target-layer metadata.
        hot_end = min(allocated_len, self.device_buffer_size)
        if current_len < hot_end:
            self.req_device_buffer_tokens[:, req_pool_idx, current_len:hot_end] = (
                self._device_buffer_arange_i32[current_len:hot_end]
            )
        extra_start = max(current_len, self.device_buffer_size)
        if extra_start < allocated_len:
            self.req_device_buffer_tokens[
                :, req_pool_idx, extra_start:allocated_len
            ] = -1
        self.req_device_buffer_token_locs[
            :, req_pool_idx, current_len:allocated_len
        ] = owner_indices[current_len:allocated_len]
        self.lru_slots[:, req_pool_idx, current_len:allocated_len] = self._lru_init[
            current_len:allocated_len
        ]
        self._skip_first_backup[req_pool_idx] = True

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def destroy(self) -> None:
        # Drain in-flight transfers so the buffer is idle, then unregister it.
        # See HostKVCache.destroy for why the explicit unregister matters.
        self.write_staging_stream.synchronize()
        self.decode_backup_stream.synchronize()
        self.mem_pool_host.destroy()

    def _hisparse_device_locs(self, compressed_locs: torch.Tensor) -> torch.Tensor:
        return self.mem_pool_device.full_to_hisparse_device_index_mapping[
            compressed_locs
        ]

    def _resident_token_device_locs(
        self, req_pool_indices: torch.Tensor, compressed_positions: torch.Tensor
    ) -> torch.Tensor:
        token_positions = compressed_positions
        if self.compress_ratio != 1:
            token_positions = compressed_positions * self.compress_ratio + (
                self.compress_ratio - 1
            )
        logical_locs = self.req_to_token_pool.req_to_token[
            req_pool_indices, token_positions
        ]
        return self._hisparse_device_locs(
            self.mem_pool_device.translate_loc_from_full_to_compressed(logical_locs)
        )

    def _free_stale_hisparse_mapping(
        self, compressed_locs: torch.Tensor, new_device_locs: torch.Tensor
    ) -> None:
        """Free stale ROCm-only temporary slots before remapping token locations.

        The decode remap can create a temporary hisparse device slot per new
        token on ROCm. If that slot is replaced by a device-buffer slot and left
        allocated, later swap-in lookups can see stale mappings. CUDA consumes
        top_k_device_locs directly, so the old mapping is harmless there.
        """
        if not _is_hip or self.is_dsv4_hisparse:
            return

        stale_locs = self._hisparse_device_locs(compressed_locs)
        stale_locs = stale_locs[(stale_locs > 0) & (stale_locs != new_device_locs)]
        if stale_locs.numel() > 0:
            self.token_to_kv_pool_allocator.free_hisparse_indices(stale_locs)

    def _device_buffer_alloc_size(self, kv_allocated_len: int) -> int:
        page_size = self.mem_pool_device.page_size
        alloc_size = min(
            ((self.host_token_len(kv_allocated_len) + page_size - 1) // page_size)
            * page_size,
            self.device_buffer_size,
        )
        return (
            self.padded_buffer_size
            if alloc_size == self.device_buffer_size
            else alloc_size
        )

    def demote_until_hisparse_available(
        self,
        need_tokens: int,
    ) -> bool:
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        if allocator.available_size() >= need_tokens:
            return True

        candidates = [
            req
            for req_idx, req in self.active_hisparse_reqs.items()
            if self._is_resident(req_idx)
            and self.host_token_len(req.kv.kv_allocated_len)
            > self._device_buffer_alloc_size(req.kv.kv_allocated_len)
        ]
        candidates.sort(key=lambda req: req.kv.kv_allocated_len, reverse=True)
        if candidates:
            self.wait_for_pending_backup()
        for req in candidates:
            self._demote_resident_request(req)
            if allocator.available_size() >= need_tokens:
                return True

        return allocator.available_size() >= need_tokens

    def schedulable_hisparse_available(self) -> int:
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        available = allocator.available_size()
        reclaimable = 0
        page_size = self.mem_pool_device.page_size
        for req_idx, req in self.active_hisparse_reqs.items():
            if not self._is_resident(req_idx):
                continue
            host_len = self.host_token_len(req.kv.kv_allocated_len)
            alloc_size = self._device_buffer_alloc_size(req.kv.kv_allocated_len)
            current_size = ((host_len + page_size - 1) // page_size) * page_size
            reclaimable += max(0, current_size - alloc_size)
        return max(0, available + reclaimable - page_size)

    @staticmethod
    def _has_free_hisparse_pages(allocator, num_new_pages: int) -> bool:
        if allocator.need_sort and num_new_pages > len(allocator.free_pages):
            allocator.merge_and_sort_free()
        return num_new_pages <= len(allocator.free_pages)

    def _alloc_resident_last_locs(
        self,
        resident_positions: List[int],
        active_seq_lens: torch.Tensor,
        resident_seq_lens_cpu: torch.Tensor,
        active_out_cache_loc: torch.Tensor,
        active_req_pool_indices: torch.Tensor,
    ) -> None:
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        if len(resident_positions) == len(active_seq_lens):
            seq_lens = active_seq_lens
            req_pool_indices = active_req_pool_indices
            out_cache_loc = active_out_cache_loc
        else:
            pos_tensor = torch.tensor(
                resident_positions, dtype=torch.int64, device=self.device
            )
            seq_lens = active_seq_lens[pos_tensor]
            req_pool_indices = active_req_pool_indices[pos_tensor]
            out_cache_loc = active_out_cache_loc[pos_tensor]

        prev_compressed_pos = seq_lens - 2
        has_previous_token = int(resident_seq_lens_cpu.min()) > 1
        prev_device_locs = self._resident_token_device_locs(
            req_pool_indices,
            (
                prev_compressed_pos
                if has_previous_token
                else prev_compressed_pos.clamp_min(0)
            ),
        )
        if not has_previous_token:
            prev_device_locs = torch.where(
                prev_compressed_pos >= 0,
                prev_device_locs,
                torch.full_like(prev_device_locs, -1),
            )
        hisparse_indices = allocator.alloc_decode(
            seq_lens,
            resident_seq_lens_cpu,
            prev_device_locs,
        )
        if hisparse_indices is None:
            raise RuntimeError("HiSparse dynamic decode allocation failed")

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            out_cache_loc
        )
        self._free_stale_hisparse_mapping(compressed_locs, hisparse_indices)
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            hisparse_indices
        )

    def get_token_stats(self) -> HiSparseTokenStats:
        self._drain_completed_promotion_migrations()
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        projected_resident_tokens = sum(
            self._projected_resident_alloc_size(req, req.kv.kv_allocated_len)
            for req_idx, req in self.active_hisparse_reqs.items()
            if self._is_resident(req_idx)
        )
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
            resident_requests=self._resident_request_count(),
            device_buffered_requests=sum(
                state == HiSparseResidencyState.DEVICE_BUFFERED
                for state in self._residency_states.values()
            ),
            promotions=self._promotion_count,
            demotions=self._demotion_count,
            promotion_failures=self._promotion_failure_count,
            promotion_migrated_bytes=self._promotion_migrated_bytes,
            promotion_migration_seconds=self._promotion_migration_seconds,
            demotion_reclaimed_bytes=self._demotion_reclaimed_bytes,
            demotion_transition_seconds=self._demotion_transition_seconds,
            repromotion_suppressed=self._repromotion_suppressed_count,
            projected_resident_tokens=projected_resident_tokens,
            resident_time_ratio=(
                self._resident_request_steps / self._active_request_steps
                if self._active_request_steps > 0
                else 0.0
            ),
        )

    def admit_request_into_staging(self, req: Req) -> None:
        if self._device_slot_owner is not self:
            raise RuntimeError(
                "Only the canonical HiSparse slot owner may admit staging requests"
            )
        req.hisparse_staging = True
        self._initialize_dsv4_retire_watermark(req)
        self._set_residency_state(
            req.req_pool_idx,
            HiSparseResidencyState.INACTIVE,
            count_transition=False,
        )

        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.extend_range.end
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            0,
            prefill_len,
        )

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def admit_request_direct(
        self,
        req: Req,
        device_slot_owner: Optional["HiSparseCoordinator"] = None,
    ) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging backup entirely. If the HiSparse device pool has room,
        promotes the complete host KV into it and admits the request as
        resident. Otherwise, allocates a small device buffer for decode-time
        swap-in.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.
        """
        if device_slot_owner is not None and device_slot_owner is not self:
            self._admit_request_from_owner(req, device_slot_owner)
            return

        self._initialize_dsv4_retire_watermark(req)

        if self.debug_validate_lifecycle:
            req_idx = req.req_pool_idx
            if req_idx in self.active_hisparse_reqs:
                raise RuntimeError(
                    f"HiSparse reused active request slot: req_pool_idx={req_idx}"
                )
            if self._state(req_idx) != HiSparseResidencyState.INACTIVE:
                raise RuntimeError(
                    "HiSparse reused non-inactive residency slot: "
                    f"req_pool_idx={req_idx} state={self._state(req_idx).value}"
                )
            if int(self.req_device_buffer_size[req_idx]) != 0:
                raise RuntimeError(
                    "HiSparse reused request slot with a live device buffer: "
                    f"req_pool_idx={req_idx} size={int(self.req_device_buffer_size[req_idx])}"
                )
            host_len_for_trace = int(self.req_to_host_pool_allocated_len[req_idx])
            host_locs_for_trace = self.req_to_host_pool[req_idx, :host_len_for_trace]
            unique_host_locs = torch.unique(host_locs_for_trace)
            if unique_host_locs.numel() != host_locs_for_trace.numel():
                raise RuntimeError(
                    "HiSparse admitted a request with duplicate host slots: "
                    f"req={req.rid} req_pool_idx={req_idx} "
                    f"allocated={host_locs_for_trace.numel()} "
                    f"unique={unique_host_locs.numel()}"
                )
            logger.warning(
                "HISPARSE_LIFECYCLE_TRACE stage=admit req=%s req_pool_idx=%d "
                "host_slots=%d host_min=%d host_max=%d host_contiguous=%s "
                "host_available=%d",
                req.rid,
                req_idx,
                host_locs_for_trace.numel(),
                (
                    int(host_locs_for_trace.min().item())
                    if host_locs_for_trace.numel()
                    else -1
                ),
                (
                    int(host_locs_for_trace.max().item())
                    if host_locs_for_trace.numel()
                    else -1
                ),
                bool(
                    host_locs_for_trace.numel() <= 1
                    or torch.all(
                        host_locs_for_trace[1:] - host_locs_for_trace[:-1] == 1
                    )
                ),
                self.mem_pool_host.available_size(),
            )

        host_len = self.host_token_len(req.kv.kv_allocated_len)
        if not self._try_promote_from_host(
            req, sync_mirrors=False, admission_boundary=True
        ):
            buffer_size = self._device_buffer_alloc_size(req.kv.kv_allocated_len)
            if not self.demote_until_hisparse_available(buffer_size):
                raise RuntimeError("HiSparse direct admission allocation failed")
            self.alloc_device_buffer(req)

            if host_len <= self.device_buffer_size:
                # Short sequences take the kernel fast path, so preload the
                # complete host KV into their device buffer.
                self._preload_to_device_buffer(req)
            else:
                # The direct-path buffer starts empty. Every top-k lookup is a
                # miss until the swap-in kernel populates it.
                self.req_device_buffer_tokens[
                    :, req.req_pool_idx, : self.device_buffer_size
                ] = -1

        req.hisparse_staging = False
        self._skip_first_backup[req.req_pool_idx] = True
        self.active_hisparse_reqs[req.req_pool_idx] = req
        logger.debug(
            "HiSparse: admitting request %s directly (%s)",
            req.rid,
            self._state(req.req_pool_idx).value,
        )

    def _admit_request_from_owner(self, req: Req, owner: "HiSparseCoordinator") -> None:
        """Mirror the target coordinator's admission decision for MTP draft KV."""
        self._device_slot_owner = owner
        self._initialize_dsv4_retire_watermark(req)
        owner_state = owner._state(req.req_pool_idx)
        host_len = self.host_token_len(req.kv.kv_allocated_len)
        if owner_state == HiSparseResidencyState.RESIDENT:
            logical_locs = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.kv.kv_allocated_len
            ]
            compressed_locs = (
                self.mem_pool_device.translate_loc_from_full_to_compressed(logical_locs)
            )
            device_locs = owner.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ]
            self._validate_mirror_device_locs(device_locs)
            self._load_host_kv(req, device_locs, host_len=host_len)
            self._set_residency_state(
                req.req_pool_idx,
                HiSparseResidencyState.RESIDENT,
                count_transition=False,
            )
        elif owner_state == HiSparseResidencyState.DEVICE_BUFFERED:
            self.mirror_device_slots_from(owner, req.req_pool_idx)
            if host_len <= self.device_buffer_size:
                self._preload_to_device_buffer(req)
            else:
                self.req_device_buffer_tokens[
                    :, req.req_pool_idx, : self.device_buffer_size
                ] = -1
            self._set_residency_state(
                req.req_pool_idx,
                HiSparseResidencyState.DEVICE_BUFFERED,
                count_transition=False,
            )
        else:
            raise RuntimeError(
                "HiSparse owner has no committed admission state: "
                f"req={req.rid}, state={owner_state.value}"
            )

        req.hisparse_staging = False
        self._skip_first_backup[req.req_pool_idx] = True
        self.active_hisparse_reqs[req.req_pool_idx] = req

    def _can_promote_by_policy(
        self,
        req: Req,
        alloc_size: int,
        logical_len: int,
        *,
        admission_boundary: bool,
    ) -> bool:
        if not self.dynamic_residency:
            return False
        if self.dynamic_residency_mode == "forced_host_backed":
            return False
        if self.dynamic_residency_mode == "admission_window" and (
            not admission_boundary or not self._admission_window_available()
        ):
            if req.req_pool_idx not in self._repromotion_suppression_reported:
                self._repromotion_suppression_reported.add(req.req_pool_idx)
                self._repromotion_suppressed_count += 1
            return False
        if (
            self.dynamic_residency_mode == "admission_once"
            and req.req_pool_idx in self._ever_resident_requests
        ):
            if req.req_pool_idx not in self._repromotion_suppression_reported:
                self._repromotion_suppression_reported.add(req.req_pool_idx)
                self._repromotion_suppressed_count += 1
            return False
        projected_logical_len = (
            logical_len + self._remaining_output_tokens(req) + self.max_num_steps
        )
        if (
            self.dynamic_residency_mode != "forced_resident"
            and projected_logical_len > self.dynamic_residency_max_tokens
        ):
            return False
        if (
            self.dynamic_residency_mode != "forced_resident"
            and self._remaining_output_tokens(req)
            < self.dynamic_residency_min_remaining_tokens
        ):
            return False
        if self._resident_request_count() >= self.dynamic_residency_max_requests:
            return False
        if (
            self.dynamic_residency_mode != "forced_resident"
            and not self._residency_cooldown_complete(req.req_pool_idx)
        ):
            return False
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        current_buffer_size = int(self.req_device_buffer_size[req.req_pool_idx])
        resident_spec_page = self.page_size if self.max_num_steps > 1 else 0
        projected_alloc_size = max(
            alloc_size,
            self._projected_resident_alloc_size(req, logical_len),
        )
        projected_available = (
            allocator.available_size()
            - projected_alloc_size
            - resident_spec_page
            + current_buffer_size
        )
        return projected_available >= int(
            allocator.size * self.dynamic_residency_promote_watermark
        )

    def _projected_resident_alloc_size(self, req: Req, logical_len: int) -> int:
        """Resident slots needed through the request's declared output budget."""
        projected_logical_len = (
            logical_len + self._remaining_output_tokens(req) + self.max_num_steps
        )
        projected_host_len = self.host_token_len(projected_logical_len)
        return (
            (projected_host_len + self.page_size - 1) // self.page_size
        ) * self.page_size

    def _try_promote_from_host(
        self,
        req: Req,
        *,
        logical_len: Optional[int] = None,
        sync_mirrors: bool = True,
        admission_boundary: bool = False,
    ) -> bool:
        """Transactionally promote host-backed target and draft KV into HBM."""
        if self._device_slot_owner is not self:
            return False
        was_device_buffered = (
            self._state(req.req_pool_idx) == HiSparseResidencyState.DEVICE_BUFFERED
        )
        logical_len = req.kv.kv_allocated_len if logical_len is None else logical_len
        logical_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :logical_len
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            logical_locs
        )
        host_len = self.host_token_len(logical_len)
        assert len(compressed_locs) == host_len

        page_size = self.mem_pool_device.page_size
        alloc_size = ((host_len + page_size - 1) // page_size) * page_size
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        num_pages = alloc_size // page_size
        can_promote = self._can_promote_by_policy(
            req,
            alloc_size,
            logical_len,
            admission_boundary=admission_boundary,
        ) and self._has_free_hisparse_pages(allocator, num_pages)
        if not self._tp_all_true(can_promote):
            return False

        self._set_residency_state(
            req.req_pool_idx,
            HiSparseResidencyState.PROMOTING,
            count_transition=False,
        )
        device_locs = allocator.alloc(alloc_size)
        if device_locs is None:
            self._promotion_failure_count += 1
            self._set_residency_state(
                req.req_pool_idx,
                HiSparseResidencyState.DEVICE_BUFFERED,
                count_transition=False,
            )
            return False
        mapped_device_locs = device_locs[:host_len]
        migration_start = None
        migration_finish = None
        if was_device_buffered:
            migration_start = device_module.Event(enable_timing=True)
            migration_finish = device_module.Event(enable_timing=True)
            migration_start.record()

        try:
            self.wait_for_pending_backup()
            self._load_host_kv(req, mapped_device_locs, host_len=host_len)
            if sync_mirrors:
                for mirror in self._device_slot_mirrors:
                    mirror.wait_for_pending_backup()
                    mirror._validate_mirror_device_locs(mapped_device_locs)
                    mirror._load_host_kv(req, mapped_device_locs, host_len=host_len)
            if migration_finish is not None:
                migration_finish.record()
        except Exception:
            self.token_to_kv_pool_allocator.free_hisparse_indices(device_locs)
            self._promotion_failure_count += 1
            self._set_residency_state(
                req.req_pool_idx,
                HiSparseResidencyState.DEVICE_BUFFERED,
                count_transition=False,
            )
            raise

        if migration_start is not None and migration_finish is not None:
            self._promotion_migration_acts.append(
                HiSparseMigrationAct(
                    start_event=migration_start,
                    finish_event=migration_finish,
                    migrated_bytes=self._device_kv_payload_bytes(host_len),
                )
            )

        # Publish only after every physical target/draft tensor is populated.
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            mapped_device_locs
        )
        self._free_device_buffer_after_promotion(req, free_physical=True)
        if sync_mirrors:
            for mirror in self._device_slot_mirrors:
                mirror._free_device_buffer_after_promotion(req, free_physical=False)
        # Initial PD admission may choose the resident path directly. Keep the
        # cumulative promotion metric reserved for runtime swap-to-resident
        # transitions so it can be used to reason about migration payback.
        self._set_residency_state(
            req.req_pool_idx,
            HiSparseResidencyState.RESIDENT,
            count_transition=was_device_buffered,
        )
        if admission_boundary:
            self._acquire_admission_window(req.req_pool_idx)
        if sync_mirrors:
            for mirror in self._device_slot_mirrors:
                mirror_was_device_buffered = (
                    mirror._state(req.req_pool_idx)
                    == HiSparseResidencyState.DEVICE_BUFFERED
                )
                mirror._set_residency_state(
                    req.req_pool_idx,
                    HiSparseResidencyState.RESIDENT,
                    count_transition=mirror_was_device_buffered,
                )
        return True

    def _device_pool_addressable_slots(self) -> int:
        """Return the numerical slot namespace backed by the device KV pool.

        Paged allocators reserve page 0 for padded writes and allocate real pages
        from ``1..size // page_size``.  The corresponding KV tensors therefore
        contain ``size + page_size`` slots.  Using ``size`` as the mirror bound
        incorrectly rejects the allocator's final real page.
        """
        return int(self.mem_pool_device.size + self.mem_pool_device.page_size)

    def _validate_mirror_device_locs(self, device_locs: torch.Tensor) -> None:
        if device_locs.numel() == 0:
            return
        max_slot = int(device_locs.max().item())
        addressable_slots = self._device_pool_addressable_slots()
        if max_slot >= addressable_slots:
            raise RuntimeError(
                "HiSparse mirror pool cannot address owner slot: "
                f"slot={max_slot}, capacity={addressable_slots}"
            )

    def _free_device_buffer_after_promotion(
        self, req: Req, *, free_physical: bool
    ) -> None:
        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        if free_physical and current_cap > 0:
            buffer_locs = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
            buffer_locs = buffer_locs[buffer_locs > 0]
            if buffer_locs.numel() > 0:
                # Promotion replaces the committed prefix with newly allocated
                # resident pages, but speculative decoding can leave logical
                # over-allocation beyond ``logical_len`` pointing at the old
                # device buffer.  Detach those stale references before the
                # buffer pages are returned.  Otherwise release_kv_cache()
                # follows the stale mapping at request finish and frees the
                # same physical page a second time.
                self._detach_request_mappings_to_physical_pages(req, buffer_locs)
                self.token_to_kv_pool_allocator.release_hisparse_ownership(
                    mapping_indices=torch.empty(
                        0, dtype=torch.int64, device=buffer_locs.device
                    ),
                    extra_owned_coordinates=buffer_locs,
                )
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)

    def _detach_request_mappings_to_physical_pages(
        self, req: Req, physical_locs: torch.Tensor
    ) -> None:
        """Clear request mappings that still reference coordinator-owned pages."""
        physical_locs = physical_locs[physical_locs > 0]
        if physical_locs.numel() == 0 or req.kv.kv_allocated_len <= 0:
            return
        logical_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.kv.kv_allocated_len
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            logical_locs
        )
        mapping = self.mem_pool_device.full_to_hisparse_device_index_mapping
        mapped_locs = mapping[compressed_locs]
        physical_pages = torch.unique(physical_locs // self.page_size)
        stale = (mapped_locs > 0) & torch.isin(
            mapped_locs // self.page_size, physical_pages
        )
        if torch.any(stale):
            mapping[compressed_locs[stale]] = 0

    def _free_resident_spec_page(self, req: Req, *, free_physical: bool) -> None:
        """Release the resident request's graph-stable speculative page."""
        req_idx = req.req_pool_idx
        current_cap = int(self.req_device_buffer_size[req_idx])
        if current_cap == 0:
            return
        if current_cap != self.padded_buffer_size:
            raise RuntimeError(
                "Resident HiSparse request has an invalid speculative-page "
                f"capacity: req={req.rid}, current={current_cap}, "
                f"expected={self.padded_buffer_size}"
            )
        page_locs = self.req_to_device_buffer[
            req_idx, self.device_buffer_size : self.padded_buffer_size
        ]
        if free_physical:
            page_locs = page_locs[page_locs > 0]
            if page_locs.numel() > 0:
                # Verify slots are deliberately over-allocated and can still
                # point at this graph-stable page when residency changes or a
                # request finishes.  Detach those logical references before
                # returning the physical page; otherwise alloc_device_buffer()
                # or the generic allocator.free() path observes the stale
                # mapping and frees the same page a second time.
                self._detach_request_mappings_to_physical_pages(req, page_locs)
                self.token_to_kv_pool_allocator.release_hisparse_ownership(
                    mapping_indices=torch.empty(
                        0, dtype=torch.int64, device=page_locs.device
                    ),
                    extra_owned_coordinates=page_locs,
                )
        self.req_to_device_buffer[req_idx, :] = 0
        self.req_device_buffer_size[req_idx] = 0
        self.req_device_buffer_tokens[:, req_idx, :] = -1
        self.req_device_buffer_token_locs[:, req_idx, :] = -1
        self.lru_slots[:, req_idx, :].copy_(self._lru_init)

    def host_token_len(self, kv_allocated_len: int) -> int:
        if self.is_dsv4_hisparse:
            return kv_allocated_len // self.compress_ratio
        return kv_allocated_len

    def _initialize_dsv4_retire_watermark(self, req: Req) -> None:
        if not self.is_dsv4_hisparse:
            return
        # The complete prompt C4 pages were transferred to host by PD/staging.
        # Keep the final partial page eligible: decode may fill its suffix on a
        # newly allocated physical page that must later be backed up and freed.
        transferred_len = min(
            int(getattr(req, "kv_committed_len", 0)),
            int(getattr(getattr(req, "kv", None), "kv_allocated_len", 0)),
        )
        committed_c4 = transferred_len // self.compress_ratio
        self._req_c4_written_len[req.req_pool_idx] = committed_c4
        self._req_c4_retired_len[req.req_pool_idx] = (
            committed_c4 // self.page_size * self.page_size
        )

    def retire_committed_dsv4_pages(
        self,
        reqs: List[Req],
        mirror: Optional["HiSparseCoordinator"] = None,
    ) -> None:
        """Persist and release fully committed DSV4 C4 allocation pages.

        EAGLE reserves a logical full-KV page before verify.  The DSV4 C4
        allocator consequently owns the corresponding physical page even when
        only a few tokens have committed.  At the start of the next EAGLE
        round, ``kv_committed_len`` is stable and no new reserve exists yet.
        This method uses that boundary to back up complete request-local C4
        pages in both target and draft pools, then clears every logical alias
        and returns the canonical target page exactly once.
        """
        if not self.is_dsv4_hisparse:
            return
        if self._device_slot_owner is not self:
            raise RuntimeError(
                "Only the canonical DSV4 HiSparse owner may retire C4 pages"
            )
        if mirror is not None:
            if not mirror.is_dsv4_hisparse:
                raise RuntimeError(
                    "DSV4 HiSparse target cannot retire pages with a non-DSV4 mirror"
                )
            if mirror._device_slot_owner is not self:
                raise RuntimeError(
                    "DSV4 HiSparse draft does not mirror the canonical target owner"
                )

        mapping = self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        for req in reqs:
            req_idx = int(req.req_pool_idx)
            if self._is_resident(req_idx):
                # Resident mappings are the request's durable device copy.  The
                # production calibration keeps dynamic residency disabled; a
                # later demotion owns their host transition.
                continue
            if req_idx not in self._req_c4_retired_len:
                self._initialize_dsv4_retire_watermark(req)

            start = int(self._req_c4_retired_len[req_idx])
            written = int(self._req_c4_written_len[req_idx])
            committed_c4 = int(req.kv_committed_len) // self.compress_ratio
            retire_end = committed_c4 // self.page_size * self.page_size
            retirement_plan = []
            while start < retire_end:
                end = start + self.page_size
                full_positions = torch.arange(
                    start, end, dtype=torch.int64, device=self.device
                ) * self.compress_ratio + (self.compress_ratio - 1)
                if int(full_positions[-1]) >= req.kv.kv_allocated_len:
                    raise RuntimeError(
                        "Committed DSV4 C4 page exceeds the request allocation: "
                        f"req={req.rid}, page=({start}, {end}), "
                        f"committed={req.kv_committed_len}, "
                        f"allocated={req.kv.kv_allocated_len}"
                    )
                full_locs = self.req_to_token_pool.req_to_token[
                    req_idx, full_positions
                ].to(torch.int64)
                compressed_locs = (
                    self.mem_pool_device.translate_loc_from_full_to_compressed(
                        full_locs
                    )
                )
                if compressed_locs.numel() != self.page_size:
                    raise RuntimeError(
                        "DSV4 C4 retirement lost request-local alignment: "
                        f"req={req.rid}, expected={self.page_size}, "
                        f"actual={compressed_locs.numel()}"
                    )

                device_locs = mapping[compressed_locs]
                mapped_mask = device_locs > 0
                c4_positions = torch.arange(
                    start, end, dtype=torch.int64, device=self.device
                )
                missing_unwritten = (~mapped_mask) & (c4_positions >= written)
                if torch.any(missing_unwritten):
                    raise RuntimeError(
                        "DSV4 C4 committed payload has neither host nor device "
                        f"ownership: req={req.rid}, "
                        f"positions={c4_positions[missing_unwritten].tolist()}, "
                        f"written={written}"
                    )
                if torch.any(mapped_mask):
                    mapped_locs = device_locs[mapped_mask].to(torch.int64)
                    page_ids = torch.unique(mapped_locs // self.page_size)

                    # A paged allocation gives one request-local C4 page
                    # exclusive physical-page ownership.  Verify that EAGLE
                    # did not leave an alias outside the now-committed page
                    # before making the page reusable.
                    positive = mapping > 0
                    aliases = torch.nonzero(
                        positive
                        & torch.isin(
                            torch.div(mapping, self.page_size, rounding_mode="floor"),
                            page_ids,
                        ),
                        as_tuple=False,
                    ).flatten()
                    if torch.any(~torch.isin(aliases, compressed_locs)):
                        raise RuntimeError(
                            "DSV4 C4 physical page still has an uncommitted alias: "
                            f"req={req.rid}, c4_page=({start}, {end}), "
                            f"physical_pages={page_ids.tolist()}"
                        )

                    host_locs = self.req_to_host_pool[req_idx, start:end][mapped_mask]
                    if torch.any(host_locs < 0):
                        raise RuntimeError(
                            "DSV4 C4 committed page has no host destination: "
                            f"req={req.rid}, c4_page=({start}, {end})"
                        )
                    if mirror is not None:
                        mirror._validate_mirror_device_locs(mapped_locs)
                    retirement_plan.append((host_locs, mapped_locs, page_ids))

                start = end
                written = end

            # Validate every page before starting DMA or freeing any owner. A
            # corrupt future alias therefore cannot partially advance a request
            # and leave target/draft watermarks at different boundaries.
            for host_locs, mapped_locs, page_ids in retirement_plan:
                self._backup_device_locs_to_host(host_locs, mapped_locs)
                if mirror is not None:
                    mirror._backup_device_locs_to_host(host_locs, mapped_locs)

                # Both physical payloads are durable.  The target owns the
                # shared numerical mapping and allocator free transaction.
                self.token_to_kv_pool_allocator.release_hisparse_mapped_pages(page_ids)

            self._req_c4_written_len[req_idx] = written
            self._req_c4_retired_len[req_idx] = start
            if mirror is not None:
                mirror._req_c4_written_len[req_idx] = written
                mirror._req_c4_retired_len[req_idx] = start

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer."""
        n = self.host_token_len(req.kv.kv_allocated_len)
        device_locs = self.req_to_device_buffer[req.req_pool_idx, :n]
        self._load_host_kv(req, device_locs)

    def _load_host_kv(
        self,
        req: Req,
        device_locs: torch.Tensor,
        *,
        host_len: Optional[int] = None,
    ) -> None:
        host_len = (
            self.host_token_len(req.kv.kv_allocated_len)
            if host_len is None
            else host_len
        )
        host_locs = self.req_to_host_pool[req.req_pool_idx, :host_len]
        if self.debug_validate_lifecycle:
            host_kernel_locs = self.mem_pool_host.dcp_kernel_indices(host_locs)
            device_kernel_locs = self.mem_pool_host.dcp_kernel_indices(device_locs)
            if host_kernel_locs.numel() != device_kernel_locs.numel():
                raise RuntimeError(
                    "HiSparse host/device preload length mismatch: "
                    f"req={req.rid} host={host_kernel_locs.numel()} "
                    f"device={device_kernel_locs.numel()}"
                )
            if self.mem_pool_host.layout == "layer_first":
                host_rows = int(self.mem_pool_host.kv_buffer.shape[1])
            else:
                host_rows = int(self.mem_pool_host.kv_buffer.shape[0])
            device_rows = int(self.mem_pool_device.kv_buffer[0].shape[0])
            host_min = (
                int(host_kernel_locs.min().item()) if host_kernel_locs.numel() else -1
            )
            host_max = (
                int(host_kernel_locs.max().item()) if host_kernel_locs.numel() else -1
            )
            device_min = (
                int(device_kernel_locs.min().item())
                if device_kernel_locs.numel()
                else -1
            )
            device_max = (
                int(device_kernel_locs.max().item())
                if device_kernel_locs.numel()
                else -1
            )
            if host_min < 0 or host_max >= host_rows:
                raise RuntimeError(
                    "HiSparse host preload index is out of bounds: "
                    f"req={req.rid} range=({host_min}, {host_max}) "
                    f"capacity={host_rows}"
                )
            if device_min < 0 or device_max >= device_rows:
                raise RuntimeError(
                    "HiSparse device preload index is out of bounds: "
                    f"req={req.rid} range=({device_min}, {device_max}) "
                    f"capacity={device_rows}"
                )
            logger.warning(
                "HISPARSE_LOAD_TRACE req=%s req_pool_idx=%d host_len=%d "
                "host_range=(%d,%d)/%d device_range=(%d,%d)/%d "
                "layers=%d kv_dim=%d",
                req.rid,
                req.req_pool_idx,
                host_len,
                host_min,
                host_max,
                host_rows,
                device_min,
                device_max,
                device_rows,
                self.mem_pool_device.layer_num,
                self.mem_pool_device.kv_cache_dim,
            )
        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(self, req: Req) -> None:
        was_resident = self._is_resident(req.req_pool_idx)
        if was_resident:
            self._set_residency_state(
                req.req_pool_idx,
                HiSparseResidencyState.DEMOTING,
                count_transition=False,
            )
        if self.is_dsv4_hisparse:
            allocated_len = req.kv.kv_allocated_len
            alloc_size = self._device_buffer_alloc_size(allocated_len)
        else:
            allocated_len = req.kv.kv_allocated_len
            page_size = self.mem_pool_device.page_size
            # Allocate only enough for current tokens (page-aligned).
            # When prefill already fills device_buffer_size, include the reserved page.
            alloc_size = min(
                ((allocated_len + page_size - 1) // page_size) * page_size,
                self.device_buffer_size,
            )
            if alloc_size == self.device_buffer_size:
                alloc_size = self.padded_buffer_size

        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :allocated_len]
            )
        )
        compressed_len = len(compressed_logical_indices)

        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            compressed_logical_indices, alloc_size
        )
        if buffer_indices is None:
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        self.token_to_kv_pool_allocator.claim_hisparse_ownership(buffer_indices)
        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size

        self.req_device_buffer_tokens[
            :, req.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32[: self.device_buffer_size]
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )
        self._skip_first_backup[req.req_pool_idx] = True
        self._set_residency_state(
            req.req_pool_idx,
            HiSparseResidencyState.DEVICE_BUFFERED,
            count_transition=was_resident,
        )

    def _demote_resident_request(self, req: Req) -> None:
        if not self._is_resident(req.req_pool_idx):
            return
        if self._device_slot_owner is not self:
            raise RuntimeError("Only the HiSparse slot owner may initiate demotion")
        if self._pending_draft_extend_backup is not None:
            raise RuntimeError(
                "Cannot demote a resident HiSparse request before speculative "
                "draft-extend backup is finalized"
            )
        for mirror in self._device_slot_mirrors:
            mirror.wait_for_pending_backup()
            if mirror._pending_draft_extend_backup is not None:
                raise RuntimeError(
                    "Cannot demote a resident HiSparse mirror before speculative "
                    "draft-extend backup is finalized"
                )
        transition_start = time.perf_counter()
        if self.dynamic_residency_mode == "admission_once":
            self._pressure_demoted_requests.add(req.req_pool_idx)
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        available_before = allocator.available_size()
        self._free_resident_spec_page(req, free_physical=True)
        for mirror in self._device_slot_mirrors:
            mirror._free_resident_spec_page(req, free_physical=False)
        self.alloc_device_buffer(req)
        # Runtime demotion occurs after the previous token has been backed up;
        # unlike initial admission, the next decode token must not be skipped.
        self._skip_first_backup[req.req_pool_idx] = False
        for mirror in self._device_slot_mirrors:
            mirror.mirror_device_slots_from(self, req.req_pool_idx)
            mirror._skip_first_backup[req.req_pool_idx] = False
            mirror._set_residency_state(
                req.req_pool_idx, HiSparseResidencyState.DEVICE_BUFFERED
            )
        reusable_slots = max(0, allocator.available_size() - available_before)
        self._demotion_reclaimed_bytes += self._device_kv_payload_bytes(reusable_slots)
        self._demotion_transition_seconds += time.perf_counter() - transition_start

    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Grow device buffers for requests whose sequence length exceeds current capacity."""
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            # Compute all grow sizes on CPU, then do a single bulk allocation
            req_idxs = []
            old_caps = []
            new_caps = []
            grow_sizes = []
            total_grow = 0
            for i in grow_indices.tolist():
                req_idx = int(req_pool_indices_cpu[i])
                current_cap = int(current_caps[i])
                seq_len = int(seq_lens_cpu[i])

                new_cap = min(
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:
                    new_cap = self.padded_buffer_size
                grow_size = new_cap - current_cap
                if grow_size <= 0:
                    continue
                req_idxs.append(req_idx)
                old_caps.append(current_cap)
                new_caps.append(new_cap)
                grow_sizes.append(grow_size)
                total_grow += grow_size

            if total_grow > 0:
                all_new_indices = (
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                        total_grow
                    )
                )
                if all_new_indices is None:
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
                    )

                self.token_to_kv_pool_allocator.claim_hisparse_ownership(
                    all_new_indices
                )
                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    hot_end = min(new_cap, self.device_buffer_size)
                    if current_cap < hot_end:
                        self.req_device_buffer_tokens[
                            :, req_idx, current_cap:hot_end
                        ] = self._device_buffer_arange_i32[current_cap:hot_end]
                    if hot_end < new_cap:
                        self.req_device_buffer_tokens[:, req_idx, hot_end:new_cap] = -1
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap

        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    def has_ongoing_staging(self) -> bool:
        return len(self.ack_staging_queue) > 0

    def collect_ready_reqs(self) -> List[Req]:
        ready_reqs: List[Req] = []
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            self._skip_first_backup[req.req_pool_idx] = True
            req.hisparse_staging = False
            self._set_residency_state(
                req.req_pool_idx,
                HiSparseResidencyState.RESIDENT,
                count_transition=False,
            )
            self.active_hisparse_reqs[req.req_pool_idx] = req
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    def rebalance_dynamic_residency(
        self, req_indices: List[int], logical_lens_cpu: torch.Tensor
    ) -> None:
        if not self.dynamic_residency or self._device_slot_owner is not self:
            return

        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        low_tokens = int(allocator.size * self.dynamic_residency_demote_watermark)
        high_tokens = int(allocator.size * self.dynamic_residency_promote_watermark)

        logical_lens = {
            req_idx: int(logical_lens_cpu[position])
            for position, req_idx in enumerate(req_indices)
        }
        residents = [
            self.active_hisparse_reqs[req_idx]
            for req_idx in req_indices
            if self._is_resident(req_idx)
        ]
        residents.sort(key=lambda req: logical_lens[req.req_pool_idx], reverse=True)
        # ``forced_resident`` is a diagnostic control, not a best-effort policy.
        # Never hide a capacity failure by demoting and immediately promoting
        # the same request.  If the resident footprint cannot grow, the owning
        # allocation path must fail explicitly so the benchmark is classified
        # as a resident-capacity failure.
        if self.dynamic_residency_mode == "forced_resident":
            return
        if any(
            logical_lens[req.req_pool_idx] > self.dynamic_residency_max_tokens
            or allocator.available_size() < low_tokens
            for req in residents
        ):
            self.wait_for_pending_backup()
        for req in residents:
            exceeds_limit = (
                self.dynamic_residency_mode != "forced_resident"
                and logical_lens[req.req_pool_idx] > self.dynamic_residency_max_tokens
            )
            under_pressure = allocator.available_size() < low_tokens
            if not exceeds_limit and not under_pressure:
                continue
            self._demote_resident_request(req)
            if not exceeds_limit and allocator.available_size() >= high_tokens:
                break

        # At most one host-to-device promotion per target decode step.
        for position, req_idx in enumerate(req_indices):
            if self._state(req_idx) != HiSparseResidencyState.DEVICE_BUFFERED:
                continue
            req = self.active_hisparse_reqs[req_idx]
            logical_len = max(0, int(logical_lens_cpu[position]))
            if self._try_promote_from_host(req, logical_len=logical_len):
                break

    def advance_dynamic_residency(
        self, req_indices: List[int], logical_lens_cpu: torch.Tensor
    ) -> None:
        """Advance the decode policy clock and rebalance at a safe boundary."""
        if not self.dynamic_residency or self._device_slot_owner is not self:
            return
        self._decode_step += 1
        self.rebalance_dynamic_residency(req_indices, logical_lens_cpu)
        self._active_request_steps += len(req_indices)
        self._resident_request_steps += sum(
            self._is_resident(req_idx) for req_idx in req_indices
        )

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: Optional[torch.Tensor] = None,
    ) -> None:
        if req_pool_indices_cpu is None:
            req_pool_indices_cpu = req_pool_indices.cpu()

        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if self.token_to_kv_pool_allocator.page_size > 1:
            self._rehome_page_boundary_owners(
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                req_pool_indices=req_pool_indices,
                seq_lens_cpu=seq_lens_cpu,
                req_pool_indices_cpu=req_pool_indices_cpu,
            )

        active_req_indices = [int(req_idx) for req_idx in req_pool_indices_cpu.tolist()]
        self.advance_dynamic_residency(
            active_req_indices, (seq_lens_cpu - 1).clamp_min(0)
        )

        if self.compress_ratio == 1:
            active_positions = torch.arange(
                len(seq_lens_cpu), dtype=torch.int64, device=seq_lens_cpu.device
            )
        else:
            active_positions = torch.where(seq_lens_cpu % self.compress_ratio == 0)[0]
        if active_positions.numel() == 0:
            return

        active_pos_tensor = active_positions.to(device=self.device)
        active_seq_lens = seq_lens[active_pos_tensor] // self.compress_ratio
        active_seq_lens_cpu = seq_lens_cpu[active_positions] // self.compress_ratio
        active_out_cache_loc = out_cache_loc[active_pos_tensor]
        active_req_pool_indices = req_pool_indices[active_pos_tensor]
        active_req_pool_indices_cpu = req_pool_indices_cpu[active_positions]

        active_req_indices = active_req_pool_indices_cpu.tolist()
        resident_positions = [
            i
            for i, req_idx in enumerate(active_req_indices)
            if self._is_resident(int(req_idx))
        ]
        swap_positions = [
            i
            for i, req_idx in enumerate(active_req_indices)
            if not self._is_resident(int(req_idx))
        ]

        if resident_positions:
            resident_seq_lens_cpu = active_seq_lens_cpu[resident_positions]
            allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
            num_new_pages = int(
                (resident_seq_lens_cpu % allocator.page_size == 1).int().sum().item()
            )
            if not self._has_free_hisparse_pages(allocator, num_new_pages):
                need_tokens = num_new_pages * allocator.page_size
                if not self.demote_until_hisparse_available(need_tokens):
                    raise RuntimeError("HiSparse dynamic decode allocation failed")
                resident_positions = [
                    i
                    for i, req_idx in enumerate(active_req_indices)
                    if self._is_resident(int(req_idx))
                ]
                swap_positions = [
                    i
                    for i, req_idx in enumerate(active_req_indices)
                    if not self._is_resident(int(req_idx))
                ]
                resident_seq_lens_cpu = active_seq_lens_cpu[resident_positions]
                num_new_pages = int(
                    (resident_seq_lens_cpu % allocator.page_size == 1)
                    .int()
                    .sum()
                    .item()
                )
                if not self._has_free_hisparse_pages(allocator, num_new_pages):
                    raise RuntimeError("HiSparse dynamic decode allocation failed")
            if resident_positions:
                self._alloc_resident_last_locs(
                    resident_positions,
                    active_seq_lens,
                    active_seq_lens_cpu[resident_positions],
                    active_out_cache_loc,
                    active_req_pool_indices,
                )

        if swap_positions:
            pos_tensor = torch.tensor(
                swap_positions, dtype=torch.int64, device=self.device
            )
            reserved_buffer_loc = self._grow_device_buffers(
                active_seq_lens[pos_tensor],
                active_req_pool_indices[pos_tensor],
                active_seq_lens_cpu[swap_positions],
                active_req_pool_indices_cpu[swap_positions],
            )
            self.req_device_buffer_token_locs[
                :, active_req_pool_indices[pos_tensor], self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)
            self.req_device_buffer_tokens[
                :,
                active_req_pool_indices[pos_tensor],
                self.device_buffer_size,
            ] = (
                seq_lens[active_pos_tensor][pos_tensor].to(torch.int32) - 1
            )
            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                active_out_cache_loc[pos_tensor]
            )
            self._free_stale_hisparse_mapping(compressed_locs, reserved_buffer_loc)
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc

    def _rehome_page_boundary_owners(
        self,
        *,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Resolve whole temporary pages before publishing decode remaps.

        A paged speculative reserve owns a complete physical page even though
        decode publishes one semantic C4 slot at a time.  At the first slot of
        each page, transfer that page to a growing generic device buffer or
        release it whole.  Replacing only the first mapping would strand the
        remaining speculative tail until process exit.
        """
        allocator = self.token_to_kv_pool_allocator
        page_size = allocator.hisparse_device_page_size
        boundary_batch_indices_cpu = []
        for batch_index in range(seq_lens_cpu.numel()):
            seq_len = int(seq_lens_cpu[batch_index])
            if self.is_dsv4_hisparse and seq_len % self.compress_ratio != 0:
                continue
            semantic_position = seq_len - 1
            if self.is_dsv4_hisparse:
                semantic_position = seq_len // self.compress_ratio - 1
            if semantic_position % page_size == 0:
                boundary_batch_indices_cpu.append(batch_index)
        if not boundary_batch_indices_cpu:
            return

        boundary_batch_indices = torch.tensor(
            boundary_batch_indices_cpu, dtype=torch.int64, device=seq_lens.device
        )
        boundary_seq_lens = seq_lens[boundary_batch_indices]
        boundary_req_indices = req_pool_indices[boundary_batch_indices]
        semantic_positions = boundary_seq_lens - 1
        if self.is_dsv4_hisparse:
            semantic_positions = boundary_seq_lens // self.compress_ratio - 1
        first_mapping_indices = allocator.get_last_loc_compressed(
            out_cache_loc[boundary_batch_indices]
        ).to(torch.int64)
        offsets = torch.arange(page_size, dtype=torch.int64, device=seq_lens.device)
        mapping_index_blocks = first_mapping_indices[:, None] + offsets
        buffer_positions = (semantic_positions[:, None] + offsets).clamp(
            max=self.device_buffer_size
        )
        mapping = self.mem_pool_device.full_to_hisparse_device_index_mapping
        mapped_coordinates = mapping[mapping_index_blocks]
        temporary_page_ids = mapped_coordinates[:, 0] // page_size
        temporary_blocks = temporary_page_ids[:, None] * page_size + offsets
        torch._assert_async(
            torch.all(first_mapping_indices % page_size == 0),
            "HiSparse temporary mapping keys must be page aligned",
        )
        torch._assert_async(
            torch.all(mapped_coordinates == temporary_blocks),
            "HiSparse temporary owners must cover complete pages",
        )
        sorted_page_ids = torch.sort(temporary_page_ids).values
        torch._assert_async(
            torch.all(sorted_page_ids[1:] != sorted_page_ids[:-1]),
            "HiSparse temporary pages must have unique request owners",
        )

        owner_rows = self.req_to_device_buffer[boundary_req_indices]
        growths = []
        for boundary_index, batch_index in enumerate(boundary_batch_indices_cpu):
            req_index = int(req_pool_indices_cpu[batch_index])
            seq_len = int(seq_lens_cpu[batch_index])
            old_cap = int(self.req_device_buffer_size[req_index])
            if self.is_dsv4_hisparse or seq_len > self.device_buffer_size:
                continue
            if seq_len <= old_cap:
                continue
            assert seq_len - 1 == old_cap
            new_cap = min(
                (seq_len + page_size - 1) // page_size * page_size,
                self.device_buffer_size,
            )
            if new_cap == self.device_buffer_size:
                new_cap = self.padded_buffer_size
            net_extra = new_cap - old_cap - page_size
            assert net_extra >= 0
            growths.append(
                (boundary_index, req_index, old_cap, new_cap, net_extra)
            )

        growth_mask_values = [False] * len(boundary_batch_indices_cpu)
        for boundary_index, *_ in growths:
            growth_mask_values[boundary_index] = True
        growth_mask = torch.tensor(growth_mask_values, device=seq_lens.device)
        torch._assert_async(
            torch.all(~torch.isin(owner_rows // page_size, temporary_page_ids)),
            "HiSparse temporary pages must not already belong to a device buffer",
        )
        existing_destinations = torch.gather(owner_rows, 1, buffer_positions)
        torch._assert_async(
            torch.all(existing_destinations[~growth_mask] > 0),
            "HiSparse release destinations must remain owned",
        )
        torch._assert_async(
            torch.all(
                ~torch.isin(
                    existing_destinations[~growth_mask] // page_size,
                    temporary_page_ids,
                )
            ),
            "HiSparse release destinations must not alias temporary pages",
        )

        total_net_extra = sum(growth[-1] for growth in growths)
        if total_net_extra > 0:
            extra_indices = allocator.hisparse_attn_allocator.alloc(total_net_extra)
            if extra_indices is None:
                raise RuntimeError(
                    "HiSparse device buffer net allocation failed "
                    f"(total_net_extra={total_net_extra})"
                )
        else:
            extra_indices = temporary_page_ids[:0]

        destinations = torch.where(
            growth_mask[:, None], temporary_blocks, existing_destinations
        )

        def install_growth_rows() -> None:
            retained_blocks = temporary_blocks[growth_mask].reshape(-1)
            if retained_blocks.numel() > 0:
                allocator.claim_hisparse_ownership(retained_blocks)
            if extra_indices.numel() > 0:
                allocator.claim_hisparse_ownership(extra_indices)
            extra_offset = 0
            for boundary_index, req_index, old_cap, new_cap, net_extra in growths:
                self.req_to_device_buffer[
                    req_index, old_cap : old_cap + page_size
                ] = temporary_blocks[boundary_index]
                self.req_to_device_buffer[
                    req_index, old_cap + page_size : new_cap
                ] = extra_indices[extra_offset : extra_offset + net_extra]
                extra_offset += net_extra
                self.req_device_buffer_size[req_index] = new_cap

        allocator.rehome_temporary_hisparse_pages(
            mapping_indices=mapping_index_blocks.reshape(-1),
            retained_page_ids=temporary_page_ids[growth_mask],
            install_retained_owner=install_growth_rows,
        )
        for _, req_index, old_cap, new_cap, _ in growths:
            self.req_device_buffer_token_locs[
                :, req_index, old_cap:new_cap
            ] = self.req_to_device_buffer[req_index, old_cap:new_cap].to(torch.int32)
        mapping[mapping_index_blocks] = destinations

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Back up the previous compressed token to host memory.

        Each newly produced compressed token (one per `compress_ratio` decode
        steps) must be backed up to host so the swap-in kernel can later
        recover it.

        Two cases are skipped:
        - The first decode step right after staging: all prefill tokens were
          already backed up during staging, so there is nothing new to save.
        - Steps where `(seq_len - 1) % compress_ratio != 0`: no new compressed
          token was produced this step.
        """
        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up),
        # and skip non-aligned steps that did not produce a new compressed token.
        backup_indices = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                continue
            if (int(seq_lens_cpu[i]) - 1) % self.compress_ratio == 0:
                backup_indices.append(i)

        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
        backup_req_indices = req_pool_indices[backup_indices_gpu]

        # The previous compressed token's position and its device buffer slot:
        #  compressed_pos = (seq_len - 1) // compress_ratio - 1
        #  - short: slot = compressed_pos          (within the regular buffer)
        #  - long:  slot = device_buffer_size      (the reserved slot)
        prev_seq_lens = seq_lens[backup_indices_gpu] - 1
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio
        actual_compressed_pos = compressed_prev_seq_lens - 1

        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)

        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]
        resident_backup_indices = [
            j
            for j, i in enumerate(backup_indices)
            if self._is_resident(int(req_pool_indices_cpu[i]))
        ]
        if resident_backup_indices:
            resident_positions = torch.tensor(
                resident_backup_indices, dtype=torch.int64, device=self.device
            )
            device_locs = device_locs.clone()
            device_locs[resident_positions] = self._resident_token_device_locs(
                backup_req_indices[resident_positions],
                actual_compressed_pos[resident_positions],
            )

        host_locs_list = []
        for i in backup_indices:
            req_idx = int(req_pool_indices_cpu[i])
            start_pos = (int(seq_lens_cpu[i]) - 1) // self.compress_ratio - 1
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req_idx,
                start_pos,
                1,
            )
            host_locs_list.append(host_locs)
        host_locs = torch.cat(host_locs_list)

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_compressed_pos.is_cuda:
                actual_compressed_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    def wait_for_pending_backup(self) -> None:
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False

    def _backup_device_locs_to_host(
        self, host_locs: torch.Tensor, device_locs: torch.Tensor
    ) -> None:
        if host_locs.numel() == 0:
            return
        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        device_locs = device_locs.contiguous()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        event = device_module.Event()
        event.record(self.decode_backup_stream)
        device_module.current_stream().wait_event(event)

    @staticmethod
    def _debug_tensor_row_fingerprints(rows: torch.Tensor) -> List[str]:
        """Return byte-level row fingerprints for an opt-in correctness probe."""
        if rows.numel() == 0:
            return []
        rows_u8 = (
            rows.detach().contiguous().view(torch.uint8).reshape(rows.shape[0], -1)
        )
        rows_cpu = rows_u8.cpu()
        return [
            hashlib.sha256(row.numpy().tobytes()).hexdigest()[:16] for row in rows_cpu
        ]

    def _debug_device_row_fingerprints(self, device_locs: torch.Tensor) -> List[str]:
        device_module.current_stream().synchronize()
        rows = self.mem_pool_device.kv_buffer[0][device_locs.to(torch.int64)]
        return self._debug_tensor_row_fingerprints(rows)

    def _debug_host_row_fingerprints(self, host_locs: torch.Tensor) -> List[str]:
        device_module.current_stream().synchronize()
        assert self.mem_pool_host.layout == "layer_first"
        host_locs_cpu = host_locs.detach().to(device="cpu", dtype=torch.int64)
        rows = self.mem_pool_host.kv_buffer[0, host_locs_cpu]
        return self._debug_tensor_row_fingerprints(rows)

    def finish_pending_draft_extend_backup(self) -> None:
        pending = self._pending_draft_extend_backup
        if pending is None:
            return
        self._pending_draft_extend_backup = None
        host_locs, device_locs, logical_locs_to_clear = pending
        debug_pending = self._debug_pending_draft_extend
        self._debug_pending_draft_extend = None
        post_extend_fingerprints = None
        if debug_pending is not None:
            post_extend_fingerprints = self._debug_device_row_fingerprints(device_locs)
        self._backup_device_locs_to_host(host_locs, device_locs)
        if debug_pending is not None:
            bucket, token_positions, debug_device_locs, pre_extend_fingerprints = (
                debug_pending
            )
            if not torch.equal(debug_device_locs, device_locs):
                raise RuntimeError(
                    "HiSparse generated-KV diagnostic device slots changed before "
                    f"draft-extend backup: expected={debug_device_locs.tolist()} "
                    f"actual={device_locs.tolist()}"
                )
            host_fingerprints = self._debug_host_row_fingerprints(host_locs)
            if host_fingerprints != post_extend_fingerprints:
                raise RuntimeError(
                    "HiSparse draft generated-KV backup content mismatch: "
                    f"bucket={bucket} token_positions={token_positions} "
                    f"device={post_extend_fingerprints} host={host_fingerprints}"
                )
            logger.warning(
                "HiSparse draft generated-KV fingerprint: bucket=%d "
                "token_positions=%s device_locs=%s changed_by_draft_extend=%s "
                "pre=%s post=%s host_match=True",
                bucket,
                token_positions,
                device_locs.tolist(),
                [
                    before != after
                    for before, after in zip(
                        pre_extend_fingerprints, post_extend_fingerprints
                    )
                ],
                pre_extend_fingerprints,
                post_extend_fingerprints,
            )
        if logical_locs_to_clear.numel() > 0:
            self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
                logical_locs_to_clear
            ] = 0

    def clear_pending_draft_extend_backup(self) -> None:
        pending = self._pending_draft_extend_backup
        if pending is None:
            return
        self._pending_draft_extend_backup = None
        self._debug_pending_draft_extend = None
        _, _, logical_locs_to_clear = pending
        if logical_locs_to_clear.numel() > 0:
            self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
                logical_locs_to_clear
            ] = 0

    def supports_hisparse_draft_slots(self) -> bool:
        return not self.is_dsv4_hisparse

    def _compute_padded_grow(self, req_indices_list: List[int]):
        """Return the per-request grow plan and total device-pool demand needed
        to bring every request up to ``padded_buffer_size`` (hot buffer + draft
        page).  Resident requests keep a virtual hot buffer and only need one
        extra graph-stable speculative page."""
        grow_reqs = []
        total_grow = 0
        for req_idx in req_indices_list:
            current_cap = int(self.req_device_buffer_size[req_idx])
            if current_cap >= self.padded_buffer_size:
                continue
            resident = self._is_resident(req_idx)
            if resident and current_cap != 0:
                raise RuntimeError(
                    "Resident HiSparse request has a partial speculative page: "
                    f"req_pool_idx={req_idx}, current={current_cap}"
                )
            grow_size = (
                self.page_size if resident else self.padded_buffer_size - current_cap
            )
            grow_reqs.append((req_idx, current_cap, resident, grow_size))
            total_grow += grow_size
        return grow_reqs, total_grow

    def _ensure_padded_buffer(self, req_pool_indices: torch.Tensor) -> None:
        """Ensure each request owns a fixed hot buffer plus one extra draft page."""
        req_indices_list = req_pool_indices.cpu().tolist()
        grow_reqs, total_grow = self._compute_padded_grow(req_indices_list)

        if total_grow == 0:
            return
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        all_new = allocator.alloc(total_grow)
        if all_new is None:
            # schedulable_hisparse_available() -- which gates PD admission
            # (hisparse_direct_admission_capacity) and KV-full decode retraction
            # (HiSparse allocator.available_size()) -- counts reclaimable resident
            # device pages as available.  The other two device-pool allocation
            # sites (admit_request_direct, the dynamic-decode grow path) honor that
            # promise by reclaiming before allocating; this spec-verify grow path
            # historically called alloc() raw and raised into the scheduler event
            # loop on None, SIGQUIT-ing PID 1 under KV-retraction pressure at
            # decode batch > ~16.  Reclaim first, then retry.  Demotion can change
            # the residency/current_cap of requests in this batch, so recompute the
            # grow plan against the post-demotion state before writing slots.
            self.demote_until_hisparse_available(total_grow)
            grow_reqs, total_grow = self._compute_padded_grow(req_indices_list)
            if total_grow == 0:
                return
            all_new = allocator.alloc(total_grow)
            if all_new is None:
                raise RuntimeError(
                    "HiSparse: failed to grow buffers for draft slots even after "
                    f"reclaiming resident device pages (need {total_grow}, "
                    f"available={allocator.available_size()}). The decode batch was "
                    "admitted beyond reclaim-adjusted HiSparse device-pool capacity."
                )

        self.token_to_kv_pool_allocator.claim_hisparse_ownership(all_new)
        offset = 0
        for req_idx, current_cap, resident, grow_size in grow_reqs:
            chunk = all_new[offset : offset + grow_size]
            offset += grow_size
            if resident:
                # Keep every hot-buffer metadata column invalid.  The JIT
                # kernel uses this sentinel to select the full-resident mapping
                # path; only the graph-stable speculative page is materialized.
                page_start = self.device_buffer_size
                self.req_to_device_buffer[
                    req_idx, page_start : self.padded_buffer_size
                ] = chunk
                self.req_device_buffer_tokens[:, req_idx, :] = -1
                self.req_device_buffer_token_locs[:, req_idx, :] = -1
                self.req_device_buffer_token_locs[
                    :, req_idx, page_start : self.padded_buffer_size
                ] = chunk
                self.req_device_buffer_size[req_idx] = self.padded_buffer_size
                continue
            self.req_to_device_buffer[
                req_idx, current_cap : self.padded_buffer_size
            ] = chunk
            hot_end = min(self.padded_buffer_size, self.device_buffer_size)
            if current_cap < hot_end:
                self.req_device_buffer_tokens[:, req_idx, current_cap:hot_end] = (
                    self._device_buffer_arange_i32[current_cap:hot_end]
                )
            if hot_end < self.padded_buffer_size:
                self.req_device_buffer_tokens[
                    :, req_idx, hot_end : self.padded_buffer_size
                ] = -1
            self.req_device_buffer_token_locs[
                :, req_idx, current_cap : self.padded_buffer_size
            ] = chunk
            self.req_device_buffer_size[req_idx] = self.padded_buffer_size

    def get_draft_device_slots(
        self,
        req_pool_indices: torch.Tensor,
        num_tokens_per_req: int,
        start_positions_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Return stable coordinator-owned physical slots for uniform drafts."""
        assert self.supports_hisparse_draft_slots()
        start = self.device_buffer_size + 1
        if start + num_tokens_per_req > self.padded_buffer_size:
            raise ValueError(
                f"Requested {num_tokens_per_req} draft slots but extra page only "
                f"has {self.padded_buffer_size - self.device_buffer_size - 1}"
            )
        self._ensure_padded_buffer(req_pool_indices)
        self.req_device_buffer_tokens[
            :, req_pool_indices, start : self.padded_buffer_size
        ] = -1
        total_slots = req_pool_indices.numel() * num_tokens_per_req
        row_indices = torch.repeat_interleave(req_pool_indices, num_tokens_per_req)
        pos_in_segment = torch.arange(total_slots, device=req_pool_indices.device) % (
            num_tokens_per_req
        )
        start_positions = start_positions_cpu.to(
            device=req_pool_indices.device, dtype=torch.int64
        )
        token_positions = (
            torch.repeat_interleave(start_positions, num_tokens_per_req)
            + pos_in_segment
        )
        resident_rows = torch.tensor(
            [self._is_resident(int(req_idx)) for req_idx in row_indices.cpu().tolist()],
            dtype=torch.bool,
            device=req_pool_indices.device,
        )
        col_indices = torch.where(
            (~resident_rows) & (token_positions < self.device_buffer_size),
            token_positions,
            start + pos_in_segment,
        )
        if torch.any(col_indices >= self.padded_buffer_size):
            raise ValueError(
                "HiSparse draft slots exceed padded buffer: "
                f"{col_indices.max().item()=} {self.padded_buffer_size=}"
            )
        self.req_device_buffer_tokens[:, row_indices, col_indices] = token_positions.to(
            torch.int32
        ).unsqueeze(0)
        return self.req_to_device_buffer[row_indices, col_indices]

    def get_draft_device_slots_variable(
        self,
        req_pool_indices: torch.Tensor,
        tokens_per_req_cpu: torch.Tensor,
        start_positions_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Return stable coordinator-owned physical slots for variable drafts."""
        assert self.supports_hisparse_draft_slots()
        if tokens_per_req_cpu.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=req_pool_indices.device)
        start = self.device_buffer_size + 1
        max_tokens = int(tokens_per_req_cpu.max().item())
        if start + max_tokens > self.padded_buffer_size:
            raise ValueError(
                f"Max per-request draft slots ({max_tokens}) exceeds extra page "
                f"capacity ({self.padded_buffer_size - self.device_buffer_size - 1})."
            )
        self._ensure_padded_buffer(req_pool_indices)
        self.req_device_buffer_tokens[
            :, req_pool_indices, start : self.padded_buffer_size
        ] = -1
        total_slots = int(tokens_per_req_cpu.sum().item())
        if total_slots == 0:
            return torch.empty(0, dtype=torch.int64, device=req_pool_indices.device)
        tokens_per_req = tokens_per_req_cpu.to(
            device=req_pool_indices.device, dtype=torch.int64
        )
        row_indices = torch.repeat_interleave(req_pool_indices, tokens_per_req)
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=tokens_per_req.device),
                tokens_per_req.cumsum(0),
            ]
        )
        pos_in_segment = torch.arange(total_slots, device=tokens_per_req.device) - (
            torch.repeat_interleave(offsets[:-1], tokens_per_req)
        )
        start_positions = start_positions_cpu.to(
            device=req_pool_indices.device, dtype=torch.int64
        )
        token_positions = torch.repeat_interleave(start_positions, tokens_per_req) + (
            pos_in_segment
        )
        resident_rows = torch.tensor(
            [self._is_resident(int(req_idx)) for req_idx in row_indices.cpu().tolist()],
            dtype=torch.bool,
            device=req_pool_indices.device,
        )
        col_indices = torch.where(
            (~resident_rows) & (token_positions < self.device_buffer_size),
            token_positions,
            start + pos_in_segment,
        )
        if torch.any(col_indices >= self.padded_buffer_size):
            raise ValueError(
                "HiSparse variable draft slots exceed padded buffer: "
                f"{col_indices.max().item()=} {self.padded_buffer_size=}"
            )
        self.req_device_buffer_tokens[:, row_indices, col_indices] = token_positions.to(
            torch.int32
        ).unsqueeze(0)
        return self.req_to_device_buffer[row_indices, col_indices]

    def prepare_verify_slots_spec_v2(
        self,
        req_pool_indices: torch.Tensor,
        verify_cache_locs: torch.Tensor,
        num_tokens_per_req: int,
        start_positions_cpu: torch.Tensor,
    ) -> None:
        """Bind spec-v2 target-verify logical locations to stable draft slots."""
        device_slots = self.get_draft_device_slots(
            req_pool_indices, num_tokens_per_req, start_positions_cpu
        )
        if verify_cache_locs.numel() != device_slots.numel():
            raise ValueError(
                "HiSparse spec-v2 verify slot mismatch: "
                f"logical={verify_cache_locs.numel()}, device={device_slots.numel()}"
            )
        self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
            verify_cache_locs
        ] = device_slots

    def finalize_accepted_tokens(
        self,
        req_pool_indices: torch.Tensor,
        accepted_cache_locs: torch.Tensor,
        draft_cache_locs: torch.Tensor,
        num_correct_drafts: torch.Tensor,
        num_correct_drafts_cpu: torch.Tensor,
        accepted_token_positions: torch.Tensor,
    ) -> None:
        """Commit accepted slots and detach rejected logical mappings exactly once."""
        assert self.supports_hisparse_draft_slots()
        if accepted_cache_locs.numel() == 0:
            return
        self.clear_pending_draft_extend_backup()

        counts = num_correct_drafts.to(torch.int64) + 1
        counts_cpu = num_correct_drafts_cpu.to(torch.int64) + 1
        total_accepted = int(counts_cpu.sum().item())
        if total_accepted != accepted_cache_locs.numel():
            raise ValueError(
                "HiSparse accepted-token bookkeeping mismatch: "
                f"expected={total_accepted}, actual={accepted_cache_locs.numel()}"
            )
        if total_accepted != accepted_token_positions.numel():
            raise ValueError(
                "HiSparse accepted-token position mismatch: "
                f"expected={total_accepted}, actual={accepted_token_positions.numel()}"
            )

        mapping = self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        accepted_token_positions = accepted_token_positions.to(
            device=accepted_cache_locs.device, dtype=torch.int64
        )
        in_hot_buffer = accepted_token_positions < self.device_buffer_size
        accepted_device_locs = mapping[accepted_cache_locs].clone()

        # From this point, ordinary allocator.free owns only logical slots. The
        # coordinator continues to own all physical slots in the request buffer.
        mapping[draft_cache_locs] = 0
        accepted_req_indices = torch.repeat_interleave(req_pool_indices, counts)
        if torch.any(in_hot_buffer):
            hot_locs = accepted_cache_locs[in_hot_buffer]
            hot_positions = accepted_token_positions[in_hot_buffer]
            hot_req_indices = accepted_req_indices[in_hot_buffer]
            mapping[hot_locs] = self.req_to_device_buffer[
                hot_req_indices, hot_positions
            ]

        needs_backup = ~in_hot_buffer
        backup_count = int(needs_backup.sum().item())
        host_locs = torch.empty(0, dtype=torch.int64, device=self.device)
        backup_device_locs = torch.empty(
            0, dtype=accepted_device_locs.dtype, device=accepted_device_locs.device
        )
        if backup_count > 0:
            backup_positions = accepted_token_positions[needs_backup]
            backup_req_indices = accepted_req_indices[needs_backup]
            backup_device_locs = accepted_device_locs[needs_backup]
            # Preserve existing preallocated slots when acceptance straddles a
            # row boundary.  Only the canonical target owner may grow the host
            # row; a draft coordinator mirrors those numerical slots.
            host_locs = self._ensure_host_slots_for_positions(
                backup_req_indices, backup_positions
            )
            self._backup_device_locs_to_host(host_locs, backup_device_locs)
            mapping[accepted_cache_locs[needs_backup]] = backup_device_locs

        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=counts.device), counts.cumsum(0)]
        )
        last_offsets = offsets[1:] - 1
        last_positions = accepted_token_positions[last_offsets]
        reserved_positions = last_positions.clamp(max=self.device_buffer_size)
        newest_slots = self.req_to_device_buffer[req_pool_indices, reserved_positions]
        last_logical = accepted_cache_locs[last_offsets]
        last_slots = accepted_device_locs[last_offsets]
        self.req_device_buffer_tokens[:, req_pool_indices, reserved_positions] = (
            last_positions.to(torch.int32).unsqueeze(0)
        )
        self.req_device_buffer_token_locs[:, req_pool_indices, reserved_positions] = (
            newest_slots.to(torch.int32)
        )
        for req_idx in req_pool_indices.tolist():
            self._skip_first_backup[req_idx] = True

        same_slot = last_slots == newest_slots
        if torch.any(~same_slot):
            self.mem_pool_device.transfer_values_on_device(
                dst_indices=newest_slots[~same_slot],
                src_indices=last_slots[~same_slot],
            )
        mapping[last_logical] = newest_slots

        if backup_count > 0:
            backup_positions_in_needs = (
                torch.cumsum(needs_backup.to(torch.int64), dim=0) - 1
            )
            last_needs_backup = needs_backup[last_offsets]
            post_backup_device_locs = backup_device_locs.clone()
            if torch.any(last_needs_backup):
                last_backup_offsets = backup_positions_in_needs[
                    last_offsets[last_needs_backup]
                ]
                post_backup_device_locs[last_backup_offsets] = newest_slots[
                    last_needs_backup
                ]
            logical_locs_to_clear_mask = needs_backup.clone()
            logical_locs_to_clear_mask[last_offsets] = False
            self._pending_draft_extend_backup = (
                host_locs,
                post_backup_device_locs,
                accepted_cache_locs[logical_locs_to_clear_mask],
            )
            if self.debug_validate_generated_kv and self._device_slot_owner is not self:
                max_position = int(last_positions.max().item())
                bucket = max_position // self.mem_pool_device.page_size
                if bucket > self._debug_last_generated_kv_bucket:
                    self._debug_last_generated_kv_bucket = bucket
                    self._debug_pending_draft_extend = (
                        bucket,
                        accepted_token_positions[needs_backup].tolist(),
                        post_backup_device_locs.clone(),
                        self._debug_device_row_fingerprints(post_backup_device_locs),
                    )

    def _finalize_buffered_tokens_spec_v2(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        verify_cache_locs: torch.Tensor,
        accept_index: torch.Tensor,
    ) -> None:
        """Commit the accepted subset of a spec-v2 target-verify window."""
        assert self.supports_hisparse_draft_slots()
        if verify_cache_locs.numel() == 0:
            return
        counts = (accept_index != -1).sum(dim=1).to(torch.int64)
        total_accepted = int(counts.sum().item())
        if total_accepted == 0:
            self.token_to_kv_pool_allocator.clear_device_mapping(verify_cache_locs)
            return
        flat_accept_index = accept_index.reshape(-1)
        accepted_offsets = flat_accept_index[flat_accept_index >= 0].to(torch.int64)
        if accepted_offsets.numel() != total_accepted:
            raise ValueError(
                "HiSparse spec-v2 accepted index mismatch: "
                f"expected={total_accepted}, actual={accepted_offsets.numel()}"
            )
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=counts.device), counts.cumsum(0)]
        )
        pos_in_segment = torch.arange(
            total_accepted, dtype=torch.int64, device=counts.device
        ) - torch.repeat_interleave(offsets[:-1], counts)
        accepted_token_positions = (
            torch.repeat_interleave(seq_lens.to(torch.int64), counts) + pos_in_segment
        )
        self.finalize_accepted_tokens(
            req_pool_indices=req_pool_indices,
            accepted_cache_locs=verify_cache_locs[accepted_offsets],
            draft_cache_locs=verify_cache_locs,
            num_correct_drafts=counts - 1,
            num_correct_drafts_cpu=(counts - 1).cpu(),
            accepted_token_positions=accepted_token_positions,
        )

    def _select_spec_v2_rows(
        self,
        row_positions: List[int],
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        verify_cache_locs: torch.Tensor,
        accept_index: torch.Tensor,
    ):
        """Select request rows and rewrite flattened accept offsets locally."""
        batch_size = accept_index.shape[0]
        if batch_size == 0 or verify_cache_locs.numel() % batch_size != 0:
            raise ValueError(
                "HiSparse spec-v2 verify window is not request-major: "
                f"batch={batch_size}, slots={verify_cache_locs.numel()}"
            )
        row_width = verify_cache_locs.numel() // batch_size
        rows = torch.tensor(
            row_positions, dtype=torch.int64, device=req_pool_indices.device
        )
        row_cols = torch.arange(
            row_width, dtype=torch.int64, device=verify_cache_locs.device
        )
        verify_offsets = (rows[:, None] * row_width + row_cols[None, :]).reshape(-1)
        selected_accept = accept_index[rows].clone()
        valid = selected_accept >= 0
        if torch.any(valid):
            source_rows = selected_accept[valid] // row_width
            expected_rows = torch.repeat_interleave(
                rows, valid.sum(dim=1).to(torch.int64)
            )
            if not torch.equal(source_rows.to(expected_rows.device), expected_rows):
                raise ValueError(
                    "HiSparse spec-v2 accept index crosses request-row boundaries"
                )
            local_columns = selected_accept[valid] % row_width
            selected_rows = torch.arange(
                len(row_positions),
                dtype=torch.int64,
                device=selected_accept.device,
            )
            selected_accept[valid] = (
                torch.repeat_interleave(selected_rows, valid.sum(dim=1).to(torch.int64))
                * row_width
                + local_columns
            ).to(dtype=selected_accept.dtype)
        return (
            req_pool_indices[rows],
            seq_lens[rows],
            verify_cache_locs[verify_offsets],
            selected_accept,
        )

    def _rollback_resident_host_growth(
        self,
        old_host_lens: dict,
        mirror: Optional["HiSparseCoordinator"],
    ) -> None:
        for req_idx, old_len in old_host_lens.items():
            new_len = int(self.req_to_host_pool_allocated_len[req_idx])
            if new_len > old_len:
                new_host_locs = self.req_to_host_pool[req_idx, old_len:new_len]
                new_host_locs = new_host_locs[new_host_locs >= 0]
                if new_host_locs.numel() > 0 and self._host_slot_owner is self:
                    self.mem_pool_host.free(new_host_locs)
                self.req_to_host_pool[req_idx, old_len:new_len] = -1
                self.req_to_host_pool_allocated_len[req_idx] = old_len
            if mirror is not None:
                mirror.req_to_host_pool[req_idx, old_len:] = -1
                mirror.req_to_host_pool_allocated_len[req_idx] = old_len

    def _finalize_resident_tokens_spec_v2(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        verify_cache_locs: torch.Tensor,
        accept_index: torch.Tensor,
        mirror: Optional["HiSparseCoordinator"],
    ) -> None:
        """Move accepted verify KV from the side page into resident pages.

        The owner allocates page-aligned permanent physical storage.  Target and
        draft payloads are copied before the shared logical mapping is
        published, so an error leaves the original speculative-page mapping
        intact and permits a clean rollback.
        """
        mapping = self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        counts = (accept_index != -1).sum(dim=1).to(torch.int64)
        total_accepted = int(counts.sum().item())
        if total_accepted == 0:
            mapping[verify_cache_locs] = 0
            return

        flat_accept_index = accept_index.reshape(-1)
        accepted_offsets = flat_accept_index[flat_accept_index >= 0].to(torch.int64)
        if accepted_offsets.numel() != total_accepted:
            raise ValueError(
                "HiSparse resident spec-v2 accepted index mismatch: "
                f"expected={total_accepted}, actual={accepted_offsets.numel()}"
            )
        accepted_cache_locs = verify_cache_locs[accepted_offsets]
        source_device_locs = mapping[accepted_cache_locs].clone()
        if torch.any(source_device_locs <= 0):
            raise RuntimeError("HiSparse resident spec-v2 source mapping is missing")

        # Logical verify slots come from a transient allocator page.  Grouping
        # accepted tokens by that page allocates one permanent physical page on
        # every speculative iteration, even when only one or two tokens were
        # accepted.  Extend the request's stable resident tail page instead and
        # allocate a new page only when that tail is full.
        permanent_parts = []
        new_page_parts = []
        accepted_offset = 0
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        for row, req_idx in enumerate(req_pool_indices.cpu().tolist()):
            count = int(counts[row].item())
            if count == 0:
                continue
            seq_len = int(seq_lens[row].item())
            if seq_len <= 0:
                raise RuntimeError(
                    "HiSparse resident spec-v2 cannot extend an empty request"
                )
            previous_logical = self.req_to_token_pool.req_to_token[
                int(req_idx), seq_len - 1
            ]
            previous_compressed = (
                self.mem_pool_device.translate_loc_from_full_to_compressed(
                    previous_logical.reshape(1)
                )
            )
            previous_device = mapping[previous_compressed][0]
            if int(previous_device.item()) <= 0:
                raise RuntimeError(
                    "HiSparse resident spec-v2 previous mapping is missing: "
                    f"req_pool_idx={req_idx}, seq_len={seq_len}"
                )

            tail_free = (
                self.page_size
                - 1
                - int(previous_device.remainder(self.page_size).item())
            )
            reuse_count = min(count, tail_free)
            row_parts = []
            if reuse_count:
                row_parts.append(
                    previous_device
                    + torch.arange(
                        1,
                        reuse_count + 1,
                        dtype=previous_device.dtype,
                        device=previous_device.device,
                    )
                )

            remaining = count - reuse_count
            if remaining:
                new_page_count = (remaining + self.page_size - 1) // self.page_size
                new_pages = allocator.alloc(new_page_count * self.page_size)
                if new_pages is None:
                    for allocated in new_page_parts:
                        self.token_to_kv_pool_allocator.free_hisparse_indices(allocated)
                    raise RuntimeError(
                        "HiSparse resident spec-v2 permanent-page allocation "
                        f"failed: req_pool_idx={req_idx}, pages={new_page_count}"
                    )
                new_page_parts.append(new_pages)
                row_parts.append(new_pages[:remaining])

            row_locs = torch.cat(row_parts)
            if row_locs.numel() != count:
                raise RuntimeError(
                    "HiSparse resident spec-v2 permanent slot mismatch: "
                    f"req_pool_idx={req_idx}, expected={count}, "
                    f"actual={row_locs.numel()}"
                )
            permanent_parts.append(row_locs)
            accepted_offset += count

        if accepted_offset != total_accepted:
            for allocated in new_page_parts:
                self.token_to_kv_pool_allocator.free_hisparse_indices(allocated)
            raise RuntimeError(
                "HiSparse resident spec-v2 row accounting mismatch: "
                f"expected={total_accepted}, actual={accepted_offset}"
            )
        permanent_locs = torch.cat(permanent_parts)

        old_host_lens = {
            int(req_idx): int(self.req_to_host_pool_allocated_len[int(req_idx)])
            for req_idx in req_pool_indices.cpu().tolist()
        }
        try:
            self.mem_pool_device.transfer_values_on_device(
                dst_indices=permanent_locs,
                src_indices=source_device_locs,
            )
            if mirror is not None:
                mirror._validate_mirror_device_locs(permanent_locs)
                mirror.mem_pool_device.transfer_values_on_device(
                    dst_indices=permanent_locs,
                    src_indices=source_device_locs,
                )

            offsets = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int64, device=counts.device),
                    counts.cumsum(0),
                ]
            )
            pos_in_segment = torch.arange(
                total_accepted, dtype=torch.int64, device=counts.device
            ) - torch.repeat_interleave(offsets[:-1], counts)
            accepted_positions = (
                torch.repeat_interleave(seq_lens.to(torch.int64), counts)
                + pos_in_segment
            )
            accepted_req_indices = torch.repeat_interleave(req_pool_indices, counts)

            # PD normally preallocates the whole host row.  The fallback below
            # preserves single-node correctness while retaining target-only
            # free-list ownership for the mirrored draft host pool.
            for row, req_idx in enumerate(req_pool_indices.cpu().tolist()):
                count = int(counts[row].item())
                if count == 0:
                    continue
                start_pos = int(seq_lens[row].item())
                self.mem_pool_host.alloc_paged_token_slots(
                    self.req_to_host_pool,
                    self.req_to_host_pool_allocated_len,
                    int(req_idx),
                    start_pos,
                    count,
                )
                if mirror is not None:
                    mirror.mirror_host_slots_from(self, int(req_idx))

            host_locs = self.req_to_host_pool[accepted_req_indices, accepted_positions]
            if torch.any(host_locs < 0):
                raise RuntimeError("HiSparse resident spec-v2 host mapping is missing")
            self._backup_device_locs_to_host(host_locs, permanent_locs)
            if mirror is not None:
                mirror._backup_device_locs_to_host(host_locs, permanent_locs)

            # Prepare reusable side-page metadata before publishing the shared
            # mapping.  Once published, no fallible operation may run before
            # returning: rollback frees the new pages and would otherwise leave
            # the mapping pointing at released storage.
            for req_idx in req_pool_indices.cpu().tolist():
                self.req_device_buffer_tokens[
                    :, int(req_idx), self.device_buffer_size :
                ] = -1
                self._skip_first_backup[int(req_idx)] = True
                if mirror is not None:
                    mirror.req_device_buffer_tokens[
                        :, int(req_idx), mirror.device_buffer_size :
                    ] = -1
                    mirror._skip_first_backup[int(req_idx)] = True

            # Publish only after both target and draft payloads and their host
            # backups are complete.  The speculative page remains allocated at
            # a fixed address for the next graph replay.
            mapping[verify_cache_locs] = 0
            mapping[accepted_cache_locs] = permanent_locs
        except Exception:
            for allocated in new_page_parts:
                self.token_to_kv_pool_allocator.free_hisparse_indices(allocated)
            self._rollback_resident_host_growth(old_host_lens, mirror)
            raise

    def finalize_accepted_tokens_spec_v2(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        verify_cache_locs: torch.Tensor,
        accept_index: torch.Tensor,
        mirror: Optional["HiSparseCoordinator"] = None,
    ) -> None:
        """Commit target/draft accepted tokens for resident and buffered rows."""
        assert self.supports_hisparse_draft_slots()
        if verify_cache_locs.numel() == 0:
            return
        if self._device_slot_owner is not self:
            raise RuntimeError(
                "Only the target HiSparse slot owner may finalize spec-v2 mappings"
            )

        req_indices_cpu = [int(x) for x in req_pool_indices.cpu().tolist()]
        resident_rows = [
            row
            for row, req_idx in enumerate(req_indices_cpu)
            if self._is_resident(req_idx)
        ]
        buffered_rows = [
            row
            for row, req_idx in enumerate(req_indices_cpu)
            if not self._is_resident(req_idx)
        ]
        if mirror is not None:
            for req_idx in req_indices_cpu:
                if mirror._is_resident(req_idx) != self._is_resident(req_idx):
                    raise RuntimeError(
                        "Target and draft HiSparse residency states diverged: "
                        f"req_pool_idx={req_idx}"
                    )

        # Resident first: buffered finalization may defer one host backup until
        # the next draft-extend boundary.
        if resident_rows:
            resident_inputs = self._select_spec_v2_rows(
                resident_rows,
                req_pool_indices,
                seq_lens,
                verify_cache_locs,
                accept_index,
            )
            self._finalize_resident_tokens_spec_v2(*resident_inputs, mirror)

        if buffered_rows:
            buffered_inputs = self._select_spec_v2_rows(
                buffered_rows,
                req_pool_indices,
                seq_lens,
                verify_cache_locs,
                accept_index,
            )
            buffered_verify_locs = buffered_inputs[2]
            if mirror is not None:
                mapping = (
                    self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
                )
                mapping_snapshot = mapping[buffered_verify_locs].clone()
                try:
                    mirror._finalize_buffered_tokens_spec_v2(*buffered_inputs)
                finally:
                    mapping[buffered_verify_locs] = mapping_snapshot
            self._finalize_buffered_tokens_spec_v2(*buffered_inputs)

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.

        Note: dsv4 hisparse is not supported — DeepSeekV4SingleKVPoolHost has no
        load_to_device_per_layer and indices live in compressed space. Currently
        only used as a kernel oracle in test_hisparse_unit.py (non-dsv4 path).

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
        assert (
            not self.is_dsv4_hisparse
        ), "naive_load_topk is not implemented for dsv4 hisparse"
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            assert torch.all(
                selected_tokens >= 0
            ), f"Req {req_idx}: selected tokens contain negative positions"
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    def abort_staging_request(self, req: Req) -> None:
        """Remove a request from the staging queue and free its host + device resources.

        Must be called when aborting a request that has been admitted into staging
        but has not yet completed (i.e. req.hisparse_staging is True).
        """
        if self._device_slot_owner is not self:
            raise RuntimeError(
                "Only the canonical HiSparse slot owner may abort staging requests"
            )

        # Remove from staging queue
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        # Wait for any in-flight staging DMA to complete before freeing
        self.write_staging_stream.synchronize()

        prefill_len = req.extend_range.end
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefill_len
        ]
        self.token_to_kv_pool_allocator.free_hisparse(allocated_locs)

        # Free host memory that was allocated during admit_request_into_staging
        host_indices = self.req_to_host_pool[req.req_pool_idx]
        host_indices = host_indices[host_indices >= 0]
        self._debug_validate_host_request_slots(
            req, host_indices, stage="abort_staging"
        )
        if host_indices.numel() > 0 and self._host_slot_owner is self:
            self.mem_pool_host.free(host_indices)
            self._debug_validate_host_allocator_after_free(
                req, host_indices.numel(), stage="abort_staging"
            )
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self._skip_first_backup[req.req_pool_idx] = False
        getattr(self, "_req_c4_retired_len", {}).pop(req.req_pool_idx, None)
        getattr(self, "_req_c4_written_len", {}).pop(req.req_pool_idx, None)
        req.hisparse_staging = False
        self._clear_residency_state(req.req_pool_idx)

    def retract_req(self, req: Req) -> None:
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def request_finished(self, req: Req):
        # release resources only after the execution of a potential overlapped batch
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()
        self.clear_pending_draft_extend_backup()

        # Use kv_allocated_len (not seqlen): speculative decoding may reserve
        # beyond the committed length. The canonical owner must retire every
        # such mapping together with the side-buffer aliases before the later
        # logical release_kv_cache step.
        allocated_len = req.kv.kv_allocated_len

        is_resident_req = self._is_resident(req.req_pool_idx)

        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        if is_resident_req and not self.is_dsv4_hisparse and current_cap > 0:
            self._free_resident_spec_page(
                req, free_physical=self._device_slot_owner is self
            )
            current_cap = 0
        # DSV4 C4 mappings are independent physical owners. Complete pages can
        # leave the coordinator side buffer before the request finishes, so a
        # zero current_cap does not mean that the request has no live mapping.
        # The canonical owner must always enumerate the request-visible mapping
        # before release_kv_cache frees its logical slots. Draft mirrors only
        # clear local buffer state and never mutate the canonical mapping.
        should_release_device_ownership = (
            current_cap > 0 and not is_resident_req
        ) or (
            self.is_dsv4_hisparse
            and (is_resident_req or self._device_slot_owner is self)
        )
        if should_release_device_ownership:
            allocated_locs = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :allocated_len
            ]
            compressed_locs = (
                self.mem_pool_device.translate_loc_from_full_to_compressed(
                    allocated_locs
                )
            )
            buffer_locs = self.req_to_device_buffer[
                req.req_pool_idx, :current_cap
            ].clone()
            HiSparseCoordinator._debug_device_lifecycle_snapshot(
                self, req, buffer_locs, stage="finish_before"
            )

            def clear_device_buffer_owner() -> None:
                self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
                self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
                self.req_to_device_buffer[req.req_pool_idx, :] = 0
                self.req_device_buffer_size[req.req_pool_idx] = 0

            if self._device_slot_owner is self:
                self.token_to_kv_pool_allocator.release_hisparse_ownership(
                    mapping_indices=compressed_locs,
                    extra_owned_coordinates=buffer_locs,
                    clear_extra_owner=clear_device_buffer_owner,
                )
                HiSparseCoordinator._debug_device_lifecycle_snapshot(
                    self, req, buffer_locs, stage="finish_after"
                )
            else:
                # Target owns the numerical physical-slot namespace shared by
                # target/draft tensors. A draft mirror clears only its local
                # side-buffer aliases. It must neither clear the canonical
                # mapping nor return the target allocator's pages, so cleanup
                # remains safe even if mirror/owner call order changes.
                clear_device_buffer_owner()

        host_indices = self.req_to_host_pool[req.req_pool_idx]
        host_indices = host_indices[host_indices >= 0]
        self._debug_validate_host_request_slots(req, host_indices, stage="finish")
        if host_indices.numel() > 0 and self._host_slot_owner is self:
            self.mem_pool_host.free(host_indices)
            self._debug_validate_host_allocator_after_free(
                req, host_indices.numel(), stage="finish"
            )

        # clear req info
        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.req_pool_idx] = False
        getattr(self, "_req_c4_retired_len", {}).pop(req.req_pool_idx, None)
        getattr(self, "_req_c4_written_len", {}).pop(req.req_pool_idx, None)
        self.active_hisparse_reqs.pop(req.req_pool_idx, None)
        self._clear_residency_state(req.req_pool_idx)

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        token_position_space: Literal["compressed", "full"] = "compressed",
        num_steps: int = 1,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices.

        Multi-step speculative calls use req-major tensors shaped
        ``[num_reqs, num_steps, top_k]`` and a flat req-major sequence-length
        vector. The kernel processes all steps for one request in a single
        block so LRU state and extra-page draft slots remain graph-stable.
        """
        num_reqs = req_pool_indices.size(0)
        needed = num_reqs * num_steps

        if needed > self.top_k_device_locs_buffer.shape[0]:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "HiSparse multi-step output buffer is too small during CUDA "
                    f"Graph capture: need {needed}, have "
                    f"{self.top_k_device_locs_buffer.shape[0]}"
                )
            self.top_k_device_locs_buffer = torch.full(
                (needed, self.top_k),
                -1,
                dtype=torch.int32,
                device=self.device,
            )

        top_k_indices = self.top_k_device_locs_buffer[:needed]
        if num_steps > 1:
            top_k_indices = top_k_indices.view(num_reqs, num_steps, self.top_k)
        top_k_indices.fill_(-1)

        swap_seq_lens = compressed_seq_lens
        swap_top_k_result = top_k_result
        if token_position_space == "full" and self.is_dsv4_hisparse:
            if num_steps > 1:
                seq_lens_for_compare = compressed_seq_lens.view(
                    num_reqs, num_steps
                ).unsqueeze(2)
            else:
                seq_lens_for_compare = compressed_seq_lens.unsqueeze(1)
            valid_compressed_token = (
                (top_k_result >= 0)
                & (top_k_result < seq_lens_for_compare)
                & ((top_k_result + 1) % self.compress_ratio == 0)
            )
            swap_top_k_result = torch.where(
                valid_compressed_token,
                top_k_result // self.compress_ratio,
                torch.full_like(top_k_result, -1),
            )
            if num_steps > 1:
                swap_seq_lens = (
                    compressed_seq_lens.view(num_reqs, num_steps) // self.compress_ratio
                ).reshape(-1)
            else:
                swap_seq_lens = compressed_seq_lens // self.compress_ratio
        elif token_position_space != "compressed":
            assert (
                token_position_space == "full"
            ), f"Unsupported token_position_space={token_position_space}"

        # This validator intentionally performs host reads (``.item()`` and
        # ``.cpu()``).  Those synchronize the current stream and are illegal
        # while a CUDA graph is being captured.  The crash under investigation
        # occurs after the long-context graph threshold forces eager execution,
        # so skipping capture preserves the exact replay signal without
        # perturbing graph construction or replay.
        if (
            self.debug_validate_swap_in
            and num_steps == 1
            and not torch.cuda.is_current_stream_capturing()
        ):
            self._validate_swap_in_metadata(
                req_pool_indices=req_pool_indices,
                seq_lens=swap_seq_lens,
                top_k_tokens=swap_top_k_result,
                layer_id=layer_id,
            )

        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
        swap_in_fn(
            top_k_tokens=swap_top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=swap_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=self.mem_pool_device.page_size if num_steps > 1 else 1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            num_steps=num_steps,
            req_to_token=self.req_to_token_pool.req_to_token,
            full_to_hisparse_device_index_mapping=(
                self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
            ),
        )
        return top_k_indices

    def _validate_swap_in_metadata(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> None:
        """Synchronously validate HiSparse swap-in metadata for crash replay.

        This is intentionally gated by ``SGLANG_HISPARSE_DEBUG_VALIDATE_SWAP_IN``:
        it performs GPU reductions and host reads that are too expensive for the
        production path.  A missing host location is valid only when the token is
        already resident in the per-request device buffer, or when it is the
        newest token handled by the reserved slot in the swap-in kernel.
        """
        num_real_reqs = int(self.num_real_reqs.item())
        if num_real_reqs <= 0:
            return

        req_pool_indices = req_pool_indices[:num_real_reqs].to(torch.int64)
        seq_lens = seq_lens[:num_real_reqs].to(torch.int64)
        top_k_tokens = top_k_tokens[:num_real_reqs].to(torch.int64)

        max_req_slot = self.req_to_host_pool.shape[0]
        invalid_req = (req_pool_indices < 0) | (req_pool_indices >= max_req_slot)
        if torch.any(invalid_req):
            raise RuntimeError(
                "HiSparse swap-in invalid request pool indices: "
                f"req_pool_indices={req_pool_indices.cpu().tolist()} "
                f"max_req_slot={max_req_slot} layer_id={layer_id}"
            )

        # The CUDA kernel has a short-sequence fast path that only consumes the
        # first ``seq_len`` entries.  CUDA-graph capture intentionally leaves
        # the remaining Top-K entries at -1, so validating all NUM_TOP_K slots
        # would reject legal capture placeholders before the fast-path return.
        # Long sequences execute the hash/miss path and must have a valid token
        # in every Top-K slot; keep the strict checks for those rows.
        long_sequence = seq_lens > self.device_buffer_size
        invalid_token = long_sequence.unsqueeze(1) & (
            (top_k_tokens < 0) | (top_k_tokens >= seq_lens.unsqueeze(1))
        )
        safe_tokens = top_k_tokens.clamp(min=0, max=self.req_to_host_pool.shape[1] - 1)
        host_locs = self.req_to_host_pool[req_pool_indices.unsqueeze(1), safe_tokens]

        buffer_tokens = self.req_device_buffer_tokens[
            layer_id, req_pool_indices, : self.device_buffer_size
        ].to(torch.int64)
        is_device_hit = torch.any(
            top_k_tokens.unsqueeze(2) == buffer_tokens.unsqueeze(1), dim=2
        )
        is_newest = top_k_tokens == (seq_lens.unsqueeze(1) - 1)
        missing_host = (
            long_sequence.unsqueeze(1) & (host_locs < 0) & ~is_device_hit & ~is_newest
        )
        invalid = invalid_token | missing_host
        if not torch.any(invalid):
            return

        bad_rows, bad_cols = torch.where(invalid)
        limit = min(16, bad_rows.numel())
        details = []
        for j in range(limit):
            row = int(bad_rows[j].item())
            col = int(bad_cols[j].item())
            req_idx = int(req_pool_indices[row].item())
            details.append(
                {
                    "batch_row": row,
                    "topk_col": col,
                    "req_pool_idx": req_idx,
                    "seq_len": int(seq_lens[row].item()),
                    "token": int(top_k_tokens[row, col].item()),
                    "host_loc": int(host_locs[row, col].item()),
                    "device_hit": bool(is_device_hit[row, col].item()),
                    "is_newest": bool(is_newest[row, col].item()),
                    "host_allocated_len": int(
                        self.req_to_host_pool_allocated_len[req_idx]
                    ),
                }
            )
        raise RuntimeError(
            "HiSparse swap-in metadata validation failed before CUDA kernel: "
            f"layer_id={layer_id} invalid_count={int(invalid.sum().item())} "
            f"num_real_reqs={num_real_reqs} details={details}"
        )
