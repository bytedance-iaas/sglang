"""
Life cycle of a request in the decode server

1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributed import ProcessGroup

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    poll_and_all_reduce,
    poll_and_all_reduce_with_staging,
    prepare_abort,
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.common import (
    kv_to_page_indices,
    maybe_cache_unfinished_req,
    page_align_floor,
    release_kv_cache,
    state_page_size_from_allocator,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    KVCache,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.observability.req_time_stats import (
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.utils import get_num_new_pages
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import ServerArgs

CLIP_MAX_NEW_TOKEN = envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.get()


def _is_fake_transfer(req: Req, server_args: ServerArgs) -> bool:
    return req.bootstrap_host == FAKE_BOOTSTRAP_HOST or (
        req.bootstrap_host is None
        and server_args.disaggregation_transfer_backend == "fake"
    )


def _bootstrap_addr(req: Req) -> str:
    # FIXME: make a property of a req
    return NetworkAddress(req.bootstrap_host, req.bootstrap_port).to_host_port_str()


class DecodeReqToTokenPool:
    """
    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        # +1 padding row at index 0; see ReqToTokenPool for rationale.
        self._alloc_size = size + pre_alloc_size + 1
        self.max_context_len = max_context_len
        self.device = device
        self.pre_alloc_size = pre_alloc_size
        with memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )

        self.free_slots = list(range(1, self._alloc_size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: List["Req"]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        assert (
            len(reusing) <= 1
        ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
            for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: "Req"):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))


class HybridMambaDecodeReqToTokenPool(HybridReqToTokenPool):
    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: "Mamba2CacheParams",
        mamba_layer_ids: List[int],
        speculative_num_draft_tokens: int,
        enable_mamba_extra_buffer: bool,
        pre_alloc_size: int,
        enable_overlap_schedule: bool,
        mamba_size: int = None,
        start_layer: int = None,
    ):
        DecodeReqToTokenPool.__init__(
            self,
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        # Each request needs 1 main mamba slot + ping-pong slots when extra_buffer is enabled.
        # Cap the pool at max concurrent requests * slots_per_req to avoid allocating failed.
        slots_per_req = 1 + (
            self.mamba_ping_pong_track_buffer_size if enable_mamba_extra_buffer else 0
        )
        max_slots_needed = (size + pre_alloc_size) * slots_per_req
        if mamba_size is not None:
            effective_mamba_size = max(mamba_size, max_slots_needed)
            if mamba_size < max_slots_needed:
                logger.warning(
                    "mamba_size (%d) is less than decode side's max_slots_needed (%d = %d reqs * %d slots/req), "
                    "raising effective_mamba_size to %d",
                    mamba_size,
                    max_slots_needed,
                    size + pre_alloc_size,
                    slots_per_req,
                    effective_mamba_size,
                )
        else:
            effective_mamba_size = max_slots_needed
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            mamba_size=effective_mamba_size,
            mamba_spec_state_size=size + pre_alloc_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=self.enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))
        self.mamba_pool.clear()


@dataclass
class DecodeRequest:
    req: Req
    kv_receiver: CommonKVReceiver
    waiting_for_input: bool = False
    metadata_buffer_index: int = -1

    @property
    def seqlen(self) -> int:
        return self.req.seqlen

    @property
    def priority(self) -> Optional[int]:
        return self.req.priority


class DecodePreallocQueue:
    """
    Store the requests that are preallocating.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        transfer_queue: DecodeTransferQueue,
        tree_cache: BasePrefixCache,
        gloo_group: ProcessGroup,
        tp_rank: int,
        tp_size: int,
        dp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        pp_rank: int,
        num_reserved_decode_tokens: int,
        transfer_backend: TransferBackend,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        self.transfer_queue = transfer_queue
        self.tree_cache = tree_cache
        self.gloo_group = gloo_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens
        self.pp_rank = pp_rank
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        self.transfer_backend = transfer_backend
        # Queue for requests pending pre-allocation
        self.queue: List[DecodeRequest] = []
        self.retracted_queue: List[Req] = []
        self.pending_reqs: List[DecodeRequest] = []
        self._ensure_retry_count: Dict[str, int] = {}
        self._max_ensure_retries: int = 15  # scheduling cycles
        self._ensure_last_attempt_time: Dict[str, float] = {}
        self._ensure_retry_interval: float = 1.0  # seconds
        self._last_prealloc_admission_log_time: float = 0.0
        self._prealloc_admission_log_interval: float = 5.0
        hisparse_coordinator = getattr(self.scheduler, "hisparse_coordinator", None)
        self._relax_decode_output_reserve = (
            envs.SGLANG_HISPARSE_RELAX_DECODE_OUTPUT_RESERVE.get()
            and self.scheduler.enable_hisparse
            and isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
            and getattr(hisparse_coordinator, "is_dsv4_hisparse", False)
        )
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        if self.enable_staging and self.is_mla_backend:
            raise RuntimeError(
                "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                "(e.g. GQA, MHA). MLA models should not set this flag."
            )
        self.kv_manager = self._init_kv_manager()
        if self.enable_staging:
            self.transfer_queue._init_staging_handler(self.kv_manager)

        if (
            self.scheduler.tp_worker.is_hybrid_swa
            and not self._uses_swa_tail_prealloc()
        ):
            # Fallback for SWA allocators that still allocate the SWA pool at
            # full prompt length.
            self.max_total_num_tokens = min(
                self.max_total_num_tokens,
                self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
            )

    def _uses_swa_tail_prealloc(self) -> bool:
        return (
            isinstance(self.token_to_kv_pool, (SWAKVPool, DeepSeekV4TokenToKVPool))
            and self.token_to_kv_pool_allocator.page_size > 1
            and hasattr(self.token_to_kv_pool_allocator, "alloc_extend_swa_tail")
        )

    def _logical_kv_page_size(self) -> int:
        return getattr(
            self.token_to_kv_pool_allocator,
            "logical_page_size",
            self.token_to_kv_pool_allocator.page_size,
        )

    def _swa_tail_len(self, seq_len: int) -> int:
        if not self._uses_swa_tail_prealloc() or seq_len <= 0:
            return max(seq_len, 0)

        window_size = self.scheduler.sliding_window_size
        if window_size is None or window_size <= 0:
            return seq_len

        page_size = self._logical_kv_page_size()
        window_start = max(0, seq_len - window_size)
        window_start = (window_start // page_size) * page_size
        return seq_len - window_start

    def _is_dsv4_hisparse_swa_tail_prealloc(self) -> bool:
        coordinator = getattr(self.scheduler, "hisparse_coordinator", None)
        return (
            self.scheduler.enable_hisparse
            and isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
            and coordinator is not None
            and coordinator.is_dsv4_hisparse
            and self._uses_swa_tail_prealloc()
        )

    def _uses_hisparse_eagle_draft_logical_slots(self) -> bool:
        coordinator = getattr(self.scheduler, "hisparse_coordinator", None)
        return (
            self._is_dsv4_hisparse_swa_tail_prealloc()
            and self.scheduler.spec_algorithm.is_eagle()
            and self.scheduler.server_args.speculative_eagle_topk == 1
            and coordinator is not None
            and coordinator.supports_hisparse_draft_slots()
            and hasattr(self.token_to_kv_pool_allocator, "alloc_extend_logical_only")
        )

    def _hisparse_eagle_draft_logical_reserve_tokens(
        self, n_active: Optional[int] = None
    ) -> int:
        if not self._uses_hisparse_eagle_draft_logical_slots():
            return 0

        if n_active is None:
            n_active = self._active_req_count()
        if n_active <= 0:
            return 0

        draft_tokens = self.scheduler.server_args.speculative_num_draft_tokens
        if draft_tokens is None:
            draft_tokens = self.scheduler.server_args.speculative_num_steps or 0

        page_size = self._logical_kv_page_size()
        tokens_per_req = max(1, int(draft_tokens) + 1)
        pages_per_req = max(1, (tokens_per_req + page_size - 1) // page_size)
        return n_active * pages_per_req * page_size

    def _enable_decode_radix_prefix_reuse(self) -> bool:
        if not self.scheduler.server_args.disaggregation_decode_enable_radix_cache:
            return False
        if not self._is_dsv4_hisparse_swa_tail_prealloc():
            return True

        # DSV4 HiSparse can safely reuse decode-side full logical pages only in
        # the C4 host-prefix-cache mode.  The matched prefix is capped later so
        # it never crosses the SWA tail and never exceeds the already-resident
        # C4 host mirror prefix.
        coordinator = getattr(self.scheduler, "hisparse_coordinator", None)
        return (
            coordinator is not None
            and coordinator.is_dsv4_hisparse
            and coordinator.host_radix_cache is None
        )

    def _dsv4_hisparse_safe_logical_prefix_len(
        self,
        *,
        fill_len: int,
        matched_prefix_len: int,
        c4_prefix_len: Optional[int],
    ) -> int:
        """Cap DSV4 HiSparse full-prefix reuse before the SWA tail.

        The current transfer protocol still uses one full-token decode_prefix_len
        to choose the prefill-side send start.  Keep the reused logical prefix
        within both the SWA-safe full range and the already-resident C4 host
        prefix, so SWA mappings stay newly allocated and the C4 host mirror has
        no holes.
        """
        if (
            matched_prefix_len <= 0
            or not self._is_dsv4_hisparse_swa_tail_prealloc()
        ):
            return matched_prefix_len

        page_size = self._logical_kv_page_size()
        swa_tail_len = self._swa_tail_len(fill_len)
        swa_tail_start = max(fill_len - swa_tail_len, 0)
        safe_prefix_len = min(
            matched_prefix_len,
            page_align_floor(swa_tail_start, page_size),
        )

        if c4_prefix_len is not None:
            c4_full_prefix_len = (
                c4_prefix_len * self.token_to_kv_pool_allocator.compress_ratio
            )
            safe_prefix_len = min(
                safe_prefix_len,
                page_align_floor(c4_full_prefix_len, page_size),
            )

        return max(0, page_align_floor(safe_prefix_len, page_size))

    def _swa_retractable_len(self, req: Req) -> int:
        if not self._uses_swa_tail_prealloc():
            return len(req.origin_input_ids) + len(req.output_ids)
        return self._swa_tail_len(len(req.origin_input_ids)) + len(req.output_ids)

    def _prealloc_kv_lens(self, req: Req) -> Tuple[int, int]:
        allocated_kv_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        if self._uses_swa_tail_prealloc():
            return allocated_kv_len, self._swa_tail_len(allocated_kv_len)
        return allocated_kv_len, allocated_kv_len

    def _prealloc_required_tokens(self, req: Req) -> Tuple[int, int]:
        full_len, swa_len = self._prealloc_kv_lens(req)
        return (
            full_len + self.num_reserved_decode_tokens,
            swa_len + self.num_reserved_decode_tokens,
        )

    def _output_reserve_tokens_for_admission(self, req: Req) -> int:
        max_new_tokens = min(
            req.sampling_params.max_new_tokens,
            CLIP_MAX_NEW_TOKEN,
        )
        if self._relax_decode_output_reserve:
            return min(max_new_tokens, self.num_reserved_decode_tokens)
        return max_new_tokens

    def _dsv4_c4_host_output_reserve_tokens_for_admission(self, req: Req) -> int:
        if not self._is_dsv4_hisparse_swa_tail_prealloc():
            return 0

        max_new_tokens = min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
        committed_output_tokens = max(len(req.output_ids) - 1, 0)
        remaining_output_tokens = max(0, max_new_tokens - committed_output_tokens)
        # C4 host mirror grows on compressed-token boundaries during decode.
        # Unlike logical full/SWA reserve, a missing C4 host slot cannot be fixed
        # by retracting a running request after the backup point is reached, so
        # keep the CLIP_MAX_NEW_TOKEN bound instead of the smaller relaxed cap.
        compress_ratio = self.token_to_kv_pool_allocator.compress_ratio
        return (remaining_output_tokens + compress_ratio - 1) // compress_ratio

    def _dsv4_c4_host_decode_reserve_tokens(
        self, extra_decode_reqs: Optional[List[DecodeRequest]] = None
    ) -> int:
        if not self._is_dsv4_hisparse_swa_tail_prealloc():
            return 0

        reqs: List[Req] = []
        reqs.extend(self.scheduler.running_batch.reqs)
        reqs.extend(decode_req.req for decode_req in self.transfer_queue.queue)
        reqs.extend(self.scheduler.waiting_queue)
        if extra_decode_reqs:
            reqs.extend(decode_req.req for decode_req in extra_decode_reqs)

        seen = set()
        reserve_tokens = 0
        for req in reqs:
            if req.rid in seen:
                continue
            seen.add(req.rid)
            reserve_tokens += self._dsv4_c4_host_output_reserve_tokens_for_admission(
                req
            )
        return reserve_tokens

    def _future_full_tokens_for_admission(
        self,
        req: Req,
        *,
        origin_input_len: int,
        prefix_len: int,
        retractable_tokens: int,
    ) -> int:
        return (
            origin_input_len
            - prefix_len
            + self._output_reserve_tokens_for_admission(req)
            - retractable_tokens
        )

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()

        attn_tp_size = get_attention_tp_size()
        kv_args.engine_rank = self.tp_rank % (attn_tp_size)

        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.ps.dp_rank
        if self.scheduler.enable_hisparse:
            # Direct-to-host: register host pool pointers so P writes to D's host memory
            host_pool = self.scheduler.hisparse_coordinator.mem_pool_host
            kv_data_ptrs, kv_data_lens, kv_item_lens = (
                host_pool.get_contiguous_buf_infos()
            )
        else:
            kv_data_ptrs, kv_data_lens, kv_item_lens = (
                self.token_to_kv_pool.get_contiguous_buf_infos()
            )
        if self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        # HiSparse Host pool has page_size=1; use it when hisparse is enabled
        kv_args.page_size = (
            1 if self.scheduler.enable_hisparse else self.token_to_kv_pool.page_size
        )

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )

        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool,
            total_kv_layers=self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=getattr(self, "req_to_token_pool", None),
        )

        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.ps.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.DECODE,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # Staging buffer setup (only when heterogeneous TP staging is enabled)
        if self.enable_staging and not self.is_mla_backend:
            kv_pool_for_heads = self.token_to_kv_pool
            if hasattr(kv_pool_for_heads, "full_kv_pool"):
                kv_pool_for_heads = kv_pool_for_heads.full_kv_pool
            per_rank_kv_heads = getattr(kv_pool_for_heads, "head_num", 0)
            if per_rank_kv_heads > 0:
                kv_args.kv_head_num = per_rank_kv_heads
                kv_args.total_kv_head_num = per_rank_kv_heads * attn_tp_size
            if hasattr(kv_manager, "set_kv_buffer_tensors"):
                kv_pool = kv_pool_for_heads
                if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                    kv_manager.set_kv_buffer_tensors(
                        kv_pool.k_buffer, kv_pool.v_buffer, kv_pool.page_size
                    )
        return kv_manager

    def add(self, req: Req, is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        if self._check_if_req_exceed_kv_capacity(req):
            return

        req.skip_swa_radix_cache_insert = (
            self._enable_decode_radix_prefix_reuse()
            and self._is_dsv4_hisparse_swa_tail_prealloc()
        )

        if is_retracted:
            req.retraction_mb_id = None
            self.retracted_queue.append(req)
        else:
            decode_req = self._create_receiver_and_enqueue(req)

            # NOTE: fake transfer does not need to resolve prefill dp rank in the pending queue
            if _is_fake_transfer(req, self.scheduler.server_args):
                decode_req.kv_receiver.init(0)
                return

            # Fast path: cache-only lookup, no network calls
            prefill_dp_rank = self._resolve_prefill_dp_rank(req)
            logger.debug(f"prefill_dp_rank: {prefill_dp_rank}")
            if prefill_dp_rank is not None:
                decode_req.kv_receiver.init(prefill_dp_rank)
                return

            self.pending_reqs.append(decode_req)

    def _match_prefix_and_lock(
        self, req: Req, token_ids: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Match a request against the decode-side radix cache, lock the matched
        node to prevent eviction, and return the matched prefix information.
        """
        result = match_prefix_for_req(
            self.tree_cache,
            req,
            token_ids if token_ids is not None else req.origin_input_ids,
            cow_mamba=self.tree_cache.supports_mamba(),
            include_req=True,
            ignore_component_types=(
                (ComponentType.SWA,)
                if getattr(req, "skip_swa_radix_cache_insert", False)
                else ()
            ),
        )
        prefix_indices = result.device_indices
        last_device_node = result.last_device_node
        # Always lock to match aggregated scheduling behavior
        self.tree_cache.inc_lock_ref(last_device_node)

        return prefix_indices, len(prefix_indices)

    def _resolve_prefill_dp_rank(self, req: Req) -> Optional[int]:
        prefill_info = self.kv_manager.prefill_info_table.get(_bootstrap_addr(req))
        # If None, it will go to the slow path and resolve prefill_info by _ensure_prefill_info then cache it
        if prefill_info is None:
            return None

        if req.disagg_prefill_dp_rank is not None:
            return req.disagg_prefill_dp_rank

        if prefill_info.dp_size == 1:
            return 0

        if (
            prefill_info.follow_bootstrap_room
            and not envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.get()
        ):
            return req.bootstrap_room % prefill_info.dp_size

        return None

    def _create_receiver_and_enqueue(self, req: Req) -> DecodeRequest:
        backend = (
            TransferBackend.FAKE
            if _is_fake_transfer(req, self.scheduler.server_args)
            else self.transfer_backend
        )
        kv_receiver_class = get_kv_class(backend, KVClassType.RECEIVER)

        kv_receiver = kv_receiver_class(
            mgr=self.kv_manager,
            bootstrap_addr=_bootstrap_addr(req),
            bootstrap_room=req.bootstrap_room,
        )

        decode_req = DecodeRequest(req=req, kv_receiver=kv_receiver)
        self.queue.append(decode_req)
        return decode_req

    def _clear_receiver(self, decode_req: DecodeRequest) -> None:
        kv_receiver = decode_req.kv_receiver
        if kv_receiver is None:
            return
        if hasattr(kv_receiver, "abort"):
            kv_receiver.abort()
        if hasattr(kv_receiver, "clear"):
            kv_receiver.clear()
        decode_req.kv_receiver = None

    def abort_requests(self, rid: str, abort_all: bool = False) -> None:
        """Abort requests before KV preallocation has been committed.

        Requests in this queue have a receiver/handshake in progress but have
        not yet been moved to the transfer queue.  Cleaning the receiver here
        keeps abort_all from leaving dead waiters that would otherwise rely on a
        later poll to notice FINISH_ABORT.
        """
        remaining_queue: List[DecodeRequest] = []
        aborted: List[Req] = []
        aborted_ids = set()

        for decode_req in self.queue:
            req = decode_req.req
            if abort_all or req.rid.startswith(rid):
                if not isinstance(req.finished_reason, FINISH_ABORT):
                    prepare_abort(req, "Request aborted.", status_code=HTTPStatus.GONE)
                self._clear_receiver(decode_req)
                aborted.append(req)
                aborted_ids.add(id(decode_req))
            else:
                remaining_queue.append(decode_req)

        if aborted_ids:
            self.queue = remaining_queue
            self.pending_reqs = [
                decode_req
                for decode_req in self.pending_reqs
                if id(decode_req) not in aborted_ids
            ]
            for req in aborted:
                self.scheduler.output_streamer.stream_output([req], req.return_logprob)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.output_streamer.stream_output([req], req.return_logprob)
            return True
        if self._uses_swa_tail_prealloc():
            _, swa_required = self._prealloc_required_tokens(req)
            swa_capacity = self.token_to_kv_pool_allocator.size_swa
            if swa_required > swa_capacity:
                message = (
                    f"Request {req.rid} requires too many SWA KV tokens for "
                    f"decode preallocation: {swa_required} > {swa_capacity}"
                )
                logger.error(message)
                prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
                self.scheduler.output_streamer.stream_output([req], req.return_logprob)
                return True
        return False

    def extend(self, reqs: List[Req], is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        for req in reqs:
            self.add(req, is_retracted=is_retracted)

    def resume_retracted_reqs(
        self, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        # TODO refactor the scheduling part, reuse with the unified engine logic as much as possible

        # allocate memory
        resumed_reqs = []
        indices_to_remove = set()
        uses_swa_tail_prealloc = self._uses_swa_tail_prealloc()
        if uses_swa_tail_prealloc:
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(count_retracted=False)
            )
        else:
            full_allocatable_tokens = self._allocatable_token_budgets(
                count_retracted=False
            )

        for i, req in enumerate(self.retracted_queue):
            if rids_to_check is not None and req.rid not in rids_to_check:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            full_required, swa_required = self._prealloc_required_tokens(req)
            if full_required > full_allocatable_tokens:
                break
            if uses_swa_tail_prealloc and swa_required > swa_allocatable_tokens:
                break

            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            self._pre_alloc(req)
            full_allocatable_tokens -= full_required
            if uses_swa_tail_prealloc:
                swa_allocatable_tokens -= swa_required

            # load from cpu, release the cpu copy
            req.load_kv_cache(self.req_to_token_pool, self.token_to_kv_pool_allocator)

        self.retracted_queue = [
            entry
            for i, entry in enumerate(self.retracted_queue)
            if i not in indices_to_remove
        ]

        return resumed_reqs

    def _update_handshake_waiters(
        self, rids_to_check: Optional[List[str]] = None
    ) -> None:
        if not self.queue:
            return

        # Still poll if any receiver was aborted, otherwise it stays stuck.
        if all(decode_req.waiting_for_input for decode_req in self.queue) and not any(
            getattr(decode_req.kv_receiver, "conclude_state", None) == KVPoll.Failed
            for decode_req in self.queue
        ):
            return

        polls = poll_and_all_reduce(
            [decode_req.kv_receiver for decode_req in self.queue], self.gloo_group
        )

        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                decode_req.waiting_for_input = True
                decode_req.req.time_stats.set_bootstrap_done_time()
            elif poll == KVPoll.Failed:
                error_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                if self.scheduler.metrics_reporter.enable_metrics:
                    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def _maybe_log_prealloc_admission_blocked(
        self,
        *,
        reason: str,
        full_allocatable_tokens: int,
        swa_allocatable_tokens: int,
        c4_host_allocatable_tokens: Optional[int],
        c4_host_required_tokens: int,
        hisparse_req_budget,
        hisparse_avail: Optional[int],
    ) -> None:
        now = time.monotonic()
        if (
            now - self._last_prealloc_admission_log_time
            < self._prealloc_admission_log_interval
        ):
            return
        self._last_prealloc_admission_log_time = now

        coordinator = getattr(self.scheduler, "hisparse_coordinator", None)
        full_evictable_tokens = (
            self.tree_cache.full_evictable_size()
            if self._enable_decode_radix_prefix_reuse()
            else 0
        )
        logical_allocator = getattr(
            self.token_to_kv_pool_allocator, "logical_attn_allocator", None
        )
        full_available_tokens = (
            logical_allocator.full_available_size()
            if logical_allocator is not None
            and hasattr(logical_allocator, "full_available_size")
            else None
        )
        logical_available_tokens = (
            logical_allocator.available_size()
            if logical_allocator is not None
            and hasattr(logical_allocator, "available_size")
            else None
        )
        c4_host_reclaimable_tokens = (
            coordinator.dsv4_c4_host_reclaimable_tokens()
            if coordinator is not None
            and hasattr(coordinator, "dsv4_c4_host_reclaimable_tokens")
            else None
        )
        eagle_draft_reserve_tokens = (
            self._hisparse_eagle_draft_logical_reserve_tokens()
        )
        ready_reqs = sum(1 for req in self.queue if req.waiting_for_input)
        running_reqs = len(self.scheduler.running_batch.reqs)
        transfer_reqs = len(self.transfer_queue.queue)
        idle_admission_deadlock = (
            ready_reqs > 0 and running_reqs == 0 and transfer_reqs == 0
        )
        logger.info(
            "Decode prealloc admission limited: reason=%s, queue=%d, "
            "ready=%d, pending=%d, transfer=%d, running=%d, retracted=%d, "
            "req_slots=%d, req_slot_capacity=%d, req_slot_prealloc_capacity=%d, "
            "metadata_slots=%d, full_allocatable_tokens=%d, "
            "full_evictable_tokens=%d, full_available_tokens=%s, "
            "logical_available_tokens=%s, swa_allocatable_tokens=%d, "
            "c4_host_allocatable_tokens=%s, c4_host_required_tokens=%d, "
            "c4_host_reclaimable_tokens=%s, hisparse_req_budget=%s, "
            "eagle_draft_reserve_tokens=%d, "
            "hisparse_avail=%s, hisparse_padded_buffer_size=%s, "
            "hisparse_top_k=%s, hisparse_device_buffer_size=%s, "
            "max_running_requests=%s, idle_admission_deadlock=%s, "
            "relaxed_output_reserve=%s, decode_output_reserve_cap=%s",
            reason,
            len(self.queue),
            ready_reqs,
            len(self.pending_reqs),
            transfer_reqs,
            running_reqs,
            len(self.retracted_queue),
            self.req_to_token_pool.available_size(),
            self.req_to_token_pool.size,
            getattr(self.req_to_token_pool, "pre_alloc_size", 0),
            self.req_to_metadata_buffer_idx_allocator.available_size(),
            full_allocatable_tokens,
            full_evictable_tokens,
            full_available_tokens,
            logical_available_tokens,
            swa_allocatable_tokens,
            c4_host_allocatable_tokens,
            c4_host_required_tokens,
            c4_host_reclaimable_tokens,
            hisparse_req_budget,
            eagle_draft_reserve_tokens,
            hisparse_avail,
            getattr(coordinator, "padded_buffer_size", None),
            getattr(coordinator, "top_k", None),
            getattr(coordinator, "device_buffer_size", None),
            getattr(self.scheduler, "max_running_requests", None),
            idle_admission_deadlock,
            self._relax_decode_output_reserve,
            self.num_reserved_decode_tokens
            if self._relax_decode_output_reserve
            else CLIP_MAX_NEW_TOKEN,
        )

    def _ensure_prefill_info(
        self, addr_to_reqs: Dict[str, List[DecodeRequest]]
    ) -> Tuple[Dict[str, List[DecodeRequest]], List[DecodeRequest]]:
        """Non-blocking ensure parallel info for each addr.
        Returns (ready_addrs, remaining_reqs)."""
        ready: Dict[str, List[DecodeRequest]] = {}
        remaining: List[DecodeRequest] = []

        now = time.monotonic()
        for bootstrap_addr, reqs in addr_to_reqs.items():
            last_attempt = self._ensure_last_attempt_time.get(bootstrap_addr)
            if last_attempt is not None and (
                now - last_attempt < self._ensure_retry_interval
            ):
                remaining.extend(reqs)
                continue

            self._ensure_last_attempt_time[bootstrap_addr] = now

            if self.kv_manager.try_ensure_parallel_info(bootstrap_addr):
                if bootstrap_addr in self._ensure_retry_count:
                    del self._ensure_retry_count[bootstrap_addr]
                if bootstrap_addr in self._ensure_last_attempt_time:
                    del self._ensure_last_attempt_time[bootstrap_addr]
                ready[bootstrap_addr] = reqs
                continue

            count = self._ensure_retry_count.get(bootstrap_addr, 0) + 1
            self._ensure_retry_count[bootstrap_addr] = count

            if count >= self._max_ensure_retries:
                error_msg = f"Could not fetch prefill parallel info from {bootstrap_addr} after {count} attempts"
                logger.error(error_msg)
                for decode_req in reqs:
                    decode_req.kv_receiver.abort()
                del self._ensure_retry_count[bootstrap_addr]
                del self._ensure_last_attempt_time[bootstrap_addr]
            else:
                remaining.extend(reqs)

        return ready, remaining

    def _resolve_pending_reqs(self) -> None:
        """Batch-resolve prefill_dp_ranks for pending requests and initialize receivers."""
        if not self.pending_reqs:
            return

        # Group pending requests by bootstrap_addr
        addr_to_reqs: Dict[str, List[DecodeRequest]] = {}
        for decode_req in self.pending_reqs:
            addr = _bootstrap_addr(decode_req.req)
            addr_to_reqs.setdefault(addr, []).append(decode_req)

        # Pass 1: ensure parallel info for each addr
        ready_addrs, remaining = self._ensure_prefill_info(addr_to_reqs)

        resolved: List[Tuple[DecodeRequest, int]] = []
        for bootstrap_addr, decode_reqs in ready_addrs.items():
            need_query: List[DecodeRequest] = []
            for decode_req in decode_reqs:
                prefill_dp_rank = self._resolve_prefill_dp_rank(decode_req.req)
                if prefill_dp_rank is not None:
                    resolved.append((decode_req, prefill_dp_rank))
                else:
                    need_query.append(decode_req)

            # Pass 2: resolve dp rank for addrs whose info is available
            if need_query:
                rooms = [decode_req.req.bootstrap_room for decode_req in need_query]
                room_to_rank = CommonKVReceiver.query_prefill_dp_ranks(
                    bootstrap_addr, rooms
                )
                for decode_req in need_query:
                    prefill_dp_rank = room_to_rank.get(
                        str(decode_req.req.bootstrap_room)
                    )
                    if prefill_dp_rank is not None:
                        resolved.append((decode_req, int(prefill_dp_rank)))
                    else:
                        remaining.append(decode_req)

        self.pending_reqs = remaining

        for decode_req, prefill_dp_rank in resolved:
            decode_req.kv_receiver.init(prefill_dp_rank)

    def pop_preallocated(
        self, rids_to_check: Optional[List[str]] = None
    ) -> Tuple[List[DecodeRequest], List[DecodeRequest]]:
        """Pop the preallocated requests from the pending queue (FIFO)."""
        self._resolve_pending_reqs()
        self._update_handshake_waiters(rids_to_check)

        failed_reqs = []
        preallocated_reqs = []
        indices_to_remove = set()

        # We need to make sure that the sum of inflight tokens and allocatable tokens is greater than maximum input+output length of each inflight request
        # Otherwise it is possible for one request running decode out of memory, while all other requests are in the transfer queue that cannot be retracted.
        retractable_tokens = sum(
            len(r.origin_input_ids) + len(r.output_ids)
            for r in self.scheduler.running_batch.reqs
        )

        uses_swa_tail_prealloc = self._uses_swa_tail_prealloc()
        swa_allocatable_tokens = 0
        if uses_swa_tail_prealloc:
            retractable_swa_tokens = sum(
                self._swa_retractable_len(r) for r in self.scheduler.running_batch.reqs
            )
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    retractable_swa_tokens=retractable_swa_tokens,
                    count_retracted=True,
                )
            )
        else:
            retractable_swa_tokens = 0
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens, count_retracted=True
            )

        # Sort by priority before any index-based bookkeeping so that both the
        # abort-scan loop and the preallocation loop operate on the same order.
        if self.scheduler.enable_priority_scheduling:
            priority_sign = (
                1 if self.scheduler.schedule_low_priority_values_first else -1
            )
            self.queue.sort(key=lambda r: r.req.priority * priority_sign)

        # First, remove all failed requests from the queue
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                self.scheduler.output_streamer.stream_output(
                    [decode_req.req],
                    decode_req.req.return_logprob,
                )
                failed_reqs.append(decode_req)
                indices_to_remove.add(i)

        # HiSparse physical constraint: max requests by device buffer capacity.
        # Each admitted req needs padded_buffer_size from hisparse device pool.
        # waiting_queue reqs already have device buffers (allocated in admit_request_direct),
        # only transfer_queue reqs are pending device buffer allocation.
        hisparse_req_budget = float("inf")
        hisparse_avail = None
        c4_host_allocatable_tokens = None
        c4_host_required_tokens = 0
        c4_host_reserved_decode_reqs: List[DecodeRequest] = []
        if self.scheduler.enable_hisparse:
            hisparse_avail = (
                self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
            )
            hisparse_req_budget = max(
                0,
                hisparse_avail // self.scheduler.hisparse_coordinator.padded_buffer_size
                - len(self.transfer_queue.queue),
            )
            if (
                self.scheduler.hisparse_coordinator.is_dsv4_hisparse
                and self.scheduler.hisparse_coordinator.host_radix_cache is None
            ):
                c4_host_decode_reserve_tokens = (
                    self._dsv4_c4_host_decode_reserve_tokens()
                )
                self.scheduler.hisparse_coordinator.reclaim_dsv4_c4_host_prefix_cache(
                    c4_host_decode_reserve_tokens
                )
                c4_host_allocatable_tokens = (
                    self.scheduler.hisparse_coordinator.mem_pool_host.available_size()
                    - c4_host_decode_reserve_tokens
                )

        # Then, preallocate the remaining requests if possible
        blocked_reason = None
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if i in indices_to_remove:
                continue

            if not decode_req.waiting_for_input:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                blocked_reason = "req_to_token_slots"
                break

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                blocked_reason = "metadata_buffer_slots"
                break

            if hisparse_req_budget <= 0:
                blocked_reason = "hisparse_c4_hot_req_budget"
                break

            # Memory estimation: don't add if the projected memory cannot be met
            # TODO: add new_token ratio
            c4_host_required_tokens = 0
            c4_host_len = None
            c4_host_prefix_len = None
            prefix_lock_acquired = False
            origin_input_len = len(decode_req.req.origin_input_ids)
            fill_len = origin_input_len + max(len(decode_req.req.output_ids) - 1, 0)
            if self._enable_decode_radix_prefix_reuse():
                # Match prefix against decode's radix cache.
                prefix_indices, prefix_len = self._match_prefix_and_lock(decode_req.req)
                prefix_lock_acquired = True
                # Align prefix_len down to page boundary so both prefill and
                # decode agree on the page-aligned split point for KV transfer.
                page_size = self._logical_kv_page_size()
                if page_size > 1 and prefix_len % page_size != 0:
                    prefix_len = page_align_floor(prefix_len, page_size)
                    prefix_indices = prefix_indices[:prefix_len]
                if self._is_dsv4_hisparse_swa_tail_prealloc():
                    (
                        c4_host_required_tokens,
                        c4_host_len,
                        c4_host_prefix_len,
                    ) = self._dsv4_c4_host_required_tokens_for_admission(
                        decode_req.req, fill_len
                    )
                    safe_prefix_len = self._dsv4_hisparse_safe_logical_prefix_len(
                        fill_len=fill_len,
                        matched_prefix_len=prefix_len,
                        c4_prefix_len=c4_host_prefix_len,
                    )
                    if safe_prefix_len < prefix_len:
                        matched_prefix_len = prefix_len
                        self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                        prefix_indices, prefix_len = self._match_prefix_and_lock(
                            decode_req.req,
                            decode_req.req.origin_input_ids[:safe_prefix_len],
                        )
                        logger.debug(
                            "DSV4 HiSparse logical prefix capped: req=%s "
                            "matched=%d safe=%d c4_prefix=%s swa_tail=%d fill=%d",
                            decode_req.req.rid,
                            matched_prefix_len,
                            prefix_len,
                            c4_host_prefix_len,
                            self._swa_tail_len(fill_len),
                            fill_len,
                        )

                required_alloc_tokens = self._required_alloc_tokens(
                    fill_len=fill_len, prefix_len=prefix_len
                )
                # Matching may lock previously-evictable radix pages, so refresh
                # the admission budget against the post-lock pool state before we
                # decide whether this request still fits.
                full_allocatable_tokens = self._allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    count_retracted=True,
                    extra_reserved_reqs=len(preallocated_reqs),
                )
            else:
                prefix_indices = None
                prefix_len = 0
                required_alloc_tokens = origin_input_len

            required_tokens_for_request = (
                required_alloc_tokens + self.num_reserved_decode_tokens
            )

            if (
                max(
                    required_tokens_for_request,
                    self._future_full_tokens_for_admission(
                        decode_req.req,
                        origin_input_len=origin_input_len,
                        prefix_len=prefix_len,
                        retractable_tokens=retractable_tokens,
                    ),
                )
                > full_allocatable_tokens
            ):
                if prefix_lock_acquired:
                    self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                blocked_reason = "full_allocatable_tokens"
                break
            if required_tokens_for_request > full_allocatable_tokens:
                if prefix_lock_acquired:
                    self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                blocked_reason = "full_required_tokens"
                break

            if uses_swa_tail_prealloc:
                _, swa_required = self._prealloc_required_tokens(decode_req.req)
                _, swa_len = self._prealloc_kv_lens(decode_req.req)
                max_new_tokens = self._output_reserve_tokens_for_admission(
                    decode_req.req
                )
                if (
                    max(
                        swa_required,
                        swa_len + max_new_tokens - retractable_swa_tokens,
                    )
                    > swa_allocatable_tokens
                ):
                    if prefix_lock_acquired:
                        self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                    blocked_reason = "swa_allocatable_tokens"
                    break

            if c4_host_allocatable_tokens is not None:
                if c4_host_len is None or c4_host_prefix_len is None:
                    (
                        c4_host_required_tokens,
                        c4_host_len,
                        c4_host_prefix_len,
                    ) = self._dsv4_c4_host_required_tokens_for_admission(
                        decode_req.req, fill_len
                    )
                c4_host_required_tokens += (
                    self._dsv4_c4_host_output_reserve_tokens_for_admission(
                        decode_req.req
                    )
                )
                if c4_host_required_tokens > c4_host_allocatable_tokens:
                    c4_host_decode_reserve_tokens = (
                        self._dsv4_c4_host_decode_reserve_tokens(
                            c4_host_reserved_decode_reqs
                        )
                    )
                    if self.scheduler.hisparse_coordinator.ensure_dsv4_c4_host_available(
                        c4_host_required_tokens + c4_host_decode_reserve_tokens
                    ):
                        c4_host_allocatable_tokens = (
                            self.scheduler.hisparse_coordinator.mem_pool_host.available_size()
                            - c4_host_decode_reserve_tokens
                        )
                if c4_host_required_tokens > c4_host_allocatable_tokens:
                    if prefix_lock_acquired:
                        self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                    blocked_reason = "c4_host_allocatable_tokens"
                    break

            dst_kv_indices = self._pre_alloc(decode_req.req, prefix_indices, prefix_len)
            hisparse_req_budget -= 1
            if c4_host_allocatable_tokens is not None:
                c4_host_reserved_decode_reqs.append(decode_req)
                c4_host_allocatable_tokens = (
                    self.scheduler.hisparse_coordinator.mem_pool_host.available_size()
                    - self._dsv4_c4_host_decode_reserve_tokens(
                        c4_host_reserved_decode_reqs
                    )
                )
            # Recompute from actual pool state for the next queue entry.
            # This accounts for page rounding and newly locked evictable cache.
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens,
                count_retracted=True,
                extra_reserved_reqs=len(preallocated_reqs) + 1,
            )
            if uses_swa_tail_prealloc:
                # SWA budget uses simple decrement (no radix cache eviction in
                # the SWA pool, so page-rounding drift is negligible).
                swa_allocatable_tokens -= swa_required
            decode_req.req.cache_protected_len = prefix_len

            metadata_decode_prefix_len = prefix_len
            decode_c4_prefix_len = None
            if self.scheduler.enable_hisparse:
                # Must cast to int32 for ZMQ serialization -- from_zmq reads np.int32.
                coordinator = self.scheduler.hisparse_coordinator
                if (
                    isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
                    and coordinator.is_dsv4_hisparse
                ):
                    if self.transfer_backend != TransferBackend.MOONCAKE:
                        raise NotImplementedError(
                            "DeepSeek V4 HiSparse direct-to-host C4 transfer "
                            "currently requires the Mooncake transfer backend."
                        )
                    page_size = 1  # DSV4 host C4 pool is token-linear.
                    host_c4_len = (
                        self.token_to_kv_pool_allocator.c4_tokens_for_full_tokens(
                            origin_input_len
                        )
                    )
                    decode_c4_prefix_len = min(
                        getattr(
                            decode_req.req, "hisparse_c4_transfer_prefix_len", 0
                        )
                        or 0,
                        host_c4_len,
                    )
                    c4_transfer_prefix_len = min(
                        prefix_len,
                        decode_c4_prefix_len
                        * self.token_to_kv_pool_allocator.compress_ratio,
                    )
                    # Prefill counts full KV pages from decode_prefix_len, so
                    # keep this prefix full-page aligned. The C4 sender still
                    # skips C4 slots below decode_c4_prefix_len.
                    metadata_decode_prefix_len = (
                        c4_transfer_prefix_len // self.token_to_kv_pool.page_size
                    ) * self.token_to_kv_pool.page_size
                    kv_indices_tensor = coordinator.req_to_host_pool[
                        decode_req.req.req_pool_idx, :host_c4_len
                    ]
                    if kv_indices_tensor.numel() != host_c4_len or torch.any(
                        kv_indices_tensor < 0
                    ):
                        raise RuntimeError(
                            "DeepSeek V4 HiSparse C4 host mirror is incomplete "
                            f"for req {decode_req.req.rid}: "
                            f"host_c4_len={host_c4_len}, "
                            f"decode_c4_prefix_len={decode_c4_prefix_len}."
                        )
                    kv_indices = kv_indices_tensor.cpu().numpy().astype(np.int32)
                else:
                    kv_indices = (
                        dst_kv_indices[: origin_input_len - prefix_len]
                        .cpu()
                        .numpy()
                        .astype(np.int32)
                    )
                    page_size = 1  # host pool page_size
            else:
                # Only send delta indices (beyond prefix) to prefill.
                kv_indices = (
                    self.req_to_token_pool.req_to_token[decode_req.req.req_pool_idx][
                        prefix_len:origin_input_len
                    ]
                    .cpu()
                    .numpy()
                )
                page_size = self.token_to_kv_pool_allocator.page_size

            seq_len = len(decode_req.req.origin_input_ids)
            swa_state_page_size = state_page_size_from_allocator(
                self.token_to_kv_pool_allocator
            )

            # Generic SWA/NSA state transfer is independent of decode_prefix_len:
            # prefill still sends full state page lists on the last chunk. Keep
            # destination state lists full-sized until DSV4 gets an explicit
            # component-wise transfer plan.
            def _mamba_payload():
                return [
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        decode_req.req.req_pool_idx
                    ]
                    .cpu()
                    .numpy()
                ]

            def _swa_payload():
                window_size = self.scheduler.sliding_window_size
                window_start = max(0, seq_len - window_size)
                window_start = page_align_floor(window_start, swa_state_page_size)
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx, window_start:seq_len
                ]
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                return kv_to_page_indices(
                    window_kv_indices_swa.cpu().numpy(), swa_state_page_size
                )

            def _nsa_payload():
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx, :seq_len
                ]
                # Indexer lives on device pool; always use device page_size
                device_page_size = self.token_to_kv_pool.page_size
                return kv_to_page_indices(
                    kv_indices_full.cpu().numpy(), device_page_size
                )

            state_types = self.kv_manager.kv_args.state_types
            state_indices: Optional[List] = []
            for st in state_types:
                if st == StateType.MAMBA:
                    state_indices.append(_mamba_payload())
                elif st == StateType.SWA:
                    state_indices.append(_swa_payload())
                elif st == StateType.NSA:
                    state_indices.append(_nsa_payload())
                else:
                    state_indices.append(None)

            decode_req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert decode_req.metadata_buffer_index is not None
            page_indices = kv_to_page_indices(kv_indices, page_size)
            metadata_kwargs = {"decode_prefix_len": metadata_decode_prefix_len}
            if decode_c4_prefix_len is not None:
                metadata_kwargs["decode_c4_prefix_len"] = decode_c4_prefix_len
            try:
                decode_req.kv_receiver.send_metadata(
                    page_indices,
                    decode_req.metadata_buffer_index,
                    state_indices,
                    **metadata_kwargs,
                )
                if (
                    self.transfer_queue.enable_staging
                    and hasattr(decode_req.kv_receiver, "require_staging")
                    and decode_req.kv_receiver.require_staging
                ):
                    self.transfer_queue.staging_handler.register_decode_req(
                        decode_req.req.bootstrap_room, decode_req
                    )
            except Exception as e:
                error_message = (
                    "Decode preallocation metadata send failed for request "
                    f"rank={self.tp_rank} {decode_req.req.rid=} "
                    f"{decode_req.req.bootstrap_room=}: {e}"
                )
                logger.exception(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.output_streamer.stream_output(
                    [decode_req.req],
                    decode_req.req.return_logprob,
                )
                if self.scheduler.enable_hisparse:
                    self.scheduler.hisparse_coordinator.request_finished(decode_req.req)
                release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                if (
                    self.transfer_queue.enable_staging
                    and self.transfer_queue.staging_handler is not None
                ):
                    self.transfer_queue.staging_handler.unregister_decode_req(
                        decode_req.req.bootstrap_room
                    )
                if decode_req.metadata_buffer_index is not None:
                    self.req_to_metadata_buffer_idx_allocator.free(
                        decode_req.metadata_buffer_index
                    )
                    decode_req.metadata_buffer_index = -1
                if decode_req.kv_receiver is not None:
                    if hasattr(decode_req.kv_receiver, "abort"):
                        decode_req.kv_receiver.abort()
                    if hasattr(decode_req.kv_receiver, "clear"):
                        decode_req.kv_receiver.clear()
                    decode_req.kv_receiver = None
                indices_to_remove.add(i)
                if self.scheduler.metrics_reporter.enable_metrics:
                    self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                continue
            preallocated_reqs.append(decode_req)
            indices_to_remove.add(i)
            decode_req.req.time_stats.set_decode_transfer_queue_entry_time()

        if blocked_reason is not None:
            self._maybe_log_prealloc_admission_blocked(
                reason=blocked_reason,
                full_allocatable_tokens=full_allocatable_tokens,
                swa_allocatable_tokens=swa_allocatable_tokens,
                c4_host_allocatable_tokens=c4_host_allocatable_tokens,
                c4_host_required_tokens=c4_host_required_tokens,
                hisparse_req_budget=hisparse_req_budget,
                hisparse_avail=hisparse_avail,
            )

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs, failed_reqs

    @property
    def num_tokens_pre_allocated(self):
        return sum(
            len(decode_req.req.fill_ids) for decode_req in self.transfer_queue.queue
        )

    def _need_space_for_single_req(
        self, retractable_tokens: Optional[int] = None
    ) -> int:
        need_space_for_single_req = (
            max(
                [
                    self._output_reserve_tokens_for_admission(x)
                    + len(x.origin_input_ids)
                    - retractable_tokens
                    for x in self.scheduler.running_batch.reqs
                ]
            )
            if retractable_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
            else 0
        )
        return need_space_for_single_req

    def _active_req_count(self, extra_reserved_reqs: int = 0) -> int:
        return (
            len(self.scheduler.running_batch.reqs)
            + len(self.transfer_queue.queue)
            + len(self.scheduler.waiting_queue)
            + extra_reserved_reqs
        )

    def _active_reserved_tokens(
        self, n_active: Optional[int] = None, extra_reserved_reqs: int = 0
    ) -> int:
        if n_active is None:
            n_active = self._active_req_count(extra_reserved_reqs)
        return self.num_reserved_decode_tokens * n_active

    def _swa_aware_allocatable_token_budgets(
        self,
        retractable_tokens: Optional[int] = None,
        retractable_swa_tokens: Optional[int] = None,
        count_retracted: bool = True,
    ) -> Tuple[int, int]:
        n_active = self._active_req_count()
        reserved_tokens = self._active_reserved_tokens(n_active)

        full_allocatable_tokens = self._allocatable_token_budgets(
            retractable_tokens=retractable_tokens,
            count_retracted=count_retracted,
            reserved_tokens=reserved_tokens,
        )

        return full_allocatable_tokens, self._swa_tail_allocatable_token_budget(
            retractable_tokens=retractable_tokens,
            retractable_swa_tokens=retractable_swa_tokens,
            count_retracted=count_retracted,
            n_active=n_active,
            reserved_tokens=reserved_tokens,
        )

    def _allocatable_token_budgets(
        self,
        retractable_tokens: Optional[int] = None,
        count_retracted: bool = True,
        extra_reserved_reqs: int = 0,
        reserved_tokens: Optional[int] = None,
    ) -> int:
        need_space_for_single_req = self._need_space_for_single_req(retractable_tokens)
        if reserved_tokens is None:
            reserved_tokens = self._active_reserved_tokens(
                extra_reserved_reqs=extra_reserved_reqs
            )

        if self.scheduler.enable_hisparse:
            # HiSparse pre-alloc allocates logical full indices.  For DSV4 SWA-tail
            # preallocation, SWA has a separate budget check below, so the full
            # budget must not be capped by logical_attn_allocator.available_size()
            # (which is min(full_available, swa_available) for hybrid SWA).
            logical_allocator = self.token_to_kv_pool_allocator.logical_attn_allocator
            if self._uses_swa_tail_prealloc():
                available_size = logical_allocator.full_available_size()
            else:
                available_size = logical_allocator.available_size()
            if self._enable_decode_radix_prefix_reuse():
                available_size += self.tree_cache.full_evictable_size()
        elif self._uses_swa_tail_prealloc():
            available_size = self.token_to_kv_pool_allocator.full_available_size()
            if self._enable_decode_radix_prefix_reuse():
                available_size += self.tree_cache.evictable_size()
        else:
            available_size = self.token_to_kv_pool_allocator.available_size()
            # Include evictable decode-radix cache entries in the budget -- they
            # can be freed on demand before allocation.
            if self._enable_decode_radix_prefix_reuse():
                available_size += self.tree_cache.evictable_size()
        allocatable_tokens = available_size - max(
            reserved_tokens, need_space_for_single_req
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            allocatable_tokens -= self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )

        if count_retracted:
            for req in self.retracted_queue:
                full_required, _ = self._prealloc_required_tokens(req)
                allocatable_tokens -= full_required

        return allocatable_tokens

    def _swa_tail_allocatable_token_budget(
        self,
        retractable_tokens: Optional[int] = None,
        retractable_swa_tokens: Optional[int] = None,
        count_retracted: bool = True,
        n_active: Optional[int] = None,
        reserved_tokens: Optional[int] = None,
    ) -> int:
        need_swa_space_for_single_req = self._need_space_for_single_req(
            retractable_tokens
        )
        if (
            retractable_swa_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
        ):
            need_swa_space_for_single_req = max(
                self._swa_tail_len(len(x.origin_input_ids))
                + self._output_reserve_tokens_for_admission(x)
                - retractable_swa_tokens
                for x in self.scheduler.running_batch.reqs
            )

        if n_active is None:
            n_active = self._active_req_count()
        if reserved_tokens is None:
            reserved_tokens = self._active_reserved_tokens(n_active)

        # SWA growth is bounded by the sliding window: once a req's SWA
        # footprint reaches `sliding_window_size`, further decode tokens
        # evict old ones and net growth is zero. The linear reservation
        # `num_reserved_decode_tokens * n_active` (correct for the full
        # pool) over-reserves SWA in steady state. Cap by the actual
        # remaining headroom up to per-req window cap.
        window_size = self.scheduler.sliding_window_size or 0
        swa_total = self.token_to_kv_pool_allocator.size_swa
        swa_used = swa_total - self.token_to_kv_pool_allocator.swa_available_size()
        swa_growth_potential = max(0, n_active * window_size - swa_used)
        swa_reserved_tokens = min(reserved_tokens, swa_growth_potential)
        eagle_draft_reserve_tokens = (
            self._hisparse_eagle_draft_logical_reserve_tokens(n_active)
        )
        swa_allocatable_tokens = (
            self.token_to_kv_pool_allocator.swa_available_size()
            - max(
                swa_reserved_tokens,
                need_swa_space_for_single_req,
                eagle_draft_reserve_tokens,
            )
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            prebuilt_reserved_tokens = self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )
            prebuilt_n = len(self.scheduler.last_batch.reqs)
            prebuilt_swa_growth = max(0, prebuilt_n * window_size - swa_used)
            swa_allocatable_tokens -= min(prebuilt_reserved_tokens, prebuilt_swa_growth)
            swa_allocatable_tokens -= (
                self._hisparse_eagle_draft_logical_reserve_tokens(prebuilt_n)
            )

        if count_retracted:
            for req in self.retracted_queue:
                _, swa_required = self._prealloc_required_tokens(req)
                swa_allocatable_tokens -= swa_required

        return swa_allocatable_tokens

    def _required_alloc_tokens(self, *, fill_len: int, prefix_len: int) -> int:
        page_size = self._logical_kv_page_size()
        if page_size == 1:
            return fill_len - prefix_len

        num_new_pages = get_num_new_pages(
            seq_lens=torch.tensor([fill_len], dtype=torch.int64),
            prefix_lens=torch.tensor([prefix_len], dtype=torch.int64),
            page_size=page_size,
        )
        return num_new_pages * page_size

    def _dsv4_c4_host_required_tokens_for_admission(
        self, req: Req, fill_len: int
    ) -> Tuple[int, int, int]:
        """Return missing C4 host slots and ready C4 prefix length.

        Pending C4 prefixes can share already allocated destination slots across
        concurrent transfers, so they reduce the allocation requirement.  They
        are not ready cache hits, so the returned prefix length remains the
        ready prefix used for logical-prefix safety and transfer skipping.
        """
        coordinator = self.scheduler.hisparse_coordinator
        host_c4_len = self.token_to_kv_pool_allocator.c4_tokens_for_full_tokens(
            fill_len
        )
        ready_prefix_c4_len, _, _ = coordinator._match_c4_host_prefix(
            req, host_c4_len
        )
        allocated_prefix_c4_len, _, _ = coordinator._match_c4_host_prefix(
            req, host_c4_len, include_pending=True
        )
        ready_prefix_c4_len = min(ready_prefix_c4_len, host_c4_len)
        allocated_prefix_c4_len = min(allocated_prefix_c4_len, host_c4_len)
        return (
            host_c4_len - allocated_prefix_c4_len,
            host_c4_len,
            ready_prefix_c4_len,
        )

    def _pre_alloc(
        self,
        req: Req,
        prefix_indices: Optional[torch.Tensor] = None,
        prefix_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Pre-allocate the memory for req_to_token and token_kv_pool"""
        if prefix_len is None:
            prefix_len = 0

        req_pool_indices = self.req_to_token_pool.alloc([req])

        assert (
            req_pool_indices is not None
        ), "req_pool_indices is full! There is a bug in memory estimation."

        fill_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        req.kv_allocated_len = fill_len
        req.kv_committed_len = fill_len
        retracted_hisparse_restore = False

        if prefix_len > 0:
            if prefix_indices is None:
                raise RuntimeError(
                    f"Missing prefix indices for req {req.rid}: prefix_len={prefix_len}"
                )
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(0, prefix_len)), prefix_indices
            )

        # TODO(retraction): when retraction is implemented with radix cache
        # awareness, a retracted request should re-match the tree here
        # instead of re-allocating from scratch. See resume_retracted_reqs.
        delta_len = fill_len - prefix_len
        required_alloc_tokens = self._required_alloc_tokens(
            fill_len=fill_len, prefix_len=prefix_len
        )

        # Evict cached entries if the pool doesn't have enough free pages.
        if self.scheduler.enable_hisparse:
            logical_allocator = self.token_to_kv_pool_allocator.logical_attn_allocator
            available_for_prealloc = (
                logical_allocator.full_available_size()
                if self._uses_swa_tail_prealloc()
                else logical_allocator.available_size()
            )
        else:
            available_for_prealloc = self.token_to_kv_pool_allocator.available_size()
        if (
            self._enable_decode_radix_prefix_reuse()
            and available_for_prealloc < required_alloc_tokens
        ):
            num_to_evict = required_alloc_tokens - available_for_prealloc
            result = self.tree_cache.evict(EvictParams(num_tokens=num_to_evict))
            if self.scheduler.enable_hisparse:
                available_after_evict = (
                    logical_allocator.full_available_size()
                    if self._uses_swa_tail_prealloc()
                    else logical_allocator.available_size()
                )
            else:
                available_after_evict = self.token_to_kv_pool_allocator.available_size()
            if available_after_evict < required_alloc_tokens:
                logger.warning(
                    f"Eviction insufficient: needed {required_alloc_tokens} tokens, "
                    f"available {available_after_evict} "
                    f"after evicting {result.num_tokens_evicted}/{num_to_evict} tokens. "
                    f"evictable_size={self.tree_cache.evictable_size()}, "
                    f"protected_size={self.tree_cache.protected_size()}, "
                    f"fill_len={fill_len}, prefix_len={prefix_len}, delta_len={delta_len}, "
                    f"page_size={self._logical_kv_page_size()}, "
                    f"req={req.rid}"
                )

        if self.scheduler.enable_hisparse:
            # Direct-to-host path: allocate logical full/SWA indices and allocate
            # host indices for RDMA destination.  DSV4 can reuse a full logical
            # prefix only when it is capped before the SWA tail; all other
            # HiSparse paths stay on the historical prefix_len == 0 semantics.
            coordinator = self.scheduler.hisparse_coordinator
            device = self.token_to_kv_pool_allocator.device
            use_dsv4_safe_prefix = (
                coordinator.is_dsv4_hisparse and self._uses_swa_tail_prealloc()
            )
            if prefix_len > 0 and not use_dsv4_safe_prefix:
                raise AssertionError(
                    "HiSparse decode prefix reuse is only supported for "
                    "DeepSeek V4 SWA-tail preallocation."
                )
            if prefix_len > 0 and prefix_indices is None:
                raise RuntimeError(
                    f"Missing HiSparse prefix indices for req {req.rid}: "
                    f"prefix_len={prefix_len}"
                )

            last_loc = (
                prefix_indices[-1:].to(dtype=torch.int64, device=device)
                if prefix_len > 0
                else torch.tensor([-1], dtype=torch.int64, device=device)
            )
            if use_dsv4_safe_prefix:
                swa_tail_len = self._swa_tail_len(fill_len)
                swa_tail_start = fill_len - swa_tail_len
                page_size = self._logical_kv_page_size()
                if page_size > 1:
                    swa_tail_start = page_align_floor(
                        swa_tail_start, page_size
                    )
                req.swa_evicted_seqlen = max(
                    req.swa_evicted_seqlen, swa_tail_start
                )
                if delta_len < swa_tail_len:
                    raise RuntimeError(
                        "Unsafe DSV4 HiSparse logical prefix crosses SWA tail: "
                        f"req={req.rid}, prefix_len={prefix_len}, "
                        f"fill_len={fill_len}, swa_tail_len={swa_tail_len}."
                    )
                kv_loc = self.token_to_kv_pool_allocator.alloc_extend_swa_tail(
                    prefix_lens=torch.tensor(
                        [prefix_len], dtype=torch.int64, device=device
                    ),
                    prefix_lens_cpu=torch.tensor([prefix_len], dtype=torch.int64),
                    seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                    seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                    last_loc=last_loc,
                    extend_num_tokens=delta_len,
                    swa_tail_len=swa_tail_len,
                )
            else:
                kv_loc = self.token_to_kv_pool_allocator.alloc_logical_only(
                    prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                    prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                    seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                    seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                    last_loc=last_loc,
                    extend_num_tokens=fill_len,
                )
            if coordinator.is_dsv4_hisparse:
                # DSV4 host mirror stores only compressed C4 tokens. Restore the
                # cached prefix first, then allocate only the suffix that prefill
                # still needs to transfer.
                host_len = self.token_to_kv_pool_allocator.c4_tokens_for_full_tokens(
                    fill_len
                )
                if getattr(req, "hisparse_retract_host_indices", None) is not None:
                    retracted_hisparse_restore = True
                    host_indices = torch.empty(
                        (host_len,), dtype=torch.int64, device=coordinator.device
                    )
                else:
                    coordinator.reserve_dsv4_c4_host_transfer_slots(req, host_len)
                    host_indices = coordinator.req_to_host_pool[
                        req.req_pool_idx, :host_len
                    ]
            else:
                # Allocate host indices for the RDMA transfer target.
                host_indices = coordinator.mem_pool_host.alloc(fill_len)
                if host_indices is None:
                    raise RuntimeError(
                        f"HiSparse host mem pool alloc failed for {fill_len} tokens "
                        f"in _pre_alloc (req {req.rid})"
                    )
                host_indices = host_indices.to(device=coordinator.device)
                coordinator.req_to_host_pool[req.req_pool_idx, :fill_len] = (
                    host_indices
                )
        elif self.token_to_kv_pool_allocator.page_size == 1:
            kv_loc = self.token_to_kv_pool_allocator.alloc(delta_len)
        else:
            device = self.token_to_kv_pool_allocator.device
            last_loc = (
                prefix_indices[-1:].to(dtype=torch.int64, device=device)
                if prefix_len > 0
                else torch.tensor([-1], dtype=torch.int64, device=device)
            )
            if self._uses_swa_tail_prealloc() and prefix_len == 0:
                # Tail-only SWA allocation: only valid when prefix_len == 0.
                # When prefix_len > 0 (radix cache hit), we fall back to
                # alloc_extend which allocates SWA at full page count; the
                # SWA budget in that case may slightly under-estimate.
                kv_loc = self.token_to_kv_pool_allocator.alloc_extend_swa_tail(
                    prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                    prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                    seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                    seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                    last_loc=last_loc,
                    extend_num_tokens=fill_len,
                    swa_tail_len=self._swa_tail_len(fill_len),
                )
            else:
                kv_loc = self.token_to_kv_pool_allocator.alloc_extend(
                    prefix_lens=torch.tensor(
                        [prefix_len], dtype=torch.int64, device=device
                    ),
                    prefix_lens_cpu=torch.tensor([prefix_len], dtype=torch.int64),
                    seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                    seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                    last_loc=last_loc,
                    extend_num_tokens=delta_len,
                )

        assert kv_loc is not None, (
            f"KV cache is full! Bug in memory estimation. "
            f"available={self.token_to_kv_pool_allocator.available_size()}, "
            f"evictable={self.tree_cache.evictable_size()}, "
            f"protected={self.tree_cache.protected_size()}, "
            f"required_alloc={required_alloc_tokens}, delta={delta_len}, "
            f"fill={fill_len}, prefix={prefix_len}, "
            f"page_size={self._logical_kv_page_size()}, "
            f"req={req.rid}"
        )

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(prefix_len, prefix_len + len(kv_loc))), kv_loc
        )

        # Truncate fill_ids to kv_committed_len so cache_unfinished_req only
        # inserts committed KV into the radix tree. The last output token
        # hasn't had KV committed yet (fill_ids is 1 ahead).
        req.fill_ids = (req.origin_input_ids + req.output_ids)[: req.kv_committed_len]
        # Set prefix_indices so downstream consumers (init_next_round_input,
        # prepare_for_extend) see the correct prefix length. In the agg path
        # this is done inside init_next_round_input, but decode-disagg needs
        # allocation info before batch assembly so we set it here.
        req.prefix_indices = (
            prefix_indices if prefix_len > 0 else torch.empty((0,), dtype=torch.int64)
        )
        req.set_extend_input_len(len(req.fill_ids) - prefix_len)

        if retracted_hisparse_restore:
            if not coordinator.restore_retracted_decode_req(req):
                raise RuntimeError(
                    f"Missing DSV4 HiSparse retracted state for req {req.rid}."
                )
            host_indices = coordinator.req_to_host_pool[req.req_pool_idx, :host_len]

        # Return the transfer destination indices:
        if self.scheduler.enable_hisparse:
            return host_indices
        return kv_loc


class DecodeTransferQueue:
    """
    Store the requests that is polling kv
    """

    def __init__(
        self,
        gloo_group: ProcessGroup,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        tp_rank: int,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[DecodeRequest] = []
        self.gloo_group = gloo_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tree_cache = tree_cache
        self.spec_algorithm = scheduler.spec_algorithm
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.staging_handler = None

    def add(self, decode_req: DecodeRequest) -> None:
        self.queue.append(decode_req)

    def extend(self, decode_reqs: List[DecodeRequest]) -> None:
        self.queue.extend(decode_reqs)
        if self.enable_staging:
            for dr in decode_reqs:
                if (
                    hasattr(dr.kv_receiver, "require_staging")
                    and dr.kv_receiver.require_staging
                ):
                    self.staging_handler.register_decode_req(dr.req.bootstrap_room, dr)

    def _commit_transfer_to_req(self, decode_req: DecodeRequest) -> bool:
        """
        Returns:
            True if the request should be removed from the queue (success or corruption)
            False if metadata not ready yet (keep in queue for next poll)
        """
        idx = decode_req.metadata_buffer_index
        (
            output_id,
            cached_tokens,
            output_token_logprobs_val,
            output_token_logprobs_idx,
            output_top_logprobs_val,
            output_top_logprobs_idx,
            output_topk_p,
            output_topk_index,
            output_hidden_states,
            output_bootstrap_room,
        ) = self.metadata_buffers.get_buf(idx)

        # Validate bootstrap_room to detect context corruption
        actual_room = output_bootstrap_room[0].item()
        expected_room = (
            decode_req.req.bootstrap_room
            if decode_req.req.bootstrap_room is not None
            else 0
        )

        if _is_fake_transfer(decode_req.req, self.scheduler.server_args):
            pass
        elif actual_room == 0:
            # Case 1: Metadata not ready yet (actual_room == 0)
            # Keep request in queue and wait for next poll
            return False
        elif actual_room != expected_room:
            # Case 2: Real corruption detected (mismatch)
            # Abort the request and remove from the queue
            error_msg = (
                f"Context corruption detected: Request {decode_req.req.rid} "
                f"(bootstrap_room={expected_room}) received metadata from "
                f"bootstrap_room={actual_room}. "
                f"Metadata buffer index: {idx}. "
                f"This indicates metadata buffer index collision."
            )
            logger.error(error_msg)
            prepare_abort(
                decode_req.req,
                "Metadata corruption detected - bootstrap_room mismatch",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return True

        # Case 3: Success - commit the transfer
        decode_req.req.output_ids.append(output_id[0].item())
        decode_req.req.cached_tokens = cached_tokens[0].item()
        decode_req.req.cached_tokens_device = cached_tokens[1].item()
        decode_req.req.cached_tokens_host = cached_tokens[2].item()
        decode_req.req.cached_tokens_storage = cached_tokens[3].item()
        if not self.spec_algorithm.is_none():
            decode_req.req.output_topk_p = output_topk_p
            decode_req.req.output_topk_index = output_topk_index
            decode_req.req.hidden_states_tensor = output_hidden_states

        if decode_req.req.return_logprob:
            decode_req.req.logprob.output_token_logprobs_val.append(
                output_token_logprobs_val[0].item()
            )
            decode_req.req.logprob.output_token_logprobs_idx.append(
                output_token_logprobs_idx[0].item()
            )
            decode_req.req.logprob.output_top_logprobs_val.append(
                output_top_logprobs_val[
                    : decode_req.req.logprob.top_logprobs_num
                ].tolist()
            )
            decode_req.req.logprob.output_top_logprobs_idx.append(
                output_top_logprobs_idx[
                    : decode_req.req.logprob.top_logprobs_num
                ].tolist()
            )

        self._publish_transferred_prefix_if_needed(decode_req.req)
        decode_req.kv_receiver.clear()
        decode_req.kv_receiver = None
        decode_req.req.time_stats.set_wait_queue_entry_time()
        return True

    def _publish_transferred_prefix_if_needed(self, req: Req) -> None:
        if not self.scheduler.server_args.disaggregation_decode_enable_radix_cache:
            return
        if not self.scheduler.enable_hisparse:
            return

        coordinator = getattr(self.scheduler, "hisparse_coordinator", None)
        if (
            coordinator is None
            or not getattr(coordinator, "is_dsv4_hisparse", False)
            or coordinator.host_radix_cache is not None
            or req.req_pool_idx is None
            or req.kv_committed_len is None
            or req.kv_committed_len <= 0
            or req.cache_protected_len >= req.kv_committed_len
        ):
            return

        # The transfer has completed, so both logical KV and the C4 host mirror
        # are ready. Publish this committed prefix immediately; otherwise later
        # same-prefix requests allocate the whole long prompt while this request
        # waits to enter decode, which can pin the full/logical pool at 0.99.
        req.fill_ids = (req.origin_input_ids + req.output_ids)[: req.kv_committed_len]
        coordinator._publish_dsv4_c4_host_prompt_prefix(req)
        maybe_cache_unfinished_req(req, self.tree_cache)

    def _poll_with_staging(self) -> list:
        return poll_and_all_reduce_with_staging(
            self.queue, self.staging_handler, self.gloo_group
        )

    def _init_staging_handler(self, kv_manager):
        """Create staging handler from kv_manager. Must be called exactly once."""
        from sglang.srt.disaggregation.common.staging_handler import (
            DecodeStagingHandler,
        )

        self.staging_handler = DecodeStagingHandler.create(
            kv_manager, self.scheduler, self.tp_rank
        )
        kv_manager._staging_handler = self.staging_handler

    def _clear_receiver(self, decode_req: DecodeRequest) -> None:
        kv_receiver = decode_req.kv_receiver
        if kv_receiver is None:
            return
        if hasattr(kv_receiver, "abort"):
            kv_receiver.abort()
        if hasattr(kv_receiver, "clear"):
            kv_receiver.clear()
        decode_req.kv_receiver = None

    def _release_metadata_buffer(self, decode_req: DecodeRequest) -> None:
        idx = decode_req.metadata_buffer_index
        if idx is not None and idx >= 0:
            self.req_to_metadata_buffer_idx_allocator.free(idx)
            decode_req.metadata_buffer_index = -1

    def _unregister_staging_req(self, decode_req: DecodeRequest) -> None:
        if (
            self.enable_staging
            and self.staging_handler is not None
            and self.staging_handler.is_staging_room(decode_req.req.bootstrap_room)
        ):
            self.staging_handler.unregister_decode_req(decode_req.req.bootstrap_room)

    def _cleanup_failed_decode_req(self, decode_req: DecodeRequest) -> None:
        self._clear_receiver(decode_req)
        if self.scheduler.enable_hisparse:
            self.scheduler.hisparse_coordinator.request_finished(decode_req.req)
        release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
        self._unregister_staging_req(decode_req)
        self._release_metadata_buffer(decode_req)

    def abort_requests(self, rid: str, abort_all: bool = False) -> None:
        """Abort requests after KV preallocation but before transfer completion."""
        remaining_queue: List[DecodeRequest] = []
        aborted = False

        for decode_req in self.queue:
            req = decode_req.req
            if abort_all or req.rid.startswith(rid):
                if not isinstance(req.finished_reason, FINISH_ABORT):
                    prepare_abort(req, "Request aborted.", status_code=HTTPStatus.GONE)
                self.scheduler.output_streamer.stream_output([req], req.return_logprob)
                self._cleanup_failed_decode_req(decode_req)
                aborted = True
            else:
                remaining_queue.append(decode_req)

        if aborted:
            self.queue = remaining_queue

    def pop_transferred(self, rids_to_check: Optional[List[str]] = None) -> List[Req]:
        if not self.queue:
            return []

        if self.enable_staging:
            polls = self._poll_with_staging()
        else:
            polls = poll_and_all_reduce(
                [dr.kv_receiver for dr in self.queue], self.gloo_group
            )

        transferred_reqs = []
        indices_to_remove = set()
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Failed:
                error_message = f"Decode transfer failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.output_streamer.stream_output(
                    [decode_req.req],
                    decode_req.req.return_logprob,
                )
                self._clear_receiver(decode_req)
                if self.scheduler.enable_hisparse:
                    self.scheduler.hisparse_coordinator.request_finished(decode_req.req)
                # release pre-allocated kv cache, but don't insert into the tree since it's failed
                release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                indices_to_remove.add(i)
                if self.scheduler.metrics_reporter.enable_metrics:
                    self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                continue
            elif poll == KVPoll.Success:
                should_remove = self._commit_transfer_to_req(decode_req)
                if should_remove:
                    indices_to_remove.add(i)
                    # Check if request was aborted due to corruption
                    if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                        self.scheduler.output_streamer.stream_output(
                            [decode_req.req],
                            decode_req.req.return_logprob,
                        )
                        if self.scheduler.enable_hisparse:
                            self.scheduler.hisparse_coordinator.request_finished(
                                decode_req.req
                            )
                        release_kv_cache(
                            decode_req.req, self.tree_cache, is_insert=False
                        )
                        if self.scheduler.metrics_reporter.enable_metrics:
                            self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                    else:
                        transferred_reqs.append(decode_req.req)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            if self.enable_staging and self.staging_handler.is_staging_room(
                self.queue[i].req.bootstrap_room
            ):
                self.staging_handler.unregister_decode_req(
                    self.queue[i].req.bootstrap_room
                )
            idx = self.queue[i].metadata_buffer_index
            assert idx != -1
            self.req_to_metadata_buffer_idx_allocator.free(idx)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return transferred_reqs


class SchedulerDisaggregationDecodeMixin:
    @torch.no_grad()
    def event_loop_normal_disagg_decode(self: Scheduler):
        """A normal scheduler loop for decode worker in disaggregation mode."""

        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            self.process_decode_queue()
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self: Scheduler):
        self.result_queue = deque()
        self.last_batch: Optional[ScheduleBatch] = None

        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            self.process_decode_queue()
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                self.on_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch

    def _run_batch_prebuilt(
        self: Scheduler, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        if batch.inner_idle_batch is not None:
            idle_batch = batch.inner_idle_batch
            # Reset the inner idle batch to avoid reusing it.
            batch.inner_idle_batch = None
            return self.run_batch(idle_batch)

        return GenerationBatchResult()

    def get_next_disagg_decode_batch_to_run(
        self: Scheduler,
    ) -> Optional[ScheduleBatch]:
        """Process prebuilt batch and schedule the next decode batch."""
        # Process pending prebuilt batch: output processing + filter + merge
        new_prebuilt_batch = self.get_new_prebuilt_batch()
        if new_prebuilt_batch:
            assert self.chunked_req is None
            self.batch_result_processor.process_batch_result_prebuilt(
                new_prebuilt_batch
            )
            new_prebuilt_batch.filter_batch()
            if not new_prebuilt_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = new_prebuilt_batch
                    if self.enable_hisparse:
                        self.running_batch.hisparse_coordinator = (
                            self.hisparse_coordinator
                        )
                else:
                    self.running_batch.merge_batch(new_prebuilt_batch)

        # Schedule decode batch
        if self.running_batch.is_empty():
            ret = None
        else:
            self.running_batch = self.update_running_batch(self.running_batch)
            ret = self.running_batch if not self.running_batch.is_empty() else None

        ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret)
        if ret:
            set_schedule_time_batch(ret)
        return ret

    def get_new_prebuilt_batch(self: Scheduler) -> Optional[ScheduleBatch]:
        """Create a schedulebatch for fake completed prefill"""
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if len(self.waiting_queue) == 0:
            return None

        if self.enable_priority_scheduling:
            self.policy.calc_priority(self.waiting_queue, self.running_batch)

        curr_batch_size = self.running_batch.batch_size()

        batch_size = min(self.req_to_token_pool.size, self.max_running_requests)

        num_not_used_batch = batch_size - curr_batch_size
        if num_not_used_batch <= 0:
            now = time.monotonic()
            last_log_time = getattr(self, "_last_prebuilt_admission_log_time", 0.0)
            if now - last_log_time >= 5.0:
                self._last_prebuilt_admission_log_time = now
                logger.info(
                    "Decode prebuilt admission limited: reason=max_running_requests, "
                    "waiting=%d, running=%d, req_pool_size=%d, max_running_requests=%d",
                    len(self.waiting_queue),
                    curr_batch_size,
                    self.req_to_token_pool.size,
                    self.max_running_requests,
                )
            return None

        # pop req from waiting queue
        can_run_list: List[Req] = []
        waiting_queue: List[Req] = []

        for i in range(len(self.waiting_queue)):
            req = self.waiting_queue[i]
            # we can only add at least `num_not_used_batch` new batch to the running queue
            if i < num_not_used_batch:
                can_run_list.append(req)
                # Decode-radix path: do NOT re-match prefix here.
                # `pop_preallocated` already took a tree snapshot and used it
                # to (1) pre-allocate KV, (2) choose delta pages for transfer,
                # and (3) set cache_protected_len/last_node for correct frees.
                # Re-matching now can observe a newer tree (other reqs may have
                # inserted the same prefix) and overwrite cache_protected_len,
                # making `cache_unfinished_req` free the wrong range (leak).
                # Non-radix decode keeps the original behavior.
                tree_cache = (
                    None
                    if self.server_args.disaggregation_decode_enable_radix_cache
                    else self.tree_cache
                )
                req.init_next_round_input(tree_cache)
                # Truncate fill_ids to kv_committed_len so cache_unfinished_req
                # only sees committed KV (fill_ids includes one uncommitted token).
                if req.kv_committed_len is not None:
                    req.fill_ids = req.fill_ids[: req.kv_committed_len]
                    req.set_extend_input_len(
                        len(req.fill_ids) - len(req.prefix_indices)
                    )
            else:
                waiting_queue.append(req)

        self.waiting_queue = waiting_queue
        if len(can_run_list) == 0:
            return None

        set_time_batch(can_run_list, "set_forward_entry_time")

        # construct a schedule batch with those requests and mark as decode
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )

        # construct fake completed prefill
        new_batch.prepare_for_prebuilt()
        new_batch.process_prebuilt(self.server_args, self.future_map)

        return new_batch

    def process_decode_queue(self: Scheduler):
        if self.server_args.disaggregation_decode_enable_offload_kvcache:
            self.decode_offload_manager.check_offload_progress()

        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return

        if not hasattr(self, "polling_count"):
            self.polling_count = 0
            self.polling_interval = (
                self.server_args.disaggregation_decode_polling_interval
            )

        self.polling_count = (self.polling_count + 1) % self.polling_interval

        if self.polling_count % self.polling_interval == 0:
            req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated()
            self.disagg_decode_transfer_queue.extend(req_conns)
            transferred_reqs = (
                self.disagg_decode_transfer_queue.pop_transferred()
            )  # the requests which kv has arrived
            if self.enable_hisparse:
                for req in transferred_reqs:
                    # Direct-to-host: KV data already in host pool, skip staging
                    self.hisparse_coordinator.admit_request_direct(req)
                self.waiting_queue.extend(transferred_reqs)
            else:
                self.waiting_queue.extend(transferred_reqs)
