# to be combined with the sparse coordinator class and sparse algorithm family

import logging
import os
import time
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch

from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.mem_cache.hisparse_memory_pool import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    DeepSeekV4SingleKVPoolHost,
    HiSparseNSATokenToKVPool,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost
from sglang.srt.utils import get_device_module

device_module = get_device_module()

from sglang.jit_kernel.hisparse import (
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseTokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float
    c4_swap_miss_tokens: Optional[int] = None
    c4_swap_hit_tokens: Optional[int] = None
    c4_swap_miss_rate: Optional[float] = None
    c4_swap_h2d_bytes: Optional[int] = None


class HiSparseStagingBackup(NamedTuple):
    host_indices: torch.Tensor
    device_indices: torch.Tensor
    indexer_ranges: Optional[List[Tuple[int, int, int]]] = None


class DSV4C4PrefixEntry:
    def __init__(self, host_indices: torch.Tensor):
        self.host_indices = host_indices
        self.ref_count = 0


class DSV4C4PrefixCache:
    """Host-prefix cache for DSV4 C4 slots.

    The cache key is the full-token prefix, while the value is the compressed
    C4 host-slot list. This deliberately does not model full KV prefix-cache
    semantics: a C4 hit can skip C4 host backup/allocation, but it cannot skip
    prefill compute by itself.
    """

    def __init__(self, compress_ratio: int, device: str):
        self.compress_ratio = compress_ratio
        self.device = device
        self.enabled = False
        self.cache: Dict[Tuple[Optional[str], Tuple[int, ...]], DSV4C4PrefixEntry] = {}
        self.req_refs: Dict[int, Tuple[Tuple[Optional[str], Tuple[int, ...]], int]] = {}

    def enable(self) -> None:
        self.enabled = True

    def full_token_len(self, c4_len: int) -> int:
        return c4_len * self.compress_ratio

    def prefix_token_ids(self, req: Req) -> List[int]:
        if getattr(req, "fill_ids", None):
            return req.fill_ids
        return req.origin_input_ids + req.output_ids

    def key(self, req: Req, c4_len: int) -> Tuple[Optional[str], Tuple[int, ...]]:
        token_ids = self.prefix_token_ids(req)
        full_len = min(len(token_ids), self.full_token_len(c4_len))
        return (req.extra_key, tuple(token_ids[:full_len]))

    def contains(self, req: Req, c4_len: int) -> bool:
        return self.enabled and c4_len > 0 and self.key(req, c4_len) in self.cache

    def match(
        self, req: Req, c4_len: int
    ) -> Tuple[int, Optional[Tuple[Optional[str], Tuple[int, ...]]], torch.Tensor]:
        if not self.enabled or c4_len <= 0:
            return 0, None, torch.empty(0, dtype=torch.int64, device=self.device)

        token_ids = self.prefix_token_ids(req)
        req_token_len = min(len(token_ids), self.full_token_len(c4_len))
        req_tokens = tuple(token_ids[:req_token_len])
        exact_key = (req.extra_key, req_tokens)
        exact_entry = self.cache.get(exact_key)
        if exact_entry is not None:
            exact_c4_len = min(c4_len, len(exact_entry.host_indices))
            return (
                exact_c4_len,
                exact_key,
                exact_entry.host_indices[:exact_c4_len].to(
                    self.device, non_blocking=True
                ),
            )

        best_key = None
        best_indices = None
        best_c4_len = 0
        for key, entry in self.cache.items():
            extra_key, token_key = key
            if extra_key != req.extra_key:
                continue
            host_indices = entry.host_indices
            max_full_len = min(
                len(token_key),
                len(req_tokens),
                self.full_token_len(min(len(host_indices), c4_len)),
            )
            common_full_len = 0
            while (
                common_full_len < max_full_len
                and req_tokens[common_full_len] == token_key[common_full_len]
            ):
                common_full_len += 1
            candidate_c4_len = common_full_len // self.compress_ratio
            candidate_full_len = self.full_token_len(candidate_c4_len)
            if (
                candidate_c4_len > best_c4_len
                and candidate_full_len > 0
            ):
                best_key = key
                best_indices = host_indices[:candidate_c4_len]
                best_c4_len = candidate_c4_len

        if best_indices is None:
            return 0, None, torch.empty(0, dtype=torch.int64, device=self.device)
        return best_c4_len, best_key, best_indices.to(self.device, non_blocking=True)

    def indices(
        self,
        key: Tuple[Optional[str], Tuple[int, ...]],
        c4_len: int,
    ) -> torch.Tensor:
        entry = self.cache[key]
        return entry.host_indices[:c4_len].to(self.device, non_blocking=True)

    def acquire(
        self,
        req_pool_idx: int,
        key: Optional[Tuple[Optional[str], Tuple[int, ...]]],
        c4_len: int,
    ) -> bool:
        if key is None or c4_len <= 0 or key not in self.cache:
            return False

        old_ref = self.req_refs.get(req_pool_idx)
        if old_ref == (key, c4_len):
            return True
        if old_ref is not None:
            self.release_req(req_pool_idx)

        self.cache[key].ref_count += 1
        self.req_refs[req_pool_idx] = (key, c4_len)
        return True

    def release_req(self, req_pool_idx: int) -> None:
        old_ref = self.req_refs.pop(req_pool_idx, None)
        if old_ref is None:
            return

        key, _ = old_ref
        entry = self.cache.get(key)
        if entry is not None:
            entry.ref_count = max(0, entry.ref_count - 1)

    def retain_req_ref(
        self, req_pool_idx: int
    ) -> Optional[Tuple[Tuple[Optional[str], Tuple[int, ...]], int]]:
        old_ref = self.req_refs.get(req_pool_idx)
        if old_ref is None:
            return None

        key, _ = old_ref
        entry = self.cache.get(key)
        if entry is None:
            return None

        entry.ref_count += 1
        return old_ref

    def release_retained(
        self, retained_ref: Optional[Tuple[Tuple[Optional[str], Tuple[int, ...]], int]]
    ) -> None:
        if retained_ref is None:
            return

        key, _ = retained_ref
        entry = self.cache.get(key)
        if entry is not None:
            entry.ref_count = max(0, entry.ref_count - 1)

    def evict_unref(self, need_tokens: int) -> torch.Tensor:
        if need_tokens <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)

        freed_indices = []
        freed_tokens = 0
        for key in list(self.cache.keys()):
            entry = self.cache[key]
            if entry.ref_count > 0:
                continue
            candidate_indices = torch.unique(entry.host_indices)
            del self.cache[key]

            if self.cache:
                remaining_indices = torch.unique(
                    torch.cat([x.host_indices for x in self.cache.values()])
                )
                candidate_indices = candidate_indices[
                    ~torch.isin(candidate_indices, remaining_indices)
                ]

            if candidate_indices.numel() > 0:
                freed_indices.append(candidate_indices)
                freed_tokens += int(candidate_indices.numel())
            if freed_tokens >= need_tokens:
                break

        if not freed_indices:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        return torch.unique(torch.cat(freed_indices)).to(
            device=self.device, non_blocking=True
        )

    def insert(
        self,
        req: Req,
        c4_len: int,
        req_to_host_pool: torch.Tensor,
    ) -> bool:
        if not self.enabled or c4_len <= 0:
            return False

        key = self.key(req, c4_len)
        if key in self.cache:
            return True

        host_indices = req_to_host_pool[req.req_pool_idx, :c4_len]
        if host_indices.numel() != c4_len or torch.any(host_indices < 0):
            logger.warning(
                "Skip DSV4 C4 host-prefix insert for req %s: "
                "c4_len=%d has missing host slots.",
                req.rid,
                c4_len,
            )
            return False

        self.cache[key] = DSV4C4PrefixEntry(host_indices.detach().clone())
        logger.debug(
            "DSV4 C4 host-prefix cache insert: req=%s c4_len=%d full_tokens=%d",
            req.rid,
            c4_len,
            len(key[1]),
        )
        return True


class HiSparseCoordinator:
    """Coordinates HiSparse-only KV movement and side buffers.

    The non-HiSparse path never enters this coordinator. Within HiSparse, generic
    staging/release helpers keep the common path separate from DSV4 C4 prefix
    reuse and EAGLE/MTP draft-slot handling.
    """

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
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        if self.device_buffer_size < self.top_k:
            raise ValueError(
                "HiSparse device_buffer_size must be no smaller than top_k: "
                f"device_buffer_size={self.device_buffer_size}, top_k={self.top_k}."
            )
        self.device = device
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        if self.is_dsv4_hisparse:
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
            host_size = self.token_to_kv_pool_allocator.size_full // self.compress_ratio
            self.mem_pool_host = DeepSeekV4SingleKVPoolHost(
                self.mem_pool_device, host_size, 1
            )
            self.item_size_bytes = (
                self.mem_pool_host.kv_cache_total_dim
                * self.mem_pool_host.dtype.itemsize
            )
        else:
            assert isinstance(
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            self.mem_pool_device: HiSparseNSATokenToKVPool = (
                self.token_to_kv_pool_allocator.get_kvcache()
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
        self.page_size = self.mem_pool_host.page_size

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
        self.req_draft_buffer_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_compressed_context_len + self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

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
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self._c4_miss_sample_interval_s = float(
            os.getenv("SGLANG_HISPARSE_C4_MISS_SAMPLE_INTERVAL", "-1")
        )
        self._last_c4_miss_sample_time = 0.0
        self._c4_swap_miss_tokens = 0
        self._c4_swap_hit_tokens = 0
        self._c4_swap_h2d_bytes = 0
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        self._skip_first_backup = [False] * max_num_req_slots

        self._c4_prefix_cache = DSV4C4PrefixCache(self.compress_ratio, device)
        self._req_c4_prefix_len: Dict[int, int] = {}
        self._req_c4_written_len: Dict[int, int] = {}
        self._req_host_written_len: Dict[int, int] = {}
        self.host_radix_cache = None

    # ---- DSV4 C4 prefix-cache integration ----

    def set_host_radix_cache(self, cache) -> None:
        self.host_radix_cache = cache
        cache.bind_hisparse_req_pools(
            self.req_to_host_pool,
            self.req_to_token_pool.req_to_token,
        )

    def enable_c4_host_prefix_cache(self) -> bool:
        if not self.is_dsv4_hisparse:
            logger.warning(
                "HiSparse C4 host-prefix cache is only supported for DeepSeek V4."
            )
            return False
        self._c4_prefix_cache.enable()
        logger.warning(
            "Enabled DeepSeek V4 HiSparse C4 host-prefix cache. "
            "This does not enable normal scheduler radix matching."
        )
        return True

    def reclaim_dsv4_c4_host_prefix_cache(self, min_available_tokens: int) -> int:
        if (
            not self.is_dsv4_hisparse
            or self.host_radix_cache is not None
            or min_available_tokens <= 0
        ):
            return 0

        need_tokens = min_available_tokens - self.mem_pool_host.available_size()
        if need_tokens <= 0:
            return 0

        evicted_indices = self._c4_prefix_cache.evict_unref(need_tokens)
        if evicted_indices.numel() == 0:
            return 0

        evicted_indices = torch.unique(evicted_indices)
        self.mem_pool_host.free(evicted_indices)
        return int(evicted_indices.numel())

    def _match_c4_host_prefix(
        self, req: Req, c4_len: int
    ) -> Tuple[int, Optional[Tuple[Optional[str], Tuple[int, ...]]], torch.Tensor]:
        return self._c4_prefix_cache.match(req, c4_len)

    def _insert_c4_host_prefix(self, req: Req, c4_len: int) -> bool:
        return self._c4_prefix_cache.insert(req, c4_len, self.req_to_host_pool)

    def _prompt_c4_len(self, req: Req) -> int:
        prompt_len = len(req.origin_input_ids)
        if prompt_len <= 0 and getattr(req, "fill_ids", None):
            prompt_len = len(req.fill_ids)
        return min(
            self._host_token_len(prompt_len),
            self.req_to_host_pool.shape[1],
        )

    def _publish_dsv4_c4_host_prompt_prefix(self, req: Req) -> int:
        if not self.is_dsv4_hisparse:
            return 0

        prompt_c4_len = self._prompt_c4_len(req)
        if prompt_c4_len <= 0:
            return 0

        written_len = self._req_c4_written_len.get(req.req_pool_idx, 0)
        prefix_len = min(prompt_c4_len, written_len)
        if prefix_len <= 0:
            return 0

        key = self._c4_prefix_cache.key(req, prefix_len)
        key_exists = self._c4_prefix_cache.contains(req, prefix_len)
        current_prefix = self.req_to_host_pool[req.req_pool_idx, :prefix_len]

        if key_exists:
            canonical_indices = self._c4_prefix_cache.indices(key, prefix_len)
        else:
            if not self._insert_c4_host_prefix(req, prefix_len):
                return 0
            canonical_indices = self._c4_prefix_cache.indices(key, prefix_len)

        if key_exists and not torch.equal(current_prefix, canonical_indices):
            current_indices = current_prefix[current_prefix >= 0]
            current_unique = torch.unique(current_indices)
            canonical_unique = torch.unique(canonical_indices)
            duplicate_indices = current_unique[
                ~torch.isin(current_unique, canonical_unique)
            ]
            if duplicate_indices.numel() > 0:
                self.mem_pool_host.free(duplicate_indices)
        self.req_to_host_pool[req.req_pool_idx, :prefix_len] = canonical_indices
        if not self._c4_prefix_cache.acquire(req.req_pool_idx, key, prefix_len):
            return 0

        self._req_c4_prefix_len[req.req_pool_idx] = max(
            self._req_c4_prefix_len.get(req.req_pool_idx, 0),
            prefix_len,
        )
        return prefix_len

    def restore_dsv4_c4_host_prefix_for_req(self, req: Req, c4_len: int) -> int:
        """Populate req_to_host_pool from the C4-only prefix cache.

        This only restores the DSV4 compressed C4 host mirror. It deliberately
        does not claim a normal logical KV prefix, so SWA/C128/indexer/MTP
        transfer semantics stay on the existing stable path.
        """
        if not self.is_dsv4_hisparse or c4_len <= 0:
            req.hisparse_c4_transfer_prefix_len = 0
            return 0

        prefix_len, prefix_key, prefix_host_indices = self._match_c4_host_prefix(
            req, c4_len
        )
        if prefix_len <= 0:
            req.hisparse_c4_transfer_prefix_len = 0
            return 0

        self.req_to_host_pool[req.req_pool_idx, :prefix_len] = prefix_host_indices[
            :prefix_len
        ]
        self._c4_prefix_cache.acquire(req.req_pool_idx, prefix_key, prefix_len)
        self._req_c4_prefix_len[req.req_pool_idx] = prefix_len
        req.hisparse_c4_transfer_prefix_len = prefix_len
        logger.debug(
            "DSV4 C4 host-prefix restore: req=%s c4_prefix_len=%d c4_total_len=%d",
            req.rid,
            prefix_len,
            c4_len,
        )
        return prefix_len

    def _prepare_staging_backup(
        self, req: Req, device_indices: torch.Tensor
    ) -> HiSparseStagingBackup:
        if self.host_radix_cache is not None:
            return self._prepare_host_radix_staging_backup(req, device_indices)
        if self.is_dsv4_hisparse:
            return self._prepare_dsv4_c4_staging_backup(req, device_indices)

        host_indices = self.ensure_host_slots(
            req.req_pool_idx, 0, len(device_indices)
        )
        return HiSparseStagingBackup(host_indices, device_indices)

    def _prepare_dsv4_c4_staging_backup(
        self, req: Req, device_indices: torch.Tensor
    ) -> HiSparseStagingBackup:
        prefill_len = len(device_indices)
        prefix_len, prefix_key, prefix_host_indices = self._match_c4_host_prefix(
            req, prefill_len
        )
        if prefix_len > 0:
            self.req_to_host_pool[req.req_pool_idx, :prefix_len] = (
                prefix_host_indices[:prefix_len]
            )
            self._c4_prefix_cache.acquire(req.req_pool_idx, prefix_key, prefix_len)
            logger.debug(
                "DSV4 C4 host-prefix hit: req=%s c4_prefix_len=%d c4_total_len=%d",
                req.rid,
                prefix_len,
                prefill_len,
            )

        suffix_len = prefill_len - prefix_len
        host_indices = (
            self.ensure_host_slots(req.req_pool_idx, prefix_len, suffix_len)
            if suffix_len > 0
            else torch.empty((0,), dtype=torch.int64, device=self.device)
        )
        self._req_c4_prefix_len[req.req_pool_idx] = prefix_len
        self._req_c4_written_len[req.req_pool_idx] = prefill_len
        return HiSparseStagingBackup(host_indices, device_indices[prefix_len:])

    def _prepare_host_radix_staging_backup(
        self, req: Req, device_indices: torch.Tensor
    ) -> HiSparseStagingBackup:
        prefill_len = len(device_indices)
        token_ids = list(req.fill_ids[:prefill_len])

        host_prefix, radix_prefix_len = self.host_radix_cache.match_and_lock_req_prefix(
            req.req_pool_idx,
            token_ids,
            req.extra_key,
            prefill_len,
        )
        if radix_prefix_len > 0:
            self.req_to_host_pool[req.req_pool_idx, :radix_prefix_len] = host_prefix[
                :radix_prefix_len
            ].to(device=self.device, non_blocking=True)

        suffix_len = prefill_len - radix_prefix_len
        host_indices = (
            self.ensure_host_slots(req.req_pool_idx, radix_prefix_len, suffix_len)
            if suffix_len > 0
            else torch.empty((0,), dtype=torch.int64, device=self.device)
        )
        self._req_host_written_len[req.req_pool_idx] = prefill_len
        return HiSparseStagingBackup(
            host_indices,
            device_indices[radix_prefix_len:],
            indexer_ranges=[(req.req_pool_idx, radix_prefix_len, prefill_len)],
        )

    def _release_host_slots(self, req: Req) -> None:
        if self.host_radix_cache is not None:
            self._release_or_cache_host_radix_slots(req)
            return
        if self.is_dsv4_hisparse:
            if isinstance(req.finished_reason, FINISH_ABORT):
                self._release_aborted_dsv4_c4_host_slots(req)
            else:
                self._release_or_cache_dsv4_c4_host_slots(req)
        else:
            self._free_request_host_indices_from(req, 0)

    def _release_or_cache_host_radix_slots(self, req: Req) -> None:
        total_len = self._prepare_radix_cache_len(
            req, self._host_token_len(req.kv_allocated_len)
        )
        host_indices = self.req_to_host_pool[req.req_pool_idx, :total_len]
        token_ids = list((req.origin_input_ids + req.output_ids)[:total_len])
        old_protected = self.host_radix_cache.req_prefix_len(req.req_pool_idx)
        cache_key_len = (len(token_ids) // self.page_size) * self.page_size

        if total_len > 0:
            _, duplicate_indices, _ = self.host_radix_cache.insert_req_host_indices(
                req.req_pool_idx,
                token_ids,
                host_indices.cpu(),
                req.extra_key,
            )
            if duplicate_indices.numel() > 0:
                self.mem_pool_host.free(duplicate_indices)

        self._free_request_host_indices_from(req, max(cache_key_len, old_protected))
        self.host_radix_cache.release_req_node(req.req_pool_idx)

    def _prepare_radix_cache_len(self, req: Req, total_len: int) -> int:
        if self.host_radix_cache is None:
            return 0

        req_idx = req.req_pool_idx
        written_len = min(self._req_host_written_len.get(req_idx, 0), total_len)
        if written_len >= total_len:
            return written_len

        if self.is_dsv4_hisparse or total_len - written_len != 1:
            logger.warning(
                "HiSparse radix cache inserts only %d/%d host-written tokens "
                "for req %s.",
                written_len,
                total_len,
                req.rid,
            )
            return written_len

        host_locs = self.ensure_host_slots(req_idx, written_len, 1)
        device_slot = min(written_len, self.device_buffer_size)
        device_locs = self.req_to_device_buffer[req_idx, device_slot : device_slot + 1]
        self.host_radix_cache.backup_from_device_all_layer(
            self.mem_pool_device,
            host_locs,
            device_locs,
            completed_indexer_pages=[(req_idx, written_len + 1)],
        )
        written_len += 1
        self._req_host_written_len[req_idx] = written_len
        return written_len

    def _release_or_cache_dsv4_c4_host_slots(self, req: Req) -> None:
        # Only the prompt C4 mirror is reusable across benchmark requests.  The
        # generated C4 suffix is request-specific; caching it keeps host slots
        # alive without improving admission for the next request.
        published_len = self._publish_dsv4_c4_host_prompt_prefix(req)
        keep_host_len = max(
            published_len,
            self._req_c4_prefix_len.get(req.req_pool_idx, 0),
        )
        self._free_request_host_indices_from(req, keep_host_len)

    def _release_aborted_dsv4_c4_host_slots(self, req: Req) -> None:
        # Do not publish aborted requests into the C4 host prefix cache.  If this
        # request already referenced an existing prefix cache entry, keep that
        # protected prefix alive and let _clear_c4_prefix_req_state drop the ref.
        keep_host_len = self._req_c4_prefix_len.get(req.req_pool_idx, 0)
        self._free_request_host_indices_from(req, keep_host_len)

    def _free_request_host_indices_from(self, req: Req, start_pos: int) -> None:
        host_len = min(
            self._round_up_to_host_page(self._host_token_len(req.kv_allocated_len)),
            self.req_to_host_pool.shape[1],
        )
        start_pos = min(max(start_pos, 0), host_len)
        req_host_row = self.req_to_host_pool[req.req_pool_idx]
        host_indices = req_host_row[start_pos:host_len]
        host_indices = host_indices[host_indices >= 0]
        if host_indices.numel() > 0:
            host_indices = torch.unique(host_indices)
            if start_pos > 0:
                protected_indices = req_host_row[:start_pos]
                protected_indices = protected_indices[protected_indices >= 0]
                if protected_indices.numel() > 0:
                    host_indices = host_indices[
                        ~torch.isin(host_indices, torch.unique(protected_indices))
                    ]
            if host_indices.numel() > 0:
                self.mem_pool_host.free(host_indices)
        req_host_row[start_pos:host_len] = -1

    def _clear_c4_prefix_req_state(self, req_pool_idx: int) -> None:
        self._c4_prefix_cache.release_req(req_pool_idx)
        self._req_c4_prefix_len.pop(req_pool_idx, None)
        self._req_c4_written_len.pop(req_pool_idx, None)
        self._req_host_written_len.pop(req_pool_idx, None)

    def _round_up_to_host_page(self, size: int) -> int:
        return (size + self.page_size - 1) // self.page_size * self.page_size

    # ---- Host slots and staging ----

    def ensure_host_slots(
        self,
        req_pool_idx: int,
        start_pos: int,
        num_tokens: int,
    ) -> torch.Tensor:
        """Ensure host slots for compressed token positions."""
        if num_tokens <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        page_start = (start_pos // self.page_size) * self.page_size
        page_end = self._round_up_to_host_page(start_pos + num_tokens)

        page_starts = torch.arange(
            page_start,
            page_end,
            self.page_size,
            dtype=torch.int64,
            device=self.device,
        )
        page_is_missing = self.req_to_host_pool[req_pool_idx, page_starts] < 0
        num_missing_pages = int(page_is_missing.sum().item())

        if num_missing_pages > 0:
            if self.host_radix_cache is not None:
                self.host_radix_cache.evict_host_if_needed(
                    self.mem_pool_host, num_missing_pages * self.page_size
                )
            elif self.is_dsv4_hisparse:
                self.reclaim_dsv4_c4_host_prefix_cache(
                    num_missing_pages * self.page_size
                )
            host_locs = self.mem_pool_host.alloc(num_missing_pages * self.page_size)
            if host_locs is None:
                logger.error(
                    "HiSparse: host mem pool alloc failed for %d host pages "
                    "(req_pool_idx=%d, start_pos=%d, num_tokens=%d)",
                    num_missing_pages,
                    req_pool_idx,
                    start_pos,
                    num_tokens,
                )
                raise RuntimeError(
                    f"HiSparse host mem pool alloc failed for {num_missing_pages} pages"
                )
            host_locs = host_locs.to(device=self.device)
            if num_missing_pages == page_starts.numel():
                self.req_to_host_pool[req_pool_idx, page_start:page_end] = host_locs
            else:
                missing_page_starts = page_starts[page_is_missing]
                offsets = torch.arange(
                    self.page_size, dtype=torch.int64, device=self.device
                )
                missing_indices = (
                    missing_page_starts[:, None] + offsets[None, :]
                ).reshape(-1)
                self.req_to_host_pool[req_pool_idx, missing_indices] = host_locs

        return self.req_to_host_pool[req_pool_idx, start_pos : start_pos + num_tokens]

    def _allocated_host_indices(self, req: Req) -> torch.Tensor:
        host_len = min(
            self._round_up_to_host_page(self._host_token_len(req.kv_allocated_len)),
            self.req_to_host_pool.shape[1],
        )
        host_indices = self.req_to_host_pool[req.req_pool_idx, :host_len]
        return host_indices[host_indices >= 0]

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def get_token_stats(self) -> HiSparseTokenStats:
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        if self.is_dsv4_hisparse:
            c4_swap_total = self._c4_swap_miss_tokens + self._c4_swap_hit_tokens
            c4_swap_miss_tokens = self._c4_swap_miss_tokens
            c4_swap_hit_tokens = self._c4_swap_hit_tokens
            c4_swap_miss_rate = (
                self._c4_swap_miss_tokens / c4_swap_total
                if c4_swap_total > 0
                else 0.0
            )
            c4_swap_h2d_bytes = self._c4_swap_h2d_bytes
        else:
            c4_swap_miss_tokens = None
            c4_swap_hit_tokens = None
            c4_swap_miss_rate = None
            c4_swap_h2d_bytes = None
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
            c4_swap_miss_tokens=c4_swap_miss_tokens,
            c4_swap_hit_tokens=c4_swap_hit_tokens,
            c4_swap_miss_rate=c4_swap_miss_rate,
            c4_swap_h2d_bytes=c4_swap_h2d_bytes,
        )

    def admit_request_into_staging(self, req: Req) -> None:
        req.hisparse_staging = True

        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        backup = self._prepare_staging_backup(req, device_indices)

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            if backup.host_indices.numel() > 0:
                if self.host_radix_cache is not None:
                    self.host_radix_cache.backup_from_device_all_layer(
                        self.mem_pool_device,
                        backup.host_indices,
                        backup.device_indices,
                        indexer_ranges=backup.indexer_ranges,
                    )
                else:
                    self.mem_pool_host.backup_from_device_all_layer(
                        self.mem_pool_device,
                        backup.host_indices,
                        backup.device_indices,
                        io_backend="kernel",
                    )
            finish_event.record()
            if backup.host_indices.is_cuda:
                backup.host_indices.record_stream(self.write_staging_stream)
            if backup.device_indices.is_cuda:
                backup.device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def admit_request_direct(self, req: Req) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.
        """
        self.alloc_device_buffer(req)

        host_len = self._host_token_len(req.kv_allocated_len)
        if host_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(req)
        else:
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty -> every top-k lookup is a miss -> host load.
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1

        req.hisparse_staging = False
        self._skip_first_backup[req.req_pool_idx] = True
        if self.is_dsv4_hisparse:
            self._req_c4_prefix_len.setdefault(
                req.req_pool_idx,
                min(
                    getattr(req, "hisparse_c4_transfer_prefix_len", 0) or 0,
                    host_len,
                ),
            )
            self._req_c4_written_len[req.req_pool_idx] = host_len
            self._publish_dsv4_c4_host_prompt_prefix(req)
        if self.host_radix_cache is not None:
            self._req_host_written_len[req.req_pool_idx] = host_len
        logger.debug("HiSparse: admitting request %s directly", req.rid)

    def _host_token_len(self, kv_allocated_len: int) -> int:
        if self.is_dsv4_hisparse:
            return self.token_to_kv_pool_allocator.c4_tokens_for_full_tokens(
                kv_allocated_len
            )
        return kv_allocated_len

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer."""
        n = self._host_token_len(req.kv_allocated_len)
        host_indices = self.req_to_host_pool[req.req_pool_idx, :n]
        device_locs = self.req_to_device_buffer[req.req_pool_idx, :n]
        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )
        # Direct PD admission runs outside the model forward stream.  The preload
        # must be complete before the request is marked ready, otherwise decode
        # can race the H2D copy and read a partially populated hot buffer.
        device_module.current_stream().synchronize()

    def alloc_device_buffer(self, req: Req) -> None:
        if self.is_dsv4_hisparse:
            allocated_len = req.kv_allocated_len or len(req.fill_ids)
            alloc_size = self.padded_buffer_size
        else:
            allocated_len = req.kv_allocated_len
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

        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size

        self.req_device_buffer_tokens[:, req.req_pool_idx, :alloc_size] = (
            self._device_buffer_arange_i32[:alloc_size]
        )
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )

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

                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap

        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    def _insert_prefill_into_radix_cache(self, req: Req) -> None:
        if self.host_radix_cache is None:
            return

        prefill_len = self._req_host_written_len.get(req.req_pool_idx, 0)
        if prefill_len <= 0:
            return
        token_ids = list(req.fill_ids[:prefill_len])
        host_indices = self.req_to_host_pool[req.req_pool_idx, :prefill_len].cpu()
        old_prefix_len = self.host_radix_cache.req_prefix_len(req.req_pool_idx)

        insert_prefix_len, duplicate_indices, canonical_indices = (
            self.host_radix_cache.insert_req_host_indices(
                req.req_pool_idx,
                token_ids,
                host_indices,
                req.extra_key,
                new_protected_len=prefill_len,
                lock_new_node=True,
                return_canonical_indices=True,
            )
        )

        if duplicate_indices.numel() > 0:
            self.mem_pool_host.free(duplicate_indices)
            assert canonical_indices is not None
            self.req_to_host_pool[
                req.req_pool_idx, old_prefix_len:insert_prefix_len
            ] = canonical_indices.to(device=self.device, non_blocking=True)

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
            # prepare device buffer and update req
            self.alloc_device_buffer(req)
            if self.host_radix_cache is not None:
                self._insert_prefill_into_radix_cache(req)
            else:
                self._publish_dsv4_c4_host_prompt_prefix(req)
            self._skip_first_backup[req.req_pool_idx] = True
            req.hisparse_staging = False
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> None:
        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if not self.is_dsv4_hisparse:
            # Grow device buffers if needed and resolve the latest-token slot.
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)

            # No need to clear prior mappings: the only consumer of the mapping
            # for past tokens is the swap-in kernel, and it goes through
            # top_k_device_locs returned by swap_in_selected_pages -- not via
            # mapping[old_out_cache_loc] -- so stale entries are harmless.
            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                out_cache_loc
            )
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc
            return

        active_reqs = seq_lens % self.compress_ratio == 0
        if not torch.any(active_reqs):
            return

        active_seq_lens = seq_lens[active_reqs]
        active_out_cache_loc = out_cache_loc[active_reqs]
        active_req_pool_indices = req_pool_indices[active_reqs]

        compressed_seq_lens = active_seq_lens // self.compress_ratio
        reserved_positions = (compressed_seq_lens - 1).clamp(
            max=self.device_buffer_size
        )
        reserved_buffer_loc = self.req_to_device_buffer[
            active_req_pool_indices, reserved_positions
        ]

        self.req_device_buffer_token_locs[
            :, active_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            active_out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            reserved_buffer_loc
        )

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

        actual_compressed_pos_cpu = actual_compressed_pos.cpu()
        completed_indexer_pages = []
        host_locs = torch.cat(
            [
                self.ensure_host_slots(
                    int(req_pool_indices_cpu[i]),
                    int(actual_compressed_pos_cpu[j]),
                    1,
                )
                for j, i in enumerate(backup_indices)
            ]
        )
        for j, i in enumerate(backup_indices):
            req_idx = int(req_pool_indices_cpu[i])
            self._req_c4_written_len[req_idx] = max(
                self._req_c4_written_len.get(req_idx, 0),
                int(actual_compressed_pos_cpu[j]) + 1,
            )
            written_end = int(actual_compressed_pos_cpu[j]) + 1
            self._req_host_written_len[req_idx] = max(
                self._req_host_written_len.get(req_idx, 0),
                written_end,
            )
            completed_indexer_pages.append((req_idx, written_end))

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            if self.host_radix_cache is not None:
                self.host_radix_cache.backup_from_device_all_layer(
                    self.mem_pool_device,
                    host_locs,
                    device_locs,
                    completed_indexer_pages=completed_indexer_pages,
                )
            else:
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

    def _backup_accepted_device_locs(
        self, host_locs: torch.Tensor, device_locs: torch.Tensor
    ) -> None:
        producer_stream = device_module.current_stream()
        device_locs = device_locs.contiguous()

        with device_module.stream(self.decode_backup_stream):
            # Accepted draft KV is produced by the verify forward on the current
            # stream. The backup stream must not read those slots before that
            # write has completed.
            self.decode_backup_stream.wait_stream(producer_stream)
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
        producer_stream.wait_event(event)

    def _free_device_buffer_slots(self, req: Req) -> None:
        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        draft_cap = int(self.req_draft_buffer_size[req.req_pool_idx])
        if current_cap > 0:
            side_buf_hi = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
        else:
            side_buf_hi = torch.empty(0, dtype=torch.int64, device=self.device)
        if draft_cap > 0:
            draft_start = self.device_buffer_size + 1
            draft_end = draft_start + draft_cap
            draft_buf_hi = self.req_to_device_buffer[
                req.req_pool_idx, draft_start:draft_end
            ]
            all_hi = torch.unique(
                torch.cat(
                    [
                        side_buf_hi[side_buf_hi > 0],
                        draft_buf_hi[draft_buf_hi > 0],
                    ]
                )
            )
        else:
            all_hi = torch.unique(side_buf_hi[side_buf_hi > 0])

        if (current_cap > 0 or draft_cap > 0) and all_hi.numel() > 0:
            self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)

    def supports_hisparse_draft_slots(self) -> bool:
        return True

    # ---- EAGLE/MTP draft slots ----

    def _ensure_draft_buffer(
        self,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        num_tokens: int,
    ) -> None:
        if num_tokens <= 0:
            return

        start = self.device_buffer_size + 1
        end = start + num_tokens
        if end > self.padded_buffer_size:
            raise ValueError(
                f"Requested {num_tokens} draft slots but extra page only "
                f"has {self.padded_buffer_size - self.device_buffer_size - 1} "
                f"available (padded_buffer_size={self.padded_buffer_size}, "
                f"device_buffer_size={self.device_buffer_size})."
            )

        req_indices_cpu = req_pool_indices_cpu.to(torch.int64).tolist()
        grow_reqs = []
        total_grow = 0
        for req_idx in req_indices_cpu:
            current_cap = int(self.req_draft_buffer_size[req_idx])
            existing_device_cap = int(self.req_device_buffer_size[req_idx])
            if existing_device_cap >= end:
                self.req_draft_buffer_size[req_idx] = max(current_cap, num_tokens)
                continue
            if current_cap >= num_tokens:
                continue
            grow_reqs.append((req_idx, current_cap))
            total_grow += num_tokens - current_cap

        if total_grow == 0:
            return

        all_new = self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
            total_grow
        )
        if all_new is None:
            raise RuntimeError(
                f"HiSparse: failed to grow buffers for draft slots (need {total_grow})"
            )

        offset = 0
        for req_idx, current_cap in grow_reqs:
            grow_size = num_tokens - current_cap
            chunk = all_new[offset : offset + grow_size]
            offset += grow_size
            slot_start = start + current_cap
            slot_end = slot_start + grow_size
            self.req_to_device_buffer[req_idx, slot_start:slot_end] = chunk
            self.req_device_buffer_tokens[:, req_idx, slot_start:slot_end] = (
                self._device_buffer_arange_i32[slot_start:slot_end]
            )
            self.req_device_buffer_token_locs[:, req_idx, slot_start:slot_end] = chunk
            self.req_draft_buffer_size[req_idx] = num_tokens

    def _ensure_padded_buffer(
        self, req_pool_indices: torch.Tensor, req_pool_indices_cpu: torch.Tensor
    ) -> None:
        self._ensure_draft_buffer(
            req_pool_indices,
            req_pool_indices_cpu,
            self.padded_buffer_size - self.device_buffer_size - 1,
        )

    def get_draft_device_slots(
        self,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        num_tokens_per_req: int,
    ) -> torch.Tensor:
        start = self.device_buffer_size + 1
        end = start + num_tokens_per_req
        if end > self.padded_buffer_size:
            raise ValueError(
                f"Requested {num_tokens_per_req} draft slots but extra page only "
                f"has {self.padded_buffer_size - self.device_buffer_size - 1} "
                f"available (padded_buffer_size={self.padded_buffer_size}, "
                f"device_buffer_size={self.device_buffer_size})."
            )
        self._ensure_draft_buffer(
            req_pool_indices, req_pool_indices_cpu, num_tokens_per_req
        )
        return self.req_to_device_buffer[req_pool_indices, start:end].reshape(-1)

    def get_draft_device_slots_variable(
        self,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        tokens_per_req_cpu: torch.Tensor,
    ) -> torch.Tensor:
        if tokens_per_req_cpu.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=req_pool_indices.device)

        if not tokens_per_req_cpu.is_cuda:
            tokens_per_req_cpu = tokens_per_req_cpu.to(torch.int64)
            first_tokens = int(tokens_per_req_cpu[0].item())
            if first_tokens < 0:
                raise ValueError(
                    f"Negative draft slot count is invalid: {first_tokens}."
                )
            if bool(torch.all(tokens_per_req_cpu == first_tokens).item()):
                if first_tokens == 0:
                    return torch.empty(
                        0, dtype=torch.int64, device=req_pool_indices.device
                    )
                return self.get_draft_device_slots(
                    req_pool_indices,
                    req_pool_indices_cpu,
                    first_tokens,
                )

        start = self.device_buffer_size + 1
        max_tokens = int(tokens_per_req_cpu.max().item())
        if start + max_tokens > self.padded_buffer_size:
            raise ValueError(
                f"Max per-request draft slots ({max_tokens}) exceeds extra page "
                f"capacity ({self.padded_buffer_size - self.device_buffer_size - 1})."
            )

        self._ensure_draft_buffer(req_pool_indices, req_pool_indices_cpu, max_tokens)

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
        col_indices = start + pos_in_segment

        return self.req_to_device_buffer[row_indices, col_indices]

    # ---- EAGLE/MTP accepted-token finalization ----

    def finalize_accepted_tokens(
        self,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        accepted_cache_locs: torch.Tensor,
        draft_cache_locs: torch.Tensor,
        num_correct_drafts: torch.Tensor,
        num_correct_drafts_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        if accepted_cache_locs.numel() == 0:
            return

        if self.is_dsv4_hisparse:
            self._finalize_accepted_tokens_dsv4(
                req_pool_indices,
                req_pool_indices_cpu,
                accepted_cache_locs,
                draft_cache_locs,
                num_correct_drafts,
                num_correct_drafts_cpu,
                seq_lens,
            )
            return

        counts = num_correct_drafts.to(torch.int64) + 1
        counts_cpu = num_correct_drafts_cpu.to(torch.int64) + 1
        total_accepted = int(counts_cpu.sum().item())
        if total_accepted != accepted_cache_locs.numel():
            raise ValueError(
                "HiSparse accepted token bookkeeping mismatch: "
                f"expected {total_accepted} cache locs, got {accepted_cache_locs.numel()}."
            )

        all_device_locs = self.mem_pool_device._translate_loc_to_hisparse_device(
            accepted_cache_locs
        )
        full_to_device_mapping = (
            self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        )
        draft_mapping_snapshot = full_to_device_mapping[draft_cache_locs].clone()

        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=counts.device), counts.cumsum(0)]
        )
        starts = seq_lens - counts
        all_indices = torch.arange(total_accepted, device=counts.device)
        req_indices_expanded = torch.repeat_interleave(req_pool_indices, counts)
        req_indices_expanded_cpu = torch.repeat_interleave(
            req_pool_indices_cpu.to(torch.int64), counts_cpu
        )
        pos_in_segment = all_indices - torch.repeat_interleave(offsets[:-1], counts)
        col_indices = torch.repeat_interleave(starts, counts) + pos_in_segment

        col_indices_cpu = col_indices.cpu()
        host_locs = torch.cat(
            [
                self.ensure_host_slots(int(req_idx), int(col_idx), 1)
                for req_idx, col_idx in zip(req_indices_expanded_cpu, col_indices_cpu)
            ]
        )
        if host_locs.numel() != total_accepted:
            full_to_device_mapping[draft_cache_locs] = draft_mapping_snapshot
            raise RuntimeError("HiSparse host slot allocation mismatch for draft accept")

        full_to_device_mapping[draft_cache_locs] = 0

        self._backup_accepted_device_locs(host_locs, all_device_locs)

        newest_slots = self.req_to_device_buffer[
            req_pool_indices, self.device_buffer_size
        ]
        for idx in req_pool_indices_cpu.tolist():
            self._skip_first_backup[idx] = True

        last_offsets = offsets[1:] - 1
        last_logical = accepted_cache_locs[last_offsets]
        last_device = all_device_locs[last_offsets]
        full_to_device_mapping[last_logical] = newest_slots
        self.mem_pool_device.transfer_values_on_device(
            dst_indices=newest_slots,
            src_indices=last_device,
        )

    def _finalize_accepted_tokens_dsv4(
        self,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        accepted_cache_locs: torch.Tensor,
        draft_cache_locs: torch.Tensor,
        num_correct_drafts: torch.Tensor,
        num_correct_drafts_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        counts = num_correct_drafts.to(torch.int64) + 1
        counts_cpu = num_correct_drafts_cpu.to(torch.int64) + 1
        total_accepted = int(counts_cpu.sum().item())
        if total_accepted != accepted_cache_locs.numel():
            raise ValueError(
                "DeepSeek V4 HiSparse accepted token bookkeeping mismatch: "
                f"expected {total_accepted} cache locs, got {accepted_cache_locs.numel()}."
            )

        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=counts.device), counts.cumsum(0)]
        )
        all_indices = torch.arange(total_accepted, device=counts.device)
        pos_in_segment = all_indices - torch.repeat_interleave(offsets[:-1], counts)
        starts = seq_lens - counts
        full_positions = torch.repeat_interleave(starts, counts) + pos_in_segment
        compressed_mask = (accepted_cache_locs + 1) % self.compress_ratio == 0

        full_to_device_mapping = (
            self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        )
        draft_compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            draft_cache_locs
        )
        draft_mapping_snapshot = full_to_device_mapping[draft_compressed_locs].clone()

        if not torch.any(compressed_mask):
            full_to_device_mapping[draft_compressed_locs] = 0
            return

        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            accepted_cache_locs
        )
        device_locs = full_to_device_mapping[compressed_locs]

        req_indices_expanded = torch.repeat_interleave(req_pool_indices, counts)
        req_indices_expanded_cpu = torch.repeat_interleave(
            req_pool_indices_cpu.to(torch.int64), counts_cpu
        )
        compressed_req_indices = req_indices_expanded[compressed_mask]
        compressed_req_indices_cpu = req_indices_expanded_cpu[compressed_mask.cpu()]
        compressed_positions = full_positions[compressed_mask] // self.compress_ratio
        if compressed_locs.numel() != compressed_req_indices.numel():
            full_to_device_mapping[draft_compressed_locs] = draft_mapping_snapshot
            raise RuntimeError(
                "DeepSeek V4 HiSparse compressed draft bookkeeping mismatch: "
                f"{compressed_locs.numel()} compressed locs vs "
                f"{compressed_req_indices.numel()} compressed positions."
            )

        compressed_positions_cpu = compressed_positions.cpu()
        host_locs = torch.cat(
            [
                self.ensure_host_slots(int(req_idx), int(pos), 1)
                for req_idx, pos in zip(
                    compressed_req_indices_cpu, compressed_positions_cpu
                )
            ]
        )
        if host_locs.numel() != compressed_locs.numel():
            full_to_device_mapping[draft_compressed_locs] = draft_mapping_snapshot
            raise RuntimeError(
                "DeepSeek V4 HiSparse host slot allocation mismatch for draft accept"
            )
        for req_idx, pos in zip(compressed_req_indices_cpu, compressed_positions_cpu):
            self._req_c4_written_len[int(req_idx)] = max(
                self._req_c4_written_len.get(int(req_idx), 0), int(pos) + 1
            )

        full_to_device_mapping[draft_compressed_locs] = 0

        self._backup_accepted_device_locs(host_locs, device_locs)

        newest_slots = self.req_to_device_buffer[
            req_pool_indices, self.device_buffer_size
        ]
        for idx in req_pool_indices_cpu.tolist():
            self._skip_first_backup[idx] = True

        compressed_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=counts.device),
                torch.cumsum(
                    torch.bincount(
                        compressed_req_indices,
                        minlength=self.req_to_device_buffer.shape[0],
                    )[req_pool_indices].to(torch.int64),
                    dim=0,
                ),
            ]
        )
        has_compressed = compressed_offsets[1:] > compressed_offsets[:-1]
        if not torch.any(has_compressed):
            return

        active_newest_slots = newest_slots[has_compressed]
        last_compressed_offsets = compressed_offsets[1:][has_compressed] - 1
        last_compressed_locs = compressed_locs[last_compressed_offsets]
        last_device_locs = device_locs[last_compressed_offsets]
        full_to_device_mapping[last_compressed_locs] = active_newest_slots
        self.mem_pool_device.transfer_values_on_device(
            dst_indices=active_newest_slots,
            src_indices=last_device_locs,
        )

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

        Note: dsv4 hisparse is not supported here. This helper is only used as
        a kernel oracle in test_hisparse_unit.py (non-dsv4 path).

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
        # Remove from staging queue
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        # Wait for any in-flight staging DMA to complete before freeing
        self.write_staging_stream.synchronize()

        prefill_len = len(req.fill_ids)
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefill_len
        ]
        self.token_to_kv_pool_allocator.free_hisparse(allocated_locs)

        if self.host_radix_cache is not None:
            self._free_request_host_indices_from(
                req, self.host_radix_cache.req_prefix_len(req.req_pool_idx)
            )
            self.host_radix_cache.release_req_node(req.req_pool_idx)
        else:
            # Free only request-owned suffix slots. Prefix slots may point to the
            # shared C4 host-prefix cache.
            self._free_request_host_indices_from(
                req, self._req_c4_prefix_len.get(req.req_pool_idx, 0)
            )
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self._skip_first_backup[req.req_pool_idx] = False
        self._clear_c4_prefix_req_state(req.req_pool_idx)
        req.hisparse_staging = False

    def _clear_request_runtime_state(self, req_pool_idx: int) -> None:
        self.req_device_buffer_tokens[:, req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req_pool_idx, :] = -1
        self.req_to_device_buffer[req_pool_idx, :] = 0
        self.req_device_buffer_size[req_pool_idx] = 0
        self.req_draft_buffer_size[req_pool_idx] = 0
        self.req_to_host_pool[req_pool_idx, :] = -1
        self.lru_slots[:, req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req_pool_idx] = False

    def retract_decode_req(self, req: Req) -> None:
        """Detach DSV4 HiSparse request state without treating the request as done.

        Decode retraction is a cold OOM recovery path.  Unlike
        request_finished(), it must preserve the authoritative C4 host mirror so
        the request can resume later without a new PD transfer.
        """
        if not self.is_dsv4_hisparse:
            self.retract_req(req)
            return

        if req.hisparse_staging:
            self.abort_staging_request(req)
            return

        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()

        req_pool_idx = req.req_pool_idx
        host_len = self._host_token_len(req.kv_allocated_len)
        host_indices = self.req_to_host_pool[req_pool_idx, :host_len]
        if host_len > 0 and torch.any(host_indices < 0):
            raise RuntimeError(
                "Cannot retract DSV4 HiSparse request with incomplete C4 host "
                f"mirror: req={req.rid}, host_len={host_len}."
            )

        req.hisparse_retract_host_indices = host_indices.detach().clone()
        req.hisparse_retract_host_len = host_len
        c4_prefix_ref = self._c4_prefix_cache.retain_req_ref(req_pool_idx)
        c4_prefix_len = self._req_c4_prefix_len.get(req_pool_idx, 0)
        if c4_prefix_ref is None:
            c4_prefix_len = 0
        req.hisparse_retract_c4_prefix_len = c4_prefix_len
        req.hisparse_retract_c4_written_len = self._req_c4_written_len.get(
            req_pool_idx, host_len
        )
        req.hisparse_retract_host_written_len = self._req_host_written_len.get(
            req_pool_idx, 0
        )
        req.hisparse_retract_c4_prefix_ref = c4_prefix_ref

        self._free_device_buffer_slots(req)

        allocated_len = req.kv_allocated_len
        allocated_locs = self.req_to_token_pool.req_to_token[
            req_pool_idx, :allocated_len
        ]
        if allocated_locs.numel() > 0:
            compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
                allocated_locs
            )
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = 0

        self._clear_request_runtime_state(req_pool_idx)
        self._clear_c4_prefix_req_state(req_pool_idx)
        self._req_host_written_len.pop(req_pool_idx, None)
        req.hisparse_staging = False

    def restore_retracted_decode_req(self, req: Req) -> bool:
        """Restore DSV4 HiSparse host/device side state saved by retraction."""
        if not self.is_dsv4_hisparse:
            return False

        host_indices = getattr(req, "hisparse_retract_host_indices", None)
        if host_indices is None:
            return False

        req_pool_idx = req.req_pool_idx
        host_len = int(getattr(req, "hisparse_retract_host_len", len(host_indices)))
        if host_len > self.req_to_host_pool.shape[1]:
            raise RuntimeError(
                "DSV4 HiSparse retracted host state is larger than req host row: "
                f"req={req.rid}, host_len={host_len}, "
                f"row={self.req_to_host_pool.shape[1]}."
            )

        self.req_to_host_pool[req_pool_idx, :] = -1
        if host_len > 0:
            self.req_to_host_pool[req_pool_idx, :host_len] = host_indices[
                :host_len
            ].to(device=self.device, non_blocking=True)

        self._req_c4_prefix_len[req_pool_idx] = min(
            int(getattr(req, "hisparse_retract_c4_prefix_len", 0)), host_len
        )
        self._req_c4_written_len[req_pool_idx] = min(
            int(getattr(req, "hisparse_retract_c4_written_len", host_len)),
            host_len,
        )
        host_written_len = int(getattr(req, "hisparse_retract_host_written_len", 0))
        if host_written_len > 0:
            self._req_host_written_len[req_pool_idx] = min(host_written_len, host_len)

        self.alloc_device_buffer(req)
        if host_len <= self.device_buffer_size:
            self._preload_to_device_buffer(req)
        else:
            self.req_device_buffer_tokens[
                :, req_pool_idx, : self.device_buffer_size
            ] = -1

        self._skip_first_backup[req_pool_idx] = True
        self._publish_dsv4_c4_host_prompt_prefix(req)
        self._c4_prefix_cache.release_retained(
            getattr(req, "hisparse_retract_c4_prefix_ref", None)
        )
        req.hisparse_staging = False

        for attr in (
            "hisparse_retract_host_indices",
            "hisparse_retract_host_len",
            "hisparse_retract_c4_prefix_len",
            "hisparse_retract_c4_written_len",
            "hisparse_retract_host_written_len",
            "hisparse_retract_c4_prefix_ref",
        ):
            if hasattr(req, attr):
                delattr(req, attr)
        return True

    def release_retracted_decode_req(self, req: Req) -> None:
        """Free DSV4 HiSparse state for a request aborted while retracted."""
        if not self.is_dsv4_hisparse:
            return

        host_indices = getattr(req, "hisparse_retract_host_indices", None)
        if host_indices is None:
            return

        prefix_len = int(getattr(req, "hisparse_retract_c4_prefix_len", 0))
        host_len = int(getattr(req, "hisparse_retract_host_len", len(host_indices)))
        host_indices = host_indices[:host_len].to(device=self.device, non_blocking=True)
        suffix_indices = host_indices[prefix_len:]
        suffix_indices = suffix_indices[suffix_indices >= 0]
        if suffix_indices.numel() > 0:
            self.mem_pool_host.free(torch.unique(suffix_indices))

        self._c4_prefix_cache.release_retained(
            getattr(req, "hisparse_retract_c4_prefix_ref", None)
        )

        for attr in (
            "hisparse_retract_host_indices",
            "hisparse_retract_host_len",
            "hisparse_retract_c4_prefix_len",
            "hisparse_retract_c4_written_len",
            "hisparse_retract_host_written_len",
            "hisparse_retract_c4_prefix_ref",
        ):
            if hasattr(req, attr):
                delattr(req, attr)

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

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi).  The C4 host prefix
        # cache may be shared across requests, but the GPU hot buffer slots are
        # request-owned, so clear the whole hot mapping range.
        allocated_len = req.kv_allocated_len

        self._free_device_buffer_slots(req)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ]
        if allocated_locs.numel() > 0:
            compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
                allocated_locs
            )
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = 0

        self._release_host_slots(req)

        # clear req info
        self._clear_request_runtime_state(req.req_pool_idx)
        self._clear_c4_prefix_req_state(req.req_pool_idx)

    def _maybe_log_c4_swap_in_sample(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> None:
        if (
            not self.is_dsv4_hisparse
            or layer_id != 0
            or self._c4_miss_sample_interval_s <= 0
            or not logger.isEnabledFor(logging.INFO)
        ):
            return

        now = time.monotonic()
        if now - self._last_c4_miss_sample_time < self._c4_miss_sample_interval_s:
            return

        is_capturing = False
        try:
            if hasattr(device_module, "is_current_stream_capturing"):
                is_capturing = bool(device_module.is_current_stream_capturing())
        except Exception:
            is_capturing = True
        if is_capturing:
            return

        self._last_c4_miss_sample_time = now
        with torch.no_grad():
            top_tokens = top_k_result.to(dtype=torch.int32)
            valid = (top_tokens >= 0) & (
                top_tokens < compressed_seq_lens.view(-1, 1).to(top_tokens.device)
            )
            if not torch.any(valid):
                return

            resident_req_indices = req_pool_indices.to(dtype=torch.long)
            resident_tokens = self.req_device_buffer_tokens[
                layer_id, resident_req_indices
            ]
            resident = (top_tokens.unsqueeze(-1) == resident_tokens.unsqueeze(1)).any(
                dim=-1
            )
            valid_count = int(valid.sum().item())
            hit_count = int((resident & valid).sum().item())
            miss_count = valid_count - hit_count

        self._c4_swap_hit_tokens += hit_count
        self._c4_swap_miss_tokens += miss_count
        self._c4_swap_h2d_bytes += miss_count * self.item_size_bytes

        logger.info(
            "HiSparse C4 swap-in sample: reqs=%d topk_tokens=%d hot_hits=%d "
            "host_misses=%d miss_rate=%.4f hot_buffer_size=%d top_k=%d",
            top_k_result.size(0),
            valid_count,
            hit_count,
            miss_count,
            miss_count / max(valid_count, 1),
            self.device_buffer_size,
            self.top_k,
        )

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices."""
        num_reqs = compressed_seq_lens.size(0)
        if top_k_result.size(0) != num_reqs:
            raise RuntimeError(
                "HiSparse swap-in batch mismatch: "
                f"compressed_seq_lens.shape={tuple(compressed_seq_lens.shape)}, "
                f"top_k_result.shape={tuple(top_k_result.shape)}"
            )
        if req_pool_indices.size(0) < num_reqs:
            raise RuntimeError(
                "HiSparse swap-in req_pool_indices shorter than metadata batch: "
                f"req_pool_indices.shape={tuple(req_pool_indices.shape)}, "
                f"num_reqs={num_reqs}"
            )
        req_pool_indices = req_pool_indices[:num_reqs]

        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]
        top_k_indices.fill_(-1)
        self._maybe_log_c4_swap_in_sample(
            req_pool_indices, compressed_seq_lens, top_k_result, layer_id
        )

        # todo, adjustable for performance
        block_size = 1024
        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
        swap_in_fn(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=block_size,
            num_real_reqs=self.num_real_reqs,
        )
        return top_k_indices
