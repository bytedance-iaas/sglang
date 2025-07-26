import logging
import threading
import time

import torch

from sglang.srt.managers.eic_cache_controller import (
    EICCacheController,
    get_content_hash,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.chunk_cache import ChunkCache, SWAChunkCache
from sglang.srt.mem_cache.eic_memory_pool import (
    EICMHATokenToKVPoolHost,
    EICMLATokenToKVPoolHost,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class EICChunkCache(ChunkCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        tp_cache_group: torch.distributed.ProcessGroup,
        server_args: ServerArgs,
    ):
        super().__init__(req_to_token_pool, token_to_kv_pool_allocator, page_size)
        self.tp_group = tp_cache_group
        self.tp_size = self.tp_group.size()
        self.rank = self.tp_group.rank()
        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        self.page_size = page_size
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = EICMHATokenToKVPoolHost(
                self.kv_cache,
                4.0,
                0,
                "cpu",
                page_size,
                self.rank,
                extra_info=self.get_extra_info(server_args),
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = EICMLATokenToKVPoolHost(
                self.kv_cache,
                4.0,
                0,
                "cpu",
                page_size,
                self.rank,
                extra_info=self.get_extra_info(server_args),
            )
        else:
            raise ValueError(f"EICChunkCache only supports MHA and MLA yet")

        self.load_cache_event = threading.Event()
        self.cache_controller = EICCacheController(
            token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            page_size,
            load_cache_event=self.load_cache_event,
            write_policy="write_through",
            server_args=server_args,
        )
        self.ongoing_writing_queue = dict()
        self.background_thread = threading.Thread(
            target=self.background_thread, daemon=True
        )
        self.background_thread.start()
        self._evictable_size = 0
        self.save_docode_cache = True

    def get_extra_info(self, server_args: ServerArgs):
        extra_info = {
            "model_path": server_args.model_path,
            "world_size": self.tp_size,
            "tp_rank": self.rank,
            "framework": "sglang",
        }
        return extra_info

    def write_backup(self, req: Req, save_decode_cache: bool = True):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx,
            : len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0),
        ]
        if (
            len(self.ongoing_writing_queue) >= 20
            or len(kv_indices) < self.page_size
            or not save_decode_cache
        ):
            self.token_to_kv_pool_allocator.free(kv_indices)
            return
        page_aligned_len = len(kv_indices) // self.page_size * self.page_size
        logger.info(f"page aligned length: {page_aligned_len}")
        paged_kv_indices = kv_indices[:page_aligned_len]
        token_ids = (req.origin_input_ids + req.output_ids)[:page_aligned_len]
        content_hash = get_content_hash(token_ids, self.page_size)
        host_indices = self.cache_controller.write_page(
            device_indices=paged_kv_indices,
            priority=None,
            node_id=req.rid,
            content_hash=content_hash,
        )
        if host_indices is not None:
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
            self.ongoing_writing_queue[req.rid] = paged_kv_indices.clone()
            self._evictable_size += len(paged_kv_indices)
            logger.debug(
                f"cache request {req.rid} started, kvcache indices: {len(paged_kv_indices)}"
            )
        else:
            self.token_to_kv_pool_allocator.free(kv_indices)

    def cache_finished_req(self, req: Req, is_decode: bool = False):
        save_cache = is_decode and self.save_docode_cache
        self.write_backup(req, save_decode_cache=save_cache)
        self.req_to_token_pool.free(req.req_pool_idx)

    def writing_check(self):
        while not self.cache_controller.ack_write_queue.empty():
            try:
                rid, success = self.cache_controller.ack_write_queue.get_nowait()
                kv_indices = self.ongoing_writing_queue.get(rid)
                self.token_to_kv_pool_allocator.free(kv_indices)
                self._evictable_size -= len(kv_indices)
                logger.debug(
                    f"cache request {rid} complete, kvcache indices: {len(kv_indices)} "
                )
                del self.ongoing_writing_queue[rid]
            except Exception as e:
                continue

    def background_thread(self):
        while True:
            self.writing_check()

    def evictable_size(self):
        return self._evictable_size

    def reset(self):
        logger.info("Reset EICChunkCache")
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        self.ongoing_writing_queue = dict()
        self._evictable_size = 0


class EICSWAChunkCache(EICChunkCache):
    """ChunkCache with support for hybrid KV cache operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        tp_cache_group: torch.distributed.ProcessGroup,
        server_args: ServerArgs,
    ):
        assert isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
        super().__init__(
            req_to_token_pool,
            token_to_kv_pool_allocator,
            page_size,
            tp_cache_group,
            server_args,
        )

    def evict(
        self,
        req: Req,
        prelen: int,
        attention_chunk_size: int,
    ):
        if prelen >= req.evicted_seqlen_local + attention_chunk_size:
            new_evicted_seqlen_local = attention_chunk_size * (
                prelen // attention_chunk_size
            )
            free_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.evicted_seqlen_local : new_evicted_seqlen_local
            ]
            self.token_to_kv_pool_allocator.free_swa(free_slots)
            req.evicted_seqlen_local = new_evicted_seqlen_local
