from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, DecLockRefParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class DecodeKVCacheOffloadManager:
    """Manage decode-side KV cache offloading lifecycle and operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        tree_cache: BasePrefixCache,
        server_args: ServerArgs,
        hisparse_coordinator: "HiSparseCoordinator" = None,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = server_args.page_size
        self.server_args = server_args
        self.hisparse_coordinator = hisparse_coordinator
        self.request_counter = 0
        self.tree_cache = tree_cache
        env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
        if env_stride is None or env_stride <= 0:
            self.offload_stride = self.page_size
        else:
            self.offload_stride = max(
                self.page_size, (env_stride // self.page_size) * self.page_size
            )
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, MHATokenToKVPool):
            self.decode_host_mem_pool = MHATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        elif isinstance(kv_cache, MLATokenToKVPool):
            self.decode_host_mem_pool = MLATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        else:
            raise ValueError("Unsupported KV cache type for decode offload")

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        hicache_storage_backend_extra_config = {}
        if server_args.hicache_storage_backend_extra_config:
            try:
                hicache_storage_backend_extra_config = json.loads(
                    server_args.hicache_storage_backend_extra_config
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid hicache storage backend extra config JSON: {e}"
                )

        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            mem_pool_host=self.decode_host_mem_pool,
            page_size=self.page_size,
            tp_group=tp_group,
            io_backend=server_args.hicache_io_backend,
            load_cache_event=threading.Event(),
            storage_backend=server_args.hicache_storage_backend,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=hicache_storage_backend_extra_config,
        )

        self.ongoing_offload = {}
        self.ongoing_backup = {}
        self.offloaded_state = {}
        self.aborted_req_ids = set()
        logger.info("Enable offload kv cache for decode side")

    def offload_kv_cache(self, req) -> bool:
        """Offload incremental KV cache for decode side."""

        if self.cache_controller is None or self.decode_host_mem_pool is None:
            return False

        if req.req_pool_idx is None or req.req_pool_idx == -1 or len(req.output_ids) == 0:
            return False

        token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            return False

        # Prefill side offloads page-aligned origin_input_ids, decode side offloads the incremental part
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        prefill_offloaded_len = (
            len(req.origin_input_ids) // self.page_size * self.page_size
        )
        state = self.offloaded_state.get(req.rid)
        is_first_offload = state is None
        if is_first_offload:
            prefill_hashes = self._compute_prefix_hash(
                req.origin_input_ids[:prefill_offloaded_len]
            )
            last_prefill_hash = (
                prefill_hashes[-1] if prefill_offloaded_len > 0 else None
            )
            state = OffloadedState(
                prefill_len=prefill_offloaded_len,
                inc_len=0,
                last_hash=last_prefill_hash,
            )
        incremental_total = len(all_tokens) - state.prefill_len
        incremental_new = incremental_total - state.inc_len
        incremental_aligned_len = (
            incremental_new // self.offload_stride * self.offload_stride
        )

        if incremental_aligned_len == 0:
            return False

        # Extract incremental tokens and indices for the newly available chunk
        start = state.prefill_len + state.inc_len
        end = start + incremental_aligned_len
        incremental_tokens = all_tokens[start:end]
        incremental_indices = token_indices[start:end]

        # Asynchronously offload incremental KV cache from device to host
        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self.cache_controller.write(
            device_indices=incremental_indices.long(),
            node_id=ack_id,
        )
        if host_indices is None:
            logger.error(f"Not enough host memory for request {req.rid}")
            return False

        if is_first_offload:
            self.offloaded_state[req.rid] = state

        # Early free prefill-offloaded GPU memory only after the incremental
        # D2H write is accepted. If host allocation fails, finish cleanup still
        # owns this prefix and must be able to release it normally.
        if state.prefill_len > 0 and state.inc_len == 0:
            self.token_to_kv_pool_allocator.free(token_indices[: state.prefill_len])

        self.ongoing_offload[ack_id] = (
            req,
            host_indices,
            incremental_indices,
            incremental_tokens,
            time.time(),
            start,
            end,
        )
        state.inc_len += incremental_aligned_len
        return True

    def check_offload_progress(self):
        """Check the progress of offload from device to host and backup from host to storage."""
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                len(cc.ack_write_queue),
                cc.ack_backup_queue.qsize(),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_write, n_backup = map(int, qsizes.tolist())
        self._check_offload_progress(n_write)
        self._check_backup_progress(n_backup)

    def _check_offload_progress(self, finish_count):
        """Check the progress of offload from device to host."""
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                if ack_id not in self.ongoing_offload:
                    continue
                (
                    req,
                    host_indices,
                    device_indices,
                    incremental_tokens,
                    start_time,
                    start,
                    end,
                ) = self.ongoing_offload.pop(ack_id)

                if req.rid in self.aborted_req_ids or req.req_pool_idx is None:
                    self.token_to_kv_pool_allocator.free(device_indices)
                    self.decode_host_mem_pool.free(host_indices)
                    if not any(
                        entry[0].rid == req.rid for entry in self.ongoing_offload.values()
                    ) and not any(
                        entry[0].rid == req.rid
                        for entry in self.ongoing_backup.values()
                    ):
                        self.aborted_req_ids.discard(req.rid)
                        if req.req_pool_idx is None or req.req_pool_idx == -1:
                            self.offloaded_state.pop(req.rid, None)
                    continue

                self.token_to_kv_pool_allocator.free(device_indices)

                prior_hash = (
                    self.offloaded_state[req.rid].last_hash
                    if req.rid in self.offloaded_state
                    else None
                )
                last_hash = self._trigger_backup(
                    req, host_indices, incremental_tokens, start_time, prior_hash
                )
                if req.rid in self.offloaded_state:
                    self.offloaded_state[req.rid].last_hash = last_hash
                if req.finished() and not any(
                    entry[0].rid == req.rid
                    for entry in self.ongoing_offload.values()
                ):
                    state = self.offloaded_state.get(req.rid)
                    if state is None:
                        start_offset = end
                    else:
                        start_offset = state.prefill_len + state.inc_len
                    self._release_finished_req(req, start_offset)
            finish_count -= 1

    def _release_finished_req(self, req: Req, start_offset: int):
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.request_finished(req)
        kv_committed_len = req.pop_committed_kv_cache()
        start = start_offset
        end = kv_committed_len
        # Free the incremental part of the request (NSA-aware)
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, start:end]
        self.token_to_kv_pool_allocator.free(kv_indices)

        # Free over-allocated KV cache slots (e.g. from speculative decoding v2).
        # Without spec v2, start_p == end_p so this is a no-op.
        start_p, end_p = req.pop_overallocated_kv_cache()
        if self.page_size > 1:
            start_p = ceil_align(start_p, self.page_size)
        if start_p < end_p:
            overalloc_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_p:end_p
            ]
            self.token_to_kv_pool_allocator.free(overalloc_indices)

        self.tree_cache.dec_lock_ref(
            req.last_node,
            DecLockRefParams(
                swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)
            ),
        )
        self.req_to_token_pool.free(req)
        if req.rid in self.offloaded_state:
            del self.offloaded_state[req.rid]

    def _check_backup_progress(self, finish_count):
        """Check the progress of backup from host to storage."""
        for _ in range(finish_count):
            storage_operation = self.cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            if ack_id not in self.ongoing_backup:
                continue
            req, host_indices, start_time = self.ongoing_backup.pop(ack_id)
            req_id = req.rid

            # Release host memory
            self.decode_host_mem_pool.free(host_indices)
            if req_id in self.aborted_req_ids and not any(
                entry[0].rid == req_id for entry in self.ongoing_offload.values()
            ) and not any(
                entry[0].rid == req_id for entry in self.ongoing_backup.values()
            ):
                if req.req_pool_idx is None or req.req_pool_idx == -1:
                    self.offloaded_state.pop(req_id, None)
                self.aborted_req_ids.discard(req_id)

            logger.debug(
                f"Finished backup request {req_id}, free host memory, len:{len(host_indices)}, cost time:{time.time() - start_time:.2f} seconds."
            )

    def _trigger_backup(
        self, req, host_indices, incremental_tokens, start_time, prior_hash
    ):
        """Trigger async backup from host to storage."""
        page_hashes = self._compute_prefix_hash(incremental_tokens, prior_hash)
        ack_id = self.cache_controller.write_storage(
            host_indices,
            incremental_tokens,
            hash_value=page_hashes,
        )
        self.ongoing_backup[ack_id] = (req, host_indices, start_time)
        return page_hashes[-1] if len(page_hashes) > 0 else prior_hash

    def _compute_prefix_hash(self, tokens, prior_hash=""):
        page_hashes = []
        last_hash = prior_hash
        for offset in range(0, len(tokens), self.page_size):
            page_tokens = tokens[offset : offset + self.page_size]
            last_hash = self.cache_controller.get_hash_str(page_tokens, last_hash)
            page_hashes.append(last_hash)
        return page_hashes

    def _release_after_scheduled_offloads(self, req: Req, *, is_abort: bool = False):
        """Free the req slot and KV ranges not already handed to D2H offload."""
        if req.req_pool_idx is None or req.req_pool_idx == -1:
            return
        state = self.offloaded_state.get(req.rid)
        has_pending_io = any(
            entry[0].rid == req.rid for entry in self.ongoing_offload.values()
        ) or any(entry[0].rid == req.rid for entry in self.ongoing_backup.values())
        if state is None and not has_pending_io:
            if self.hisparse_coordinator is not None:
                self.hisparse_coordinator.request_finished(req)
            release_kv_cache(req, self.tree_cache, is_insert=not is_abort)
            return
        if state is None:
            prefill_len = len(req.origin_input_ids) // self.page_size * self.page_size
            inc_len = 0
        else:
            prefill_len = state.prefill_len
            inc_len = state.inc_len
        # A state with no accepted incremental offload must not take the offload
        # release path: no prefill range was early-freed, and normal release_kv_cache
        # still owns radix/SWA lock handling for that request.
        if inc_len == 0 and not has_pending_io:
            self.offloaded_state.pop(req.rid, None)
            if self.hisparse_coordinator is not None:
                self.hisparse_coordinator.request_finished(req)
            release_kv_cache(req, self.tree_cache, is_insert=not is_abort)
            return
        start_offset = prefill_len + inc_len
        self._release_finished_req(req, start_offset)

    def finalize_release_on_finish(self, req: Req):
        """Free any remaining tail KV that was not offloaded due to non-aligned length."""
        self._release_after_scheduled_offloads(req)

    def release_on_abort(self, req: Req) -> None:
        """Release an aborted request without double-freeing offloaded chunks."""
        self.abort_request(req)
        self._release_after_scheduled_offloads(req, is_abort=True)

    def abort_request(self, req: Req) -> None:
        """Drop pending decode offload state for an aborted request.

        If the scheduler can release the request immediately, it should call
        release_on_abort().  For in-flight forwards we only mark pending async
        IO so ack handlers do not touch a freed req_pool_idx later.
        """
        has_pending_io = any(
            entry[0].rid == req.rid for entry in self.ongoing_offload.values()
        ) or any(entry[0].rid == req.rid for entry in self.ongoing_backup.values())
        if has_pending_io:
            self.aborted_req_ids.add(req.rid)
        elif req.req_pool_idx is None or req.req_pool_idx == -1:
            self.offloaded_state.pop(req.rid, None)
