from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import queue
import threading
import time
from collections import deque
from functools import reduce
from typing import Callable, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.utils import FastQueue, group_concurrent_contiguous
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    TransferInfo,
    TransferKVChunk,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    FAKE_BOOTSTRAP_HOST,
    kv_to_page_indices,
)
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.schedule_batch import ScheduleBatch, Req


_async_context = threading.local()


def set_layer_ready_callback(callback: Optional[Callable[[int], None]]) -> None:
    _async_context.layer_ready_callback = callback


def get_layer_ready_callback() -> Optional[Callable[[int], None]]:
    return getattr(_async_context, "layer_ready_callback", None)


@dataclasses.dataclass
class TransferKVChunkSet:
    rooms: Tuple[int] = dataclasses.field(default_factory=tuple)
    prefill_kv_indices: Tuple[npt.NDArray[np.int64]] = dataclasses.field(
        default_factory=tuple
    )
    index_slices: Tuple[slice] = dataclasses.field(default_factory=tuple)


@dataclasses.dataclass
class AsyncInfo:
    layer_ids: Tuple[int] = dataclasses.field(default_factory=tuple)
    kv_chunk_info: TransferKVChunkSet = dataclasses.field(
        default_factory=TransferKVChunkSet
    )


class StreamAsyncSubmitter:
    def __init__(self, submit_func):
        self._submit_func = submit_func
        self._queue: queue.Queue[int] = queue.Queue()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._submitted = 0
        self._finished = 0
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            self._queue.get()
            try:
                self._submit_func()
            finally:
                with self._cond:
                    self._finished += 1
                    self._cond.notify_all()

    def step_async(self):
        with self._cond:
            self._submitted += 1
            self._queue.put(self._submitted)
            return self._submitted

    def get_step_count(self):
        with self._cond:
            return self._submitted

    def wait_sent_finish(self, target_count: int):
        with self._cond:
            while self._finished < target_count:
                self._cond.wait()


def cached_group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int64], dst_indices: npt.NDArray[np.int64]
):
    src = np.asarray(src_indices, dtype=np.int32)
    dst = np.asarray(dst_indices, dtype=np.int32)
    return group_concurrent_contiguous(src, dst)


class MooncakeAsyncKVManager(MooncakeKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self._async_submitter = StreamAsyncSubmitter(self._put_kvcache_func)
        self._notify_queue = deque()
        self._waiting_rooms = deque()
        self._current_kv_chunk_infos: Optional[TransferKVChunkSet] = None
        self._req_begin_count: Dict[int, deque] = {}
        self._req_bids: Dict[int, deque] = {}
        self._lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self._kv_cache_ntensors = len(self.kv_args.kv_data_ptrs)
        self._kv_cache_nlayers = (
            len(self.kv_args.kv_data_ptrs)
            if self.is_mla_backend
            else len(self.kv_args.kv_data_ptrs) // 2
        )

    @property
    def is_support_async(self):
        return True

    def _put_kvcache_func(self):
        try:
            with self._queue_lock:
                if not self._notify_queue:
                    return
                info = self._notify_queue.pop()
            self._put_kv_cache_internal(info)
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.info(f"Error in put_kvcache_thread: {e}")
            import os

            os._exit(1)

    def get_info_with_risk(self, room: int) -> TransferInfo:
        while room not in self.transfer_infos:
            time.sleep(1e-3)
            logger.info(f"room {room} is not in transfer_infos")
        return self.transfer_infos[room]

    def submit_layer(
        self,
        session_id: str,
        src_ptr: int,
        dst_ptr: int,
        prefill_kv_blocks: npt.NDArray[np.int64],
        dst_kv_blocks: npt.NDArray[np.int64],
        item_len: int,
    ) -> int:
        prefill_kv_blocks_tmp, dst_kv_blocks_tmp = cached_group_concurrent_contiguous(
            prefill_kv_blocks, dst_kv_blocks
        )
        if not prefill_kv_blocks_tmp:
            return 0
        transfer_blocks = []
        for prefill_index, decode_index in zip(
            prefill_kv_blocks_tmp, dst_kv_blocks_tmp
        ):
            src_addr = src_ptr + int(prefill_index[0]) * item_len
            dst_addr = dst_ptr + int(decode_index[0]) * item_len
            length = item_len * len(prefill_index)
            transfer_blocks.append((src_addr, dst_addr, length))
        return self._transfer_data(session_id, transfer_blocks)

    def _put_kv_cache_internal(self, async_info: AsyncInfo):
        kv_chunk_info = async_info.kv_chunk_info
        if not kv_chunk_info.rooms:
            return
        infos = [self.get_info_with_risk(room) for room in kv_chunk_info.rooms]
        for layer_id in async_info.layer_ids:
            for room_id, transfer_info_dict, kv_indice, index_slice in zip(
                kv_chunk_info.rooms,
                infos,
                kv_chunk_info.prefill_kv_indices,
                kv_chunk_info.index_slices,
            ):
                for transfer_info in transfer_info_dict.values():
                    if not transfer_info.is_dummy:
                        dst = transfer_info.dst_kv_indices[index_slice]
                        session_id = transfer_info.mooncake_session_id
                        dst_kv_ptrs = self.decode_kv_args_table[
                            transfer_info.mooncake_session_id
                        ].dst_kv_ptrs
                        src_ptr = self.kv_args.kv_data_ptrs[layer_id]
                        dst_ptr = dst_kv_ptrs[layer_id]
                        item_len = self.kv_args.kv_item_lens[layer_id]
                        status = self.submit_layer(
                            session_id, src_ptr, dst_ptr, kv_indice, dst, item_len
                        )
                        with self._lock:
                            if room_id not in self._req_bids:
                                self._req_bids[room_id] = deque()
                            self._req_bids[room_id].appendleft(status)

    def mark_layer_ready(self, layer_id: int):
        if layer_id == 0:
            begin_count = self._async_submitter.get_step_count()
            with self._queue_lock:
                self._current_kv_chunk_infos = (
                    self._waiting_rooms.pop() if self._waiting_rooms else None
                )
            if self._current_kv_chunk_infos:
                for rid in self._current_kv_chunk_infos.rooms:
                    if rid not in self._req_begin_count:
                        self._req_begin_count[rid] = deque()
                    self._req_begin_count[rid].appendleft(begin_count)
        if self._current_kv_chunk_infos:
            send_layers = [layer_id]
            if not self.is_mla_backend:
                send_layers.append(layer_id + self._kv_cache_nlayers)
            with self._queue_lock:
                self._notify_queue.appendleft(
                    AsyncInfo(
                        layer_ids=tuple(send_layers),
                        kv_chunk_info=self._current_kv_chunk_infos,
                    )
                )
            self._async_submitter.step_async()

    def _is_bids_finished_func(self, rid: int):
        bids_not_finished = (
            rid not in self._req_bids or len(self._req_bids[rid]) < self._kv_cache_ntensors
        )
        return not bids_not_finished

    def pop_req_bids(self, rid: int, is_remove: bool):
        if is_remove:
            q = self._req_bids.pop(rid)
        else:
            q = self._req_bids[rid]
        rsts = []
        for _ in range(self._kv_cache_ntensors):
            rsts.append(q.pop())
        return rsts

    def _flush_all_layers(self, rid: int, is_last: bool):
        if rid not in self._req_begin_count:
            return
        start_time = time.time()
        if is_last:
            statuses = []
            while len(self._req_begin_count[rid]):
                begin_count = self._req_begin_count[rid].pop()
                self._async_submitter.wait_sent_finish(
                    begin_count + self._kv_cache_nlayers
                )
                while not self._is_bids_finished_func(rid):
                    time.sleep(1e-3)
                current_last = len(self._req_begin_count[rid]) == 0
                statuses = self.pop_req_bids(rid, current_last)
                for status in statuses:
                    assert status == 0, f"status is {status} in {rid}"
            self._req_begin_count.pop(rid, None)
            logger.info(
                f"finish send (rid={rid}, n_blocks={len(statuses)}) in {1000*(time.time() - start_time)} ms."
            )

    def prepare_batch(self, sch: "Scheduler", batch: "ScheduleBatch"):
        rooms = []
        prefill_kv_indices = []
        index_slices = []
        for req in batch.reqs:
            if req.is_chunked > 0:
                continue
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
                continue
            if not self._is_async_eligible_room(req.bootstrap_room):
                continue
            page_size = sch.token_to_kv_pool_allocator.page_size
            start_idx = req.start_send_idx
            end_idx = min(len(req.fill_ids), len(req.origin_input_ids))
            if end_idx <= start_idx:
                continue
            kv_indices = (
                sch.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
                .cpu()
                .numpy()
            )
            page_indices = kv_to_page_indices(kv_indices, page_size)
            if len(page_indices) == 0:
                continue
            index_slice = slice(
                req.disagg_kv_sender.curr_idx,
                req.disagg_kv_sender.curr_idx + len(page_indices),
            )
            bootstrap_room = req.bootstrap_room
            rooms.append(bootstrap_room)
            prefill_kv_indices.append(page_indices)
            index_slices.append(index_slice)
        if len(rooms):
            kv_chunk_info_set = TransferKVChunkSet(
                rooms=tuple(rooms),
                prefill_kv_indices=tuple(prefill_kv_indices),
                index_slices=tuple(index_slices),
            )
        else:
            kv_chunk_info_set = None
        with self._queue_lock:
            self._waiting_rooms.appendleft(kv_chunk_info_set)

    def get_layer_count(self) -> int:
        return self._kv_cache_nlayers

    def _is_async_eligible_room(self, room: int) -> bool:
        transfer_info_dict = self.transfer_infos.get(room)
        if transfer_info_dict is None:
            return False
        if self.is_mla_backend:
            for transfer_info in transfer_info_dict.values():
                if transfer_info.is_dummy:
                    continue
                registration = self.decode_kv_args_table.get(
                    transfer_info.mooncake_session_id
                )
                if registration is None:
                    return False
                if registration.dst_attn_tp_size != self.attn_tp_size:
                    return False
            return True
        for transfer_info in transfer_info_dict.values():
            if transfer_info.is_dummy:
                continue
            registration = self.decode_kv_args_table.get(transfer_info.mooncake_session_id)
            if registration is None:
                return False
            if registration.dst_attn_tp_size != self.attn_tp_size:
                return False
        return True

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []
                local_rank = self.attn_tp_rank * self.pp_size + self.pp_rank
                use_async = kv_chunk.room in self._req_begin_count
                if use_async and kv_chunk.is_last_chunk:
                    self._flush_all_layers(kv_chunk.room, True)
                for req in reqs_to_be_processed:
                    if not req.is_dummy:
                        with self.session_lock:
                            if req.mooncake_session_id in self.failed_sessions:
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Decode instance could be dead, remote mooncake session {req.mooncake_session_id} is not alive",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                    local_rank,
                                )
                                break

                        chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]

                        if len(chunked_dst_kv_indice) < len(
                            kv_chunk.prefill_kv_indices
                        ):
                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                                : len(chunked_dst_kv_indice)
                            ]

                        target_rank_registration_info: KVArgsRegisterInfo = (
                            self.decode_kv_args_table[req.mooncake_session_id]
                        )
                        if not use_async:
                            if self.is_mla_backend or (
                                self.attn_tp_size
                                == target_rank_registration_info.dst_attn_tp_size
                            ):
                                ret = self.send_kvcache(
                                    req.mooncake_session_id,
                                    kv_chunk.prefill_kv_indices,
                                    target_rank_registration_info.dst_kv_ptrs,
                                    chunked_dst_kv_indice,
                                    executor,
                                )
                            else:
                                ret = self.send_kvcache_slice(
                                    req.mooncake_session_id,
                                    kv_chunk.prefill_kv_indices,
                                    target_rank_registration_info.dst_kv_ptrs,
                                    chunked_dst_kv_indice,
                                    target_rank_registration_info.dst_tp_rank,
                                    target_rank_registration_info.dst_attn_tp_size,
                                    target_rank_registration_info.dst_kv_item_len,
                                    executor,
                                )
                            if ret != 0:
                                with self.session_lock:
                                    self.session_failures[req.mooncake_session_id] += 1
                                    if self.session_failures[req.mooncake_session_id] >= 1:
                                        self.failed_sessions.add(req.mooncake_session_id)
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Failed to send kv chunk of {kv_chunk.room} to {req.endpoint}:{req.dst_port}",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                    local_rank,
                                )
                                break

                        if kv_chunk.is_last_chunk:
                            if kv_chunk.state_indices is not None:
                                self.maybe_send_extra(
                                    req,
                                    kv_chunk.state_indices,
                                    target_rank_registration_info.dst_state_data_ptrs,
                                    executor,
                                    target_rank_registration_info,
                                )

                            ret = self.send_aux(
                                req,
                                kv_chunk.prefill_aux_index,
                                target_rank_registration_info.dst_aux_ptrs,
                            )
                            polls.append(True if ret == 0 else False)
                            dst_ranks_infos.append(
                                (req.endpoint, req.dst_port, req.room)
                            )

                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint, dst_port, room, status, local_rank
                                    )
                    else:
                        if kv_chunk.is_last_chunk and req.room in self.request_status:
                            self.update_status(req.room, KVPoll.Success)

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

            except Exception as e:
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )
