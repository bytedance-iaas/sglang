from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import queue
import threading
import time
import os
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


@dataclasses.dataclass
class TransferKVChunkSet:
    rooms: Tuple[int] = dataclasses.field(default_factory=tuple)
    prefill_kv_indices: Tuple[npt.NDArray[np.int64]] = dataclasses.field(
        default_factory=tuple
    )
    index_slices: Tuple[slice] = dataclasses.field(default_factory=tuple)
    prefill_state_indices: Tuple[int] = dataclasses.field(default_factory=tuple)


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

    def get_progress(self):
        with self._cond:
            return self._submitted, self._finished


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
        self._req_tensor_seen: Dict[int, set[int]] = {}
        self._room_to_kv_chunk_info: Dict[int, tuple[TransferKVChunkSet, int]] = {}
        self._lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self._debug_mamba_extra_logged_rooms: set[int] = set()
        self._kv_tensor_ntensors = len(self.kv_args.kv_data_ptrs)
        self._state_tensor_ntensors = len(getattr(self.kv_args, "state_data_ptrs", []))
        self._tensor_ntensors_total = self._kv_tensor_ntensors + self._state_tensor_ntensors
        self._mamba_num_layers_debug = 0
        self._mamba_state_tensors_per_layer_debug = 0
        self._kv_cache_nlayers = (
            self._kv_tensor_ntensors
            if self.is_mla_backend
            else self._kv_tensor_ntensors // 2
        )
        self._kv_per_layer_event_sync = (
            os.getenv("SGLANG_ASYNC_KV_GQA_PER_LAYER_EVENT_SYNC", "1") == "1"
        )
        self._mamba_per_layer_event_sync = (
            os.getenv("SGLANG_ASYNC_KV_MAMBA_PER_LAYER_EVENT_SYNC", "1") == "1"
        )
        self._mamba_per_layer_cuda_sync = (
            os.getenv("SGLANG_ASYNC_KV_MAMBA_PER_LAYER_CUDA_SYNC", "0") == "1"
        )
        self._async_kv_missing_wait_ms = int(
            os.getenv("SGLANG_ASYNC_KV_MISSING_WAIT_MS", "20")
        )
        self._mamba_per_layer_cuda_sync_warned = False
        self._mamba_per_layer_event_sync_warned = False
        self._mamba_layer_ready_events: Dict[Tuple[int, int], object] = {}

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
        if room not in self.transfer_infos:
            status = self.request_status.get(room)
            if status != KVPoll.Success:
                logger.warning(
                    "async kv skip: room=%s not in transfer_infos status=%s",
                    room,
                    status,
                )
            return {}
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
            for room_id, transfer_info_dict, kv_indice, index_slice, prefill_state_idx in zip(
                kv_chunk_info.rooms,
                infos,
                kv_chunk_info.prefill_kv_indices,
                kv_chunk_info.index_slices,
                kv_chunk_info.prefill_state_indices,
            ):
                if not transfer_info_dict:
                    continue
                for transfer_info in transfer_info_dict.values():
                    if not transfer_info.is_dummy:
                        dst = transfer_info.dst_kv_indices[index_slice]
                        session_id = transfer_info.mooncake_session_id
                        registration = self.decode_kv_args_table.get(
                            transfer_info.mooncake_session_id
                        )
                        if registration is None:
                            logger.warning(
                                "async kv skip: missing registration room=%s session=%s layer=%s",
                                room_id,
                                session_id,
                                layer_id,
                            )
                            continue

                        is_state_tensor = layer_id >= self._kv_tensor_ntensors
                        if not is_state_tensor:
                            if self._kv_per_layer_event_sync:
                                event_key = (int(room_id), int(layer_id))
                                with self._lock:
                                    event = self._mamba_layer_ready_events.pop(
                                        event_key, None
                                    )
                                if event is not None:
                                    try:
                                        import torch

                                        if torch.cuda.is_available():
                                            event.synchronize()
                                    except Exception:
                                        pass
                            dst = transfer_info.dst_kv_indices[index_slice]
                            if len(dst) < len(kv_indice):
                                logger.warning(
                                    "async kv indices mismatch: room=%s session=%s tensor=%s dst_len=%s src_len=%s slice=%s",
                                    room_id,
                                    session_id,
                                    layer_id,
                                    len(dst),
                                    len(kv_indice),
                                    index_slice,
                                )
                                kv_indice = kv_indice[: len(dst)]
                            dst_ptrs = registration.dst_kv_ptrs
                            kv_ptrs_len = self._kv_tensor_ntensors
                            dst_ptrs_len = len(dst_ptrs)
                            if layer_id >= kv_ptrs_len or layer_id >= dst_ptrs_len:
                                logger.warning(
                                    "async kv skip: invalid kv tensor index room=%s session=%s tensor=%s kv_ptrs_len=%s dst_ptrs_len=%s",
                                    room_id,
                                    session_id,
                                    layer_id,
                                    kv_ptrs_len,
                                    dst_ptrs_len,
                                )
                                continue
                            src_ptr = self.kv_args.kv_data_ptrs[layer_id]
                            dst_ptr = dst_ptrs[layer_id]
                            item_len = self.kv_args.kv_item_lens[layer_id]
                            status = self.submit_layer(
                                session_id, src_ptr, dst_ptr, kv_indice, dst, item_len
                            )
                            if status != 0:
                                logger.warning(
                                    f"async kv layer transfer failed: layer={layer_id} room={room_id} status={status}"
                                )
                        else:
                            state_tensor_id = layer_id - self._kv_tensor_ntensors
                            if prefill_state_idx is None or prefill_state_idx < 0:
                                continue
                            if not transfer_info.dst_state_indices:
                                continue
                            if (
                                getattr(self.kv_args, "state_type", "none") == "mamba"
                                and self._mamba_per_layer_event_sync
                            ):
                                event_key = (int(room_id), int(layer_id))
                                with self._lock:
                                    event = self._mamba_layer_ready_events.pop(
                                        event_key, None
                                    )
                                if event is not None:
                                    try:
                                        import torch

                                        if torch.cuda.is_available():
                                            event.synchronize()
                                    except Exception:
                                        pass
                            dst_state_idx = int(transfer_info.dst_state_indices[0])
                            src_state_ptrs = getattr(self.kv_args, "state_data_ptrs", [])
                            src_state_item_lens = getattr(self.kv_args, "state_item_lens", [])
                            dst_state_ptrs = getattr(registration, "dst_state_data_ptrs", [])
                            if (
                                state_tensor_id >= len(src_state_ptrs)
                                or state_tensor_id >= len(dst_state_ptrs)
                                or state_tensor_id >= len(src_state_item_lens)
                            ):
                                logger.warning(
                                    "async kv skip: invalid state tensor index room=%s session=%s tensor=%s state_ptrs_len=%s dst_state_ptrs_len=%s",
                                    room_id,
                                    session_id,
                                    state_tensor_id,
                                    len(src_state_ptrs),
                                    len(dst_state_ptrs),
                                )
                                continue
                            item_len = int(src_state_item_lens[state_tensor_id])
                            src_addr = int(src_state_ptrs[state_tensor_id]) + item_len * int(
                                prefill_state_idx
                            )
                            dst_addr = int(dst_state_ptrs[state_tensor_id]) + item_len * int(
                                dst_state_idx
                            )
                            status = self._transfer_data(
                                session_id, [(src_addr, dst_addr, item_len)]
                            )

                        with self._lock:
                            if room_id not in self._req_tensor_seen:
                                self._req_tensor_seen[room_id] = set()
                            self._req_tensor_seen[room_id].add(int(layer_id))
                            if room_id not in self._req_bids:
                                self._req_bids[room_id] = deque()
                            self._req_bids[room_id].appendleft(status)

    def mark_layer_ready(self, layer_id: int):
        tensor_id = layer_id
        if tensor_id == -1 or (tensor_id >= 0 and self._current_kv_chunk_infos is None):
            begin_count = self._async_submitter.get_step_count()
            with self._queue_lock:
                self._current_kv_chunk_infos = (
                    self._waiting_rooms.pop() if self._waiting_rooms else None
                )
                waiting_len = len(self._waiting_rooms)
            if self._current_kv_chunk_infos:
                for idx, rid in enumerate(self._current_kv_chunk_infos.rooms):
                    if rid not in self._req_begin_count:
                        self._req_begin_count[rid] = deque()
                    self._req_begin_count[rid].appendleft(begin_count)
                    self._room_to_kv_chunk_info[rid] = (self._current_kv_chunk_infos, idx)
            else:
                logger.warning(
                    f"async kv layer0: no waiting rooms, waiting_len={waiting_len}"
                )
            if tensor_id == -1:
                return
        if tensor_id < 0 or tensor_id >= self._tensor_ntensors_total:
            logger.warning(
                "async kv layer ready skipped: tensor=%s total_ntensors=%s kv_ntensors=%s nlayers=%s kv_ptrs_len=%s state_ptrs_len=%s start_layer=%s",
                tensor_id,
                self._tensor_ntensors_total,
                self._kv_tensor_ntensors,
                self._kv_cache_nlayers,
                self._kv_tensor_ntensors,
                self._state_tensor_ntensors,
                self.kv_args.prefill_start_layer,
            )
            return

        if self._current_kv_chunk_infos:
            rooms = self._current_kv_chunk_infos.rooms
            if rooms:
                keep_indices = []
                for idx, rid in enumerate(rooms):
                    if (
                        rid in self.transfer_infos
                        and self.request_status.get(rid) != KVPoll.Success
                    ):
                        keep_indices.append(idx)
                if not keep_indices:
                    return
                if len(keep_indices) != len(rooms):
                    filtered_rooms = tuple(rooms[i] for i in keep_indices)
                    filtered_prefill_kv = tuple(
                        self._current_kv_chunk_infos.prefill_kv_indices[i]
                        for i in keep_indices
                    )
                    filtered_index_slices = tuple(
                        self._current_kv_chunk_infos.index_slices[i]
                        for i in keep_indices
                    )
                    filtered_state_indices = tuple(
                        self._current_kv_chunk_infos.prefill_state_indices[i]
                        for i in keep_indices
                    )
                    self._current_kv_chunk_infos = TransferKVChunkSet(
                        rooms=filtered_rooms,
                        prefill_kv_indices=filtered_prefill_kv,
                        index_slices=filtered_index_slices,
                        prefill_state_indices=filtered_state_indices,
                    )
                    with self._lock:
                        for rid in rooms:
                            if rid not in filtered_rooms:
                                self._room_to_kv_chunk_info.pop(rid, None)
                        for idx, rid in enumerate(filtered_rooms):
                            self._room_to_kv_chunk_info[rid] = (
                                self._current_kv_chunk_infos,
                                idx,
                            )
            if tensor_id < self._kv_tensor_ntensors:
                kind = "kv"
            else:
                kind = "state"
            if (
                kind == "state"
                and getattr(self.kv_args, "state_type", "none") == "mamba"
                and not self._mamba_per_layer_event_sync
                and self._mamba_per_layer_cuda_sync
            ):
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        if not self._mamba_per_layer_cuda_sync_warned:
                            self._mamba_per_layer_cuda_sync_warned = True
                            logger.warning(
                                "async kv mamba per-layer cuda sync enabled; may reduce overlap"
                            )
                except Exception:
                    pass
            if (
                kind == "state"
                and getattr(self.kv_args, "state_type", "none") == "mamba"
                and self._mamba_per_layer_event_sync
            ):
                try:
                    import torch

                    if torch.cuda.is_available():
                        event = torch.cuda.Event(
                            enable_timing=False, blocking=False, interprocess=False
                        )
                        event.record()
                        with self._lock:
                            for rid in self._current_kv_chunk_infos.rooms:
                                self._mamba_layer_ready_events[(int(rid), int(tensor_id))] = event
                        if not self._mamba_per_layer_event_sync_warned:
                            self._mamba_per_layer_event_sync_warned = True
                            logger.info("async kv mamba per-layer event sync enabled")
                except Exception:
                    pass
            if kind == "kv" and self._kv_per_layer_event_sync:
                try:
                    import torch

                    if torch.cuda.is_available():
                        event = torch.cuda.Event(
                            enable_timing=False, blocking=False, interprocess=False
                        )
                        event.record()
                        with self._lock:
                            for rid in self._current_kv_chunk_infos.rooms:
                                self._mamba_layer_ready_events[(int(rid), int(tensor_id))] = event
                except Exception:
                    pass
            kv_role = None
            kv_layer = None
            if kind == "kv" and not self.is_mla_backend and self._kv_cache_nlayers > 0:
                if tensor_id < self._kv_cache_nlayers:
                    kv_role = "k"
                    kv_layer = int(tensor_id)
                else:
                    kv_role = "v"
                    kv_layer = int(tensor_id - self._kv_cache_nlayers)
            send_layers = [tensor_id]
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
            rid not in self._req_bids
            or len(self._req_bids[rid]) < self._tensor_ntensors_total
        )
        return not bids_not_finished

    def _resend_missing_state_tensors(
        self, room: int, missing_state_tensor_ids: list[int]
    ) -> None:
        if not missing_state_tensor_ids:
            return
        info = self._room_to_kv_chunk_info.get(room)
        if info is None:
            logger.warning(
                "async kv resend missing state skipped: room=%s reason=no_kv_chunk_info",
                room,
            )
            return
        kv_chunk_info, idx = info
        if idx >= len(kv_chunk_info.prefill_state_indices):
            logger.warning(
                "async kv resend missing state skipped: room=%s reason=idx_oob idx=%s",
                room,
                idx,
            )
            return
        prefill_state_idx = kv_chunk_info.prefill_state_indices[idx]
        if prefill_state_idx is None or prefill_state_idx < 0:
            logger.warning(
                "async kv resend missing state skipped: room=%s reason=prefill_state_idx_invalid idx=%s",
                room,
                prefill_state_idx,
            )
            return
        transfer_info_dict = self.transfer_infos.get(room)
        if not transfer_info_dict:
            logger.warning(
                "async kv resend missing state skipped: room=%s reason=no_transfer_info",
                room,
            )
            return
        src_state_ptrs = getattr(self.kv_args, "state_data_ptrs", [])
        src_state_item_lens = getattr(self.kv_args, "state_item_lens", [])
        for transfer_info in transfer_info_dict.values():
            if transfer_info.is_dummy:
                continue
            session_id = transfer_info.mooncake_session_id
            registration = self.decode_kv_args_table.get(session_id)
            if registration is None:
                continue
            if not transfer_info.dst_state_indices:
                continue
            dst_state_ptrs = getattr(registration, "dst_state_data_ptrs", [])
            dst_state_idx = int(transfer_info.dst_state_indices[0])
            transfer_blocks = []
            for state_tensor_id in missing_state_tensor_ids:
                if (
                    state_tensor_id >= len(src_state_ptrs)
                    or state_tensor_id >= len(dst_state_ptrs)
                    or state_tensor_id >= len(src_state_item_lens)
                ):
                    continue
                item_len = int(src_state_item_lens[state_tensor_id])
                src_addr = int(src_state_ptrs[state_tensor_id]) + item_len * int(
                    prefill_state_idx
                )
                dst_addr = int(dst_state_ptrs[state_tensor_id]) + item_len * int(
                    dst_state_idx
                )
                transfer_blocks.append((src_addr, dst_addr, item_len))
            if transfer_blocks:
                status = self._transfer_data(session_id, transfer_blocks)
                logger.warning(
                    "async kv resend missing state: room=%s session=%s tensors=%s status=%s",
                    room,
                    session_id,
                    len(transfer_blocks),
                    status,
                )

    def pop_req_bids(self, rid: int, is_remove: bool):
        if is_remove:
            q = self._req_bids.pop(rid)
        else:
            q = self._req_bids[rid]
        rsts = []
        for _ in range(self._tensor_ntensors_total):
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
                    begin_count + self._tensor_ntensors_total
                )
                while not self._is_bids_finished_func(rid):
                    time.sleep(1e-3)
                current_last = len(self._req_begin_count[rid]) == 0
                statuses = self.pop_req_bids(rid, current_last)
                for status in statuses:
                    assert status == 0, f"status is {status} in {rid}"
                if current_last:
                    seen = self._req_tensor_seen.get(rid, set())
                    missing_kv = [
                        i for i in range(self._kv_tensor_ntensors) if i not in seen
                    ]
                    missing_state = [
                        i
                        for i in range(self._kv_tensor_ntensors, self._tensor_ntensors_total)
                        if i not in seen
                    ]
                    if missing_state and self._async_kv_missing_wait_ms > 0:
                        wait_start = time.time()
                        while time.time() - wait_start < self._async_kv_missing_wait_ms / 1000.0:
                            time.sleep(1e-3)
                            seen = self._req_tensor_seen.get(rid, set())
                            missing_kv = [
                                i for i in range(self._kv_tensor_ntensors) if i not in seen
                            ]
                            missing_state = [
                                i
                                for i in range(self._kv_tensor_ntensors, self._tensor_ntensors_total)
                                if i not in seen
                            ]
                            if not missing_state and not missing_kv:
                                break
                    if missing_state:
                        missing_state_tensor_ids = [
                            i - self._kv_tensor_ntensors for i in missing_state
                        ]
                        self._resend_missing_state_tensors(
                            rid, missing_state_tensor_ids
                        )
                    if missing_kv:
                        logger.warning(
                            "async kv flush kv: room=%s missing=%s missing_count=%s seen_kv_count=%s kv_total=%s",
                            rid,
                            tuple(missing_kv[:50]),
                            len(missing_kv),
                            self._kv_tensor_ntensors - len(missing_kv),
                            self._kv_tensor_ntensors,
                        )
                    missing = missing_kv + missing_state
                    if missing:
                        logger.warning(
                            "async kv flush: room=%s missing_tensors=%s missing_count=%s seen_count=%s total=%s",
                            rid,
                            tuple(missing[:50]),
                            len(missing),
                            len(seen),
                            self._tensor_ntensors_total,
                        )
                    with self._lock:
                        self._req_tensor_seen.pop(rid, None)
                    self._room_to_kv_chunk_info.pop(rid, None)
                    if self._mamba_layer_ready_events:
                        with self._lock:
                            keys_to_remove = [
                                k for k in self._mamba_layer_ready_events.keys() if k[0] == rid
                            ]
                            for k in keys_to_remove:
                                self._mamba_layer_ready_events.pop(k, None)
            self._req_begin_count.pop(rid, None)

    def prepare_batch(self, sch: "Scheduler", batch: "ScheduleBatch"):
        rooms = []
        prefill_kv_indices = []
        index_slices = []
        prefill_state_indices = []
        skip_chunked = 0
        skip_fake = 0
        skip_ineligible = 0
        skip_empty = 0
        if (
            getattr(self.kv_args, "state_type", "none") == "mamba"
            and self._mamba_num_layers_debug == 0
        ):
            mamba_pool = getattr(getattr(sch, "req_to_token_pool", None), "mamba_pool", None)
            if mamba_pool is not None:
                self._mamba_num_layers_debug = int(getattr(mamba_pool, "num_mamba_layers", 0))
                mamba_cache = getattr(mamba_pool, "mamba_cache", None)
                state_tensors = []
                if mamba_cache is not None:
                    for field in vars(mamba_cache):
                        if field in ("intermediate_ssm", "intermediate_conv_window"):
                            continue
                        value = getattr(mamba_cache, field)
                        if isinstance(value, list):
                            state_tensors.extend(value)
                        else:
                            state_tensors.append(value)
                self._mamba_state_tensors_per_layer_debug = len(state_tensors)
        for req in batch.reqs:
            if req.is_chunked > 0:
                skip_chunked += 1
                continue
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
                skip_fake += 1
                continue
            eligible = self._is_async_eligible_room(req.bootstrap_room)
            if not eligible:
                skip_ineligible += 1
                continue
            page_size = sch.token_to_kv_pool_allocator.page_size
            start_idx = req.start_send_idx
            end_idx = min(len(req.fill_ids), len(req.origin_input_ids))
            if end_idx <= start_idx:
                skip_empty += 1
                continue
            kv_indices = (
                sch.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
                .cpu()
                .numpy()
            )
            page_indices = kv_to_page_indices(kv_indices, page_size)
            if len(page_indices) == 0:
                skip_empty += 1
                continue
            index_slice = slice(
                req.disagg_kv_sender.curr_idx,
                req.disagg_kv_sender.curr_idx + len(page_indices),
            )
            bootstrap_room = req.bootstrap_room
            rooms.append(bootstrap_room)
            prefill_kv_indices.append(page_indices)
            index_slices.append(index_slice)
            state_idx = -1
            if getattr(self.kv_args, "state_type", "none") == "mamba":
                try:
                    state_idx = int(
                        sch.req_to_token_pool.req_index_to_mamba_index_mapping[
                            req.req_pool_idx
                        ].item()
                    )
                except Exception:
                    state_idx = -1
            prefill_state_indices.append(state_idx)
        if len(rooms):
            kv_chunk_info_set = TransferKVChunkSet(
                rooms=tuple(rooms),
                prefill_kv_indices=tuple(prefill_kv_indices),
                index_slices=tuple(index_slices),
                prefill_state_indices=tuple(prefill_state_indices),
            )
        else:
            kv_chunk_info_set = None
        with self._queue_lock:
            self._waiting_rooms.appendleft(kv_chunk_info_set)
        waiting_len = len(self._waiting_rooms)

    def get_layer_count(self) -> int:
        return self._kv_cache_nlayers

    def _is_async_eligible_room(self, room: int) -> bool:
        transfer_info_dict = self.transfer_infos.get(room)
        if transfer_info_dict is None:
            logger.warning(f"async kv ineligible: room={room} reason=missing_transfer_info")
            return False
        if not transfer_info_dict:
            logger.warning(f"async kv ineligible: room={room} reason=empty_transfer_info")
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
        if all(t.is_dummy for t in transfer_info_dict.values()):
            logger.warning(f"async kv ineligible: room={room} reason=all_dummy")
            return False
        return True

    def transfer_worker(
        self,
        queue: FastQueue,
        executor: concurrent.futures.ThreadPoolExecutor,
        staging_buffer=None,
    ):
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                worker_start = time.time()
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []
                local_rank = self.attn_tp_rank * self.pp_size + self.pp_rank
                use_async = kv_chunk.room in self._req_begin_count
                if not use_async:
                    with self._queue_lock:
                        waiting_len = len(self._waiting_rooms)
                        current_rooms = (
                            self._current_kv_chunk_infos.rooms
                            if self._current_kv_chunk_infos
                            else None
                        )
                    logger.warning(
                        "async kv transfer: use_async=False "
                        f"room={kv_chunk.room} last={kv_chunk.is_last_chunk} "
                        f"req_begin_count_keys={len(self._req_begin_count)} "
                        f"waiting_len={waiting_len} current_rooms={current_rooms}"
                    )
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
                                if not use_async:
                                    ret_extra = self.maybe_send_extra(
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
