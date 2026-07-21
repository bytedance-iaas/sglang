from __future__ import annotations

import logging
import queue
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import torch
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.utils import FastQueue, TransferKVChunk

logger = logging.getLogger(__name__)


class PDHiddenKVManagerMixin:
    """Backend-agnostic PD hidden streaming state helpers.

    Backends still own the transport-specific READY/ACK packets and hidden-row
    data transfer. This mixin centralizes the shared waiter, lifetime, and
    request-done state needed by Mooncake, NIXL, MORI, or future backends.
    """

    def supports_pd_hidden_streaming(self) -> bool:
        return True

    def _init_prefill_pd_hidden_state(self) -> None:
        self.pd_hidden_done_rooms = set()
        self.pd_hidden_done_lock = threading.Lock()
        self.pd_hidden_chunk_acks = defaultdict(int)
        self.pd_hidden_chunk_ack_cv = threading.Condition()
        self.pd_hidden_ack_waiters = {}
        self.pd_hidden_room_waiters = defaultdict(deque)
        self.pd_hidden_inflight_chunks = {}
        self.pd_hidden_inflight_lock = threading.Lock()
        self.pd_hidden_active_transfers = defaultdict(int)
        self.pd_hidden_active_cv = threading.Condition()

    def _init_decode_pd_hidden_state(self) -> None:
        self.pd_hidden_ready_chunks: Dict[int, List[dict]] = defaultdict(list)
        self.pd_hidden_ready_lock = threading.Lock()
        self.pd_hidden_ack_completions = queue.SimpleQueue()
        self.pd_hidden_ack_pending_counts = defaultdict(int)
        self.pd_hidden_ack_completion_cv = threading.Condition()
        self.pd_hidden_acked_chunks: Dict[int, List[dict]] = defaultdict(list)
        self.pd_hidden_acked_lock = threading.Lock()
        self.pd_hidden_ack_wakeup_endpoint = f"inproc://pd-hidden-ack-{id(self)}"
        self.pd_hidden_ack_wakeup_receiver = self._zmq_ctx.socket(zmq.PULL)
        self.pd_hidden_ack_wakeup_receiver.bind(self.pd_hidden_ack_wakeup_endpoint)

    def mark_pd_hidden_request_done(
        self,
        bootstrap_room: int,
        state_indices: Optional[List] = None,
    ) -> None:
        if not hasattr(self, "pd_hidden_done_rooms"):
            return
        with self.pd_hidden_done_lock:
            room = int(bootstrap_room)
            if room in self.pd_hidden_done_rooms:
                return
            self.pd_hidden_done_rooms.add(room)
        pool = getattr(self, "pd_hidden_pool", None)
        state_idx = self._pd_hidden_state_index()
        if (
            pool is not None
            and state_indices is not None
            and state_idx is not None
            and state_idx < len(state_indices)
        ):
            indices = state_indices[state_idx]
            if indices is not None and len(indices) > 0:
                pool.free([int(idx) for idx in indices])

    def pop_pd_hidden_request_done(self, bootstrap_room: int) -> bool:
        if not hasattr(self, "pd_hidden_done_rooms"):
            return False
        with self.pd_hidden_done_lock:
            room = int(bootstrap_room)
            if room not in self.pd_hidden_done_rooms:
                return False
            self.pd_hidden_done_rooms.remove(room)
            return True

    def mark_pd_hidden_done(
        self,
        bootstrap_room: int,
        state_indices: Optional[List] = None,
    ) -> None:
        self.mark_pd_hidden_request_done(bootstrap_room, state_indices)

    def pop_pd_hidden_done(self, bootstrap_room: int) -> bool:
        return self.pop_pd_hidden_request_done(bootstrap_room)

    def park_pd_hidden_chunk_for_ack(
        self,
        *,
        transfer_queue: FastQueue,
        kv_chunk: TransferKVChunk,
        prefill_rank: int,
        expected_count: int,
        timeout_s: float = 300.0,
    ) -> bool:
        """Park a streaming hidden chunk until all Decode ACKs arrive.

        Returns True when the chunk was parked. False means ACKs were already
        available and the caller can finish the chunk immediately.
        """
        if kv_chunk.pd_hidden_start is None:
            return False
        key = (
            int(kv_chunk.room),
            int(prefill_rank),
            int(kv_chunk.pd_hidden_start),
        )
        expected_count = int(expected_count)
        if expected_count <= 0:
            kv_chunk.pd_hidden_ack_ready = True
            return False
        with self.pd_hidden_chunk_ack_cv:
            if self.pd_hidden_chunk_acks.get(key, 0) >= expected_count:
                self.pd_hidden_chunk_acks[key] -= expected_count
                if self.pd_hidden_chunk_acks[key] <= 0:
                    self.pd_hidden_chunk_acks.pop(key, None)
                kv_chunk.pd_hidden_ack_ready = True
                return False
            if key in self.pd_hidden_ack_waiters:
                raise RuntimeError(
                    "PD hidden ACK waiter already exists: "
                    f"room={key[0]}, prefill_rank={key[1]}, hidden_start={key[2]}"
                )
            kv_chunk.pd_hidden_ack_expected_count = expected_count
            self.pd_hidden_ack_waiters[key] = (transfer_queue, kv_chunk)

        def on_timeout() -> None:
            with self.pd_hidden_chunk_ack_cv:
                waiter = self.pd_hidden_ack_waiters.pop(key, None)
            if waiter is None:
                return
            _, timed_out_chunk = waiter
            timed_out_chunk.pd_hidden_ack_timed_out = True
            self.record_failure(
                key[0],
                "Timed out waiting for PD hidden chunk ACK: "
                f"prefill_rank={key[1]}, hidden_start={key[2]}",
            )
            self.update_status(key[0], KVPoll.Failed)
            transfer_queue.put(timed_out_chunk)
            self._wake_pd_hidden_ack_waiters(key[0])

        timer = threading.Timer(float(timeout_s), on_timeout)
        timer.daemon = True
        timer.start()
        return True

    def _wake_pd_hidden_ack_waiters(self, room: int) -> None:
        room = int(room)
        with self.pd_hidden_chunk_ack_cv:
            waiters = [
                self.pd_hidden_ack_waiters.pop(key)
                for key in list(self.pd_hidden_ack_waiters)
                if key[0] == room
            ]
            room_waiters = list(self.pd_hidden_room_waiters.pop(room, []))
        for transfer_queue, kv_chunk in waiters:
            transfer_queue.put(kv_chunk)
        for transfer_queue, kv_chunk in room_waiters:
            transfer_queue.put(kv_chunk)

    def _park_pd_hidden_chunk_behind_room(
        self, transfer_queue: FastQueue, kv_chunk: TransferKVChunk
    ) -> None:
        with self.pd_hidden_chunk_ack_cv:
            self.pd_hidden_room_waiters[int(kv_chunk.room)].append(
                (transfer_queue, kv_chunk)
            )

    def _wake_next_pd_hidden_room_waiter(self, room: int) -> None:
        with self.pd_hidden_chunk_ack_cv:
            room_waiters = self.pd_hidden_room_waiters.get(int(room))
            if not room_waiters:
                return
            transfer_queue, kv_chunk = room_waiters.popleft()
            if not room_waiters:
                self.pd_hidden_room_waiters.pop(int(room), None)
        transfer_queue.put(kv_chunk)

    def _handle_pd_hidden_chunk_ack(
        self, room: int, prefill_rank: int, hidden_start: int
    ) -> None:
        key = (int(room), int(prefill_rank), int(hidden_start))
        waiter_to_wake = None
        with self.pd_hidden_chunk_ack_cv:
            self.pd_hidden_chunk_acks[key] += 1
            waiter = self.pd_hidden_ack_waiters.get(key)
            if waiter is not None:
                _, kv_chunk = waiter
                expected_count = kv_chunk.pd_hidden_ack_expected_count
                if self.pd_hidden_chunk_acks[key] >= expected_count:
                    self.pd_hidden_chunk_acks[key] -= expected_count
                    if self.pd_hidden_chunk_acks[key] <= 0:
                        self.pd_hidden_chunk_acks.pop(key, None)
                    self.pd_hidden_ack_waiters.pop(key, None)
                    kv_chunk.pd_hidden_ack_ready = True
                    waiter_to_wake = waiter
            self.pd_hidden_chunk_ack_cv.notify_all()
        if waiter_to_wake is not None:
            transfer_queue, kv_chunk = waiter_to_wake
            transfer_queue.put(kv_chunk)

    def pop_pd_hidden_ready_chunks(self, room: int) -> List[dict]:
        if not hasattr(self, "pd_hidden_ready_chunks"):
            return []
        with self.pd_hidden_ready_lock:
            return self.pd_hidden_ready_chunks.pop(int(room), [])

    def submit_pd_hidden_chunk_ack(
        self,
        *,
        event,
        remote: str,
        dst_port: int,
        room: int,
        prefill_rank: int,
        hidden_start: int,
        is_last_hidden_chunk: bool,
    ) -> None:
        completion = {
            "remote": remote,
            "dst_port": int(dst_port),
            "room": int(room),
            "prefill_rank": int(prefill_rank),
            "hidden_start": int(hidden_start),
            "is_last_hidden_chunk": bool(is_last_hidden_chunk),
        }
        with self.pd_hidden_ack_completion_cv:
            self.pd_hidden_ack_pending_counts[int(room)] += 1

        def wait_for_injection() -> None:
            try:
                if event is not None:
                    with torch.cuda.device(self.kv_args.gpu_id):
                        event.synchronize()
                completion["success"] = True
            except Exception:
                logger.exception(
                    "PD hidden injection completion failed: room=%s start=%s",
                    room,
                    hidden_start,
                )
                completion["success"] = False
            self.pd_hidden_ack_completions.put(completion)
            wakeup_sender = self._zmq_ctx.socket(zmq.PUSH)
            wakeup_sender.setsockopt(zmq.LINGER, 0)
            try:
                wakeup_sender.connect(self.pd_hidden_ack_wakeup_endpoint)
                wakeup_sender.send(b"ACK_READY")
            finally:
                wakeup_sender.close()

        threading.Thread(
            target=wait_for_injection,
            name=f"PDHiddenAckWaiter-{room}",
            daemon=True,
        ).start()

    def _drain_pd_hidden_ack_completions(self) -> None:
        while True:
            try:
                completion = self.pd_hidden_ack_completions.get_nowait()
            except queue.Empty:
                return

            room = int(completion["room"])
            try:
                if completion.pop("success"):
                    self.ack_pd_hidden_chunk(
                        remote=completion["remote"],
                        dst_port=int(completion["dst_port"]),
                        room=room,
                        prefill_rank=int(completion["prefill_rank"]),
                        hidden_start=int(completion["hidden_start"]),
                    )
                    with self.pd_hidden_acked_lock:
                        self.pd_hidden_acked_chunks[room].append(completion)
                else:
                    self.record_failure(
                        room,
                        "PD hidden injection CUDA completion failed: "
                        f"hidden_start={completion['hidden_start']}",
                    )
                    self.update_status(room, KVPoll.Failed)
            except Exception:
                logger.exception(
                    "Failed to send PD hidden chunk ACK: room=%s start=%s",
                    room,
                    completion["hidden_start"],
                )
                self.record_failure(
                    room,
                    "Failed to send PD hidden chunk ACK: "
                    f"hidden_start={completion['hidden_start']}",
                )
                self.update_status(room, KVPoll.Failed)
            finally:
                with self.pd_hidden_ack_completion_cv:
                    self.pd_hidden_ack_pending_counts[room] -= 1
                    if self.pd_hidden_ack_pending_counts[room] <= 0:
                        self.pd_hidden_ack_pending_counts.pop(room, None)
                    self.pd_hidden_ack_completion_cv.notify_all()

    def pop_pd_hidden_acked_chunks(self, room: int) -> List[dict]:
        with self.pd_hidden_acked_lock:
            return self.pd_hidden_acked_chunks.pop(int(room), [])

    def wait_pd_hidden_ack_completions(
        self, room: int, timeout_s: float = 300.0
    ) -> bool:
        room = int(room)
        deadline = time.monotonic() + float(timeout_s)
        with self.pd_hidden_ack_completion_cv:
            while self.pd_hidden_ack_pending_counts.get(room, 0) > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.pd_hidden_ack_completion_cv.wait(timeout=min(remaining, 1.0))
        return True

    def _begin_pd_hidden_transfer(self, room: int) -> None:
        if not hasattr(self, "pd_hidden_active_cv"):
            return
        with self.pd_hidden_active_cv:
            self.pd_hidden_active_transfers[int(room)] += 1

    def _end_pd_hidden_transfer(self, room: int) -> None:
        if not hasattr(self, "pd_hidden_active_cv"):
            return
        with self.pd_hidden_active_cv:
            room = int(room)
            count = self.pd_hidden_active_transfers.get(room, 0) - 1
            if count <= 0:
                self.pd_hidden_active_transfers.pop(room, None)
            else:
                self.pd_hidden_active_transfers[room] = count
            self.pd_hidden_active_cv.notify_all()

    def _wait_pd_hidden_transfers_quiesced(
        self, room: int, timeout_s: float = 300.0
    ) -> bool:
        if not hasattr(self, "pd_hidden_active_cv"):
            return True
        deadline = time.monotonic() + float(timeout_s)
        room = int(room)
        with self.pd_hidden_active_cv:
            while self.pd_hidden_active_transfers.get(room, 0) > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error(
                        "Timed out waiting for PD hidden transfers to quiesce: "
                        "room=%s active=%s",
                        room,
                        self.pd_hidden_active_transfers.get(room, 0),
                    )
                    return False
                self.pd_hidden_active_cv.wait(timeout=min(remaining, 1.0))
            return True

    def _pd_hidden_state_index(self) -> Optional[int]:
        for idx, state_type in enumerate(getattr(self.kv_args, "state_types", [])):
            if state_type == StateType.PD_HIDDEN:
                return idx
        return None

    def _has_pd_hidden_state(self, state_indices: Optional[List]) -> bool:
        idx = self._pd_hidden_state_index()
        if idx is None or not state_indices or idx >= len(state_indices):
            return False
        indices = state_indices[idx]
        return indices is not None and len(indices) > 0

    def _without_pd_hidden_state(
        self, state_indices: Optional[List]
    ) -> Optional[List]:
        idx = self._pd_hidden_state_index()
        if idx is None or not state_indices or idx >= len(state_indices):
            return state_indices
        ret = list(state_indices)
        ret[idx] = None
        return ret

    def _pd_hidden_release_state_indices(
        self, kv_chunk: TransferKVChunk
    ) -> Optional[List]:
        release_indices = kv_chunk.pd_hidden_release_indices
        if not release_indices:
            return kv_chunk.state_indices
        idx = self._pd_hidden_state_index()
        if idx is None or not kv_chunk.state_indices or idx >= len(kv_chunk.state_indices):
            return kv_chunk.state_indices
        ret = list(kv_chunk.state_indices)
        ret[idx] = [int(x) for x in release_indices]
        return ret

    def _free_pd_hidden_state_indices(self, state_indices: Optional[List]) -> None:
        pool = getattr(self, "pd_hidden_pool", None)
        state_idx = self._pd_hidden_state_index()
        if (
            pool is None
            or state_idx is None
            or state_indices is None
            or state_idx >= len(state_indices)
        ):
            return
        indices = state_indices[state_idx]
        if indices is not None and len(indices) > 0:
            pool.free([int(idx) for idx in indices])

    def _free_pd_hidden_chunk_rows(self, kv_chunk: TransferKVChunk) -> None:
        self._free_pd_hidden_state_indices(
            self._pd_hidden_release_state_indices(kv_chunk)
        )

    def _release_or_mark_pd_hidden_done(self, kv_chunk: TransferKVChunk) -> None:
        if kv_chunk.pd_hidden_start is not None:
            self._free_pd_hidden_chunk_rows(kv_chunk)
        else:
            self.mark_pd_hidden_request_done(
                kv_chunk.room,
                self._pd_hidden_release_state_indices(kv_chunk),
            )

    def _finish_pd_hidden_streaming_chunk(
        self,
        kv_chunk: TransferKVChunk,
        hidden_inflight_key: Optional[Tuple[int, int]],
    ) -> None:
        self._free_pd_hidden_chunk_rows(kv_chunk)
        if hidden_inflight_key is not None:
            with self.pd_hidden_inflight_lock:
                if self.pd_hidden_inflight_chunks.get(kv_chunk.room) == hidden_inflight_key:
                    self.pd_hidden_inflight_chunks.pop(kv_chunk.room, None)
            self._wake_next_pd_hidden_room_waiter(kv_chunk.room)

    def _mark_session_failed_and_sync(
        self,
        *,
        kv_chunk: TransferKVChunk,
        req,
        prefill_unique_rank: int,
        failure_reason: str,
    ) -> None:
        session_id = getattr(req, "mooncake_session_id", None)
        if session_id is None:
            session_id = getattr(req, "session_id")
        with self.session_lock:
            self.session_failures[session_id] += 1
            if self.session_failures[session_id] >= 1:
                self.failed_sessions.add(session_id)
                logger.error("Session %s failed.", session_id)
        self.record_failure(kv_chunk.room, failure_reason)
        self.update_status(kv_chunk.room, KVPoll.Failed)
        self._wake_pd_hidden_ack_waiters(kv_chunk.room)
        self.sync_status_to_decode_endpoint(
            req.endpoint,
            req.dst_port,
            req.room,
            KVPoll.Failed,
            prefill_unique_rank,
        )
