"""Unit tests for completion-driven PD hidden row allocation."""

from __future__ import annotations

import threading
import unittest
from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    PDHiddenRequestState,
)
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.hidden_events import PDHiddenEventManager
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import PDHiddenRowPool
from sglang.srt.speculative.dspark_components.dspark_disaggregation import (
    resolve_hidden_bootstrap_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _alloc_request(
    room: int,
    rows: int,
    *,
    hidden_start: int = 0,
    hidden_end: int | None = None,
) -> dict:
    hidden_end = hidden_start + rows if hidden_end is None else hidden_end
    return {
        "room": room,
        "prefill_rank": 0,
        "hidden_start": hidden_start,
        "row_len": rows,
        "is_last_hidden_chunk": hidden_start + rows == hidden_end,
        "session_id": f"session-{room}",
        "reply_host": "127.0.0.1",
        "reply_port": 12345,
    }


def _decode_req(room: int, hidden_end: int):
    return SimpleNamespace(
        req=SimpleNamespace(rid=f"rid-{room}", bootstrap_room=room),
        pd_hidden_dynamic_allocation=True,
        pd_hidden_dynamic_allocations={},
        pd_hidden_state=PDHiddenRequestState.streaming_state(0, hidden_end),
    )


class _FakeKVManager:
    def __init__(self):
        self.request_status = {}
        self.alloc_requests = deque()
        self.grants = []
        self.acked_chunks = defaultdict(list)
        self.failures = []

    def pop_pd_hidden_alloc_requests(self):
        requests = list(self.alloc_requests)
        self.alloc_requests.clear()
        return requests

    def requeue_pd_hidden_alloc_requests(self, requests):
        self.alloc_requests.extendleft(reversed(requests))

    def grant_pd_hidden_rows(self, request, dst_indices):
        self.grants.append((dict(request), list(dst_indices)))

    def pop_pd_hidden_acked_chunks(self, room):
        return self.acked_chunks.pop(room, [])

    def record_failure(self, room, reason):
        self.failures.append((room, reason))

    def update_status(self, room, status):
        self.request_status[room] = status


class TestPDHiddenAllocationEvents(CustomTestCase):
    def test_mooncake_control_send_supports_legacy_common_manager(self):
        socket = MagicMock()
        manager = SimpleNamespace(
            _socket_lock=threading.Lock(),
            _connect=MagicMock(return_value=socket),
        )

        MooncakeKVManager._send_multipart(
            manager,
            "tcp://127.0.0.1:12345",
            [b"header", b"payload"],
        )

        manager._connect.assert_called_once_with(
            "tcp://127.0.0.1:12345", is_ipv6=False
        )
        socket.send_multipart.assert_called_once_with([b"header", b"payload"])
        self.assertIn("tcp://127.0.0.1:12345", manager._socket_send_locks)

    def test_allocation_requests_preserve_prefill_completion_order(self):
        events = PDHiddenEventManager(MagicMock())
        first = _alloc_request(1, 4)
        second = _alloc_request(2, 2)
        third = _alloc_request(3, 1)

        events.append_alloc_request(first)
        events.append_alloc_request(second)
        self.assertEqual(events.pop_alloc_requests(), [first, second])

        events.append_alloc_request(third)
        events.requeue_alloc_requests_front([first, second])
        self.assertEqual(events.pop_alloc_requests(), [first, second, third])

    def test_chunk_wakes_only_after_every_decode_session_grants_rows(self):
        owner = MagicMock()
        events = PDHiddenEventManager(owner)
        transfer_queue = FastQueue()
        chunk = SimpleNamespace(room=9, pd_hidden_start=32)

        grants = events.take_alloc_grants_or_park(
            transfer_queue=transfer_queue,
            kv_chunk=chunk,
            prefill_rank=3,
            expected_session_ids={"decode-a", "decode-b"},
        )
        self.assertIsNone(grants)

        events.handle_alloc_grant(
            room=9,
            prefill_rank=3,
            hidden_start=32,
            session_id="decode-a",
            dst_indices=[4, 5],
        )
        self.assertEqual(len(transfer_queue._buf), 0)

        events.handle_alloc_grant(
            room=9,
            prefill_rank=3,
            hidden_start=32,
            session_id="decode-b",
            dst_indices=[8, 9],
        )
        self.assertIs(transfer_queue.get(), chunk)
        self.assertEqual(
            events.take_alloc_grants_or_park(
                transfer_queue=transfer_queue,
                kv_chunk=chunk,
                prefill_rank=3,
                expected_session_ids={"decode-a", "decode-b"},
            ),
            {"decode-a": [4, 5], "decode-b": [8, 9]},
        )


class TestPDHiddenDynamicBootstrap(CustomTestCase):
    def test_dynamic_bootstrap_accepts_layout_without_reserved_dst_rows(self):
        pool = PDHiddenRowPool(4, hidden_size=4, dtype=torch.float32)
        req = SimpleNamespace(rid="dynamic-bootstrap", origin_input_ids=list(range(8)))
        metadata = {
            "hidden_start": 0,
            "hidden_len": 8,
            "streaming_hidden": True,
            "dynamic_hidden_allocation": True,
            "streaming_window_rows": 4,
            "target_layer_ids": [3, 5],
            "pp_slices": {
                "0": {
                    "layer_ids": [3, 5],
                    "slice_len": 4,
                    "dst_indices": [],
                }
            },
        }
        model_config = SimpleNamespace(hidden_size=2)
        model_runner = SimpleNamespace(
            spec_aux_config=SimpleNamespace(dflash_target_layer_ids=[3, 5])
        )

        plan, error = resolve_hidden_bootstrap_plan(
            req=req,
            metadata=metadata,
            decode_prefix_len=0,
            pp_rank=0,
            model_config=model_config,
            model_runner=model_runner,
            metadata_buffers=SimpleNamespace(pd_hidden_pool=pool),
        )

        self.assertIsNone(error)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.dst_indices, [])
        self.assertEqual(plan.source_window_rows, 4)
        self.assertIs(plan.pool, pool)


class TestDecodePDHiddenDynamicAllocator(CustomTestCase):
    def _make_queue(self, pool_rows: int, decode_reqs):
        manager = _FakeKVManager()
        manager.request_status.update(
            {
                req.req.bootstrap_room: KVPoll.Transferring
                for req in decode_reqs
            }
        )
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = list(decode_reqs)
        queue.kv_manager = manager
        queue.metadata_buffers = SimpleNamespace(
            pd_hidden_pool=PDHiddenRowPool(
                pool_rows, hidden_size=1, dtype=torch.float32
            )
        )
        queue._last_pd_hidden_dynamic_credit_warning_time = 0.0
        return queue, manager

    def test_rows_are_allocated_only_after_prefill_requests_them(self):
        decode_req = _decode_req(room=1, hidden_end=4)
        queue, manager = self._make_queue(4, [decode_req])
        pool = queue.metadata_buffers.pd_hidden_pool

        self.assertEqual(pool.available_size(), 4)
        manager.alloc_requests.append(_alloc_request(1, 4))
        queue._drain_pd_hidden_alloc_requests()

        self.assertEqual(pool.available_size(), 0)
        self.assertEqual(len(manager.grants), 1)
        self.assertEqual(manager.grants[0][1], [0, 1, 2, 3])

    def test_live_room_accepts_hidden_request_after_kv_success(self):
        decode_req = _decode_req(room=1, hidden_end=8)
        queue, manager = self._make_queue(4, [decode_req])
        manager.request_status[1] = KVPoll.Success
        manager.alloc_requests.append(
            _alloc_request(1, 4, hidden_start=4, hidden_end=8)
        )

        queue._drain_pd_hidden_alloc_requests()

        self.assertEqual(len(manager.grants), 1)
        self.assertEqual(manager.grants[0][0]["hidden_start"], 4)
        self.assertEqual(manager.grants[0][1], [0, 1, 2, 3])
        self.assertEqual(
            decode_req.pd_hidden_dynamic_allocations[(0, 4, "session-1")][
                "dst_indices"
            ],
            [0, 1, 2, 3],
        )

    def test_oldest_completed_prefill_chunk_cannot_be_bypassed(self):
        first_req = _decode_req(room=1, hidden_end=4)
        second_req = _decode_req(room=2, hidden_end=1)
        queue, manager = self._make_queue(4, [first_req, second_req])

        first = _alloc_request(1, 4)
        second = _alloc_request(2, 1)
        manager.alloc_requests.extend([first, second])
        queue._drain_pd_hidden_alloc_requests()

        self.assertEqual([grant[0]["room"] for grant in manager.grants], [1])
        self.assertEqual(list(manager.alloc_requests), [second])

    def test_ack_reclaims_rows_for_the_next_completed_chunk(self):
        first_req = _decode_req(room=1, hidden_end=8)
        second_req = _decode_req(room=2, hidden_end=4)
        queue, manager = self._make_queue(4, [first_req, second_req])
        pool = queue.metadata_buffers.pd_hidden_pool

        manager.alloc_requests.extend(
            [
                _alloc_request(1, 4, hidden_end=8),
                _alloc_request(2, 4),
            ]
        )
        queue._drain_pd_hidden_alloc_requests()
        self.assertEqual([grant[0]["room"] for grant in manager.grants], [1])

        manager.acked_chunks[1].append(
            {
                "prefill_rank": 0,
                "hidden_start": 0,
                "is_last_hidden_chunk": False,
                "release_indices": [0, 1, 2, 3],
            }
        )
        queue._consume_pd_hidden_acked_chunks(first_req)
        self.assertEqual(pool.available_size(), 4)

        queue._drain_pd_hidden_alloc_requests()
        self.assertEqual(
            [grant[0]["room"] for grant in manager.grants], [1, 2]
        )
        self.assertEqual(pool.available_size(), 0)

    def test_late_request_for_terminal_room_is_dropped(self):
        decode_req = _decode_req(room=1, hidden_end=2)
        queue, manager = self._make_queue(2, [decode_req])
        manager.request_status[1] = KVPoll.Failed
        manager.alloc_requests.append(_alloc_request(1, 2))

        queue._drain_pd_hidden_alloc_requests()

        self.assertEqual(queue.metadata_buffers.pd_hidden_pool.available_size(), 2)
        self.assertEqual(manager.grants, [])
        self.assertEqual(list(manager.alloc_requests), [])


if __name__ == "__main__":
    unittest.main()
