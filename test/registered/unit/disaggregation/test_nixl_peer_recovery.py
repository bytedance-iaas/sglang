"""Recover invalidated peers without replaying a failed room or losing writes."""

import threading
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.nixl.conn import (
    _NIXL_TRANSPORT_ERRORS,
    NixlKVManager,
    _NixlPostError,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MODULE = "sglang.srt.disaggregation.nixl.conn"


class FakeAgent:
    def __init__(self):
        self.valid = set()
        self.events = []
        self.states = {}
        self.fail_load = False
        self.fail_post = False
        self.unreadable = False

    def check_remote_metadata(self, peer):
        return peer in self.valid

    def add_remote_agent(self, metadata):
        self.events.append("load")
        if self.fail_load:
            raise RuntimeError("unreachable")
        self.valid.add(metadata.decode())
        return metadata.decode()

    def remove_remote_agent(self, peer):
        self.events.append("remove")
        self.valid.discard(peer)

    def transfer(self, handle):
        self.events.append("post")
        if self.fail_post:
            self.fail_post = False
            self.valid.discard("peer")
            self.states[handle] = "ERR"
            raise RuntimeError("remote disconnected after posting")
        self.states[handle] = "DONE"
        return "DONE"

    def check_xfer_state(self, handle):
        self.events.append("check")
        if self.unreadable:
            raise RuntimeError("cannot read status")
        return self.states[handle]

    def release_xfer_handle(self, handle):
        self.events.append("release")
        assert self.states[handle] in ("DONE", "ERR")


def manager():
    mgr = NixlKVManager.__new__(NixlKVManager)
    mgr.agent = FakeAgent()
    mgr._transfer_handle_owner = threading.local()
    mgr._peer_reload_lock = threading.Lock()
    mgr._peer_reload_times = {}
    mgr._unsettled_peers = set()
    mgr.prep_handles = {"": "source", "peer": "old-destination"}
    mgr.prep_handles_slice_dst = {"peer": "old-slice"}
    mgr.decode_kv_args_table = {
        "peer": SimpleNamespace(agent_metadata=b"peer", kv_xfer_segments=["old"])
    }

    def prepare(info):
        assert "peer" not in mgr.prep_handles
        assert "peer" not in mgr.prep_handles_slice_dst
        assert info.kv_xfer_segments is None
        mgr.agent.events.append("prepare")
        mgr.prep_handles["peer"] = "new-destination"

    mgr._prepare_payload_xfer = MagicMock(side_effect=prepare)
    return mgr


class TestPeerRecovery(CustomTestCase):
    def test_invalidated_peer_rebuilds_only_destination_cache(self):
        mgr = manager()
        mgr._reload_invalidated_peers({"peer": object()})
        self.assertEqual(mgr.agent.events, ["load", "prepare"])
        self.assertEqual(mgr.prep_handles, {"": "source", "peer": "new-destination"})
        self.assertEqual(mgr.prep_handles_slice_dst, {})
        self.assertTrue(mgr.agent.check_remote_metadata("peer"))

    def test_valid_peer_keeps_prepared_handles(self):
        mgr = manager()
        mgr.agent.valid.add("peer")
        mgr._reload_invalidated_peers({"peer": object()})
        self.assertEqual(mgr.agent.events, [])
        self.assertEqual(mgr.prep_handles["peer"], "old-destination")

    def test_rebuild_failure_rolls_back_new_handles_and_validity(self):
        mgr = manager()

        def fail(info):
            mgr.prep_handles["peer"] = "half-built"
            raise RuntimeError("descriptor preparation failed")

        mgr._prepare_payload_xfer.side_effect = fail
        mgr._reload_invalidated_peers({"peer": object()})
        self.assertNotIn("peer", mgr.agent.valid)
        self.assertNotIn("peer", mgr.prep_handles)
        self.assertEqual(mgr.prep_handles_slice_dst, {})
        self.assertEqual(mgr.agent.events, ["load", "remove"])

    def test_failed_import_is_rate_limited_then_retried(self):
        mgr = manager()
        mgr.agent.fail_load = True
        with patch(MODULE + ".time.monotonic", return_value=10.0):
            mgr._reload_invalidated_peers({"peer": object()})
            mgr._reload_invalidated_peers({"peer": object()})
        self.assertEqual(mgr.agent.events.count("load"), 1)
        mgr.agent.fail_load = False
        with patch(MODULE + ".time.monotonic", return_value=12.0):
            mgr._reload_invalidated_peers({"peer": object()})
        self.assertEqual(mgr.agent.events.count("load"), 2)
        self.assertIn("peer", mgr.agent.valid)

    def test_concurrent_reload_imports_and_prepares_once(self):
        mgr = manager()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(
                    lambda _: mgr._reload_invalidated_peers({"peer": object()}),
                    range(16),
                )
            )
        self.assertEqual(mgr.agent.events, ["load", "prepare"])

    def test_unsettled_or_unknown_peer_is_not_reloaded(self):
        mgr = manager()
        mgr._unsettled_peers.add("peer")
        mgr._reload_invalidated_peers({"peer": object(), "unknown": object()})
        self.assertEqual(mgr.agent.events, [])


class TestTransferOwnership(CustomTestCase):
    def test_worker_owns_handle_before_post_raises(self):
        mgr = manager()
        handles = []
        mgr._transfer_handle_owner.handles = handles
        h = object()
        mgr.agent.fail_post = True
        with self.assertRaises(_NixlPostError):
            mgr._post_xfer(h, "failed")
        self.assertEqual(handles, [h])

    def test_nested_partial_state_keeps_every_posted_handle(self):
        mgr = manager()
        mgr._transfer_handle_owner.handles = []
        first, second = object(), object()
        mgr._post_xfer(first, "first")
        mgr.agent.fail_post = True
        with self.assertRaises(_NixlPostError):
            mgr._post_xfer(second, "second")
        self.assertEqual(mgr._transfer_handle_owner.handles, [first, second])
        self.assertEqual(
            mgr._await_handles([first, second], failure_seen=True), (True, True)
        )

    def test_ownership_is_thread_local(self):
        mgr = manager()
        barrier = threading.Barrier(2)

        def post(_):
            own = []
            mgr._transfer_handle_owner.handles = own
            h = object()
            barrier.wait()
            mgr._post_xfer(h, "failed")
            return own, h

        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(post, range(2)))
        self.assertTrue(all(own == [h] for own, h in rows))

    def test_failed_handle_does_not_skip_running_sibling(self):
        mgr = manager()
        failed, sibling = object(), object()
        states = iter(["ERR", "PROC", "ERR", "DONE"])
        mgr.agent.check_xfer_state = MagicMock(side_effect=lambda _: next(states))
        self.assertEqual(
            mgr._await_handles([failed, sibling], failure_seen=False), (True, True)
        )
        self.assertEqual(mgr.agent.check_xfer_state.call_count, 4)

    def test_duplicate_handle_is_polled_once(self):
        mgr = manager()
        h = object()
        mgr.agent.states[h] = "DONE"
        self.assertEqual(mgr._await_handles([h, h], failure_seen=False), (True, False))
        self.assertEqual(mgr.agent.events, ["check"])

    def test_unreadable_status_never_proves_drained(self):
        mgr = manager()
        mgr.agent.unreadable = True
        with patch(MODULE + "._ERR_SETTLE_TIMEOUT_S", 0.002), patch(
            MODULE + "._ERR_SETTLE_POLL_S", 0
        ):
            self.assertEqual(
                mgr._await_handles([object()], failure_seen=True), (False, True)
            )

    def test_first_native_poll_error_retained_once_per_handle_across_drains(self):
        mgr = manager()
        owner = mgr._transfer_handle_owner
        owner.handles = []
        owner.payload_tags = {}
        owner.poll_errors = {}
        owner.first_poll_error = None
        owner.room, owner.worker_index, owner.chunk_id = 123, 1, 96
        state, aux = object(), object()
        mgr._post_xfer(state, "state post", payload_tag="123_state_0_0")
        mgr._post_xfer(aux, "aux post", payload_tag="123_aux")
        first = _NIXL_TRANSPORT_ERRORS[0]("NIXL_ERR_REMOTE_DISCONNECT")
        second = RuntimeError("unreadable sibling")

        def poll(handle):
            raise first if handle is state else second

        mgr.agent.check_xfer_state = poll
        with patch(MODULE + ".logger") as log, patch(
            MODULE + "._ERR_SETTLE_TIMEOUT_S", 0
        ):
            self.assertEqual(
                mgr._await_handles([state, state, aux], failure_seen=False),
                (False, True),
            )
            self.assertEqual(
                mgr._await_handles([state, aux], failure_seen=True), (False, True)
            )
        self.assertIs(owner.first_poll_error, first)
        self.assertEqual(owner.poll_errors, {id(state): first, id(aux): second})
        self.assertEqual(log.exception.call_count, 2)
        self.assertEqual(log.exception.call_args_list[0].args[1:4], (123, 1, 96))
        self.assertEqual(log.exception.call_args_list[0].args[5], "123_state_0_0")
        self.assertNotIn("release", mgr.agent.events)


class StopWorker(BaseException):
    pass


class TestRecoveryThroughWorker(CustomTestCase):
    def worker_manager(self):
        mgr = manager()
        mgr.enable_staging = False
        mgr.transfer_source_rank = 0
        mgr.enable_deferred_decode_kv_release = True
        mgr._staging_outstanding = defaultdict(int)
        mgr._deferred_ack_targets = {1: ("decode", 1234)}
        mgr._deferred_ack_poisoned_rooms = set()
        mgr._send_abort_ack = MagicMock()
        mgr.request_status = {1: KVPoll.WaitingForInput, 2: KVPoll.WaitingForInput}
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr.exceptions = {}
        mgr.req_to_decode_prefix_len = {}
        mgr.kv_args = SimpleNamespace(kv_data_ptrs=[])
        mgr.decode_kv_args_table["peer"].decode_tp_size = 1
        mgr.decode_kv_args_table["peer"].dst_aux_ptrs = []
        mgr.transfer_infos = {
            room: {
                "peer": SimpleNamespace(
                    room=room,
                    agent_name="peer",
                    dst_aux_index=room,
                    is_dummy=lambda: False,
                )
            }
            for room in [1, 2]
        }

        def send_aux(*args):
            h = object()
            mgr._post_xfer(h, "aux post failed")
            return h

        mgr.send_aux = MagicMock(side_effect=send_aux)
        chunks = [
            TransferKVChunk(
                room=room,
                prefill_kv_indices=np.array([], dtype=np.int32),
                index_slice=slice(0, 0),
                chunk_id=0,
                is_last_chunk=True,
                state_indices=[],
                prefill_aux_index=room,
            )
            for room in [1, 2]
        ]
        q = MagicMock()
        q.get.side_effect = chunks + [StopWorker()]
        return mgr, q

    def test_failed_room_stays_failed_and_next_room_uses_reloaded_peer(self):
        mgr, q = self.worker_manager()
        mgr.agent.fail_post = True
        with self.assertRaises(StopWorker):
            mgr.transfer_worker(q)
        self.assertEqual(mgr.request_status[1], KVPoll.Failed)
        self.assertEqual(mgr.request_status[2], KVPoll.Success)
        self.assertEqual(mgr.send_aux.call_count, 2)
        events = mgr.agent.events
        self.assertLess(events.index("release"), events.index("load"))
        self.assertLess(events.index("prepare"), len(events) - 1)
        self.assertEqual(events.count("load"), 1)
        mgr._send_abort_ack.assert_called_once_with("decode", 1234, 1)
        self.assertFalse(hasattr(mgr._transfer_handle_owner, "handles"))

    def test_unsettled_failure_blocks_peer_and_never_acks_old_room(self):
        mgr, q = self.worker_manager()
        mgr.agent.fail_post = True
        mgr.agent.unreadable = True
        with patch(MODULE + "._ERR_SETTLE_TIMEOUT_S", 0.002), patch(
            MODULE + "._ERR_SETTLE_POLL_S", 0
        ):
            with self.assertRaises(StopWorker):
                mgr.transfer_worker(q)
        self.assertEqual(mgr.send_aux.call_count, 1)
        self.assertEqual(mgr._unsettled_peers, {"peer"})
        self.assertEqual(mgr.agent.events.count("load"), 0)
        mgr._send_abort_ack.assert_not_called()

    def test_poll_timeout_preserves_native_cause_and_ack_poison(self):
        mgr, q = self.worker_manager()
        native_error = _NIXL_TRANSPORT_ERRORS[0]("NIXL_ERR_REMOTE_DISCONNECT")
        mgr.agent.check_xfer_state = MagicMock(side_effect=native_error)
        with patch(MODULE + "._ERR_SETTLE_TIMEOUT_S", 0), patch(
            MODULE + "._ERR_SETTLE_POLL_S", 0
        ):
            with self.assertRaises(StopWorker):
                mgr.transfer_worker(q)
        self.assertIs(mgr.exceptions[1].__cause__, native_error)
        self.assertEqual(mgr._unsettled_peers, {"peer"})
        self.assertIn(1, mgr._deferred_ack_poisoned_rooms)
        mgr._send_abort_ack.assert_not_called()
        self.assertNotIn("release", mgr.agent.events)
        self.assertFalse(hasattr(mgr._transfer_handle_owner, "poll_errors"))

    def test_release_failure_prevents_peer_reload_and_drain_ack(self):
        mgr, q = self.worker_manager()
        mgr.agent.fail_post = True
        mgr.agent.release_xfer_handle = MagicMock(
            side_effect=RuntimeError("not released")
        )
        with self.assertRaises(StopWorker):
            mgr.transfer_worker(q)
        self.assertEqual(mgr.send_aux.call_count, 1)
        self.assertEqual(mgr._unsettled_peers, {"peer"})
        self.assertNotIn("load", mgr.agent.events)
        mgr._send_abort_ack.assert_not_called()

    def test_late_abort_of_cleared_poisoned_room_never_acks(self):
        mgr, _ = self.worker_manager()
        mgr.poison_deferred_ack_room(1)
        mgr.request_status.pop(1)
        self.assertTrue(
            mgr._handle_abort_notification([b"ABORT", b"1", b"127.0.0.1", b"1234"])
        )
        mgr._send_abort_ack.assert_not_called()
        mgr.register_deferred_ack_target(1, "decode", 1234)
        mgr._staging_outstanding.pop(1, None)
        mgr._maybe_ack_drained_abort(1)
        mgr._send_abort_ack.assert_not_called()


class TestFailedRoomState(CustomTestCase):
    def test_new_room_lifecycle_clears_poison(self):
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr.request_status = {}
        mgr._deferred_ack_poisoned_rooms = {1}
        mgr.update_status(1, KVPoll.Bootstrapping)
        self.assertNotIn(1, mgr._deferred_ack_poisoned_rooms)


if __name__ == "__main__":
    unittest.main()
