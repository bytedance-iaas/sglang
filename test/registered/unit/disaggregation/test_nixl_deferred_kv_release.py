"""Deferred decode-side KV release on the NIXL backend.

When a decode request is aborted while its prefill->decode transfer may still be
in flight, the decode holds its KV pages until every prefill rank acks that its
transfer drained. NIXL transfers are asynchronous (agent.transfer() posts, the
worker polls check_xfer_state), so the ack must come from the transfer worker
after its DONE barrier -- never from the bootstrap thread for an active room.
"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.nixl.conn import (
    NixlKVManager,
    NixlKVSender,
    NixlTransferKVChunk,
    TransferInfo,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _prefill_mgr(cls=CommonKVManager, enabled=True):
    """Bare manager carrying only the prefill-side deferred-ack state."""
    mgr = cls.__new__(cls)
    mgr.enable_deferred_decode_kv_release = enabled
    mgr._deferred_ack_targets = {}
    mgr._staging_outstanding = {}
    mgr._failed_transfer_handles = {}
    mgr._generation_ack_targets = {}
    mgr._abort_generations = {}
    mgr._source_lease_lock = threading.RLock()
    mgr.transfer_infos = {}
    mgr.req_to_decode_prefix_len = {}
    mgr.request_status = {}
    mgr.enable_staging = False
    mgr._staging_ctx = None
    mgr._sent = []
    # Capture acks instead of opening a socket.
    mgr._send_abort_ack = lambda ip, port, room: mgr._sent.append((ip, port, room))
    return mgr


class TestDeferredAckTargets(CustomTestCase):
    def test_ack_held_until_outstanding_drains(self):
        mgr = _prefill_mgr()
        mgr.register_deferred_ack_target(7, "10.0.0.1", 5000)

        mgr._staging_outstanding[7] = 1
        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(mgr._sent, [])  # still writing -> no ack

        mgr._staging_outstanding[7] = 0
        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(mgr._sent, [("10.0.0.1", 5000, 7)])

    def test_ack_fires_at_most_once(self):
        mgr = _prefill_mgr()
        mgr.register_deferred_ack_target(8, "10.0.0.2", 5001)
        mgr._maybe_ack_drained_abort(8)
        mgr._maybe_ack_drained_abort(8)
        self.assertEqual(len(mgr._sent), 1)
        self.assertNotIn(8, mgr._deferred_ack_targets)

    def test_unregistered_room_is_noop(self):
        mgr = _prefill_mgr()
        mgr._maybe_ack_drained_abort(999)
        self.assertEqual(mgr._sent, [])

    def test_sender_clear_preserves_target_until_last_chunk_drains(self):
        mgr = _prefill_mgr()
        sender = NixlKVSender.__new__(NixlKVSender)
        sender.kv_mgr, sender.bootstrap_room = mgr, 7
        mgr.request_status[7] = KVPoll.Failed
        mgr._staging_outstanding[7] = 2
        mgr.register_deferred_ack_target(7, "10.0.0.1", 5000)

        NixlKVSender.clear(sender)
        self.assertNotIn(7, mgr.request_status)
        self.assertEqual(mgr._deferred_ack_targets[7], ("10.0.0.1", 5000))
        mgr._staging_outstanding[7] = 1
        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(mgr._sent, [])
        mgr._staging_outstanding[7] = 0
        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(mgr._sent, [("10.0.0.1", 5000, 7)])
        NixlKVSender.clear(sender)
        self.assertEqual(len(mgr._sent), 1)

    def test_sender_clear_acks_quiescent_target_once(self):
        mgr = _prefill_mgr()
        sender = NixlKVSender.__new__(NixlKVSender)
        sender.kv_mgr, sender.bootstrap_room = mgr, 8
        mgr.register_deferred_ack_target(8, "10.0.0.2", 5001)

        NixlKVSender.clear(sender)
        NixlKVSender.clear(sender)

        self.assertEqual(mgr._sent, [("10.0.0.2", 5001, 8)])
        self.assertEqual(mgr._deferred_ack_targets, {})

    def test_prefill_unique_rank_matches_success_sync_formula(self):
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr.attn_tp_rank, mgr.pp_size, mgr.attn_cp_size = 2, 3, 4
        mgr.pp_rank, mgr.attn_cp_rank = 1, 3
        self.assertEqual(mgr._prefill_unique_rank(), 2 * (3 * 4) + 1 * 4 + 3)

    def test_old_sender_clear_cannot_remove_new_active_attempt(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr._active_transfer_epochs = {7: 2}
        new_info = TransferInfo(
            7,
            "10.0.0.2",
            6001,
            "new",
            np.array([2], dtype=np.int32),
            2,
            1,
            [],
            transfer_epoch=2,
        )
        mgr._transfer_infos_by_attempt = {(7, 2): {"new": new_info}}
        mgr.transfer_infos = {7: mgr._transfer_infos_by_attempt[(7, 2)]}
        mgr.request_status[7] = KVPoll.WaitingForInput
        sender = NixlKVSender.__new__(NixlKVSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = 7
        sender.transfer_epoch = 1

        sender.clear()

        self.assertEqual(mgr._active_transfer_epochs[7], 2)
        self.assertIs(mgr.transfer_infos[7]["new"], new_info)
        self.assertIn(7, mgr.request_status)

    def test_sender_clear_preserves_attempt_marker_until_decode_release(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr._active_transfer_epochs = {7: 1}
        mgr._transfer_infos_by_attempt = {(7, 1): {}}
        mgr._aborted_transfer_attempts = {(7, 1): {"101"}}
        mgr.request_status[7] = KVPoll.Failed
        sender = NixlKVSender.__new__(NixlKVSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = 7
        sender.transfer_epoch = 1

        sender.clear()

        self.assertEqual(mgr._aborted_transfer_attempts[(7, 1)], {"101"})
        self.assertNotIn(7, mgr._active_transfer_epochs)


class TestNixlEnqueuedDestinationBinding(CustomTestCase):
    def test_enqueue_snapshot_survives_guard_replacement_and_clear(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr.disaggregation_mode = DisaggregationMode.PREFILL
        mgr.request_status[7] = KVPoll.WaitingForInput
        mgr._staging_outstanding[7] = 0
        old = TransferInfo(
            7, "10.0.0.1", 5000, "old", np.array([1], dtype=np.int32), 1, 1, []
        )
        mgr.transfer_infos = {7: {"old": old}}
        mgr.transfer_queues = [MagicMock(), MagicMock()]
        mgr.add_transfer_request(
            7, np.array([2], dtype=np.int32), slice(0, 1), False, 0
        )
        chunk = mgr.transfer_queues[0].put.call_args.args[0]
        self.assertIsInstance(chunk, NixlTransferKVChunk)
        self.assertEqual(chunk.transfer_infos, (old,))
        self.assertTrue(chunk.staging_counted)
        self.assertEqual(mgr._staging_outstanding[7], 1)
        # Late GUARD replaces the map object, not the chunk's destinations.
        replacement = TransferInfo(
            7, "10.0.0.2", 5001, "new", np.array([3], dtype=np.int32), 3, 1, []
        )
        mgr.transfer_infos[7] = {"new": replacement}
        self.assertIs(chunk.transfer_infos[0], old)
        mgr.transfer_infos.pop(7)
        self.assertIs(chunk.transfer_infos[0], old)
        # Preserve the same live staging object; copying it would lose replies.
        old.staging = SimpleNamespace(offsets={0: (16, 0)})
        self.assertIs(chunk.transfer_infos[0].staging, old.staging)
        mgr._finish_counted_chunk(chunk)
        self.assertFalse(mgr.has_pending_source_reads(7))

    def test_enqueue_failure_releases_only_its_chunk_contribution(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr.disaggregation_mode = DisaggregationMode.PREFILL
        mgr.request_status[7] = KVPoll.WaitingForInput
        info = TransferInfo(
            7, "10.0.0.1", 5000, "old", np.array([1], dtype=np.int32), 1, 1, []
        )
        mgr.transfer_infos = {7: {"old": info}}
        queue = MagicMock()
        queue.put.side_effect = RuntimeError("enqueue failed")
        mgr.transfer_queues = [queue]
        mgr._staging_outstanding[7] = 1
        with self.assertRaisesRegex(RuntimeError, "enqueue failed"):
            mgr.add_transfer_request(
                7, np.array([2], dtype=np.int32), slice(0, 1), False, 0
            )
        self.assertEqual(mgr._staging_outstanding[7], 1)
        self.assertFalse(queue.put.call_args.args[0].staging_counted)


class TestNixlAbortNotification(CustomTestCase):
    """_handle_abort_notification is the prefill bootstrap-thread entry point."""

    @staticmethod
    def _abort_msg(room=11, ip="10.0.0.3", port=6000):
        return [
            b"ABORT",
            str(room).encode("ascii"),
            ip.encode("ascii"),
            str(port).encode("ascii"),
        ]

    def _mgr(self, enabled=True, room=11, status=KVPoll.WaitingForInput):
        mgr = _prefill_mgr(NixlKVManager, enabled=enabled)
        if status is not None:
            mgr.request_status[room] = status
        mgr.record_failure = MagicMock()
        mgr.update_status = MagicMock(
            side_effect=lambda r, s: mgr.request_status.__setitem__(r, s)
        )
        mgr.check_status = lambda r: mgr.request_status[r]
        return mgr

    def test_in_flight_room_registers_target_and_does_not_ack_yet(self):
        # A counted chunk holds the ack: only the worker knows when it landed.
        mgr = self._mgr()
        mgr._staging_outstanding[11] = 1
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._deferred_ack_targets[11], ("10.0.0.3", 6000))
        self.assertEqual(mgr._sent, [])
        # Marked Failed first, so no new chunk can be enqueued for the room.
        self.assertEqual(mgr.request_status[11], KVPoll.Failed)

    def test_quiescent_active_room_acks_without_waiting_for_a_worker_visit(self):
        # Window 2: chunks already drained with none left to come, so the worker
        # never revisits the room -- acking here keeps it off the timeout path.
        mgr = self._mgr()
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._sent, [("10.0.0.3", 6000, 11)])
        self.assertEqual(mgr._deferred_ack_targets, {})

    def test_worker_skip_before_registration_still_acks(self):
        # Window 1: the worker can pass its skip point between the Failed flip
        # and registration; the ack attempt at registration covers that.
        mgr = self._mgr()
        mgr._staging_outstanding[11] = 1

        real_update = mgr.update_status.side_effect

        def failed_then_worker_skips(room, status):
            real_update(room, status)
            # Worker dequeues, sees Failed, uncounts, and finds no target yet.
            mgr._staging_outstanding.pop(room, None)
            mgr._maybe_ack_drained_abort(room)

        mgr.update_status = MagicMock(side_effect=failed_then_worker_skips)
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._sent, [("10.0.0.3", 6000, 11)])
        self.assertEqual(mgr._deferred_ack_targets, {})

    def test_concluded_room_acks_immediately(self):
        # Concluded and quiescent: ack straight away.
        mgr = self._mgr(status=None)
        mgr.check_status = lambda r: KVPoll.Success
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._sent, [("10.0.0.3", 6000, 11)])
        self.assertEqual(mgr._deferred_ack_targets, {})

    def test_cleared_room_with_outstanding_chunk_retains_late_ack_until_drain(self):
        # The ERR path abandons sibling handles that may still be writing and
        # leaves the chunk counted; clear() then drops the room. Acking on
        # "unknown room" alone would release decode pages under those writes.
        mgr = self._mgr(status=None)  # room absent == cleared/unknown
        mgr._staging_outstanding[11] = 1
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._sent, [])
        self.assertEqual(mgr._deferred_ack_targets[11], ("10.0.0.3", 6000))
        self.assertNotIn(11, mgr.request_status)
        mgr._staging_outstanding.pop(11)
        mgr._maybe_ack_drained_abort(11)
        mgr._maybe_ack_drained_abort(11)
        self.assertEqual(mgr._sent, [("10.0.0.3", 6000, 11)])
        self.assertEqual(mgr._deferred_ack_targets, {})

    def test_success_with_pending_sibling_is_failed_before_late_ack(self):
        mgr = self._mgr(status=KVPoll.Success)
        mgr._staging_outstanding[11] = 1
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))
        self.assertEqual(mgr.request_status[11], KVPoll.Failed)
        self.assertEqual(mgr._sent, [])
        self.assertEqual(mgr._deferred_ack_targets[11], ("10.0.0.3", 6000))
        mgr._staging_outstanding.pop(11)
        mgr._maybe_ack_drained_abort(11)
        self.assertEqual(mgr._sent, [("10.0.0.3", 6000, 11)])

    def test_cleared_room_retained_handle_blocks_late_ack_even_without_count(self):
        mgr = self._mgr(status=None)
        mgr._failed_transfer_handles[11] = [object()]
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))
        mgr._maybe_ack_drained_abort(11)
        self.assertEqual(mgr._sent, [])
        self.assertEqual(mgr._deferred_ack_targets[11], ("10.0.0.3", 6000))

    def test_feature_off_registers_nothing_and_acks_nothing(self):
        mgr = self._mgr(enabled=False)
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._deferred_ack_targets, {})
        self.assertEqual(mgr._sent, [])
        # Legacy behavior preserved: the room is still failed.
        self.assertEqual(mgr.request_status[11], KVPoll.Failed)

    def test_legacy_two_frame_abort_is_tolerated(self):
        # Older peers send [ABORT, room] with no return address.
        mgr = self._mgr()
        self.assertTrue(mgr._handle_abort_notification([b"ABORT", b"11"]))
        self.assertEqual(mgr._deferred_ack_targets, {})
        self.assertEqual(mgr._sent, [])

    def test_non_abort_message_is_not_claimed(self):
        mgr = self._mgr()
        self.assertFalse(mgr._handle_abort_notification([b"STAGING_REQ", b"11"]))


class TestNixlDecodeAckIngest(CustomTestCase):
    def test_abort_ack_is_aggregated_per_rank(self):
        # Mirrors the decode listener thread's ABORT_ACK branch.
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr._deferred_abort_ack_tracker = {}
        mgr.register_deferred_abort_room(21)

        for rank in (b"0", b"1", b"1"):
            msg = [b"ABORT_ACK", b"21", rank]
            mgr.note_abort_ack(int(msg[1].decode()), int(msg[2].decode()))

        self.assertFalse(mgr.is_abort_release_safe(21, required_acks=3))
        self.assertTrue(mgr.is_abort_release_safe(21, required_acks=2))

    def test_old_or_legacy_ack_cannot_release_new_room_generation(self):
        mgr = _prefill_mgr(NixlKVManager)
        old = mgr.register_deferred_abort_room(21, expected_ranks={0})
        mgr.note_abort_ack(21, 0, old)
        self.assertTrue(mgr.is_abort_release_safe(21, 1, old))
        mgr.clear_deferred_abort_state(21, old)
        new = mgr.register_deferred_abort_room(21, expected_ranks={1})
        self.assertNotEqual(old, new)
        for msg in (
            [b"ABORT_ACK", b"21", b"1", old.encode()],
            [b"ABORT_ACK", b"21", b"1"],
            [b"ABORT_ACK", b"21", b"bad", new.encode()],
        ):
            self.assertTrue(mgr.handle_abort_ack_message(msg))
        self.assertFalse(mgr.is_abort_release_safe(21, 1, new))
        mgr.clear_deferred_abort_state(21, old)
        self.assertIn((21, 0), mgr._abort_generations)
        mgr.handle_abort_ack_message([b"ABORT_ACK", b"21", b"1", new.encode()])
        self.assertTrue(mgr.is_abort_release_safe(21, 1, new))
        self.assertFalse(mgr.is_abort_release_safe(21, 1, old))
        self.assertFalse(mgr.is_abort_release_safe(21, 1))

    def test_new_receiver_cannot_replace_live_destination_hold(self):
        mgr = _prefill_mgr(NixlKVManager)
        generation = mgr.register_deferred_abort_room(21, expected_ranks={0})
        mgr.note_abort_ack(21, 0, generation)
        with self.assertRaisesRegex(RuntimeError, "destination owner"):
            mgr.register_deferred_abort_room(21)
        self.assertTrue(mgr.is_abort_release_safe(21, 1, generation))

    def test_same_room_epochs_keep_independent_destination_ack_state(self):
        mgr = _prefill_mgr(NixlKVManager)
        old = mgr.register_deferred_abort_room(21, expected_ranks={0}, transfer_epoch=1)
        new = mgr.register_deferred_abort_room(21, expected_ranks={1}, transfer_epoch=2)
        mgr.note_abort_ack(21, 0, old, transfer_epoch=1)
        self.assertTrue(mgr.is_abort_release_safe(21, 1, old, transfer_epoch=1))
        self.assertFalse(mgr.is_abort_release_safe(21, 1, new, transfer_epoch=2))
        mgr.clear_deferred_abort_state(21, old, transfer_epoch=1)
        self.assertNotIn((21, 1), mgr._abort_generations)
        self.assertIn((21, 2), mgr._abort_generations)

    def test_old_epoch_abort_does_not_fail_new_active_sender(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr._active_transfer_epochs = {}
        mgr.request_status[21] = KVPoll.Transferring
        mgr._active_transfer_epochs[21] = 2
        mgr.record_failure = MagicMock()
        mgr.update_status = MagicMock()
        mgr._send_generation_abort_ack = MagicMock()

        self.assertTrue(
            mgr._handle_abort_notification(
                [b"ABORT", b"21", b"10.0.0.1", b"6000", b"101", b"1"]
            )
        )

        mgr.update_status.assert_not_called()
        mgr._send_generation_abort_ack.assert_called_once_with(
            "10.0.0.1", 6000, 21, "101", 1
        )
        self.assertNotIn((21, 1), mgr._generation_ack_targets)

    def test_activate_uses_only_matching_early_metadata_epoch(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr._active_transfer_epochs = {}
        mgr._transfer_infos_by_attempt = {}
        mgr.record_failure = MagicMock()
        mgr.update_status = MagicMock(
            side_effect=lambda room, status: mgr.request_status.__setitem__(
                room, status
            )
        )
        mgr.resolve_kv_replica_factor = MagicMock()
        old = TransferInfo(
            21,
            "10.0.0.1",
            6000,
            "old",
            np.array([1], dtype=np.int32),
            1,
            1,
            [],
            transfer_epoch=1,
        )
        new = TransferInfo(
            21,
            "10.0.0.2",
            6001,
            "new",
            np.array([2], dtype=np.int32),
            2,
            1,
            [],
            transfer_epoch=2,
        )
        mgr._transfer_infos_by_attempt[(21, 1)] = {"old": old}
        mgr._transfer_infos_by_attempt[(21, 2)] = {"new": new}

        self.assertTrue(mgr.activate_transfer_attempt(21, 2))

        self.assertIs(mgr.transfer_infos[21]["new"], new)
        self.assertNotIn("old", mgr.transfer_infos[21])
        self.assertEqual(mgr._active_transfer_epochs[21], 2)
        self.assertEqual(mgr.request_status[21], KVPoll.WaitingForInput)

    def test_stale_sender_cannot_roll_back_new_active_epoch(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr._active_transfer_epochs = {21: 2}
        mgr._transfer_infos_by_attempt = {}
        current = TransferInfo(
            21,
            "10.0.0.2",
            6001,
            "new",
            np.array([2], dtype=np.int32),
            2,
            1,
            [],
            transfer_epoch=2,
        )
        stale = TransferInfo(
            21,
            "10.0.0.1",
            6000,
            "old",
            np.array([1], dtype=np.int32),
            1,
            1,
            [],
            transfer_epoch=1,
        )
        mgr.transfer_infos = {21: {"new": current}}
        mgr._transfer_infos_by_attempt[(21, 2)] = mgr.transfer_infos[21]
        mgr._transfer_infos_by_attempt[(21, 1)] = {"old": stale}
        mgr.request_status[21] = KVPoll.Transferring
        mgr.record_failure = MagicMock()
        mgr.update_status = MagicMock()

        self.assertFalse(mgr.activate_transfer_attempt(21, 1))

        self.assertEqual(mgr._active_transfer_epochs[21], 2)
        self.assertIs(mgr.transfer_infos[21]["new"], current)
        self.assertEqual(mgr.request_status[21], KVPoll.Transferring)
        mgr.record_failure.assert_not_called()
        mgr.update_status.assert_not_called()

    def test_abort_released_cleans_only_known_attempt_and_is_idempotent(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr._active_transfer_epochs = {21: 2}
        mgr._aborted_transfer_attempts = {(21, 1): {"101"}}
        mgr._released_transfer_attempts = {}
        mgr._generation_ack_targets = {
            (21, 1): {
                ("10.0.0.1", 6000): "101",
                ("10.0.0.2", 6001): "202",
            }
        }
        old = TransferInfo(
            21,
            "10.0.0.1",
            6000,
            "old",
            np.array([1], dtype=np.int32),
            1,
            1,
            [],
            transfer_epoch=1,
        )
        mgr._transfer_infos_by_attempt = {(21, 1): {"old": old}}
        mgr._send_multipart_locked = MagicMock()
        mgr._prefill_unique_rank = MagicMock(return_value=7)
        msg = [
            b"ABORT_RELEASED",
            b"21",
            b"10.0.0.9",
            b"7000",
            b"101",
            b"1",
        ]

        self.assertTrue(mgr._handle_abort_released_notification(msg))
        self.assertNotIn((21, 1), mgr._aborted_transfer_attempts)
        self.assertNotIn((21, 1), mgr._transfer_infos_by_attempt)
        self.assertEqual(
            mgr._generation_ack_targets[(21, 1)],
            {("10.0.0.2", 6001): "202"},
        )
        frames = mgr._send_multipart_locked.call_args.args[1]
        self.assertEqual(frames, [b"ABORT_RELEASED_ACK", b"21", b"7", b"101", b"1"])

        mgr._send_multipart_locked.reset_mock()
        self.assertTrue(mgr._handle_abort_released_notification(msg))
        mgr._send_multipart_locked.assert_called_once()
        unknown = [*msg[:4], b"202", msg[5]]
        mgr._send_multipart_locked.reset_mock()
        self.assertTrue(mgr._handle_abort_released_notification(unknown))
        mgr._send_multipart_locked.assert_not_called()

    def test_abort_released_notice_retries_only_unconfirmed_ranks(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr._abort_release_notices = {}
        mgr.local_ip, mgr.rank_port = "10.0.0.9", 7000
        receiver = SimpleNamespace(
            bootstrap_room=21,
            transfer_epoch=3,
            _abort_generation="101",
            bootstrap_infos=[
                {"abort_ack_rank": 0, "rank_ip": "10.0.0.1", "rank_port": 6000},
                {"abort_ack_rank": 1, "rank_ip": "10.0.0.2", "rank_port": 6001},
            ],
        )
        sockets = [MagicMock(), MagicMock()]
        with patch.object(
            CommonKVReceiver,
            "_connect_to_bootstrap_server",
            side_effect=[
                (sockets[0], threading.Lock()),
                (sockets[1], threading.Lock()),
            ],
        ):
            mgr.register_abort_released_notice(receiver)
        self.assertIn((21, 3), mgr._abort_release_notices)
        for socket in sockets:
            self.assertEqual(
                socket.send_multipart.call_args.args[0][0], b"ABORT_RELEASED"
            )

        mgr.handle_abort_released_ack_message(
            [b"ABORT_RELEASED_ACK", b"21", b"0", b"101", b"3"]
        )
        retry_socket = MagicMock()
        with patch.object(
            CommonKVReceiver,
            "_connect_to_bootstrap_server",
            return_value=(retry_socket, threading.Lock()),
        ) as connect:
            mgr.retry_abort_released_notices()
        connect.assert_called_once_with(receiver.bootstrap_infos[1])
        mgr.handle_abort_released_ack_message(
            [b"ABORT_RELEASED_ACK", b"21", b"1", b"101", b"3"]
        )
        self.assertNotIn((21, 3), mgr._abort_release_notices)

    def test_unexpected_ranks_cannot_satisfy_the_expected_sender_set(self):
        mgr = _prefill_mgr(NixlKVManager)
        expected = {3, 11}
        generation = mgr.register_deferred_abort_room(21, expected_ranks=expected)
        expected.add(99)  # Tracker keeps the arm-time immutable snapshot.
        for rank in (0, 1, 99, -1, 3, 3):
            mgr.note_abort_ack(21, rank, generation)
        self.assertFalse(mgr.is_abort_release_safe(21, 2, generation))
        self.assertFalse(mgr.is_abort_release_safe(21, 1, generation))
        mgr.note_abort_ack(21, 11, generation)
        self.assertTrue(mgr.is_abort_release_safe(21, 2, generation))
        self.assertFalse(mgr.is_abort_release_safe(21, 3, generation))

    def test_unknown_or_invalid_expected_identities_never_prove_drain(self):
        for expected in (None, set(), {-1}, {True}):
            with self.subTest(expected=expected):
                mgr = _prefill_mgr(NixlKVManager)
                generation = mgr.register_deferred_abort_room(
                    21, expected_ranks=expected
                )
                for rank in range(8):
                    mgr.note_abort_ack(21, rank, generation)
                self.assertFalse(mgr.is_abort_release_safe(21, 1, generation))
                self.assertFalse(mgr.is_abort_release_safe(21, 0, generation))

    def test_per_endpoint_generation_targets_survive_clear_and_failed_send(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr.request_status[21] = KVPoll.Failed
        mgr._staging_outstanding[21] = 1
        mgr._send_generation_abort_ack = MagicMock()
        mgr.register_deferred_ack_target(21, "10.0.0.1", 6000, "101")
        mgr.register_deferred_ack_target(21, "10.0.0.2", 6000, "202")
        sender = NixlKVSender.__new__(NixlKVSender)
        sender.kv_mgr, sender.bootstrap_room = mgr, 21
        sender.clear()
        mgr._send_generation_abort_ack.assert_not_called()
        mgr._staging_outstanding.pop(21)

        def fail_first_endpoint(ip, port, room, generation, transfer_epoch):
            if ip == "10.0.0.1":
                raise RuntimeError("send unavailable")

        mgr._send_generation_abort_ack.side_effect = fail_first_endpoint
        mgr._maybe_ack_drained_abort(21)
        self.assertEqual(
            mgr._generation_ack_targets[(21, 0)], {("10.0.0.1", 6000): "101"}
        )
        mgr._send_generation_abort_ack.side_effect = None
        mgr.retry_drained_abort_acks()
        self.assertNotIn((21, 0), mgr._generation_ack_targets)
        self.assertEqual(mgr._send_generation_abort_ack.call_count, 3)

    def test_generation_ack_wire_contains_room_rank_and_integer_nonce(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr._prefill_unique_rank = MagicMock(return_value=7)
        mgr._send_multipart_locked = MagicMock()
        mgr._send_generation_abort_ack("10.0.0.1", 6000, 21, "101", 3)
        call = mgr._send_multipart_locked.call_args
        self.assertEqual(call.args[1], [b"ABORT_ACK", b"21", b"7", b"101", b"3"])

    def test_retry_does_not_ack_unproven_source_group(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr._staging_outstanding[21] = 1
        mgr._send_generation_abort_ack = MagicMock()
        mgr.register_deferred_ack_target(21, "10.0.0.1", 6000, "101")
        mgr.retry_drained_abort_acks()
        mgr._send_generation_abort_ack.assert_not_called()
        self.assertEqual(mgr._staging_outstanding[21], 1)

    def test_generation_abort_echo_waits_for_retained_source_handles(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr.request_status[21] = KVPoll.Transferring
        mgr.record_failure = MagicMock()
        mgr._failed_transfer_handles[21] = [object()]
        mgr._send_generation_abort_ack = MagicMock()
        mgr._handle_abort_notification([b"ABORT", b"21", b"10.0.0.1", b"6000", b"101"])
        self.assertEqual(
            mgr._generation_ack_targets[(21, 0)], {("10.0.0.1", 6000): "101"}
        )
        mgr._send_generation_abort_ack.assert_not_called()
        self.assertTrue(mgr.has_pending_source_reads(21))

    def test_conflicting_generation_does_not_kill_bootstrap_or_erase_old_owner(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr.request_status[21] = KVPoll.Transferring
        mgr.record_failure = MagicMock()
        mgr._staging_outstanding[21] = 1
        mgr._handle_abort_notification([b"ABORT", b"21", b"10.0.0.1", b"6000", b"101"])
        self.assertTrue(
            mgr._handle_abort_notification(
                [b"ABORT", b"21", b"10.0.0.1", b"6000", b"202"]
            )
        )
        self.assertEqual(
            mgr._generation_ack_targets[(21, 0)], {("10.0.0.1", 6000): "101"}
        )
        self.assertEqual(mgr._staging_outstanding[21], 1)


if __name__ == "__main__":
    unittest.main()
