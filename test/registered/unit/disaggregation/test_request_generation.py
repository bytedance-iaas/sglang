import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVSender
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    TransferInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_manager():
    manager = object.__new__(CommonKVManager)
    manager.request_status = {}
    manager.request_generation = {}
    manager.next_request_generation = {}
    manager.request_status_history = {}
    manager.request_failure_history = {}
    manager.request_bootstrap_activity = {}
    manager.request_status_lock = threading.RLock()
    manager.failure_records = {}
    manager.failure_lock = threading.Lock()
    manager.transfer_infos = {}
    manager.req_to_decode_prefix_len = {}
    manager.required_prefill_response_num_table = {}
    manager.prefill_response_tracker = {}
    return manager


class TestRequestGeneration(unittest.TestCase):
    @patch(
        "sglang.srt.disaggregation.mooncake.conn.envs."
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get",
        return_value=600,
    )
    @patch(
        "sglang.srt.disaggregation.mooncake.conn.time.monotonic",
        return_value=31.0,
    )
    def test_decode_admission_keepalive_is_generation_scoped_and_throttled(
        self, _monotonic, _timeout
    ):
        receiver = object.__new__(MooncakeKVReceiver)
        receiver.bootstrap_room = 24
        receiver.generation = 7
        receiver.conclude_state = None
        receiver.bootstrap_infos = [{"rank_ip": "127.0.0.1", "rank_port": 5000}]
        receiver._last_bootstrap_keepalive_time = 0.0
        receiver.kv_mgr = MagicMock()
        receiver.kv_mgr.check_status.return_value = KVPoll.WaitingForInput
        sock = MagicMock()
        lock = MagicMock()
        lock.__enter__.return_value = lock
        lock.__exit__.return_value = False

        with patch.object(
            receiver, "_connect_to_bootstrap_server", return_value=(sock, lock)
        ):
            self.assertTrue(receiver.renew_bootstrap_lease())
            self.assertFalse(receiver.renew_bootstrap_lease())

        sock.send_multipart.assert_called_once_with(
            [b"BOOTSTRAP_KEEPALIVE", b"24", b"7"]
        )

    def test_bootstrap_lease_renews_only_current_bootstrapping_generation(self):
        manager = make_manager()
        generation, _ = manager.begin_request(21, KVPoll.Bootstrapping)

        self.assertTrue(manager.renew_bootstrap_activity(21, generation, 123.0))
        self.assertEqual(manager.get_bootstrap_activity(21, generation), 123.0)

        manager.update_status(21, KVPoll.WaitingForInput, generation)
        self.assertFalse(manager.renew_bootstrap_activity(21, generation, 456.0))
        self.assertEqual(manager.get_bootstrap_activity(21, generation), 123.0)

    def test_stale_bootstrap_lease_cannot_renew_reused_room(self):
        manager = make_manager()
        old_generation, _ = manager.begin_request(22, KVPoll.Bootstrapping)
        manager.update_status(22, KVPoll.Success, old_generation)
        new_generation, _ = manager.begin_request(22, KVPoll.Bootstrapping)

        self.assertFalse(manager.renew_bootstrap_activity(22, old_generation, 456.0))
        self.assertIsNone(manager.get_bootstrap_activity(22, old_generation))
        self.assertIsNone(manager.get_bootstrap_activity(22, new_generation))

    def test_sender_timeout_uses_latest_decode_admission_lease(self):
        manager = make_manager()
        manager.bootstrap_timeout = 10
        generation, _ = manager.begin_request(23, KVPoll.Bootstrapping)
        sender = object.__new__(CommonKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = 23
        sender.generation = generation
        sender.init_time = time.time() - 20

        manager.renew_bootstrap_activity(23, generation, time.time())
        self.assertIsNone(sender._check_bootstrap_timeout())

        manager.request_bootstrap_activity[(23, generation)] = time.time() - 20
        self.assertEqual(sender._check_bootstrap_timeout(), KVPoll.Failed)

    def test_sender_and_metadata_race_join_one_generation(self):
        for _ in range(100):
            manager = make_manager()
            barrier = threading.Barrier(3)
            results = []

            def begin(initial_status, generation=None):
                barrier.wait()
                results.append(manager.begin_request(15, initial_status, generation))

            sender = threading.Thread(target=begin, args=(KVPoll.Bootstrapping,))
            metadata = threading.Thread(target=begin, args=(KVPoll.WaitingForInput, 1))
            sender.start()
            metadata.start()
            barrier.wait()
            sender.join()
            metadata.join()

            self.assertEqual({generation for generation, _ in results}, {1})
            self.assertEqual(manager.check_status(15, 1), KVPoll.WaitingForInput)

    def test_terminal_room_starts_new_generation(self):
        manager = make_manager()
        generation_1, started = manager.begin_request(7, KVPoll.Bootstrapping)
        self.assertTrue(started)
        manager.update_status(7, KVPoll.Success, generation_1)

        generation_2, started = manager.begin_request(7, KVPoll.Bootstrapping)

        self.assertTrue(started)
        self.assertEqual(generation_2, generation_1 + 1)
        self.assertEqual(manager.check_status(7, generation_2), KVPoll.Bootstrapping)
        self.assertEqual(manager.check_status(7, generation_1), KVPoll.Success)

    def test_sender_joins_metadata_generation_without_downgrading_ready_state(self):
        manager = make_manager()
        generation, started = manager.begin_request(8, KVPoll.Bootstrapping, 1)
        self.assertTrue(started)
        manager.update_status(8, KVPoll.WaitingForInput, generation)

        joined_generation, started = manager.begin_request(8, KVPoll.Bootstrapping)

        self.assertFalse(started)
        self.assertEqual(joined_generation, generation)
        self.assertEqual(manager.check_status(8, generation), KVPoll.WaitingForInput)

    def test_authoritative_generation_cannot_overtake_nonterminal_transfer(self):
        manager = make_manager()
        generation_1, _ = manager.begin_request(18, KVPoll.Bootstrapping)

        active_generation, started = manager.begin_request(
            18, KVPoll.Bootstrapping, generation_1 + 1
        )

        self.assertFalse(started)
        self.assertEqual(active_generation, generation_1)
        self.assertTrue(manager.update_status(18, KVPoll.Success, generation_1))

    def test_sender_fails_closed_when_authoritative_generation_overlaps(self):
        manager = make_manager()
        manager.is_dummy_cp_rank = False
        manager.supports_request_generation = True
        generation_1, _ = manager.begin_request(20, KVPoll.Bootstrapping)

        sender = object.__new__(CommonKVSender)
        # Avoid unrelated routing setup; exercise the generation admission
        # portion of the real constructor through a single-rank parallel mock.
        with unittest.mock.patch(
            "sglang.srt.disaggregation.common.conn.get_parallel"
        ) as parallel:
            parallel.return_value.dp_size = 1
            CommonKVSender.__init__(
                sender, manager, "127.0.0.1:8998", 20, [0], 0, generation=2
            )

        self.assertEqual(sender.generation, 2)
        self.assertEqual(sender.conclude_state, KVPoll.Failed)
        self.assertIn(
            "overlapped active generation 1", sender._generation_rejected_reason
        )
        self.assertEqual(manager.check_status(20, generation_1), KVPoll.Bootstrapping)

    def test_authoritative_generation_replaces_terminal_old_sender(self):
        manager = make_manager()
        generation_1, _ = manager.begin_request(19, KVPoll.Bootstrapping)
        manager.update_status(19, KVPoll.Success, generation_1)

        generation_2, started = manager.begin_request(
            19, KVPoll.Bootstrapping, generation_1 + 1
        )

        self.assertTrue(started)
        self.assertEqual(generation_2, 2)
        self.assertEqual(manager.check_status(19, generation_1), KVPoll.Success)
        self.assertEqual(manager.check_status(19, generation_2), KVPoll.Bootstrapping)
        self.assertFalse(manager.update_status(19, KVPoll.Failed, generation_1))

    def test_stale_updates_and_clear_do_not_touch_new_generation(self):
        manager = make_manager()
        old_generation, _ = manager.begin_request(9, KVPoll.Bootstrapping)
        manager.update_status(9, KVPoll.Success, old_generation)
        new_generation, _ = manager.begin_request(9, KVPoll.Bootstrapping)
        manager.transfer_infos[9] = {"new": object()}
        manager.req_to_decode_prefix_len[9] = 12

        self.assertFalse(manager.update_status(9, KVPoll.Failed, old_generation))
        manager.record_failure(9, "stale failure", old_generation)
        self.assertFalse(manager.clear_request(9, old_generation))

        self.assertEqual(manager.check_status(9, new_generation), KVPoll.Bootstrapping)
        self.assertNotIn(9, manager.failure_records)
        self.assertIn(9, manager.transfer_infos)
        self.assertIn(9, manager.req_to_decode_prefix_len)

    def test_failure_reason_remains_owned_by_old_generation(self):
        manager = make_manager()
        generation_1, _ = manager.begin_request(16, KVPoll.Bootstrapping)
        manager.record_failure(16, "old transfer failed", generation_1)
        manager.update_status(16, KVPoll.Failed, generation_1)

        generation_2, _ = manager.begin_request(16, KVPoll.Bootstrapping)
        manager.record_failure(16, "new transfer failed", generation_2)

        self.assertEqual(manager.pop_failure(16, generation_1), "old transfer failed")
        self.assertEqual(manager.pop_failure(16, generation_2), "new transfer failed")

    def test_old_generation_clear_only_drops_its_failure_reason(self):
        manager = make_manager()
        generation_1, _ = manager.begin_request(17, KVPoll.Bootstrapping)
        manager.record_failure(17, "old transfer failed", generation_1)
        manager.update_status(17, KVPoll.Failed, generation_1)
        generation_2, _ = manager.begin_request(17, KVPoll.Bootstrapping)
        manager.record_failure(17, "new transfer failed", generation_2)

        self.assertFalse(manager.clear_request(17, generation_1))
        self.assertIsNone(manager.pop_failure(17, generation_1))
        self.assertEqual(manager.pop_failure(17, generation_2), "new transfer failed")

    def test_duplicate_metadata_cannot_resurrect_cleared_generation(self):
        manager = make_manager()
        generation, _ = manager.begin_request(10, KVPoll.Bootstrapping, 1)
        self.assertTrue(manager.clear_request(10, generation))

        active_generation, started = manager.begin_request(
            10, KVPoll.Bootstrapping, generation
        )

        self.assertFalse(started)
        self.assertEqual(active_generation, generation)
        self.assertFalse(manager.is_current_generation(10, generation))
        self.assertNotIn(10, manager.request_status)

    def test_generationless_message_is_rejected_after_room_reuse(self):
        manager = make_manager()
        generation_1, _ = manager.begin_request(14, KVPoll.Bootstrapping)
        manager.update_status(14, KVPoll.Success, generation_1)
        generation_2, _ = manager.begin_request(14, KVPoll.Bootstrapping)

        self.assertEqual(generation_2, 2)
        self.assertFalse(manager.is_current_generation(14, None))

    def test_mooncake_metadata_round_trip_includes_generation(self):
        kv_indices = np.array([2, 4], dtype=np.int32)
        info = TransferInfo.from_zmq(
            [
                b"11",
                b"127.0.0.1",
                b"5000",
                b"session",
                kv_indices.tobytes(),
                b"3",
                b"",
                b"1",
                b"42",
                b"",
                b"7",
            ]
        )

        self.assertEqual(info.generation, 7)
        np.testing.assert_array_equal(info.dst_kv_indices, kv_indices)

    def test_mooncake_old_metadata_remains_compatible(self):
        info = TransferInfo.from_zmq(
            [
                b"12",
                b"127.0.0.1",
                b"5001",
                b"session",
                np.array([1], dtype=np.int32).tobytes(),
                b"0",
                b"",
                b"1",
            ]
        )

        self.assertIsNone(info.generation)

    def test_mooncake_status_wire_only_appends_generation_when_supported(self):
        manager = object.__new__(MooncakeKVManager)
        manager._send_multipart_locked = MagicMock()

        manager.sync_status_to_decode_endpoint("127.0.0.1", 5002, 13, KVPoll.Success, 4)
        legacy_message = manager._send_multipart_locked.call_args.args[1]
        self.assertEqual(legacy_message, [b"13", b"4", b"4"])

        manager.sync_status_to_decode_endpoint(
            "127.0.0.1", 5002, 13, KVPoll.Success, 4, generation=9
        )
        generation_message = manager._send_multipart_locked.call_args.args[1]
        self.assertEqual(generation_message, [b"13", b"4", b"4", b"9"])


if __name__ == "__main__":
    unittest.main()
