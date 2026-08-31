import threading
import time
import unittest
from collections import defaultdict
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.fake.conn import FakeKVSender
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
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
    manager.request_owner_lease_expected = {}
    manager.request_owner_lease_activity = {}
    manager.request_owner_lease_started = {}
    manager.request_status_lock = threading.RLock()
    manager.failure_records = {}
    manager.failure_lock = threading.Lock()
    manager.transfer_infos = {}
    manager.req_to_decode_prefix_len = {}
    manager.required_prefill_response_num_table = {}
    manager.prefill_response_tracker = {}
    return manager


class MinimalCommonKVReceiver(CommonKVReceiver):
    def poll(self) -> KVPoll:
        return KVPoll.WaitingForInput


class TestRequestGeneration(unittest.TestCase):
    def test_mooncake_transfer_worker_failed_session_propagates_failure(self):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_trace = False
        manager.enable_staging = False
        manager.request_status = {46: KVPoll.WaitingForInput}
        manager.request_status_lock = threading.RLock()
        manager.transfer_infos = {
            46: {
                "session": MagicMock(
                    is_dummy=False,
                    mooncake_session_id="session",
                    endpoint="decode",
                    dst_port=9000,
                    room=46,
                    generation=3,
                )
            }
        }
        manager.session_lock = threading.Lock()
        manager.failed_sessions = {"session"}
        manager.is_current_generation = MagicMock(return_value=True)
        manager.check_status = MagicMock(return_value=KVPoll.WaitingForInput)
        manager.record_failure = MagicMock()
        manager.update_status = MagicMock(return_value=True)
        manager.sync_status_to_decode_endpoint = MagicMock()
        manager._finish_transfer_attempt = MagicMock()
        manager.attn_tp_rank = 0
        manager.pp_size = 1
        manager.attn_cp_size = 1
        manager.pp_rank = 0
        manager.attn_cp_rank = 0
        chunk = TransferKVChunk(
            room=46,
            prefill_kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(0, 0),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
            generation=3,
        )
        queue = MagicMock()
        queue.get.side_effect = [chunk, SystemExit]

        with self.assertRaises(SystemExit):
            manager.transfer_worker(queue, executor=None)

        manager.update_status.assert_called_once_with(46, KVPoll.Failed, 3)
        manager.sync_status_to_decode_endpoint.assert_called_once_with(
            "decode", 9000, 46, KVPoll.Failed, 0, 3
        )
        manager._finish_transfer_attempt.assert_called_once_with(46, 3)
        self.assertEqual(queue.get.call_count, 2)

    def test_mooncake_failed_session_does_not_notify_reused_room(self):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_trace = False
        manager.enable_staging = False
        manager.request_status = {46: KVPoll.WaitingForInput}
        manager.request_status_lock = threading.RLock()
        manager.transfer_infos = {
            46: {
                "old-generation": MagicMock(
                    is_dummy=False,
                    mooncake_session_id="failed-session",
                    endpoint="decode-old",
                    dst_port=9000,
                    room=46,
                    generation=3,
                )
            }
        }
        manager.session_lock = threading.Lock()
        manager.failed_sessions = {"failed-session"}
        manager.is_current_generation = MagicMock(return_value=True)
        manager.check_status = MagicMock(return_value=KVPoll.WaitingForInput)
        manager.record_failure = MagicMock()
        manager.update_status = MagicMock(return_value=False)
        manager.sync_status_to_decode_endpoint = MagicMock()
        manager._finish_transfer_attempt = MagicMock()
        manager.attn_tp_rank = 0
        manager.pp_size = 1
        manager.attn_cp_size = 1
        manager.pp_rank = 0
        manager.attn_cp_rank = 0
        chunk = TransferKVChunk(
            room=46,
            prefill_kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(0, 0),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
            generation=3,
        )
        queue = MagicMock()
        queue.get.side_effect = [chunk, SystemExit]

        with self.assertRaises(SystemExit):
            manager.transfer_worker(queue, executor=None)

        manager.update_status.assert_called_once_with(46, KVPoll.Failed, 3)
        manager.sync_status_to_decode_endpoint.assert_not_called()
        manager._finish_transfer_attempt.assert_called_once_with(46, 3)
        self.assertEqual(queue.get.call_count, 2)

    def test_mooncake_transfer_worker_exception_propagates_and_continues(self):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_trace = False
        manager.enable_staging = False
        manager.request_status_lock = threading.RLock()
        manager.transfer_infos = {
            47: {
                "session": MagicMock(
                    endpoint="decode",
                    dst_port=9001,
                    generation=4,
                )
            }
        }
        manager.is_current_generation = MagicMock(
            side_effect=RuntimeError("unexpected worker error")
        )
        manager.record_failure = MagicMock()
        manager.update_status = MagicMock()
        manager.sync_status_to_decode_endpoint = MagicMock()
        manager._prefill_unique_rank = MagicMock(return_value=6)
        manager._finish_transfer_attempt = MagicMock()
        chunk = TransferKVChunk(
            room=47,
            prefill_kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(0, 0),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
            generation=4,
        )
        queue = MagicMock()
        queue.get.side_effect = [chunk, SystemExit]

        with self.assertRaises(SystemExit):
            manager.transfer_worker(queue, executor=None)

        manager.update_status.assert_called_once_with(47, KVPoll.Failed, 4)
        manager.sync_status_to_decode_endpoint.assert_called_once_with(
            "decode", 9001, 47, KVPoll.Failed, 6, 4
        )
        manager._finish_transfer_attempt.assert_called_once_with(47, 4)
        self.assertEqual(queue.get.call_count, 2)

    def test_mooncake_transfer_worker_snapshots_failure_targets(self):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_trace = False
        manager.enable_staging = False
        manager.request_status_lock = threading.RLock()
        manager.transfer_infos = {
            48: {
                "session-a": MagicMock(
                    endpoint="decode-a", dst_port=9002, generation=5
                ),
                "session-b": MagicMock(
                    endpoint="decode-b", dst_port=9003, generation=5
                ),
                "stale": MagicMock(
                    endpoint="decode-stale", dst_port=9004, generation=4
                ),
            }
        }
        manager.is_current_generation = MagicMock(
            side_effect=RuntimeError("unexpected worker error")
        )
        manager.record_failure = MagicMock()
        manager.update_status = MagicMock(return_value=True)
        manager._prefill_unique_rank = MagicMock(return_value=7)
        manager._finish_transfer_attempt = MagicMock()

        def mutate_transfer_infos(*_args):
            manager.transfer_infos[48].clear()

        manager.sync_status_to_decode_endpoint = MagicMock(
            side_effect=mutate_transfer_infos
        )
        chunk = TransferKVChunk(
            room=48,
            prefill_kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(0, 0),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
            generation=5,
        )
        queue = MagicMock()
        queue.get.side_effect = [chunk, SystemExit]

        with self.assertRaises(SystemExit):
            manager.transfer_worker(queue, executor=None)

        manager.sync_status_to_decode_endpoint.assert_has_calls(
            [
                unittest.mock.call("decode-a", 9002, 48, KVPoll.Failed, 7, 5),
                unittest.mock.call("decode-b", 9003, 48, KVPoll.Failed, 7, 5),
            ],
            any_order=True,
        )
        self.assertEqual(manager.sync_status_to_decode_endpoint.call_count, 2)
        manager._finish_transfer_attempt.assert_called_once_with(48, 5)
        self.assertEqual(queue.get.call_count, 2)

    def test_mooncake_transfer_worker_does_not_notify_reused_room(self):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_trace = False
        manager.enable_staging = False
        manager.request_status_lock = threading.RLock()
        manager.transfer_infos = {
            49: {
                "new-generation": MagicMock(
                    endpoint="decode-new", dst_port=9005, generation=6
                )
            }
        }
        manager.is_current_generation = MagicMock(
            side_effect=RuntimeError("old generation worker error")
        )
        manager.record_failure = MagicMock()
        manager.update_status = MagicMock(return_value=False)
        manager.sync_status_to_decode_endpoint = MagicMock()
        manager._finish_transfer_attempt = MagicMock()
        chunk = TransferKVChunk(
            room=49,
            prefill_kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(0, 0),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
            generation=5,
        )
        queue = MagicMock()
        queue.get.side_effect = [chunk, SystemExit]

        with self.assertRaises(SystemExit):
            manager.transfer_worker(queue, executor=None)

        manager.sync_status_to_decode_endpoint.assert_not_called()
        manager._finish_transfer_attempt.assert_called_once_with(49, 5)
        self.assertEqual(queue.get.call_count, 2)

    def test_deferred_abort_arm_is_idempotent_and_preserves_early_ack(self):
        manager = make_manager()
        generation, _ = manager.begin_request(50, KVPoll.WaitingForInput)
        self.assertTrue(manager.register_deferred_abort_room(50, generation, {3, 4}))
        self.assertTrue(manager.note_abort_ack(50, 3, generation))
        self.assertTrue(manager.register_deferred_abort_room(50, generation, {3, 4}))
        self.assertEqual(manager._deferred_abort_ack_tracker[(50, generation)], {3})
        self.assertFalse(manager.register_deferred_abort_room(50, generation, {3}))

    def test_deferred_abort_arm_rejects_empty_expected_rank_set(self):
        manager = make_manager()
        generation, _ = manager.begin_request(53, KVPoll.WaitingForInput)

        self.assertFalse(manager.register_deferred_abort_room(53, generation, set()))
        self.assertNotIn((53, generation), manager._deferred_abort_ack_tracker)
        self.assertNotIn((53, generation), manager._deferred_abort_expected_ranks)
        self.assertFalse(manager.is_abort_release_safe(53, generation))

    def test_cleared_terminal_generation_cannot_rearm_abort_tracker(self):
        manager = make_manager()
        generation, _ = manager.begin_request(54, KVPoll.WaitingForInput)
        self.assertTrue(manager.register_deferred_abort_room(54, generation, {2}))
        manager.update_status(54, KVPoll.Failed, generation)
        manager.clear_deferred_abort_state(54, generation)

        self.assertFalse(
            manager.register_deferred_abort_room(54, generation, expected_ranks={2})
        )
        self.assertFalse(manager.note_abort_ack(54, 2, generation))
        self.assertFalse(manager.is_abort_release_safe(54, generation))

        next_generation, started = manager.begin_request(54, KVPoll.WaitingForInput)
        self.assertTrue(started)
        self.assertGreater(next_generation, generation)
        self.assertTrue(
            manager.register_deferred_abort_room(
                54, next_generation, expected_ranks={2}
            )
        )

    def test_old_generation_abort_ack_uses_armed_tracker_after_room_reuse(self):
        manager = make_manager()
        old, _ = manager.begin_request(51, KVPoll.WaitingForInput)
        manager.register_deferred_abort_room(51, old, {7})
        manager.update_status(51, KVPoll.Success, old)
        new, _ = manager.begin_request(51, KVPoll.WaitingForInput)
        self.assertNotEqual(old, new)
        self.assertTrue(manager.note_abort_ack(51, 7, old))
        self.assertTrue(manager.is_abort_release_safe(51, old))
        self.assertFalse(manager.note_abort_ack(51, 7, new))
        self.assertFalse(manager.is_abort_release_safe(51, new))

    def test_owner_lease_requires_all_expected_ranks_to_stay_fresh(self):
        manager = make_manager()
        generation, _ = manager.begin_request(31, KVPoll.WaitingForInput)
        self.assertTrue(manager.arm_decode_owner_lease(31, generation, {2, 5}, 100.0))

        with patch(
            "sglang.srt.disaggregation.common.conn.envs."
            "SGLANG_DISAGGREGATION_OWNER_LEASE_TIMEOUT.get",
            return_value=300.0,
        ), patch(
            "sglang.srt.disaggregation.common.conn.envs."
            "SGLANG_DISAGGREGATION_OWNER_LEASE_MAX_LIFETIME.get",
            return_value=3600.0,
        ):
            self.assertTrue(manager.note_decode_owner_lease(31, 2, generation, 399.0))
            self.assertFalse(manager.decode_owner_lease_expired(31, generation, 399.0))
            self.assertTrue(manager.decode_owner_lease_expired(31, generation, 401.0))

    def test_owner_lease_rearm_does_not_extend_absolute_cap(self):
        manager = make_manager()
        generation, _ = manager.begin_request(52, KVPoll.WaitingForInput)
        self.assertTrue(manager.arm_decode_owner_lease(52, generation, {0}, 100.0))
        self.assertTrue(manager.note_decode_owner_lease(52, 0, generation, 200.0))
        self.assertTrue(manager.arm_decode_owner_lease(52, generation, {0}, 300.0))
        self.assertEqual(manager.request_owner_lease_started[(52, generation)], 100.0)
        self.assertEqual(
            manager.request_owner_lease_activity[(52, generation)][0], 200.0
        )
        self.assertFalse(manager.arm_decode_owner_lease(52, generation, {1}, 300.0))

    def test_owner_lease_absolute_cap_and_terminal_state(self):
        manager = make_manager()
        generation, _ = manager.begin_request(32, KVPoll.WaitingForInput)
        manager.arm_decode_owner_lease(32, generation, {0}, 100.0)
        manager.note_decode_owner_lease(32, 0, generation, 3699.0)
        with patch(
            "sglang.srt.disaggregation.common.conn.envs."
            "SGLANG_DISAGGREGATION_OWNER_LEASE_TIMEOUT.get",
            return_value=300.0,
        ), patch(
            "sglang.srt.disaggregation.common.conn.envs."
            "SGLANG_DISAGGREGATION_OWNER_LEASE_MAX_LIFETIME.get",
            return_value=3600.0,
        ):
            self.assertTrue(manager.decode_owner_lease_expired(32, generation, 3700.0))

        manager.update_status(32, KVPoll.Success, generation)
        receiver = object.__new__(MinimalCommonKVReceiver)
        receiver.bootstrap_room = 32
        receiver.generation = generation
        receiver.kv_mgr = manager
        receiver.conclude_state = None
        receiver.poll = MagicMock()
        self.assertFalse(manager.note_decode_owner_lease(32, 0, generation, 3701.0))

    def test_expired_owner_lease_fails_without_waiting_for_legacy_timeout(self):
        manager = make_manager()
        manager.enable_decode_owner_lease = True
        manager.waiting_timeout = 1800
        generation, _ = manager.begin_request(38, KVPoll.WaitingForInput)
        manager.arm_decode_owner_lease(38, generation, {0, 1}, 100.0)
        manager.note_decode_owner_lease(38, 0, generation, 399.0)
        receiver = object.__new__(MinimalCommonKVReceiver)
        receiver.bootstrap_room = 38
        receiver.generation = generation
        receiver.kv_mgr = manager
        receiver.init_time = 10_000.0
        receiver.abort_notified = True

        with patch(
            "sglang.srt.disaggregation.common.conn.time.monotonic",
            return_value=401.0,
        ), patch(
            "sglang.srt.disaggregation.common.conn.time.time",
            return_value=10_001.0,
        ), patch(
            "sglang.srt.disaggregation.common.conn.envs."
            "SGLANG_DISAGGREGATION_OWNER_LEASE_TIMEOUT.get",
            return_value=300.0,
        ), patch(
            "sglang.srt.disaggregation.common.conn.envs."
            "SGLANG_DISAGGREGATION_OWNER_LEASE_MAX_LIFETIME.get",
            return_value=3600.0,
        ):
            self.assertEqual(receiver._check_waiting_timeout(), KVPoll.Failed)

        self.assertEqual(manager.check_status(38, generation), KVPoll.Failed)
        self.assertIn("owner lease expired", manager.pop_failure(38, generation))

    def test_legacy_waiting_timeout_uses_wall_clock(self):
        manager = make_manager()
        manager.enable_decode_owner_lease = False
        manager.waiting_timeout = 10
        generation, _ = manager.begin_request(39, KVPoll.WaitingForInput)
        receiver = object.__new__(MinimalCommonKVReceiver)
        receiver.bootstrap_room = 39
        receiver.generation = generation
        receiver.kv_mgr = manager
        receiver.init_time = 1_000.0
        receiver.abort_notified = True
        with patch(
            "sglang.srt.disaggregation.common.conn.time.time", return_value=1_005.0
        ), patch(
            "sglang.srt.disaggregation.common.conn.time.monotonic",
            return_value=99_999.0,
        ):
            self.assertIsNone(receiver._check_waiting_timeout())

    def test_owner_lease_stale_generation_and_clear_reuse(self):
        manager = make_manager()
        old, _ = manager.begin_request(33, KVPoll.WaitingForInput)
        manager.arm_decode_owner_lease(33, old, {0}, 10.0)
        manager.update_status(33, KVPoll.Success, old)
        new, _ = manager.begin_request(33, KVPoll.WaitingForInput)
        self.assertFalse(manager.note_decode_owner_lease(33, 0, old, 20.0))
        self.assertFalse(manager.has_decode_owner_lease(33, old))
        manager.arm_decode_owner_lease(33, new, {1}, 30.0)
        self.assertTrue(manager.clear_request(33, new))
        self.assertFalse(manager.has_decode_owner_lease(33, new))

    def test_expected_prefill_ranks_match_topology_and_exclude_mla_dummies(self):
        manager = MagicMock()
        manager.supports_request_generation = True
        manager.begin_request.return_value = (1, True)
        manager.addr_to_rooms_tracker = {"host": set()}
        manager.prefill_info_table = {
            "host": MagicMock(
                target_tp_rank=1,
                target_tp_ranks=[0, 1],
                target_cp_ranks=[0, 1],
                target_pp_ranks=[0, 1],
                required_dst_info_num=1,
                required_prefill_response_num=4,
                pp_size=2,
                attn_cp_size=2,
                attn_tp_size=2,
            )
        }
        manager.required_prefill_response_num_table = {}
        manager.enable_staging = False
        manager.is_mla_backend = True
        receiver = MinimalCommonKVReceiver(manager, "host", 34)
        receiver._setup_bootstrap_infos = MagicMock()
        receiver.init(0)
        self.assertEqual(receiver.expected_prefill_ranks, {4, 5, 6, 7})

    def test_fake_sender_accepts_request_generation_for_health_requests(self):
        sender = FakeKVSender(
            MagicMock(),
            "127.0.0.1:8998",
            25,
            [0],
            0,
            generation=7,
        )

        self.assertEqual(sender.generation, 7)
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

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

    def test_legacy_bootstrap_timeout_uses_wall_clock(self):
        manager = make_manager()
        manager.bootstrap_timeout = 10
        generation, _ = manager.begin_request(40, KVPoll.Bootstrapping)
        sender = object.__new__(CommonKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = 40
        sender.generation = generation
        sender.init_time = 1_000.0
        with patch(
            "sglang.srt.disaggregation.common.conn.time.time", return_value=1_005.0
        ), patch(
            "sglang.srt.disaggregation.common.conn.time.monotonic",
            return_value=99_999.0,
        ):
            self.assertIsNone(sender._check_bootstrap_timeout())

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

    def test_mooncake_owner_lease_wire_shape_and_throttle(self):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_decode_owner_lease = True
        manager.request_status_lock = threading.RLock()
        manager.request_status = {35: KVPoll.WaitingForInput}
        manager.request_generation = {35: 8}
        manager.request_status_history = {}
        manager.request_failure_history = {}
        manager.request_bootstrap_activity = {}
        manager.transfer_infos = {
            35: {
                "session": MagicMock(
                    is_dummy=False,
                    generation=8,
                    endpoint="127.0.0.1",
                    dst_port=5003,
                )
            }
        }
        manager._owner_lease_last_sent = {}
        manager.attn_tp_rank = 1
        manager.pp_size = 2
        manager.attn_cp_size = 2
        manager.pp_rank = 0
        manager.attn_cp_rank = 1
        manager._send_multipart_locked = MagicMock()
        sender = object.__new__(MooncakeKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = 35
        sender.generation = 8
        sender.conclude_state = None

        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic",
            side_effect=[100.0, 101.0],
        ):
            self.assertTrue(sender.renew_decode_owner_lease())
            self.assertFalse(sender.renew_decode_owner_lease())

        self.assertEqual(
            manager._send_multipart_locked.call_args.args[1],
            [b"PREFILL_OWNER_LEASE", b"35", b"5", b"8"],
        )

    def test_mooncake_owner_lease_does_not_renew_terminal_request(self):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_decode_owner_lease = True
        manager.request_status_lock = threading.RLock()
        manager.request_status = {35: KVPoll.Failed}
        manager.request_generation = {35: 8}
        manager.request_status_history = {}
        manager.request_failure_history = {}
        manager.request_bootstrap_activity = {}
        manager.transfer_infos = {35: {}}
        manager._owner_lease_last_sent = {}
        manager._send_multipart_locked = MagicMock()
        sender = object.__new__(MooncakeKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = 35
        sender.generation = 8
        sender.conclude_state = None

        self.assertFalse(sender.renew_decode_owner_lease())
        manager._send_multipart_locked.assert_not_called()

    def test_mooncake_poll_observes_terminal_before_expired_lease(self):
        receiver = object.__new__(MooncakeKVReceiver)
        receiver.bootstrap_room = 35
        receiver.generation = 8
        receiver.conclude_state = None
        receiver.kv_mgr = MagicMock()
        receiver.kv_mgr.check_status.return_value = KVPoll.Success
        receiver._check_waiting_timeout = MagicMock(return_value=KVPoll.Failed)

        self.assertEqual(receiver.poll(), KVPoll.Success)
        receiver._check_waiting_timeout.assert_not_called()

    def test_mooncake_abort_ack_waits_for_generation_attempts(self):
        manager = object.__new__(MooncakeKVManager)
        manager.request_status_lock = threading.RLock()
        manager._transfer_attempts = {(36, 9): 1}
        manager._deferred_ack_targets = {(36, 9): {("127.0.0.1", 5004)}}
        manager._source_cleanup_complete = set()
        manager._deferred_ack_retry = {}
        manager._send_abort_ack = MagicMock()

        manager._maybe_ack_drained_abort(36, 9)
        manager._send_abort_ack.assert_not_called()
        manager._finish_transfer_attempt(36, 9)
        manager._send_abort_ack.assert_not_called()
        manager.mark_source_cleanup_complete(36, 9)
        manager._send_abort_ack.assert_called_once_with("127.0.0.1", 5004, 36, 9)

    def test_mooncake_abort_ack_targets_are_not_overwritten(self):
        manager = object.__new__(MooncakeKVManager)
        manager.request_status_lock = threading.RLock()
        manager._transfer_attempts = {}
        manager._deferred_ack_targets = {}
        manager._source_cleanup_complete = set()
        manager._deferred_ack_retry = {}
        manager._send_abort_ack = MagicMock()
        manager.register_deferred_ack_target(41, 12, "127.0.0.1", 5001)
        manager.register_deferred_ack_target(41, 12, "127.0.0.2", 5002)
        manager.mark_source_cleanup_complete(41, 12)

        manager._maybe_ack_drained_abort(41, 12)

        manager._send_abort_ack.assert_has_calls(
            [
                unittest.mock.call("127.0.0.1", 5001, 41, 12),
                unittest.mock.call("127.0.0.2", 5002, 41, 12),
            ],
            any_order=True,
        )
        self.assertEqual(manager._send_abort_ack.call_count, 2)

    def test_mooncake_abort_ack_retries_only_failed_endpoints(self):
        manager = object.__new__(MooncakeKVManager)
        manager.request_status_lock = threading.RLock()
        manager._transfer_attempts = {}
        manager._deferred_ack_targets = {(43, 13): {("ok", 1), ("retry", 2)}}
        manager._source_cleanup_complete = {(43, 13)}
        manager._deferred_ack_retry = {}
        attempts = defaultdict(int)

        def send_ack(host, port, room, generation):
            endpoint = (host, port)
            attempts[endpoint] += 1
            return endpoint == ("ok", 1) or attempts[endpoint] > 1

        manager._send_abort_ack = MagicMock(side_effect=send_ack)

        manager._maybe_ack_drained_abort(43, 13)
        self.assertEqual(manager._deferred_ack_targets[(43, 13)], {("retry", 2)})
        manager._maybe_ack_drained_abort(43, 13)
        self.assertNotIn((43, 13), manager._deferred_ack_targets)

    def test_mooncake_abort_cleanup_latch_sends_ack_in_both_orders(self):
        for cleanup_first in (False, True):
            manager = object.__new__(MooncakeKVManager)
            manager.request_status_lock = threading.RLock()
            manager._transfer_attempts = {}
            manager._deferred_ack_targets = {}
            manager._source_cleanup_complete = set()
            manager._deferred_ack_retry = {}
            manager._send_abort_ack = MagicMock(return_value=True)
            if cleanup_first:
                manager.mark_source_cleanup_complete(44, 2)
            manager.register_deferred_ack_target(44, 2, "decode", 9000)
            manager._maybe_ack_drained_abort(44, 2)
            if not cleanup_first:
                manager.mark_source_cleanup_complete(44, 2)
            self.assertEqual(manager._send_abort_ack.call_count, 1)
            manager._send_abort_ack.assert_called_with("decode", 9000, 44, 2)

    def test_mooncake_abort_ack_retry_exhaustion_is_fail_stop(self):
        manager = object.__new__(MooncakeKVManager)
        manager.request_status_lock = threading.RLock()
        manager._transfer_attempts = {}
        manager._source_cleanup_complete = {(45, 3)}
        manager._deferred_ack_targets = {(45, 3): {("dead", 1)}}
        manager._deferred_ack_retry = {(45, 3): (0.0, 6)}
        manager._send_abort_ack = MagicMock(return_value=False)
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=1.0
        ):
            with self.assertRaises(RuntimeError):
                manager.retry_deferred_abort_acks()
        self.assertIn((45, 3), manager._deferred_ack_targets)
        self.assertIn((45, 3), manager._source_cleanup_complete)

    def test_decode_abort_notification_retries_failed_endpoint(self):
        receiver = object.__new__(MinimalCommonKVReceiver)
        receiver.bootstrap_room = 44
        receiver.generation = 14
        receiver.bootstrap_infos = [{"rank_ip": "prefill", "rank_port": 1}]
        receiver.kv_mgr = MagicMock(local_ip="decode")
        receiver.kv_mgr.rank_port = 2
        sock = MagicMock()
        sock.send_multipart.side_effect = [RuntimeError("temporary"), None]
        receiver._connect_to_bootstrap_server = MagicMock(
            return_value=(sock, threading.Lock())
        )

        self.assertFalse(receiver._send_abort_notification())
        self.assertTrue(receiver._send_abort_notification())

    def test_room_reuse_preserves_old_generation_drain_state(self):
        manager = make_manager()
        manager._transfer_attempts = {(42, 1): 1}
        manager._deferred_ack_targets = {(42, 1): {("127.0.0.1", 5003)}}
        generation, _ = manager.begin_request(42, KVPoll.Bootstrapping, 1)
        manager.update_status(42, KVPoll.Success, generation)

        new_generation, _ = manager.begin_request(42, KVPoll.Bootstrapping, 2)

        self.assertEqual(new_generation, 2)
        self.assertEqual(manager._transfer_attempts[(42, 1)], 1)
        self.assertEqual(manager._deferred_ack_targets[(42, 1)], {("127.0.0.1", 5003)})

    def test_mooncake_abort_ack_wire_shape(self):
        manager = object.__new__(MooncakeKVManager)
        manager.attn_tp_rank = 1
        manager.pp_size = 2
        manager.attn_cp_size = 2
        manager.pp_rank = 0
        manager.attn_cp_rank = 1
        manager._send_multipart_locked = MagicMock()

        manager._send_abort_ack("127.0.0.1", 5005, 37, 10)

        self.assertEqual(
            manager._send_multipart_locked.call_args.args[1],
            [b"ABORT_ACK", b"37", b"5", b"10"],
        )

    def test_await_transfer_futures_drains_after_first_failure(self):
        manager = object.__new__(MooncakeKVManager)
        failed = Future()
        running = Future()
        failed.set_result(7)
        running.set_running_or_notify_cancel()
        finished = threading.Event()
        result = []

        waiter = threading.Thread(
            target=lambda: (
                result.append(manager._await_transfer_futures([failed, running])),
                finished.set(),
            )
        )
        waiter.start()
        self.assertFalse(finished.wait(0.05))
        running.set_result(0)
        waiter.join(timeout=1)
        self.assertTrue(finished.is_set())
        self.assertEqual(result, [7])

    def test_await_transfer_futures_drains_before_reraising(self):
        manager = object.__new__(MooncakeKVManager)
        failed = Future()
        running = Future()
        failed.set_exception(RuntimeError("first failure"))
        running.set_running_or_notify_cancel()
        finished = threading.Event()
        errors = []

        def await_all():
            try:
                manager._await_transfer_futures([failed, running])
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()

        waiter = threading.Thread(target=await_all)
        waiter.start()
        self.assertFalse(finished.wait(0.05))
        running.set_result(0)
        waiter.join(timeout=1)
        self.assertTrue(finished.is_set())
        self.assertEqual(str(errors[0]), "first failure")

    def test_deferred_release_is_backend_capability_gated(self):
        self.assertFalse(CommonKVManager.supports_deferred_decode_kv_release)
        self.assertTrue(MooncakeKVManager.supports_deferred_decode_kv_release)


if __name__ == "__main__":
    unittest.main()
