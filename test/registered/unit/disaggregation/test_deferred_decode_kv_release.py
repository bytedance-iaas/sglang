"""Generation-safe decode ownership tests for aborts during KV transfer."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeRequest,
    DecodeTransferQueue,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _manager(room=41, generation=3):
    manager = CommonKVManager.__new__(CommonKVManager)
    manager.request_generation = {room: generation}
    manager.next_request_generation = {room: generation + 1}
    manager.request_status = {room: object()}
    manager.request_status_history = {}
    manager.request_failure_history = {}
    manager.request_bootstrap_activity = {}
    manager.request_status_lock = threading.RLock()
    manager._deferred_abort_ack_tracker = {}
    manager._deferred_abort_expected_ranks = {}
    return manager


class TestGenerationScopedAbortAcks(CustomTestCase):
    def test_all_expected_ranks_are_required_and_duplicates_do_not_count(self):
        manager = _manager()
        manager.register_deferred_abort_room(41, 3, expected_ranks={0, 1})
        manager.note_abort_ack(41, 0, 3)
        manager.note_abort_ack(41, 0, 3)
        self.assertFalse(manager.is_abort_release_safe(41, 3))
        manager.note_abort_ack(41, 1, 3)
        self.assertTrue(manager.is_abort_release_safe(41, 3))

    def test_stale_generation_ack_cannot_release_reused_room(self):
        manager = _manager()
        manager.register_deferred_abort_room(41, 3, expected_ranks={0, 1})
        manager.note_abort_ack(41, 0, 3)
        manager.clear_deferred_abort_state(41, 3)
        manager.request_generation[41] = 4
        manager.register_deferred_abort_room(41, 4, expected_ranks={0, 1})
        manager.note_abort_ack(41, 1, 3)
        manager.note_abort_ack(41, 0, 4)
        self.assertFalse(manager.is_abort_release_safe(41, 4))
        manager.note_abort_ack(41, 1, 4)
        self.assertTrue(manager.is_abort_release_safe(41, 4))


class _Manager:
    def __init__(self, events):
        self.events = events
        self.supports_request_generation = True
        self.supports_deferred_decode_kv_release = True
        self.enable_deferred_decode_kv_release = True
        self.expected = {}
        self.acks = {}

    def register_deferred_abort_room(self, room, generation, expected_ranks):
        self.events.append("arm")
        key = (room, generation)
        self.expected[key] = set(expected_ranks)
        self.acks[key] = set()
        return True

    def is_abort_release_safe(self, room, generation):
        key = (room, generation)
        return self.expected.get(key, set()).issubset(self.acks.get(key, set()))

    def clear_deferred_abort_state(self, room, generation):
        self.events.append("clear-tracker")
        self.expected.pop((room, generation), None)
        self.acks.pop((room, generation), None)


class _Receiver:
    def __init__(self, manager, events, generation=3):
        self.kv_mgr = manager
        self.generation = generation
        self.expected_prefill_ranks = {0, 1}
        self.abort_notified = False
        self.events = events
        self.abort_attempts = 0
        self.abort_succeeds = True

    def abort(self):
        self.abort_attempts += 1
        self.events.append("abort")
        if self.abort_succeeds:
            self.abort_notified = True

    def clear(self):
        self.events.append("clear-receiver")


class _Staging:
    def __init__(self, events, enabled):
        self.events = events
        self.enabled = enabled

    def is_staging_room(self, room):
        return self.enabled

    def unregister_decode_req(self, room):
        self.events.append("staging")


def _queue_and_entry(*, staging=True, hisparse=True):
    events = []
    manager = _Manager(events)
    receiver = _Receiver(manager, events)
    req = SimpleNamespace(
        rid="request-child",
        bootstrap_room=41,
        return_logprob=False,
        finished_output=False,
        finished_reason=None,
        req_pool_idx=2,
        kv=object(),
        mamba_pool_idx=None,
    )
    entry = DecodeRequest(
        req=req,
        kv_receiver=receiver,
        metadata_buffer_index=7,
        metadata_sent=True,
    )
    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue.queue = [entry]
    queue.deferred_abort_holds = []
    queue.enable_deferred_kv_release = True
    queue.deferred_kv_release_timeout = 30.0
    queue.enable_staging = staging
    queue.staging_handler = _Staging(events, staging)
    queue.metadata_buffers = SimpleNamespace(bootstrap_room={7: 41})
    queue.req_to_metadata_buffer_idx_allocator = MagicMock()
    queue.req_to_metadata_buffer_idx_allocator.free.side_effect = (
        lambda _: events.append("metadata")
    )
    queue.tp_rank = 0
    queue.tree_cache = object()
    queue._clean_hicache_prefetch_resources = MagicMock(
        side_effect=lambda _: events.append("hicache")
    )
    queue.scheduler = SimpleNamespace(
        enable_hisparse=hisparse,
        finish_hisparse_request=MagicMock(
            side_effect=lambda _: events.append("hisparse")
        ),
        output_streamer=MagicMock(),
        server_args=SimpleNamespace(),
        ps=SimpleNamespace(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0),
    )
    return queue, entry, manager, receiver, events


class TestDeferredDecodeRelease(CustomTestCase):
    @patch.object(decode_mod, "prepare_abort")
    def test_real_pop_preallocated_preflight_has_no_admission_side_effects(
        self, prepare_abort
    ):
        """Drive pop_preallocated itself, not only the preflight helper."""
        req = SimpleNamespace(
            rid="legacy-pop",
            finished_reason=None,
            waiting_for_input=True,
            return_logprob=False,
            priority=0,
            origin_input_ids=[1, 2],
            output_ids=[],
            sampling_params=SimpleNamespace(max_new_tokens=1),
        )
        receiver = SimpleNamespace(
            generation=None,
            expected_prefill_ranks=None,
            kv_mgr=SimpleNamespace(),
        )
        entry = DecodeRequest(req=req, kv_receiver=receiver, waiting_for_input=True)
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [entry]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.held_rebootstrap_reqs = []
        queue.transfer_queue = SimpleNamespace(enable_deferred_kv_release=True)
        queue.scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                disaggregation_decode_max_inflight_transfers=None,
                disaggregation_decode_admission_policy="fifo",
                disaggregation_decode_admission_max_bypasses=8,
            ),
            running_batch=SimpleNamespace(reqs=[]),
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            enable_hisparse=False,
        )
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: 8)
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 8, alloc=MagicMock()
        )
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue.num_reserved_decode_tokens = 0
        queue.max_total_num_tokens = 1024
        queue._allocatable_token_budgets = MagicMock(return_value=1024)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._renew_bootstrap_leases = MagicMock()
        queue._abort_and_release = MagicMock()
        queue._pre_alloc = MagicMock()

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [entry])
        queue._pre_alloc.assert_not_called()
        queue.req_to_metadata_buffer_idx_allocator.alloc.assert_not_called()
        queue._abort_and_release.assert_not_called()
        prepare_abort.assert_called_once()
        self.assertTrue(entry.deferred_preflight_failed)

    def test_admission_preflight_rejects_missing_generation_or_expected_ranks(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.transfer_queue = SimpleNamespace(
            enable_deferred_kv_release=True,
            _supports_deferred_abort=MagicMock(return_value=True),
        )
        manager = SimpleNamespace(
            supports_request_generation=True,
            supports_deferred_decode_kv_release=True,
            enable_deferred_decode_kv_release=True,
        )
        receiver = SimpleNamespace(
            kv_mgr=manager, generation=3, expected_prefill_ranks={0, 1}
        )
        entry = DecodeRequest(req=SimpleNamespace(rid="r"), kv_receiver=receiver)
        for attr, value in (("generation", None), ("expected_prefill_ranks", set())):
            setattr(receiver, attr, value)
            self.assertFalse(queue._deferred_admission_preflight(entry))
            setattr(receiver, attr, 3 if attr == "generation" else {0, 1})

    def test_admission_preflight_rejects_missing_manager_capability(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.transfer_queue = SimpleNamespace(
            enable_deferred_kv_release=True,
            _supports_deferred_abort=MagicMock(return_value=True),
        )
        receiver = SimpleNamespace(
            kv_mgr=SimpleNamespace(), generation=3, expected_prefill_ranks={0}
        )
        entry = DecodeRequest(req=SimpleNamespace(rid="r"), kv_receiver=receiver)
        self.assertFalse(queue._deferred_admission_preflight(entry))

    def test_preflight_failure_does_not_call_allocation_or_publication(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.transfer_queue = SimpleNamespace(enable_deferred_kv_release=True)
        receiver = SimpleNamespace(kv_mgr=SimpleNamespace(), generation=None)
        receiver.send_metadata = MagicMock()
        entry = DecodeRequest(req=SimpleNamespace(rid="r"), kv_receiver=receiver)
        queue._pre_alloc = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(alloc=MagicMock())
        self.assertFalse(queue._deferred_admission_preflight(entry))
        entry.deferred_preflight_failed = True
        queue._pre_alloc.assert_not_called()
        queue.req_to_metadata_buffer_idx_allocator.alloc.assert_not_called()
        receiver.send_metadata.assert_not_called()

    def test_backend_capability_gate_rejects_unsupported_manager(self):
        queue, _, manager, _, _ = _queue_and_entry(staging=False)
        manager.supports_deferred_decode_kv_release = False

        with self.assertRaisesRegex(RuntimeError, "lacks.*abort/ACK"):
            queue.bind_kv_manager_capabilities(manager)

        self.assertFalse(queue.enable_deferred_kv_release)

    @patch.object(decode_mod, "release_kv_cache")
    def test_external_abort_removes_active_owner_but_holds_all_buffers(self, release):
        queue, entry, manager, receiver, events = _queue_and_entry()
        removed = queue.remove_aborted("request")
        self.assertEqual(removed, [entry])
        self.assertEqual(queue.queue, [])
        self.assertIs(queue.deferred_abort_holds[0].decode_req, entry)
        self.assertEqual(queue.deferred_abort_holds[0].expected_prefill_ranks, {0, 1})
        self.assertEqual(events[:2], ["arm", "abort"])
        release.assert_not_called()
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [entry.req], False
        )

    @patch.object(decode_mod, "release_kv_cache")
    def test_all_rank_ack_releases_every_resource_once_in_order(self, release):
        queue, entry, manager, receiver, events = _queue_and_entry()
        release.side_effect = lambda *_args, **_kwargs: events.append("kv")
        queue.remove_aborted("request")
        queue.resolve_deferred_abort_holds()
        release.assert_not_called()
        manager.acks[(41, 3)].update({0, 1})
        queue.resolve_deferred_abort_holds()
        queue.resolve_deferred_abort_holds()
        self.assertEqual(queue.deferred_abort_holds, [])
        self.assertEqual(
            events,
            [
                "arm",
                "abort",
                "staging",
                "hisparse",
                "hicache",
                "metadata",
                "kv",
                "clear-tracker",
                "clear-receiver",
            ],
        )
        release.assert_called_once_with(entry.req, queue.tree_cache, is_insert=False)
        self.assertIsNone(entry.kv_receiver)
        self.assertEqual(entry.metadata_buffer_index, -1)

    @patch.object(decode_mod, "release_kv_cache")
    def test_second_abort_does_not_rearm_or_emit_again(self, release):
        queue, entry, manager, _, events = _queue_and_entry(staging=False)
        queue.remove_aborted("request")
        queue.remove_aborted("request")
        self.assertEqual(events.count("arm"), 1)
        self.assertEqual(len(queue.deferred_abort_holds), 1)
        queue.scheduler.output_streamer.stream_output.assert_called_once()
        release.assert_not_called()

    @patch.object(decode_mod, "release_kv_cache")
    def test_deferred_abort_retries_partial_notification_without_resetting_deadline(
        self, release
    ):
        queue, entry, manager, receiver, events = _queue_and_entry(staging=False)
        receiver.abort_succeeds = False
        queue.remove_aborted("request")
        hold = queue.deferred_abort_holds[0]
        deadline = hold.abort_drain_deadline
        receiver.abort_succeeds = True
        queue.resolve_deferred_abort_holds()
        self.assertEqual(receiver.abort_attempts, 2)
        self.assertEqual(hold.abort_drain_deadline, deadline)
        self.assertEqual(queue.deferred_abort_holds, [hold])
        release.assert_not_called()

    @patch.object(decode_mod, "release_kv_cache")
    @patch.object(decode_mod, "poll_and_all_reduce")
    def test_metadata_send_started_failure_moves_to_deferred_hold(self, poll, release):
        queue, entry, manager, receiver, events = _queue_and_entry(staging=False)
        entry.metadata_sent = False
        entry.metadata_publication_started = True
        entry.abort_tracker_armed = True
        manager.register_deferred_abort_room(41, 3, {0, 1})
        queue.gloo_group = object()
        queue.scheduler.enable_decode_hicache = False
        queue.scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        poll.return_value = [decode_mod.KVPoll.Failed]
        receiver.failure_exception = lambda: None

        self.assertEqual(queue.pop_transferred(), [])
        self.assertTrue(queue.has_pending_deferred_abort_holds())
        release.assert_not_called()
        self.assertNotIn("metadata", events)

    @patch.object(decode_mod, "release_kv_cache")
    def test_timeout_is_fatal_and_does_not_release_without_ack(self, release):
        queue, entry, _, _, _ = _queue_and_entry(staging=False, hisparse=False)
        queue.remove_aborted("request")
        hold = queue.deferred_abort_holds[0]
        hold.abort_drain_deadline = float("-inf")
        with self.assertRaisesRegex(RuntimeError, "Fatal deferred decode abort"):
            queue.resolve_deferred_abort_holds()
        release.assert_not_called()
        self.assertEqual(len(queue.deferred_abort_holds), 1)

    @patch.object(decode_mod, "release_kv_cache")
    def test_teardown_rejects_unquiesced_owner_and_retains_tracker(self, release):
        queue, entry, manager, _, _ = _queue_and_entry(staging=False, hisparse=False)
        queue.remove_aborted("request")
        with self.assertRaisesRegex(RuntimeError, "pending deferred abort holds"):
            queue.release_memory_occupation()
        self.assertEqual(len(queue.deferred_abort_holds), 1)
        self.assertEqual(manager.expected, {(41, 3): {0, 1}})
        release.assert_not_called()

    @patch.object(decode_mod, "release_kv_cache")
    @patch.object(decode_mod, "poll_and_all_reduce")
    def test_metadata_published_failure_moves_to_hold_before_cleanup(
        self, poll, release
    ):
        queue, entry, manager, receiver, events = _queue_and_entry(staging=False)
        entry.abort_tracker_armed = True
        manager.register_deferred_abort_room(41, 3, {0, 1})
        queue.gloo_group = object()
        queue.scheduler.enable_decode_hicache = False
        queue.scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        poll.return_value = [decode_mod.KVPoll.Failed]
        receiver.failure_exception = lambda: None

        self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(queue.has_pending_deferred_abort_holds())
        self.assertTrue(receiver.abort_notified)
        self.assertEqual(events.count("abort"), 1)
        release.assert_not_called()
        self.assertNotIn("metadata", events)
        self.assertNotIn("staging", events)
        self.assertNotIn("hicache", events)
        self.assertNotIn("hisparse", events)

    @patch.object(decode_mod, "release_kv_cache")
    @patch.object(decode_mod, "poll_and_all_reduce")
    def test_abort_notified_failure_still_moves_to_hold(self, poll, release):
        queue, entry, manager, receiver, events = _queue_and_entry(staging=False)
        receiver.abort_notified = True
        entry.abort_tracker_armed = True
        manager.register_deferred_abort_room(41, 3, {0, 1})
        queue.gloo_group = object()
        queue.scheduler.enable_decode_hicache = False
        queue.scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        poll.return_value = [decode_mod.KVPoll.Failed]
        receiver.failure_exception = lambda: None

        self.assertEqual(queue.pop_transferred(), [])

        self.assertEqual(queue.queue, [])
        self.assertTrue(queue.has_pending_deferred_abort_holds())
        self.assertEqual(events.count("abort"), 0)
        release.assert_not_called()
        self.assertNotIn("metadata", events)
        self.assertNotIn("hicache", events)
        self.assertNotIn("hisparse", events)

    @patch.object(decode_mod, "release_kv_cache")
    @patch.object(decode_mod, "poll_and_all_reduce")
    def test_metadata_published_failure_without_capability_fails_closed(
        self, poll, release
    ):
        queue, entry, manager, receiver, events = _queue_and_entry(staging=False)
        manager.supports_deferred_decode_kv_release = False
        queue.gloo_group = object()
        queue.scheduler.enable_decode_hicache = False
        queue.scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        poll.return_value = [decode_mod.KVPoll.Failed]
        receiver.failure_exception = lambda: None

        with self.assertRaisesRegex(RuntimeError, "does not support"):
            queue.pop_transferred()

        self.assertEqual(queue.queue, [entry])
        self.assertEqual(queue.deferred_abort_holds, [])
        release.assert_not_called()
        self.assertNotIn("metadata", events)
        self.assertNotIn("staging", events)
        self.assertNotIn("hicache", events)
        self.assertNotIn("hisparse", events)

    @patch.object(decode_mod, "release_kv_cache")
    def test_non_leader_marks_output_done_without_external_stream(self, release):
        queue, entry, _, _, events = _queue_and_entry(staging=False)
        queue.scheduler.ps.pp_rank = 1
        queue.remove_aborted("request")
        self.assertTrue(entry.abort_output_emitted)
        self.assertTrue(entry.req.finished_output)
        queue.scheduler.output_streamer.stream_output.assert_not_called()
        self.assertEqual(events.count("arm"), 1)
        release.assert_not_called()


if __name__ == "__main__":
    unittest.main()
