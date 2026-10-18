"""Unit tests for the deferred decode-side KV release mechanism.

When a decode request is aborted while its prefill->decode KV transfer may still
be in flight, the decode side holds its KV pages / req-slot instead of freeing
them immediately (which could let the still-in-flight write land on pages already
reused by another request). The pages are released once every prefill rank acks
that its transfer drained (CommonKVManager.is_abort_release_safe). A timeout
reports a recovery gap, not inert writes. See resolve_deferred_releases.
"""

import asyncio
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    PrefillRankInfo,
    prefill_abort_ack_rank,
)
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.nixl.conn import NixlKVManager, NixlKVReceiver
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_manager():
    """A bare CommonKVManager carrying only the deferred-ack state the helpers
    touch (avoids the heavy real __init__)."""
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr._deferred_abort_ack_tracker = {}
    return mgr


class TestRebootstrapPrefillBinding(CustomTestCase):
    def test_init_normalizes_numpy_rank_and_supports_fake_backend(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        req = SimpleNamespace(pd_bound_prefill_dp_rank=None)
        receiver = SimpleNamespace(conclude_state=None, init=MagicMock())
        queue._init_receiver(
            SimpleNamespace(req=req, kv_receiver=receiver), np.int64(0)
        )
        receiver.init.assert_called_once_with(0)
        self.assertIs(type(receiver.init.call_args.args[0]), int)
        self.assertIs(type(req.pd_bound_prefill_dp_rank), int)
        self.assertEqual(req.pd_bound_prefill_dp_rank, 0)
        for rank in (True, np.bool_(False), -1, "2"):
            with self.subTest(rank=rank):
                with self.assertRaises(ValueError):
                    queue._init_receiver(
                        SimpleNamespace(req=req, kv_receiver=receiver), rank
                    )
        receiver.init.assert_called_once()

    def test_resolved_binding_survives_receiver_recreation(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.kv_manager = SimpleNamespace(prefill_info_table={})
        req = SimpleNamespace(
            pd_rebootstrap_in_progress=False,
            pd_bound_prefill_dp_rank=None,
            disagg_prefill_dp_rank=None,
            bootstrap_room=5,
        )
        receiver = SimpleNamespace(conclude_state=None)

        def initialize(rank):
            receiver.prefill_dp_rank = rank

        receiver.init = initialize
        queue._init_receiver(SimpleNamespace(req=req, kv_receiver=receiver), 2)
        self.assertEqual(req.pd_bound_prefill_dp_rank, 2)
        # The original room modulo / client hint disagree, and topology cache
        # may have expired. Neither is used to reselect an already-bound DP.
        req.pd_rebootstrap_in_progress = True
        req.disagg_prefill_dp_rank = 3
        self.assertEqual(queue._resolve_prefill_dp_rank(req), 2)
        replacement = SimpleNamespace(conclude_state=None)
        replacement.init = lambda rank: setattr(replacement, "prefill_dp_rank", rank)
        queue._init_receiver(
            SimpleNamespace(req=req, kv_receiver=replacement),
            queue._resolve_prefill_dp_rank(req),
        )
        self.assertEqual(replacement.prefill_dp_rank, 2)
        self.assertEqual(req.pd_bound_prefill_dp_rank, 2)

    def test_failed_init_does_not_overwrite_prior_binding(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        req = SimpleNamespace(pd_bound_prefill_dp_rank=2)
        receiver = SimpleNamespace(
            init=MagicMock(), conclude_state=KVPoll.Failed, prefill_dp_rank=3
        )
        queue._init_receiver(SimpleNamespace(req=req, kv_receiver=receiver), 3)
        self.assertEqual(req.pd_bound_prefill_dp_rank, 2)

    def test_add_pins_fast_fake_and_rebootstrap_paths(self):
        for mode in ("fast", "fake", "rebootstrap"):
            with self.subTest(mode=mode):
                queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
                queue._check_if_req_exceed_kv_capacity = MagicMock(return_value=False)
                queue._resolve_prefill_dp_rank = MagicMock(return_value=3)
                queue.pending_reqs = []
                req = SimpleNamespace(
                    pd_bound_prefill_dp_rank=2 if mode == "rebootstrap" else None
                )
                receiver = SimpleNamespace(conclude_state=None)
                receiver.init = lambda rank, target=receiver: setattr(
                    target, "prefill_dp_rank", rank
                )
                decode_req = SimpleNamespace(req=req, kv_receiver=receiver)
                queue._create_receiver_and_enqueue = MagicMock(return_value=decode_req)
                with patch.object(
                    decode_mod, "_is_fake_transfer", return_value=mode == "fake"
                ):
                    queue.add(req, is_rebootstrap=mode == "rebootstrap")
                expected = {"fast": 3, "fake": 0, "rebootstrap": 2}[mode]
                self.assertEqual(req.pd_bound_prefill_dp_rank, expected)
                self.assertEqual(receiver.prefill_dp_rank, expected)
                self.assertEqual(queue.pending_reqs, [])
                if mode == "rebootstrap":
                    queue._resolve_prefill_dp_rank.assert_not_called()

    def test_missing_binding_aborts_before_receiver_construction(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._check_if_req_exceed_kv_capacity = MagicMock(return_value=False)
        queue._create_receiver_and_enqueue = MagicMock()
        queue.scheduler = MagicMock()
        req = SimpleNamespace(
            rid="missing-binding", return_logprob=False, pd_bound_prefill_dp_rank=None
        )
        with patch.object(decode_mod, "prepare_abort") as abort:
            queue.add(req, is_rebootstrap=True)
        abort.assert_called_once()
        queue._create_receiver_and_enqueue.assert_not_called()
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], False
        )

    def test_payload_uses_bound_prefill_rank_not_decode_route(self):
        req = Req(
            rid="bound-payload",
            origin_input_text="",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(),
            routed_dp_rank=7,
            disagg_prefill_dp_rank=None,
        )
        for rank in (0, 2):
            with self.subTest(rank=rank):
                req.pd_bound_prefill_dp_rank = rank
                payload = req.build_rebootstrap_payload()
                self.assertEqual(payload["routed_dp_rank"], rank)
                self.assertEqual(payload["disagg_prefill_dp_rank"], rank)
                self.assertEqual(req.routed_dp_rank, 7)
                self.assertEqual(payload["pd_transfer_epoch"], 0)
        req.pd_bound_prefill_dp_rank = None
        with self.assertRaisesRegex(ValueError, "initialized prefill DP binding"):
            req.build_rebootstrap_payload()

    def test_executor_post_pins_actual_receiver_without_mutating_payload(self):
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr.attn_tp_rank = mgr.attn_cp_rank = mgr.pp_rank = 0
        mgr._resolve_rebootstrap_prefill_url = MagicMock(return_value="http://prefill")
        executor = MagicMock()
        mgr._ensure_prefill_recompute_executor = MagicMock(return_value=executor)
        receiver = SimpleNamespace(prefill_dp_rank=2)
        payload = {"rid": "recompute", "routed_dp_rank": 7, "disagg_prefill_dp_rank": 3}
        mgr.submit_prefill_recompute(receiver, payload)
        submitted = executor.submit.call_args.args[3]
        self.assertEqual(submitted["routed_dp_rank"], 2)
        self.assertEqual(submitted["disagg_prefill_dp_rank"], 2)
        self.assertEqual(payload["routed_dp_rank"], 7)
        self.assertEqual(payload["disagg_prefill_dp_rank"], 3)
        mgr.pp_rank = 1
        mgr.submit_prefill_recompute(receiver, payload)
        executor.submit.assert_called_once()

    def test_recompute_payload_preserves_incremented_transfer_epoch(self):
        req = Req(
            rid="epoch-payload",
            origin_input_text="",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(),
            bootstrap_room=7,
            pd_transfer_epoch=4,
        )
        req.pd_bound_prefill_dp_rank = 0
        self.assertEqual(req.build_rebootstrap_payload()["pd_transfer_epoch"], 4)

    def test_executor_missing_binding_fails_before_http_dispatch(self):
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr.attn_tp_rank = mgr.attn_cp_rank = mgr.pp_rank = 0
        mgr._fail_prefill_recompute = MagicMock()
        mgr._resolve_rebootstrap_prefill_url = MagicMock()
        mgr._ensure_prefill_recompute_executor = MagicMock()
        for rank in (None, -1, True):
            with self.subTest(rank=rank):
                receiver = SimpleNamespace(prefill_dp_rank=rank)
                mgr.submit_prefill_recompute(receiver, {"rid": "missing-binding"})
                self.assertEqual(
                    mgr._fail_prefill_recompute.call_args.args[0], receiver
                )
        mgr._resolve_rebootstrap_prefill_url.assert_not_called()
        mgr._ensure_prefill_recompute_executor.assert_not_called()


class TestAbortAckAggregation(CustomTestCase):
    def test_release_safe_only_after_all_required_ranks_ack(self):
        mgr = _make_manager()
        room = 100
        mgr.register_deferred_abort_room(room)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))

        mgr.note_abort_ack(room, 0)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))

        mgr.note_abort_ack(room, 1)
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=2))

    def test_duplicate_rank_ack_does_not_over_count(self):
        mgr = _make_manager()
        room = 101
        mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0)
        mgr.note_abort_ack(room, 0)  # same rank twice
        # Two acks arrived but from one rank: not safe for a 2-rank prefill.
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))

    def test_single_rank_fast_path(self):
        mgr = _make_manager()
        room = 102
        mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0)
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=1))

    def test_clear_deferred_abort_state(self):
        mgr = _make_manager()
        room = 103
        mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0)
        mgr.clear_deferred_abort_state(room)
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=1))

    def test_ack_before_register_is_dropped(self):
        # An ack for a room that isn't actively held must not be recorded (it
        # would otherwise pollute a later request reusing the same room).
        mgr = _make_manager()
        room = 104
        mgr.note_abort_ack(room, 0)  # no register yet
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=1))

    def test_late_ack_after_release_does_not_pollute_reused_room(self):
        # Regression for bootstrap_room reuse: req A (room R) releases, then a
        # late ack from A arrives, then req B reuses room R. B must start from a
        # clean slate and not inherit A's ack (which would release B early while
        # its transfer is still in flight -> KV corruption).
        mgr = _make_manager()
        room = 105

        # Req A: held, one of two ranks acks, then released (e.g. timed out).
        mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0)
        mgr.clear_deferred_abort_state(room)

        # Late ack from A's other rank arrives after release -> dropped.
        mgr.note_abort_ack(room, 1)
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)

        # Req B reuses room R.
        mgr.register_deferred_abort_room(room)
        # Only B's rank-0 has acked so far; a 2-rank prefill is NOT safe yet.
        mgr.note_abort_ack(room, 0)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))
        mgr.note_abort_ack(room, 1)
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=2))

    def test_register_resets_stale_acks(self):
        mgr = _make_manager()
        room = 106
        mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0)
        mgr.note_abort_ack(room, 1)
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=2))
        # Re-registering (a later reuse) wipes the prior acks.
        mgr.register_deferred_abort_room(room)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))


class TestReceiverArmBeforeAbort(CustomTestCase):
    def test_bootstrap_registration_publishes_the_actual_ack_sender_identity(self):
        async def register():
            server = CommonKVBootstrapServer.__new__(CommonKVBootstrapServer)
            for name in (
                "attn_tp_size",
                "attn_cp_size",
                "dp_size",
                "pp_size",
                "page_size",
                "kv_cache_dtype",
                "prefill_http_port",
                "follow_bootstrap_room",
                "enable_dsa_cache_layer_split",
            ):
                setattr(server, name, None)
            server.lock = asyncio.Lock()
            server.prefill_port_table = {}
            server._registered_count = 0
            seen = set()
            for tp_rank in range(2):
                for cp_rank in range(2):
                    for pp_rank in range(2):
                        payload = {
                            "attn_tp_size": 2,
                            "attn_tp_rank": tp_rank,
                            "attn_cp_size": 2,
                            "attn_cp_rank": cp_rank,
                            "attn_dp_size": 1,
                            "attn_dp_rank": 0,
                            "pp_size": 2,
                            "pp_rank": pp_rank,
                            "system_dp_size": 1,
                            "system_dp_rank": 0,
                            "rank_ip": "10.0.0.1",
                            "rank_port": 6000 + len(seen),
                            "page_size": 64,
                            "kv_cache_dtype": "fp8",
                        }
                        request = SimpleNamespace(json=AsyncMock(return_value=payload))
                        response = await server._handle_route_put(request)
                        self.assertEqual(response.status, 200)
                        info = server.prefill_port_table[0][cp_rank][tp_rank][pp_rank]
                        manager = CommonKVManager.__new__(CommonKVManager)
                        manager.attn_tp_rank, manager.attn_cp_rank = tp_rank, cp_rank
                        manager.pp_rank, manager.attn_cp_size, manager.pp_size = (
                            pp_rank,
                            2,
                            2,
                        )
                        self.assertEqual(
                            info.abort_ack_rank, manager._prefill_unique_rank()
                        )
                        seen.add(info.abort_ack_rank)
            self.assertEqual(seen, set(range(8)))
            for tp_rank in range(2):
                for cp_rank in range(2):
                    for pp_rank in range(2):
                        request = SimpleNamespace(
                            query={
                                "prefill_dp_rank": "0",
                                "prefill_cp_rank": str(cp_rank),
                                "target_tp_rank": str(tp_rank),
                                "target_pp_rank": str(pp_rank),
                            }
                        )
                        response = await server._handle_route_get(request)
                        self.assertEqual(response.status, 200)
                        returned = json.loads(response.text)
                        self.assertEqual(
                            returned["abort_ack_rank"],
                            prefill_abort_ack_rank(tp_rank, cp_rank, pp_rank, 2, 2),
                        )

        asyncio.run(register())

    def test_invalid_rank_coordinates_do_not_generate_drain_identity(self):
        for coords in (
            (0, -1, 0, 8, 1),
            (0, 8, 0, 8, 1),
            (0, 0, 1, 8, 1),
            (0, 0, 0, 0, 1),
            (-1, 0, 0, 8, 1),
            (True, 0, 0, 8, 1),
        ):
            with self.subTest(coords=coords), self.assertRaises(ValueError):
                prefill_abort_ack_rank(*coords)

    def test_registered_ack_rank_survives_serialization_and_missing_identity_is_unknown(
        self,
    ):
        import dataclasses

        info = PrefillRankInfo("10.0.0.1", 6000, abort_ack_rank=23)
        self.assertEqual(dataclasses.asdict(info)["abort_ack_rank"], 23)
        legacy = PrefillRankInfo("10.0.0.1", 6000)
        self.assertIsNone(legacy.abort_ack_rank)

    def test_expected_ack_rank_set_includes_dummy_and_pp_senders(self):
        receiver, _, _ = self._receiver()
        receiver.bootstrap_infos = [
            {"rank_ip": "10.0.0.1", "rank_port": 6000, "abort_ack_rank": 3},
            {
                "rank_ip": "10.0.0.1",
                "rank_port": 6001,
                "abort_ack_rank": 11,
                "is_dummy": True,
            },
        ]
        self.assertEqual(receiver.expected_abort_ack_ranks(), {3, 11})

    def test_missing_duplicate_or_ambiguous_bootstrap_identities_are_unknown(self):
        receiver, _, _ = self._receiver()
        good = {"rank_ip": "10.0.0.1", "rank_port": 6000, "abort_ack_rank": 3}
        cases = [
            None,
            [],
            [{"rank_ip": "10.0.0.1", "rank_port": 6000}],
            [good, {"rank_ip": "10.0.0.1", "rank_port": 6001, "abort_ack_rank": 3}],
            [good, {"rank_ip": "10.0.0.1", "rank_port": 6000, "abort_ack_rank": 11}],
            [{"rank_ip": "10.0.0.1", "rank_port": 6000, "abort_ack_rank": -1}],
            [{"rank_ip": "10.0.0.1", "rank_port": 6000, "abort_ack_rank": True}],
        ]
        for infos in cases:
            with self.subTest(infos=infos):
                receiver.bootstrap_infos = infos
                self.assertIsNone(receiver.expected_abort_ack_ranks())

    def _receiver(self, *, published=True, enabled=True):
        mgr = _make_manager()
        mgr.enable_deferred_decode_kv_release = enabled
        mgr.local_ip, mgr.rank_port = "10.0.0.4", 6001
        mgr.record_failure = MagicMock()
        mgr.update_status = MagicMock()
        receiver = CommonKVReceiver.__new__(CommonKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_room = 107
        receiver.init_time = 1.0 if published else None
        receiver.abort_notified = False
        receiver.conclude_state = None
        receiver.bootstrap_infos = [{"rank_ip": "10.0.0.5", "rank_port": 6002}]
        receiver.invalidate_cached_bootstrap_infos = MagicMock()
        socket = MagicMock()
        receiver._connect_to_bootstrap_server = MagicMock(
            return_value=(socket, threading.Lock())
        )
        return receiver, mgr, socket

    def test_immediate_ack_is_counted_and_redundant_abort_does_not_reset_it(self):
        receiver, mgr, socket = self._receiver()

        def immediate_ack(frames):
            self.assertIn(receiver.bootstrap_room, mgr._deferred_abort_ack_tracker)
            self.assertEqual(frames[0], b"ABORT")
            mgr.note_abort_ack(receiver.bootstrap_room, 0)

        socket.send_multipart.side_effect = immediate_ack
        receiver.abort()
        receiver.abort()
        socket.send_multipart.assert_called_once()
        self.assertTrue(mgr.is_abort_release_safe(receiver.bootstrap_room, 1))

    def test_unpublished_prealloc_abort_does_not_leak_tracker(self):
        receiver, mgr, socket = self._receiver(published=False)
        receiver.abort()
        socket.send_multipart.assert_called_once()
        self.assertEqual(mgr._deferred_abort_ack_tracker, {})

    def test_disabled_deferred_release_keeps_tracker_empty(self):
        receiver, mgr, socket = self._receiver(enabled=False)
        receiver.abort()
        socket.send_multipart.assert_called_once()
        self.assertEqual(mgr._deferred_abort_ack_tracker, {})

    def test_partial_publication_forces_arm_without_overwriting_failure(self):
        receiver, mgr, socket = self._receiver(published=False)
        receiver.destination_metadata_published = True
        mgr.failure_records = {receiver.bootstrap_room: "partial metadata send failed"}

        def immediate_ack(_frames):
            self.assertIn(receiver.bootstrap_room, mgr._deferred_abort_ack_tracker)
            mgr.note_abort_ack(receiver.bootstrap_room, 0)

        socket.send_multipart.side_effect = immediate_ack
        self.assertTrue(receiver.requires_abort_drain())
        receiver.ensure_abort_notified(force_arm=True)
        receiver.ensure_abort_notified(force_arm=True)
        socket.send_multipart.assert_called_once()
        mgr.record_failure.assert_not_called()
        self.assertEqual(
            mgr.failure_records[receiver.bootstrap_room], "partial metadata send failed"
        )
        self.assertTrue(mgr.is_abort_release_safe(receiver.bootstrap_room, 1))

    def test_force_arm_after_unarmed_notification_resends_once(self):
        receiver, mgr, socket = self._receiver(published=False)
        receiver.ensure_abort_notified()
        self.assertEqual(mgr._deferred_abort_ack_tracker, {})
        receiver.ensure_abort_notified(force_arm=True)
        mgr.note_abort_ack(receiver.bootstrap_room, 0)
        receiver.ensure_abort_notified(force_arm=True)
        self.assertEqual(socket.send_multipart.call_count, 2)
        self.assertTrue(mgr.is_abort_release_safe(receiver.bootstrap_room, 1))

    def test_publication_unknown_is_not_proof_of_no_destination_writes(self):
        receiver, _, _ = self._receiver(published=False)
        self.assertTrue(receiver.requires_abort_drain())
        receiver.destination_metadata_published = False
        self.assertFalse(receiver.requires_abort_drain())

    def test_waiting_timeout_arms_before_send_and_preserves_timeout_cause(self):
        receiver, mgr, socket = self._receiver()
        mgr.waiting_timeout = 5

        def immediate_ack(_frames):
            self.assertIn(receiver.bootstrap_room, mgr._deferred_abort_ack_tracker)
            mgr.note_abort_ack(receiver.bootstrap_room, 0)

        socket.send_multipart.side_effect = immediate_ack
        with patch("sglang.srt.disaggregation.common.conn.time.time", return_value=10):
            self.assertEqual(
                receiver._check_waiting_timeout(), decode_mod.KVPoll.Failed
            )
        mgr.record_failure.assert_called_once()
        self.assertIn("timed out", mgr.record_failure.call_args.args[1])
        self.assertTrue(mgr.is_abort_release_safe(receiver.bootstrap_room, 1))


class _FakeIdxAllocator:
    def __init__(self):
        self.freed = []

    def free(self, idx):
        self.freed.append(idx)


def _make_queue(timeout=30.0):
    q = DecodeTransferQueue.__new__(DecodeTransferQueue)
    q._deferred_releases = []
    q.queue = []
    q.scheduler = SimpleNamespace(enable_hisparse=False)
    q._clean_hicache_prefetch_resources = MagicMock()
    q.deferred_kv_release_timeout = timeout
    q.enable_staging = False
    q.staging_handler = None
    q.tree_cache = object()
    q.metadata_buffers = SimpleNamespace(bootstrap_room={})
    q.req_to_metadata_buffer_idx_allocator = _FakeIdxAllocator()
    return q


def _make_decode_req(room, idx, mgr, n_prefill_ranks=1):
    receiver = SimpleNamespace(
        kv_mgr=mgr,
        # One entry per prefill rank the decode notified of the abort; its length
        # is the required drain-ack count (see DecodeTransferQueue._defer_release).
        bootstrap_infos=[{"rank": r} for r in range(n_prefill_ranks)],
        clear=lambda: None,
    )
    return SimpleNamespace(
        req=SimpleNamespace(bootstrap_room=room),
        kv_receiver=receiver,
        metadata_buffer_index=idx,
        abort_drain_timeout_reported=False,
        abort_release_failed=False,
        abort_release_phase=0,
        abort_release_failed_phase=None,
        abort_release_control_retry_at=0.0,
    )


class TestResolveDeferredReleases(CustomTestCase):
    def test_noop_when_nothing_deferred(self):
        q = _make_queue()
        with patch.object(decode_mod, "release_kv_cache") as rel:
            q.resolve_deferred_releases()
        rel.assert_not_called()

    def test_holds_until_drained_then_releases(self):
        mgr = _make_manager()
        room, idx = 200, 7
        q = _make_queue()
        dreq = _make_decode_req(room, idx, mgr, n_prefill_ranks=2)
        # In production the room is armed in abort_request when the ABORT is
        # sent, before the scheduler defers here.
        mgr.register_deferred_abort_room(room)
        q._defer_release(dreq)

        with patch.object(decode_mod, "release_kv_cache") as rel:
            # Not yet acked -> held, not released.
            q.resolve_deferred_releases()
            rel.assert_not_called()
            self.assertEqual(len(q._deferred_releases), 1)

            # One of two ranks acked -> still held.
            mgr.note_abort_ack(room, 0)
            q.resolve_deferred_releases()
            rel.assert_not_called()
            self.assertEqual(len(q._deferred_releases), 1)

            # Both ranks acked -> released exactly once.
            mgr.note_abort_ack(room, 1)
            q.resolve_deferred_releases()
            rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)

        # Held state fully cleaned up.
        self.assertEqual(q._deferred_releases, [])
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [idx])
        self.assertEqual(q.metadata_buffers.bootstrap_room[idx], 0)
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertIsNone(dreq.kv_receiver)

    def test_timeout_retains_destinations_until_later_drain_proof(self):
        mgr = _make_manager()
        room, idx = 300, 3
        q = _make_queue(timeout=30.0)
        dreq = _make_decode_req(room, idx, mgr, n_prefill_ranks=1)
        # Force an already-expired deadline (no ack will ever arrive).
        q._deferred_releases.append((dreq, float("-inf"), idx, 1))

        with patch.object(decode_mod, "release_kv_cache") as rel:
            with patch.object(decode_mod.logger, "error") as log:
                q.resolve_deferred_releases()
                q.resolve_deferred_releases()
                log.assert_called_once()
            rel.assert_not_called()
            q._clean_hicache_prefetch_resources.assert_not_called()
            self.assertEqual(len(q._deferred_releases), 1)
            self.assertTrue(q.has_pending_deferred_releases())
            mgr.register_deferred_abort_room(room)
            mgr.note_abort_ack(room, 0)
            q.resolve_deferred_releases()
            rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)

        self.assertEqual(q._deferred_releases, [])
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [idx])
        self.assertIsNone(dreq.kv_receiver)

    def test_failed_release_is_quarantined_without_replaying_partial_frees(self):
        # Unknown partial frees must retain ownership but never be blindly
        # replayed. A failed release must not block a separately drained owner.
        mgr = _make_manager()
        q = _make_queue()
        good = _make_decode_req(700, 1, mgr)
        bad = _make_decode_req(701, 2, mgr)
        # Only actual drain proof permits release, regardless of the deadline.
        for room in (700, 701):
            mgr.register_deferred_abort_room(room)
            mgr.note_abort_ack(room, 0)
        q._deferred_releases.append((bad, float("-inf"), 2, 1))
        q._deferred_releases.append((good, float("-inf"), 1, 1))

        calls = []

        def fake_release(req, tree_cache, is_insert):
            calls.append(req)
            if req is bad.req:
                raise RuntimeError("boom")

        with patch.object(decode_mod, "release_kv_cache", side_effect=fake_release):
            q.resolve_deferred_releases()  # must not raise
            # The good one still released despite the bad one throwing.
            self.assertIn(good.req, calls)
            self.assertEqual(len(q._deferred_releases), 1)
            self.assertIs(q._deferred_releases[0][0], bad)
            self.assertTrue(bad.abort_release_failed)
            self.assertTrue(q.has_pending_deferred_releases())
            attempts = len(calls)
            q.resolve_deferred_releases()
            self.assertEqual(len(calls), attempts)

    def test_control_cleanup_retries_without_replaying_data_frees(self):
        mgr = _make_manager()
        mgr.register_abort_released_notice = MagicMock(
            side_effect=[RuntimeError("notice unavailable"), None]
        )
        q = _make_queue()
        dreq = _make_decode_req(702, 3, mgr)
        mgr.register_deferred_abort_room(702)
        mgr.note_abort_ack(702, 0)
        q._deferred_releases.append((dreq, float("-inf"), 3, 1))

        with patch.object(decode_mod, "release_kv_cache") as release:
            q.resolve_deferred_releases()
            self.assertFalse(dreq.abort_release_failed)
            self.assertEqual(dreq.abort_release_phase, 5)
            self.assertEqual(dreq.abort_release_failed_phase, "release_notice")
            release.assert_called_once()
            self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [3])

            dreq.abort_release_control_retry_at = 0.0
            q.resolve_deferred_releases()
            release.assert_called_once()

        self.assertEqual(mgr.register_abort_released_notice.call_count, 2)
        self.assertEqual(q._deferred_releases, [])
        self.assertEqual(dreq.abort_release_phase, 8)
        self.assertIsNone(dreq.abort_release_failed_phase)
        self.assertIsNone(dreq.kv_receiver)

    def test_receiver_cleanup_resumes_after_tracker_already_cleared(self):
        mgr = _make_manager()
        q = _make_queue()
        dreq = _make_decode_req(703, 4, mgr)
        mgr.register_deferred_abort_room(703)
        mgr.note_abort_ack(703, 0)
        clear = MagicMock(side_effect=[RuntimeError("receiver clear failed"), None])
        dreq.kv_receiver.clear = clear
        q._deferred_releases.append((dreq, float("-inf"), 4, 1))

        with patch.object(decode_mod, "release_kv_cache") as release:
            q.resolve_deferred_releases()
            self.assertEqual(dreq.abort_release_phase, 7)
            self.assertFalse(dreq.abort_release_failed)
            self.assertNotIn(703, mgr._deferred_abort_ack_tracker)
            dreq.abort_release_control_retry_at = 0.0
            q.resolve_deferred_releases()
            release.assert_called_once()

        self.assertEqual(clear.call_count, 2)
        self.assertEqual(q._deferred_releases, [])
        self.assertIsNone(dreq.kv_receiver)

    def test_defer_release_records_deadline_and_idx(self):
        mgr = _make_manager()
        q = _make_queue(timeout=12.5)
        dreq = _make_decode_req(room=400, idx=9, mgr=mgr)
        q._defer_release(dreq)
        self.assertEqual(len(q._deferred_releases), 1)
        held_req, deadline, held_idx, required = q._deferred_releases[0]
        self.assertIs(held_req, dreq)
        self.assertEqual(held_idx, 9)
        self.assertIsInstance(deadline, float)

    def test_missing_bootstrap_targets_do_not_make_zero_acks_safe(self):
        mgr = _make_manager()
        q = _make_queue()
        dreq = _make_decode_req(402, 9, mgr, n_prefill_ranks=0)
        q._defer_release(dreq)
        self.assertEqual(q._deferred_releases[0][3], 1)
        with patch.object(decode_mod, "release_kv_cache") as rel:
            q.resolve_deferred_releases()
        rel.assert_not_called()

    def test_memory_release_refuses_owned_transfer_destinations(self):
        mgr = _make_manager()
        q = _make_queue()
        dreq = _make_decode_req(403, 9, mgr)
        q._defer_release(dreq)
        with self.assertRaisesRegex(RuntimeError, "destinations remain owned"):
            q.release_memory_occupation()
        self.assertEqual(len(q._deferred_releases), 1)

    def test_prealloc_callback_cannot_deregister_held_destination_pool(self):
        mgr = _make_manager()
        q = _make_queue()
        q._defer_release(_make_decode_req(404, 9, mgr))
        prealloc = DecodePreallocQueue.__new__(DecodePreallocQueue)
        prealloc.transfer_queue = q
        prealloc._num_published_destinations = 0
        prealloc.kv_manager = SimpleNamespace(deregister_buffer_to_engine=MagicMock())
        prealloc._cancel_prefill_dp_rank_queries = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "destinations remain owned"):
            prealloc.release_memory_occupation()
        prealloc.kv_manager.deregister_buffer_to_engine.assert_not_called()
        prealloc._cancel_prefill_dp_rank_queries.assert_not_called()


class TestFailedTransferDestinationOwnership(CustomTestCase):
    def _failed_queue(self, *, published=True, notify=True, enabled=True):
        mgr = _make_manager()
        mgr.enable_deferred_decode_kv_release = enabled
        mgr.local_ip, mgr.rank_port = "10.0.0.4", 6001
        mgr.record_failure = MagicMock()
        mgr.update_status = MagicMock()
        receiver = NixlKVReceiver.__new__(NixlKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_room = 500
        receiver.transfer_epoch = 0
        receiver.init_time = None
        receiver.destination_metadata_published = published
        receiver.abort_notified = False
        receiver._abort_tracker_armed = False
        receiver.bootstrap_infos = (
            [{"rank_ip": "10.0.0.5", "rank_port": 6002}] if notify else None
        )
        receiver.failure_exception = MagicMock(
            side_effect=RuntimeError("prefill partial-post ERR")
        )
        receiver.clear = MagicMock()
        receiver._connect_to_bootstrap_server = MagicMock(
            side_effect=RuntimeError("peer unavailable")
        )
        q = _make_queue()
        q.enable_deferred_kv_release = enabled
        q.tp_rank = 0
        q.scheduler = SimpleNamespace(
            enable_hisparse=True,
            enable_decode_hicache=False,
            hisparse_coordinator=SimpleNamespace(request_finished=MagicMock()),
            output_streamer=SimpleNamespace(stream_output=MagicMock()),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
        )
        dreq = SimpleNamespace(
            req=SimpleNamespace(
                bootstrap_room=500, rid="failed-transfer", return_logprob=False
            ),
            kv_receiver=receiver,
            metadata_buffer_index=9,
            hicache_restore_status=decode_mod.HiCacheRestoreResult.READY,
            abort_drain_timeout_reported=False,
            abort_release_failed=False,
        )
        q.queue = [dreq]
        q._poll_with_metadata_gate = MagicMock(return_value=[decode_mod.KVPoll.Failed])
        return q, dreq, receiver, mgr

    def test_prefill_failure_or_failed_notification_keeps_all_destinations(self):
        for notify in (True, False):
            with self.subTest(notify=notify):
                q, dreq, receiver, mgr = self._failed_queue(notify=notify)
                with patch.object(decode_mod, "prepare_abort"), patch.object(
                    decode_mod, "release_kv_cache"
                ) as rel:
                    self.assertEqual(q.pop_transferred(), [])
                self.assertEqual(q.queue, [])
                self.assertEqual(len(q._deferred_releases), 1)
                self.assertIs(dreq.kv_receiver, receiver)
                receiver.clear.assert_not_called()
                rel.assert_not_called()
                q._clean_hicache_prefetch_resources.assert_not_called()
                q.scheduler.hisparse_coordinator.request_finished.assert_not_called()
                self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [])
                mgr.record_failure.assert_not_called()

    def test_explicit_unpublished_or_optout_failure_releases_without_notification(self):
        for published, enabled in ((False, True), (True, False)):
            with self.subTest(published=published, enabled=enabled):
                q, dreq, receiver, _ = self._failed_queue(
                    published=published, enabled=enabled
                )
                with patch.object(decode_mod, "prepare_abort"), patch.object(
                    decode_mod, "release_kv_cache"
                ) as rel:
                    q.pop_transferred()
                rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)
                receiver._connect_to_bootstrap_server.assert_not_called()
                receiver.clear.assert_called_once()
                self.assertEqual(q._deferred_releases, [])
                self.assertIsNone(dreq.kv_receiver)

    def test_hicache_or_hisparse_ownership_released_only_after_drain(self):
        q, dreq, _, mgr = self._failed_queue()
        with patch.object(decode_mod, "prepare_abort"), patch.object(
            decode_mod, "release_kv_cache"
        ) as rel:
            q.pop_transferred()
            mgr.note_abort_ack(500, 0)
            q.resolve_deferred_releases()
        q._clean_hicache_prefetch_resources.assert_called_once_with(dreq)
        q.scheduler.hisparse_coordinator.request_finished.assert_called_once_with(
            dreq.req
        )
        rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)

    def test_real_nixl_tracker_roundtrip_rejects_old_generation_at_release(self):
        q, dreq, receiver, _ = self._failed_queue()
        receiver.transfer_epoch = 3
        receiver.bootstrap_infos[0]["abort_ack_rank"] = 0
        mgr = NixlKVManager.__new__(NixlKVManager)
        mgr._source_lease_lock = threading.RLock()
        mgr._abort_generations = {}
        mgr.enable_deferred_decode_kv_release = True
        mgr.local_ip, mgr.rank_port = "10.0.0.4", 6001
        receiver.kv_mgr = mgr
        socket = MagicMock()
        receiver._connect_to_bootstrap_server = MagicMock(
            return_value=(socket, threading.Lock())
        )
        with patch.object(decode_mod, "prepare_abort"), patch.object(
            decode_mod, "release_kv_cache"
        ) as rel:
            q.pop_transferred()
            frames = socket.send_multipart.call_args.args[0]
            self.assertEqual(frames[0], b"ABORT")
            self.assertEqual(frames[4].decode(), receiver._abort_generation)
            self.assertEqual(frames[5], b"3")
            mgr.note_abort_ack(500, 0, "old-generation", 3)
            mgr.note_abort_ack(500, 9, receiver._abort_generation, 3)
            q.resolve_deferred_releases()
            rel.assert_not_called()
            mgr.handle_abort_ack_message(
                [b"ABORT_ACK", b"500", b"0", frames[4], frames[5]]
            )
            q.resolve_deferred_releases()
            rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)
        self.assertNotIn((500, 3), mgr._abort_generations)
        self.assertIsNone(dreq.kv_receiver)

    def test_missing_registered_sender_identity_keeps_real_nixl_hold_unproven(self):
        q, dreq, receiver, _ = self._failed_queue()
        mgr = NixlKVManager.__new__(NixlKVManager)
        mgr._source_lease_lock = threading.RLock()
        mgr._abort_generations = {}
        mgr.enable_deferred_decode_kv_release = True
        mgr.local_ip, mgr.rank_port = "10.0.0.4", 6001
        receiver.kv_mgr = mgr
        with patch.object(decode_mod, "prepare_abort"), patch.object(
            decode_mod, "release_kv_cache"
        ) as rel:
            q.pop_transferred()
            for rank in range(8):
                mgr.note_abort_ack(500, rank, receiver._abort_generation)
            q.resolve_deferred_releases()
            rel.assert_not_called()
        self.assertEqual(len(q._deferred_releases), 1)
        self.assertIs(dreq.kv_receiver, receiver)


if __name__ == "__main__":
    unittest.main()
