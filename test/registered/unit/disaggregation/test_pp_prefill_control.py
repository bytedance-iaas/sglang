"""Deterministic PP-prefill consensus and ownership regressions; no GPUs."""

import queue
import unittest
from collections import OrderedDict, defaultdict
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.disaggregation.pp_prefill_control import (
    PPControlFrame,
    PPDecisionLog,
    PPPrefillControl,
    TransferLeases,
    poll_prefill_controlled,
    request_key,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ReplicaStore:
    """Manual delivery models a delayed background replication worker."""

    def __init__(self, network, rank):
        self.network = network
        self.rank = rank
        self.local = {}

    def __setitem__(self, key, value):
        self.local[key] = value
        if self.rank == 0:
            self.network.visible[0][key] = value

    def pop(self, key, default=None):
        self.network.visible[self.rank].pop(key, None)
        return self.local.pop(key, default)

    def get(self, key, default=None):
        return self.local.get(key, default)

    def collect(self, key):
        return [replica.get(key) for replica in self.network.visible]


class Pipeline:
    def __init__(self, size):
        self.now = 0
        self.visible = [{} for _ in range(size)]
        self.stores = [ReplicaStore(self, rank) for rank in range(size)]
        self.controls = [
            PPPrefillControl(
                PPDecisionLog(rank == 0, "epoch"), store, 10, lambda: self.now
            )
            for rank, store in enumerate(self.stores)
        ]
        self.reqs = [
            NS(rid="request", bootstrap_room=123, owns_kv=True, owns_metadata=True)
            for _ in range(size)
        ]
        self.leases = [TransferLeases() for _ in range(size)]
        self.retired = [0] * size
        for control, req in zip(self.controls, self.reqs):
            control.register(req)

    def deliver(self):
        self.visible = [dict(store.local) for store in self.stores]

    def boundary(self):
        frame = None
        for rank, (control, req, leases) in enumerate(
            zip(self.controls, self.reqs, self.leases)
        ):
            control.log.begin(frame)

            def fence(req, reason):
                leases.close(req.bootstrap_room)

            def retire(req):
                self.assert_owned(req)
                req.owns_kv = req.owns_metadata = False
                self.retired[rank] += 1

            control.boundary(
                fence, retire, lambda req: leases.drained(req.bootstrap_room)
            )
            frame = control.log.finish()
        return frame

    @staticmethod
    def assert_owned(req):
        assert req.owns_kv and req.owns_metadata


class TestPPPrefillControl(CustomTestCase):
    def test_pp0_decision_delay_keeps_gpu_and_transport_ownership(self):
        for size in (2, 4, 8):
            with self.subTest(size=size):
                pipeline = Pipeline(size)
                batches = [NS(reqs=[req]) for req in pipeline.reqs]
                for control, batch in zip(pipeline.controls, batches):
                    control.bind_batch(batch)
                # One queued read and one running read: a terminal poll is not a drain.
                pipeline.leases[-1].acquire(123)
                pipeline.leases[-1].acquire(123)
                pipeline.controls[-1].propose(pipeline.reqs[-1], "late failure")
                for _ in range(3):
                    pipeline.boundary()  # PP0 has not received the proposal.
                self.assertEqual(pipeline.retired, [0] * size)
                self.assertTrue(
                    all(not control.fenced for control in pipeline.controls)
                )
                self.assertTrue(
                    all(req.owns_kv and req.owns_metadata for req in pipeline.reqs)
                )
                pipeline.deliver()
                pipeline.boundary()  # Immutable abort fence.
                self.assertTrue(all(control.fenced for control in pipeline.controls))
                for leases in pipeline.leases:
                    self.assertFalse(leases.acquire(123))
                for control, batch in zip(pipeline.controls, batches):
                    control.complete_batch(batch)  # Old work drains after the fence.
                pipeline.deliver()
                pipeline.boundary()
                self.assertEqual(pipeline.retired, [0] * size)
                pipeline.leases[-1].release(123)
                pipeline.deliver()
                pipeline.boundary()
                self.assertEqual(pipeline.retired, [0] * size)
                pipeline.leases[-1].release(123)
                pipeline.boundary()  # Publishes readiness, delivery still delayed.
                self.assertEqual(pipeline.retired, [0] * size)
                pipeline.deliver()
                pipeline.boundary()  # PP0's retire commit, once at every stage.
                self.assertEqual(pipeline.retired, [1] * size)
                pipeline.deliver()
                pipeline.boundary()
                self.assertEqual(pipeline.retired, [1] * size)

    def test_timeout_does_not_free_without_pp0_or_drain(self):
        pipeline = Pipeline(2)
        pipeline.controls[1].propose(pipeline.reqs[1], "failure")
        pipeline.now = 11
        with self.assertRaises(TimeoutError):
            pipeline.controls[1].check_deadlines()
        self.assertEqual(pipeline.retired, [0, 0])
        self.assertTrue(all(req.owns_kv and req.owns_metadata for req in pipeline.reqs))

    def test_retire_decision_delay_and_timeout_keep_resources(self):
        pipeline = Pipeline(2)
        pipeline.leases[1].acquire(123)
        pipeline.controls[0].propose(pipeline.reqs[0], "abort")
        pipeline.boundary()
        pipeline.deliver()
        pipeline.boundary()
        pipeline.now = 11
        for control in pipeline.controls:
            with self.assertRaises(TimeoutError):
                control.check_deadlines()
        self.assertTrue(all(req.owns_kv and req.owns_metadata for req in pipeline.reqs))
        self.assertEqual(pipeline.retired, [0, 0])

    def test_watchdog_observes_failure_while_scheduler_waits_for_pp0(self):
        from sglang.srt.disaggregation import pp_prefill_control as module

        pipeline = Pipeline(2)
        control = pipeline.controls[1]
        # The scheduling thread never gets a frame and never calls propose().
        with patch.object(module.threading, "Thread") as thread:
            control.start_watchdog(lambda req: True)
            watchdog_step = thread.call_args.kwargs["target"]

        def advance_clock(_):
            pipeline.now += 11

        with patch.object(
            module.time, "sleep", side_effect=advance_clock
        ), patch.object(module.os, "kill") as kill, patch.object(
            module.os, "_exit", side_effect=SystemExit("group fail-stop")
        ):
            with self.assertLogs(module.logger, level="CRITICAL"):
                with self.assertRaisesRegex(SystemExit, "group fail-stop"):
                    watchdog_step()
        self.assertFalse(control.proposals)
        self.assertTrue(pipeline.reqs[1].owns_kv and pipeline.reqs[1].owns_metadata)
        self.assertEqual(pipeline.retired, [0, 0])
        kill.assert_called_once_with(module.os.getppid(), module.signal.SIGQUIT)

    def test_proposal_capacity_is_bounded_and_repeats_are_deduplicated(self):
        pipeline = Pipeline(2)
        control = pipeline.controls[0]
        control.max_pending = 1
        control.propose(pipeline.reqs[0], "abort")
        control.propose(pipeline.reqs[0], "repeat")
        with self.assertRaises(RuntimeError):
            control.propose(NS(rid="another", bootstrap_room=456), "second abort")
        self.assertEqual(len(control.proposals), 1)
        self.assertTrue(pipeline.reqs[0].owns_kv and pipeline.reqs[0].owns_metadata)

    def test_late_abort_cannot_veto_frozen_ready(self):
        from sglang.srt.disaggregation.base import KVPoll

        pipeline = Pipeline(2)
        schedulers = []
        for rank, (control, req) in enumerate(zip(pipeline.controls, pipeline.reqs)):
            req.disagg_kv_sender = NS(poll_pp_consensus=lambda: KVPoll.WaitingForInput)
            schedulers.append(
                NS(
                    pp_prefill_control=control,
                    attn_cp_cpu_group=None,
                    attn_tp_cpu_group=None,
                )
            )
        local = [KVPoll.WaitingForInput]
        with patch(
            "sglang.srt.disaggregation.utils.poll_and_all_reduce_attn_cp_tp_group",
            side_effect=lambda *args: local,
        ), patch(
            "sglang.srt.disaggregation.utils._all_reduce_polls",
            side_effect=lambda polls, group: polls,
        ):
            leader, follower = pipeline.controls
            leader.log.begin()
            good = poll_prefill_controlled(
                schedulers[0], [pipeline.reqs[0]], "chunk-bootstrap"
            )
            frame = leader.log.finish()
            # Failure is observed AFTER PP0 froze the decision, before PP1 applies it.
            local[:] = [KVPoll.Failed]
            follower.log.begin(frame)
            delayed = poll_prefill_controlled(
                schedulers[1], [pipeline.reqs[1]], "chunk-bootstrap"
            )
            follower.log.finish()
        self.assertEqual(good, (KVPoll.WaitingForInput,))
        self.assertEqual(delayed, good)
        self.assertIn(request_key(pipeline.reqs[1]), follower.proposals)
        self.assertFalse(follower.fenced)
        self.assertTrue(pipeline.reqs[1].owns_kv)
        pipeline.deliver()
        pipeline.boundary()
        self.assertTrue(all(control.fenced for control in pipeline.controls))

    def test_failure_and_abort_entrypoints_only_propose_before_commit(self):
        from sglang.srt.disaggregation.prefill import (
            SchedulerDisaggregationPrefillMixin,
        )
        from sglang.srt.managers.scheduler import Scheduler

        pipeline = Pipeline(2)
        control = pipeline.controls[1]
        req = pipeline.reqs[1]
        scheduler = NS(pp_prefill_control=control)
        SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure(scheduler, req)
        SchedulerDisaggregationPrefillMixin.handle_inflight_transfer_failure(
            scheduler, req
        )
        SchedulerDisaggregationPrefillMixin.release_aborted_prefill_waiting_req(
            scheduler, req
        )
        SchedulerDisaggregationPrefillMixin._retire_aborted_prefill_result(
            scheduler, req
        )
        Scheduler.abort_request(
            scheduler, NS(abort_all=False, rid="request", abort_message="late abort")
        )
        self.assertTrue(req.owns_kv and req.owns_metadata)
        self.assertFalse(control.fenced)
        self.assertEqual(len(control.proposals), 1)

    def test_abort_cancels_unadmitted_grammar_and_mm_without_freeing_owned_kv(self):
        from concurrent.futures import Future

        from sglang.srt.constrained.grammar_manager import GrammarManager
        from sglang.srt.disaggregation.encoder.receiver import (
            MMReceiverHTTP,
            WaitingMMRequestStatus,
            WaitingZmqRequest,
        )
        from sglang.srt.managers.io_struct import AbortReq
        from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.sampling.sampling_params import SamplingParams

        for abort_all in (False, True):
            with self.subTest(abort_all=abort_all):
                p = Pipeline(2)
                reqs = [
                    Req(rid, "", [1, 2], SamplingParams())
                    for rid in ("request-grammar", "other")
                ]
                futures = [Future(), Future()]
                for req, future in zip(reqs, futures):
                    req.grammar = future
                grammar = GrammarManager.__new__(GrammarManager)
                grammar.grammar_queue = reqs
                mm = MMReceiverHTTP.__new__(MMReceiverHTTP)
                waiting = WaitingZmqRequest.__new__(WaitingZmqRequest)
                waiting.rid, waiting.status = (
                    "request-mm",
                    WaitingMMRequestStatus.PENDING,
                )
                waiting.registration_future = Future()
                registration = waiting.registration_future
                waiting._pool_slot_id, waiting.embedding_pool = None, None
                waiting.embeddings_buffer, waiting.recv_socket = object(), None
                released = []
                waiting._mm_finalizer = lambda: released.append(True)
                mm.waiting_list = [waiting]
                scheduler = NS(
                    pp_prefill_control=p.controls[0],
                    grammar_manager=grammar,
                    mm_receiver=mm,
                )
                from sglang.srt.runtime_context import get_parallel

                with get_parallel().override(tp_rank=0):
                    Scheduler.abort_request(
                        scheduler, AbortReq(rid="request", abort_all=abort_all)
                    )
                self.assertTrue(futures[0].cancelled())
                self.assertIsInstance(reqs[0].to_finish, FINISH_ABORT)
                self.assertEqual(futures[1].cancelled(), abort_all)
                self.assertEqual(isinstance(reqs[1].to_finish, FINISH_ABORT), abort_all)
                self.assertEqual(waiting.status, WaitingMMRequestStatus.FAIL)
                self.assertTrue(registration.cancelled())
                self.assertIsNone(waiting.embeddings_buffer)
                self.assertEqual(released, [True])
                self.assertTrue(p.reqs[0].owns_kv and p.reqs[0].owns_metadata)
                self.assertIn(request_key(p.reqs[0]), p.controls[0].proposals)
                self.assertFalse(p.controls[0].fenced)

    def test_fenced_resources_block_idle_cleanup_and_cache_flush(self):
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.scheduler import Scheduler

        pipeline = Pipeline(2)
        control = pipeline.controls[0]
        scheduler = NS(
            pp_prefill_control=control,
            running_batch=NS(is_empty=lambda: True),
            chunked_req=None,
            dllm_manager=NS(any_staging_reqs=lambda: False),
            last_batch=None,
            enable_overlap=False,
            _pp_microbatches_drained=lambda: True,
            waiting_queue=[],
            grammar_manager=NS(grammar_queue=[]),
            disaggregation_mode=DisaggregationMode.PREFILL,
            disagg_prefill_inflight_queue=[],
            disagg_prefill_bootstrap_queue=NS(queue=[]),
            enable_hisparse=False,
            enable_hierarchical_cache=False,
            enable_lmcache=False,
        )
        self.assertTrue(Scheduler.is_fully_idle(scheduler))
        control.propose(pipeline.reqs[0], "abort")
        self.assertFalse(Scheduler.is_fully_idle(scheduler))
        pipeline.boundary()
        self.assertFalse(Scheduler.is_fully_idle(scheduler, ignore_waiting=True))
        pipeline.deliver()
        pipeline.boundary()
        pipeline.deliver()
        pipeline.boundary()
        self.assertTrue(Scheduler.is_fully_idle(scheduler))

    def test_fenced_request_cannot_checkpoint_or_start_new_kv_reads(self):
        from sglang.srt.disaggregation.prefill import (
            SchedulerDisaggregationPrefillMixin,
        )

        pipeline = Pipeline(2)
        control = pipeline.controls[0]
        control.propose(pipeline.reqs[0], "abort")
        pipeline.boundary()
        scheduler = NS(pp_prefill_control=control)
        # No cache/checksum/transport adapter is present: a fenced operation
        # must return before accessing any of those resource-bearing objects.
        SchedulerDisaggregationPrefillMixin.checkpoint_disagg_prefill(
            scheduler, pipeline.reqs[0]
        )
        SchedulerDisaggregationPrefillMixin.send_kv_chunk(
            scheduler, pipeline.reqs[0], last_chunk=True
        )
        self.assertTrue(pipeline.reqs[0].owns_kv)

    def test_bootstrap_ready_waits_for_real_metadata_reservations(self):
        from sglang.srt.disaggregation.base import KVPoll
        from sglang.srt.disaggregation.pp_prefill_control import (
            reserve_bootstrap_metadata,
        )

        pipeline = Pipeline(2)
        schedulers = [
            NS(pp_prefill_control=ctl, attn_cp_cpu_group=None, attn_tp_cpu_group=None)
            for ctl in pipeline.controls
        ]
        available = [True, False]
        for req in pipeline.reqs:
            req.disagg_kv_sender = NS(poll_pp_consensus=lambda: KVPoll.WaitingForInput)
            req.metadata_buffer_index = -1
        with patch(
            "sglang.srt.disaggregation.utils.poll_and_all_reduce_attn_cp_tp_group",
            return_value=[KVPoll.WaitingForInput],
        ), patch(
            "sglang.srt.disaggregation.utils._all_reduce_polls",
            side_effect=lambda polls, group: polls,
        ):
            for rank in range(2):

                def reserve(req, rank=rank):
                    if available[rank]:
                        req.metadata_buffer_index = rank
                        return True
                    return False

                reserve_bootstrap_metadata(
                    NS(
                        scheduler=schedulers[rank],
                        queue=[pipeline.reqs[rank]],
                        ensure_metadata_buffer=reserve,
                    )
                )
            pipeline.deliver()
            leader = pipeline.controls[0]
            leader.log.begin()
            self.assertEqual(
                poll_prefill_controlled(schedulers[0], [pipeline.reqs[0]], "bootstrap"),
                (KVPoll.Bootstrapping,),
            )
            leader.log.finish()
            self.assertEqual(pipeline.reqs[0].metadata_buffer_index, 0)
            self.assertEqual(pipeline.reqs[1].metadata_buffer_index, -1)
            available[1] = True
            reserve_bootstrap_metadata(
                NS(
                    scheduler=schedulers[1],
                    queue=[pipeline.reqs[1]],
                    ensure_metadata_buffer=lambda req: setattr(
                        req, "metadata_buffer_index", 1
                    )
                    or True,
                )
            )
            pipeline.deliver()
            leader.log.begin()
            self.assertEqual(
                poll_prefill_controlled(schedulers[0], [pipeline.reqs[0]], "bootstrap"),
                (KVPoll.WaitingForInput,),
            )
            leader.log.finish()
            self.assertTrue(
                all(req.metadata_buffer_index >= 0 for req in pipeline.reqs)
            )

    def test_runtime_broadcast_envelope_handles_empty_and_boolean_decisions(self):
        from sglang.srt.disaggregation import pp_prefill_control as module
        from sglang.srt.disaggregation.pp_consensus_store import PPConsensusStore

        store = PPConsensusStore.__new__(PPConsensusStore)
        scheduler = NS(
            disagg_prefill_bootstrap_queue=NS(
                kv_manager=NS(
                    request_status=store, _pp_transfer_leases=TransferLeases()
                )
            ),
            pp_group=NS(broadcast_object=lambda value, src: value),
            policy=NS(),
            server_args=NS(watchdog_timeout=3600),
        )

        def list_broadcast(data):
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 1)
            return data

        with patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=NS(pp_rank=0, pp_size=2),
        ), patch(
            "sglang.srt.distributed.communication_op.attn_cp_tp_broadcast_pyobj",
            side_effect=list_broadcast,
        ), patch.object(
            module.PPPrefillControl, "start_watchdog"
        ):
            module.init_pp_prefill_control(scheduler)
            log = scheduler.pp_prefill_control.log
            self.assertIsInstance(log.epoch, str)
            self.assertEqual(len(log.epoch), 32)
            log.begin()
            log.check("empty-batch", None)
            self.assertFalse(log.decide("refresh", (), lambda: False))
            self.assertTrue(log.decide("refresh", (), lambda: True))
            log.finish()

    def test_uninstrumented_backend_keeps_existing_pp_loop_protocol(self):
        from sglang.srt.disaggregation.pp_prefill_control import init_pp_prefill_control

        scheduler = NS(disagg_prefill_bootstrap_queue=NS(kv_manager=NS()), policy=NS())
        self.assertIsNone(init_pp_prefill_control(scheduler))
        self.assertFalse(hasattr(scheduler, "pp_prefill_control"))

    def test_fake_health_sender_abort_can_retire_without_rdma(self):
        from sglang.srt.disaggregation.fake.conn import FakeKVSender
        from sglang.srt.disaggregation.pp_prefill_control import apply_pp_boundary

        pipeline = Pipeline(2)
        schedulers = []
        for rank, (control, req) in enumerate(zip(pipeline.controls, pipeline.reqs)):
            req.disagg_kv_sender = FakeKVSender.__new__(FakeKVSender)

            def retire(req, pp_retire_commit, rank=rank):
                self.assertTrue(pp_retire_commit)
                req.owns_kv = req.owns_metadata = False
                pipeline.retired[rank] += 1

            schedulers.append(
                NS(
                    pp_prefill_control=control,
                    waiting_queue=[req],
                    disagg_prefill_bootstrap_queue=NS(queue=[]),
                    disagg_prefill_inflight_queue=[],
                    chunked_req=None,
                    attn_tp_cpu_group=None,
                    attn_cp_cpu_group=None,
                    handle_bootstrap_failure=retire,
                )
            )
        pipeline.controls[0].propose(pipeline.reqs[0], "health cancelled")
        with patch(
            "sglang.srt.disaggregation.utils._all_reduce_polls",
            side_effect=lambda polls, group: polls,
        ):
            for _ in range(3):
                frame = None
                for control, scheduler in zip(pipeline.controls, schedulers):
                    control.log.begin(frame)
                    apply_pp_boundary(scheduler)
                    frame = control.log.finish()
                pipeline.deliver()
        self.assertEqual(pipeline.retired, [1, 1])

    def test_epoch_gap_duplicate_and_batch_mismatch_fail_closed(self):
        for frame in (
            None,
            PPControlFrame("old", 0, ()),
            PPControlFrame("epoch", 1, ()),
        ):
            with self.subTest(frame=frame):
                with self.assertRaises(RuntimeError):
                    PPDecisionLog(False, "epoch").begin(frame)
        leader, follower = PPDecisionLog(True, "epoch"), PPDecisionLog(False, "epoch")
        leader.begin()
        leader.check("batch", None)
        frame = leader.finish()
        follower.begin(frame)
        with self.assertRaises(RuntimeError):
            follower.check("batch", ("nonempty",))
        with self.assertRaises(RuntimeError):
            follower.begin(frame)

    def test_stale_incarnation_proposal_cannot_abort_reused_rid(self):
        pipeline = Pipeline(2)
        old = NS(rid="request", bootstrap_room=122)
        control = pipeline.controls[1]
        control.propose(old, "old timeout")
        control.forget_success(old)
        pipeline.deliver()
        pipeline.boundary()
        self.assertTrue(all(not ctl.fenced for ctl in pipeline.controls))
        self.assertTrue(all(req.owns_kv for req in pipeline.reqs))

    def test_fenced_request_cannot_bind_new_forward(self):
        pipeline = Pipeline(2)
        pipeline.controls[0].propose(pipeline.reqs[0], "abort")
        pipeline.boundary()
        for control, req in zip(pipeline.controls, pipeline.reqs):
            with self.assertRaises(RuntimeError):
                control.bind_batch(NS(reqs=[req]))

    def test_transfer_lease_includes_queued_read_and_defer(self):
        leases = TransferLeases()
        self.assertTrue(leases.acquire(1))  # Enqueue, before worker dequeue.
        leases.close(1)
        self.assertFalse(leases.drained(1))
        self.assertFalse(leases.acquire(1))
        # A defer keeps the SAME lease, so no transient zero is published.
        self.assertFalse(leases.drained(1))
        with self.assertRaises(RuntimeError):
            leases.forget(1)
        leases.release(1)
        self.assertTrue(leases.drained(1))
        leases.forget(1)
        with self.assertRaises(RuntimeError):
            leases.release(1)

    def test_actual_mooncake_queue_drains_aborted_queued_read(self):
        from sglang.srt.disaggregation.base import KVPoll
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
        from sglang.srt.disaggregation.utils import DisaggregationMode

        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.request_status = {1: KVPoll.WaitingForInput}
        manager.transfer_infos = {1: {"peer:1": NS()}}
        manager.transfer_queues = [queue.Queue()]
        manager._pp_transfer_leases = TransferLeases()
        manager.check_status = lambda room: manager.request_status[room]
        manager.enable_trace = False
        manager.enable_deferred_decode_kv_release = False
        manager._staging_outstanding = defaultdict(int)
        manager.add_transfer_request(1, [], slice(0, 0), False, trace_ctx=NS())
        manager._pp_transfer_leases.close(1)
        manager.request_status[1] = KVPoll.Failed
        self.assertFalse(manager._pp_transfer_leases.drained(1))
        manager.transfer_queues[0].put(None)
        manager.transfer_worker(manager.transfer_queues[0], None)
        self.assertTrue(manager._pp_transfer_leases.drained(1))
        manager.add_transfer_request(1, [], slice(0, 0), False, trace_ctx=NS())
        self.assertTrue(manager.transfer_queues[0].empty())

    def test_lru_refresh_descriptor_and_touch_are_bounded_to_head_128(self):
        from sglang.srt.managers import schedule_policy as module

        policy = module.SchedulePolicy.__new__(module.SchedulePolicy)
        policy.tree_cache = NS(
            supports_fast_match_prefix=lambda: False,
            supports_prefix_sharing=lambda: True,
        )
        policy.pp_decision_log = PPDecisionLog(True, "epoch")
        policy._last_waiting_prefix_refresh = float("-inf")
        requests = [
            NS(rid=str(i), bootstrap_room=i, origin_input_ids=[i]) for i in range(200)
        ]
        touched = []
        with patch.object(
            module, "get_disagg", return_value=NS(disaggregation_mode="prefill")
        ), patch.object(
            module, "get_memory", return_value=NS(radix_eviction_policy="lru")
        ), patch.object(
            module,
            "touch_waiting_prefix",
            side_effect=lambda req, cache: touched.append(req.rid),
        ), patch.object(
            module.envs.SGLANG_ENABLE_WAITING_PREFIX_REFRESH, "get", return_value=True
        ):
            policy.pp_decision_log.begin()
            policy._touch_waiting_prefixes(module.CacheAgnosticPolicy.FCFS, requests)
            frame = policy.pp_decision_log.finish()
        self.assertEqual(len(frame.events[0][1]), 128)
        self.assertEqual(touched, [str(i) for i in reversed(range(128))])

    def test_lru_refresh_uses_pp0_clock_and_preserves_order(self):
        from sglang.srt.managers import schedule_policy as policy_module

        class LRU:
            def __init__(self):
                self.pages = OrderedDict((key, None) for key in ("hot", "other"))

            def supports_fast_match_prefix(self):
                return False

            def supports_prefix_sharing(self):
                return True

        reqs = [NS(rid="hot", bootstrap_room=1, origin_input_ids=[1])]
        logs = [PPDecisionLog(rank == 0, "epoch") for rank in range(2)]
        policies = []
        for log in logs:
            policy = policy_module.SchedulePolicy.__new__(policy_module.SchedulePolicy)
            policy.tree_cache = LRU()
            policy.pp_decision_log = log
            policy._last_waiting_prefix_refresh = 10
            policies.append(policy)
        with patch.object(
            policy_module, "get_disagg", return_value=NS(disaggregation_mode="prefill")
        ), patch.object(
            policy_module, "get_memory", return_value=NS(radix_eviction_policy="lru")
        ), patch.object(
            policy_module,
            "touch_waiting_prefix",
            side_effect=lambda req, cache: cache.pages.move_to_end(req.rid),
        ), patch.object(
            policy_module.envs.SGLANG_ENABLE_WAITING_PREFIX_REFRESH,
            "get",
            return_value=True,
        ):
            for leader_now, follower_now, expected_victim in (
                (10.169, 10.648, "hot"),
                (10.648, 10.169, "other"),
            ):
                for policy in policies:
                    policy.tree_cache = LRU()
                    policy._last_waiting_prefix_refresh = 10
                logs[0].begin()
                with patch.object(
                    policy_module.time, "monotonic", return_value=leader_now
                ):
                    policies[0]._touch_waiting_prefixes(
                        policy_module.CacheAgnosticPolicy.FCFS, reqs
                    )
                logs[1].begin(logs[0].finish())
                with patch.object(
                    policy_module.time, "monotonic", return_value=follower_now
                ):
                    policies[1]._touch_waiting_prefixes(
                        policy_module.CacheAgnosticPolicy.FCFS, reqs
                    )
                logs[1].finish()
                victims = [next(iter(policy.tree_cache.pages)) for policy in policies]
                self.assertEqual(victims, [expected_victim, expected_victim])


if __name__ == "__main__":
    unittest.main()
