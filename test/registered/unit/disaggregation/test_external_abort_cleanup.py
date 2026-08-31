"""Scheduler-visible ownership tests for PD external aborts."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.prefill import (
    PrefillAbortDrainTimeout,
    PrefillBootstrapQueue,
    process_pending_prefill_external_aborts,
    renew_disagg_prefill_owner_leases,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _base_scheduler(mode):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.chunked_req = None
    scheduler._pending_chunked_abort_req = None
    scheduler.waiting_queue = []
    scheduler.enable_hicache_storage = False
    scheduler.enable_hisparse = False
    scheduler.dllm_config = None
    scheduler.grammar_manager = MagicMock()
    scheduler.disaggregation_mode = mode
    scheduler.ps = SimpleNamespace(pp_size=1, pp_rank=0, attn_tp_rank=0, attn_cp_rank=0)
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler.last_batch = None
    scheduler.output_streamer = MagicMock()
    return scheduler


class TestDecodeExternalAbort(CustomTestCase):
    def test_pp1_routes_abort_locally_without_collective(self):
        scheduler = _base_scheduler(DisaggregationMode.DECODE)
        scheduler.disagg_decode_prealloc_queue = MagicMock()
        scheduler.disagg_decode_transfer_queue = MagicMock()

        with patch("sglang.srt.managers.scheduler.barrier") as barrier:
            scheduler.abort_request(AbortReq(rid="rid-prefix"))

        scheduler.disagg_decode_prealloc_queue.remove_aborted.assert_called_once_with(
            "rid-prefix", False
        )
        scheduler.disagg_decode_transfer_queue.remove_aborted.assert_called_once_with(
            "rid-prefix", False
        )
        barrier.assert_not_called()

    def test_pp_stage_abort_is_local_and_does_not_start_a_poll_collective(self):
        scheduler = _base_scheduler(DisaggregationMode.DECODE)
        scheduler.ps.pp_size = 2
        scheduler.running_mbs = []
        scheduler.mbs = []
        scheduler.disagg_decode_prealloc_queue = MagicMock()
        scheduler.disagg_decode_transfer_queue = MagicMock()

        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce_pp"
        ) as poll_collective:
            scheduler.abort_request(AbortReq(rid="rid-prefix"))

        scheduler.disagg_decode_prealloc_queue.remove_aborted.assert_called_once_with(
            "rid-prefix", False
        )
        scheduler.disagg_decode_transfer_queue.remove_aborted.assert_called_once_with(
            "rid-prefix", False
        )
        poll_collective.assert_not_called()

    def test_deferred_decode_hold_keeps_scheduler_non_idle(self):
        scheduler = _base_scheduler(DisaggregationMode.DECODE)
        scheduler.running_batch.is_empty = MagicMock(return_value=True)
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.enable_overlap = False
        scheduler._pp_microbatches_drained = MagicMock(return_value=True)
        scheduler.grammar_manager.grammar_queue = []
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            queue=[],
            deferred_abort_holds=[object()],
            has_pending_deferred_abort_holds=lambda: True,
        )
        scheduler.decode_offload_manager = None
        scheduler.enable_hierarchical_cache = False

        self.assertFalse(scheduler.is_fully_idle())


class TestPrefillExternalAbort(CustomTestCase):
    @patch("sglang.srt.disaggregation.prefill.release_kv_cache")
    def test_inflight_transfer_failure_defers_source_release_until_attempt_drains(
        self, release_kv
    ):
        manager = SimpleNamespace(
            _transfer_attempts={(51, 7): 1},
            request_status_lock=None,
            _maybe_ack_drained_abort=MagicMock(),
        )
        sender = SimpleNamespace(
            kv_mgr=manager,
            bootstrap_room=51,
            generation=7,
            abort=MagicMock(),
            failure_exception=MagicMock(),
        )
        req = SimpleNamespace(
            rid="failed-inflight",
            bootstrap_room=51,
            disagg_kv_sender=sender,
            metadata_buffer_index=8,
            pending_bootstrap=False,
            req_pool_idx=12,
            kv=object(),
            mamba_pool_idx=None,
            return_logprob=False,
            finished_output=False,
            finished_reason=None,
            time_stats=MagicMock(),
        )
        scheduler = _base_scheduler(DisaggregationMode.PREFILL)
        scheduler.ps.tp_rank = 0
        scheduler.tree_cache = object()
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = []
        queue.scheduler = scheduler
        queue.req_to_metadata_buffer_idx_allocator = (
            scheduler.req_to_metadata_buffer_idx_allocator
        )
        scheduler.disagg_prefill_bootstrap_queue = queue
        scheduler._pending_prefill_external_aborts = []

        scheduler.handle_inflight_transfer_failure(req)

        release_kv.assert_not_called()
        self.assertEqual(scheduler._pending_prefill_external_aborts, [req])
        self.assertEqual(manager._transfer_attempts, {(51, 7): 2})
        sender.abort.assert_called_once_with()
        manager._maybe_ack_drained_abort.assert_not_called()

        manager._transfer_attempts[(51, 7)] = 1
        process_pending_prefill_external_aborts(scheduler)

        release_kv.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
        self.assertEqual(manager._transfer_attempts, {})
        manager._maybe_ack_drained_abort.assert_called_once_with(51, 7)

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache")
    def test_bootstrap_abort_releases_sender_metadata_and_kv_once(self, release_kv):
        sender = MagicMock()
        req = SimpleNamespace(
            rid="prefill-child",
            disagg_kv_sender=sender,
            metadata_buffer_index=5,
            pending_bootstrap=True,
            req_pool_idx=11,
            kv=object(),
            mamba_pool_idx=None,
            return_logprob=False,
            finished_output=False,
            finished_reason=None,
        )
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = [req]
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.scheduler = SimpleNamespace(
            enable_hicache_storage=False,
            enable_hisparse=False,
            tree_cache=object(),
            output_streamer=MagicMock(),
            ps=SimpleNamespace(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0),
            req_to_metadata_buffer_idx_allocator=(
                queue.req_to_metadata_buffer_idx_allocator
            ),
        )

        queue.remove_aborted("prefill")
        queue.remove_aborted("prefill")

        self.assertEqual(queue.queue, [])
        sender.abort.assert_called_once_with()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(5)
        self.assertEqual(req.metadata_buffer_index, -1)
        self.assertFalse(req.pending_bootstrap)
        release_kv.assert_called_once_with(
            req, queue.scheduler.tree_cache, is_insert=False
        )
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], False
        )
        self.assertTrue(req.finished_output)

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache")
    def test_bootstrap_abort_preserves_concurrent_terminal_completion(self, release_kv):
        sender = MagicMock()
        terminal_reason = FINISH_LENGTH(length=0)
        req = SimpleNamespace(
            rid="prefill-terminal",
            disagg_kv_sender=sender,
            metadata_buffer_index=-1,
            pending_bootstrap=True,
            req_pool_idx=None,
            kv=None,
            mamba_pool_idx=None,
            return_logprob=False,
            finished_output=False,
            finished_reason=terminal_reason,
        )
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = [req]
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.scheduler = SimpleNamespace(
            enable_hicache_storage=False,
            enable_hisparse=False,
            tree_cache=object(),
            output_streamer=MagicMock(),
            ps=SimpleNamespace(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0),
            req_to_metadata_buffer_idx_allocator=(
                queue.req_to_metadata_buffer_idx_allocator
            ),
        )

        queue.remove_aborted("prefill")

        self.assertIs(req.finished_reason, terminal_reason)
        release_kv.assert_not_called()
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], False
        )
        self.assertTrue(req.finished_output)

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache")
    def test_source_abort_waits_for_generation_attempts_before_release(
        self, release_kv
    ):
        events = []
        manager = SimpleNamespace(
            _transfer_attempts={(41, 3): 1},
            request_status_lock=None,
            _maybe_ack_drained_abort=MagicMock(
                side_effect=lambda *_args: events.append("ack")
            ),
        )
        sender = SimpleNamespace(
            kv_mgr=manager,
            bootstrap_room=41,
            generation=3,
            abort=MagicMock(),
        )
        req = SimpleNamespace(
            rid="draining",
            disagg_kv_sender=sender,
            metadata_buffer_index=5,
            pending_bootstrap=False,
            req_pool_idx=11,
            kv=object(),
            mamba_pool_idx=None,
            return_logprob=False,
            finished_output=False,
            finished_reason=None,
        )
        scheduler = _base_scheduler(DisaggregationMode.PREFILL)
        scheduler.tree_cache = object()
        scheduler.req_to_metadata_buffer_idx_allocator = MagicMock()
        scheduler.req_to_metadata_buffer_idx_allocator.free.side_effect = (
            lambda *_args: events.append("metadata")
        )
        release_kv.side_effect = lambda *_args, **_kwargs: events.append("kv")
        scheduler.output_streamer.stream_output.side_effect = (
            lambda *_args, **_kwargs: events.append("output")
        )
        scheduler._pending_prefill_external_aborts = []
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = [req]
        queue.scheduler = scheduler
        queue.req_to_metadata_buffer_idx_allocator = (
            scheduler.req_to_metadata_buffer_idx_allocator
        )
        scheduler.disagg_prefill_bootstrap_queue = queue

        queue.remove_aborted("draining")

        sender.abort.assert_called_once_with()
        self.assertEqual(manager._transfer_attempts, {(41, 3): 2})
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(req.pending_bootstrap)
        self.assertEqual(scheduler._pending_prefill_external_aborts, [req])
        release_kv.assert_not_called()
        scheduler.output_streamer.stream_output.assert_not_called()
        manager._maybe_ack_drained_abort.assert_not_called()

        # The real attempt drains, leaving the source-cleanup sentinel to hold
        # back Mooncake's automatic ABORT_ACK.
        manager._transfer_attempts[(41, 3)] = 1
        process_pending_prefill_external_aborts(scheduler)

        self.assertEqual(scheduler._pending_prefill_external_aborts, [])
        self.assertEqual(manager._transfer_attempts, {})
        release_kv.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
        sender.abort.assert_called_once_with()
        manager._maybe_ack_drained_abort.assert_called_once_with(41, 3)
        self.assertEqual(events, ["metadata", "kv", "output", "ack"])
        self.assertTrue(req.finished_output)

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache")
    def test_cleanup_retry_does_not_repeat_successful_resources(self, release_kv):
        manager = SimpleNamespace(
            _transfer_attempts={},
            request_status_lock=None,
            _maybe_ack_drained_abort=MagicMock(),
        )
        sender = SimpleNamespace(
            kv_mgr=manager,
            bootstrap_room=42,
            generation=4,
            abort=MagicMock(),
        )
        req = SimpleNamespace(
            rid="retry",
            disagg_kv_sender=sender,
            metadata_buffer_index=5,
            pending_bootstrap=False,
            req_pool_idx=11,
            kv=object(),
            mamba_pool_idx=None,
            return_logprob=False,
            finished_output=False,
            finished_reason=None,
        )
        scheduler = _base_scheduler(DisaggregationMode.PREFILL)
        scheduler.enable_hisparse = True
        scheduler.finish_hisparse_request = MagicMock()
        scheduler.tree_cache = object()
        scheduler.req_to_metadata_buffer_idx_allocator = MagicMock()
        scheduler._pending_prefill_external_aborts = [req]
        release_kv.side_effect = [RuntimeError("retry"), None]

        process_pending_prefill_external_aborts(scheduler)
        self.assertEqual(scheduler._pending_prefill_external_aborts, [req])
        self.assertFalse(getattr(req, "_disagg_external_abort_finalized", False))
        self.assertEqual(manager._transfer_attempts, {(42, 4): 1})
        manager._maybe_ack_drained_abort.assert_not_called()
        process_pending_prefill_external_aborts(scheduler)

        scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(5)
        scheduler.finish_hisparse_request.assert_called_once_with(req)
        self.assertEqual(release_kv.call_count, 2)
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
        self.assertEqual(manager._transfer_attempts, {})
        manager._maybe_ack_drained_abort.assert_called_once_with(42, 4)
        self.assertTrue(req._disagg_external_abort_finalized)
        self.assertTrue(req.finished_output)

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache")
    @patch(
        "sglang.srt.disaggregation.prefill.envs."
        "SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT.get",
        return_value=30.0,
    )
    def test_source_abort_deadline_retains_then_fail_stops(self, _timeout, release_kv):
        manager = SimpleNamespace(
            _transfer_attempts={(43, 5): 1},
            request_status_lock=None,
            _maybe_ack_drained_abort=MagicMock(),
        )
        sender = SimpleNamespace(
            kv_mgr=manager,
            bootstrap_room=43,
            generation=5,
            abort=MagicMock(),
        )
        req = SimpleNamespace(
            rid="deadline",
            disagg_kv_sender=sender,
            metadata_buffer_index=7,
            pending_bootstrap=False,
            req_pool_idx=13,
            kv=object(),
            mamba_pool_idx=None,
            return_logprob=False,
            finished_output=False,
            finished_reason=None,
        )
        scheduler = _base_scheduler(DisaggregationMode.PREFILL)
        scheduler.tree_cache = object()
        scheduler.req_to_metadata_buffer_idx_allocator = MagicMock()
        scheduler._pending_prefill_external_aborts = []
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = [req]
        queue.scheduler = scheduler
        queue.req_to_metadata_buffer_idx_allocator = (
            scheduler.req_to_metadata_buffer_idx_allocator
        )
        scheduler.disagg_prefill_bootstrap_queue = queue

        with patch(
            "sglang.srt.disaggregation.prefill.time.monotonic", return_value=100.0
        ):
            queue.remove_aborted("deadline")
        self.assertEqual(req._disagg_external_abort_drain_deadline, 130.0)

        with patch(
            "sglang.srt.disaggregation.prefill.time.monotonic", return_value=129.0
        ):
            process_pending_prefill_external_aborts(scheduler)
        self.assertEqual(scheduler._pending_prefill_external_aborts, [req])
        scheduler.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
        release_kv.assert_not_called()
        manager._maybe_ack_drained_abort.assert_not_called()

        with patch(
            "sglang.srt.disaggregation.prefill.time.monotonic", return_value=130.0
        ):
            with self.assertRaisesRegex(PrefillAbortDrainTimeout, "rid=deadline"):
                process_pending_prefill_external_aborts(scheduler)
        self.assertEqual(scheduler._pending_prefill_external_aborts, [req])
        scheduler.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
        release_kv.assert_not_called()
        scheduler.output_streamer.stream_output.assert_not_called()
        manager._maybe_ack_drained_abort.assert_not_called()
        self.assertFalse(getattr(req, "_disagg_external_abort_finalized", False))

    def test_non_output_pp_rank_does_not_emit_abort_output(self):
        sender = MagicMock(is_abort_drained=MagicMock(return_value=True))
        req = SimpleNamespace(
            rid="pp-stage",
            disagg_kv_sender=sender,
            metadata_buffer_index=-1,
            pending_bootstrap=False,
            req_pool_idx=None,
            kv=None,
            mamba_pool_idx=None,
            return_logprob=False,
            finished_output=False,
            finished_reason=None,
        )
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = [req]
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.scheduler = _base_scheduler(DisaggregationMode.PREFILL)
        queue.scheduler.ps.pp_rank = 1
        queue.scheduler.tree_cache = object()
        queue.scheduler.req_to_metadata_buffer_idx_allocator = (
            queue.req_to_metadata_buffer_idx_allocator
        )

        queue.remove_aborted("pp-stage")

        queue.scheduler.output_streamer.stream_output.assert_not_called()
        self.assertFalse(req.finished_output)
        self.assertTrue(req._disagg_external_abort_finalized)

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache")
    @patch("sglang.srt.disaggregation.prefill.maybe_release_metadata_buffer")
    def test_inflight_abort_detaches_owner_and_outputs_once(
        self, release_metadata, release_kv
    ):
        scheduler = _base_scheduler(DisaggregationMode.PREFILL)
        scheduler.req_to_metadata_buffer_idx_allocator = object()
        scheduler.tree_cache = object()
        scheduler._pending_prefill_external_aborts = []
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = []
        queue.scheduler = scheduler
        queue.req_to_metadata_buffer_idx_allocator = (
            scheduler.req_to_metadata_buffer_idx_allocator
        )
        scheduler.disagg_prefill_bootstrap_queue = queue
        sender = MagicMock()
        req = SimpleNamespace(
            rid="inflight-child",
            disagg_kv_sender=sender,
            metadata_buffer_index=6,
            pending_bootstrap=False,
            req_pool_idx=11,
            kv=object(),
            mamba_pool_idx=None,
            return_logprob=False,
            finished_output=False,
            finished_reason=None,
        )
        scheduler.disagg_prefill_inflight_queue = [req]

        scheduler.abort_request(AbortReq(rid="inflight"))
        scheduler.abort_request(AbortReq(rid="inflight"))

        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        sender.abort.assert_called_once_with()
        release_metadata.assert_called_once_with(
            req, scheduler.req_to_metadata_buffer_idx_allocator
        )
        release_kv.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        sender.clear.assert_not_called()
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)

    def test_owner_lease_renews_each_unique_live_owner_once(self):
        scheduler = _base_scheduler(DisaggregationMode.PREFILL)
        bootstrap_sender = MagicMock()
        inflight_sender = MagicMock()
        terminal_sender = MagicMock()

        bootstrap = SimpleNamespace(
            disagg_kv_sender=bootstrap_sender,
            bootstrap_host="decode",
            to_finish=None,
            finished_reason=None,
            finished_output=False,
        )
        inflight = SimpleNamespace(
            disagg_kv_sender=inflight_sender,
            bootstrap_host="decode",
            to_finish=None,
            finished_reason=None,
            finished_output=False,
        )
        terminal = SimpleNamespace(
            disagg_kv_sender=terminal_sender,
            bootstrap_host="decode",
            to_finish=None,
            finished_reason=FINISH_LENGTH(length=0),
            finished_output=False,
        )
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[bootstrap])
        scheduler.waiting_queue = [bootstrap, terminal]
        scheduler.disagg_prefill_inflight_queue = [inflight]
        scheduler.running_batch = SimpleNamespace(reqs=[inflight])
        scheduler.last_batch = SimpleNamespace(reqs=[terminal])
        scheduler.chunked_req = inflight

        renew_disagg_prefill_owner_leases(scheduler)

        bootstrap_sender.renew_decode_owner_lease.assert_called_once_with(
            phase="bootstrap"
        )
        inflight_sender.renew_decode_owner_lease.assert_called_once_with(
            phase="inflight"
        )
        terminal_sender.renew_decode_owner_lease.assert_not_called()


if __name__ == "__main__":
    unittest.main()
