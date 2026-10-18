"""Source-slot lifetime regressions; execute in the target Docker image.

Test doubles distinguish lease bookkeeping from actual NIXL/GPU quiescence.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.prefill import (
    SchedulerDisaggregationPrefillMixin,
    maybe_release_metadata_buffer,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Scheduler(SchedulerDisaggregationPrefillMixin):
    pass


class TestPrefillSourceRelease(unittest.TestCase):
    def _make(self, pending=True):
        scheduler = _Scheduler()
        scheduler.tree_cache = object()
        scheduler.disagg_prefill_drain_releases = []
        scheduler.req_to_metadata_buffer_idx_allocator = MagicMock()
        manager = SimpleNamespace(
            has_pending_source_reads=MagicMock(return_value=pending)
        )
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(kv_manager=manager)
        req = SimpleNamespace(
            bootstrap_room=7,
            disagg_kv_sender=SimpleNamespace(kv_mgr=manager),
            disagg_prefill_release_deferred=False,
            kv=SimpleNamespace(holds_kv=True, holds_mamba=False),
            metadata_buffer_index=3,
            inflight_middle_chunks=0,
            disagg_prefill_forward_done=None,
        )
        return scheduler, req, manager

    def test_ack_retry_runs_even_when_retired_source_hold_list_is_empty(self):
        scheduler, _, manager = self._make(pending=False)
        manager.retry_drained_abort_acks = MagicMock()
        scheduler.resolve_prefill_drain_releases()
        manager.retry_drained_abort_acks.assert_called_once()
        self.assertEqual(scheduler.disagg_prefill_drain_releases, [])

    def test_hold_keeps_kv_state_and_aux_until_drain(self):
        scheduler, req, manager = self._make()

        def release(request, cache, is_insert):
            self.assertIs(request, req)
            self.assertIs(cache, scheduler.tree_cache)
            self.assertFalse(is_insert)
            self.assertEqual(req.metadata_buffer_index, 3)
            req.kv.holds_kv = False

        with patch(
            "sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=release
        ) as free:
            self.assertFalse(scheduler.release_prefill_kv_after_drain(req))
            self.assertFalse(
                scheduler.release_prefill_kv_after_drain(req, is_insert=False)
            )
            maybe_release_metadata_buffer(
                req, scheduler.req_to_metadata_buffer_idx_allocator
            )
            scheduler.resolve_prefill_drain_releases()
            free.assert_not_called()
            scheduler.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
            self.assertEqual(len(scheduler.disagg_prefill_drain_releases), 1)

            manager.has_pending_source_reads.return_value = False
            scheduler.resolve_prefill_drain_releases()
            scheduler.resolve_prefill_drain_releases()
            free.assert_called_once()

        scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        self.assertEqual(req.metadata_buffer_index, -1)
        self.assertFalse(req.disagg_prefill_release_deferred)
        self.assertEqual(scheduler.disagg_prefill_drain_releases, [])

    def test_quiescent_retirement_keeps_immediate_owner_release(self):
        scheduler, req, _ = self._make(pending=False)
        with patch("sglang.srt.disaggregation.prefill.release_kv_cache") as free:
            self.assertTrue(
                scheduler.release_prefill_kv_after_drain(req, is_insert=False)
            )
            free.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
        self.assertEqual(scheduler.disagg_prefill_drain_releases, [])
        self.assertFalse(req.disagg_prefill_release_deferred)

    def test_abort_after_drain_still_narrows_held_insertion(self):
        scheduler, req, manager = self._make()
        scheduler.release_prefill_kv_after_drain(req)
        manager.has_pending_source_reads.return_value = False
        with patch("sglang.srt.disaggregation.prefill.release_kv_cache") as free:
            scheduler.release_prefill_kv_after_drain(req, is_insert=False)
            free.assert_not_called()
            scheduler.resolve_prefill_drain_releases()
            free.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)

    def test_failed_owner_release_keeps_hold_and_metadata_visible(self):
        scheduler, req, manager = self._make()
        scheduler.release_prefill_kv_after_drain(req, is_insert=False)
        manager.has_pending_source_reads.return_value = False
        with patch(
            "sglang.srt.disaggregation.prefill.release_kv_cache",
            side_effect=RuntimeError("owner free failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "owner free failed"):
                scheduler.resolve_prefill_drain_releases()
        self.assertTrue(req.disagg_prefill_release_deferred)
        self.assertEqual(len(scheduler.disagg_prefill_drain_releases), 1)
        self.assertEqual(req.metadata_buffer_index, 3)
        scheduler.req_to_metadata_buffer_idx_allocator.free.assert_not_called()

    def test_inflight_forward_also_holds_source_slots(self):
        scheduler, req, _ = self._make(pending=False)
        req.inflight_middle_chunks = 1
        with patch("sglang.srt.disaggregation.prefill.release_kv_cache") as free:
            self.assertFalse(
                scheduler.release_prefill_kv_after_drain(req, is_insert=False)
            )
            scheduler.resolve_prefill_drain_releases()
            free.assert_not_called()
            req.inflight_middle_chunks = 0
            scheduler.resolve_prefill_drain_releases()
            free.assert_called_once()
        self.assertEqual(scheduler.disagg_prefill_drain_releases, [])

    def test_aborted_overlap_result_does_not_send_or_rewrite_output(self):
        scheduler, req, _ = self._make()
        req.to_finish = FINISH_ABORT("cancelled")
        req.finished_reason = None
        req.return_logprob = False
        req.rid = "cancelled"
        req.output_ids = [5]
        req.disagg_kv_sender.abort = MagicMock()
        req.disagg_kv_sender.clear = MagicMock()
        scheduler.disagg_prefill_pending_chunk_rids = {req.rid}
        scheduler.spec_algorithm = SimpleNamespace(is_eagle=lambda: False)
        scheduler.batch_result_processor = MagicMock()
        scheduler.metrics_reporter = MagicMock()
        scheduler.output_streamer = MagicMock()
        scheduler.send_kv_chunk = MagicMock()
        scheduler.disagg_prefill_inflight_queue = []
        batch = SimpleNamespace(
            reqs=[req], spec_info=None, prefill_stats=None, dp_cooperation_info=None
        )
        result = SimpleNamespace(
            logits_output=MagicMock(),
            next_token_ids=torch.tensor([99]),
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            copy_done=MagicMock(),
            auxiliary_host_output=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            next_draft_input=None,
            can_run_cuda_graph=False,
        )
        with patch("sglang.srt.disaggregation.prefill.release_kv_cache") as free:
            scheduler.process_batch_result_disagg_prefill(batch, result)
            free.assert_not_called()
        self.assertEqual(req.output_ids, [5])
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        scheduler.send_kv_chunk.assert_not_called()
        scheduler.output_streamer.stream_output.assert_called_once()
        scheduler.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
        self.assertTrue(req.disagg_prefill_release_deferred)

    def test_latest_forward_event_blocks_release_after_old_result(self):
        scheduler, req, _ = self._make(pending=False)
        req.disagg_prefill_forward_done = MagicMock()
        req.disagg_prefill_forward_done.query.return_value = False
        with patch("sglang.srt.disaggregation.prefill.release_kv_cache") as free:
            self.assertFalse(
                scheduler.release_prefill_kv_after_drain(req, is_insert=False)
            )
            scheduler.resolve_prefill_drain_releases()
            free.assert_not_called()
            req.disagg_prefill_forward_done.query.return_value = True
            scheduler.resolve_prefill_drain_releases()
            free.assert_called_once()

    def test_source_hold_prevents_fully_idle_and_flush_eligibility(self):
        scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(is_empty=lambda: True),
            chunked_req=None,
            dllm_manager=SimpleNamespace(any_staging_reqs=lambda: False),
            last_batch=None,
            enable_overlap=False,
            _pp_microbatches_drained=lambda: True,
            waiting_queue=[],
            grammar_manager=SimpleNamespace(grammar_queue=[]),
            disaggregation_mode=DisaggregationMode.PREFILL,
            disagg_prefill_inflight_queue=[],
            disagg_prefill_bootstrap_queue=SimpleNamespace(queue=[]),
            disagg_prefill_drain_releases=[(object(), False)],
            enable_hisparse=False,
            enable_hierarchical_cache=False,
        )
        self.assertFalse(Scheduler.is_fully_idle(scheduler))

    def test_optimistic_retry_waits_latest_forward_before_free(self):
        scheduler, req, _ = self._make(pending=False)
        req.disagg_prefill_forward_done = MagicMock()
        event = req.disagg_prefill_forward_done
        order = []
        event.synchronize.side_effect = lambda: order.append("producer")
        req.reset_for_retract = MagicMock()
        req.time_stats = MagicMock()
        req.prefill_attempt_count = 1
        req.rid = "retry"
        scheduler.waiting_queue = []
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.disagg_prefill_pending_chunk_rids = set()
        with (
            patch(
                "sglang.srt.disaggregation.prefill.get_disagg",
                return_value=SimpleNamespace(optimistic_prefill_attempts=2),
            ),
            patch(
                "sglang.srt.disaggregation.prefill.maybe_cache_unfinished_req",
                side_effect=lambda *args: order.append("cache"),
            ),
            patch(
                "sglang.srt.disaggregation.prefill.release_kv_cache",
                side_effect=lambda *args: order.append("free"),
            ),
        ):
            scheduler.optimistic_release_and_requeue(req)
        self.assertEqual(order, ["producer", "cache", "free"])
        self.assertIsNone(req.disagg_prefill_forward_done)
        self.assertEqual(scheduler.waiting_queue, [req])

    def test_optimistic_retry_refuses_live_transfer_source(self):
        scheduler, req, _ = self._make()
        with (
            patch(
                "sglang.srt.disaggregation.prefill.get_disagg",
                return_value=SimpleNamespace(optimistic_prefill_attempts=2),
            ),
            patch("sglang.srt.disaggregation.prefill.release_kv_cache") as free,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "still owns transfer source leases"
            ):
                scheduler.optimistic_release_and_requeue(req)
            free.assert_not_called()


if __name__ == "__main__":
    unittest.main()
