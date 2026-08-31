import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeRequest,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeReceiver:
    def __init__(self):
        self.clear_called = False
        self.generation = None
        self.kv_mgr = MagicMock()

    def clear(self):
        self.clear_called = True

    def abort(self):
        self.abort_called = True

    def failure_exception(self):
        return None

    def renew_bootstrap_lease(self):
        self.renew_called = getattr(self, "renew_called", 0) + 1
        return True


class TestDecodeQueueCleanup(CustomTestCase):
    def _make_hisparse_prealloc_abort_owner(
        self, *, rid, req_pool_idx, kv, mamba_pool_idx=None
    ):
        receiver = MagicMock()
        receiver.abort_notified = False
        req = SimpleNamespace(
            rid=rid,
            bootstrap_room=9,
            finished_reason=None,
            return_logprob=False,
            finished_output=False,
            req_pool_idx=req_pool_idx,
            kv=kv,
            mamba_pool_idx=mamba_pool_idx,
        )
        decode_req = DecodeRequest(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = [decode_req]
        # The add() slow path aliases the same owner in both collections.
        queue.pending_reqs = [decode_req]
        queue.retracted_queue = []
        queue.held_rebootstrap_reqs = []
        queue.metadata_buffers = SimpleNamespace(bootstrap_room={})
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_token_pool = MagicMock()
        queue.tree_cache = MagicMock()
        queue.transfer_queue = SimpleNamespace(
            enable_staging=False,
            staging_handler=None,
            _clean_hicache_prefetch_resources=MagicMock(),
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            finish_hisparse_request=MagicMock(),
            output_streamer=MagicMock(),
            ps=SimpleNamespace(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0),
        )
        return queue, decode_req, receiver

    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_remove_aborted_returns_wrappers_for_retracted_and_held(self, prepare):
        retracted = SimpleNamespace(
            rid="abort-retracted",
            bootstrap_room=1,
            finished_reason=None,
            return_logprob=False,
            finished_output=False,
            req_pool_idx=None,
            kv=None,
            mamba_pool_idx=None,
        )
        held = SimpleNamespace(
            rid="abort-held",
            bootstrap_room=2,
            finished_reason=None,
            return_logprob=False,
            finished_output=False,
            req_pool_idx=None,
            kv=None,
            mamba_pool_idx=None,
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = []
        queue.pending_reqs = []
        queue.retracted_queue = [retracted]
        queue.held_rebootstrap_reqs = [held]
        queue.metadata_buffers = SimpleNamespace(bootstrap_room={})
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tree_cache = MagicMock()
        queue.transfer_queue = SimpleNamespace(
            enable_staging=False,
            staging_handler=None,
            _clean_hicache_prefetch_resources=MagicMock(),
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=False,
            output_streamer=MagicMock(),
            ps=SimpleNamespace(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0),
        )

        removed = queue.remove_aborted("abort-")

        self.assertEqual([owner.req for owner in removed], [retracted, held])
        self.assertEqual(queue.retracted_queue, [])
        self.assertEqual(queue.held_rebootstrap_reqs, [])
        self.assertEqual(queue.scheduler.output_streamer.stream_output.call_count, 2)
        self.assertEqual(prepare.call_count, 2)

    def test_prealloc_abort_clears_receiver_before_removing_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="abort-prealloc",
            bootstrap_room=7,
            finished_reason=FINISH_ABORT("aborted"),
            return_logprob=False,
            finished_output=False,
            req_pool_idx=None,
            kv=None,
            mamba_pool_idx=None,
        )
        decode_req = DecodeRequest(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.held_rebootstrap_reqs = []
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue.metadata_buffers = SimpleNamespace(bootstrap_room={})
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tree_cache = MagicMock()
        queue.transfer_queue = SimpleNamespace(
            enable_staging=False,
            staging_handler=None,
            _clean_hicache_prefetch_resources=MagicMock(),
        )

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        scheduler.ps.pp_rank = 0
        scheduler.ps.attn_tp_rank = 0
        scheduler.ps.attn_cp_rank = 0
        queue.scheduler = scheduler

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertFalse(hasattr(receiver, "renew_called"))
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_capacity_blocked_request_renews_bootstrap_lease(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(rid="capacity-blocked", finished_reason=None)
        decode_req = SimpleNamespace(
            req=req, kv_receiver=receiver, waiting_for_input=True
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = [decode_req]

        queue._renew_bootstrap_leases(set())

        self.assertEqual(receiver.renew_called, 1)

    def test_aborted_request_is_excluded_from_bootstrap_lease_renewal(self):
        receiver = FakeReceiver()
        decode_req = SimpleNamespace(
            req=SimpleNamespace(rid="aborted"),
            kv_receiver=receiver,
            waiting_for_input=True,
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = [decode_req]

        queue._renew_bootstrap_leases({0})

        self.assertFalse(hasattr(receiver, "renew_called"))

    def test_prealloc_abort_also_drops_from_pending_reqs(self):
        # Same DecodeRequest lives in both queue and pending_reqs (add() slow
        # path). Aborting must drop it from both, and compare by identity since
        # DecodeRequest's dataclass __eq__ would compare the tensor receiver.
        class BadEqReceiver(FakeReceiver):
            def __eq__(self, other):
                raise TypeError("use identity comparison, not value equality")

            __hash__ = object.__hash__

        receiver = BadEqReceiver()
        req = SimpleNamespace(
            rid="abort-shared",
            bootstrap_room=8,
            finished_reason=FINISH_ABORT("aborted"),
            return_logprob=False,
            finished_output=False,
            req_pool_idx=None,
            kv=None,
            mamba_pool_idx=None,
        )
        decode_req = DecodeRequest(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]  # same object, dual ownership
        queue.retracted_queue = []
        queue.held_rebootstrap_reqs = []
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue.metadata_buffers = SimpleNamespace(bootstrap_room={})
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tree_cache = MagicMock()
        queue.transfer_queue = SimpleNamespace(
            enable_staging=False,
            staging_handler=None,
            _clean_hicache_prefetch_resources=MagicMock(),
        )

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        scheduler.ps.pp_rank = 0
        scheduler.ps.attn_tp_rank = 0
        scheduler.ps.attn_cp_rank = 0
        queue.scheduler = scheduler

        # Must not raise on the receiver __eq__ above.
        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(all(r is not decode_req for r in queue.pending_reqs))
        self.assertIsNone(decode_req.kv_receiver)

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    def test_hisparse_external_abort_before_allocation_is_idempotent(
        self, mock_release_kv_cache
    ):
        queue, decode_req, receiver = self._make_hisparse_prealloc_abort_owner(
            rid="hisparse-unallocated", req_pool_idx=None, kv=None
        )

        removed = queue.remove_aborted("hisparse-unallocated")
        removed_again = queue.remove_aborted("hisparse-unallocated")
        queue._abort_and_release(decode_req)

        self.assertEqual(len(removed), 1)
        self.assertIs(removed[0], decode_req)
        self.assertEqual(removed_again, [])
        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.pending_reqs, [])
        receiver.abort.assert_called_once_with()
        receiver.clear.assert_called_once_with()
        self.assertIsNone(decode_req.kv_receiver)
        queue.scheduler.finish_hisparse_request.assert_not_called()
        mock_release_kv_cache.assert_not_called()
        queue.req_to_token_pool.free.assert_not_called()
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [decode_req.req], decode_req.req.return_logprob
        )

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    def test_hisparse_external_abort_releases_allocated_owner_once(
        self, mock_release_kv_cache
    ):
        cleanup_order = []
        queue, decode_req, receiver = self._make_hisparse_prealloc_abort_owner(
            rid="hisparse-allocated", req_pool_idx=3, kv=object()
        )
        receiver.abort.side_effect = lambda: cleanup_order.append("receiver.abort")
        receiver.clear.side_effect = lambda: cleanup_order.append("receiver.clear")
        queue.scheduler.finish_hisparse_request.side_effect = (
            lambda _req: cleanup_order.append("hisparse.finish")
        )
        mock_release_kv_cache.side_effect = (
            lambda *_args, **_kwargs: cleanup_order.append("kv.release")
        )
        queue.scheduler.output_streamer.stream_output.side_effect = (
            lambda *_args, **_kwargs: cleanup_order.append("output")
        )

        removed = queue.remove_aborted("hisparse-allocated")
        queue.remove_aborted("hisparse-allocated")
        queue._abort_and_release(decode_req)

        self.assertEqual(len(removed), 1)
        self.assertIs(removed[0], decode_req)
        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.pending_reqs, [])
        receiver.abort.assert_called_once_with()
        receiver.clear.assert_called_once_with()
        queue.scheduler.finish_hisparse_request.assert_called_once_with(decode_req.req)
        mock_release_kv_cache.assert_called_once_with(
            decode_req.req, queue.tree_cache, is_insert=False
        )
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [decode_req.req], decode_req.req.return_logprob
        )
        self.assertEqual(
            cleanup_order,
            [
                "receiver.abort",
                "hisparse.finish",
                "kv.release",
                "receiver.clear",
                "output",
            ],
        )

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    def test_hisparse_external_abort_rolls_back_req_pool_only_allocation(
        self, mock_release_kv_cache
    ):
        queue, decode_req, receiver = self._make_hisparse_prealloc_abort_owner(
            rid="hisparse-req-pool-only", req_pool_idx=4, kv=None
        )

        removed = queue.remove_aborted("hisparse-req-pool-only")
        queue._abort_and_release(decode_req)

        self.assertEqual(removed, [decode_req])
        receiver.abort.assert_called_once_with()
        receiver.clear.assert_called_once_with()
        queue.scheduler.finish_hisparse_request.assert_not_called()
        mock_release_kv_cache.assert_not_called()
        queue.req_to_token_pool.free_mamba_cache.assert_not_called()
        queue.req_to_token_pool.free.assert_called_once_with(decode_req.req)
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [decode_req.req], decode_req.req.return_logprob
        )

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    def test_hisparse_external_abort_rolls_back_partial_allocation(
        self, mock_release_kv_cache
    ):
        mamba_pool_idx = object()
        queue, decode_req, receiver = self._make_hisparse_prealloc_abort_owner(
            rid="hisparse-partial",
            req_pool_idx=4,
            kv=None,
            mamba_pool_idx=mamba_pool_idx,
        )

        removed = queue.remove_aborted("hisparse-partial")
        queue.remove_aborted("hisparse-partial")
        queue._abort_and_release(decode_req)

        self.assertEqual(len(removed), 1)
        self.assertIs(removed[0], decode_req)
        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.pending_reqs, [])
        receiver.abort.assert_called_once_with()
        receiver.clear.assert_called_once_with()
        queue.scheduler.finish_hisparse_request.assert_not_called()
        mock_release_kv_cache.assert_not_called()
        self.assertEqual(
            queue.req_to_token_pool.method_calls,
            [call.free_mamba_cache(decode_req.req), call.free(decode_req.req)],
        )
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [decode_req.req], decode_req.req.return_logprob
        )

    @patch("sglang.srt.disaggregation.decode.logger.exception")
    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    def test_hisparse_external_abort_notification_failure_still_rolls_back(
        self, mock_release_kv_cache, mock_log_exception
    ):
        queue, decode_req, receiver = self._make_hisparse_prealloc_abort_owner(
            rid="hisparse-abort-notify-failure", req_pool_idx=5, kv=None
        )
        receiver.abort.side_effect = RuntimeError("transport unavailable")

        removed = queue.remove_aborted("hisparse-abort-notify-failure")
        queue._abort_and_release(decode_req)

        self.assertEqual(removed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.pending_reqs, [])
        receiver.abort.assert_called_once_with()
        receiver.clear.assert_called_once_with()
        mock_log_exception.assert_called_once()
        queue.req_to_token_pool.free.assert_called_once_with(decode_req.req)
        queue.scheduler.finish_hisparse_request.assert_not_called()
        mock_release_kv_cache.assert_not_called()
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [decode_req.req], decode_req.req.return_logprob
        )

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    def test_hisparse_external_abort_output_side_effect_is_not_retried(
        self, mock_release_kv_cache
    ):
        queue, decode_req, receiver = self._make_hisparse_prealloc_abort_owner(
            rid="hisparse-output-side-effect", req_pool_idx=None, kv=None
        )

        def output_then_fail(reqs, _return_logprob):
            reqs[0].finished_output = True
            raise RuntimeError("detokenizer transport failed after accept")

        queue.scheduler.output_streamer.stream_output.side_effect = output_then_fail

        with self.assertRaisesRegex(RuntimeError, "after accept"):
            queue.remove_aborted("hisparse-output-side-effect")
        queue._abort_and_release(decode_req)

        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [decode_req.req], decode_req.req.return_logprob
        )
        self.assertTrue(decode_req.abort_output_emitted)
        self.assertTrue(decode_req.req.finished_output)
        receiver.abort.assert_called_once_with()
        receiver.clear.assert_called_once_with()
        mock_release_kv_cache.assert_not_called()

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    def test_hisparse_external_abort_rejects_kv_only_owner_before_detach(
        self, mock_release_kv_cache
    ):
        queue, decode_req, receiver = self._make_hisparse_prealloc_abort_owner(
            rid="hisparse-kv-only", req_pool_idx=None, kv=object()
        )

        with self.assertRaisesRegex(RuntimeError, "no request-pool owner"):
            queue.remove_aborted("hisparse-kv-only")

        self.assertEqual(queue.queue, [decode_req])
        self.assertEqual(queue.pending_reqs, [decode_req])
        receiver.abort.assert_not_called()
        receiver.clear.assert_not_called()
        queue.scheduler.finish_hisparse_request.assert_not_called()
        mock_release_kv_cache.assert_not_called()
        queue.req_to_token_pool.free.assert_not_called()
        queue.scheduler.output_streamer.stream_output.assert_not_called()

        queue.queue = []
        queue.pending_reqs = []
        queue.held_rebootstrap_reqs = [decode_req.req]
        with self.assertRaisesRegex(RuntimeError, "no request-pool owner"):
            queue.remove_aborted("hisparse-kv-only")
        self.assertEqual(queue.held_rebootstrap_reqs, [decode_req.req])

    def test_ensure_prefill_info_tolerates_cleared_receiver(self):
        # A req whose kv_receiver was already cleared must not crash on .abort().
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 1
        queue._ensure_retry_interval = 0
        queue._ensure_retry_count = {"127.0.0.1:11500": 0}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = False

        cleared_req = SimpleNamespace(
            req=SimpleNamespace(rid="cleared"), kv_receiver=None
        )
        addr_to_reqs = {"127.0.0.1:11500": [cleared_req]}

        ready, remaining = queue._ensure_prefill_info(addr_to_reqs)

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_transfer_failure_clears_receiver_before_removing_request(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="failed-transfer",
            bootstrap_room=7,
            return_logprob=False,
            finished_output=False,
            req_pool_idx=2,
            kv=object(),
            mamba_pool_idx=None,
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            metadata_sent=False,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_deferred_kv_release = False
        queue.deferred_abort_holds = []
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        scheduler.metrics_reporter.enable_metrics = False
        scheduler.ps.pp_rank = 0
        scheduler.ps.attn_tp_rank = 0
        scheduler.ps.attn_cp_rank = 0
        queue.scheduler = scheduler

        mock_poll.return_value = [KVPoll.Failed]

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )
        mock_prepare_abort.assert_called_once()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_transfer_success_budget_leaves_excess_requests_queued(self, mock_poll):
        decode_reqs = []
        for i in range(3):
            decode_reqs.append(
                SimpleNamespace(
                    req=SimpleNamespace(rid=f"success-{i}", finished_reason=None),
                    kv_receiver=FakeReceiver(),
                    metadata_buffer_index=i,
                    hicache_restore_status=HiCacheRestoreResult.READY,
                )
            )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = decode_reqs
        queue.enable_staging = False
        queue.deferred_abort_holds = []
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 3)
        queue._commit_transfer_to_req = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = True
        scheduler.server_args = MagicMock()
        queue.scheduler = scheduler

        mock_poll.return_value = [KVPoll.Success] * 3

        transferred = queue.pop_transferred(max_successes=2)

        self.assertEqual([req.rid for req in transferred], ["success-0", "success-1"])
        self.assertEqual([entry.req.rid for entry in queue.queue], ["success-2"])
        self.assertEqual(queue._commit_transfer_to_req.call_count, 2)
        self.assertEqual(
            queue.req_to_metadata_buffer_idx_allocator.free.call_args_list,
            [call(0), call(1)],
        )

    def test_hisparse_direct_admission_capacity_uses_target_draft_minimum(self):
        scheduler = Scheduler.__new__(Scheduler)

        def coordinator(available, padded):
            allocator = SimpleNamespace(
                hisparse_attn_allocator=SimpleNamespace(
                    available_size=MagicMock(return_value=available)
                )
            )
            return SimpleNamespace(
                token_to_kv_pool_allocator=allocator,
                padded_buffer_size=padded,
            )

        scheduler.hisparse_coordinator = coordinator(18_624, 6_208)
        scheduler.draft_hisparse_coordinator = coordinator(12_416, 6_208)

        self.assertEqual(scheduler.hisparse_direct_admission_capacity(), 2)

    def test_hisparse_direct_admission_capacity_counts_mirrored_slot_once(self):
        scheduler = Scheduler.__new__(Scheduler)
        shared_physical_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=18_624)
        )
        shared_allocator = SimpleNamespace(
            hisparse_attn_allocator=shared_physical_allocator
        )

        def coordinator(padded):
            return SimpleNamespace(
                token_to_kv_pool_allocator=shared_allocator,
                padded_buffer_size=padded,
            )

        scheduler.hisparse_coordinator = coordinator(6_208)
        scheduler.draft_hisparse_coordinator = coordinator(6_208)

        # Target and draft mirror one numerical slot namespace. Their KV tensors
        # are distinct, but a shared allocator must reserve the slot only once.
        self.assertEqual(scheduler.hisparse_direct_admission_capacity(), 3)
        shared_physical_allocator.available_size.assert_called_once_with()

    @patch("sglang.srt.disaggregation.decode." "poll_and_all_reduce_attn_cp_tp_group")
    def test_pp_transfer_success_ignores_hicache_state_when_hicache_disabled(
        self, mock_poll
    ):
        decode_req = SimpleNamespace(
            req=SimpleNamespace(rid="ready", bootstrap_host="2.2.2.2"),
            kv_receiver=FakeReceiver(),
            metadata_buffer_index=0,
            hicache_restore_status=HiCacheRestoreResult.PENDING,
        )
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.scheduler = SimpleNamespace(
            enable_decode_hicache=False,
            attn_cp_cpu_group=object(),
            attn_tp_cpu_group=object(),
            server_args=object(),
        )
        mock_poll.return_value = [KVPoll.Success]

        success, failed = queue.get_transferred_status_for_pp()

        self.assertEqual(success, ["ready"])
        self.assertEqual(failed, [])
        self.assertEqual(mock_poll.call_args.kwargs["ordered_keys"], ["ready"])

    def test_retracted_decode_requests_keep_scheduler_non_idle(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.enable_overlap = False
        scheduler.ps = ParallelState.trivial()
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[object()]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            queue=[], has_pending_deferred_abort_holds=lambda: False
        )
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False

        self.assertFalse(scheduler.is_fully_idle())

    def test_deferred_decode_abort_hold_keeps_scheduler_non_idle(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.enable_overlap = False
        scheduler.ps = ParallelState.trivial()
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            queue=[], has_pending_deferred_abort_holds=lambda: True
        )
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False

        self.assertFalse(scheduler.is_fully_idle())

    def test_pp_decode_tick_resolves_holds_without_release_status(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            resolve_deferred_abort_holds=MagicMock()
        )

        result = scheduler.process_decode_transfer_queue(None)

        self.assertIsNone(result)
        scheduler.disagg_decode_transfer_queue.resolve_deferred_abort_holds.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
