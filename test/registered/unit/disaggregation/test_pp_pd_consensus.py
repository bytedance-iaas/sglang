import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.prefill import (  # noqa: E402
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    _pp_acknowledge_release_status,
    _pp_merge_pending_release_status,
    _pp_merge_transfer_status,
    _pp_ready_release_status,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPPPDConsensus(CustomTestCase):
    def test_empty_bootstrap_queue_still_participates_in_poll_consensus(self):
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = []
        queue.scheduler = SimpleNamespace(
            attn_cp_cpu_group=object(),
            attn_tp_cpu_group=object(),
            pp_disagg_prefill_poll_groups={
                "bootstrap": {"attn_cp": object(), "attn_tp": object()}
            },
        )
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 1
        )

        with patch(
            "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
            return_value=[],
        ) as poll:
            self.assertEqual(queue.get_ready_bootstrapped_rids_for_pp(), ([], []))

        poll.assert_called_once_with(
            [],
            queue.scheduler.pp_disagg_prefill_poll_groups["bootstrap"]["attn_cp"],
            queue.scheduler.pp_disagg_prefill_poll_groups["bootstrap"]["attn_tp"],
            ordered_keys=[],
        )

    def test_transfer_failure_overrides_ordered_success_intersection(self):
        """A failure on one PP rank must terminate an otherwise successful rid."""
        status = _pp_merge_transfer_status(
            previous=(["req-a", "req-b", "req-c"], ["req-x"]),
            current=(["req-c", "req-a", "req-b"], ["req-b", "req-y"]),
        )

        self.assertEqual(
            status,
            (["req-a", "req-c"], ["req-x", "req-b", "req-y"]),
        )

    def test_bootstrap_probe_respects_local_metadata_credit_prefix(self):
        """A slower PP rank must not advertise requests it cannot admit."""
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = [
            SimpleNamespace(
                rid="req-failed",
                metadata_buffer_index=-1,
                disagg_kv_sender=object(),
            ),
            SimpleNamespace(
                rid="req-ready",
                metadata_buffer_index=-1,
                disagg_kv_sender=object(),
            ),
            SimpleNamespace(
                rid="req-blocked",
                metadata_buffer_index=-1,
                disagg_kv_sender=object(),
            ),
        ]
        queue.scheduler = SimpleNamespace(
            attn_cp_cpu_group=object(),
            attn_tp_cpu_group=object(),
            pp_disagg_prefill_poll_groups={
                "bootstrap": {"attn_cp": object(), "attn_tp": object()}
            },
        )
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 1
        )

        with patch(
            "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
            return_value=[
                KVPoll.Failed,
                KVPoll.WaitingForInput,
                KVPoll.WaitingForInput,
            ],
        ):
            good_rids, failed_rids = queue.get_ready_bootstrapped_rids_for_pp()

        self.assertEqual(good_rids, ["req-ready"])
        self.assertEqual(failed_rids, ["req-failed"])
        self.assertEqual(
            [req.metadata_buffer_index for req in queue.queue],
            [-1, -1, -1],
        )

    def test_transfer_probe_uses_dedicated_pp_poll_groups(self):
        req = SimpleNamespace(
            rid="req-transfer",
            disagg_kv_sender=object(),
            pending_bootstrap=False,
        )
        scheduler = SimpleNamespace(
            disagg_prefill_inflight_queue=[req],
            pp_disagg_prefill_poll_groups={
                "transfer": {"attn_cp": object(), "attn_tp": object()}
            },
        )

        with patch(
            "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
            return_value=[KVPoll.Success],
        ) as poll:
            status = SchedulerDisaggregationPrefillMixin.get_transferred_rids(scheduler)

        self.assertEqual(status, (["req-transfer"], []))
        poll.assert_called_once_with(
            [req.disagg_kv_sender],
            scheduler.pp_disagg_prefill_poll_groups["transfer"]["attn_cp"],
            scheduler.pp_disagg_prefill_poll_groups["transfer"]["attn_tp"],
            ordered_keys=["req-transfer"],
        )

    def test_pp_release_consumes_consensus_without_repolling(self):
        sender = SimpleNamespace(clear=Mock())
        time_stats = SimpleNamespace(
            set_prefill_kv_transfer_finish_time=Mock(),
            set_completion_time=Mock(),
        )
        req = SimpleNamespace(
            rid="req-release",
            disagg_kv_sender=sender,
            pending_bootstrap=False,
            finished_reason=None,
            return_logprob=False,
            bootstrap_host="2.2.2.2",
            time_stats=time_stats,
        )
        scheduler = SimpleNamespace(
            disagg_prefill_inflight_queue=[req],
            tree_cache=object(),
            output_streamer=SimpleNamespace(stream_output=Mock()),
            req_to_metadata_buffer_idx_allocator=object(),
        )

        with (
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group"
            ) as poll,
            patch("sglang.srt.disaggregation.prefill.release_kv_cache") as release_kv,
            patch(
                "sglang.srt.disaggregation.prefill.maybe_release_metadata_buffer"
            ) as release_metadata,
        ):
            done = SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                scheduler, (["req-release"], [])
            )

        self.assertEqual(done, [req])
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        poll.assert_not_called()
        release_kv.assert_called_once_with(req, scheduler.tree_cache)
        release_metadata.assert_called_once_with(
            req, scheduler.req_to_metadata_buffer_idx_allocator
        )
        sender.clear.assert_called_once_with()

    def test_release_authority_waits_for_local_stage_and_acks_exactly_once(self):
        pending = _pp_merge_pending_release_status(
            pending=(["req-old"], []),
            incoming=(["req-new", "req-failed"], ["req-failed"]),
            committed_rids={"req-old"},
        )

        self.assertEqual(pending, (["req-new"], ["req-failed"]))
        self.assertIsNone(
            _pp_ready_release_status(pending, success_ready_rids=["other"])
        )

        ready = _pp_ready_release_status(
            pending,
            success_ready_rids=["req-new"],
            failure_ready_rids=["req-failed"],
        )
        self.assertEqual(ready, pending)
        self.assertIsNone(
            _pp_acknowledge_release_status(pending, ["req-new", "req-failed"])
        )


if __name__ == "__main__":
    unittest.main()
