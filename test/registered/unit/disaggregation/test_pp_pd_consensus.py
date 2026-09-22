import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue  # noqa: E402
from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    _pp_acknowledge_release_status,
    _pp_merge_pending_release_status,
    _pp_merge_transfer_status,
    _pp_ready_release_status,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPPPDConsensus(CustomTestCase):
    def test_prefill_bootstrap_passes_full_request_page_coordinates(self):
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(page_size=64)
        )
        sender = Mock()
        sender.pop_decode_prefix_len.return_value = 128
        req = SimpleNamespace(
            rid="prefix-hit",
            pending_bootstrap=True,
            metadata_buffer_index=7,
            time_stats=SimpleNamespace(set_bootstrap_done_time=lambda: None),
            disagg_kv_sender=sender,
            origin_input_ids=list(range(640)),
        )

        self.assertTrue(queue.finalize_bootstrap(req))
        self.assertEqual(req.start_send_idx, 128)
        sender.init.assert_called_once_with(
            8,
            7,
            num_request_pages=10,
            send_page_offset=2,
        )

    def test_prefill_bootstrap_rejects_unaligned_decode_prefix(self):
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(page_size=64)
        )
        sender = Mock()
        sender.pop_decode_prefix_len.return_value = 65
        req = SimpleNamespace(
            rid="unaligned-prefix",
            pending_bootstrap=True,
            metadata_buffer_index=7,
            time_stats=SimpleNamespace(set_bootstrap_done_time=lambda: None),
            disagg_kv_sender=sender,
            origin_input_ids=list(range(640)),
        )

        with self.assertRaisesRegex(AssertionError, "must be page aligned"):
            queue.finalize_bootstrap(req)
        sender.init.assert_not_called()

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
        )
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 1
        )

        with patch(
            "sglang.srt.disaggregation.prefill." "poll_and_all_reduce_attn_cp_tp_group",
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
