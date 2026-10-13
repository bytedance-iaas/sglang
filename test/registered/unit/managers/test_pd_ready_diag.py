"""CPU contracts for the opt-in PD request-ready diagnostic."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    _maybe_log_pd_ready_first_decode,
)
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPdReadyDiag(CustomTestCase):
    @patch("sglang.srt.observability.req_time_stats._PD_READY_DIAG", True)
    @patch(
        "sglang.srt.observability.req_time_stats.get_global_tracing_enabled",
        return_value=False,
    )
    def test_diag_records_decode_schedule_without_enabling_trace(self, _tracing):
        req = SimpleNamespace(
            time_stats=SimpleNamespace(set_last_scheduled_time=MagicMock())
        )
        batch = SimpleNamespace(
            reqs=[req],
            forward_mode=SimpleNamespace(
                is_decode=lambda: True, is_prefill=lambda: False
            ),
        )

        set_schedule_time_batch(batch)

        req.time_stats.set_last_scheduled_time.assert_called_once()
        self.assertEqual(
            req.time_stats.set_last_scheduled_time.call_args.args[0],
            batch.forward_mode,
        )

    @patch(
        "sglang.srt.managers.scheduler_components.batch_result_processor._PD_READY_DIAG",
        True,
    )
    @patch(
        "sglang.srt.managers.scheduler_components.batch_result_processor.get_parallel",
        return_value=SimpleNamespace(attn_dp_rank=2, attn_tp_rank=3),
    )
    @patch(
        "sglang.srt.managers.scheduler_components.batch_result_processor.logger.info"
    )
    def test_first_decode_log_uses_decode_local_timestamps(self, log_info, _parallel):
        stats = SimpleNamespace(
            disagg_mode=DisaggregationMode.DECODE,
            decode_transfer_queue_entry_time=10.0,
            wait_queue_entry_time=11.0,
            last_decode_scheduled_time=11.25,
            last_decode_finish_time=11.5,
        )
        req = SimpleNamespace(rid="rid-1", bootstrap_room=42, time_stats=stats)

        _maybe_log_pd_ready_first_decode(req, was_first_decode=True)

        log_info.assert_called_once()
        self.assertEqual(log_info.call_args.args[1:5], ("rid-1", 42, 2, 3))
        self.assertEqual(log_info.call_args.args[5:], (1000.0, 250.0, 250.0, 500.0))

    @patch(
        "sglang.srt.managers.scheduler_components.batch_result_processor._PD_READY_DIAG",
        True,
    )
    @patch(
        "sglang.srt.managers.scheduler_components.batch_result_processor.logger.info"
    )
    def test_non_first_decode_is_silent(self, log_info):
        req = SimpleNamespace(
            time_stats=SimpleNamespace(disagg_mode=DisaggregationMode.DECODE)
        )

        _maybe_log_pd_ready_first_decode(req, was_first_decode=False)

        log_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
