"""Regression tests for PP disaggregated chunked-prefill cancellation."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPPDisaggPrefillAbort(CustomTestCase):
    def test_pp_loop_consumes_pending_abort_before_scheduling(self):
        class AbortConsumed(Exception):
            pass

        scheduler = SimpleNamespace(
            pp_loop_size=1,
            ps=SimpleNamespace(pp_size=1),
            running_mbs=[MagicMock()],
            last_mbs=[None],
            request_receiver=SimpleNamespace(
                recv_requests=MagicMock(return_value=[])
            ),
            process_input_requests=MagicMock(),
            process_pending_chunked_abort=MagicMock(
                side_effect=AbortConsumed
            ),
            init_pp_loop_state=MagicMock(),
        )

        with self.assertRaises(AbortConsumed):
            SchedulerPPMixin.event_loop_pp_disagg_prefill.__wrapped__(scheduler)

        scheduler.process_pending_chunked_abort.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
