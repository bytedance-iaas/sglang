import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    ExpertDistributionReq,
    ExpertDistributionReqType,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(action):
    return ExpertDistributionReq(action=action)


def _scheduler(*, is_last_rank):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.pp_group = SimpleNamespace(is_last_rank=is_last_rank)
    scheduler.send_req_work = [MagicMock()]
    scheduler._pp_commit_comm_work = MagicMock(side_effect=lambda work: work.clear())
    scheduler._pp_send_pyobj_to_next_stage = MagicMock(return_value=[])
    return scheduler


class TestSchedulerPPCollectiveControl(CustomTestCase):
    def test_first_stage_preforwards_ordered_batch_once_before_local_dispatch(self):
        scheduler = _scheduler(is_last_rank=False)
        dump = _req(ExpertDistributionReqType.DUMP_RECORD)
        start = _req(ExpertDistributionReqType.START_RECORD)
        ordinary = object()
        scheduler._deferred_input_requests = []
        scheduler._poll_timeout_aborts = MagicMock(return_value=[])
        scheduler.request_receiver = SimpleNamespace(
            recv_requests=MagicMock(return_value=[ordinary, dump, start])
        )
        scheduler.metrics_reporter = SimpleNamespace(
            record_scheduler_active=MagicMock()
        )
        scheduler.process_input_requests = MagicMock()

        with get_parallel().override(
            pp_size=2,
            pp_rank=0,
            attn_tp_rank=0,
            attn_cp_rank=0,
        ):
            relay_reqs = scheduler.ingest_requests()

        scheduler._pp_commit_comm_work.assert_called_once()
        scheduler._pp_send_pyobj_to_next_stage.assert_called_once_with(
            [ordinary, dump, start]
        )
        scheduler.process_input_requests.assert_called_once_with(
            [ordinary, dump, start]
        )
        self.assertEqual(relay_reqs, [])

    def test_multiple_dumps_preforward_the_batch_once(self):
        scheduler = _scheduler(is_last_rank=False)
        first = _req(ExpertDistributionReqType.DUMP_RECORD)
        ordinary = object()
        second = _req(ExpertDistributionReqType.DUMP_RECORD)
        reqs = [first, ordinary, second]

        with get_parallel().override(pp_size=2, pp_rank=0):
            relay_reqs = scheduler._preforward_pp_collective_control_requests(reqs)

        self.assertEqual(relay_reqs, [])
        scheduler._pp_commit_comm_work.assert_called_once()
        scheduler._pp_send_pyobj_to_next_stage.assert_called_once_with(reqs)

    def test_last_stage_processes_dump_without_preforwarding(self):
        scheduler = _scheduler(is_last_rank=True)
        dump = _req(ExpertDistributionReqType.DUMP_RECORD)

        with get_parallel().override(pp_size=2, pp_rank=1):
            relay_reqs = scheduler._preforward_pp_collective_control_requests([dump])

        self.assertEqual(relay_reqs, [dump])
        scheduler._pp_commit_comm_work.assert_not_called()
        scheduler._pp_send_pyobj_to_next_stage.assert_not_called()

    def test_start_and_stop_keep_regular_post_dispatch_relay(self):
        scheduler = _scheduler(is_last_rank=False)
        reqs = [
            _req(ExpertDistributionReqType.START_RECORD),
            _req(ExpertDistributionReqType.STOP_RECORD),
        ]

        with get_parallel().override(pp_size=2, pp_rank=0):
            relay_reqs = scheduler._preforward_pp_collective_control_requests(reqs)

        self.assertEqual(relay_reqs, reqs)
        scheduler._pp_commit_comm_work.assert_not_called()
        scheduler._pp_send_pyobj_to_next_stage.assert_not_called()

    def test_pp_size_one_is_unchanged(self):
        scheduler = _scheduler(is_last_rank=True)
        dump = _req(ExpertDistributionReqType.DUMP_RECORD)

        with get_parallel().override(pp_size=1, pp_rank=0):
            relay_reqs = scheduler._preforward_pp_collective_control_requests([dump])

        self.assertEqual(relay_reqs, [dump])
        scheduler._pp_commit_comm_work.assert_not_called()
        scheduler._pp_send_pyobj_to_next_stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
