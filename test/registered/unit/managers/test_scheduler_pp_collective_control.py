import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def _scheduler(*, is_last_rank=False):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.pp_group = SimpleNamespace(is_last_rank=is_last_rank)
    scheduler.world_group = SimpleNamespace(cpu_group=object())
    scheduler.send_req_work = [MagicMock()]
    scheduler._pp_commit_comm_work = MagicMock(side_effect=lambda work: work.clear())
    scheduler._pp_send_pyobj_to_next_stage = MagicMock(return_value=[])
    return scheduler


class TestSchedulerPPCollectiveControl(CustomTestCase):
    def test_ingest_defers_collective_control_without_reordering(self):
        scheduler = _scheduler()
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
            recv_reqs = scheduler.ingest_requests(defer_pp_collective_control=True)

        scheduler.process_input_requests.assert_not_called()
        self.assertEqual(recv_reqs, [ordinary, dump, start])

    def test_ingest_dispatches_non_collective_controls_normally(self):
        scheduler = _scheduler()
        start = _req(ExpertDistributionReqType.START_RECORD)
        stop = _req(ExpertDistributionReqType.STOP_RECORD)
        scheduler._deferred_input_requests = []
        scheduler._poll_timeout_aborts = MagicMock(return_value=[])
        scheduler.request_receiver = SimpleNamespace(
            recv_requests=MagicMock(return_value=[start, stop])
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
            recv_reqs = scheduler.ingest_requests(defer_pp_collective_control=True)

        scheduler.process_input_requests.assert_called_once_with([start, stop])
        self.assertEqual(recv_reqs, [start, stop])

    def test_default_ingest_dispatches_dump_for_non_pp_callers(self):
        scheduler = _scheduler()
        dump = _req(ExpertDistributionReqType.DUMP_RECORD)
        scheduler._deferred_input_requests = []
        scheduler._poll_timeout_aborts = MagicMock(return_value=[])
        scheduler.request_receiver = SimpleNamespace(
            recv_requests=MagicMock(return_value=[dump])
        )
        scheduler.metrics_reporter = SimpleNamespace(
            record_scheduler_active=MagicMock()
        )
        scheduler.process_input_requests = MagicMock()

        with get_parallel().override(
            pp_size=1,
            pp_rank=0,
            attn_tp_rank=0,
            attn_cp_rank=0,
        ):
            recv_reqs = scheduler.ingest_requests()

        scheduler.process_input_requests.assert_called_once_with([dump])
        self.assertEqual(recv_reqs, [dump])

    def test_dump_barriers_across_pp_before_recorder_collective(self):
        scheduler = _scheduler()
        dump = _req(ExpertDistributionReqType.DUMP_RECORD)
        recorder = MagicMock()

        with (
            get_parallel().override(pp_size=2, pp_rank=0),
            patch("torch.distributed.barrier") as barrier,
            patch(
                "sglang.srt.managers.scheduler.get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            scheduler.expert_distribution_handle(dump)

        barrier.assert_called_once_with(group=scheduler.world_group.cpu_group)
        recorder.dump_record.assert_called_once_with()

    def test_dump_skips_barrier_without_pp(self):
        scheduler = _scheduler()
        dump = _req(ExpertDistributionReqType.DUMP_RECORD)
        recorder = MagicMock()

        with (
            get_parallel().override(pp_size=1, pp_rank=0),
            patch("torch.distributed.barrier") as barrier,
            patch(
                "sglang.srt.managers.scheduler.get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            scheduler.expert_distribution_handle(dump)

        barrier.assert_not_called()
        recorder.dump_record.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
