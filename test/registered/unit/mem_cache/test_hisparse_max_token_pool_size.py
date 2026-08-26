"""Regression tests for HiSparse host-backed request capacity."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _model_runner(*, enable_hisparse, size_full=None, max_total=1024):
    runner = object.__new__(ModelRunner)
    runner.enable_hisparse = enable_hisparse
    runner.token_to_kv_pool_allocator = SimpleNamespace()
    if size_full is not None:
        runner.token_to_kv_pool_allocator.size_full = size_full
    runner.is_hybrid_swa = False
    runner.max_total_num_tokens = max_total
    runner.full_max_total_num_tokens = None
    runner.swa_max_total_num_tokens = None
    return runner


class TestHiSparseMaxTokenPoolSize(CustomTestCase):
    def test_hisparse_uses_host_backed_logical_capacity(self):
        runner = _model_runner(
            enable_hisparse=True, size_full=4096, max_total=1024
        )
        self.assertEqual(runner.max_token_pool_size, 4096)
        self.assertEqual(runner.effective_max_total_num_tokens, 4096)

    def test_non_hisparse_keeps_device_capacity(self):
        runner = _model_runner(
            enable_hisparse=False, size_full=4096, max_total=1024
        )
        self.assertEqual(runner.max_token_pool_size, 1024)

    def test_hisparse_falls_back_when_allocator_has_no_size_full(self):
        runner = _model_runner(enable_hisparse=True, max_total=2048)
        self.assertEqual(runner.max_token_pool_size, 2048)

    def test_decode_admits_between_device_and_host_capacity(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.max_total_num_tokens = 1024
        queue.token_to_kv_pool_allocator = SimpleNamespace(size_swa=10**9)
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(max_token_pool_size=4096)
            ),
            output_streamer=MagicMock(),
        )
        req = SimpleNamespace(
            rid="hisparse-long",
            origin_input_ids=[0] * 2048,
            output_ids=[],
            return_logprob=False,
            pd_rebootstrap_in_progress=False,
            finished_reason=None,
        )

        self.assertFalse(queue._check_if_req_exceed_kv_capacity(req))
        queue.scheduler.output_streamer.stream_output.assert_not_called()
