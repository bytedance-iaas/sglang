import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation.decode import DecodePreallocQueue

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


class TestDecodePreallocQueue(CustomTestCase):
    def _new_queue(self) -> DecodePreallocQueue:
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.kv_manager = MagicMock()
        queue._ensure_last_attempt_time = {}
        queue._ensure_retry_count = {}
        queue._ensure_retry_interval = 0.0
        queue._max_ensure_retries = 3
        return queue

    def test_ensure_prefill_info_runtime_error_aborts_requests(self):
        queue = self._new_queue()
        queue.kv_manager.try_ensure_parallel_info.side_effect = RuntimeError(
            "PD disaggregation does not support decode-only DCP"
        )
        decode_req_0 = SimpleNamespace(kv_receiver=MagicMock())
        decode_req_1 = SimpleNamespace(kv_receiver=MagicMock())

        with self.assertLogs(decode_mod.logger, level="ERROR") as captured_logs:
            ready, remaining = queue._ensure_prefill_info(
                {"127.0.0.1:1234": [decode_req_0, decode_req_1]}
            )

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])
        decode_req_0.kv_receiver.abort.assert_called_once_with()
        decode_req_1.kv_receiver.abort.assert_called_once_with()
        self.assertEqual(queue._ensure_retry_count, {})
        self.assertEqual(queue._ensure_last_attempt_time, {})
        self.assertIn(
            "Failed to resolve prefill parallel info from 127.0.0.1:1234",
            "\n".join(captured_logs.output),
        )


if __name__ == "__main__":
    unittest.main()
