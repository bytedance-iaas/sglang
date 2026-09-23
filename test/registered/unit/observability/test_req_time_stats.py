"""Unit tests for ReqTimeStats IPC serialization.

ReqTimeStatsBase.__setstate__ rebases perf_counter fields onto the receiving
process's clock anchor. Rebasing a field that was never stamped (0.0) turns
the sentinel into a tiny epsilon (sender_diff - receiver_diff), which defeats
== 0.0 / > 0.0 "was this stamped?" checks downstream. Concretely, a PD decode
server never stamps prefill_finished_time locally; if the sentinel arrives at
the tokenizer as an epsilon, first-token bookkeeping mistakes it for a real
stamp and the TTFT / inter-token-latency histograms record ~node-uptime-sized
garbage samples.
"""

import pickle
import unittest
from unittest import mock

import sglang.srt.observability.req_time_stats as rts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSetstatePreservesUnsetTimeSentinels(CustomTestCase):
    def test_two_hop_round_trip(self):
        src = rts.SchedulerReqTimeStats()
        src.enable_metrics = True
        src.wait_queue_entry_time = 123.456
        src.prefill_finished_time = 0.0

        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            blob = pickle.dumps(src)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_005.0):
            hop1 = pickle.loads(blob)
            blob2 = pickle.dumps(hop1)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_009.0):
            hop2 = pickle.loads(blob2)

        self.assertEqual(hop2.prefill_finished_time, 0.0)
        self.assertAlmostEqual(hop2.wait_queue_entry_time, 123.456 - 9.0)

    def test_diagnostic_timestamps_round_trip_only_when_enabled(self):
        src = rts.SchedulerReqTimeStats()
        src.scheduler_recv_time = 101.0
        src.decode_prealloc_queue_entry_time = 102.0
        src.bootstrap_done_time = 103.0
        src.decode_transfer_queue_entry_time = 104.0
        src.wait_queue_entry_time = 105.0
        src.forward_entry_time = 106.0
        src.decode_prebuilt_finish_time = 107.0
        src.last_decode_finish_time = 108.0

        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000.0):
            disabled_state = src.__getstate__()
            self.assertEqual(disabled_state, {})

            src.has_timing_data = True
            blob = pickle.dumps(src)

        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_005.0):
            restored = pickle.loads(blob)

        self.assertAlmostEqual(restored.scheduler_recv_time, 96.0)
        self.assertAlmostEqual(restored.decode_prebuilt_finish_time, 102.0)
        self.assertAlmostEqual(restored.last_decode_finish_time, 103.0)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_005.0):
            timestamps = restored.convert_to_diagnostic_timestamps()
        self.assertEqual(timestamps["scheduler_recv_ts"], 1101.0)
        self.assertEqual(timestamps["last_decode_finish_ts"], 1108.0)


if __name__ == "__main__":
    unittest.main()
