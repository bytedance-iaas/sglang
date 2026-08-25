"""fix(a) coverage: decode-side retraction routing when KV offload is disabled.

Root cause of the HiSparse "bs16" ceiling: on a KV-full decode retract,
Scheduler.update_running_batch -> retract_decode -> release_req unconditionally
offloaded the retracted KV via allocator.get_cpu_copy, which the HiSparse device
pools raise NotImplementedError for, crashing the scheduler once the decode batch
grew past the retract line.

fix(a) gates that device->host copy in release_req on
--disaggregation-decode-enable-offload-kvcache and, when the flag is off, routes
the retracted request through the existing PD true-retraction rebootstrap path
(recompute the prefix on the prefill worker) instead of the
retracted_queue/load_kv_cache resume path -- there is no CPU copy to resume from.

These tests pin that routing in Scheduler._add_request_to_queue:
- offload OFF + retracted  -> rebootstrap enqueue (add(is_rebootstrap=True)),
  boundary output id popped into pd_rebootstrap_forced_output_id, flags set.
- offload ON  + retracted  -> legacy retracted_queue enqueue
  (add(is_retracted=True)), unchanged.
- non-retracted            -> normal prealloc enqueue, unchanged.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestHiSparseRetractRebootstrapRouting(CustomTestCase):
    def _new_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.enable_priority_scheduling = False
        scheduler.abort_on_priority_when_disabled = False
        scheduler.waiting_queue = []
        scheduler.disagg_decode_prealloc_queue = MagicMock()
        return scheduler

    def _new_req(self, output_ids):
        req = MagicMock()
        req.priority = None
        req.rid = "req"
        req.output_ids = list(output_ids)
        req.pd_rebootstrap_in_progress = False
        req.pd_rebootstrap_forced_output_id = None
        req.time_stats = MagicMock()
        return req

    def _patch_offload(self, enabled: bool):
        disagg = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=enabled,
        )
        return patch(
            "sglang.srt.managers.scheduler.get_disagg",
            return_value=disagg,
        )

    def test_offload_disabled_retract_routes_to_rebootstrap(self):
        scheduler = self._new_scheduler()
        req = self._new_req(output_ids=[11, 22, 33])

        with self._patch_offload(enabled=False):
            scheduler._add_request_to_queue(req, is_retracted=True)

        # Rebootstrap enqueue, not the retracted_queue resume path.
        scheduler.disagg_decode_prealloc_queue.add.assert_called_once_with(
            req, is_rebootstrap=True
        )
        # Boundary token popped so the prefill worker re-emits it deterministically.
        self.assertEqual(req.pd_rebootstrap_forced_output_id, 33)
        self.assertEqual(req.output_ids, [11, 22])
        self.assertTrue(req.pd_rebootstrap_in_progress)
        req.time_stats.set_retract_time.assert_called_once()
        # Must not also fall through to the legacy is_retracted enqueue.
        self.assertEqual(scheduler.disagg_decode_prealloc_queue.add.call_count, 1)

    def test_offload_disabled_retract_with_empty_output_ids(self):
        scheduler = self._new_scheduler()
        req = self._new_req(output_ids=[])

        with self._patch_offload(enabled=False):
            scheduler._add_request_to_queue(req, is_retracted=True)

        scheduler.disagg_decode_prealloc_queue.add.assert_called_once_with(
            req, is_rebootstrap=True
        )
        # No boundary token to pop; forced id stays unset.
        self.assertIsNone(req.pd_rebootstrap_forced_output_id)
        self.assertEqual(req.output_ids, [])
        self.assertTrue(req.pd_rebootstrap_in_progress)

    def test_offload_enabled_retract_keeps_legacy_resume_path(self):
        scheduler = self._new_scheduler()
        req = self._new_req(output_ids=[11, 22, 33])

        with self._patch_offload(enabled=True):
            scheduler._add_request_to_queue(req, is_retracted=True)

        # Unchanged behaviour: enqueue for load_kv_cache resume.
        scheduler.disagg_decode_prealloc_queue.add.assert_called_once_with(
            req, is_retracted=True
        )
        # No rebootstrap mutation.
        self.assertIsNone(req.pd_rebootstrap_forced_output_id)
        self.assertEqual(req.output_ids, [11, 22, 33])
        self.assertFalse(req.pd_rebootstrap_in_progress)
        req.time_stats.set_retract_time.assert_called_once()

    def test_non_retracted_decode_request_unaffected(self):
        scheduler = self._new_scheduler()
        req = self._new_req(output_ids=[11, 22, 33])

        with self._patch_offload(enabled=False):
            scheduler._add_request_to_queue(req, is_retracted=False)

        scheduler.disagg_decode_prealloc_queue.add.assert_called_once_with(
            req, is_retracted=False
        )
        self.assertIsNone(req.pd_rebootstrap_forced_output_id)
        self.assertFalse(req.pd_rebootstrap_in_progress)
        req.time_stats.set_decode_prealloc_queue_entry_time.assert_called_once()


if __name__ == "__main__":
    unittest.main()
