"""Runtime prefill chunk updates preserve allocation limits and request progress."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GetInternalStateReq, SetInternalStateReq
from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
)
from sglang.srt.runtime_context import (
    get_context,
    get_schedule,
    get_server_args,
    publish,
    restore_context,
    snapshot_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestChunkedPrefillUpdate(unittest.TestCase):
    def setUp(self):
        self.addCleanup(restore_context, snapshot_context())
        self.enterContext(envs.SGLANG_DISAGG_STAGING_BUFFER.override(False))
        publish(
            ServerArgs(model_path="dummy", chunked_prefill_size=8192, page_size=64),
            role="scheduler",
        )
        self.scheduler = SimpleNamespace(
            chunked_prefill_size=8192,
            page_size=64,
            truncation_align_size=None,
            chunked_req=object(),
            is_mixed_chunk=True,
            max_running_requests=8,
            dllm_config=None,
        )
        self.controller = SchedulerInternalStateController(self.scheduler, False)

    def _update(self, value, **other):
        return self.controller.set_internal_state(
            SetInternalStateReq(server_args={"chunked_prefill_size": value, **other})
        ).updated

    def _assert_unchanged(self):
        self.assertEqual(self.scheduler.chunked_prefill_size, 8192)
        self.assertEqual(get_schedule().chunked_prefill_size, 8192)

    def test_update_reaches_scheduler_and_resolved_config(self):
        self.assertTrue(self._update(2048))
        self.assertEqual(self.scheduler.chunked_prefill_size, 2048)
        self.assertEqual(get_schedule().chunked_prefill_size, 2048)
        # The startup allocation remains diagnostic input, not the live budget.
        self.assertEqual(get_server_args().chunked_prefill_size, 8192)

    def test_can_restore_startup_budget_after_reducing_it(self):
        for value in (2048, 4096, 64, 8192, 8192):
            with self.subTest(value=value):
                self.assertTrue(self._update(value))
                self.assertEqual(self.scheduler.chunked_prefill_size, value)

    def test_update_preserves_inflight_request_and_mixed_chunk_mode(self):
        req = self.scheduler.chunked_req
        self.assertTrue(self._update(2048))
        self.assertIs(self.scheduler.chunked_req, req)
        self.assertTrue(self.scheduler.is_mixed_chunk)

    def test_rejects_exceeding_startup_budget_after_an_update(self):
        self.assertTrue(self._update(2048))
        self.assertFalse(self._update(16384))
        self.assertEqual(self.scheduler.chunked_prefill_size, 2048)
        self.assertEqual(get_schedule().chunked_prefill_size, 2048)

    def test_readback_reports_live_budget(self):
        self.assertTrue(self._update(2048))
        contributors = {
            name: MagicMock()
            for name in vars(SchedulerInternalStateController)
            if name.startswith("_add_")
        }
        with patch.multiple(SchedulerInternalStateController, **contributors):
            state = self.controller.get_internal_state(GetInternalStateReq())
        self.assertEqual(state.internal_state["chunked_prefill_size"], 2048)

    def test_rejects_invalid_values_without_mutation(self):
        for value in (0, -1, -64, 8193, 16384, 65, 1.5, 2048.0, True, None, "2048"):
            with self.subTest(value=value):
                self.assertFalse(self._update(value))
                self._assert_unchanged()

    def test_rejects_enabling_chunked_prefill_at_runtime(self):
        for value in (None, -1, 0):
            with self.subTest(value=value):
                self.scheduler.chunked_prefill_size = value
                controller = SchedulerInternalStateController(self.scheduler, False)
                result = controller.set_internal_state(
                    SetInternalStateReq(server_args={"chunked_prefill_size": 2048})
                )
                self.assertFalse(result.updated)
                self.assertEqual(self.scheduler.chunked_prefill_size, value)
                self.assertEqual(get_schedule().chunked_prefill_size, 8192)

    def test_rejects_chunk_smaller_than_truncation_alignment(self):
        self.scheduler.truncation_align_size = 4096
        self.assertFalse(self._update(2048))
        self._assert_unchanged()
        self.assertTrue(self._update(4096))

    def test_rejects_dynamic_chunking(self):
        get_context().override("test", enable_dynamic_chunking=True)
        self.assertFalse(self._update(2048))
        self._assert_unchanged()

    def test_rejects_staging_grid_changes(self):
        with envs.SGLANG_DISAGG_STAGING_BUFFER.override(True):
            self.assertFalse(self._update(2048))
        self._assert_unchanged()

    def test_rejects_diffusion_llm_chunk_updates(self):
        self.scheduler.dllm_config = SimpleNamespace(block_size=32)
        self.assertFalse(self._update(2048))
        self._assert_unchanged()

    def test_unknown_key_rejects_whole_update(self):
        self.assertFalse(self._update(2048, unsupported_knob=1))
        self._assert_unchanged()

    def test_invalid_existing_key_rejects_whole_update(self):
        self.assertFalse(self._update(2048, pp_max_micro_batch_size=0))
        self._assert_unchanged()

    def test_invalid_chunk_does_not_apply_other_keys(self):
        before = get_context().resolved_server_args_dict()["pp_max_micro_batch_size"]
        result = self.controller.set_internal_state(
            SetInternalStateReq(
                server_args={"pp_max_micro_batch_size": 1, "chunked_prefill_size": 65}
            )
        )
        self.assertFalse(result.updated)
        self._assert_unchanged()
        self.assertEqual(
            get_context().resolved_server_args_dict()["pp_max_micro_batch_size"], before
        )

    def test_attention_dp_update_is_already_per_rank(self):
        get_context().override("test", attn_dp_size=2)
        self.assertTrue(self._update(2048))
        self.assertEqual(self.scheduler.chunked_prefill_size, 2048)
        self.assertEqual(get_schedule().chunked_prefill_size, 2048)


if __name__ == "__main__":
    unittest.main()
