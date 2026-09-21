import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.prefill import (  # noqa: E402
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (  # noqa: E402
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPPDisaggChunkResult(unittest.TestCase):
    def make_case(self, spec_info, middle_chunks=1):
        req = SimpleNamespace(
            rid="chunked-request",
            inflight_middle_chunks=middle_chunks,
            output_ids=[],
            time_stats=Mock(),
            return_logprob=False,
            return_sampling_mask=False,
            pending_bootstrap=False,
            is_retracted=False,
            finished=lambda: False,
            to_finish=None,
            finished_reason=None,
            extend_range=None,
            grammar=None,
        )
        batch = SimpleNamespace(
            reqs=[req],
            spec_info=spec_info,
            forward_mode=ForwardMode.EXTEND,
            contains_last_prefill_chunk=middle_chunks == 0,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        processor = Mock()
        processor._validate_pp_skip_output_comm = (
            SchedulerBatchResultProcessor._validate_pp_skip_output_comm
        )
        processor.snapshot_auxiliary_output_starts.return_value = []
        scheduler = SimpleNamespace(
            device="cpu",
            device_module=SimpleNamespace(Event=Mock, current_stream=lambda: None),
            batch_result_processor=processor,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
            chunked_req=req,
            enable_overlap=False,
            disagg_prefill_inflight_queue=[],
            tree_cache=Mock(),
            metrics_reporter=Mock(),
            send_kv_chunk=Mock(),
        )
        _, result, _ = SchedulerPPMixin._pp_make_skip_output_result(
            scheduler, batch, None
        )
        return scheduler, batch, result, req

    def process(self, scheduler, batch, result):
        with patch(
            "sglang.srt.managers.scheduler_components.batch_result_processor."
            "envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get",
            return_value=True,
        ):
            SchedulerDisaggregationPrefillMixin.process_batch_result_disagg_prefill(
                scheduler, batch, result
            )

    def test_middle_chunk_preserves_draft_without_consuming_placeholder(self):
        for spec_info in (None, Mock()):
            with self.subTest(has_local_draft=spec_info is not None):
                scheduler, batch, result, req = self.make_case(spec_info)
                self.process(scheduler, batch, result)
                self.assertEqual(req.inflight_middle_chunks, 0)
                self.assertEqual(req.output_ids, [])
                self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
                self.assertIs(batch.spec_info, spec_info)
                scheduler.send_kv_chunk.assert_not_called()
                if spec_info is not None:
                    spec_info.hidden_states.to.assert_not_called()
                    spec_info.dsa_topk_indices.to.assert_not_called()

    def test_skipped_output_cannot_complete_a_request(self):
        scheduler, batch, result, req = self.make_case(None, middle_chunks=0)
        with self.assertRaisesRegex(AssertionError, "PP skip output comm"):
            self.process(scheduler, batch, result)
        self.assertEqual(req.output_ids, [])
        scheduler.send_kv_chunk.assert_not_called()

    def test_real_output_keeps_draft_identity_check(self):
        scheduler, batch, result, req = self.make_case(Mock(), middle_chunks=0)
        result.skipped_output_comm = False
        with self.assertRaises(AssertionError):
            self.process(scheduler, batch, result)
        self.assertEqual(req.output_ids, [])

    def test_final_chunk_transfers_real_token_and_draft(self):
        draft = SimpleNamespace(
            topk_p=torch.tensor([[0.75]]),
            topk_index=torch.tensor([[9]]),
            hidden_states=torch.tensor([[1.0, 2.0]]),
            dsa_topk_indices=torch.tensor([[3, 4]]),
        )
        scheduler, batch, result, req = self.make_case(draft, middle_chunks=0)
        result.skipped_output_comm = False
        result.next_token_ids = torch.tensor([42])
        result.next_draft_input = draft
        with patch("sglang.srt.disaggregation.prefill.maybe_cache_unfinished_req"):
            self.process(scheduler, batch, result)
        self.assertEqual(req.output_ids, [42])
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])
        self.assertTrue(torch.equal(req.output_topk_p, draft.topk_p[0]))
        self.assertTrue(torch.equal(req.output_topk_index, draft.topk_index[0]))
        self.assertTrue(torch.equal(req.hidden_states_tensor, draft.hidden_states[0]))
        self.assertTrue(
            torch.equal(req.output_dsa_topk_indices, draft.dsa_topk_indices[0])
        )
        scheduler.send_kv_chunk.assert_called_once_with(req, last_chunk=True)


if __name__ == "__main__":
    unittest.main()
