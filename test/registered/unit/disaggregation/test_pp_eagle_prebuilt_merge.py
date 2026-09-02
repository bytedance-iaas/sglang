import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EaglePPVerifyInputRaw,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPPEaglePrebuiltMerge(unittest.TestCase):
    def test_seedless_draft_vote_is_set_before_dp_metadata_sync(self):
        batch = SimpleNamespace(
            is_empty=lambda: False, force_disable_draft_cuda_graph=False
        )
        draft_worker = SimpleNamespace(
            requires_dp_attention_eager_forward=lambda candidate: candidate is batch
        )

        def prepare(candidate):
            self.assertTrue(candidate.force_disable_draft_cuda_graph)
            return candidate

        scheduler = SimpleNamespace(
            get_new_prebuilt_batch=lambda _running: None,
            update_running_batch=lambda running: running,
            draft_worker=draft_worker,
            dp_attn_adapter=SimpleNamespace(maybe_prepare_mlp_sync_batch=prepare),
        )
        with patch(
            "sglang.srt.disaggregation.decode.set_schedule_time_batch"
        ) as set_schedule_time:
            plan = (
                SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run(
                    scheduler, batch
                )
            )

        self.assertIs(plan.batch_to_run, batch)
        set_schedule_time.assert_called_once_with(batch)

    @staticmethod
    def _batch(draft_input):
        return SimpleNamespace(
            spec_algorithm=SimpleNamespace(
                build_disagg_draft_input=lambda *_args: draft_input
            ),
            reqs=[],
            device=torch.device("cpu"),
            input_ids=None,
            spec_info=None,
        )

    @staticmethod
    def _draft_input(tokens):
        bonus_tokens = torch.tensor(tokens, dtype=torch.int64)
        return EagleDraftInput(
            topk_p=torch.ones((len(tokens), 1), dtype=torch.float32),
            topk_index=bonus_tokens[:, None],
            hidden_states=torch.zeros((len(tokens), 4), dtype=torch.float32),
            bonus_tokens=bonus_tokens,
        )

    def test_new_pd_request_is_normalized_before_running_batch_merge(self):
        draft_input = self._draft_input([101, 202])
        batch = self._batch(draft_input)
        with (
            patch(
                "sglang.srt.disaggregation.decode_schedule_batch_mixin.get_parallel",
                return_value=SimpleNamespace(pp_size=2),
            ),
            patch(
                "sglang.srt.disaggregation.decode_schedule_batch_mixin.get_spec",
                return_value=SimpleNamespace(speculative_num_draft_tokens=4),
            ),
        ):
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(batch, None)

        self.assertTrue(torch.equal(batch.input_ids, draft_input.bonus_tokens))
        self.assertIsInstance(batch.spec_info, EaglePPVerifyInputRaw)
        self.assertEqual(
            batch.spec_info.draft_tokens,
            [[101, 101, 101, 101], [202, 202, 202, 202]],
        )
        running_raw = EaglePPVerifyInputRaw(
            draft_tokens=[[11, 12, 13, 14]],
            bonus_tokens=[11],
            top_scores_index=[[0, 1, 2]],
            parent_list=[[-1, 0, 1]],
            accept_lens=[2],
        )
        running_raw.merge_batch(batch.spec_info)
        self.assertEqual(running_raw.bonus_tokens, [11, 101, 202])

    def test_non_pp_keeps_eagle_draft_input(self):
        draft_input = self._draft_input([303])
        batch = self._batch(draft_input)
        with patch(
            "sglang.srt.disaggregation.decode_schedule_batch_mixin.get_parallel",
            return_value=SimpleNamespace(pp_size=1),
        ):
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(batch, None)
        self.assertIs(batch.spec_info, draft_input)


if __name__ == "__main__":
    unittest.main()
