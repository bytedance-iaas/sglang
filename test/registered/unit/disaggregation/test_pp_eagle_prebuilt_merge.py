import unittest
from types import SimpleNamespace

import torch

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
    def test_new_pd_request_is_normalized_before_running_batch_merge(self):
        bonus_tokens = torch.tensor([101, 202], dtype=torch.int64)
        draft_input = EagleDraftInput(
            topk_p=torch.ones((2, 1), dtype=torch.float32),
            topk_index=torch.tensor([[101], [202]], dtype=torch.int64),
            hidden_states=torch.zeros((2, 4), dtype=torch.float32),
            bonus_tokens=bonus_tokens,
        )
        batch = SimpleNamespace(
            spec_algorithm=SimpleNamespace(
                build_disagg_draft_input=lambda *_args: draft_input
            ),
            reqs=[],
            device=torch.device("cpu"),
            enable_overlap=False,
            input_ids=torch.empty((0,), dtype=torch.int64),
            spec_info=None,
        )
        server_args = SimpleNamespace(
            pp_size=4,
            speculative_num_draft_tokens=4,
        )

        ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(
            batch,
            server_args,
            future_map=None,
        )

        self.assertTrue(torch.equal(batch.input_ids, bonus_tokens))
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
        self.assertEqual(len(running_raw.draft_tokens), 3)
        self.assertEqual(running_raw.bonus_tokens, [11, 101, 202])

    def test_non_pp_keeps_eagle_draft_input(self):
        bonus_tokens = torch.tensor([303], dtype=torch.int64)
        draft_input = EagleDraftInput(
            topk_p=torch.ones((1, 1), dtype=torch.float32),
            topk_index=torch.tensor([[303]], dtype=torch.int64),
            hidden_states=torch.zeros((1, 4), dtype=torch.float32),
            bonus_tokens=bonus_tokens,
        )
        batch = SimpleNamespace(
            spec_algorithm=SimpleNamespace(
                build_disagg_draft_input=lambda *_args: draft_input
            ),
            reqs=[],
            device=torch.device("cpu"),
            enable_overlap=False,
            input_ids=torch.empty((0,), dtype=torch.int64),
            spec_info=None,
        )
        server_args = SimpleNamespace(
            pp_size=1,
            speculative_num_draft_tokens=4,
        )

        ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(
            batch,
            server_args,
            future_map=None,
        )

        self.assertIs(batch.spec_info, draft_input)


if __name__ == "__main__":
    unittest.main()
