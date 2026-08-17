import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EaglePPVerifyInputRaw,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPPEaglePrebuiltMerge(unittest.TestCase):
    @staticmethod
    def _draft_input(bonus_tokens):
        batch_size = bonus_tokens.shape[0]
        return EagleDraftInput(
            topk_p=torch.ones((batch_size, 1), dtype=torch.float32),
            topk_index=bonus_tokens.reshape(-1, 1),
            hidden_states=torch.zeros((batch_size, 4), dtype=torch.float32),
            bonus_tokens=bonus_tokens,
        )

    def test_new_pd_request_is_normalized_before_running_batch_merge(self):
        bonus_tokens = torch.tensor([101, 202], dtype=torch.int64)
        draft_input = self._draft_input(bonus_tokens)
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
            pp_size=2,
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
        draft_input = self._draft_input(bonus_tokens)
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

    def test_dummy_tree_uses_bonus_tokens_as_roots(self):
        raw = EaglePPVerifyInputRaw.build_dummy_from_bonus_tokens(
            torch.tensor([7, 9], dtype=torch.int64), num_draft=4
        )

        self.assertEqual(raw.draft_tokens, [[7, 7, 7, 7], [9, 9, 9, 9]])
        self.assertEqual(raw.parent_list, [[-1, 0, 1], [-1, 0, 1]])
        self.assertEqual(raw.top_scores_index, [[0, 1, 2], [0, 1, 2]])
        self.assertEqual(raw.accept_lens, [1, 1])
        self.assertIsNone(raw.accept_index)

    def test_worker_fallback_normalizes_direct_pd_handoff(self):
        bonus_tokens = torch.tensor([401, 402], dtype=torch.int64)
        batch = SimpleNamespace(
            spec_info=self._draft_input(bonus_tokens),
            input_ids=None,
        )
        worker = SimpleNamespace(speculative_num_draft_tokens=4)

        EAGLEWorkerV2._normalize_pp_verify_input_from_pd(worker, batch)

        self.assertTrue(torch.equal(batch.input_ids, bonus_tokens))
        self.assertIsInstance(batch.spec_info, EaglePPVerifyInputRaw)
        self.assertEqual(
            batch.spec_info.draft_tokens,
            [[401, 401, 401, 401], [402, 402, 402, 402]],
        )

    def test_pp_non_last_idle_does_not_require_draft_worker(self):
        worker = SimpleNamespace(
            _draft_worker=None,
            topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            device="cpu",
        )

        verify_input = EAGLEWorkerV2._build_idle_verify_input(worker, SimpleNamespace())

        self.assertTrue(verify_input.is_verify_input())
        self.assertEqual(verify_input.draft_token_num, 4)

    def test_pp_raw_rebuild_uses_current_verify_mask_contract(self):
        raw = EaglePPVerifyInputRaw(
            draft_tokens=[[10, 11, 12, 13], [20, 21, 22, 23]],
            bonus_tokens=[10, 20],
            top_scores_index=[[0, 1, 2], [0, 1, 2]],
            parent_list=[[-1, 0, 1], [-1, 0, 1]],
            accept_lens=[2, 3],
        )
        mask_buffer = torch.empty(32, dtype=torch.bool)
        verify_mask = SimpleNamespace(
            mode=TreeMaskMode.QLEN_ONLY,
            is_read=False,
            buffer=mask_buffer,
            fits=lambda bs: bs <= 8,
        )
        backend = SimpleNamespace(verify_mask=verify_mask, max_context_len=4096)
        worker = SimpleNamespace(
            topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(attn_backend=backend)
            ),
        )
        batch = SimpleNamespace(
            spec_info=raw,
            seq_lens=torch.tensor([10, 12], dtype=torch.int64),
            seq_lens_cpu=None,
            seq_lens_sum=None,
            input_ids=None,
        )
        arranged = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23])
        kernel_result = (
            mask_buffer,
            torch.tensor([2]),
            torch.tensor([3]),
            torch.tensor([4]),
            torch.tensor([5]),
            arranged,
        )

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.build_tree_kernel_efficient",
            return_value=kernel_result,
        ) as build_tree:
            verify = EAGLEWorkerV2._build_verify_input_from_pp_raw(worker, batch)

        self.assertEqual(build_tree.call_args.args[5], 0)
        self.assertIs(build_tree.call_args.args[10], mask_buffer)
        self.assertEqual(build_tree.call_args.args[9], TreeMaskMode.QLEN_ONLY)
        self.assertFalse(build_tree.call_args.kwargs["fill_prefix_mask"])
        self.assertEqual(
            build_tree.call_args.args[3].tolist(), [[11, 12, 13], [21, 22, 23]]
        )
        self.assertIs(batch.input_ids, arranged)
        self.assertEqual(verify.draft_token_num, 4)

    @patch(
        "sglang.srt.speculative.eagle_worker_v2.get_plan_stream",
        return_value=(object(), nullcontext()),
    )
    @patch("sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker")
    def test_pp_non_last_uses_target_war_runner(
        self, draft_worker_cls, _get_plan_stream
    ):
        server_args = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_algorithm="EAGLE",
            speculative_adaptive=False,
            speculative_adaptive_config=None,
            pp_size=2,
            device="cpu",
            page_size=1,
            override=lambda *_args, **_kwargs: None,
        )
        target = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=False),
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(context_len=4096),
                attn_backend=object(),
            ),
        )

        worker = EAGLEWorkerV2(
            server_args,
            gpu_id=0,
            ps=object(),
            nccl_port=1234,
            target_worker=target,
        )

        draft_worker_cls.assert_not_called()
        self.assertIsNone(worker.draft_worker)
        self.assertIs(worker.war_fastpath_runner, target.model_runner)


if __name__ == "__main__":
    unittest.main()
