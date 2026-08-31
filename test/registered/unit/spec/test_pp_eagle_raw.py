import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.layers.attention.verify_mask import VerifyMask
from sglang.srt.speculative.eagle_info import EaglePPVerifyInputRaw
from sglang.srt.speculative.eagle_utils import TreeMaskMode
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2


class TestEaglePPVerifyInputRaw(unittest.TestCase):
    @staticmethod
    def _raw():
        return EaglePPVerifyInputRaw(
            draft_tokens=[[10, 11, 12, 13], [20, 21, 22, 23]],
            bonus_tokens=[10, 20],
            top_scores_index=[[0, 1, 2], [0, 1, 2]],
            parent_list=[[-1, 0, 1], [-1, 0, 1]],
            accept_lens=[2, 3],
            accept_index=[[0, 1], [0, 1, 2]],
        )

    def test_tensor_dict_round_trip_preserves_fields(self):
        raw = self._raw()

        restored = EaglePPVerifyInputRaw.from_pp_outputs(raw.to_tensor_dict())

        self.assertEqual(restored.draft_tokens, raw.draft_tokens)
        self.assertEqual(restored.bonus_tokens, raw.bonus_tokens)
        self.assertEqual(restored.top_scores_index, raw.top_scores_index)
        self.assertEqual(restored.parent_list, raw.parent_list)
        self.assertEqual(restored.accept_lens, raw.accept_lens)
        self.assertEqual(restored.accept_index, raw.accept_index)

    def test_dummy_tree_uses_bonus_tokens_as_roots(self):
        raw = EaglePPVerifyInputRaw.build_dummy_from_bonus_tokens(
            torch.tensor([7, 9]), num_draft=4
        )

        self.assertEqual(raw.draft_tokens, [[7, 7, 7, 7], [9, 9, 9, 9]])
        self.assertEqual(raw.parent_list, [[-1, 0, 1], [-1, 0, 1]])
        self.assertEqual(raw.top_scores_index, [[0, 1, 2], [0, 1, 2]])
        self.assertEqual(raw.accept_lens, [1, 1])
        self.assertIsNone(raw.accept_index)

    def test_filter_and_merge_keep_row_alignment(self):
        raw = self._raw()
        raw.filter_batch(torch.tensor([1]), new_indices_cpu=[1])
        raw.merge_batch(
            EaglePPVerifyInputRaw.build_dummy_from_bonus_tokens(
                torch.tensor([30]), num_draft=4
            )
        )

        self.assertEqual(raw.bonus_tokens, [20, 30])
        self.assertEqual(raw.draft_tokens[0], [20, 21, 22, 23])
        self.assertEqual(raw.draft_tokens[1], [30, 30, 30, 30])
        self.assertEqual(raw.accept_lens, [3, 1])
        self.assertIsNone(raw.accept_index)

    def test_filter_rejects_missing_required_field(self):
        raw = self._raw()
        raw.parent_list = None

        with self.assertRaisesRegex(RuntimeError, "required field was None"):
            raw.filter_batch(torch.tensor([0]))


class TestEaglePPVerifyRebuild(unittest.TestCase):
    @staticmethod
    def _worker(verify_mask=None):
        return SimpleNamespace(
            topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    attn_backend=SimpleNamespace(
                        verify_mask=verify_mask,
                        max_context_len=4096,
                    )
                )
            ),
        )

    @staticmethod
    def _batch(raw):
        return SimpleNamespace(
            spec_info=raw,
            seq_lens=torch.tensor([10, 12], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([10, 12], dtype=torch.int64),
            seq_lens_sum=22,
            input_ids=None,
        )

    def test_rebuild_strips_bonus_column_before_tree_kernel(self):
        raw = TestEaglePPVerifyInputRaw._raw()
        batch = self._batch(raw)
        arranged = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23])
        kernel_result = (
            torch.tensor([1]),
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
            verify = EAGLEWorkerV2._build_verify_input_from_pp_raw(
                self._worker(), batch
            )

        draft_without_bonus = build_tree.call_args.args[3]
        self.assertEqual(tuple(draft_without_bonus.shape), (2, 3))
        self.assertEqual(draft_without_bonus.tolist(), [[11, 12, 13], [21, 22, 23]])
        self.assertIs(batch.input_ids, arranged)
        self.assertEqual(verify.topk, 1)
        self.assertEqual(verify.spec_steps, 3)
        self.assertEqual(verify.draft_token_num, 4)
        self.assertEqual(build_tree.call_args.args[9], TreeMaskMode.FULL_MASK)
        self.assertIsNone(build_tree.call_args.args[10])
        self.assertTrue(build_tree.call_args.kwargs["fill_prefix_mask"])

    def test_rebuild_uses_current_verify_mask_contract(self):
        raw = TestEaglePPVerifyInputRaw._raw()
        batch = self._batch(raw)
        mask_buffer = torch.empty(128, dtype=torch.bool)
        verify_mask = VerifyMask(
            buffer=mask_buffer,
            mode=TreeMaskMode.QLEN_ONLY,
            max_bs=2,
            is_read=False,
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
            EAGLEWorkerV2._build_verify_input_from_pp_raw(
                self._worker(verify_mask), batch
            )

        self.assertEqual(build_tree.call_args.args[5], 22)
        self.assertEqual(build_tree.call_args.args[9], TreeMaskMode.QLEN_ONLY)
        self.assertIs(build_tree.call_args.args[10], mask_buffer)
        self.assertFalse(build_tree.call_args.kwargs["fill_prefix_mask"])

    def test_rebuild_rejects_parent_shape_mismatch(self):
        raw = TestEaglePPVerifyInputRaw._raw()
        raw.parent_list = [[-1, 0], [-1, 0]]

        with self.assertRaisesRegex(AssertionError, "topology shape mismatch"):
            EAGLEWorkerV2._build_verify_input_from_pp_raw(
                self._worker(), self._batch(raw)
            )


class TestEaglePPLastRankDraftOwnership(unittest.TestCase):
    @staticmethod
    def _server_args():
        return SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_algorithm="EAGLE",
            speculative_adaptive=False,
            speculative_adaptive_config=None,
            pp_size=2,
            device="cpu",
            page_size=1,
            override=lambda *args, **kwargs: None,
        )

    @staticmethod
    def _target(is_last_rank):
        return SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=is_last_rank),
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(context_len=4096),
                attn_backend=object(),
            ),
        )

    @patch(
        "sglang.srt.speculative.eagle_worker_v2.get_plan_stream",
        return_value=(object(), nullcontext()),
    )
    @patch("sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker")
    def test_non_last_rank_does_not_construct_draft_worker(
        self, draft_worker_cls, _get_plan_stream
    ):
        target = self._target(is_last_rank=False)

        worker = EAGLEWorkerV2(
            self._server_args(),
            gpu_id=0,
            ps=object(),
            nccl_port=1234,
            target_worker=target,
        )

        draft_worker_cls.assert_not_called()
        self.assertIsNone(worker.draft_worker)
        self.assertEqual(
            worker.spec_v2_attn_backends,
            (target.model_runner.attn_backend,),
        )
        self.assertIs(worker.war_fastpath_runner, target.model_runner)
        worker.init_cuda_graphs()

    @patch(
        "sglang.srt.speculative.eagle_worker_v2.get_plan_stream",
        return_value=(object(), nullcontext()),
    )
    @patch("sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker")
    def test_last_rank_constructs_draft_worker(
        self, draft_worker_cls, _get_plan_stream
    ):
        sentinel = object()
        draft_worker_cls.return_value = sentinel

        worker = EAGLEWorkerV2(
            self._server_args(),
            gpu_id=0,
            ps=object(),
            nccl_port=1234,
            target_worker=self._target(is_last_rank=True),
        )

        self.assertIs(worker.draft_worker, sentinel)
        draft_worker_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
