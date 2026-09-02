import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.verify_mask import VerifyMask
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    _assert_pp_mtp_compat,
)
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.speculative.eagle_info import EaglePPVerifyInputRaw
from sglang.srt.speculative.eagle_utils import TreeMaskMode
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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

    def test_dummy_filter_and_merge_keep_rows_aligned(self):
        raw = self._raw()
        raw.filter_batch(torch.tensor([1]), new_indices_cpu=[1])
        raw.merge_batch(
            EaglePPVerifyInputRaw.build_dummy_from_bonus_tokens(
                torch.tensor([30]), num_draft=4
            )
        )
        self.assertEqual(raw.bonus_tokens, [20, 30])
        self.assertEqual(raw.draft_tokens[1], [30, 30, 30, 30])
        self.assertEqual(raw.parent_list[1], [-1, 0, 1])
        self.assertEqual(raw.accept_lens, [3, 1])
        self.assertIsNone(raw.accept_index)

    def test_filter_rejects_missing_required_field(self):
        raw = self._raw()
        raw.parent_list = None
        with self.assertRaisesRegex(RuntimeError, "requires a relayed or dummy"):
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
                        verify_mask=verify_mask, max_context_len=4096
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
        kernel_result = tuple(torch.tensor([i]) for i in range(1, 6)) + (arranged,)
        with patch(
            "sglang.srt.speculative.eagle_worker_v2.build_tree_kernel_efficient",
            return_value=kernel_result,
        ) as build_tree:
            verify = EAGLEWorkerV2._build_verify_input_from_pp_raw(
                self._worker(), batch
            )
        self.assertEqual(
            build_tree.call_args.args[3].tolist(), [[11, 12, 13], [21, 22, 23]]
        )
        self.assertIs(batch.input_ids, arranged)
        self.assertEqual(verify.draft_token_num, 4)

    def test_rebuild_uses_rank_local_verify_mask(self):
        raw = TestEaglePPVerifyInputRaw._raw()
        batch = self._batch(raw)
        mask_buffer = torch.empty(128, dtype=torch.bool)
        verify_mask = VerifyMask(
            buffer=mask_buffer,
            mode=TreeMaskMode.QLEN_ONLY,
            max_bs=2,
            is_read=False,
        )
        arranged = torch.arange(8)
        kernel_result = (
            (mask_buffer,) + tuple(torch.tensor([i]) for i in range(2, 6)) + (arranged,)
        )
        with patch(
            "sglang.srt.speculative.eagle_worker_v2.build_tree_kernel_efficient",
            return_value=kernel_result,
        ) as build_tree:
            EAGLEWorkerV2._build_verify_input_from_pp_raw(
                self._worker(verify_mask), batch
            )
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
    @patch("sglang.srt.speculative.eagle_worker_v2.get_pp_group")
    @patch("sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker")
    def test_non_last_rank_does_not_construct_draft_worker(
        self, draft_worker_cls, get_pp_group, _get_plan_stream
    ):
        get_pp_group.return_value.is_last_rank = False
        target = self._target(is_last_rank=False)
        server_args = SimpleNamespace(pp_size=2)
        with (
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_parallel",
                return_value=SimpleNamespace(pp_size=2),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_spec",
                return_value=SimpleNamespace(
                    speculative_eagle_topk=1,
                    speculative_num_steps=3,
                    speculative_num_draft_tokens=4,
                    speculative_algorithm="EAGLE",
                    speculative_adaptive=False,
                ),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_device",
                return_value=SimpleNamespace(device="cpu"),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_schedule",
                return_value=SimpleNamespace(page_size=1),
            ),
        ):
            worker = EAGLEWorkerV2(server_args, 0, object(), 1234, target_worker=target)
        draft_worker_cls.assert_not_called()
        self.assertIsNone(worker.draft_worker)
        self.assertEqual(
            worker.spec_v2_attn_backends, (target.model_runner.attn_backend,)
        )

    def test_pp_idle_build_does_not_call_draft_worker(self):
        draft_worker = SimpleNamespace(draft=MagicMock())
        worker = SimpleNamespace(
            draft_worker=draft_worker,
            topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            device="cpu",
        )
        verify_input = EAGLEWorkerV2._build_idle_verify_input(worker, SimpleNamespace())
        draft_worker.draft.assert_not_called()
        self.assertTrue(verify_input.is_verify_input())

    def test_pp_draft_keeps_checkpoint_embedding_and_shares_only_head(self):
        target_model = SimpleNamespace(
            get_head=MagicMock(return_value=object()),
            get_embed_and_head=MagicMock(side_effect=AssertionError("missing embed")),
            lm_head=None,
        )
        draft_model = SimpleNamespace(
            set_head=MagicMock(),
            set_embed_and_head=MagicMock(),
            hot_token_id=None,
        )
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(
                pp_group=SimpleNamespace(world_size=2),
                model_runner=SimpleNamespace(model=target_model),
            ),
            draft_runner=SimpleNamespace(model=draft_model),
            speculative_algorithm=SpeculativeAlgorithm.EAGLE,
            hot_token_id=None,
        )

        EagleDraftWorker.init_lm_head(worker)

        target_model.get_head.assert_called_once_with()
        target_model.get_embed_and_head.assert_not_called()
        draft_model.set_head.assert_called_once_with(target_model.get_head.return_value)
        draft_model.set_embed_and_head.assert_not_called()


class TestPPMTPCompatibility(unittest.TestCase):
    def test_eagle_and_dspark_allow_partitioned_mtp_target(self):
        for algorithm in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.DSPARK):
            with self.subTest(algorithm=algorithm):
                _assert_pp_mtp_compat(
                    model_has_mtp_layers=True,
                    spec_algorithm=algorithm,
                    num_effective_layers=39,
                    model_num_layers=78,
                )

    def test_other_spec_algorithm_still_rejects_partitioned_mtp_target(self):
        with self.assertRaisesRegex(AssertionError, "not compatible with MTP"):
            _assert_pp_mtp_compat(
                model_has_mtp_layers=True,
                spec_algorithm=SpeculativeAlgorithm.NGRAM,
                num_effective_layers=39,
                model_num_layers=78,
            )


class TestPPDraftKVAccounting(unittest.TestCase):
    @staticmethod
    def _config(pp_rank):
        spec_algorithm = MagicMock()
        spec_algorithm.is_eagle.return_value = True
        spec_algorithm.is_standalone.return_value = False
        spec_algorithm.is_dflash_family.return_value = False
        return SimpleNamespace(
            kv_cache_dtype_str="auto",
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
                context_len=4096,
            ),
            layer_info=SimpleNamespace(
                start_layer=pp_rank * 16,
                end_layer=(pp_rank + 1) * 16,
                num_effective_layers=16,
            ),
            ps=SimpleNamespace(pp_size=2),
            pp_group=SimpleNamespace(is_last_rank=pp_rank == 1),
            spec_algorithm=spec_algorithm,
            spec_aux_config=SimpleNamespace(eagle_draft_num_layers=4),
            is_draft_worker=False,
        )

    def test_only_last_target_stage_reserves_draft_kv(self):
        with (
            patch.object(
                DefaultPoolConfigurator, "_compute_cell_size", return_value=1600
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.mambaish_config",
                return_value=None,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_schedule",
                return_value=SimpleNamespace(max_total_tokens=4096),
            ),
        ):
            first = DefaultPoolConfigurator(self._config(0))
            last = DefaultPoolConfigurator(self._config(1))

        self.assertEqual(first._cell_size, 1600)
        self.assertEqual(last._cell_size, 2000)


if __name__ == "__main__":
    unittest.main()
