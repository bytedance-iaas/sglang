import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    _should_materialize_idle_eagle_megamoe_dummy,
)
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestEagleDraftDPPadding(unittest.TestCase):
    def test_idle_eagle_megamoe_materializes_collective_valid_dummy(self):
        algorithm = SimpleNamespace(is_eagle=lambda: True)
        draft_info = SimpleNamespace(spec_input_type=SpecInputType.EAGLE_DRAFT)
        backend = SimpleNamespace(is_megamoe=lambda: True)
        supported_runner = SimpleNamespace(
            model=SimpleNamespace(supports_symmetric_spec_megamoe_dummy=True)
        )

        with patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=backend,
        ):
            self.assertTrue(
                _should_materialize_idle_eagle_megamoe_dummy(
                    model_runner=supported_runner,
                    forward_mode=ForwardMode.IDLE,
                    spec_algorithm=algorithm,
                    spec_info=draft_info,
                    dp_padding_mode=DpPaddingMode.MAX_LEN,
                    num_tokens=1,
                )
            )
            for mode, padding, tokens in (
                (ForwardMode.DECODE, DpPaddingMode.MAX_LEN, 1),
                (ForwardMode.IDLE, DpPaddingMode.SUM_LEN, 1),
                (ForwardMode.IDLE, DpPaddingMode.MAX_LEN, 0),
            ):
                self.assertFalse(
                    _should_materialize_idle_eagle_megamoe_dummy(
                        model_runner=supported_runner,
                        forward_mode=mode,
                        spec_algorithm=algorithm,
                        spec_info=draft_info,
                        dp_padding_mode=padding,
                        num_tokens=tokens,
                    )
                )
            self.assertFalse(
                _should_materialize_idle_eagle_megamoe_dummy(
                    model_runner=SimpleNamespace(model=SimpleNamespace()),
                    forward_mode=ForwardMode.IDLE,
                    spec_algorithm=algorithm,
                    spec_info=draft_info,
                    dp_padding_mode=DpPaddingMode.MAX_LEN,
                    num_tokens=1,
                )
            )
            self.assertFalse(
                _should_materialize_idle_eagle_megamoe_dummy(
                    model_runner=supported_runner,
                    forward_mode=ForwardMode.IDLE,
                    spec_algorithm=algorithm,
                    spec_info=SimpleNamespace(
                        spec_input_type=SpecInputType.EAGLE_DRAFT_EXTEND
                    ),
                    dp_padding_mode=DpPaddingMode.MAX_LEN,
                    num_tokens=1,
                )
            )

    def test_idle_megamoe_prepare_and_post_forward_round_trip(self):
        spec_info = SimpleNamespace(
            is_draft_input=lambda: True,
            spec_input_type=SpecInputType.EAGLE_DRAFT,
            num_tokens_per_req=1,
            topk_p=torch.empty((0, 1)),
            topk_index=torch.empty((0, 1), dtype=torch.int64),
            draft_probs=None,
            num_correct_drafts=None,
            hidden_states=torch.empty((0, 8)),
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.IDLE,
            batch_size=0,
            input_ids=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            seq_lens=torch.empty(0, dtype=torch.int64),
            seq_lens_sum=0,
            out_cache_loc=torch.empty(0, dtype=torch.int64),
            positions=torch.empty(0, dtype=torch.int64),
            seq_lens_cpu=torch.empty(0, dtype=torch.int64),
            spec_info=spec_info,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
            global_num_tokens_cpu=[1, 0],
            global_num_tokens_gpu=torch.tensor([1, 0]),
            global_num_tokens_for_logprob_cpu=[1, 0],
            num_token_non_padded=torch.tensor(0),
        )
        # This legacy host mirror is populated during MLP-sync preparation,
        # not accepted as a ForwardBatch constructor field.
        forward_batch.num_token_non_padded_cpu = 0
        model_runner = SimpleNamespace(
            model=SimpleNamespace(supports_symmetric_spec_megamoe_dummy=True),
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
            is_draft_worker=True,
            enable_elastic_ep=False,
            attn_tp_sequence_sharded=lambda _num_tokens: False,
            attn_backend=SimpleNamespace(
                get_cpu_graph_seq_len_fill_value=lambda: 1,
                get_cuda_graph_seq_len_fill_value=lambda: 1,
            ),
        )
        parallel = SimpleNamespace(attn_tp_size=1, attn_dp_rank=1)
        execution = SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(bs=[]))
            )
        )
        backend = SimpleNamespace(is_megamoe=lambda: True)

        with patch("sglang.srt.model_executor.forward_batch_info._is_cpu", True), patch(
            "sglang.srt.model_executor.forward_batch_info.get_parallel",
            return_value=parallel,
        ), patch(
            "sglang.srt.model_executor.forward_batch_info.get_exec",
            return_value=execution,
        ), patch(
            "sglang.srt.model_executor.forward_batch_info.set_dp_buffer_len"
        ), patch(
            "sglang.srt.model_executor.forward_batch_info.set_is_extend_in_batch"
        ), patch(
            "sglang.srt.model_executor.forward_batch_info.mambaish_config",
            return_value=None,
        ), patch(
            "sglang.srt.layers.dp_attention.DpPaddingMode.get_dp_padding_mode",
            return_value=DpPaddingMode.MAX_LEN,
        ), patch(
            "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
            return_value=1,
        ), patch(
            "sglang.srt.layers.cp.utils.enable_cp_v2", return_value=False
        ), patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=backend,
        ), patch(
            "sglang.srt.batch_overlap.two_batch_overlap.TboForwardBatchPreparer.prepare"
        ):
            forward_batch.prepare_mlp_sync_batch(model_runner)

        self.assertTrue(forward_batch.symmetric_spec_megamoe_dummy)
        self.assertEqual(forward_batch.forward_mode, ForwardMode.IDLE)
        self.assertEqual(forward_batch.batch_size, 1)
        self.assertEqual(forward_batch.input_ids.shape[0], 1)
        self.assertEqual(forward_batch.num_token_non_padded_cpu, 1)
        self.assertEqual(forward_batch.num_token_non_padded.item(), 1)
        self.assertEqual(spec_info.hidden_states.shape[0], 1)

        logits_output = SimpleNamespace(
            next_token_logits=torch.randn(1, 16), hidden_states=torch.randn(1, 8)
        )
        forward_batch.post_forward_mlp_sync_batch(logits_output)

        self.assertEqual(forward_batch.batch_size, 0)
        self.assertEqual(logits_output.next_token_logits.shape[0], 0)
        self.assertEqual(logits_output.hidden_states.shape[0], 0)
        self.assertEqual(spec_info.hidden_states.shape[0], 0)
        self.assertEqual(forward_batch.out_cache_loc.shape[0], 0)

    def test_megamoe_dummy_bypasses_attention_but_preserves_rows(self):
        hidden_states = torch.randn(1, 8)
        forward_batch = SimpleNamespace(symmetric_spec_megamoe_dummy=True)

        state = DeepseekV2AttentionMLA.forward_prepare(
            SimpleNamespace(),
            positions=torch.zeros(1, dtype=torch.int64),
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=MagicMock(),
        )

        self.assertIs(state[0], hidden_states)
        self.assertIsNone(state[1])
        self.assertIs(state[2], forward_batch)
        self.assertIsNone(state[3])

    def test_idle_runner_does_not_plan_fake_kv_for_megamoe_dummy(self):
        forward_batch = SimpleNamespace(batch_size=1, symmetric_spec_megamoe_dummy=True)
        loaded_batch = SimpleNamespace(
            input_ids=torch.zeros(1, dtype=torch.int64),
            positions=torch.zeros(1, dtype=torch.int64),
            symmetric_spec_megamoe_dummy=True,
        )
        attn_backend = MagicMock()
        model = SimpleNamespace(forward=MagicMock(return_value=object()))
        model_runner = SimpleNamespace(
            attn_backend=attn_backend,
            model=model,
            device_timer=None,
            _pp_kwargs=lambda _proxy: {},
        )
        runner = SimpleNamespace(
            model_runner=model_runner,
            enable_pdmux=False,
            load_batch=MagicMock(return_value=loaded_batch),
        )

        with patch(
            "sglang.srt.model_executor.runner.eager_runner.device_timer_ctx",
            return_value=nullcontext(),
        ):
            result = EagerRunner._execute_idle(runner, forward_batch)

        runner.load_batch.assert_called_once_with(forward_batch, None)
        attn_backend.init_forward_metadata.assert_not_called()
        self.assertIsNone(attn_backend.forward_metadata)
        model.forward.assert_called_once_with(
            loaded_batch.input_ids, loaded_batch.positions, loaded_batch
        )
        self.assertIs(result, model.forward.return_value)

if __name__ == "__main__":
    unittest.main()
