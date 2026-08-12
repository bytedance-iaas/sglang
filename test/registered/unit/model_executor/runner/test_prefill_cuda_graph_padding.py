import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.utils import (
    cal_padded_tokens,
    prepare_dsa_cache_seqlens,
)
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPrefillCudaGraphPadding(CustomTestCase):
    def _make_runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.enable_lora = False
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = Backend.TC_PIECEWISE
        runner.has_mha_companion_layers = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [4, 16]
        runner.max_num_tokens = 16
        return runner

    def _make_forward_batch(self, num_tokens):
        return SimpleNamespace(
            batch_size=1,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=None,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            input_ids=list(range(num_tokens)),
            extend_prefix_lens_cpu=[0],
        )

    def test_rejects_more_than_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertFalse(runner.can_run_graph(self._make_forward_batch(5)))

    def test_accepts_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertTrue(runner.can_run_graph(self._make_forward_batch(8)))

    @patch("sglang.srt.layers.cp.padding.get_cp_padding_align_size")
    @patch("sglang.srt.layers.attention.dsa.utils.get_parallel")
    def test_dsa_metadata_applies_attention_tp_alignment(
        self, get_parallel, get_cp_padding_align_size
    ):
        get_parallel.return_value = SimpleNamespace(
            attn_cp_size=1, attn_cp_rank=0, attn_tp_size=4
        )
        get_cp_padding_align_size.return_value = 1
        dp_padding_mode = SimpleNamespace(is_max_len=lambda: False)
        forward_batch = SimpleNamespace(
            attn_cp_metadata=None,
            dp_padding_mode=dp_padding_mode,
            global_num_tokens_cpu=[1],
            is_extend_in_batch=False,
            forward_mode=ForwardMode.DECODE,
        )

        with patch(
            "sglang.srt.layers.cp.utils.is_cp_v2_active", return_value=False
        ):
            self.assertEqual(cal_padded_tokens(forward_batch), 4)

    @patch("sglang.srt.layers.attention.dsa.utils.cal_padded_tokens", return_value=16)
    @patch("sglang.srt.layers.attention.dsa.utils.get_parallel")
    def test_dsa_flashmla_keeps_live_axis_when_dp_metadata_is_padded(
        self, get_parallel, _cal_padded_tokens
    ):
        get_parallel.return_value = SimpleNamespace(attn_cp_size=1)
        forward_batch = SimpleNamespace(
            global_num_tokens_cpu=[13], dsa_flashmla_use_live_query_axis=True
        )
        raw_cache_seqlens = torch.arange(1, 14, dtype=torch.int32)

        flashmla_seqlens, indexer_seqlens = prepare_dsa_cache_seqlens(
            forward_batch, raw_cache_seqlens
        )

        self.assertIs(flashmla_seqlens, raw_cache_seqlens)
        self.assertEqual(flashmla_seqlens.shape[0], 13)
        self.assertEqual(indexer_seqlens.shape[0], 16)
        torch.testing.assert_close(indexer_seqlens[:13], raw_cache_seqlens)
        torch.testing.assert_close(
            indexer_seqlens[13:], torch.zeros(3, dtype=torch.int32)
        )

    @patch("sglang.srt.layers.attention.dsa.utils.cal_padded_tokens", return_value=16)
    @patch("sglang.srt.layers.attention.dsa.utils.get_parallel")
    def test_dsa_flashmla_uses_padded_axis_for_eager_attention(
        self, get_parallel, _cal_padded_tokens
    ):
        get_parallel.return_value = SimpleNamespace(attn_cp_size=1)
        forward_batch = SimpleNamespace(
            global_num_tokens_cpu=[13], dsa_flashmla_use_live_query_axis=False
        )
        raw_cache_seqlens = torch.arange(1, 14, dtype=torch.int32)

        flashmla_seqlens, indexer_seqlens = prepare_dsa_cache_seqlens(
            forward_batch, raw_cache_seqlens
        )

        self.assertIs(flashmla_seqlens, indexer_seqlens)
        self.assertEqual(flashmla_seqlens.shape[0], 16)
        torch.testing.assert_close(flashmla_seqlens[:13], raw_cache_seqlens)
        torch.testing.assert_close(
            flashmla_seqlens[13:], torch.zeros(3, dtype=torch.int32)
        )


if __name__ == "__main__":
    unittest.main()
