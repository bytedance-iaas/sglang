"""CPU-only coverage for PP speculative decode proxy buffer sizing."""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner.base_runner import _allocate_decode_buffers
from sglang.srt.model_executor.runner_utils.buffers import DecodeInputBuffers
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPPSpecEnvironment(unittest.TestCase):
    def test_gate_is_registered_and_disabled_by_default(self):
        envs.SGLANG_ENABLE_PP_SPEC.clear()
        self.assertFalse(envs.SGLANG_ENABLE_PP_SPEC.get())

    def test_gate_parses_explicit_opt_in(self):
        with envs.SGLANG_ENABLE_PP_SPEC.override("1"):
            self.assertTrue(envs.SGLANG_ENABLE_PP_SPEC.get())


class TestPPSpecDecodeBuffers(unittest.TestCase):
    _MAX_BS = 4
    _MAX_NUM_TOKEN = 24
    _HIDDEN_SIZE = 16
    _TOPK_SIZE = 8

    def _common_kwargs(self):
        return dict(
            device=torch.device("cpu"),
            max_bs=self._MAX_BS,
            max_num_token=self._MAX_NUM_TOKEN,
            hidden_size=self._HIDDEN_SIZE,
            dtype=torch.float32,
            dp_size=1,
            pp_size=2,
            is_encoder_decoder=False,
            require_mlp_tp_gather=False,
            seq_len_fill_value=32,
            encoder_len_fill_value=0,
            num_tokens_per_req=6,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
            pp_proxy_topk_size=self._TOPK_SIZE,
        )

    def _assert_token_major_proxy(self, proxy):
        self.assertEqual(
            proxy["hidden_states"].shape,
            (self._MAX_NUM_TOKEN, self._HIDDEN_SIZE),
        )
        self.assertEqual(
            proxy["residual"].shape,
            (self._MAX_NUM_TOKEN, self._HIDDEN_SIZE),
        )
        self.assertEqual(
            proxy["topk_indices"].shape,
            (self._MAX_NUM_TOKEN, self._TOPK_SIZE),
        )

    def test_legacy_allocator_uses_verify_token_width(self):
        buffers = _allocate_decode_buffers(vocab_size=32, **self._common_kwargs())
        self._assert_token_major_proxy(buffers.pp_proxy_tensors)

    def test_decode_input_buffers_uses_verify_token_width(self):
        buffers = DecodeInputBuffers.create(
            next_token_logits_buffer=torch.zeros(
                (self._MAX_NUM_TOKEN, 32), dtype=torch.float32
            ),
            **self._common_kwargs(),
        )
        self._assert_token_major_proxy(buffers.pp_proxy_tensors)


if __name__ == "__main__":
    unittest.main()
