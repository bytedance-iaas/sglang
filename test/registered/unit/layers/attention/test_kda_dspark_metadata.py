"""CPU coverage for the dense Kimi-K3 DSpARK metadata contract."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKdaDsparkMetadata(CustomTestCase):
    @staticmethod
    def _layer(*, onorm: bool, physical_tokens: int):
        layer = SimpleNamespace(
            layer_id=0,
            q_dim=2,
            k_dim=2,
            v_dim=2,
            num_v_heads=1,
            head_q_dim=2,
            head_k_dim=2,
            head_v_dim=2,
            conv_weights=torch.zeros(6, 4, dtype=torch.float32),
            A_log=torch.zeros(1, dtype=torch.float32),
            dt_bias=torch.zeros(2, dtype=torch.float32),
            lower_bound=-5.0,
            _k3_onorm_gate=(
                torch.arange(physical_tokens * 2, dtype=torch.bfloat16).view(
                    physical_tokens, 2
                )
                if onorm
                else None
            ),
            _k3_fused_decode_args=(None, None, None, None, None, torch.ones(2), 1e-5)
            if onorm
            else None,
        )
        return layer

    @staticmethod
    def _run(
        *,
        seq_len: int,
        draft_token_num: int,
        real_batch_size: int,
        physical_batch_size: int,
        onorm: bool,
        calls,
    ):
        physical_tokens = physical_batch_size * draft_token_num
        cache_indices = torch.arange(1, physical_batch_size + 1, dtype=torch.int32)
        if physical_batch_size > real_batch_size:
            cache_indices[real_batch_size:] = -1
        query_start_loc = torch.arange(
            0, seq_len + 1, draft_token_num, dtype=torch.int32
        )
        if physical_batch_size > real_batch_size:
            query_start_loc = torch.cat(
                (
                    query_start_loc,
                    torch.full(
                        (physical_batch_size - real_batch_size,),
                        seq_len,
                        dtype=torch.int32,
                    ),
                )
            )
        intermediate_state_indices = torch.arange(
            1, physical_batch_size + 1, dtype=torch.int32
        )
        mixed_qkv = torch.zeros(seq_len, 6, dtype=torch.bfloat16)
        a = torch.zeros(1, seq_len, 1, 2, dtype=torch.bfloat16)
        b = torch.zeros(1, seq_len, 1, dtype=torch.bfloat16)
        pool_size = physical_batch_size + 1
        conv_states = torch.zeros(pool_size, 3, 6, dtype=torch.bfloat16)
        intermediate_conv_window = torch.zeros(
            pool_size, draft_token_num, 3, 6, dtype=torch.bfloat16
        )
        ssm_states = torch.zeros(pool_size, 1, 2, 2, dtype=torch.float32)
        intermediate_ssm = torch.zeros(
            pool_size, draft_token_num, 1, 2, 2, dtype=torch.float32
        )

        cache = SimpleNamespace(
            conv=[conv_states],
            temporal=ssm_states,
            intermediate_ssm=intermediate_ssm,
            intermediate_conv_window=[intermediate_conv_window],
            replayssm_rawv=None,
            replayssm_rawk=None,
            replayssm_g=None,
            replayssm_beta=None,
        )
        backend = object.__new__(KDAAttnBackend)
        backend.forward_metadata = SimpleNamespace(
            query_start_loc=query_start_loc,
            mamba_cache_indices=cache_indices,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_parent_token=None,
        )
        backend.req_to_token_pool = SimpleNamespace(
            mamba2_layer_cache=lambda _layer_id: cache
        )
        backend.verify_intermediate_state_indices = intermediate_state_indices
        # The CUDA capability predicate is outside this adapter regression; the
        # actual _forward_target_verify -> _run path remains exercised below.
        backend._can_run_dspark_cutedsl_mtp = lambda **_kwargs: True

        def fake_dspark(**kwargs):
            calls.append(kwargs)
            return kwargs["x_v"]

        leaf = ModuleType("sglang.kernels.ops.kimi_k3.kda_decode_mtp")
        leaf.fused_kda_decode_mtp_dspark = fake_dspark
        with patch.dict(sys.modules, {leaf.__name__: leaf}):
            return backend._forward_target_verify(
                layer=TestKdaDsparkMetadata._layer(
                    onorm=onorm, physical_tokens=physical_tokens
                ),
                forward_batch=SimpleNamespace(
                    spec_info=SimpleNamespace(
                        draft_token_num=draft_token_num,
                        ragged_verify_layout=None,
                    )
                ),
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
            )

    def test_padded_requests_use_logical_metadata_and_norm_gate(self):
        calls = []
        output = self._run(
            seq_len=4,
            draft_token_num=4,
            real_batch_size=1,
            physical_batch_size=2,
            onorm=True,
            calls=calls,
        )

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(tuple(output.shape), (1, 4, 1, 2))
        torch.testing.assert_close(
            call["ssm_state_indices"], torch.tensor([1], dtype=torch.int32)
        )
        torch.testing.assert_close(
            call["intermediate_state_indices"], torch.tensor([1], dtype=torch.int32)
        )
        torch.testing.assert_close(
            call["cu_seqlens"], torch.tensor([0, 4], dtype=torch.int32)
        )
        self.assertEqual(tuple(call["onorm_gate"].shape), (1, 4, 1, 2))
        torch.testing.assert_close(
            call["onorm_gate"],
            torch.arange(8 * 2, dtype=torch.bfloat16)[:8].view(1, 4, 1, 2),
        )

    def test_padded_requests_use_logical_metadata_without_norm(self):
        calls = []
        output = self._run(
            seq_len=18,
            draft_token_num=6,
            real_batch_size=3,
            physical_batch_size=4,
            onorm=False,
            calls=calls,
        )

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(tuple(output.shape), (1, 18, 1, 2))
        torch.testing.assert_close(
            call["ssm_state_indices"], torch.tensor([1, 2, 3], dtype=torch.int32)
        )
        torch.testing.assert_close(
            call["intermediate_state_indices"],
            torch.tensor([1, 2, 3], dtype=torch.int32),
        )
        torch.testing.assert_close(
            call["cu_seqlens"], torch.tensor([0, 6, 12, 18], dtype=torch.int32)
        )
        self.assertIsNone(call["onorm_gate"])

    def test_dense_no_padding_preserves_full_request_metadata_and_norm_off(self):
        calls = []
        output = self._run(
            seq_len=8,
            draft_token_num=4,
            real_batch_size=2,
            physical_batch_size=2,
            onorm=False,
            calls=calls,
        )

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(tuple(output.shape), (1, 8, 1, 2))
        torch.testing.assert_close(
            call["ssm_state_indices"], torch.tensor([1, 2], dtype=torch.int32)
        )
        torch.testing.assert_close(
            call["intermediate_state_indices"], torch.tensor([1, 2], dtype=torch.int32)
        )
        torch.testing.assert_close(
            call["cu_seqlens"], torch.tensor([0, 4, 8], dtype=torch.int32)
        )
        self.assertIsNone(call["onorm_gate"])

    def test_divisible_padding_preserves_draft_width_without_norm(self):
        calls = []
        self._run(
            seq_len=4,
            draft_token_num=4,
            real_batch_size=1,
            physical_batch_size=2,
            onorm=False,
            calls=calls,
        )
        call = calls[0]
        self.assertEqual(call["x_q"].shape[1] // (call["cu_seqlens"].numel() - 1), 4)
        torch.testing.assert_close(
            call["cu_seqlens"], torch.tensor([0, 4], dtype=torch.int32)
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
