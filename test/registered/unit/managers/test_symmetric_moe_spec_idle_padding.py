"""Regression coverage for EAGLE idle ranks with symmetric MoE A2A."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.managers.scheduler_components.dp_attn import _update_gather_batch
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    _should_bypass_attention_for_symmetric_spec_moe_dummy,
    _should_force_symmetric_spec_moe_padding,
    _should_materialize_idle_spec_moe,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _sync_info(global_num_tokens):
    return SimpleNamespace(
        num_tokens=0,
        num_tokens_for_logprob=0,
        global_num_tokens=global_num_tokens,
        global_num_tokens_for_logprob=global_num_tokens,
        is_extend_in_batch=False,
        tbo_split_seq_index=None,
        global_forward_mode=None,
        can_run_decode_cuda_graph=False,
        can_run_prefill_cuda_graph=False,
    )


def _backend_patch(backend_name):
    backend = SimpleNamespace(
        is_deepep=lambda: backend_name == "deepep",
        is_megamoe=lambda: backend_name == "megamoe",
    )
    return patch(
        "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
        return_value=backend,
    )


class TestSymmetricMoeSpecIdlePadding(unittest.TestCase):
    def test_mixed_eagle_round_keeps_peer_counts(self):
        sync_info = _sync_info([1, 0, 1, 0])
        batch = SimpleNamespace(spec_algorithm=SimpleNamespace(is_eagle=lambda: True))
        with _backend_patch("megamoe"):
            _update_gather_batch(batch, sync_info, require_mlp_tp_gather=False)
        self.assertEqual(batch.global_num_tokens, [1, 0, 1, 0])

    def test_non_spec_round_keeps_rank_local_count(self):
        batch = SimpleNamespace(spec_algorithm=None)
        sync_info = _sync_info([1, 0, 1, 0])
        with _backend_patch("megamoe"):
            _update_gather_batch(batch, sync_info, require_mlp_tp_gather=False)
        self.assertEqual(batch.global_num_tokens, [0])

    def test_empty_peer_census_keeps_rank_local_count(self):
        batch = SimpleNamespace(spec_algorithm=SimpleNamespace(is_eagle=lambda: True))
        sync_info = _sync_info([])
        with _backend_patch("megamoe"):
            _update_gather_batch(batch, sync_info, require_mlp_tp_gather=False)
        self.assertEqual(batch.global_num_tokens, [0])

    def test_symmetric_padding_requires_mixed_eagle_round(self):
        eagle = SimpleNamespace(is_eagle=lambda: True)
        info = SimpleNamespace(is_draft_input=lambda: False)
        with _backend_patch("megamoe"):
            self.assertTrue(
                _should_force_symmetric_spec_moe_padding(
                    spec_algorithm=eagle,
                    spec_info=info,
                    is_extend_in_batch=False,
                    global_num_tokens=[1, 0, 1, 0],
                )
            )
            for counts in ([1, 1, 1, 1], [0, 0, 0, 0]):
                self.assertFalse(
                    _should_force_symmetric_spec_moe_padding(
                        spec_algorithm=eagle,
                        spec_info=info,
                        is_extend_in_batch=False,
                        global_num_tokens=counts,
                    )
                )

    def test_only_idle_draft_dummy_bypasses_attention(self):
        draft = SimpleNamespace(is_draft_input=lambda: True)
        verify = SimpleNamespace(is_draft_input=lambda: False)
        self.assertTrue(
            _should_bypass_attention_for_symmetric_spec_moe_dummy(
                forward_mode=ForwardMode.IDLE,
                spec_info=draft,
                force_symmetric_spec_moe_padding=True,
            )
        )
        self.assertFalse(
            _should_bypass_attention_for_symmetric_spec_moe_dummy(
                forward_mode=ForwardMode.IDLE,
                spec_info=verify,
                force_symmetric_spec_moe_padding=True,
            )
        )

    def test_idle_verify_and_draft_materialize(self):
        for is_draft in (False, True):
            self.assertTrue(
                _should_materialize_idle_spec_moe(
                    forward_mode=ForwardMode.IDLE,
                    spec_info=SimpleNamespace(is_draft_input=lambda: is_draft),
                    dp_padding_mode=DpPaddingMode.MAX_LEN,
                    num_tokens=1,
                    force_symmetric_spec_moe_padding=True,
                )
            )
        self.assertFalse(
            _should_materialize_idle_spec_moe(
                forward_mode=ForwardMode.IDLE,
                spec_info=SimpleNamespace(is_draft_input=lambda: True),
                dp_padding_mode=DpPaddingMode.SUM_LEN,
                num_tokens=1,
                force_symmetric_spec_moe_padding=True,
            )
        )
        self.assertFalse(
            _should_materialize_idle_spec_moe(
                forward_mode=ForwardMode.IDLE,
                spec_info=SimpleNamespace(is_draft_input=lambda: True),
                dp_padding_mode=DpPaddingMode.MAX_LEN,
                num_tokens=1,
                force_symmetric_spec_moe_padding=False,
            )
        )

    def test_post_forward_restores_synthetic_idle_state(self):
        spec_info = SimpleNamespace(
            is_draft_input=lambda: True,
            hidden_states=torch.empty((0, 4)),
        )
        batch = ForwardBatch(
            forward_mode=ForwardMode.IDLE,
            batch_size=0,
            input_ids=torch.zeros(1, dtype=torch.long),
            req_pool_indices=torch.zeros(1, dtype=torch.int32),
            seq_lens=torch.zeros(1, dtype=torch.int32),
            out_cache_loc=torch.zeros(4, dtype=torch.int64),
            seq_lens_sum=1,
            spec_info=spec_info,
            positions=torch.zeros(1, dtype=torch.int64),
            seq_lens_cpu=torch.zeros(1, dtype=torch.int32),
            num_token_non_padded=torch.ones((), dtype=torch.int32),
            num_token_non_padded_cpu=1,
            symmetric_spec_moe_dummy=True,
            _original_forward_mode=ForwardMode.IDLE,
            _original_batch_size=0,
            _original_num_tokens=0,
        )
        batch.hidden_states_backup = spec_info.hidden_states
        batch.output_cache_loc_backup = torch.empty(0, dtype=torch.int64)
        logits = SimpleNamespace(
            next_token_logits=torch.ones((1, 8)),
            hidden_states=torch.ones((1, 4)),
        )

        batch.post_forward_mlp_sync_batch(logits)

        self.assertEqual(batch.forward_mode, ForwardMode.IDLE)
        self.assertEqual(batch.batch_size, 0)
        self.assertEqual(batch.input_ids.numel(), 0)
        self.assertEqual(batch.out_cache_loc.numel(), 0)
        self.assertEqual(batch.num_token_non_padded.item(), 0)
        self.assertEqual(batch.num_token_non_padded_cpu, 0)
        self.assertEqual(batch.seq_lens_sum, 0)
        self.assertFalse(batch.symmetric_spec_moe_dummy)
        self.assertEqual(logits.next_token_logits.shape[0], 0)
        self.assertEqual(logits.hidden_states.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
