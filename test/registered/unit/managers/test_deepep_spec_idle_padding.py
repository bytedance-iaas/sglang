"""Regression coverage for EAGLE idle ranks with DeepEP low-latency."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.managers.scheduler_components.dp_attn import _update_gather_batch
from sglang.srt.model_executor.forward_batch_info import (
    ForwardMode,
    _should_force_symmetric_spec_deepep_padding,
    _should_materialize_idle_target_verify,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

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


def _deepep_low_latency_patches():
    backend = SimpleNamespace(is_deepep=lambda: True)
    mode = SimpleNamespace(
        resolve=lambda _is_extend: SimpleNamespace(is_low_latency=lambda: True)
    )
    return (
        patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=backend,
        ),
        patch("sglang.srt.layers.moe.utils.get_deepep_mode", return_value=mode),
    )


class TestDeepEPSpecIdlePadding(CustomTestCase):
    def test_eagle_mixed_active_idle_keeps_peer_counts(self):
        batch = SimpleNamespace(spec_algorithm=SimpleNamespace(is_eagle=lambda: True))
        sync_info = _sync_info([6, 0, 6, 6])
        backend_patch, mode_patch = _deepep_low_latency_patches()

        with backend_patch, mode_patch:
            _update_gather_batch(
                batch,
                sync_info,
                require_mlp_tp_gather=False,
            )

        self.assertEqual(batch.global_num_tokens, [6, 0, 6, 6])
        self.assertEqual(batch.global_num_tokens_for_logprob, [6, 0, 6, 6])

    def test_non_spec_deepep_keeps_rank_local_counts(self):
        batch = SimpleNamespace(spec_algorithm=None)
        sync_info = _sync_info([6, 0, 6, 6])
        backend_patch, mode_patch = _deepep_low_latency_patches()

        with backend_patch, mode_patch:
            _update_gather_batch(
                batch,
                sync_info,
                require_mlp_tp_gather=False,
            )

        self.assertEqual(batch.global_num_tokens, [0])
        self.assertEqual(batch.global_num_tokens_for_logprob, [0])

    def test_idle_verify_materializes_only_under_symmetric_padding(self):
        verify_info = SimpleNamespace(is_draft_input=lambda: False)
        draft_info = SimpleNamespace(is_draft_input=lambda: True)

        self.assertTrue(
            _should_materialize_idle_target_verify(
                forward_mode=ForwardMode.IDLE,
                spec_info=verify_info,
                dp_padding_mode=DpPaddingMode.MAX_LEN,
                num_tokens=6,
            )
        )
        for spec_info, padding, num_tokens in (
            (verify_info, DpPaddingMode.SUM_LEN, 6),
            (verify_info, DpPaddingMode.MAX_LEN, 0),
            (draft_info, DpPaddingMode.MAX_LEN, 6),
        ):
            self.assertFalse(
                _should_materialize_idle_target_verify(
                    forward_mode=ForwardMode.IDLE,
                    spec_info=spec_info,
                    dp_padding_mode=padding,
                    num_tokens=num_tokens,
                )
            )

    def test_sparse_eagle_verify_forces_symmetric_deepep_padding(self):
        algorithm = SimpleNamespace(is_eagle=lambda: True)
        verify_info = SimpleNamespace(is_draft_input=lambda: False)
        draft_info = SimpleNamespace(is_draft_input=lambda: True)
        backend_patch, mode_patch = _deepep_low_latency_patches()

        with backend_patch, mode_patch:
            self.assertTrue(
                _should_force_symmetric_spec_deepep_padding(
                    spec_algorithm=algorithm,
                    spec_info=verify_info,
                    is_extend_in_batch=False,
                    global_num_tokens=[0, 6, 6, 0, 0, 0, 0, 0],
                )
            )
            self.assertFalse(
                _should_force_symmetric_spec_deepep_padding(
                    spec_algorithm=algorithm,
                    spec_info=draft_info,
                    is_extend_in_batch=False,
                    global_num_tokens=[0, 1, 1, 0, 0, 0, 0, 0],
                )
            )


if __name__ == "__main__":
    unittest.main()
