"""Draft metadata ownership and dispatch tests; not a GPU replay substitute."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DeepseekSparseAttnMultiStepBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MODULE = "sglang.srt.layers.attention.dsa_backend"


class TestDSAMultiStepMetadataSharing(unittest.TestCase):
    def _multi(self, steps=3):
        multi = object.__new__(DeepseekSparseAttnMultiStepBackend)
        multi.speculative_num_steps = steps
        multi.attn_backends = [MagicMock() for _ in range(steps - 1)]
        return multi

    def test_only_first_step_allocates_full_cuda_metadata(self):
        for steps in (2, 3, 5):
            with self.subTest(steps=steps), patch(
                MODULE + ".is_cuda", return_value=True
            ), patch(MODULE + "._is_hip", False):
                multi = self._multi(steps)
                multi.init_cuda_graph_state(40, 160)
                multi.attn_backends[0].init_cuda_graph_state.assert_called_once_with(
                    40, 160
                )
                for backend in multi.attn_backends[1:]:
                    backend.init_cuda_graph_state.assert_not_called()
                    backend._init_cuda_graph_state_shared.assert_called_once_with(
                        40, 160
                    )

    def test_hip_retains_per_step_allocation(self):
        with patch(MODULE + ".is_cuda", return_value=True), patch(
            MODULE + "._is_hip", True
        ):
            multi = self._multi()
            multi.init_cuda_graph_state(40, 160)
            for backend in multi.attn_backends:
                backend.init_cuda_graph_state.assert_called_once_with(40, 160)
                backend._init_cuda_graph_state_shared.assert_not_called()

    def test_replay_refreshes_once_and_rebinds_each_tier(self):
        with patch(MODULE + ".is_cuda", return_value=True), patch(
            MODULE + "._is_hip", False
        ):
            multi = self._multi()
            first, sibling = multi.attn_backends
            for bs in (8, 1, 40, 8):
                batch = SimpleNamespace(
                    batch_size=bs,
                    req_pool_indices=object(),
                    seq_lens=object(),
                    seq_lens_cpu=None,
                )
                first.reset_mock()
                sibling.reset_mock()
                multi.init_forward_metadata_out_graph(batch)
                first._precompute_replay_metadata.assert_called_once_with(
                    bs=bs,
                    req_pool_indices=batch.req_pool_indices,
                    seq_lens=batch.seq_lens,
                    seq_lens_cpu=None,
                    forward_mode=ForwardMode.DECODE,
                )
                first.init_forward_metadata_replay_cuda_graph_from_precomputed.assert_called_once()
                sibling.init_forward_metadata_replay_cuda_graph_from_precomputed.assert_not_called()
                sibling._share_cuda_graph_metadata.assert_called_once_with(
                    src=first, bs=bs
                )

    def test_sharing_preserves_scratch_and_step_local_state(self):
        first = object.__new__(DeepseekSparseAttnBackend)
        sibling = object.__new__(DeepseekSparseAttnBackend)
        first.dsa_drop_wide_page_table = False
        first.decode_cuda_graph_metadata = {1: object(), 8: object()}
        sibling.decode_cuda_graph_metadata = {}
        sibling.set_dsa_prefill_impl = MagicMock()
        first._multi_ctas_kv_counter_buffer = object()
        sibling._multi_ctas_kv_counter_buffer = object()
        scratch = sibling._multi_ctas_kv_counter_buffer
        sibling.speculative_step_id = 1
        for bs in (8, 1, 8):
            sibling._share_cuda_graph_metadata(first, bs)
            self.assertIs(
                sibling.forward_metadata, first.decode_cuda_graph_metadata[bs]
            )
            self.assertIs(
                sibling.decode_cuda_graph_metadata[bs], sibling.forward_metadata
            )
            self.assertIs(sibling._multi_ctas_kv_counter_buffer, scratch)
            self.assertIsNot(scratch, first._multi_ctas_kv_counter_buffer)
            self.assertEqual(sibling.speculative_step_id, 1)
        self.assertEqual(sibling.set_dsa_prefill_impl.call_count, 3)

    def test_eager_metadata_initialization_stays_independent(self):
        multi = self._multi()
        batch = object()
        multi.init_forward_metadata(batch)
        for backend in multi.attn_backends:
            backend.init_forward_metadata.assert_called_once_with(batch)
            backend._share_cuda_graph_metadata.assert_not_called()

    def test_capture_initializes_only_first_step(self):
        batch = SimpleNamespace(batch_size=8)
        inner = object()
        with patch(MODULE + ".is_cuda", return_value=True), patch(
            MODULE + "._is_hip", False
        ), patch(
            "sglang.srt.model_executor.forward_batch_info.build_inner_fb_view",
            return_value=inner,
        ):
            multi = self._multi()
            first, sibling = multi.attn_backends
            multi.init_forward_metadata_out_graph(batch, in_capture=True)
            first.init_forward_metadata_out_graph.assert_called_once_with(
                inner, in_capture=True
            )
            first._precompute_replay_metadata.assert_not_called()
            sibling.init_forward_metadata_out_graph.assert_not_called()
            sibling._share_cuda_graph_metadata.assert_called_once_with(src=first, bs=8)

    def test_backend_without_trt_counter_needs_no_reservation(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend._reserve_cuda_graph_counter(40, 160)
        backend._init_cuda_graph_state_shared(40, 160)
        self.assertEqual(backend.decode_cuda_graph_metadata, {})


if __name__ == "__main__":
    unittest.main()
