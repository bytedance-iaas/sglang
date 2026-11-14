# SPDX-License-Identifier: Apache-2.0
"""Control-flow checks for read-only DSA metadata sharing across draft steps."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DeepseekSparseAttnMultiStepBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSAMetadataShare(CustomTestCase):
    def test_empty_backend_list_is_a_noop(self):
        multi = DeepseekSparseAttnMultiStepBackend.__new__(
            DeepseekSparseAttnMultiStepBackend
        )
        multi.attn_backends = []
        multi.init_cuda_graph_state(max_bs=4, max_num_tokens=8)
        multi.init_forward_metadata_out_graph(SimpleNamespace(batch_size=1))

    def test_only_first_step_allocates_full_graph_metadata(self):
        multi = DeepseekSparseAttnMultiStepBackend.__new__(
            DeepseekSparseAttnMultiStepBackend
        )
        first, second = MagicMock(), MagicMock()
        multi.attn_backends = [first, second]
        multi.init_cuda_graph_state(max_bs=4, max_num_tokens=8)
        first.init_cuda_graph_state.assert_called_once_with(4, 8)
        second._init_cuda_graph_state_shared.assert_called_once_with()

    def test_capture_aliases_later_steps_after_first(self):
        multi = DeepseekSparseAttnMultiStepBackend.__new__(
            DeepseekSparseAttnMultiStepBackend
        )
        first, second = MagicMock(), MagicMock()
        multi.attn_backends = [first, second]
        batch = SimpleNamespace(batch_size=2)
        inner = object()
        with patch(
            "sglang.srt.model_executor.forward_batch_info.build_inner_fb_view",
            return_value=inner,
        ):
            multi.init_forward_metadata_out_graph(batch, in_capture=True)
        first.init_forward_metadata_out_graph.assert_called_once_with(
            inner, in_capture=True
        )
        second._share_cuda_graph_metadata.assert_called_once_with(src=first, bs=2)

    def test_share_reuses_the_exact_metadata_object(self):
        metadata = object()
        src = SimpleNamespace(decode_cuda_graph_metadata={3: metadata})
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.decode_cuda_graph_metadata = {}
        backend.set_dsa_prefill_impl = MagicMock()
        backend._share_cuda_graph_metadata(src=src, bs=3)
        self.assertIs(backend.decode_cuda_graph_metadata[3], metadata)
        self.assertIs(backend.forward_metadata, metadata)


if __name__ == "__main__":
    unittest.main()
