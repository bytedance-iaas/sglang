"""V4.1 low-ratio sources must not consume EP communication padding."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention import deepseek_v4_backend as backend_module
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestV41MegaMoEPadding(CustomTestCase):
    def test_compact_verify_sources_follow_packed_request_boundaries(self):
        """A shortened verify prefix must move the next request's source rows."""
        backend = backend_module.DeepseekV4AttnBackend.__new__(
            backend_module.DeepseekV4AttnBackend
        )
        backend.speculative_num_draft_tokens = 6
        backend.MAX_SEQ_LEN_FOR_CAPTURE = 64
        backend.has_c4 = backend.has_c128 = False
        backend.req_to_token = torch.empty(8, 64, dtype=torch.int64)
        backend.token_to_kv_pool = object()
        backend.init_forward_metadata_indexer = lambda *args, **kwargs: None
        backend.make_core_attn_metadata = lambda **kw: SimpleNamespace(
            low_ratios=[1, 2], positions_casual=kw["seq_lens_casual"] - 1
        )
        raw = backend_module.DSV4RawVerifyMetadata(
            req_pool_indices=torch.tensor([2, 5, 0]),
            seq_lens=torch.tensor([10, 30, 1]),
            out_cache_loc=torch.zeros(12, dtype=torch.int64),
            extend_seq_lens=torch.tensor([1, 4, 7], dtype=torch.int32),
            seq_lens_cpu=None,
            c128_compress_metadata=None,
            extend_start_loc=torch.tensor([0, 1, 5], dtype=torch.int32),
            verify_lens=torch.tensor([1, 4, 7], dtype=torch.int32),
            total_verify_tokens=12,
        )
        metadata = backend.make_forward_metadata_from_raw_verify(raw)
        self.assertEqual(metadata.low_ratio_req_indices[:5].tolist(), [2, 5, 5, 5, 5])
        self.assertEqual(metadata.low_ratio_pos_i64[:5].tolist(), [10, 30, 31, 32, 33])

    def test_target_verify_keeps_all_rows_with_or_without_extend_lengths(self):
        for extend_lengths in (None, [1, 1]):
            with self.subTest(extend_lengths=extend_lengths):
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    encoder_swa_replay=False,
                    extend_seq_lens_cpu=extend_lengths,
                )
                x = torch.randn(12, 8)
                q = torch.randn(12, 4)
                pos = torch.arange(12)
                req = torch.tensor([2] * 6 + [5] * 6)
                backend = SimpleNamespace(
                    forward_metadata=SimpleNamespace(
                        low_ratio_req_indices=req,
                        low_ratio_pos_i64=pos,
                    ),
                    _low_ratio_in_prefill_graph=Mock(return_value=False),
                    _low_ratio_compress=Mock(),
                    _low_ratio_index_topk=Mock(),
                )
                layer = SimpleNamespace(compressor=object(), indexer=object())
                with (
                    patch.object(
                        backend_module, "dsa_use_prefill_cp", return_value=False
                    ),
                    patch.object(
                        backend_module,
                        "get_moe_a2a_backend",
                        return_value=SimpleNamespace(is_megamoe=lambda: True),
                    ),
                ):
                    backend_module.DeepseekV4AttnBackend.forward_low_ratio_sources(
                        backend,
                        layer=layer,
                        x=x,
                        q_lora=q,
                        positions=pos,
                        forward_batch=batch,
                    )
                args = backend._low_ratio_compress.call_args.args
                torch.testing.assert_close(args[1], x)
                torch.testing.assert_close(args[2], req)
                torch.testing.assert_close(args[3], pos)
                index_args = backend._low_ratio_index_topk.call_args.args
                torch.testing.assert_close(index_args[1], x)
                torch.testing.assert_close(index_args[2], q)

    def test_prefill_sources_use_real_rows_without_mutating_moe_inputs(self):
        for megamoe, padded in ((True, 8), (True, 6), (False, 6)):
            with self.subTest(megamoe=megamoe, padded=padded):
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    encoder_swa_replay=False,
                    req_pool_indices=torch.tensor([2, 5]),
                    extend_seq_lens=torch.tensor([2, 4]),
                    extend_seq_lens_cpu=[2, 4],
                )
                backend = SimpleNamespace(
                    forward_metadata=SimpleNamespace(),
                    _low_ratio_in_prefill_graph=Mock(return_value=False),
                    _low_ratio_compress=Mock(),
                    _low_ratio_index_topk=Mock(),
                )
                layer = SimpleNamespace(compressor=object(), indexer=object())
                x = torch.randn(padded, 8)
                q = torch.randn(padded, 4)
                pos = torch.arange(padded)
                with (
                    patch.object(
                        backend_module, "dsa_use_prefill_cp", return_value=False
                    ),
                    patch.object(
                        backend_module,
                        "get_moe_a2a_backend",
                        return_value=SimpleNamespace(is_megamoe=lambda: megamoe),
                    ),
                ):
                    backend_module.DeepseekV4AttnBackend.forward_low_ratio_sources(
                        backend,
                        layer=layer,
                        x=x,
                        q_lora=q,
                        positions=pos,
                        forward_batch=batch,
                    )
                args = backend._low_ratio_compress.call_args.args
                torch.testing.assert_close(args[1], x[:6])
                self.assertEqual(args[2].tolist(), [2, 2, 5, 5, 5, 5])
                torch.testing.assert_close(args[3], pos[:6])
                index_args = backend._low_ratio_index_topk.call_args.args
                torch.testing.assert_close(index_args[1], x[:6])
                torch.testing.assert_close(index_args[2], q[:6])
                self.assertEqual(x.shape[0], padded)
                self.assertEqual(batch.extend_seq_lens_cpu, [2, 4])


if __name__ == "__main__":
    unittest.main()
