"""V4.1 low-ratio sources must not consume EP communication padding."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention import deepseek_v4_backend as backend_module
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestV41MegaMoEPadding(unittest.TestCase):
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
