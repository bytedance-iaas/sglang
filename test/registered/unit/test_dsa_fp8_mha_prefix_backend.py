"""FP8 dense-prefix reads must use the full-attention child's metadata."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDSAFP8MHAPrefixBackend(CustomTestCase):
    def test_resolves_wrapped_metadata_before_dequantization(self):
        indices = torch.tensor([17, 2, 31], dtype=torch.int32)
        child = SimpleNamespace(
            forward_metadata=SimpleNamespace(page_table_1_flattened=indices)
        )
        mode = object()
        batch = SimpleNamespace(forward_mode=mode)
        split = SimpleNamespace(
            prefill_backend=child,
            decode_backend=object(),
            _select_backend=Mock(return_value=child),
        )
        hybrid = SimpleNamespace(full_attn_backend=split, forward_metadata=None)
        tbo = object.__new__(TboAttnBackend)
        tbo.primary = hybrid
        cache = object()
        pool = SimpleNamespace(get_key_buffer=Mock(return_value=cache))
        decoded = (
            torch.arange(3 * 512, dtype=torch.float32).reshape(3, 1, 512).bfloat16()
        )
        attention = SimpleNamespace(
            attn_mha=SimpleNamespace(layer_id=5), kv_lora_rank=512
        )
        for backend in (child, hybrid, tbo):
            with (
                self.subTest(backend=type(backend).__name__),
                patch.object(forward_mha, "get_attn_backend", return_value=backend),
                patch.object(forward_mha, "get_token_to_kv_pool", return_value=pool),
                patch.object(forward_mha, "_use_aiter_gfx95", False),
                patch.object(
                    forward_mha, "dequantize_k_cache_paged", return_value=decoded
                ) as dequant,
            ):
                kv, rope = (
                    forward_mha.DeepseekMHAForwardMixin._get_mla_kv_buffer_from_fp8_for_dsa(
                        attention, batch
                    )
                )
                self.assertIs(dequant.call_args.args[0], cache)
                self.assertIs(dequant.call_args.args[1], indices)
                torch.testing.assert_close(kv, decoded[:, 0], rtol=0, atol=0)
                self.assertEqual(rope.shape, (3, 1, 0))
        split._select_backend.assert_called_with(mode)
        pool.get_key_buffer.assert_called_with(5)


if __name__ == "__main__":
    unittest.main()
