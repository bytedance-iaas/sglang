"""CPU contract tests for the opt-in SM90 GLM-5.3 KDA PTPC adapter."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.quantization.unquant as unquant
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.unquant import (
    Glm53KdaPackedPtpcLinearMethod,
    Glm53KdaPtpcLinearMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.models import glm5_next
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


def _fake_weight_quant(weight: torch.Tensor):
    return weight.t(), torch.ones(1, weight.shape[0], dtype=torch.float32)


class TestGlm53KdaPtpcMethod(unittest.TestCase):
    def test_small_m_falls_back_and_large_m_uses_fp8_copy(self):
        torch.manual_seed(0)
        layer = _FakeLinear(torch.randn(32, 16, dtype=torch.bfloat16))
        method = Glm53KdaPtpcLinearMethod("o_proj", fp8_min_m=4)
        small = torch.randn(3, 16, dtype=torch.bfloat16)

        with patch.object(
            unquant,
            "_quantize_glm53_kda_ptpc_weight",
            side_effect=_fake_weight_quant,
        ):
            method.process_weights_after_loading(layer)

        torch.testing.assert_close(
            method.apply(layer, small), F.linear(small, layer.weight)
        )
        self.assertNotIn("_fp8_ptpc_weight", layer.state_dict())

        large = torch.randn(4, 16, dtype=torch.bfloat16)
        sentinel = torch.randn(4, 32, dtype=torch.bfloat16)
        with patch(
            "sglang.srt.layers.quantization.fp8_utils.apply_fp8_linear",
            return_value=sentinel,
        ) as apply_fp8:
            actual = method.apply(layer, large)
        self.assertIs(actual, sentinel)
        args, kwargs = apply_fp8.call_args
        self.assertIs(args[0], large)
        self.assertEqual(args[1].shape, torch.Size([16, 32]))
        self.assertEqual(args[1].stride(), (1, 16))
        self.assertEqual(args[2].shape, torch.Size([1, 32]))
        self.assertTrue(kwargs["cutlass_fp8_supported"])

    def test_packed_copy_excludes_beta_and_keeps_beta_bf16(self):
        qkv_size, beta_size, fg_size, k = 32, 4, 16, 16
        weight = (
            torch.arange((qkv_size + beta_size + fg_size) * k, dtype=torch.float32)
            .view(-1, k)
            .to(torch.bfloat16)
        )
        layer = _FakeLinear(weight.clone())
        method = Glm53KdaPackedPtpcLinearMethod(
            fp8_min_m=2,
            qkv_size=qkv_size,
            beta_size=beta_size,
            fg_size=fg_size,
        )
        captured = []

        def quantize(value):
            captured.append(value.clone())
            return _fake_weight_quant(value)

        with patch.object(
            unquant, "_quantize_glm53_kda_ptpc_weight", side_effect=quantize
        ):
            method.process_weights_after_loading(layer)

        expected_packed = torch.cat((weight[:qkv_size], weight[qkv_size + beta_size :]))
        torch.testing.assert_close(captured[0], expected_packed)

        x = torch.randn(2, k, dtype=torch.bfloat16)
        packed_output = torch.randn(2, qkv_size + fg_size, dtype=torch.bfloat16)
        with patch(
            "sglang.srt.layers.quantization.fp8_utils.apply_fp8_linear",
            return_value=packed_output,
        ):
            qkv, beta, fg = method.apply_ptpc_prefill(layer, x)
        torch.testing.assert_close(qkv, packed_output[:, :qkv_size])
        torch.testing.assert_close(fg, packed_output[:, qkv_size:])
        torch.testing.assert_close(
            beta,
            F.linear(x, weight[qkv_size : qkv_size + beta_size]),
        )
        # The normal module call remains the complete fused BF16 fallback.
        torch.testing.assert_close(method.apply(layer, x), F.linear(x, weight))

    def test_invalid_weight_shape_fails_before_install(self):
        layer = _FakeLinear(torch.randn(17, 16, dtype=torch.bfloat16))
        method = Glm53KdaPtpcLinearMethod("o_proj", fp8_min_m=1)
        with (
            patch.object(
                unquant,
                "_quantize_glm53_kda_ptpc_weight",
                side_effect=ValueError("N and K divisible by 16"),
            ),
            self.assertRaisesRegex(ValueError, "divisible by 16"),
        ):
            method.process_weights_after_loading(layer)
        self.assertFalse(method._fp8_ptpc_ready)


class TestGlm53KdaPtpcSelection(unittest.TestCase):
    def setUp(self):
        self._ptpc_env_name = "SGLANG_OPT_GLM53_KDA_PTPC_MODULES"
        self._ptpc_env_value = os.environ.get(self._ptpc_env_name)
        self.addCleanup(self._restore_ptpc_env)
        override = get_context().override_server_args(
            device="cpu", enable_lora=False, lora_paths=None
        )
        override.install()
        self.addCleanup(override.restore)

    def _restore_ptpc_env(self):
        if self._ptpc_env_value is None:
            os.environ.pop(self._ptpc_env_name, None)
        else:
            os.environ[self._ptpc_env_name] = self._ptpc_env_value

    def _attention(self, modules: tuple[str, ...]):
        with (
            envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override(",".join(modules)),
            patch.object(
                glm5_next.Glm5NextLinearAttention,
                "_ptpc_backend_supported",
                return_value=True,
            ),
            patch.object(glm5_next, "GLM53_KDA_PTPC_PACKED_LAYOUT", (96, 4, 16)),
            patch.object(glm5_next, "GLM53_KDA_PTPC_ALLOWED_O_PROJ_K", (32,)),
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_dp_size=1,
                attn_dp_rank=0,
                attn_cp_size=1,
                attn_cp_rank=0,
                moe_tp_size=1,
            ),
        ):
            return glm5_next.Glm5NextLinearAttention(
                layer_idx=0,
                hidden_size=16,
                config=SimpleNamespace(
                    linear_attn_config={
                        "head_dim": 8,
                        "num_heads": 4,
                        "short_conv_kernel_size": 4,
                    }
                ),
                quant_config=None,
                prefix="model.layers.0.self_attn",
            )

    def test_empty_selection_keeps_bf16_methods(self):
        attention = self._attention(())
        self.assertIsInstance(
            attention.fused_qkvbfg_a_proj.quant_method, UnquantizedLinearMethod
        )
        self.assertIsInstance(attention.o_proj.quant_method, UnquantizedLinearMethod)

    def test_target_selection_installs_packed_and_o_methods(self):
        attention = self._attention(("qkv_proj", "f_a_proj", "g_a_proj", "o_proj"))
        self.assertIsInstance(
            attention.fused_qkvbfg_a_proj.quant_method,
            Glm53KdaPackedPtpcLinearMethod,
        )
        self.assertIsInstance(attention.o_proj.quant_method, Glm53KdaPtpcLinearMethod)

    def test_partial_shared_input_selection_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "selected together"):
            self._attention(("qkv_proj",))

    def test_unknown_module_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self._attention(("not_a_projection",))


if __name__ == "__main__":
    unittest.main()
