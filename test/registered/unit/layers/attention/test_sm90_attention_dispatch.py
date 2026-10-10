"""CPU checks for SM90 attention dispatch and resolved GEMM contracts."""

import sys
import unittest
from functools import partial
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv4.sm90_q_lora import (
    QLoRAQuantSpec,
    q_lora_quant_spec,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


class _Fp8Method:
    pass


class TestSM90QLoRADispatch(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.deepgemm = lambda *args: None
        self.triton = lambda *args: None
        self.humming = lambda *args: None
        modules = {}

        def module(name, **attrs):
            value = ModuleType(name)
            value.__dict__.update(attrs)
            modules[name] = value
            return value

        self.dg = module(
            "sglang.srt.layers.deep_gemm_wrapper", DEEPGEMM_SCALE_UE8M0=False
        )
        module("sglang.srt.layers.quantization.fp8", Fp8LinearMethod=_Fp8Method)
        module(
            "sglang.srt.layers.quantization.fp8_utils",
            deepgemm_w8a8_block_fp8_linear_with_fallback=self.deepgemm,
            triton_w8a8_block_fp8_linear=self.triton,
        )
        module(
            "sglang.srt.layers.quantization.humming_fp8",
            humming_w8a8_block_fp8_linear=self.humming,
            can_use_humming_fp8_linear=lambda layer, x: layer.humming_fp8_ready,
        )
        self.enterContext(patch.dict(sys.modules, modules))
        import sglang.srt.layers as layers

        self.enterContext(
            patch.object(layers, "deep_gemm_wrapper", self.dg, create=True)
        )
        self.x = torch.empty(192, 1280, dtype=torch.bfloat16)
        self.method = _Fp8Method()
        self.method.__dict__.update(
            use_marlin=False,
            use_mxfp8=False,
            block_quant=True,
            block_fp8_as_mxfp8=False,
            use_humming=False,
            weight_block_size=[128, 128],
            w8a8_block_fp8_linear=self.deepgemm,
        )
        self.layer = SimpleNamespace(
            quant_method=self.method,
            weight=torch.empty(256, 1280, dtype=torch.float8_e4m3fn),
            humming_fp8_ready=False,
        )

    def test_deepgemm_requires_plain_group128(self):
        self.assertEqual(
            q_lora_quant_spec(self.layer, self.x), QLoRAQuantSpec(128, True)
        )
        self.dg.DEEPGEMM_SCALE_UE8M0 = True
        self.assertIsNone(q_lora_quant_spec(self.layer, self.x))
        self.dg.DEEPGEMM_SCALE_UE8M0 = False
        self.method.weight_block_size = [32, 32]
        self.assertIsNone(q_lora_quant_spec(self.layer, self.x))
        self.method.weight_block_size = [128, 128]
        self.layer.weight = torch.empty(33, 1280, dtype=torch.float8_e4m3fn)
        self.assertIsNone(q_lora_quant_spec(self.layer, self.x))

    def test_ue8m0_triton_preserves_existing_bf16_gemm(self):
        self.method.weight_block_size = [32, 32]
        self.method.w8a8_block_fp8_linear = partial(self.triton, act_scale_ue8m0=True)
        self.assertEqual(
            q_lora_quant_spec(self.layer, self.x), QLoRAQuantSpec(32, False, True)
        )
        self.layer._block_fp8_bf16_weight = torch.empty(0)
        self.assertIsNone(q_lora_quant_spec(self.layer, self.x))
        self.assertIsNotNone(q_lora_quant_spec(self.layer, self.x[:32]))
        self.method.w8a8_block_fp8_linear = partial(self.triton, act_scale_ue8m0=False)
        self.assertIsNone(q_lora_quant_spec(self.layer, self.x[:32]))

    def test_humming_requires_packed_weights_and_resolved_runner(self):
        self.method.weight_block_size = [32, 32]
        self.method.use_humming = True
        self.method.w8a8_block_fp8_linear = partial(self.humming, act_scale_ue8m0=True)
        self.assertIsNone(q_lora_quant_spec(self.layer, self.x))
        self.layer.humming_fp8_ready = True
        self.assertEqual(
            q_lora_quant_spec(self.layer, self.x), QLoRAQuantSpec(32, False, True)
        )
        self.method.w8a8_block_fp8_linear = partial(self.triton, act_scale_ue8m0=True)
        self.assertIsNone(q_lora_quant_spec(self.layer, self.x))

    def test_unsupported_paths_keep_bf16_input(self):
        for attr, value in (
            ("use_marlin", True),
            ("use_mxfp8", True),
            ("block_quant", False),
        ):
            with self.subTest(attr=attr), patch.object(self.method, attr, value):
                self.assertIsNone(q_lora_quant_spec(self.layer, self.x))
        with patch.object(self.method, "block_fp8_as_mxfp8", True):
            self.layer.block_fp8_mxfp8_ready = True
            self.assertIsNone(q_lora_quant_spec(self.layer, self.x))
        self.method.w8a8_block_fp8_linear = lambda *args: None
        self.assertIsNone(q_lora_quant_spec(self.layer, self.x))
        self.layer.quant_method = SimpleNamespace()
        self.assertIsNone(q_lora_quant_spec(self.layer, self.x))

    def test_weight_and_input_shape_contract(self):
        self.assertIsNone(q_lora_quant_spec(self.layer, self.x[:, :512]))
        self.layer.weight = self.layer.weight.to(torch.bfloat16)
        self.assertIsNone(q_lora_quant_spec(self.layer, self.x))


class TestSM90C2VerifyDispatch(CustomTestCase):
    def setUp(self):
        super().setUp()
        from sglang.test.test_utils import maybe_stub_sgl_kernel

        maybe_stub_sgl_kernel()
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention import deepseek_v4_backend as backend
        from sglang.srt.speculative.ragged_verify import RaggedVerifyMode

        self.backend = backend
        self.envs = envs
        self.ragged = RaggedVerifyMode
        self.enterContext(envs.SGLANG_OPT_DSV41_SM90_C2_VERIFY_FUSION.override(True))
        self.enterContext(patch.object(torch.version, "cuda", "12.0"))
        self.enterContext(patch.object(backend, "_is_sm90", return_value=True))
        self.mode = self.enterContext(
            patch.object(
                backend, "read_ragged_verify_mode", return_value=RaggedVerifyMode.STATIC
            )
        )
        self.invariant = self.enterContext(
            patch(
                "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
                return_value=False,
            )
        )
        self.execution = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False)
        )
        self.enterContext(
            patch("sglang.srt.runtime_context.get_exec", return_value=self.execution)
        )
        self.state = SimpleNamespace(
            ring_size=8, kv_score_buffer=SimpleNamespace(kv=torch.empty(9, 512))
        )
        self.owner = SimpleNamespace(
            is_dspark_draft=False,
            speculative_num_draft_tokens=6,
            token_to_kv_pool=SimpleNamespace(
                get_attention_compress_states=lambda _: self.state
            ),
        )
        self.layer = SimpleNamespace(
            layer_id=2,
            compress_ratio=2,
            compressor=SimpleNamespace(
                use_fused_compress=False, norm=SimpleNamespace(weight=torch.empty(512))
            ),
        )
        self.x = SimpleNamespace(is_cuda=True, shape=(192, 5120))
        self.batch = SimpleNamespace(
            batch_size=32, forward_mode=SimpleNamespace(is_target_verify=lambda: True)
        )

    def eligible(self):
        return self.backend.DeepseekV4AttnBackend._can_use_sm90_c2_verify_fusion(
            self.owner, self.layer, self.x, self.batch
        )

    def test_static_verify_and_default_off(self):
        self.assertTrue(self.eligible())
        with self.envs.SGLANG_OPT_DSV41_SM90_C2_VERIFY_FUSION.override(False):
            self.assertFalse(self.eligible())
        self.mode.return_value = self.ragged.CAP_ACCEPT
        self.assertTrue(self.eligible())

    def test_incompatible_verify_layouts_fall_back(self):
        self.mode.return_value = self.ragged.COMPACT
        self.assertFalse(self.eligible())
        self.mode.return_value = self.ragged.STATIC
        for rows in (0, 191, 193):
            with self.subTest(rows=rows):
                self.x.shape = (rows, 5120)
                self.assertFalse(self.eligible())
        self.x.shape = (192, 5120)
        self.owner.is_dspark_draft = True
        self.assertFalse(self.eligible())
        self.owner.is_dspark_draft = False
        self.batch.forward_mode.is_target_verify = lambda: False
        self.assertFalse(self.eligible())

    def test_arch_state_and_determinism_fall_back(self):
        with patch.object(self.backend, "_is_sm90", return_value=False):
            self.assertFalse(self.eligible())
        self.state.kv_score_buffer.kv = torch.empty(9, 512, dtype=torch.bfloat16)
        self.assertFalse(self.eligible())
        self.state.kv_score_buffer.kv = torch.empty(9, 512)
        self.state.ring_size = 0
        self.assertFalse(self.eligible())
        self.state.ring_size = 8
        self.invariant.return_value = True
        self.assertFalse(self.eligible())
        self.invariant.return_value = False
        self.execution.deterministic.enable_deterministic_inference = True
        self.assertFalse(self.eligible())

    def test_other_compressors_fall_back(self):
        for ratio in (0, 1, 4, 128):
            with self.subTest(ratio=ratio):
                self.layer.compress_ratio = ratio
                self.assertFalse(self.eligible())
        self.layer.compress_ratio = 2
        self.layer.compressor.use_fused_compress = True
        self.assertFalse(self.eligible())


if __name__ == "__main__":
    unittest.main()
