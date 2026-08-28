"""CPU contracts for compressed-tensors MXFP4 on SM90 MegaMoE."""

from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.mega_moe_sm90 import (
    _resolve_sm90_fp4_weight_transform,
    build_sm90_fp4_mega_moe_experts_weights,
    run_sm90_mega_routed,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsFusedMoEMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A8Mxfp4MoE,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


MXFP4_GROUP = {
    "format": "mxfp4-pack-quantized",
    "targets": ["Linear"],
    "weights": {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "strategy": "group",
        "group_size": 32,
        "dynamic": False,
    },
    "input_activations": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "token",
        "dynamic": True,
    },
}


def _config(group=MXFP4_GROUP, *, top_format="mxfp4-pack-quantized"):
    return CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": top_format,
            "config_groups": {"group_0": group},
            "ignore": [],
        }
    )


def _scheme(config=None):
    config = config or _config()
    scheme_dict = config.target_scheme_map["Linear"]
    with (
        mock.patch(
            "sglang.srt.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w4a8_mxfp4_moe.is_sm90_supported",
            return_value=True,
        ),
        mock.patch(
            "sglang.srt.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w4a8_mxfp4_moe.is_sm100_supported",
            return_value=False,
        ),
    ):
        return CompressedTensorsW4A8Mxfp4MoE(
            config,
            scheme_dict["weights"],
            scheme_dict["input_activations"],
            scheme_dict["format"],
        )


class _DummyFusedMoE(torch.nn.Module):
    pass


class TestMxfp4SchemeSelection(CustomTestCase):
    def test_mxfp4_format_preserves_dynamic_fp8_activations(self):
        scheme_dict = _config().target_scheme_map["Linear"]
        self.assertEqual(scheme_dict["format"], "mxfp4-pack-quantized")
        self.assertIsNotNone(scheme_dict["input_activations"])
        self.assertEqual(scheme_dict["input_activations"].num_bits, 8)
        self.assertTrue(scheme_dict["input_activations"].dynamic)

    def test_exact_glm_scheme_bypasses_generic_mxfp4_method(self):
        config = _config()
        layer = _DummyFusedMoE()
        with (
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.FusedMoE",
                _DummyFusedMoE,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.is_sm90_supported",
                return_value=True,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.is_sm100_supported",
                return_value=False,
            ),
        ):
            method = config.get_quant_method(layer, "model.layers.0.mlp.experts")

        self.assertIsInstance(method, CompressedTensorsFusedMoEMethod)
        self.assertIsInstance(layer.scheme, CompressedTensorsW4A8Mxfp4MoE)

    def test_other_mxfp4_shape_keeps_generic_method(self):
        other_group = dict(MXFP4_GROUP)
        other_group["input_activations"] = None
        config = _config(other_group)
        layer = _DummyFusedMoE()
        sentinel = object()
        with (
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.FusedMoE",
                _DummyFusedMoE,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.mxfp4.Mxfp4MoEMethod",
                return_value=sentinel,
            ) as generic_method,
        ):
            method = config.get_quant_method(layer, "model.layers.0.mlp.experts")

        self.assertIs(method, sentinel)
        generic_method.assert_called_once_with(prefix="model.layers.0.mlp.experts")

    def test_megamoe_scheme_rejects_sm100_without_sm90_fp4_runtime(self):
        config = _config()
        scheme_dict = config.target_scheme_map["Linear"]
        backend = SimpleNamespace(is_marlin=lambda: False, value="deep_gemm")
        with (
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.get_moe_runner_backend",
                return_value=backend,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.is_sm90_supported",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.is_sm100_supported",
                return_value=True,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "requires SM90"):
                CompressedTensorsW4A8Mxfp4MoE(
                    config,
                    scheme_dict["weights"],
                    scheme_dict["input_activations"],
                    scheme_dict["format"],
                )

    def test_marlin_scheme_keeps_sm100_support(self):
        config = _config()
        scheme_dict = config.target_scheme_map["Linear"]
        backend = SimpleNamespace(is_marlin=lambda: True, value="marlin")
        with (
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.get_moe_runner_backend",
                return_value=backend,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.is_sm90_supported",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.is_sm100_supported",
                return_value=True,
            ),
        ):
            scheme = CompressedTensorsW4A8Mxfp4MoE(
                config,
                scheme_dict["weights"],
                scheme_dict["input_activations"],
                scheme_dict["format"],
            )
        self.assertIsInstance(scheme, CompressedTensorsW4A8Mxfp4MoE)


class TestMxfp4PackedLoaderContract(CustomTestCase):
    def test_loader_mapping_hits_registered_gate_and_down_params(self):
        layer = torch.nn.Module()
        scheme = _scheme()
        scheme.create_weights(
            layer,
            num_experts=2,
            hidden_size=64,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )
        registered = dict(layer.named_parameters())

        mappings = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=2,
        )
        mapped = set()
        for param_prefix, checkpoint_prefix, _, _ in mappings:
            for suffix in ("weight_packed", "weight_scale"):
                mapped.add((param_prefix + suffix).removeprefix("experts."))
                self.assertTrue(checkpoint_prefix.startswith("experts."))

        self.assertIn("w13_weight_packed", mapped)
        self.assertIn("w2_weight_packed", mapped)
        self.assertIn("w13_weight_packed", registered)
        self.assertIn("w2_weight_packed", registered)
        self.assertNotIn("w13_weight", registered)
        self.assertNotIn("w2_weight", registered)

    def test_post_load_decodes_and_renames_without_generic_upcast(self):
        layer = torch.nn.Module()
        scheme = _scheme()
        scheme.create_weights(
            layer,
            num_experts=1,
            hidden_size=64,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )
        layer.w13_weight_scale.data.fill_(127)
        layer.w2_weight_scale.data.fill_(127)

        with mock.patch.object(scheme, "_build_mega_moe_weights") as build:
            scheme.process_weights_after_loading(layer)

        build.assert_called_once_with(layer)
        self.assertEqual(layer.w13_weight.dtype, torch.int8)
        self.assertEqual(layer.w2_weight.dtype, torch.int8)
        self.assertEqual(layer.w13_weight_scale_inv.dtype, torch.float32)
        self.assertEqual(layer.w2_weight_scale_inv.dtype, torch.float32)
        self.assertTrue(torch.all(layer.w13_weight_scale_inv == 1.0))
        self.assertTrue(torch.all(layer.w2_weight_scale_inv == 1.0))
        self.assertFalse(hasattr(layer, "w13_weight_packed"))
        self.assertFalse(hasattr(layer, "w2_weight_packed"))
        self.assertTrue(layer.is_mxfp4_converted)

    def test_megamoe_backend_fails_closed_without_sm90(self):
        scheme = _scheme()
        layer = torch.nn.Module()
        backend = SimpleNamespace(is_megamoe=lambda: True)

        with (
            mock.patch(
                "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                return_value=backend,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a8_mxfp4_moe.is_sm90_supported",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires the SM90 FP4"):
                scheme._build_mega_moe_weights(layer)


class TestSm90Fp4MegaMoEContract(CustomTestCase):
    @staticmethod
    def _make_experts():
        experts = torch.nn.Module()
        experts.register_parameter(
            "w13_weight",
            torch.nn.Parameter(
                torch.zeros((2, 128, 64), dtype=torch.int8), requires_grad=False
            ),
        )
        experts.register_parameter(
            "w2_weight",
            torch.nn.Parameter(
                torch.zeros((2, 64, 64), dtype=torch.int8), requires_grad=False
            ),
        )
        experts.register_parameter(
            "w13_weight_scale_inv",
            torch.nn.Parameter(
                torch.ones((2, 128, 4), dtype=torch.float32), requires_grad=False
            ),
        )
        experts.register_parameter(
            "w2_weight_scale_inv",
            torch.nn.Parameter(
                torch.ones((2, 64, 4), dtype=torch.float32), requires_grad=False
            ),
        )
        return experts

    def test_weight_builder_uses_sm90_fp4_transform(self):
        experts = self._make_experts()
        l1 = (torch.ones((1,), dtype=torch.int8), torch.ones((1,)))
        l2 = (torch.ones((1,), dtype=torch.int8), torch.ones((1,)))
        transform = mock.Mock(return_value=(l1, l2))
        deep_gemm = SimpleNamespace(
            transform_weights_for_mega_moe_sm90_fp4=transform,
            fp8_fp4_mega_moe=mock.Mock(),
            mega_moe_pre_dispatch_sm90=mock.Mock(),
            _C=SimpleNamespace(fp8_fp4_mega_moe_sm90=mock.Mock()),
        )

        with (
            mock.patch.dict("sys.modules", {"deep_gemm": deep_gemm}),
            mock.patch(
                "sglang.srt.layers.moe.mega_moe_sm90.envs."
                "SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get",
                return_value=False,
            ),
        ):
            build_sm90_fp4_mega_moe_experts_weights(experts)

        transform.assert_called_once()
        self.assertIs(experts.mega_l1_weights, l1)
        self.assertIs(experts.mega_l2_weights, l2)
        self.assertTrue(experts._mega_moe_sm90_fp4_weights)
        self.assertTrue(experts._mega_moe_weights_built)

    def test_weight_builder_compat_transform_matches_pr53_layout(self):
        deep_gemm = SimpleNamespace(
            fp8_fp4_mega_moe=mock.Mock(),
            mega_moe_pre_dispatch_sm90=mock.Mock(),
            _C=SimpleNamespace(fp8_fp4_mega_moe_sm90=mock.Mock()),
        )
        transform = _resolve_sm90_fp4_weight_transform(deep_gemm)

        l1_weight = torch.arange(32 * 4, dtype=torch.uint8).reshape(1, 32, 4)
        l2_weight = torch.arange(8 * 4, dtype=torch.uint8).reshape(1, 8, 4)
        l1_scale = (
            torch.pow(2.0, torch.arange(32, dtype=torch.float32))
            .reshape(1, 32, 1)
            .expand(-1, -1, 4)
            .contiguous()
        )
        l2_scale = (
            torch.pow(2.0, torch.arange(8, dtype=torch.float32))
            .reshape(1, 8, 1)
            .expand(-1, -1, 4)
            .contiguous()
        )

        (l1_weight_out, l1_scale_out), (l2_weight_out, l2_scale_out) = transform(
            (l1_weight, l1_scale), (l2_weight, l2_scale)
        )

        row_order = torch.tensor(
            list(range(0, 8))
            + list(range(16, 24))
            + list(range(8, 16))
            + list(range(24, 32))
        )
        torch.testing.assert_close(
            l1_weight_out, l1_weight.index_select(1, row_order).view(torch.int8)
        )
        torch.testing.assert_close(l2_weight_out, l2_weight.view(torch.int8))

        expected_l1_exponents = (127 + row_order).to(torch.uint8)
        expected_l2_exponents = (127 + torch.arange(8)).to(torch.uint8)
        torch.testing.assert_close(
            l1_scale_out.view(torch.uint8),
            expected_l1_exponents.reshape(1, 32, 1).expand(-1, -1, 4),
        )
        torch.testing.assert_close(
            l2_scale_out.view(torch.uint8),
            expected_l2_exponents.reshape(1, 8, 1).expand(-1, -1, 4),
        )
        self.assertEqual(l1_scale_out.dtype, torch.int32)
        self.assertEqual(l2_scale_out.dtype, torch.int32)
        self.assertTrue(l1_scale_out.is_contiguous())
        self.assertTrue(l2_scale_out.is_contiguous())

    def test_weight_builder_does_not_require_native_transform_helper(self):
        experts = self._make_experts()
        deep_gemm = SimpleNamespace(
            fp8_fp4_mega_moe=mock.Mock(),
            mega_moe_pre_dispatch_sm90=mock.Mock(),
            _C=SimpleNamespace(fp8_fp4_mega_moe_sm90=mock.Mock()),
        )
        with (
            mock.patch.dict("sys.modules", {"deep_gemm": deep_gemm}),
            mock.patch(
                "sglang.srt.layers.moe.mega_moe_sm90.envs."
                "SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get",
                return_value=False,
            ),
        ):
            build_sm90_fp4_mega_moe_experts_weights(experts)

        self.assertEqual(experts.mega_l1_weights[1].dtype, torch.int32)
        self.assertEqual(experts.mega_l2_weights[1].dtype, torch.int32)
        self.assertTrue(experts._mega_moe_sm90_fp4_weights)
        self.assertTrue(experts._mega_moe_weights_built)

    def test_compat_transform_fails_closed_on_bad_scale_shape(self):
        deep_gemm = SimpleNamespace(
            fp8_fp4_mega_moe=mock.Mock(),
            mega_moe_pre_dispatch_sm90=mock.Mock(),
            _C=SimpleNamespace(fp8_fp4_mega_moe_sm90=mock.Mock()),
        )
        transform = _resolve_sm90_fp4_weight_transform(deep_gemm)
        weight = torch.zeros((1, 16, 4), dtype=torch.int8)
        bad_scale = torch.ones((1, 16, 3), dtype=torch.float32)
        with self.assertRaisesRegex(AssertionError, "must be a multiple of 4"):
            transform((weight, bad_scale), (weight, bad_scale))

    def test_weight_builder_fails_closed_without_sm90_fp4_kernel(self):
        deep_gemm = SimpleNamespace(mega_moe_pre_dispatch_sm90=mock.Mock())
        with self.assertRaisesRegex(RuntimeError, "missing fp8_fp4_mega_moe"):
            _resolve_sm90_fp4_weight_transform(deep_gemm)

    def test_weight_builder_fails_closed_without_native_sm90_fp4_kernel(self):
        deep_gemm = SimpleNamespace(
            fp8_fp4_mega_moe=mock.Mock(),
            mega_moe_pre_dispatch_sm90=mock.Mock(),
            _C=SimpleNamespace(),
        )
        with self.assertRaisesRegex(RuntimeError, "_C.fp8_fp4_mega_moe_sm90"):
            _resolve_sm90_fp4_weight_transform(deep_gemm)

    def test_dispatch_uses_fp8_fp4_kernel_not_fp8_kernel(self):
        pre_dispatch = mock.Mock()
        fp8_fp4 = mock.Mock()
        fp8 = mock.Mock()
        deep_gemm = SimpleNamespace(
            mega_moe_pre_dispatch_sm90=pre_dispatch,
            fp8_fp4_mega_moe=fp8_fp4,
            fp8_mega_moe=fp8,
        )
        experts = SimpleNamespace(
            _mega_moe_sm90_fp4_weights=True,
            should_fuse_routed_scaling_factor_in_topk=True,
            mega_l1_weights=(torch.empty(0), torch.empty(0)),
            mega_l2_weights=(torch.empty(0), torch.empty(0)),
        )
        moe = SimpleNamespace(
            experts=experts,
            config=SimpleNamespace(hidden_size=4, swiglu_limit=None),
            routed_scaling_factor=1.0,
        )
        buf = SimpleNamespace(
            x=torch.empty((1, 4)),
            x_sf=torch.empty((1, 1)),
            topk_idx=torch.empty((1, 1), dtype=torch.int32),
            topk_weights=torch.empty((1, 1)),
        )

        with (
            mock.patch.dict("sys.modules", {"deep_gemm": deep_gemm}),
            mock.patch(
                "sglang.srt.layers.moe.mega_moe_sm90.envs."
                "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get",
                return_value=False,
            ),
        ):
            output = run_sm90_mega_routed(
                moe,
                torch.ones((1, 4), dtype=torch.bfloat16),
                torch.zeros((1, 1), dtype=torch.int32),
                torch.ones((1, 1), dtype=torch.float32),
                buf,
                1,
            )

        self.assertEqual(output.shape, (1, 4))
        pre_dispatch.assert_called_once()
        fp8_fp4.assert_called_once()
        fp8.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()
