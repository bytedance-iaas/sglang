"""CPU contracts for generic MXFP4 loading on SM90 MegaMoE."""

from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.mega_moe_sm90 import (
    _resolve_sm90_fp4_weight_transform,
    build_sm90_fp4_mega_moe_experts_weights,
    is_sm90_fp4_mega_moe_available,
    run_sm90_mega_routed,
)
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _normalize_packed_expert_param_name,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMxfp4PackedLoaderContract(CustomTestCase):
    def test_sm90_megamoe_rejects_non_deep_gemm_runner(self):
        runner = SimpleNamespace(
            is_triton_kernels=lambda: False,
            is_flashinfer_mxfp4=lambda: False,
            is_marlin=lambda: False,
            is_deep_gemm=lambda: False,
        )
        backend = SimpleNamespace(is_megamoe=lambda: True)
        with (
            mock.patch(
                "sglang.srt.layers.quantization.mxfp4.get_moe_runner_backend",
                return_value=runner,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.mxfp4.get_moe_a2a_backend",
                return_value=backend,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.mxfp4.is_sm90_supported",
                return_value=True,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "requires --moe-runner-backend deep_gemm"
            ):
                Mxfp4MoEMethod(prefix="model.layers.0.mlp.experts")

    def test_packed_checkpoint_name_maps_to_generic_weight(self):
        param = torch.nn.Parameter(torch.empty(1), requires_grad=False)
        params = {"model.experts.w13_weight": param}
        self.assertEqual(
            _normalize_packed_expert_param_name(
                "model.experts.w13_weight_packed", params
            ),
            "model.experts.w13_weight",
        )

    def test_existing_packed_parameter_name_is_preserved(self):
        packed = torch.nn.Parameter(torch.empty(1), requires_grad=False)
        params = {
            "model.experts.w13_weight": torch.nn.Parameter(
                torch.empty(1), requires_grad=False
            ),
            "model.experts.w13_weight_packed": packed,
        }
        self.assertEqual(
            _normalize_packed_expert_param_name(
                "model.experts.w13_weight_packed", params
            ),
            "model.experts.w13_weight_packed",
        )

    def test_sm90_megamoe_post_load_uses_raw_scale_layout(self):
        layer = torch.nn.Module()
        layer.register_parameter(
            "w13_weight",
            torch.nn.Parameter(
                torch.zeros((1, 16, 4), dtype=torch.uint8), requires_grad=False
            ),
        )
        layer.register_parameter(
            "w2_weight",
            torch.nn.Parameter(
                torch.zeros((1, 8, 4), dtype=torch.uint8), requires_grad=False
            ),
        )
        for name, shape in (
            ("w13_weight_scale", (1, 16, 4)),
            ("w2_weight_scale", (1, 8, 4)),
        ):
            layer.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.full(shape, 127, dtype=torch.uint8), requires_grad=False
                ),
            )

        method = object.__new__(Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deep_gemm = True
        backend = SimpleNamespace(is_megamoe=lambda: True)
        build = mock.Mock()
        generic_transform = mock.Mock(
            side_effect=AssertionError("generic SM100 transform must not run")
        )
        deep_gemm = SimpleNamespace(transform_sf_into_required_layout=generic_transform)
        with (
            mock.patch.dict("sys.modules", {"deep_gemm": deep_gemm}),
            mock.patch(
                "sglang.srt.layers.quantization.mxfp4.get_moe_a2a_backend",
                return_value=backend,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.mxfp4.is_sm90_supported",
                return_value=True,
            ),
            mock.patch(
                "sglang.srt.layers.moe.mega_moe_sm90."
                "build_sm90_fp4_mega_moe_experts_weights",
                build,
            ),
        ):
            method.process_weights_after_loading(layer)

        build.assert_called_once_with(layer)
        generic_transform.assert_not_called()
        self.assertEqual(layer.w13_weight.dtype, torch.int8)
        self.assertEqual(layer.w2_weight.dtype, torch.int8)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.float32)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.float32)
        self.assertTrue(torch.all(layer.w13_weight_scale == 1.0))
        self.assertTrue(torch.all(layer.w2_weight_scale == 1.0))
        self.assertEqual(layer._mxfp4_backend, "deep_gemm")


class TestSm90Fp4MegaMoEContract(CustomTestCase):
    @staticmethod
    def _make_experts():
        experts = torch.nn.Module()
        for name, tensor in (
            ("w13_weight", torch.zeros((2, 32, 4), dtype=torch.int8)),
            ("w2_weight", torch.zeros((2, 8, 4), dtype=torch.int8)),
            ("w13_weight_scale", torch.ones((2, 32, 4), dtype=torch.float32)),
            ("w2_weight_scale", torch.ones((2, 8, 4), dtype=torch.float32)),
        ):
            experts.register_parameter(
                name, torch.nn.Parameter(tensor, requires_grad=False)
            )
        return experts

    def test_compat_transform_matches_pr53_layout(self):
        deep_gemm = SimpleNamespace(
            fp8_fp4_mega_moe=mock.Mock(),
            mega_moe_pre_dispatch_sm90=mock.Mock(),
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
        torch.testing.assert_close(
            l1_scale_out.view(torch.uint8),
            (127 + row_order).to(torch.uint8).reshape(1, 32, 1).expand(-1, -1, 4),
        )
        torch.testing.assert_close(
            l2_scale_out.view(torch.uint8),
            (127 + torch.arange(8)).to(torch.uint8).reshape(1, 8, 1).expand(-1, -1, 4),
        )
        self.assertEqual(l1_scale_out.dtype, torch.int32)
        self.assertEqual(l2_scale_out.dtype, torch.int32)
        self.assertTrue(l1_scale_out.is_contiguous())
        self.assertTrue(l2_scale_out.is_contiguous())

    def test_builder_uses_native_transform_when_available(self):
        experts = self._make_experts()
        l1 = (torch.ones(1, dtype=torch.int8), torch.ones(1))
        l2 = (torch.ones(1, dtype=torch.int8), torch.ones(1))
        transform = mock.Mock(return_value=(l1, l2))
        deep_gemm = SimpleNamespace(
            transform_weights_for_mega_moe_sm90_fp4=transform,
            fp8_fp4_mega_moe=mock.Mock(),
            mega_moe_pre_dispatch_sm90=mock.Mock(),
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

    def test_builder_works_without_native_transform(self):
        experts = self._make_experts()
        deep_gemm = SimpleNamespace(
            fp8_fp4_mega_moe=mock.Mock(),
            mega_moe_pre_dispatch_sm90=mock.Mock(),
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

    def test_transform_fails_closed_without_kernel(self):
        with self.assertRaisesRegex(RuntimeError, "missing fp8_fp4_mega_moe"):
            _resolve_sm90_fp4_weight_transform(
                SimpleNamespace(mega_moe_pre_dispatch_sm90=mock.Mock())
            )

    def test_availability_requires_sm90_kernel_and_built_weights(self):
        experts = SimpleNamespace(_mega_moe_sm90_fp4_weights=True)
        deep_gemm = SimpleNamespace(
            fp8_fp4_mega_moe=mock.Mock(),
            mega_moe_pre_dispatch_sm90=mock.Mock(),
        )
        with (
            mock.patch.dict("sys.modules", {"deep_gemm": deep_gemm}),
            mock.patch("sglang.srt.layers.moe.mega_moe_sm90._device_sm", 90),
        ):
            self.assertTrue(is_sm90_fp4_mega_moe_available(experts))

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
