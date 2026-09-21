"""Real FP8 Linear / DSpark coverage for the opt-in SM90 packed GEMM runner."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative.dspark import dspark_draft_model as dspark
from sglang.srt.layers.quantization import fp8_utils, humming_fp8
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import (
    init_single_process_dist,
    load_linear_weights,
    make_tp1_column_parallel_linear,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(get_device_sm() == 90, "Humming block-FP8 path requires SM90")
class TestHummingFp8Linear(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29677)

    def setUp(self):
        torch.manual_seed(7)
        backend = mock.patch.object(
            fp8_utils, "FP8_GEMM_RUNNER_BACKEND", Fp8GemmRunnerBackend.HUMMING
        )
        backend.start()
        self.addCleanup(backend.stop)
        runtime = mock.patch.object(
            humming_fp8,
            "get_exec",
            return_value=SimpleNamespace(
                deterministic=SimpleNamespace(enable_deterministic_inference=False)
            ),
        )
        runtime.start()
        self.addCleanup(runtime.stop)

    def _make_layer(self, n=576, k=1280, *, skip=False, scale_fmt="ue8m0"):
        layer = make_tp1_column_parallel_linear(
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=[32, 32],
                scale_fmt=scale_fmt,
            ),
            n,
            k,
            skip_block_quant_check=True,
        )
        w = torch.randn((n, k), device="cuda") * 0.1
        tiles = w.reshape(n // 32, 32, k // 32, 32)
        scales = torch.pow(
            2.0, torch.ceil(torch.log2(tiles.abs().amax((1, 3)).clamp_min(1e-10) / 448))
        )
        qw = (tiles / scales[:, None, :, None]).to(torch.float8_e4m3fn).reshape(n, k)
        load_linear_weights(
            layer, weight=qw, weight_scale_inv=scales.to(torch.float8_e8m0fnu)
        )
        original_weight, original_scale = layer.weight, layer.weight_scale_inv
        layer.keep_plain_weight_layout = skip
        layer.quant_method.process_weights_after_loading(layer)
        # This PR does not release or replace original checkpoint parameters.
        self.assertIs(layer.weight, original_weight)
        self.assertIs(layer.weight_scale_inv, original_scale)
        torch.testing.assert_close(layer.weight.view(torch.uint8), qw.view(torch.uint8))
        torch.testing.assert_close(layer.weight_scale_inv, scales, rtol=0, atol=0)
        self.assertFalse(any("humming" in name for name in layer.state_dict()))
        return layer

    def _reference(self, layer, x):
        n, k = layer.weight.shape
        if x.numel() == 0:
            return torch.empty((*x.shape[:-1], n), device=x.device)
        w = (
            layer.weight.float().reshape(n // 32, 32, k // 32, 32)
            * layer.weight_scale_inv.float()[:, None, :, None]
        ).reshape(n, k)
        qx, sx = fp8_utils.sglang_per_token_group_quant_fp8(
            x.reshape(-1, k).contiguous(), 32, scale_ue8m0=True
        )
        dx = qx.float() * sx.repeat_interleave(32, 1)
        return (dx @ w.T).reshape(*x.shape[:-1], n)

    def _assert_close(self, out, ref):
        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(out.isfinite().all().item())
        if out.numel():
            self.assertLess(
                ((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item(), 0.01
            )

    def _forbid_fallback(self):
        return mock.patch.object(
            fp8_utils,
            "triton_w8a8_block_fp8_linear",
            side_effect=AssertionError("packed Humming must not fall back to Triton"),
        )

    def test_dense_shapes_use_humming_without_m_limit(self):
        for n, k in (
            (1792, 5120),
            (4096, 1280),
            (5120, 1024),
            (576, 5120),
            (5120, 288),
            (25600, 6144),
            (5120, 15360),
            (512, 5120),
        ):
            with self.subTest(n=n, k=k), torch.no_grad():
                layer = self._make_layer(n, k)
                self.assertTrue(layer.humming_fp8_ready)
                self.assertIs(
                    layer.quant_method.w8a8_block_fp8_linear.func,
                    humming_fp8.humming_w8a8_block_fp8_linear,
                )
                for m in (0, 1, 6, 64, 65, 96, 1024, 1025, 8192):
                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                    with self._forbid_fallback():
                        out, _ = layer(x)
                    self._assert_close(out, self._reference(layer, x))

    def test_large_prefill_and_prequantized_inputs(self):
        for k in (288, 512, 1280):
            with self.subTest(k=k), torch.no_grad():
                layer = self._make_layer(k=k)
                for m in (96, 1025, 65536):
                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                    qx, sx = fp8_utils.sglang_per_token_group_quant_fp8(
                        x, 32, scale_ue8m0=True
                    )
                    self.assertFalse(
                        humming_fp8.can_use_humming_fp8_linear(layer, (qx, sx.t()))
                    )
                    with self._forbid_fallback():
                        self._assert_close(
                            layer.quant_method.apply(layer, (qx, sx)),
                            self._reference(layer, x),
                        )
                        self._assert_close(layer(x)[0], self._reference(layer, x))

    def test_bias_leading_dims_noncontiguous_and_cuda_graph(self):
        for k in (288, 1280):
            with torch.no_grad():
                layer = self._make_layer(k=k)
                for m in (6, 96, 1026, 8192):
                    with self.subTest(k=k, m=m), self._forbid_fallback():
                        x = torch.randn(
                            (2, m // 2, k * 2), device="cuda", dtype=torch.bfloat16
                        )[..., ::2]
                        bias = torch.randn(576, device="cuda", dtype=torch.bfloat16)
                        for _ in range(3):
                            layer.quant_method.apply(layer, x, bias)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            out = layer.quant_method.apply(layer, x, bias)
                        for _ in range(3):
                            x.normal_()
                            bias.normal_()
                            graph.replay()
                            self._assert_close(
                                out, self._reference(layer, x) + bias.float()
                            )

    def test_deterministic_and_batch_invariant_paths(self):
        deterministic = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=True)
        )
        for patcher in (
            mock.patch.object(humming_fp8, "get_exec", return_value=deterministic),
            mock.patch.object(
                humming_fp8, "is_batch_invariant_mode_enabled", return_value=True
            ),
        ):
            with patcher, torch.no_grad(), self._forbid_fallback():
                layer = self._make_layer()
                self.assertTrue(layer.humming_fp8_ready)
                x = torch.randn((1025, 1280), device="cuda", dtype=torch.bfloat16)
                first = layer(x[:1])[0]
                for m in (1, 6, 64, 96, 1025):
                    out = layer(x[:m])[0]
                    self._assert_close(out, self._reference(layer, x[:m]))
                    self.assertTrue(torch.equal(first, out[:1]))
                    self.assertTrue(torch.equal(out, layer(x[:m])[0]))

    def test_packed_runner_needs_no_original_weight(self):
        with torch.no_grad():
            layer = self._make_layer(k=288)
            x = torch.randn((96, 288), device="cuda", dtype=torch.bfloat16)
            ref = self._reference(layer, x)
            packed = humming_fp8.get_humming_fp8_weight(layer)
            runner = layer.quant_method.w8a8_block_fp8_linear
            del layer
            with self._forbid_fallback():
                out = runner(
                    input=x, weight=packed, block_size=[32, 32], weight_scale=None
                )
            self._assert_close(out, ref)
            with self.assertRaisesRegex(ValueError, "dtype or K"):
                runner(
                    input=x.half(),
                    weight=packed,
                    block_size=[32, 32],
                    weight_scale=None,
                )
            qx, sx = fp8_utils.sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
            with self.assertRaisesRegex(ValueError, "per-row group-32"):
                runner(
                    input=qx,
                    weight=packed,
                    block_size=[32, 32],
                    weight_scale=None,
                    input_scale=sx.t(),
                )

    def test_dspark_stacked_kv_uses_cached_humming(self):
        with torch.no_grad():
            stages = [self._make_layer(n=512, k=5120) for _ in range(3)]
            key = id(stages[0])
            self.addCleanup(dspark._STACKED_WEIGHT_CACHE.pop, key, None)
            x = torch.randn((96, 5120), device="cuda", dtype=torch.bfloat16)
            originals = [stage.weight.data_ptr() for stage in stages]
            with self._forbid_fallback():
                out = dspark.CommitKvProj.execute(main_x=x, wkv_linears=stages)
                self.assertIsNotNone(dspark._STACKED_WEIGHT_CACHE[key].humming_weight)
                for y, stage in zip(out, stages):
                    self._assert_close(y, self._reference(stage, x))
                # Cache construction/JIT must be finished before graph capture.
                with mock.patch.object(
                    dspark,
                    "pack_humming_fp8_weight",
                    side_effect=AssertionError("repacked during forward"),
                ):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        outputs = dspark.CommitKvProj.execute(
                            main_x=x, wkv_linears=stages, allow_strided_output=True
                        )
                    for _ in range(3):
                        x.normal_()
                        graph.replay()
                        for y, stage in zip(outputs, stages):
                            self._assert_close(y, self._reference(stage, x))
            self.assertEqual(originals, [stage.weight.data_ptr() for stage in stages])

    def test_direct_reader_and_sliced_layout_keep_raw_compatibility(self):
        with torch.no_grad():
            layer = self._make_layer(skip=True)
            self.assertFalse(layer.humming_fp8_ready)
            self.assertFalse(hasattr(layer, "_humming_fp8_weight"))
            x = torch.randn((96, 1280), device="cuda", dtype=torch.bfloat16)
            expected = fp8_utils.triton_w8a8_block_fp8_linear(
                x, layer.weight, [32, 32], layer.weight_scale_inv, act_scale_ue8m0=True
            )
            torch.testing.assert_close(layer(x)[0], expected, rtol=0, atol=0)
            layer = self._make_layer()
            half = x.half()
            self.assertFalse(humming_fp8.can_use_humming_fp8_linear(layer, half))
            expected = fp8_utils.triton_w8a8_block_fp8_linear(
                half,
                layer.weight,
                [32, 32],
                layer.weight_scale_inv,
                act_scale_ue8m0=True,
            )
            torch.testing.assert_close(layer(half)[0], expected, rtol=0, atol=0)
            for dim in (0, 1):
                layer = self._make_layer()
                if dim == 0:
                    layer.weight.data = layer.weight.data[:288]
                    layer.weight_scale_inv.data = layer.weight_scale_inv.data[:9]
                else:
                    layer.weight.data = layer.weight.data[:, :640].contiguous()
                    layer.weight_scale_inv.data = layer.weight_scale_inv.data[
                        :, :20
                    ].contiguous()
                value = x[:, : layer.weight.shape[1]].contiguous()
                self.assertFalse(humming_fp8.can_use_humming_fp8_linear(layer, value))
                expected = fp8_utils.triton_w8a8_block_fp8_linear(
                    value,
                    layer.weight,
                    [32, 32],
                    layer.weight_scale_inv,
                    act_scale_ue8m0=True,
                )
                torch.testing.assert_close(
                    layer.quant_method.apply(layer, value), expected, rtol=0, atol=0
                )

    def test_reload_refreshes_cache_without_rebinding_graph_storage(self):
        for k in (288, 1280):
            with self.subTest(k=k), torch.no_grad():
                layer = self._make_layer(k=k)
                names = (
                    "_humming_fp8_weight",
                    "_humming_fp8_weight_scale",
                    "_humming_fp8_locks",
                )
                pointers = [getattr(layer, name).data_ptr() for name in names]
                x = torch.randn((96, k), device="cuda", dtype=torch.bfloat16)
                layer(x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out, _ = layer(x)
                layer.weight.copy_((-layer.weight.float()).to(torch.float8_e4m3fn))
                layer.weight_scale_inv.mul_(2)
                layer.quant_method.process_weights_after_loading(layer)
                self.assertEqual(
                    pointers, [getattr(layer, name).data_ptr() for name in names]
                )
                graph.replay()
                self._assert_close(out, self._reference(layer, x))

    def test_opt_in_format_gate_and_hardware_validation(self):
        with torch.no_grad():
            for backend, scale_fmt in (("triton", "ue8m0"), ("humming", None)):
                with mock.patch.object(
                    fp8_utils, "FP8_GEMM_RUNNER_BACKEND", Fp8GemmRunnerBackend(backend)
                ):
                    layer = self._make_layer(scale_fmt=scale_fmt)
                self.assertFalse(layer.quant_method.use_humming)
                self.assertFalse(hasattr(layer, "_humming_fp8_weight"))
            with mock.patch.object(
                fp8_utils, "get_platform", return_value=SimpleNamespace(is_sm90=False)
            ):
                with self.assertRaisesRegex(RuntimeError, "requires SM90"):
                    fp8_utils.dispatch_w8a8_block_fp8_linear([32, 32], True)


if __name__ == "__main__":
    unittest.main()
