"""Real Fp8LinearMethod coverage for the opt-in SM90 Humming dense path."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp8, fp8_utils, humming_fp8
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

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(get_device_sm() == 90, "Humming block-FP8 path requires SM90")
class TestHummingFp8Linear(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29677)

    def setUp(self):
        torch.manual_seed(7)
        self.backend = mock.patch.object(
            fp8_utils, "FP8_GEMM_RUNNER_BACKEND", Fp8GemmRunnerBackend.HUMMING
        )
        self.backend.start()
        self.addCleanup(self.backend.stop)
        # Layer UTs do not publish a serving runtime context. Mock only the
        # configuration boundary; quantization, packing and GEMM stay real.
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
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[32, 32],
            scale_fmt=scale_fmt,
        )
        layer = make_tp1_column_parallel_linear(
            config, n, k, skip_block_quant_check=True
        )
        # Checkpoint-format UE8M0 block scales, including real float8 scale
        # loading rather than only testing an already-transformed weight.
        w = torch.randn((n, k), device="cuda") * 0.1
        tiles = w.reshape(n // 32, 32, k // 32, 32)
        scales = torch.pow(
            2.0, torch.ceil(torch.log2(tiles.abs().amax((1, 3)).clamp_min(1e-10) / 448))
        )
        qw = (tiles / scales[:, None, :, None]).to(torch.float8_e4m3fn).reshape(n, k)
        load_linear_weights(
            layer, weight=qw, weight_scale_inv=scales.to(torch.float8_e8m0fnu)
        )
        original_weight = layer.weight
        original_scale = layer.weight_scale_inv
        layer.keep_plain_weight_layout = skip
        layer.quant_method.process_weights_after_loading(layer)
        self.assertIs(layer.weight, original_weight)
        self.assertIs(layer.weight_scale_inv, original_scale)
        torch.testing.assert_close(layer.weight.view(torch.uint8), qw.view(torch.uint8))
        torch.testing.assert_close(layer.weight_scale_inv, scales, rtol=0, atol=0)
        return layer

    def _reference(self, layer, x):
        n, k = layer.weight.shape
        w = (
            layer.weight.float().reshape(n // 32, 32, k // 32, 32)
            * layer.weight_scale_inv.float()[:, None, :, None]
        ).reshape(n, k)
        qx, sx = fp8_utils.sglang_per_token_group_quant_fp8(
            x.reshape(-1, k).contiguous(), 32, scale_ue8m0=True
        )
        # FP32 reference of the exact group-32 quantized inputs, not a
        # different per-token quantizer or an A16 reference with loose bounds.
        dx = qx.float() * sx.repeat_interleave(32, 1)
        return (dx @ w.T).reshape(*x.shape[:-1], n)

    def _assert_close(self, out, ref):
        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(out.isfinite().all().item())
        rel_l2 = (out.float() - ref).norm() / ref.norm().clamp_min(1e-12)
        self.assertLess(rel_l2.item(), 0.01)

    def test_dense_shapes_use_humming_and_preserve_checkpoint(self):
        # QKV-A, Q-B/indexer, O-B, gate/up (N padding), Engram, main proj,
        # and per-stage KV. The stacked KV direct reader remains on Triton.
        for n, k in (
            (1792, 5120),
            (4096, 1280),
            (5120, 1024),
            (576, 5120),
            (25600, 6144),
            (5120, 15360),
            (512, 5120),
        ):
            with self.subTest(n=n, k=k), torch.no_grad():
                layer = self._make_layer(n, k)
                self.assertTrue(layer.humming_fp8_ready)
                self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
                self.assertEqual(layer.weight.shape, (n, k))
                self.assertEqual(layer.weight_scale_inv.shape, (n // 32, k // 32))
                self.assertFalse(any("humming" in name for name in layer.state_dict()))
                for m in (1, 6, 64):
                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                    with mock.patch.object(
                        fp8, "humming_fp8_linear", wraps=humming_fp8.humming_fp8_linear
                    ) as run:
                        out, _ = layer(x)
                    self.assertEqual(run.call_count, 1)
                    self._assert_close(out, self._reference(layer, x))

    def test_k576_padding_uses_humming_only_for_large_m(self):
        with (
            torch.no_grad(),
            mock.patch.object(humming_fp8, "_HUMMING_FP8_MAX_M", 8192),
        ):
            layer = self._make_layer(n=5120, k=576)
            self.assertTrue(layer.humming_fp8_ready)
            self.assertEqual(layer.weight.shape, (5120, 576))
            self.assertEqual(layer.weight_scale_inv.shape, (160, 18))
            self.assertEqual(layer._humming_fp8_config.shape_k, 1024)
            self.assertEqual(layer._humming_fp8_input_padding, 448)
            self.assertFalse(any("humming" in name for name in layer.state_dict()))

            small = torch.randn((256, 576), device="cuda", dtype=torch.bfloat16)
            with mock.patch.object(
                fp8, "humming_fp8_linear", side_effect=AssertionError
            ):
                small_out, _ = layer(small)
            expected = fp8_utils.triton_w8a8_block_fp8_linear(
                small,
                layer.weight,
                [32, 32],
                layer.weight_scale_inv,
                act_scale_ue8m0=True,
            )
            torch.testing.assert_close(small_out, expected, rtol=0, atol=0)

            large = torch.randn((1024, 576), device="cuda", dtype=torch.bfloat16)
            with mock.patch.object(
                fp8, "humming_fp8_linear", wraps=humming_fp8.humming_fp8_linear
            ) as run:
                large_out, _ = layer(large)
            self.assertEqual(run.call_count, 1)
            self._assert_close(large_out, self._reference(layer, large))

    def test_bias_leading_dims_noncontiguous_and_cuda_graph(self):
        with torch.no_grad():
            layer = self._make_layer()
            x = torch.randn((2, 3, 2560), device="cuda", dtype=torch.bfloat16)[..., ::2]
            bias = torch.randn(576, device="cuda", dtype=torch.bfloat16)

            def fn():
                return layer.quant_method.apply(layer, x, bias)

            self._assert_close(fn(), self._reference(layer, x) + bias.float())
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fn()
            graph.replay()
            torch.cuda.synchronize()
            self._assert_close(out, self._reference(layer, x) + bias.float())

    def test_short_k_and_direct_reader_fallback(self):
        for k, skip in ((288, False), (512, False), (1280, True)):
            with self.subTest(k=k, skip=skip), torch.no_grad():
                layer = self._make_layer(k=k, skip=skip)
                self.assertFalse(layer.humming_fp8_ready)
                self.assertFalse(hasattr(layer, "_humming_fp8_weight"))
                x = torch.randn((6, k), device="cuda", dtype=torch.bfloat16)
                with mock.patch.object(
                    fp8, "humming_fp8_linear", side_effect=AssertionError
                ):
                    out, _ = layer(x)
                expected = fp8_utils.triton_w8a8_block_fp8_linear(
                    x,
                    layer.weight,
                    [32, 32],
                    layer.weight_scale_inv,
                    act_scale_ue8m0=True,
                )
                torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_large_m_prequantized_and_deterministic_fallback(self):
        with torch.no_grad():
            layer = self._make_layer()
            x = torch.randn((6, 1280), device="cuda", dtype=torch.bfloat16)
            qx, sx = fp8_utils.sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
            large = torch.randn((65, 1280), device="cuda", dtype=torch.bfloat16)
            for value in (large, (qx, sx)):
                with mock.patch.object(
                    fp8, "humming_fp8_linear", side_effect=AssertionError
                ):
                    out = layer.quant_method.apply(layer, value)
                kwargs = (
                    {"input": value[0], "input_scale": value[1]}
                    if isinstance(value, tuple)
                    else {"input": value}
                )
                expected = fp8_utils.triton_w8a8_block_fp8_linear(
                    weight=layer.weight,
                    block_size=[32, 32],
                    weight_scale=layer.weight_scale_inv,
                    act_scale_ue8m0=True,
                    **kwargs,
                )
                torch.testing.assert_close(out, expected, rtol=0, atol=0)

            with (
                mock.patch.object(humming_fp8, "_HUMMING_FP8_MAX_M", 65),
                mock.patch.object(
                    fp8, "humming_fp8_linear", wraps=humming_fp8.humming_fp8_linear
                ) as run,
            ):
                out = layer.quant_method.apply(layer, large)
            self.assertEqual(run.call_count, 1)
            self._assert_close(out, self._reference(layer, large))

            deterministic = SimpleNamespace(
                deterministic=SimpleNamespace(enable_deterministic_inference=True)
            )
            for patcher in (
                mock.patch.object(humming_fp8, "get_exec", return_value=deterministic),
                mock.patch.object(
                    humming_fp8, "is_batch_invariant_mode_enabled", return_value=True
                ),
            ):
                with (
                    patcher,
                    mock.patch.object(
                        fp8, "humming_fp8_linear", side_effect=AssertionError
                    ),
                ):
                    out, _ = layer(x)
                    unloaded = self._make_layer()
                    self.assertFalse(unloaded.humming_fp8_ready)
                expected = fp8_utils.triton_w8a8_block_fp8_linear(
                    x,
                    layer.weight,
                    [32, 32],
                    layer.weight_scale_inv,
                    act_scale_ue8m0=True,
                )
                torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_reload_refreshes_cache_without_rebinding_graph_storage(self):
        with torch.no_grad():
            layer = self._make_layer()
            names = (
                "_humming_fp8_weight",
                "_humming_fp8_weight_scale",
                "_humming_fp8_locks",
            )
            pointers = [getattr(layer, name).data_ptr() for name in names]
            x = torch.randn((6, 1280), device="cuda", dtype=torch.bfloat16)
            layer(x)  # warm up before capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out, _ = layer(x)
            # Reload both weights and scales after capturing the first layout.
            layer.weight.copy_((-layer.weight.float()).to(torch.float8_e4m3fn))
            layer.weight_scale_inv.mul_(2)
            layer.quant_method.process_weights_after_loading(layer)
            self.assertEqual(
                pointers, [getattr(layer, name).data_ptr() for name in names]
            )
            graph.replay()
            torch.cuda.synchronize()
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
