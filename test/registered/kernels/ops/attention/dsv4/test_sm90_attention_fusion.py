"""SM90 attention fusions against the unfused operators, including graph replay."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def c2_reference(kv, score, pos, raw, out_loc, req, state, weight, eps, ring):
    """Torch pooling oracle. Snapshot partners before updating any ring slots."""
    n, d = kv.shape
    read = req * ring + (pos - 1) % ring
    read = read.masked_fill((pos == 0) | (raw == 0), state.shape[0] - 1)
    carried = state[read]
    in_batch = torch.zeros_like(raw, dtype=torch.bool)
    in_batch[1:] = (
        (req[1:] == req[:-1])
        & (pos[1:] == pos[:-1] + 1)
        & (raw[1:] != 0)
        & (raw[:-1] != 0)
    )
    pk = torch.where(in_batch[:, None], kv.roll(1, 0), carried[:, :d])
    ps = torch.where(in_batch[:, None], score.roll(1, 0), carried[:, d:])
    pooled = (
        (torch.stack((pk, kv), 1) * torch.stack((ps, score), 1).softmax(1))
        .sum(1)
        .to(torch.bfloat16)
    )
    # Use the production unfused RMSNorm, so reduction/rounding drift is visible.
    from sglang.kernels.ops.layernorm.rmsnorm_fp32 import rmsnorm_fp32

    latent = rmsnorm_fp32(pooled, weight, eps)
    keep = raw != 0
    if n > ring:
        keep[:-ring] &= (req[:-ring] != req[ring:]) | (raw[ring:] == 0)
    rows = (req * ring + pos % ring).masked_fill(~keep, -1)
    state[rows] = torch.cat((kv, score), -1)
    state[-1].zero_()
    return latent, pos - pos % 2, out_loc.clamp_min(0)


def c2_inputs(batch, draft, ring, *, padded=1, strided=False, device="cuda"):
    d = 512
    live = batch - padded
    # Reordered request IDs include live request zero, also used by padding.
    ids = torch.arange(live, device=device).flip(0)
    req = torch.cat((ids, torch.zeros(padded, device=device, dtype=torch.int64)))
    req = req.repeat_interleave(draft)
    prefixes = torch.tensor([0, 1, 7, 8191, 16384, 32767, 131071], device=device)
    starts = prefixes[torch.arange(batch, device=device) % prefixes.numel()]
    pos = (starts[:, None] + torch.arange(draft, device=device)).flatten()
    raw = torch.arange(1, batch * draft + 1, device=device, dtype=torch.int64)
    if padded:
        raw[-padded * draft :] = 0
        pos[-padded * draft :] = 0
    loc = torch.where(pos % 2 == 1, raw // 2 + 1, -1).masked_fill(raw == 0, 0)
    projection = torch.randn(batch * draft, 2 * d, device=device, dtype=torch.float32)
    kv, score = projection[:, :d], projection[:, d:]
    if not strided:
        kv, score = kv.contiguous(), score.contiguous()
    state = torch.randn(batch * ring + 1, 2 * d, device=device)
    state[-1].zero_()
    weight = torch.randn(d, device=device, dtype=torch.float32)
    return kv, score, pos, raw, loc, req, state, weight


@unittest.skipUnless(torch.cuda.is_available() and torch.version.cuda, "CUDA required")
class TestSM90AttentionFusion(CustomTestCase):
    def test_empty_and_all_padding(self):
        from sglang.kernels.ops.attention.dsv4.c2_verify_pool import c2_verify_pool_norm
        from sglang.kernels.ops.layernorm.rmsnorm_group_fp8 import rmsnorm_group_fp8

        kv, score, pos, raw, loc, req, state, w = c2_inputs(4, 6, 8, padded=4)
        saved = state.clone()
        for n in (0, 24):
            y, gp, slots = c2_verify_pool_norm(
                kv[:n],
                score[:n],
                pos[:n],
                raw[:n],
                loc[:n],
                req[:n],
                state[:, :512],
                state[:, 512:],
                w,
                1e-6,
                ring_size=8,
            )
            self.assertEqual(y.shape, (n, 512))
            self.assertEqual(gp.numel(), n)
            torch.testing.assert_close(slots, torch.zeros_like(slots), rtol=0, atol=0)
            torch.testing.assert_close(state, saved, rtol=0, atol=0)
        x = torch.empty(0, 1280, device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(1280, device="cuda", dtype=torch.bfloat16)
        y, q, s = rmsnorm_group_fp8(
            x, weight, 1e-6, group_size=128, column_major_scales=True
        )
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(q.shape, x.shape)
        self.assertEqual(s.shape, (0, 10))

    def test_q_lora_deepgemm_contract(self):
        from sglang.srt.layers import deep_gemm_wrapper

        if (
            torch.cuda.get_device_capability()[0] != 9
            or not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        ):
            self.skipTest("Requires an available Hopper DeepGEMM installation")
        from sglang.kernels.ops.layernorm.rmsnorm_group_fp8 import rmsnorm_group_fp8
        from sglang.srt.layers.quantization.fp8_utils import (
            deepgemm_w8a8_block_fp8_linear_with_fallback as linear,
        )

        weight = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
        w = (torch.randn(256, 1280, device="cuda") * 10).to(torch.float8_e4m3fn)
        ws = torch.full((2, 10), 0.03125, device="cuda")
        for m in (5, 192, 193):
            x = torch.randn(m, 1280, device="cuda", dtype=torch.bfloat16)
            y, q, s = rmsnorm_group_fp8(
                x, weight, 1e-6, group_size=128, column_major_scales=True
            )
            actual = linear(q, w, [128, 128], ws, input_scale=s)
            expected = linear(y, w, [128, 128], ws)
            self.assertEqual(actual.shape, (m, 256))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_c2_pool_norm_and_ring(self):
        from sglang.kernels.ops.attention.dsv4.c2_verify_pool import c2_verify_pool_norm

        torch.manual_seed(17)
        for batch, draft, ring in ((2, 6, 2), (9, 6, 8), (33, 6, 8), (49, 10, 4)):
            for strided in (False, True):
                with self.subTest(batch=batch, draft=draft, ring=ring, strided=strided):
                    kv, score, pos, raw, loc, req, state, w = c2_inputs(
                        batch, draft, ring, strided=strided
                    )
                    if strided:
                        w = w.to(torch.bfloat16)
                    ref_state = state.clone()
                    # Replay a rejected suffix, then advance: stale state or early
                    # ring writes corrupt the first completing pair of a request.
                    for advance in (0, 2, 0, draft):
                        p = torch.where(raw != 0, pos + advance, pos)
                        expected = c2_reference(
                            kv, score, p, raw, loc, req, ref_state, w, 1e-6, ring
                        )
                        actual = c2_verify_pool_norm(
                            kv,
                            score,
                            p,
                            raw,
                            loc,
                            req,
                            state[:, :512],
                            state[:, 512:],
                            w,
                            1e-6,
                            ring_size=ring,
                        )
                        for got, want in zip(actual, expected):
                            torch.testing.assert_close(got, want, rtol=0, atol=0)
                        torch.testing.assert_close(state, ref_state, rtol=0, atol=0)

    def test_c2_graph_replay_with_live_metadata(self):
        from sglang.kernels.ops.attention.dsv4.c2_verify_pool import c2_verify_pool_norm

        kv, score, pos, raw, loc, req, state, w = c2_inputs(33, 6, 8)
        original = state.clone()

        def run():
            return c2_verify_pool_norm(
                kv,
                score,
                pos,
                raw,
                loc,
                req,
                state[:, :512],
                state[:, 512:],
                w,
                1e-6,
                ring_size=8,
            )

        run()  # Compile before capture.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run()
        for advance in (0, 1, 6):
            state.copy_(original)
            pos.add_(advance)
            kv.normal_()
            expected = c2_reference(
                kv, score, pos, raw, loc, req, original.clone(), w, 1e-6, 8
            )
            graph.replay()
            for got, want in zip(result, expected):
                torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_q_lora_norm_quant(self):
        from sglang.kernels.ops.layernorm.rmsnorm_group_fp8 import rmsnorm_group_fp8
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )
        from sglang.srt.layers.layernorm import RMSNorm

        torch.manual_seed(29)
        norm = RMSNorm(1280, eps=1e-6).cuda().to(torch.bfloat16)
        norm.weight.data.normal_()
        for m in (1, 5, 32, 48, 96, 144, 192, 240, 288, 384):
            # WQA/WKV fused projection produces this noncontiguous Q-LoRA view.
            x = torch.randn(m, 1792, device="cuda", dtype=torch.bfloat16)[:, :1280]
            x[0].zero_()
            for group, column, ue in ((128, True, False), (32, False, True)):
                with self.subTest(m=m, group=group):
                    y, q, s = rmsnorm_group_fp8(
                        x,
                        norm.weight,
                        1e-6,
                        group_size=group,
                        column_major_scales=column,
                        scale_ue8m0=ue,
                    )
                    ref_y = norm(x)
                    torch.testing.assert_close(y, ref_y, rtol=1e-2, atol=1e-3)
                    # Check quant against the emitted BF16, independently of
                    # any RMSNorm reduction-order differences.
                    ref_q, ref_s = sglang_per_token_group_quant_fp8(
                        y,
                        group,
                        column_major_scales=column,
                        scale_tma_aligned=column,
                        scale_ue8m0=ue,
                    )
                    torch.testing.assert_close(s, ref_s, rtol=0, atol=0)
                    torch.testing.assert_close(q.float(), ref_q.float(), rtol=0, atol=0)
                    self.assertEqual(s.stride(), ref_s.stride())

    def test_q_lora_graph_and_gemm(self):
        from sglang.kernels.ops.layernorm.rmsnorm_group_fp8 import rmsnorm_group_fp8
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )
        from sglang.srt.layers.quantization.fp8_utils import (
            triton_w8a8_block_fp8_linear,
        )

        x = torch.randn(192, 1280, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
        w = (torch.randn(256, 1280, device="cuda") * 10).to(torch.float8_e4m3fn)
        ws = torch.full((8, 40), 0.03125, device="cuda")

        def run():
            y, q, s = rmsnorm_group_fp8(
                x, weight, 1e-6, group_size=32, scale_ue8m0=True
            )
            out = triton_w8a8_block_fp8_linear(
                q, w, [32, 32], ws, s, act_scale_ue8m0=True
            )
            return y, out

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y, out = run()
        for _ in range(2):
            x.normal_()
            graph.replay()
            q_ref, s_ref = sglang_per_token_group_quant_fp8(y, 32, scale_ue8m0=True)
            ref = triton_w8a8_block_fp8_linear(
                q_ref, w, [32, 32], ws, s_ref, act_scale_ue8m0=True
            )
            torch.testing.assert_close(out, ref, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
