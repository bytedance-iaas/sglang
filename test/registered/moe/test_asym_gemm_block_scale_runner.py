"""Runner-level numerical tests for the unified AsymGEMM MoE runner on SM89/SM90.

Exercises the merged `_run_contiguous_gemm` / `_run_masked_gemm` paths with
native 1x128 activation + 128x128 weight block scales (no dequant-requant),
comparing the full FFN (gateup GEMM -> SiLU*mul -> down GEMM) against an
fp32 reference.

Run:
    SGLANG_ENABLE_JIT_ASYMGEMM=1 python test/registered/moe/test_asym_gemm_block_scale_runner.py
"""

import os
import unittest

os.environ.setdefault("SGLANG_ENABLE_JIT_ASYMGEMM", "1")

import torch

from sglang.srt.layers.moe.moe_runner import asym_gemm as asym_gemm_runner
from sglang.srt.layers.moe.moe_runner.asym_gemm import (
    AsymGemmMoeQuantInfo,
    AsymGemmRunnerCore,
    AsymGemmRunnerInput,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8


def calc_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    scale = b.abs().max().clamp(min=1e-6)
    return (((a - b) / scale).norm() / (a.numel() ** 0.5)).item()


def rand_fp8(shape):
    return (torch.rand(shape, device="cuda") * 2 - 1).to(torch.float8_e4m3fn)


def rand_scale(shape):
    return (torch.rand(shape, device="cuda") * 1.5 + 0.5).float()


def dequant_rows(x_fp8, sx):
    """[.., R, K] fp8 with [.., R, ceil(K/128)] scales -> fp32."""
    K = x_fp8.shape[-1]
    return x_fp8.float() * sx.repeat_interleave(128, dim=-1)[..., :K]

def dequant_w(w_fp8, sw):
    """[E, N, K] fp8 with [E, ceil(N/128), ceil(K/128)] scales -> fp32."""
    _, N, K = w_fp8.shape
    sw_full = sw.repeat_interleave(128, dim=1)[:, :N, :]
    sw_full = sw_full.repeat_interleave(128, dim=2)[:, :, :K]
    return w_fp8.float() * sw_full


def ref_ffn(x_deq, w13_deq, w2_deq):
    """fp32 reference: gateup -> silu(gate)*up -> down. x_deq [M, K]."""
    g = x_deq @ w13_deq.t()
    half = g.shape[-1] // 2
    act = torch.nn.functional.silu(g[..., :half]) * g[..., half:]
    return act @ w2_deq.t()


def make_quant_info(E, N, K):
    """w13 [E, N, K] (gateup), w2 [E, K, N//2] (down)."""
    w13 = rand_fp8((E, N, K))
    w13_scale = rand_scale((E, (N + 127) // 128, (K + 127) // 128))
    w2 = rand_fp8((E, K, N // 2))
    w2_scale = rand_scale((E, (K + 127) // 128, (N // 2 + 127) // 128))
    return AsymGemmMoeQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        use_fp8=True,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=[128, 128],
    )


def make_core():
    return AsymGemmRunnerCore(MoeRunnerConfig(activation="silu", is_gated=True))


class TestAsymGemmBlockScaleRunner(unittest.TestCase):
    TOL = 2e-2  # fp8 intermediate quant of the activation dominates

    def test_contiguous_runner(self):
        torch.manual_seed(0)
        E, N, K = 4, 512, 512
        token_counts = [40, 0, 130, 7]
        quant_info = make_quant_info(E, N, K)

        total = sum(token_counts)
        x_bf16 = torch.randn(total, K, device="cuda", dtype=torch.bfloat16)
        x_fp8, x_scale = sglang_per_token_group_quant_fp8(x_bf16, 128)

        ends, active = [], []
        acc = 0
        for e, cnt in enumerate(token_counts):
            acc += cnt
            if cnt:
                active.append(e)
                ends.append(acc)
        runner_input = AsymGemmRunnerInput(
            hidden_states=x_fp8.clone(),
            hidden_states_scale=x_scale.contiguous().clone(),
            use_masked_gemm=False,
            offsets=torch.tensor(ends, dtype=torch.int32, device="cuda"),
            experts=torch.tensor(active, dtype=torch.int32, device="cuda"),
            list_size=len(active),
        )
        running_state = {
            "all_tokens": total,
            "hidden_states_device": x_fp8.device,
            "hidden_states_dtype": torch.bfloat16,
            "hidden_states_shape": (total, K),
        }

        out = make_core()._run_contiguous_gemm(runner_input, quant_info, running_state)

        x_deq = dequant_rows(x_fp8, x_scale.float())
        w13_deq = dequant_w(quant_info.w13_weight, quant_info.w13_scale)
        w2_deq = dequant_w(quant_info.w2_weight, quant_info.w2_scale)
        start = 0
        for e, end in zip(active, ends):
            ref = ref_ffn(x_deq[start:end], w13_deq[e], w2_deq[e])
            diff = calc_diff(out[start:end].float(), ref)
            print(f"  contiguous expert {e} rows [{start}:{end}): diff={diff:.3e}")
            self.assertLess(diff, self.TOL)
            start = end

    def _masked_case(self, chunk_size):
        torch.manual_seed(1)
        G, M_max, N, K = 4, 256, 512, 512
        masked = torch.tensor([40, 0, 130, 256], dtype=torch.int32, device="cuda")
        quant_info = make_quant_info(G, N, K)

        x_bf16 = torch.randn(G, M_max, K, device="cuda", dtype=torch.bfloat16)
        x_fp8, x_scale = sglang_per_token_group_quant_fp8(x_bf16.view(-1, K), 128)
        x_fp8 = x_fp8.view(G, M_max, K)
        x_scale = x_scale.view(G, M_max, -1).contiguous()

        runner_input = AsymGemmRunnerInput(
            hidden_states=x_fp8.clone(),
            hidden_states_scale=x_scale.clone(),
            use_masked_gemm=True,
            masked_m=masked,
            expected_m=int(masked.max()),
        )
        running_state = {
            "hidden_states_device": x_fp8.device,
            "hidden_states_dtype": torch.bfloat16,
            "hidden_states_shape": (G * M_max, K),
        }

        old_chunk = asym_gemm_runner._MASKED_GEMM_CHUNK_SIZE
        asym_gemm_runner._MASKED_GEMM_CHUNK_SIZE = chunk_size
        try:
            out = make_core()._run_masked_gemm(runner_input, quant_info, running_state)
        finally:
            asym_gemm_runner._MASKED_GEMM_CHUNK_SIZE = old_chunk

        x_deq = dequant_rows(x_fp8, x_scale.float())
        w13_deq = dequant_w(quant_info.w13_weight, quant_info.w13_scale)
        w2_deq = dequant_w(quant_info.w2_weight, quant_info.w2_scale)
        for g in range(G):
            m = int(masked[g])
            if m == 0:
                continue
            ref = ref_ffn(x_deq[g, :m], w13_deq[g], w2_deq[g])
            diff = calc_diff(out[g, :m].float(), ref)
            print(f"  masked(chunk={chunk_size}) group {g} m={m}: diff={diff:.3e}")
            self.assertLess(diff, self.TOL)

    def test_masked_runner_nonchunked(self):
        self._masked_case(chunk_size=0)

    def test_masked_runner_chunked(self):
        self._masked_case(chunk_size=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
