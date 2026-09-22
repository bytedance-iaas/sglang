# SPDX-License-Identifier: Apache-2.0
"""Compare the K=576 Triton W8A8 path with K=1024-padded Humming."""

import argparse

import torch
import torch.nn.functional as F
import triton
from humming.forward import humming_forward
from humming.layer import HummingLayer

from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import (
    triton_w8a8_block_fp8_linear,
)


def bench(fn) -> tuple[float, float, float]:
    p20, p50, p80 = triton.testing.do_bench(
        fn, warmup=200, rep=500, quantiles=[0.2, 0.5, 0.8]
    )
    return p20, p50, p80


def fmt(times: tuple[float, float, float]) -> str:
    return f"{times[1] * 1000:9.2f} us [{times[0] * 1000:.2f}, {times[2] * 1000:.2f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, nargs="+", default=[256, 8000, 8192])
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    n, k, padded_k = 5120, 576, 1024

    weight = torch.randn((n, k), device=device).to(torch.float8_e4m3fn)
    weight_scale = torch.ones((n // 32, k // 32), device=device)
    weight_padded = F.pad(weight, (0, padded_k - k))
    weight_scale_padded = F.pad(weight_scale, (0, padded_k // 32 - k // 32), value=1.0)
    with torch.device(device):
        packed = HummingLayer(
            shape_n=n,
            shape_k=padded_k,
            weight_config={"quant_method": "fp8", "weight_block_size": [32, 32]},
            input_config={
                "dtype": "float8e4m3",
                "group_size": 32,
                "scale_dtype": "float32",
            },
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            torch_dtype=torch.bfloat16,
        )
    packed.load_from_tensors(
        {"weight": weight_padded, "weight_scale_inv": weight_scale_padded}
    )
    packed.transform()
    print(
        f"weight={tuple(weight.shape)}, padded_config="
        f"(N={packed.humming_config.shape_n}, K={packed.humming_config.shape_k})"
    )

    for m in args.m:
        x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
        x_padded = F.pad(x, (0, padded_k - k))
        q576, s576 = sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
        q1024, s1024 = sglang_per_token_group_quant_fp8(x_padded, 32, scale_ue8m0=True)

        def triton_full():
            return triton_w8a8_block_fp8_linear(
                x, weight, [32, 32], weight_scale, act_scale_ue8m0=True
            )

        def triton_gemm():
            return triton_w8a8_block_fp8_linear(
                q576,
                weight,
                [32, 32],
                weight_scale,
                input_scale=s576,
                act_scale_ue8m0=True,
            )

        def pad_bf16():
            return F.pad(x, (0, padded_k - k))

        def quant_576():
            return sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)

        def quant_1024():
            return sglang_per_token_group_quant_fp8(x_padded, 32, scale_ue8m0=True)

        def humming_gemm():
            return humming_forward(
                packed.humming_config,
                inputs=q1024,
                input_scale=s1024,
                weight=packed.weight,
                weight_scale=packed.weight_scale,
                locks=packed.locks,
            )

        def humming_full():
            padded = F.pad(x, (0, padded_k - k))
            q_input, input_scale = sglang_per_token_group_quant_fp8(
                padded, 32, scale_ue8m0=True
            )
            return humming_forward(
                packed.humming_config,
                inputs=q_input,
                input_scale=input_scale,
                weight=packed.weight,
                weight_scale=packed.weight_scale,
                locks=packed.locks,
            )

        reference = triton_gemm()
        candidate = humming_gemm()
        torch.cuda.synchronize()
        diff = (reference.float() - candidate.float()).abs()
        print(
            f"\nM={m}: max_abs={diff.max().item():.6f}, mean_abs={diff.mean().item():.6f}"
        )
        for name, fn in (
            ("triton full", triton_full),
            ("triton gemm", triton_gemm),
            ("quant K=576", quant_576),
            ("pad bf16", pad_bf16),
            ("quant K=1024", quant_1024),
            ("humming gemm", humming_gemm),
            ("humming full", humming_full),
        ):
            print(f"  {name:16s} {fmt(bench(fn))}")


if __name__ == "__main__":
    main()
