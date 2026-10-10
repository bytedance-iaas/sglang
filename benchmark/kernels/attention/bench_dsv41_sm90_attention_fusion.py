"""Microbench the two SM90 attention fusions, without model weights.

Run from a source checkout. Numbers exclude GEMMs, attention, communication and
KV stores and must not be interpreted as serving TPOT. CUDA Graph timing uses
the same per-rank row counts as DP attention with static DSpark verification.
"""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.attention.dsv4.c2_verify_pool import c2_verify_pool_norm
from sglang.kernels.ops.layernorm.rmsnorm_group_fp8 import rmsnorm_group_fp8
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.layernorm import RMSNorm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches", type=int, nargs="+", default=[64, 128, 192, 256, 320, 384]
    )
    parser.add_argument("--dp", type=int, default=8)
    parser.add_argument("--verify-tokens", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9):
        parser.error("This experiment targets SM90 CUDA GPUs")
    if (
        args.dp <= 0
        or args.verify_tokens < 2
        or args.repeats <= 0
        or any(b <= 0 or b % args.dp for b in args.batches)
    ):
        parser.error(
            "Use positive batches divisible by DP, verify-tokens >= 2 and repeats > 0"
        )

    # Share the unfused oracle and input geometry with the correctness tests.
    test_file = (
        Path(__file__).resolve().parents[3]
        / "test/registered/kernels/ops/attention/dsv4/test_sm90_attention_fusion.py"
    )
    spec = importlib.util.spec_from_file_location("sm90_fusion_reference", test_file)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(17)
    norm = RMSNorm(1280, eps=1e-6).cuda().to(torch.bfloat16)
    norm.weight.data.normal_()
    records = []

    def measure(name, batch, rows, shape, baseline, fused):
        baseline()
        fused()
        torch.cuda.synchronize()
        samples = {"baseline_us": [], "fused_us": []}
        for i in range(args.repeats):
            order = [("baseline_us", baseline), ("fused_us", fused)]
            for key, fn in order if i % 2 == 0 else reversed(order):
                samples[key].append(
                    1000 * triton.testing.do_bench_cudagraph(fn, rep=100)
                )
        medians = {key: statistics.median(value) for key, value in samples.items()}
        record = dict(
            operation=name,
            gpu=torch.cuda.get_device_name(),
            total_batch=batch,
            dp=args.dp,
            rows_per_rank=rows,
            shape=shape,
            timing="CUDA graph, median microseconds",
            **medians,
            speedup=medians["baseline_us"] / medians["fused_us"],
            samples=samples,
        )
        records.append(record)
        print(json.dumps(record), flush=True)

    with torch.inference_mode():
        for batch in args.batches:
            requests = batch // args.dp
            rows = requests * args.verify_tokens
            # The real ring reserves enough history for speculative rollback.
            ring = ((args.verify_tokens + 3) // 2) * 2
            kv, score, pos, raw, loc, req, state, w = reference.c2_inputs(
                requests, args.verify_tokens, ring, padded=0
            )
            baseline_state = state.clone()

            def baseline_c2():
                return reference.c2_reference(
                    kv, score, pos, raw, loc, req, baseline_state, w, 1e-6, ring
                )

            def fused_c2():
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
                    ring_size=ring,
                )

            measure(
                "c2_verify_pool_norm_fp32_to_bf16",
                batch,
                rows,
                [rows, 512],
                baseline_c2,
                fused_c2,
            )
            for phase, m in (("draft", requests), ("verify", rows)):
                x = torch.randn(m, 1792, device="cuda", dtype=torch.bfloat16)[:, :1280]
                for group, column, ue in ((128, True, False), (32, False, True)):

                    def baseline_q():
                        y = norm(x)
                        q, scale = sglang_per_token_group_quant_fp8(
                            y,
                            group,
                            column_major_scales=column,
                            scale_tma_aligned=column,
                            scale_ue8m0=ue,
                        )
                        return y, q, scale

                    def fused_q():
                        return rmsnorm_group_fp8(
                            x,
                            norm.weight,
                            1e-6,
                            group_size=group,
                            column_major_scales=column,
                            scale_ue8m0=ue,
                        )

                    measure(
                        f"q_lora_{phase}_bf16_fp8_group{group}",
                        batch,
                        m,
                        [m, 1280],
                        baseline_q,
                        fused_q,
                    )
    if args.output:
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
