# ruff: noqa
"""Standalone validation + microbench for the dispatch-level Q8KV8 split-topk.

Validates ``sparse_mla_q8kv8_prefill_fwd_split2`` (two half-topk kernel calls
+ Triton LSE merge, CUDA kernel untouched) against:

  Gate A: the single full-topk kernel call on identical inputs
          (atol/rtol 2e-2 on the fp32 view of the bf16 out; the merge changes
          accumulation order so bitwise equality is not expected).
  Gate B: a masked fp32 reference (adapted from the kernel unit test's
          ``_torch_sparse_attention_ref`` / ``_ref_masked_blocked``), same
          tolerance as the kernel's own test (atol/rtol 8e-2); the single
          call is checked against the same reference for context.

Perf: CUDA-event microbench at the GLM-5.2 3.5k-il production shape
(s_q=437, topk=2048, s_kv=65536 + 2048 appended zero pad rows, realistic
causal -1 tails) plus s_q=4096 (where the split must NOT be used).

Run (GPU 0):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python \
      test/manual/attention/test_q8kv8_split_topk_dispatch.py
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

from sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90 import (
    sparse_mla_q8kv8_prefill_fwd,
)
from sglang.srt.layers.attention.dsa.split_topk import (
    Split2Buffers,
    combine_split2,
    sparse_mla_q8kv8_prefill_fwd_split2,
)

DTYPE_FP8 = torch.float8_e4m3fn
D_QK = 576
D_V = 512
H_Q = 64  # GLM-5.2 native tile (64 heads = one CTA per query row)
H_KV = 1
INF = float("inf")


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------


def make_case(s_q: int, topk: int, num_kv_tokens: int, valid_counts, seed: int = 0):
    """Production-layout case: kv has ``topk`` trailing ZERO pad rows appended
    (the -1 clamp band), indices are random valid entries with trailing -1
    runs per ``valid_counts`` (len s_q, values in [0, topk])."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((s_q, H_Q, D_QK), device="cuda", generator=g).to(DTYPE_FP8)
    s_kv = num_kv_tokens + topk
    kv = torch.zeros((s_kv, H_KV, D_QK), dtype=DTYPE_FP8, device="cuda")
    kv[:num_kv_tokens] = torch.randn(
        (num_kv_tokens, H_KV, D_QK), device="cuda", generator=g
    ).to(DTYPE_FP8)

    idx = torch.randint(
        0,
        num_kv_tokens,
        (s_q, H_KV, topk),
        dtype=torch.int32,
        device="cuda",
        generator=g,
    )
    valid = torch.as_tensor(valid_counts, dtype=torch.int64, device="cuda")
    assert valid.shape == (s_q,)
    slot = torch.arange(topk, device="cuda")
    idx[:, 0, :] = torch.where(
        slot[None, :] < valid[:, None], idx[:, 0, :], torch.full_like(idx[:, 0, :], -1)
    )

    one = torch.ones(1, dtype=torch.float32, device="cuda")
    sm_scale = 1.0 / math.sqrt(D_QK)
    return q, kv, idx, sm_scale, one


def production_valid_counts(s_q: int, topk: int, il: int):
    """Causal tails at input length ``il``: row i sits at position
    ~ i * il / s_q, valid = min(pos + 1, topk)."""
    pos = (torch.arange(s_q, dtype=torch.float64) * (il / s_q)).long()
    return torch.clamp(pos + 1, max=topk).tolist()


def mixed_valid_counts(s_q: int, topk: int):
    """Corner-heavy mix: full rows, all-pad, single-valid, exactly half,
    half +/- 1, and everything in between."""
    half = topk // 2
    pattern = [topk, 1, 100, 517, half - 1, half, half + 1, 1500, topk - 1, 0]
    return [pattern[i % len(pattern)] for i in range(s_q)]


def backscan_topk_length(idx: torch.Tensor) -> torch.Tensor:
    """Reference backscan (matches q8kv8_topk_length_from_indices):
    last non-negative position + 1, clamped to min 1."""
    s_q, _, topk = idx.shape
    ramp = torch.arange(1, topk + 1, dtype=torch.int32, device=idx.device)
    return ((idx[:, 0, :] >= 0).int() * ramp).amax(dim=-1).clamp_(min=1)


# ---------------------------------------------------------------------------
# fp32 masked reference (blocked; adapted from the kernel unit test)
# ---------------------------------------------------------------------------


def ref_masked_blocked(q, kv, idx, sm_scale, row0, row1):
    """Returns (out, max_logits, lse) fp32 for rows [row0, row1).
    -1 slots are masked to -inf; all-pad rows yield (0, -inf, +inf) to
    mirror the kernel's empty-row epilogue."""
    q_f = q.float()[row0:row1]
    kv_f = kv.float()[:, 0, :]
    ib = idx[row0:row1, 0, :].long()
    gathered = kv_f[ib.clamp(min=0)]  # [r, k, d]
    scores = torch.einsum("rhd,rkd->rhk", q_f, gathered) * sm_scale
    scores = scores.masked_fill((ib < 0)[:, None, :], -INF)
    m = scores.amax(dim=-1)  # [r, h]; -inf for all-pad rows
    row_has_valid = (ib >= 0).any(dim=-1)  # [r]
    m_safe = torch.where(m == -INF, torch.zeros_like(m), m)
    e = torch.exp(scores - m_safe[..., None])
    denom = e.sum(dim=-1)  # [r, h]; 0 for all-pad rows
    lse = torch.where(
        row_has_valid[:, None], m_safe + torch.log(denom.clamp(min=1e-30)), INF
    )
    ml = torch.where(row_has_valid[:, None], m, -INF)
    out = torch.einsum(
        "rhk,rkd->rhd", e / denom.clamp(min=1e-30)[..., None], gathered[:, :, :D_V]
    )
    out = torch.where(row_has_valid[:, None, None], out, torch.zeros_like(out))
    return out, ml, lse


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def run_single(q, kv, idx, sm_scale, one, topk_length):
    return sparse_mla_q8kv8_prefill_fwd(
        q=q,
        kv=kv,
        indices=idx,
        sm_scale=sm_scale,
        q_scale=one,
        kv_scale=one,
        d_v=D_V,
        attn_sink=None,
        topk_length=topk_length,
    )


def gate_a_case(name, q, kv, idx, sm_scale, one, valid_counts, use_tl, overlap):
    s_q, _, topk = idx.shape
    half = topk // 2
    tl = backscan_topk_length(idx) if use_tl else None

    out_s, ml_s, lse_s = run_single(q, kv, idx, sm_scale, one, tl)

    bufs = Split2Buffers.allocate(s_q, H_Q, H_KV, half, D_V, q.device, use_tl)
    out_2, ml_2, lse_2 = sparse_mla_q8kv8_prefill_fwd_split2(
        q,
        kv,
        idx,
        sm_scale,
        one,
        one,
        D_V,
        topk_length=tl_arg,
        out_bufs=bufs,
        overlap=overlap,
    )
    torch.cuda.synchronize()

    # Empty-half semantics actually exercised: rows with <= half valid slots
    # must mark half2 empty (lse == +inf, ml == -inf) in ALL heads.
    valid_t = torch.as_tensor(valid_counts, device="cuda")
    empty2 = valid_t <= half
    if empty2.any():
        assert (bufs.lse_halves[1, :s_q][empty2] == INF).all(), "half2 lse marker"
        assert (bufs.ml_halves[1, :s_q][empty2] == -INF).all(), "half2 ml marker"
    all_pad = valid_t == 0
    if all_pad.any():
        assert (out_2[all_pad].float() == 0).all(), "all-pad out"
        assert (lse_2[all_pad] == INF).all(), "all-pad lse"
        assert (ml_2[all_pad] == -INF).all(), "all-pad ml"

    torch.testing.assert_close(out_2.float(), out_s.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(ml_2, ml_s, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse_2, lse_s, atol=5e-3, rtol=5e-3)

    fin = torch.isfinite(lse_s)
    d_out = (out_2.float() - out_s.float()).abs().max().item()
    d_lse = (lse_2[fin] - lse_s[fin]).abs().max().item() if fin.any() else 0.0
    print(
        f"  [gate A PASS] {name} tl={'backscan' if use_tl else 'None':8s} "
        f"overlap={overlap!s:5s}  max|dout|={d_out:.3e} max|dlse|={d_lse:.3e}"
    )


def gate_b_case(name, q, kv, idx, sm_scale, one, use_tl, block=64):
    s_q, _, topk = idx.shape
    half = topk // 2
    tl = backscan_topk_length(idx) if use_tl else None

    out_s, _, lse_s = run_single(q, kv, idx, sm_scale, one, tl)
    bufs = Split2Buffers.allocate(s_q, H_Q, H_KV, half, D_V, q.device, use_tl)
    out_2, _, lse_2 = sparse_mla_q8kv8_prefill_fwd_split2(
        q, kv, idx, sm_scale, one, one, D_V, topk_length=tl_arg, out_bufs=bufs
    )
    torch.cuda.synchronize()

    worst_split = worst_single = 0.0
    worst_lse = 0.0
    for r0 in range(0, s_q, block):
        r1 = min(r0 + block, s_q)
        ref, _, ref_lse = ref_masked_blocked(q, kv, idx, sm_scale, r0, r1)
        torch.testing.assert_close(out_2[r0:r1].float(), ref, atol=8e-2, rtol=8e-2)
        torch.testing.assert_close(out_s[r0:r1].float(), ref, atol=8e-2, rtol=8e-2)
        fin = torch.isfinite(ref_lse)
        if fin.any():
            torch.testing.assert_close(
                lse_2[r0:r1][fin], ref_lse[fin], atol=5e-3, rtol=5e-3
            )
            worst_lse = max(
                worst_lse, (lse_2[r0:r1][fin] - ref_lse[fin]).abs().max().item()
            )
        worst_split = max(worst_split, (out_2[r0:r1].float() - ref).abs().max().item())
        worst_single = max(
            worst_single, (out_s[r0:r1].float() - ref).abs().max().item()
        )
        del ref
        torch.cuda.empty_cache()
    print(
        f"  [gate B PASS] {name} tl={'backscan' if use_tl else 'None':8s}  "
        f"max|split-ref|={worst_split:.3e} max|single-ref|={worst_single:.3e} "
        f"max|lse-ref|={worst_lse:.3e}"
    )


def run_gates():
    print("=== Correctness gates ===")
    # Shape 1: production 3.5k-il shape, corner-heavy valid mix.
    s_q, topk, nkv = 437, 2048, 65536
    vc = mixed_valid_counts(s_q, topk)
    q, kv, idx, sm, one = make_case(s_q, topk, nkv, vc, seed=101)
    for use_tl in (False, True):
        for overlap in (True, False):
            gate_a_case(
                f"s_q={s_q} s_kv={nkv}+{topk}", q, kv, idx, sm, one, vc, use_tl, overlap
            )
    for use_tl in (False, True):
        gate_b_case(f"s_q={s_q} s_kv={nkv}+{topk}", q, kv, idx, sm, one, use_tl)
    del q, kv, idx
    torch.cuda.empty_cache()

    # Shape 2: tiny s_q, small kv; valid counts straddle the half boundary.
    s_q, topk, nkv = 8, 2048, 4096
    vc = [2048, 1, 1023, 1024, 1025, 517, 0, 2047]
    q, kv, idx, sm, one = make_case(s_q, topk, nkv, vc, seed=202)
    for use_tl in (False, True):
        for overlap in (True, False):
            gate_a_case(
                f"s_q={s_q} s_kv={nkv}+{topk}", q, kv, idx, sm, one, vc, use_tl, overlap
            )
    for use_tl in (False, True):
        gate_b_case(f"s_q={s_q} s_kv={nkv}+{topk}", q, kv, idx, sm, one, use_tl)
    del q, kv, idx
    torch.cuda.empty_cache()
    print("All correctness gates PASSED.\n")


# ---------------------------------------------------------------------------
# Perf microbench
# ---------------------------------------------------------------------------


def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000.0 / iters  # us


def run_perf(iters: int):
    print("=== Perf microbench (100-iter CUDA events, us per call) ===")

    for s_q, il_tag, vc in (
        (437, "il=3.5k", production_valid_counts(437, 2048, 3500)),
        (4096, "long-ctx", [2048] * 4096),
    ):
        topk, nkv = 2048, 65536
        half = topk // 2
        q, kv, idx, sm, one = make_case(s_q, topk, nkv, vc, seed=303)
        tl_bs = backscan_topk_length(idx)
        n_tail = sum(1 for v in vc if v < topk)
        print(
            f"\n-- s_q={s_q} ({il_tag}), topk={topk}, s_kv={nkv}+{topk} pad rows, "
            f"h={H_Q}, rows with -1 tails: {n_tail}/{s_q} --"
        )

        # Pre-allocated buffers for both arms.
        out_s = torch.empty(s_q, H_Q, D_V, dtype=torch.bfloat16, device="cuda")
        ml_s = torch.empty(s_q, H_Q, dtype=torch.float32, device="cuda")
        lse_s = torch.empty(s_q, H_Q, dtype=torch.float32, device="cuda")
        bufs = Split2Buffers.allocate(s_q, H_Q, H_KV, half, D_V, q.device, True)

        for tl_name, tl_arg in (("None", None), ("backscan", tl_bs)):
            t_single = bench(
                lambda: sparse_mla_q8kv8_prefill_fwd(
                    q=q,
                    kv=kv,
                    indices=idx,
                    sm_scale=sm,
                    q_scale=one,
                    kv_scale=one,
                    d_v=D_V,
                    attn_sink=None,
                    topk_length=tl_arg,
                    out=out_s,
                    max_logits=ml_s,
                    lse=lse_s,
                ),
                iters,
            )
            t_ovl = bench(
                lambda: sparse_mla_q8kv8_prefill_fwd_split2(
                    q,
                    kv,
                    idx,
                    sm,
                    one,
                    one,
                    D_V,
                    topk_length=tl_arg,
                    out_bufs=bufs,
                    overlap=True,
                ),
                iters,
            )
            t_seq = bench(
                lambda: sparse_mla_q8kv8_prefill_fwd_split2(
                    q,
                    kv,
                    idx,
                    sm,
                    one,
                    one,
                    D_V,
                    topk_length=tl_arg,
                    out_bufs=bufs,
                    overlap=False,
                ),
                iters,
            )
            gain = (t_single - t_ovl) / t_single * 100.0
            print(
                f"  topk_length={tl_name:8s}  single={t_single:8.1f}  "
                f"split2(overlap)={t_ovl:8.1f} ({gain:+5.1f}% vs single; "
                f"{'split2 FASTER' if gain > 0 else 'single FASTER'})  "
                f"split2(seq)={t_seq:8.1f}"
            )

        # Component costs (production shape only).
        s = s_q
        t_combine = bench(
            lambda: combine_split2(
                bufs.o_halves[0, :s],
                bufs.ml_halves[0, :s],
                bufs.lse_halves[0, :s],
                bufs.o_halves[1, :s],
                bufs.ml_halves[1, :s],
                bufs.lse_halves[1, :s],
                bufs.out[:s],
                bufs.ml_out[:s],
                bufs.lse_out[:s],
            ),
            iters,
        )
        idx_view = idx.view(s_q, H_KV, 2, half).permute(2, 0, 1, 3)
        t_idxcopy = bench(lambda: bufs.idx_halves[:, :s].copy_(idx_view), iters)
        print(
            f"  components: combine={t_combine:.1f} us, idx-split copy={t_idxcopy:.1f} us"
        )

        del q, kv, idx, bufs, out_s, ml_s, lse_s
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-gates", action="store_true")
    p.add_argument("--skip-perf", action="store_true")
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    torch.cuda.init()
    dev = torch.cuda.get_device_properties(0)
    print(f"Device: {dev.name}, {dev.multi_processor_count} SMs\n")

    if not args.skip_gates:
        run_gates()
    if not args.skip_perf:
        run_perf(args.iters)


if __name__ == "__main__":
    sys.exit(main())
