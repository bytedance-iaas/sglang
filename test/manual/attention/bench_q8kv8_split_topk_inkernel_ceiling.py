#!/usr/bin/env python3
"""In-kernel split-topk GROSS-CEILING probe for the Q8KV8 sparse prefill
kernel -- the measurement that closed the question (NEGATIVE verdict,
2026-07-14, H200).

The in-kernel split (any merge variant: (a) global fp32 partials, (b)
atomic single-pass, or (c) SM90 cluster/DSMEM) runs 2*s_q half-topk CTAs.
Its compute phase is EXACTLY reproducible today with zero kernel changes:
one kernel launch at s_q'=2*s_q, topk'=topk/2, with interleaved-block index
halves (balanced per-row lengths under backscan -- the best split policy)
and q rows duplicated.  This launch has PERFECT scheduling (no cluster
pairing constraint) and NO merge, so:

    t_doubled    = strict LOWER bound on any in-kernel split's kernel time
    ceiling_gain = t_single - t_doubled = the most ANY merge design can win

VERDICT (H200 132 SMs, 100-iter CUDA events, reproduced twice):
    prod tail s_q=437 topk=2048 WITH backscan: single 174.4/174.5 us vs
    doubled 175.4/175.6 us -> ceiling gain -1.1 us (-0.6%).  The split's
    compute phase alone already loses, and every merge design adds strictly
    positive cost on top ((a)/(b): +57..115 MB DRAM = +19..38 us; (c)
    DSMEM: ~+2-5 us/pair staging + fabric + sync, plus cluster
    pair-scheduling fragmentation).  Without backscan the pure wave win is
    only +9.2 us (+3.9%), still below the >=8% acceptance bar before any
    merge cost.  s_q=1024: -6%; s_q=4096: -13.5% (per-CTA fixed-cost
    doubling dominates once waves are full).  Root cause: the accepted
    topk_length backscan already self-balances the small-s_q tail (short
    rows free their SMs for backfill), while splitting doubles every
    per-CTA fixed cost (36 KB Q load, prologue, softmax bookkeeping,
    epilogue) -- the wave-quantization prize this lever targeted no longer
    exists on top of backscan.

Also validates the interleaved-split semantics end-to-end: LSE-merging the
two halves (combine_split2) lands within ~2e-2 of the single call (worst
0.0215 at s_q=1024, fp8 noise at the tolerance boundary).
"""

import sys

import torch

sys.path.insert(0, "/data07/jackc/sglang_q8kv8_pr3_glm52/test/manual/attention")
from test_q8kv8_split_topk_dispatch import (
    D_V,
    H_KV,
    H_Q,
    backscan_topk_length,
    bench,
    make_case,
    production_valid_counts,
)

from sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90 import sparse_mla_q8kv8_prefill_fwd
from sglang.srt.layers.attention.dsa.split_topk import combine_split2

BLOCK = 64


def interleaved_halves(idx: torch.Tensor, topk: int):
    """[s_q, 1, topk] -> [2*s_q, 1, topk/2]: pair rows (2r, 2r+1) take the
    even / odd 64-token blocks of row r (balanced lengths for any backscan L)."""
    s_q = idx.shape[0]
    nb = topk // BLOCK
    v = idx.view(s_q, H_KV, nb // 2, 2, BLOCK)  # [s, 1, nb/2, parity, 64]
    # [s, parity, 1, nb/2*64] -> [2s, 1, topk/2]
    h = v.permute(0, 3, 1, 2, 4).reshape(s_q * 2, H_KV, (nb // 2) * BLOCK)
    return h.contiguous()


def run_shape(s_q: int, vc, name: str, iters: int = 100):
    topk, nkv = 2048, 65536
    q, kv, idx, sm, one = make_case(s_q, topk, nkv, vc, seed=303)
    tl = backscan_topk_length(idx)

    out_s = torch.empty(s_q, H_Q, D_V, dtype=torch.bfloat16, device="cuda")
    ml_s = torch.empty(s_q, H_Q, dtype=torch.float32, device="cuda")
    lse_s = torch.empty(s_q, H_Q, dtype=torch.float32, device="cuda")

    # --- doubled-grid arm (in-kernel split compute phase, no merge) ---
    idx2 = interleaved_halves(idx, topk)  # [2s, 1, 1024]
    tl2 = backscan_topk_length(idx2)  # per-half true lengths (min 1)
    q2 = q.repeat_interleave(2, dim=0).contiguous()
    out_d = torch.empty(2 * s_q, H_Q, D_V, dtype=torch.bfloat16, device="cuda")
    ml_d = torch.empty(2 * s_q, H_Q, dtype=torch.float32, device="cuda")
    lse_d = torch.empty(2 * s_q, H_Q, dtype=torch.float32, device="cuda")

    def single(tl_arg):
        sparse_mla_q8kv8_prefill_fwd(
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
        )

    def doubled(tl_arg):
        sparse_mla_q8kv8_prefill_fwd(
            q=q2,
            kv=kv,
            indices=idx2,
            sm_scale=sm,
            q_scale=one,
            kv_scale=one,
            d_v=D_V,
            attn_sink=None,
            topk_length=tl_arg,
            out=out_d,
            max_logits=ml_d,
            lse=lse_d,
        )

    # Correctness of the interleaved-split semantics: LSE-merge halves vs single.
    single(tl)
    doubled(tl2)
    torch.cuda.synchronize()
    o1 = out_d[0::2].contiguous()
    o2 = out_d[1::2].contiguous()
    m1, m2 = ml_d[0::2].contiguous(), ml_d[1::2].contiguous()
    l1, l2 = lse_d[0::2].contiguous(), lse_d[1::2].contiguous()
    out_m = torch.empty_like(out_s)
    ml_m = torch.empty_like(ml_s)
    lse_m = torch.empty_like(lse_s)
    combine_split2(o1, m1, l1, o2, m2, l2, out_m, ml_m, lse_m)
    torch.cuda.synchronize()
    max_do = (out_m.float() - out_s.float()).abs().max().item()
    fin = torch.isfinite(lse_s)
    max_dl = (lse_m[fin] - lse_s[fin]).abs().max().item()
    ok = max_do <= 2e-2 and max_dl <= 1e-3 and torch.equal(torch.isfinite(lse_m), fin)
    print(
        f"  interleaved-split merged-vs-single: max|dO|={max_do:.4f} "
        f"max|dLSE|={max_dl:.2e}  {'PASS' if ok else 'FAIL'}"
    )

    t_single_bs = bench(lambda: single(tl), iters)
    t_single_no = bench(lambda: single(None), iters)
    t_doubled_bs = bench(lambda: doubled(tl2), iters)
    t_doubled_no = bench(lambda: doubled(None), iters)

    gain_bs = t_single_bs - t_doubled_bs
    gain_no = t_single_no - t_doubled_no
    print(
        f"-- {name}: s_q={s_q} topk={topk} (single grid {s_q} CTAs = "
        f"{s_q/132:.2f} waves; doubled {2*s_q} CTAs = {2*s_q/132:.2f} waves) --"
    )
    print(f"  single  backscan: {t_single_bs:8.1f} us | no-tl: {t_single_no:8.1f} us")
    print(f"  doubled backscan: {t_doubled_bs:8.1f} us | no-tl: {t_doubled_no:8.1f} us")
    print(
        f"  GROSS CEILING gain (backscan): {gain_bs:+7.1f} us "
        f"({gain_bs/t_single_bs*100:+.1f}%)   [no-tl: {gain_no:+.1f} us "
        f"({gain_no/t_single_no*100:+.1f}%)]"
    )
    return t_single_bs, t_doubled_bs


def main():
    dev = torch.cuda.get_device_properties(0)
    print(f"Device: {dev.name}, {dev.multi_processor_count} SMs")
    run_shape(437, production_valid_counts(437, 2048, 3500), "prod tail il=3.5k")
    print()
    run_shape(1024, production_valid_counts(1024, 2048, 8192), "s_q=1024")
    print()
    run_shape(4096, [2048] * 4096, "long-ctx s_q=4096")


if __name__ == "__main__":
    main()
