"""Pack fake-FP4 queries in head batches without changing their scale floor."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    _ceil_ue8m0_exp,
    _fp4_e2m1_code_rne,
)


@triton.jit
def _pack_queries(
    x, payload, scales, N, HEADS_PER_CTA: tl.constexpr, SCALE_FLOOR: tl.constexpr
):
    heads = tl.program_id(0) * HEADS_PER_CTA + tl.arange(0, HEADS_PER_CTA)
    d = tl.arange(0, 128)
    values = tl.load(x + heads[:, None] * 128 + d[None, :], heads[:, None] < N, 0).to(
        tl.float32
    )
    blocks = tl.reshape(values, (HEADS_PER_CTA, 4, 32))
    amax = tl.max(tl.abs(blocks), axis=2)
    exponent = _ceil_ue8m0_exp(tl.maximum(amax / 6.0, SCALE_FLOOR))
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    codes = _fp4_e2m1_code_rne(blocks / scale[:, :, None])
    low, high = tl.split(tl.reshape(codes, (HEADS_PER_CTA, 64, 2)))
    packed = low | (high << 4)
    tl.store(
        payload + heads[:, None] * 64 + tl.arange(0, 64)[None, :],
        packed,
        heads[:, None] < N,
    )
    sf = tl.sum(exponent.to(tl.uint32) << (tl.arange(0, 4)[None, :] * 8), axis=1)
    tl.store(scales + heads, sf, heads < N)


def pack_queries(x):
    heads_per_cta = 8
    assert x.shape[-1] == 128 and x.dtype == torch.bfloat16
    x = x.contiguous().view(-1, 128)
    payload = torch.empty((x.shape[0], 64), device=x.device, dtype=torch.int8)
    sf = torch.empty((x.shape[0],), device=x.device, dtype=torch.int32)
    if x.shape[0]:
        _pack_queries[(triton.cdiv(x.shape[0], heads_per_cta),)](
            x,
            payload,
            sf,
            x.shape[0],
            HEADS_PER_CTA=heads_per_cta,
            SCALE_FLOOR=2.0**-126,
            num_warps=4,
        )
    return payload, sf
