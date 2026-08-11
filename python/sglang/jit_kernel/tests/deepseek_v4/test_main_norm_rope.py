from __future__ import annotations

import torch

from sglang.jit_kernel.benchmark.bench_activation import register_cuda_ci
from sglang.jit_kernel.deepseek_v4 import fused_k_norm_rope_flashmla


register_cuda_ci(est_time=20, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=20, suite="nightly-kernel-1-gpu", nightly=True)


def test_fused_k_norm_rope_skips_negative_out_loc() -> None:
    """Rejected DSpark verify rows use -1 and must not write the SWA cache."""
    device = torch.device("cuda")
    batch_size = 6
    head_dim = 512
    rope_dim = 64
    page_size = 256
    page_bytes = ((584 * page_size + 575) // 576) * 576

    kv = torch.randn(batch_size, head_dim, dtype=torch.bfloat16, device=device)
    kv_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    freqs_cis = torch.ones(
        batch_size, rope_dim // 2, dtype=torch.complex64, device=device
    )
    positions = torch.arange(batch_size, dtype=torch.int64, device=device)

    # DSpark gamma=5 verifies six rows. With commit_lens=1, only the first row
    # has a write target and the five rejected rows carry the -1 sentinel.
    out_loc = torch.tensor(
        [0, -1, -1, -1, -1, -1], dtype=torch.int32, device=device
    )
    kvcache = torch.full((1, page_bytes), 0xA5, dtype=torch.uint8, device=device)
    original = kvcache.clone()

    fused_k_norm_rope_flashmla(
        kv=kv,
        kv_weight=kv_weight,
        eps=1e-6,
        freqs_cis=freqs_cis,
        positions=positions,
        out_loc=out_loc,
        kvcache=kvcache,
        page_size=page_size,
    )
    torch.cuda.synchronize(device)

    # The valid row is written, while the rest of the page stays untouched.
    assert not torch.equal(kvcache[0, :576], original[0, :576])
    torch.testing.assert_close(kvcache[0, 576:], original[0, 576:])
