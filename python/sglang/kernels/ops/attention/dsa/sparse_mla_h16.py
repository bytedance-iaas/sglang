"""SM90 native-H16 prefill over the existing DSA paged cache format."""

import torch


def sparse_mla_h16_paged_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """Read BF16 or 528-byte group-scaled NoPE KV at physical token indices.

    FP8 conversion materializes one BF16 cache workspace (1024 bytes per
    physical token). It is part of this path's latency and peak-memory cost.
    Invalid indices, including holes before KPool tail tokens, remain masked.
    """
    from sglang.kernels.ops.attention.dsa.dequant_k_cache import dequantize_k_cache
    from sglang.kernels.ops.attention.sparse_mla_h16_sm90 import sparse_mla_h16_fwd

    if q.dtype != torch.bfloat16 or q.ndim != 3 or q.shape[1:] != (16, 512):
        raise ValueError("H16 prefill requires BF16 Q with shape [tokens, 16, 512]")
    if kv_cache.ndim != 3 or kv_cache.shape[1] != 1:
        raise ValueError("KV cache must have shape [physical tokens, 1, cache dim]")
    if kv_cache.dtype == torch.float8_e4m3fn and kv_cache.shape[-1] == 528:
        kv_cache = dequantize_k_cache(kv_cache)
    elif kv_cache.dtype != torch.bfloat16 or kv_cache.shape[-1] != 512:
        raise ValueError("H16 prefill requires BF16 NoPE or 528-byte group-scaled KV")

    width = indices.shape[-1]
    padding = (-width) % 64
    if padding:
        indices = torch.nn.functional.pad(indices, (0, padding), value=-1)
    # A count of valid indices is incorrect when masked holes precede live
    # tail tokens. The kernel checks each physical index independently.
    lengths = torch.full((q.shape[0],), width, dtype=torch.int32, device=q.device)
    if attn_sink is None:
        attn_sink = torch.full((16,), -torch.inf, dtype=torch.float32, device=q.device)
    return sparse_mla_h16_fwd(
        q.contiguous(),
        kv_cache.view(-1, 512),
        indices.contiguous(),
        lengths,
        attn_sink,
        sm_scale,
    )
