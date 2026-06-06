"""Async invariant probes without CPU sync.

Compatibility shim for newer speculative workers. This branch uses separate
speculative debug gates instead of main's unified SGLANG_ENABLE_ASYNC_ASSERT.
"""

import torch

from sglang.srt.environ import envs


def maybe_detect_nan(tensor: torch.Tensor, msg: str = ""):
    """Async NaN check; error surfaces at the next CUDA sync point."""
    if not envs.SGLANG_SPEC_NAN_DETECTION.get():
        return
    torch._assert_async(~torch.any(torch.isnan(tensor)), f"NaN detected! {msg}")


def maybe_detect_inf(tensor: torch.Tensor, msg: str = ""):
    """Async Inf check; share the NaN detection gate for this branch."""
    if not envs.SGLANG_SPEC_NAN_DETECTION.get():
        return
    torch._assert_async(~torch.any(torch.isinf(tensor)), f"Inf detected! {msg}")


def maybe_detect_oob(indices: torch.Tensor, low: int, high: int, msg: str):
    """Async OOB check; error surfaces at the next CUDA sync point."""
    if not envs.SGLANG_SPEC_OOB_DETECTION.get():
        return
    if indices.numel() == 0:
        return
    torch._assert_async(
        (indices.min() >= low) & (indices.max() < high),
        f"OOB indices not in [{low}, {high}): {msg}",
    )


def maybe_detect_page_aligned(indices: torch.Tensor, page_size: int, msg: str):
    """Async page-alignment check on slot ids."""
    if not envs.SGLANG_SPEC_OOB_DETECTION.get():
        return
    if indices.numel() == 0 or page_size <= 1:
        return
    torch._assert_async(
        (indices % page_size == 0).all(),
        f"page-misaligned indices (page_size={page_size}): {msg}",
    )
