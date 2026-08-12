"""Minimal async invariant probes required by the DSpark runtime."""

from typing import Optional

import torch

from sglang.srt.environ import envs


def maybe_assert_async(cond: torch.Tensor, msg: str = "") -> None:
    if envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        torch._assert_async(cond, msg)


def maybe_detect_in_closed_range(
    tensor: Optional[torch.Tensor], low: float, high: float, msg: str = ""
) -> None:
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if tensor is None or tensor.numel() == 0:
        return
    torch._assert_async(
        ((tensor >= low) & (tensor <= high)).all(),
        f"value outside [{low}, {high}]: {msg}",
    )
