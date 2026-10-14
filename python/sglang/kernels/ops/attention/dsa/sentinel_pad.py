"""Single-pass fixed-width KPool index copy with -1 sentinel padding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sentinel_copy_kernel(
    src,
    dst,
    WIDTH: tl.constexpr,
    PADDED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + row * WIDTH + col, col < WIDTH, other=-1)
    tl.store(dst + row * PADDED + col, value, col < PADDED)


def pad_kpool_indices(indices: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Copy every query independently; output must be owned by the caller stream."""
    assert indices.ndim == output.ndim == 2
    assert indices.is_cuda and output.device == indices.device
    assert indices.dtype == output.dtype == torch.int32
    assert indices.is_contiguous() and output.is_contiguous()
    rows, width = indices.shape
    assert output.shape == (rows, triton.cdiv(width, 64) * 64)
    if rows:
        _sentinel_copy_kernel[(rows, triton.cdiv(output.shape[1], 256))](
            indices, output, width, output.shape[1], 256
        )
    return output
