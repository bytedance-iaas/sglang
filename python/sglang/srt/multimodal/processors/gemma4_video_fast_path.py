"""CUDA fast path for Gemma4 Unified video preprocessing.

Resize intentionally remains in Transformers. This module consumes the resized
NCHW uint8 tensor and directly writes the final model-patch representation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


GEMMA4_VIDEO_PATCH_SIZE = 16
GEMMA4_VIDEO_POOLING_SIZE = 3
GEMMA4_VIDEO_MERGED_PATCH_SIZE = (
    GEMMA4_VIDEO_PATCH_SIZE * GEMMA4_VIDEO_POOLING_SIZE
)
GEMMA4_VIDEO_MAX_SOFT_TOKENS = 70
GEMMA4_VIDEO_CHANNELS = 3
GEMMA4_VIDEO_FEATURE_SIZE = (
    GEMMA4_VIDEO_MERGED_PATCH_SIZE**2 * GEMMA4_VIDEO_CHANNELS
)


@triton.jit
def _gemma4_video_pack_kernel(
    input_ptr,
    output_ptr,
    input_stride_t,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    total_output_elements,
    rescale_divisor,
    GRID_WIDTH: tl.constexpr,
    VALID_PATCHES: tl.constexpr,
    MAX_SOFT_TOKENS: tl.constexpr,
    MERGED_PATCH_SIZE: tl.constexpr,
    CHANNELS: tl.constexpr,
    FEATURE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = offsets < total_output_elements

    feature_offset = offsets % FEATURE_SIZE
    frame_patch = offsets // FEATURE_SIZE
    patch_index = frame_patch % MAX_SOFT_TOKENS
    frame_index = frame_patch // MAX_SOFT_TOKENS

    channel = feature_offset % CHANNELS
    spatial_offset = feature_offset // CHANNELS
    local_x = spatial_offset % MERGED_PATCH_SIZE
    local_y = spatial_offset // MERGED_PATCH_SIZE

    patch_x = patch_index % GRID_WIDTH
    patch_y = patch_index // GRID_WIDTH
    input_x = patch_x * MERGED_PATCH_SIZE + local_x
    input_y = patch_y * MERGED_PATCH_SIZE + local_y
    input_offsets = (
        frame_index * input_stride_t
        + channel * input_stride_c
        + input_y * input_stride_h
        + input_x * input_stride_w
    )

    valid = output_mask & (patch_index < VALID_PATCHES)
    pixels = tl.load(input_ptr + input_offsets, mask=valid, other=0).to(tl.float32)
    # Gemma4's configured do_rescale+identity-normalize path is fused by
    # Transformers into float32 normalization with std=(255, 255, 255), i.e.
    # an actual division. Preserve that arithmetic so the fast path is bit-exact.
    tl.store(
        output_ptr + offsets,
        tl.div_rn(pixels, rescale_divisor),
        mask=output_mask,
    )


def fused_gemma4_video_pack(
    video: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    max_soft_tokens: int = GEMMA4_VIDEO_MAX_SOFT_TOKENS,
    rescale_factor: float = 1.0 / 255.0,
) -> torch.Tensor:
    """Pack one resized video directly into Gemma4 model patches.

    Args:
        video: CUDA uint8 tensor shaped ``(frames, 3, height, width)``. Height
            and width must be multiples of 48.
        output: Optional contiguous float32 destination shaped
            ``(frames, max_soft_tokens, 6912)``.
    """
    if video.ndim != 4 or video.shape[1] != GEMMA4_VIDEO_CHANNELS:
        raise ValueError(f"Expected NCHW RGB video, got shape={tuple(video.shape)}")
    if not video.is_cuda or video.dtype != torch.uint8:
        raise ValueError(
            f"Expected CUDA uint8 video, got device={video.device} dtype={video.dtype}"
        )

    frames, _, height, width = video.shape
    merged = GEMMA4_VIDEO_MERGED_PATCH_SIZE
    if height % merged or width % merged:
        raise ValueError(
            f"Video dimensions must be divisible by {merged}, got {height}x{width}"
        )

    grid_height = height // merged
    grid_width = width // merged
    valid_patches = grid_height * grid_width
    if valid_patches > max_soft_tokens:
        raise ValueError(
            f"Video produces {valid_patches} patches, limit is {max_soft_tokens}"
        )

    expected_shape = (frames, max_soft_tokens, GEMMA4_VIDEO_FEATURE_SIZE)
    if output is None:
        output = torch.empty(expected_shape, device=video.device, dtype=torch.float32)
    elif (
        tuple(output.shape) != expected_shape
        or output.device != video.device
        or output.dtype != torch.float32
        or not output.is_contiguous()
    ):
        raise ValueError(
            "Output must be contiguous float32 on the input device with shape "
            f"{expected_shape}, got shape={tuple(output.shape)} "
            f"device={output.device} dtype={output.dtype}"
        )

    total_elements = output.numel()
    if total_elements == 0:
        return output

    block_size = 1024
    grid = (triton.cdiv(total_elements, block_size),)
    _gemma4_video_pack_kernel[grid](
        video,
        output,
        video.stride(0),
        video.stride(1),
        video.stride(2),
        video.stride(3),
        total_elements,
        1.0 / rescale_factor,
        GRID_WIDTH=grid_width,
        VALID_PATCHES=valid_patches,
        MAX_SOFT_TOKENS=max_soft_tokens,
        MERGED_PATCH_SIZE=merged,
        CHANNELS=GEMMA4_VIDEO_CHANNELS,
        FEATURE_SIZE=GEMMA4_VIDEO_FEATURE_SIZE,
        BLOCK_SIZE=block_size,
    )
    return output


class Gemma4VideoPositionCache:
    """Precomputed position IDs for every legal merged-patch grid."""

    def __init__(
        self,
        *,
        device: torch.device | str,
        max_frames: int = 32,
        max_soft_tokens: int = GEMMA4_VIDEO_MAX_SOFT_TOKENS,
    ):
        keys: list[tuple[int, int]] = []
        entries: list[torch.Tensor] = []

        for grid_height in range(1, max_soft_tokens + 1):
            for grid_width in range(1, max_soft_tokens // grid_height + 1):
                valid = grid_height * grid_width
                positions = torch.full(
                    (max_frames, max_soft_tokens, 2),
                    -1,
                    dtype=torch.long,
                )
                patch_indices = torch.arange(valid, dtype=torch.long)
                positions[:, :valid, 0] = patch_indices.remainder(grid_width)
                positions[:, :valid, 1] = torch.div(
                    patch_indices,
                    grid_width,
                    rounding_mode="floor",
                )
                keys.append((grid_height, grid_width))
                entries.append(positions)

        self._key_to_index = {key: index for index, key in enumerate(keys)}
        self._table = torch.stack(entries, dim=0).to(device=device)
        self.max_frames = max_frames
        self.max_soft_tokens = max_soft_tokens

    def get(
        self,
        grid_height: int,
        grid_width: int,
        num_frames: int,
    ) -> torch.Tensor:
        if num_frames > self.max_frames:
            raise ValueError(
                f"Position cache supports {self.max_frames} frames, got {num_frames}"
            )
        index = self._key_to_index[(grid_height, grid_width)]
        return self._table[index, :num_frames]

    @property
    def num_entries(self) -> int:
        return len(self._key_to_index)

    @property
    def nbytes(self) -> int:
        return self._table.numel() * self._table.element_size()

    @property
    def device(self) -> torch.device:
        return self._table.device
