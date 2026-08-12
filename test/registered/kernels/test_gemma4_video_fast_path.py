"""Correctness tests for the Gemma4 CUDA video preprocessing fast path."""

from __future__ import annotations

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Gemma4 video preprocessing fast path requires CUDA",
)


def _reference(video: torch.Tensor, max_soft_tokens: int = 70):
    from transformers.models.gemma4_unified.video_processing_gemma4_unified import (
        convert_video_to_patches,
        pad_to_max_patches,
        patches_merge,
    )

    patch_size = 16
    pooling_size = 3
    # Transformers fuses identity normalization and rescaling into
    # ``float32_video.div_(tensor([255, 255, 255]))``. Keep the tensor divisor
    # here so this reference catches a reciprocal-multiply rounding change.
    divisor = torch.full((3, 1, 1), 255.0, device=video.device)
    video = video.to(torch.float32).div_(divisor)
    frames, _, height, width = video.shape
    patch_height = height // patch_size
    patch_width = width // patch_size
    patches = convert_video_to_patches(video, patch_size)

    patch_grid = torch.meshgrid(
        torch.arange(patch_width, device=video.device),
        torch.arange(patch_height, device=video.device),
        indexing="xy",
    )
    teacher_positions = torch.stack(patch_grid, dim=-1).reshape(-1, 2)
    teacher_positions = teacher_positions[None].repeat(frames, 1, 1)
    merged, positions = patches_merge(
        patches,
        teacher_positions,
        patches.shape[1] // (pooling_size**2),
    )
    return pad_to_max_patches(merged, positions, max_soft_tokens)


@pytest.mark.parametrize(
    "frames,grid_height,grid_width",
    [
        (1, 11, 6),
        (16, 6, 11),
        (2, 10, 7),
        (32, 5, 8),
    ],
)
def test_fused_pack_matches_transformers(frames, grid_height, grid_width):
    from sglang.srt.multimodal.processors.gemma4_video_fast_path import (
        fused_gemma4_video_pack,
    )

    torch.manual_seed(frames * 1000 + grid_height * 100 + grid_width)
    video = torch.randint(
        0,
        256,
        (frames, 3, grid_height * 48, grid_width * 48),
        dtype=torch.uint8,
        device="cuda",
    )
    expected_pixels, expected_positions = _reference(video)
    actual_pixels = fused_gemma4_video_pack(video)

    assert actual_pixels.shape == expected_pixels.shape
    assert actual_pixels.dtype == torch.float32
    torch.testing.assert_close(
        actual_pixels,
        expected_pixels,
        atol=1e-6,
        rtol=0,
    )
    assert torch.equal(actual_pixels, expected_pixels)

    valid = grid_height * grid_width
    assert torch.count_nonzero(actual_pixels[:, valid:]) == 0

    from sglang.srt.multimodal.processors.gemma4_video_fast_path import (
        Gemma4VideoPositionCache,
    )

    cache = Gemma4VideoPositionCache(device="cuda")
    actual_positions = cache.get(grid_height, grid_width, frames)
    assert torch.equal(actual_positions, expected_positions)
    assert torch.all(actual_positions[:, valid:] == -1)


def test_multiple_videos_write_to_independent_output_slices():
    from sglang.srt.multimodal.processors.gemma4_video_fast_path import (
        GEMMA4_VIDEO_FEATURE_SIZE,
        fused_gemma4_video_pack,
    )

    torch.manual_seed(2026)
    videos = [
        torch.randint(
            0,
            256,
            (2, 3, 5 * 48, 8 * 48),
            dtype=torch.uint8,
            device="cuda",
        )
        for _ in range(2)
    ]
    output = torch.empty(
        (2, 2, 70, GEMMA4_VIDEO_FEATURE_SIZE),
        dtype=torch.float32,
        device="cuda",
    )
    for index, video in enumerate(videos):
        fused_gemma4_video_pack(video, output=output[index])

    for index, video in enumerate(videos):
        expected, _ = _reference(video)
        torch.testing.assert_close(output[index], expected, atol=1e-6, rtol=0)

    assert not torch.equal(output[0], output[1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
