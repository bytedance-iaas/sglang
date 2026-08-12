"""CPU contract tests for the device-preserving video decoder wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors import base_processor
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import video_decoder
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _StubProcessor(BaseMultimodalProcessor):
    gpu_video_decode = False
    video_decode_dimension_order = "NHWC"


class _FakeDecoder:
    def __init__(self, tensor):
        self.tensor = tensor

    def get_frames_at(self, _indices):
        return SimpleNamespace(data=self.tensor)


def _wrapper(tensor, *, dimension_order="NCHW", device="cpu"):
    wrapper = VideoDecoderWrapper.__new__(VideoDecoderWrapper)
    wrapper._source = "unused.mp4"
    wrapper._decoder = _FakeDecoder(tensor)
    wrapper._dimension_order = dimension_order
    wrapper._requested_device = device
    wrapper._num_decode_threads = 1
    wrapper._tc_kwargs = {"dimension_order": dimension_order}
    if device == "cuda":
        wrapper._tc_kwargs["device"] = "cuda"
    wrapper._source_bytes = None
    wrapper._source_path = None
    wrapper._tmp_path = None
    return wrapper


def test_nchw_decoder_preserves_tensor_layout(monkeypatch):
    monkeypatch.setattr(video_decoder, "_BACKEND", "torchcodec")
    frames = torch.arange(2 * 3 * 4 * 5, dtype=torch.uint8).reshape(2, 3, 4, 5)
    wrapper = _wrapper(frames)

    actual = wrapper.get_frames_as_tensor([0, 1])
    assert torch.equal(actual, frames)
    assert actual.shape == (2, 3, 4, 5)


def test_legacy_numpy_api_is_always_nhwc(monkeypatch):
    monkeypatch.setattr(video_decoder, "_BACKEND", "torchcodec")
    frames = torch.arange(2 * 3 * 4 * 5, dtype=torch.uint8).reshape(2, 3, 4, 5)
    wrapper = _wrapper(frames)

    actual = wrapper.get_frames_at([0, 1])
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(actual, frames.permute(0, 2, 3, 1).numpy())


def test_cuda_decode_error_recreates_decoder_on_cpu(monkeypatch):
    monkeypatch.setattr(video_decoder, "_BACKEND", "torchcodec")
    frames = torch.zeros((2, 3, 4, 5), dtype=torch.uint8)
    wrapper = _wrapper(frames, device="cuda")

    class _FailingDecoder:
        def get_frames_at(self, _indices):
            raise RuntimeError("synthetic CUDA decode failure")

    wrapper._decoder = _FailingDecoder()
    monkeypatch.setattr(
        video_decoder,
        "_create_torchcodec_decoder",
        lambda _source, _kwargs: _FakeDecoder(frames),
    )

    actual = wrapper.get_frames_as_tensor([0, 1])
    assert torch.equal(actual, frames)
    assert wrapper.decode_device == "cpu"
    assert "device" not in wrapper._tc_kwargs


def test_frame_limit_is_not_bound_to_gpu_flag(monkeypatch):
    captured = {}

    def fake_load_video(data, *, use_gpu, dimension_order):
        captured.update(
            data=data,
            use_gpu=use_gpu,
            dimension_order=dimension_order,
        )
        return "decoded"

    monkeypatch.setattr(base_processor, "load_video", fake_load_video)
    result = _StubProcessor._load_single_item(
        "synthetic-video",
        Modality.VIDEO,
        frame_count_limit=17,
    )

    assert result == "decoded"
    assert captured == {
        "data": "synthetic-video",
        "use_gpu": False,
        "dimension_order": "NHWC",
    }
