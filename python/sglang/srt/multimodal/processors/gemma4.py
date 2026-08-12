# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import functools
import inspect
import logging
import os
import re
import threading
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import BatchFeature

from sglang.srt.environ import envs
from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.gemma4_audio import _SSCP_CONV_STRIDE_SIZES
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.utils.video_decoder import VideoDecoderWrapper


logger = logging.getLogger(__name__)

_VIDEO_TIMING_ENABLED = os.getenv("SGLANG_GEMMA4_VIDEO_TIMING", "0") == "1"
_VIDEO_FAST_PATH_ENABLED = envs.SGLANG_GEMMA4_VIDEO_FAST_PATH.get()
_video_timing_state = threading.local()


def _channel_values(value, default):
    if value is None:
        value = default
    if isinstance(value, (int, float)):
        return (float(value),) * 3
    return tuple(float(item) for item in value)


def _fast_video_config_supported(arguments) -> tuple[bool, str]:
    """Validate the exact Gemma4-12B preprocessing configuration."""
    rescale_factor = arguments.get("rescale_factor")
    checks = {
        "patch_size": arguments.get("patch_size") == 16,
        "pooling_kernel_size": arguments.get("pooling_kernel_size") == 3,
        "max_soft_tokens": arguments.get("max_soft_tokens") == 70,
        "do_rescale": arguments.get("do_rescale") is True,
        "rescale_factor": isinstance(rescale_factor, (int, float))
        and abs(float(rescale_factor) - 1.0 / 255.0) < 1e-12,
    }
    for name, supported in checks.items():
        if not supported:
            return False, f"unsupported_{name}={arguments.get(name)!r}"

    if arguments.get("do_normalize"):
        mean = _channel_values(arguments.get("image_mean"), (0.0, 0.0, 0.0))
        std = _channel_values(arguments.get("image_std"), (1.0, 1.0, 1.0))
        if mean != (0.0, 0.0, 0.0) or std != (1.0, 1.0, 1.0):
            return False, f"non_identity_normalize mean={mean} std={std}"

    return True, "supported"


def _install_fast_video_preprocess(video_processor, device):
    """Replace only Gemma4's post-prepare video preprocessing implementation."""
    if not _VIDEO_FAST_PATH_ENABLED or video_processor is None:
        return
    if getattr(video_processor, "_sglang_gemma4_fast_path_installed", False):
        return

    original_preprocess = getattr(video_processor, "_preprocess", None)
    if original_preprocess is None:
        logger.warning(
            "[Gemma4VideoFastPath] disabled reason=missing_video_preprocess"
        )
        return

    try:
        from sglang.srt.multimodal.processors.gemma4_video_fast_path import (
            GEMMA4_VIDEO_FEATURE_SIZE,
            GEMMA4_VIDEO_MERGED_PATCH_SIZE,
            Gemma4VideoPositionCache,
            fused_gemma4_video_pack,
        )

        position_cache = Gemma4VideoPositionCache(device=device)
    except Exception as exc:
        logger.warning(
            "[Gemma4VideoFastPath] disabled reason=position_cache_init_failed "
            "error=%s",
            exc,
        )
        return

    signature = inspect.signature(original_preprocess)

    @functools.wraps(original_preprocess)
    def fast_preprocess(*args, **kwargs):
        try:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError as exc:
            logger.warning(
                "[Gemma4VideoFastPath] path=reference reason=signature_mismatch "
                "error=%s",
                exc,
            )
            return original_preprocess(*args, **kwargs)

        arguments = dict(bound.arguments)
        extra_kwargs = arguments.pop("kwargs", {})
        arguments.update(extra_kwargs)
        supported, reason = _fast_video_config_supported(arguments)
        videos = arguments.get("videos")
        if not supported or not isinstance(videos, list) or not videos:
            logger.info(
                "[Gemma4VideoFastPath] path=reference reason=%s",
                reason if not supported else "invalid_video_batch",
            )
            return original_preprocess(*args, **kwargs)

        tensor_inputs = [video for video in videos if isinstance(video, torch.Tensor)]
        frame_counts = {
            video.shape[0] for video in tensor_inputs if video.ndim == 4
        }
        tensor_inputs_supported = (
            len(tensor_inputs) == len(videos)
            and len(frame_counts) == 1
            and all(
                video.ndim == 4
                and video.shape[1] == 3
                and video.is_cuda
                and video.dtype == torch.uint8
                and video.device == position_cache.device
                for video in tensor_inputs
            )
        )
        if not tensor_inputs_supported:
            logger.info(
                "[Gemma4VideoFastPath] path=reference "
                "reason=unsupported_tensor_inputs inputs=%s",
                [_tensor_trace(video) for video in videos],
            )
            return original_preprocess(*args, **kwargs)

        try:
            preprocess_started = time.perf_counter()
            resized_videos = []
            for video in videos:
                if arguments.get("do_resize"):
                    video = video_processor.aspect_ratio_preserving_resize(
                        video=video,
                        patch_size=arguments["patch_size"],
                        max_patches=(
                            arguments["max_soft_tokens"]
                            * arguments["pooling_kernel_size"] ** 2
                        ),
                        pooling_kernel_size=arguments["pooling_kernel_size"],
                        resample=arguments["resample"],
                    )
                height, width = video.shape[-2:]
                if (
                    height % GEMMA4_VIDEO_MERGED_PATCH_SIZE
                    or width % GEMMA4_VIDEO_MERGED_PATCH_SIZE
                ):
                    raise ValueError(
                        "resize output is not aligned to merged patch size: "
                        f"{height}x{width}"
                    )
                resized_videos.append(video)

            num_frames = next(iter(frame_counts))
            pixel_values = torch.empty(
                (
                    len(resized_videos),
                    num_frames,
                    arguments["max_soft_tokens"],
                    GEMMA4_VIDEO_FEATURE_SIZE,
                ),
                device=resized_videos[0].device,
                dtype=torch.float32,
            )
            position_ids = []
            num_soft_tokens_per_video = []

            if _VIDEO_TIMING_ENABLED:
                _sync_cuda_tensors(resized_videos)
                fused_started = time.perf_counter()

            for video_index, video in enumerate(resized_videos):
                fused_gemma4_video_pack(
                    video,
                    output=pixel_values[video_index],
                    max_soft_tokens=arguments["max_soft_tokens"],
                    rescale_factor=arguments["rescale_factor"],
                )
                grid_height = video.shape[-2] // GEMMA4_VIDEO_MERGED_PATCH_SIZE
                grid_width = video.shape[-1] // GEMMA4_VIDEO_MERGED_PATCH_SIZE
                num_soft_tokens_per_video.append(grid_height * grid_width)
                position_ids.append(
                    position_cache.get(grid_height, grid_width, num_frames)
                )

            if len(position_ids) == 1:
                video_position_ids = position_ids[0].unsqueeze(0)
            else:
                video_position_ids = torch.stack(position_ids, dim=0)

            if _VIDEO_TIMING_ENABLED:
                _sync_cuda_tensors((pixel_values, video_position_ids))
                logger.info(
                    "[Gemma4VideoTiming] step=9_13_fused elapsed_ms=%.3f "
                    "videos=%d",
                    (time.perf_counter() - fused_started) * 1000,
                    len(resized_videos),
                )
                logger.info(
                    "[Gemma4VideoTiming] step=8_13_fast_preprocess "
                    "elapsed_ms=%.3f videos=%d",
                    (time.perf_counter() - preprocess_started) * 1000,
                    len(resized_videos),
                )

            logger.info(
                "[Gemma4VideoFastPath] path=fused resized_shapes=%s "
                "valid_soft_tokens=%s position_cache=hit",
                [tuple(video.shape) for video in resized_videos],
                num_soft_tokens_per_video,
            )
            return BatchFeature(
                data={
                    "pixel_values_videos": pixel_values,
                    "video_position_ids": video_position_ids,
                    "num_soft_tokens_per_video": num_soft_tokens_per_video,
                },
                tensor_type=arguments.get("return_tensors"),
            )
        except Exception as exc:
            logger.warning(
                "[Gemma4VideoFastPath] path=reference "
                "reason=fused_preprocess_failed error=%s",
                exc,
                exc_info=True,
            )
            return original_preprocess(*args, **kwargs)

    video_processor._preprocess = fast_preprocess
    video_processor._sglang_gemma4_fast_path_installed = True
    video_processor._sglang_gemma4_position_cache = position_cache
    logger.info(
        "[Gemma4VideoFastPath] installed device=%s position_cache_entries=%d "
        "position_cache_mb=%.2f",
        device,
        position_cache.num_entries,
        position_cache.nbytes / (1024 * 1024),
    )


def _sync_cuda_tensors(value):
    """Synchronize CUDA tensors contained in a nested processor value."""
    devices = set()

    def collect(current):
        if isinstance(current, torch.Tensor) and current.is_cuda:
            devices.add(current.device)
        elif isinstance(current, dict):
            for nested in current.values():
                collect(nested)
        elif isinstance(current, (list, tuple)):
            for nested in current:
                collect(nested)

    collect(value)
    for device in devices:
        torch.cuda.synchronize(device)


def _install_video_timing_hooks(video_processor):
    """Instrument the Transformers Gemma4 video processor without patching it.

    CUDA preprocessing is asynchronous, so every measured boundary explicitly
    synchronizes.  This mode is intentionally for profiling rather than throughput
    benchmarking and is enabled only by SGLANG_GEMMA4_VIDEO_TIMING=1.
    """
    if not _VIDEO_TIMING_ENABLED or video_processor is None:
        return
    if getattr(video_processor, "_sglang_gemma4_timing_installed", False):
        return

    def wrap_method(name, step):
        original = getattr(video_processor, name, None)
        if original is None:
            logger.warning(
                "[Gemma4VideoTiming] step=%s unavailable_method=%s",
                step,
                name,
            )
            return

        @functools.wraps(original)
        def timed(*args, **kwargs):
            is_prepare = name == "_prepare_input_videos"
            if not is_prepare and not getattr(_video_timing_state, "active", False):
                return original(*args, **kwargs)
            _sync_cuda_tensors((args, kwargs))
            started = time.perf_counter()
            result = original(*args, **kwargs)
            _sync_cuda_tensors(result)
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "[Gemma4VideoTiming] step=%s elapsed_ms=%.3f",
                step,
                elapsed_ms,
            )
            return result

        setattr(video_processor, name, timed)

    # BaseVideoProcessor.preprocess calls this before _preprocess; it performs
    # input layout preparation and the CPU -> GPU transfer requested by device=.
    wrap_method("_prepare_input_videos", "7_prepare_and_move")
    wrap_method("aspect_ratio_preserving_resize", "8_resize")
    wrap_method("rescale_and_normalize", "9_rescale_normalize")

    processor_module = inspect.getmodule(type(video_processor))
    if processor_module is not None:
        helper_steps = {
            "convert_video_to_patches": "10_patchify",
            "patches_merge": "12_merge_3x3",
            "pad_to_max_patches": "13a_pad",
        }
        for name, step in helper_steps.items():
            original = getattr(processor_module, name, None)
            if original is None:
                logger.warning(
                    "[Gemma4VideoTiming] step=%s unavailable_helper=%s",
                    step,
                    name,
                )
                continue
            if getattr(original, "_sglang_gemma4_timing_wrapped", False):
                continue

            def make_timed_helper(helper, helper_name, helper_step):
                @functools.wraps(helper)
                def timed_helper(*args, **kwargs):
                    if not getattr(_video_timing_state, "active", False):
                        return helper(*args, **kwargs)

                    # Position-ID construction is inline between patchify and
                    # patches_merge in Transformers, so the next helper boundary
                    # is the exact point at which step 11 can be observed.
                    if helper_name == "patches_merge":
                        patchify_end = getattr(
                            _video_timing_state, "patchify_end", None
                        )
                        if patchify_end is not None:
                            _sync_cuda_tensors((args, kwargs))
                            position_ms = (
                                time.perf_counter() - patchify_end
                            ) * 1000
                            logger.info(
                                "[Gemma4VideoTiming] "
                                "step=11_position_ids elapsed_ms=%.3f",
                                position_ms,
                            )
                            _video_timing_state.patchify_end = None

                    _sync_cuda_tensors((args, kwargs))
                    started = time.perf_counter()
                    result = helper(*args, **kwargs)
                    _sync_cuda_tensors(result)
                    ended = time.perf_counter()
                    elapsed_ms = (ended - started) * 1000
                    logger.info(
                        "[Gemma4VideoTiming] step=%s elapsed_ms=%.3f",
                        helper_step,
                        elapsed_ms,
                    )
                    if helper_name == "convert_video_to_patches":
                        _video_timing_state.patchify_end = ended
                    elif helper_name == "pad_to_max_patches":
                        _video_timing_state.pad_ms += elapsed_ms
                        _video_timing_state.last_pad_end = ended
                    return result

                timed_helper._sglang_gemma4_timing_wrapped = True
                return timed_helper

            setattr(
                processor_module,
                name,
                make_timed_helper(original, name, step),
            )

    original_preprocess = getattr(video_processor, "_preprocess", None)
    if original_preprocess is None:
        logger.warning(
            "[Gemma4VideoTiming] step=13_pad_stack unavailable_method=_preprocess"
        )
    else:

        @functools.wraps(original_preprocess)
        def timed_preprocess(*args, **kwargs):
            _video_timing_state.active = True
            _video_timing_state.patchify_end = None
            _video_timing_state.pad_ms = 0.0
            _video_timing_state.last_pad_end = None
            try:
                result = original_preprocess(*args, **kwargs)
                _sync_cuda_tensors(result)
                ended = time.perf_counter()
                last_pad_end = getattr(_video_timing_state, "last_pad_end", None)
                finalize_ms = (
                    (ended - last_pad_end) * 1000
                    if last_pad_end is not None
                    else 0.0
                )
                pad_ms = getattr(_video_timing_state, "pad_ms", 0.0)
                logger.info(
                    "[Gemma4VideoTiming] step=13_pad_stack elapsed_ms=%.3f "
                    "pad_ms=%.3f stack_finalize_ms=%.3f",
                    pad_ms + finalize_ms,
                    pad_ms,
                    finalize_ms,
                )
                return result
            finally:
                _video_timing_state.active = False

        setattr(video_processor, "_preprocess", timed_preprocess)

    video_processor._sglang_gemma4_timing_installed = True
    logger.info(
        "[Gemma4VideoTiming] instrumentation=installed processor=%s",
        type(video_processor).__name__,
    )


def _tensor_trace(value):
    """Return the small, stable tensor fields needed by the video trace logs."""
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__}
    return {
        "shape": tuple(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _video_input_trace(value):
    """Describe a video input without dumping bytes/base64 payloads into logs."""
    if isinstance(value, VideoDecoderWrapper):
        decoder = getattr(value, "_decoder", None)
        try:
            total_frames = len(value)
        except Exception:
            total_frames = "unknown"
        return {
            "type": type(value).__name__,
            "decoder": type(decoder).__name__ if decoder is not None else "unknown",
            "decode_device": value.decode_device,
            "dimension_order": value.dimension_order,
            "cpu_fallback": value.cpu_fallback_status,
            "total_frames": total_frames,
        }
    if isinstance(value, str):
        # Local paths are useful for correlating a trace with a request, but avoid
        # logging arbitrarily large data URLs/base64 strings.
        source = value if len(value) <= 256 else f"{value[:96]}...({len(value)} chars)"
        return {"type": "str", "source": source}
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    if isinstance(value, torch.Tensor):
        return _tensor_trace(value)
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": "cpu",
        }
    return {"type": type(value).__name__}


class Gemma4SGLangProcessor(SGLangBaseProcessor):
    """Multimodal processor for Gemma4 supporting image, video, and audio inputs."""

    models = [Gemma4ForConditionalGeneration]
    gpu_video_decode = _VIDEO_FAST_PATH_ENABLED
    video_decode_dimension_order = "NCHW" if _VIDEO_FAST_PATH_ENABLED else "NHWC"

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IM_START_TOKEN_ID = hf_config.boi_token_id
        self.IM_END_TOKEN_ID = hf_config.eoi_token_id

        self.AUDIO_START_TOKEN_ID = hf_config.boa_token_id
        self.AUDIO_END_TOKEN_ID = hf_config.eoa_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|image|>",
            image_token_id=hf_config.image_token_id,
            image_token_regex=re.compile(
                r"<\|image>(?:<\|image\|>)+<image\|>|<\|image\|>"
            ),
            video_token="<|video|>",
            video_token_id=hf_config.video_token_id,
            video_token_regex=re.compile(
                r"<\|image>(?:<\|video\|>)+<image\|>|<\|video\|>"
            ),
            audio_token="<|audio|>",
            audio_token_id=hf_config.audio_token_id,
            audio_token_regex=re.compile(
                r"<\|audio>(?:<\|audio\|>)+<audio\|>|<\|audio\|>"
            ),
        ).build(_processor)

        # Register image-processor and video-processor outputs so they are stored on
        # MultimodalDataItem via collect_mm_items_from_processor_output.
        self.ATTR_NAME_TO_MODALITY["image_position_ids"] = Modality.IMAGE
        self.ATTR_NAME_TO_MODALITY["video_position_ids"] = Modality.VIDEO
        video_processor = getattr(_processor, "video_processor", None)
        processor_device = getattr(server_args, "device", "cpu")
        if processor_device == "cuda":
            processor_device = f"cuda:{getattr(server_args, 'base_gpu_id', 0)}"
        _install_fast_video_preprocess(video_processor, processor_device)
        # Install timing after the fast path so the profiler wraps the active
        # implementation rather than the displaced Transformers method.
        _install_video_timing_hooks(video_processor)

    def _get_audio_pad_multiple(self) -> int:
        """Derive the waveform padding alignment from processor config.

        The HF processor's ceil(duration_ms / audio_ms_per_token) formula can
        overshoot by 1 token relative to what the SSCP convolutions produce.
        Padding waveforms to a multiple of (hop_length * first_conv_stride)
        aligns the two calculations.
        See: gemma-4-eap-extras/examples/gemma-4-audio-examples.ipynb
        """
        fe = getattr(self._processor, "feature_extractor", None)
        hop = getattr(fe, "hop_length", 160)
        first_stride = _SSCP_CONV_STRIDE_SIZES[0][0]
        return hop * first_stride

    def _video_decoder_to_tensor(self, vdw: VideoDecoderWrapper) -> torch.Tensor:
        """Convert a VideoDecoderWrapper to a (sampled_frames, C, H, W) uint8 tensor.

        SGLang's load_video returns VideoDecoderWrapper which the HF
        Gemma4VideoProcessor does not recognise (expects torch.Tensor or
        np.ndarray).  We replicate HF's uniform frame sampling here to
        avoid materialising the entire video in memory, then delegate the
        rest (resize, patchify, position IDs) to the HF video processor.
        """
        step_started = time.perf_counter()
        total = len(vdw)
        num_frames = getattr(
            getattr(self._processor, "video_processor", None),
            "num_frames",
            32,
        )
        if total <= num_frames:
            indices = list(range(total))
        else:
            indices = torch.arange(0, total, total / num_frames).int().tolist()
        logger.info(
            "[Gemma4VideoTrace] stage=sample_plan total_frames=%d configured_num_frames=%d "
            "sampled_frames=%d indices=%s decoder=%s",
            total,
            num_frames,
            len(indices),
            indices,
            _video_input_trace(vdw),
        )
        sampling_ms = (time.perf_counter() - step_started) * 1000
        decode_started = time.perf_counter()
        if _VIDEO_FAST_PATH_ENABLED:
            frames = vdw.get_frames_as_tensor(indices)
            if vdw.dimension_order == "NHWC":
                frames = frames.permute(0, 3, 1, 2).contiguous()
            elif not frames.is_contiguous():
                frames = frames.contiguous()
        else:
            frames_np = vdw.get_frames_at(indices)  # (N, H, W, C)
            frames = None
        decode_ms = (time.perf_counter() - decode_started) * 1000
        if _VIDEO_TIMING_ENABLED:
            logger.info(
                "[Gemma4VideoTiming] step=5_sample_and_decode elapsed_ms=%.3f "
                "sampling_ms=%.3f decode_ms=%.3f total_frames=%d sampled_frames=%d",
                sampling_ms + decode_ms,
                sampling_ms,
                decode_ms,
                total,
                len(indices),
            )
        if _VIDEO_FAST_PATH_ENABLED:
            logger.info(
                "[Gemma4VideoTrace] stage=decode_sampled tensor=%s layout=NCHW "
                "decoder=%s",
                _tensor_trace(frames),
                _video_input_trace(vdw),
            )
        else:
            logger.info(
                "[Gemma4VideoTrace] stage=decode_sampled output_shape=%s dtype=%s "
                "layout=NHWC device=cpu_numpy",
                tuple(frames_np.shape),
                frames_np.dtype,
            )
            layout_started = time.perf_counter()
            frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()
            layout_ms = (time.perf_counter() - layout_started) * 1000
            if _VIDEO_TIMING_ENABLED:
                logger.info(
                    "[Gemma4VideoTiming] step=6_numpy_to_nchw elapsed_ms=%.3f "
                    "bytes=%d",
                    layout_ms,
                    frames.numel() * frames.element_size(),
                )
        logger.info(
            "[Gemma4VideoTrace] stage=to_nchw sampled_tensor=%s",
            _tensor_trace(frames),
        )
        return frames

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ):
        if audios:
            pad_multiple = self._get_audio_pad_multiple()
            padded = []
            for a in audios:
                a = np.asarray(a)
                remainder = len(a) % pad_multiple
                if remainder != 0:
                    a = np.pad(a, (0, pad_multiple - remainder), mode="constant")
                padded.append(a)
            audios = padded
        if videos:
            logger.info(
                "[Gemma4VideoTrace] stage=processor_input videos=%s",
                [_video_input_trace(v) for v in videos],
            )
            videos = [
                (
                    self._video_decoder_to_tensor(v)
                    if isinstance(v, VideoDecoderWrapper)
                    else v
                )
                for v in videos
            ]
            kwargs.setdefault("do_sample_frames", False)
            processor_device = getattr(self.server_args, "device", "unknown")
            if processor_device == "cuda":
                processor_device = f"cuda:{getattr(self.server_args, 'base_gpu_id', 0)}"
            video_processor = getattr(self._processor, "video_processor", None)
            logger.info(
                "[Gemma4VideoTrace] stage=hf_preprocess_plan processor=%s execution_device=%s "
                "do_sample_frames=%s operations=%s inputs=%s config=%s",
                type(video_processor).__name__,
                processor_device,
                kwargs.get("do_sample_frames"),
                "prepare/move -> aspect_ratio_resize -> rescale/normalize -> "
                "patchify -> position_ids -> merge_3x3 -> pad/stack",
                [_tensor_trace(v) for v in videos],
                {
                    "num_frames": getattr(video_processor, "num_frames", None),
                    "patch_size": getattr(video_processor, "patch_size", None),
                    "pooling_kernel_size": getattr(
                        video_processor, "pooling_kernel_size", None
                    ),
                    "max_soft_tokens": getattr(video_processor, "max_soft_tokens", None),
                    "rescale_factor": getattr(video_processor, "rescale_factor", None),
                    "image_mean": getattr(video_processor, "image_mean", None),
                    "image_std": getattr(video_processor, "image_std", None),
                },
            )
        result = super().process_mm_data(
            input_text, images=images, videos=videos, audios=audios, **kwargs
        )
        if videos:
            logger.info(
                "[Gemma4VideoTrace] stage=hf_preprocess_output pixel_values_videos=%s "
                "video_position_ids=%s num_soft_tokens_per_video=%s "
                "keep_mm_feature_on_device=%s",
                _tensor_trace(result.get("pixel_values_videos")),
                _tensor_trace(result.get("video_position_ids")),
                result.get("num_soft_tokens_per_video"),
                self.server_args.keep_mm_feature_on_device,
            )
        return result

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,
        audio_data: Optional[List[Union[str, bytes, Dict]]] = None,
        input_text: str = "",
        request_obj=None,
        *args,
        **kwargs,
    ):
        """Process multimodal data including images, video, and audio."""
        video_data = request_obj.video_data if request_obj else None
        if video_data:
            logger.info(
                "[Gemma4VideoTrace] stage=raw_request count=%d inputs=%s",
                len(video_data),
                [_video_input_trace(v) for v in video_data],
            )
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output.videos:
            logger.info(
                "[Gemma4VideoTrace] stage=load_complete videos=%s",
                [_video_input_trace(v) for v in base_output.videos],
            )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
        )
