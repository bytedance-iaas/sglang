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

import logging
import re
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.gemma4_audio import _SSCP_CONV_STRIDE_SIZES
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.utils.video_decoder import VideoDecoderWrapper


logger = logging.getLogger(__name__)


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
        tc_kwargs = getattr(value, "_tc_kwargs", {})
        try:
            total_frames = len(value)
        except Exception:
            total_frames = "unknown"
        return {
            "type": type(value).__name__,
            "decoder": type(decoder).__name__ if decoder is not None else "unknown",
            "decode_device": tc_kwargs.get("device", "cpu"),
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
        frames_np = vdw.get_frames_at(indices)  # (N, H, W, C)
        logger.info(
            "[Gemma4VideoTrace] stage=decode_sampled output_shape=%s dtype=%s "
            "layout=NHWC device=cpu_numpy",
            tuple(frames_np.shape),
            frames_np.dtype,
        )
        frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()
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
