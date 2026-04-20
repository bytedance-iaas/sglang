# Copyright 2023-2024 SGLang Team
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
"""
The definition of objects transferred between different
processes (TokenizerManager, DetokenizerManager, Scheduler).
"""

from __future__ import annotations

import copy
import uuid
from abc import ABC
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Literal, Optional, Union

import msgspec
import msgspec.msgpack

from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.mm_utils import has_valid_data
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
    SchedulerReqTimeStatsIPC,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import ImageData

# Handle serialization of Image for pydantic
if TYPE_CHECKING:
    from PIL.Image import Image
else:
    Image = Any


@dataclass
class BaseReq(ABC):
    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)
    http_worker_ipc: Optional[str] = field(default=None, kw_only=True)

    def regenerate_rid(self):
        """Generate a new request ID and return it."""
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid

    def _validate_rid_uniqueness(self):
        """Validate that request IDs within a batch are unique."""
        if isinstance(self.rid, list) and len(set(self.rid)) != len(self.rid):
            counts = Counter(self.rid)
            duplicates = [rid for rid, count in counts.items() if count > 1]
            raise ValueError(
                f"Duplicate request IDs detected within the request: {duplicates}"
            )


@dataclass
class BaseBatchReq(ABC):
    rids: Optional[List[str]] = field(default=None, kw_only=True)
    http_worker_ipcs: Optional[List[str]] = field(default=None, kw_only=True)

    def regenerate_rids(self):
        """Generate new request IDs and return them."""
        self.rids = [uuid.uuid4().hex for _ in range(len(self.rids))]
        return self.rids


@dataclass
class SpeculativeDecodingMetricsMixin:
    """
    Mixin class containing speculative decoding metrics.

    This class consolidates speculative decoding metrics that are shared across
    batch output types that support speculative decoding to avoid code duplication.
    """

    # Verify count: number of verification forward passes
    spec_verify_ct: List[int]

    # Accepted tokens: Number of accepted tokens during speculative decoding
    spec_accepted_tokens: List[int]

    # Acceptance histogram: List of lists, where each inner list represents histogram counts.
    # List index = number of accepted tokens in a step, List value = count of steps with that many accepted tokens.
    # Example: histogram[0] = 5 means 5 steps with 0 accepted tokens, histogram[3] = 10 means 10 steps with 3 accepted tokens.
    # Empty list [] when speculative decoding is disabled.
    spec_acceptance_histogram: List[List[int]]


# Parameters for a session
@dataclass
class SessionParams(msgspec.Struct, tag=True):
    id: Optional[str] = None
    rid: Optional[str] = None
    offset: Optional[int] = None
    replace: Optional[bool] = None
    drop_previous_output: Optional[bool] = None


# Type definitions for multimodal input data
# Individual data item types for each modality
ImageDataInputItem = Union[Image, str, ImageData, Dict]
AudioDataInputItem = Union[str, Dict]
VideoDataInputItem = Union[str, Dict]
# Union type for any multimodal data item
MultimodalDataInputItem = Union[
    ImageDataInputItem, VideoDataInputItem, AudioDataInputItem
]
# Format types supporting single items, lists, or nested lists for batch processing
MultimodalDataInputFormat = Union[
    List[List[MultimodalDataInputItem]],
    List[MultimodalDataInputItem],
    MultimodalDataInputItem,
]


@dataclass
class GenerateReqInput(BaseReq):
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can specify either text or input_ids
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The embeddings for input_ids; one can specify either text or input_ids or input_embeds.
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
    # Embedding overrides to place at specific token positions.
    # Runtime type: Optional[Union[PositionalEmbeds, List[Optional[PositionalEmbeds]]]]
    # Typed as Any to avoid Pydantic/FastAPI schema errors (PositionalEmbeds contains torch.Tensor).
    positional_embed_overrides: Any = None
    # The image input. It can be an image instance, file name, URL, or base64 encoded string.
    # Can be formatted as:
    # - Single image for a single request
    # - List of images (one per request in a batch)
    # - List of lists of images (multiple images per request)
    # See also python/sglang/srt/utils.py:load_image for more details.
    image_data: Optional[MultimodalDataInputFormat] = None
    # The video input. Like image data, it can be a file name, a url, or base64 encoded string.
    video_data: Optional[MultimodalDataInputFormat] = None
    # The audio input. Like image data, it can be a file name, a url, or base64 encoded string.
    audio_data: Optional[MultimodalDataInputFormat] = None
    # The sampling_params. See descriptions below.
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    # Whether to return logprobs.
    return_logprob: Optional[Union[List[bool], bool]] = None
    # If return logprobs, the start location in the prompt for returning logprobs.
    # By default, this value is "-1", which means it will only return logprobs for output tokens.
    logprob_start_len: Optional[Union[List[int], int]] = None
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: Optional[Union[List[int], int]] = None
    # If return logprobs, the token ids to return logprob for.
    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None
    # Whether to detokenize tokens in text in the returned logprobs.
    return_text_in_logprobs: bool = False
    # Whether to stream output.
    stream: bool = False
    # Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    log_metrics: bool = True
    # Whether to return hidden states
    return_hidden_states: Union[List[bool], bool] = False
    # Whether to return captured routed experts
    return_routed_experts: bool = False
    # The start location in the prompt for returning routed experts.
    routed_experts_start_len: int = 0

    # The modalities of the image data [image, multi-images, video]
    modalities: Optional[List[str]] = None
    # Session info for continual prompting
    session_params: Optional[Union[List[Dict], Dict]] = None

    # The path to the LoRA adaptors
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
    # The uid of LoRA adaptors, should be initialized by tokenizer manager
    lora_id: Optional[Union[List[Optional[str]], Optional[str]]] = None

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None

    # For disaggregated inference
    bootstrap_host: Optional[Union[List[str], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[int], int]] = None
    bootstrap_pair_key: Optional[Union[List[str], str]] = None
    decode_tp_size: Optional[Union[List[Optional[int]], int]] = None

    # Require reasoning for the request (hybrid reasoning model only)
    require_reasoning: bool = False

    # For DP routing — external router assigns a specific DP worker
    routed_dp_rank: Optional[int] = None
    # For PD disagg — hint telling decode which prefill DP worker has the KV cache
    disagg_prefill_dp_rank: Optional[int] = None
    # Deprecated: use routed_dp_rank instead
    data_parallel_rank: Optional[int] = None

    # For background responses (OpenAI responses API)
    background: bool = False

    # Conversation id used for tracking requests
    conversation_id: Optional[str] = None

    # Priority for the request
    priority: Optional[int] = None

    # Extra key for classifying the request (e.g. cache_salt)
    extra_key: Optional[Union[List[str], str]] = None

    # Routing key for routing-key schedule policy
    routing_key: Optional[str] = None

    # Whether to disallow logging for this request (e.g. due to ZDR)
    no_logs: bool = False

    # For custom metric labels
    custom_labels: Optional[Dict[str, str]] = None

    # (Internal) Whether to return bytes for image generation
    return_bytes: bool = False

    # Whether to return entropy
    return_entropy: bool = False

    # Propagates trace context via Engine.generate/async_generate
    external_trace_header: Optional[Dict] = None
    received_time: Optional[float] = None

    # For EPD-disaggregated inference
    need_wait_for_mm_inputs: Optional[bool] = None
    num_items_assigned: Optional[Dict[Modality, List[int]]] = None

    # Multimodal tiling controls (extensions)
    max_dynamic_patch: Optional[int] = None
    min_dynamic_patch: Optional[int] = None
    image_max_dynamic_patch: Optional[int] = None
    video_max_dynamic_patch: Optional[int] = None

    def contains_mm_input(self) -> bool:
        return (
            has_valid_data(self.image_data)
            or has_valid_data(self.video_data)
            or has_valid_data(self.audio_data)
        )

    def normalize_batch_and_arguments(self):
        """
        Normalize the batch size and arguments for the request.

        This method resolves various input formats and ensures all parameters
        are properly formatted as either single values or batches depending on the input.
        It also handles parallel sampling expansion and sets default values for
        unspecified parameters.

        Raises:
            ValueError: If inputs are not properly specified (e.g., none or all of
                       text, input_ids, input_embeds are provided)
        """
        if self.data_parallel_rank is not None:
            import warnings

            warnings.warn(
                "'data_parallel_rank' is deprecated, use 'routed_dp_rank' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.routed_dp_rank is None:
                self.routed_dp_rank = self.data_parallel_rank
            self.data_parallel_rank = None

        self._validate_inputs()
        self._determine_batch_size()
        self._handle_parallel_sampling()

        if self.is_single:
            self._normalize_single_inputs()
        else:
            self._normalize_batch_inputs()

        self._validate_rid_uniqueness()

    def _validate_inputs(self):
        """Validate that the input configuration is valid."""
        if (
            self.text is None and self.input_ids is None and self.input_embeds is None
        ) or (
            self.text is not None
            and self.input_ids is not None
            and self.input_embeds is not None
        ):
            raise ValueError(
                "Either text, input_ids or input_embeds should be provided."
            )

    def _determine_batch_size(self):
        """Determine if this is a single example or a batch and the batch size."""
        if self.text is not None:
            if isinstance(self.text, str):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.text)
            self.input_embeds = None
        elif self.input_ids is not None:
            if len(self.input_ids) == 0:
                raise ValueError("input_ids cannot be empty.")
            if isinstance(self.input_ids[0], int):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_ids)
            self.input_embeds = None
        else:
            if isinstance(self.input_embeds[0][0], float):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_embeds)

    def _handle_parallel_sampling(self):
        """Handle parallel sampling parameters and adjust batch size if needed."""
        # Determine parallel sample count
        if self.sampling_params is None:
            self.parallel_sample_num = 1
            return
        elif isinstance(self.sampling_params, dict):
            self.parallel_sample_num = self.sampling_params.get("n", 1)
        else:  # isinstance(self.sampling_params, list):
            self.parallel_sample_num = self.sampling_params[0].get("n", 1)
            for sampling_params in self.sampling_params:
                if self.parallel_sample_num != sampling_params.get("n", 1):
                    raise ValueError(
                        "The parallel_sample_num should be the same for all samples in sample params."
                    )

        # If using parallel sampling with a single example, convert to batch
        if self.parallel_sample_num > 1 and self.is_single:
            self.is_single = False
            if self.text is not None:
                self.text = [self.text]
            if self.input_ids is not None:
                self.input_ids = [self.input_ids]
            if self.input_embeds is not None:
                self.input_embeds = [self.input_embeds]

    def _normalize_single_inputs(self):
        """Normalize inputs for a single example."""
        if self.sampling_params is None:
            self.sampling_params = {}
        if self.rid is None:
            self.rid = uuid.uuid4().hex
        if self.return_logprob is None:
            self.return_logprob = False
        if self.logprob_start_len is None:
            self.logprob_start_len = -1
        if self.top_logprobs_num is None:
            self.top_logprobs_num = 0
        if not self.token_ids_logprob:  # covers both None and []
            self.token_ids_logprob = None

    def _normalize_batch_inputs(self):
        """Normalize inputs for a batch of examples, including parallel sampling expansion."""
        # Calculate expanded batch size
        if self.parallel_sample_num == 1:
            num = self.batch_size
        else:
            # Expand parallel_sample_num
            num = self.batch_size * self.parallel_sample_num

        # Expand input based on type
        self._expand_inputs(num)
        self._normalize_rid(num)
        self._normalize_lora_paths(num)
        self._normalize_image_data(num)
        self._normalize_video_data(num)
        self._normalize_audio_data(num)
        self._normalize_sampling_params(num)
        self._normalize_logprob_params(num)
        self._normalize_custom_logit_processor(num)
        self._normalize_bootstrap_params(num)

    def _expand_inputs(self, num):
        """Expand the main inputs (text, input_ids, input_embeds) for parallel sampling."""
        if self.text is not None:
            if not isinstance(self.text, list):
                raise ValueError("Text should be a list for batch processing.")
            self.text = self.text * self.parallel_sample_num
        elif self.input_ids is not None:
            if not isinstance(self.input_ids, list) or not isinstance(
                self.input_ids[0], list
            ):
                raise ValueError(
                    "input_ids should be a list of lists for batch processing."
                )
            self.input_ids = self.input_ids * self.parallel_sample_num
        elif self.input_embeds is not None:
            if not isinstance(self.input_embeds, list):
                raise ValueError("input_embeds should be a list for batch processing.")
            self.input_embeds = self.input_embeds * self.parallel_sample_num

    def _normalize_lora_paths(self, num):
        """Normalize LoRA paths for batch processing."""
        if self.lora_path is not None:
            if isinstance(self.lora_path, str):
                self.lora_path = [self.lora_path] * num
            elif isinstance(self.lora_path, list):
                self.lora_path = self.lora_path * self.parallel_sample_num
            else:
                raise ValueError("lora_path should be a list or a string.")

    def _normalize_image_data(self, num):
        """Normalize image data for batch processing."""
        if self.image_data is None:
            self.image_data = [None] * num
        elif not isinstance(self.image_data, list):
            # Single image, convert to list of single-image lists
            self.image_data = [[self.image_data]] * num
            self.modalities = ["image"] * num
        elif isinstance(self.image_data, list):
            # Handle empty list case - treat as no images
            if len(self.image_data) == 0:
                self.image_data = [None] * num
                return

            if len(self.image_data) != self.batch_size:
                raise ValueError(
                    "The length of image_data should be equal to the batch size."
                )

            self.modalities = []
            if len(self.image_data) > 0 and isinstance(self.image_data[0], list):
                # Already a list of lists, keep as is
                for i in range(len(self.image_data)):
                    if self.image_data[i] is None or self.image_data[i] == [None]:
                        self.modalities.append(None)
                    elif len(self.image_data[i]) == 1:
                        self.modalities.append("image")
                    elif len(self.image_data[i]) > 1:
                        self.modalities.append("multi-images")
                    else:
                        # Ensure len(self.modalities) == len(self.image_data)
                        self.modalities.append(None)
                # Expand parallel_sample_num
                self.image_data = self.image_data * self.parallel_sample_num
                self.modalities = self.modalities * self.parallel_sample_num
            else:
                # List of images for a batch, wrap each in a list
                wrapped_images = [[img] for img in self.image_data]
                # Expand for parallel sampling
                self.image_data = wrapped_images * self.parallel_sample_num
                self.modalities = ["image"] * num

    def _normalize_video_data(self, num):
        """Normalize video data for batch processing."""
        if self.video_data is None:
            self.video_data = [None] * num
        elif not isinstance(self.video_data, list):
            self.video_data = [self.video_data] * num
        elif isinstance(self.video_data, list):
            self.video_data = self.video_data * self.parallel_sample_num

    def _normalize_audio_data(self, num):
        """Normalize audio data for batch processing."""
        if self.audio_data is None:
            self.audio_data = [None] * num
        elif not isinstance(self.audio_data, list):
            self.audio_data = [self.audio_data] * num
        elif isinstance(self.audio_data, list):
            self.audio_data = self.audio_data * self.parallel_sample_num

    def _normalize_sampling_params(self, num):
        """Normalize sampling parameters for batch processing."""
        if self.sampling_params is None:
            self.sampling_params = [{}] * num
        elif isinstance(self.sampling_params, dict):
            self.sampling_params = [self.sampling_params] * num
        else:  # Already a list
            self.sampling_params = self.sampling_params * self.parallel_sample_num

    def _normalize_rid(self, num):
        """Normalize request IDs for batch processing."""
        if self.rid is None:
            self.rid = [uuid.uuid4().hex for _ in range(num)]
        elif isinstance(self.rid, str):
            new_rids = [f"{self.rid}_{i}" for i in range(num)]
            self.rid = new_rids
        elif isinstance(self.rid, list):
            # Note: the length of rid shall be the same as the batch_size,
            # as the rid would be expanded for parallel sampling in tokenizer_manager
            if len(self.rid) != self.batch_size:
                raise ValueError(
                    "The specified rids length mismatch with the batch_size for batch processing."
                )
        else:
            raise ValueError("The rid should be a string or a list of strings.")

    def _normalize_logprob_params(self, num):
        """Normalize logprob-related parameters for batch processing."""

        # Helper function to normalize a parameter
        def normalize_param(param, default_value, param_name):
            if param is None:
                return [default_value] * num
            elif not isinstance(param, list):
                return [param] * num
            else:
                if self.parallel_sample_num > 1:
                    raise ValueError(
                        f"Cannot use list {param_name} with parallel_sample_num > 1"
                    )
                return param

        # Normalize each logprob parameter
        self.return_logprob = normalize_param(
            self.return_logprob, False, "return_logprob"
        )
        self.logprob_start_len = normalize_param(
            self.logprob_start_len, -1, "logprob_start_len"
        )
        self.top_logprobs_num = normalize_param(
            self.top_logprobs_num, 0, "top_logprobs_num"
        )

        # Handle token_ids_logprob specially due to its nested structure
        if not self.token_ids_logprob:  # covers both None and []
            self.token_ids_logprob = [None] * num
        elif not isinstance(self.token_ids_logprob, list):
            self.token_ids_logprob = [[self.token_ids_logprob] for _ in range(num)]
        elif not isinstance(self.token_ids_logprob[0], list):
            self.token_ids_logprob = [
                copy.deepcopy(self.token_ids_logprob) for _ in range(num)
            ]
        elif self.parallel_sample_num > 1:
            raise ValueError(
                "Cannot use list token_ids_logprob with parallel_sample_num > 1"
            )

    def _normalize_custom_logit_processor(self, num):
        """Normalize custom logit processor for batch processing."""
        if self.custom_logit_processor is None:
            self.custom_logit_processor = [None] * num
        elif not isinstance(self.custom_logit_processor, list):
            self.custom_logit_processor = [self.custom_logit_processor] * num
        elif self.parallel_sample_num > 1:
            raise ValueError(
                "Cannot use list custom_logit_processor with parallel_sample_num > 1"
            )

    def _normalize_bootstrap_params(self, num):
        """Normalize bootstrap parameters for batch processing."""
        # Normalize bootstrap_host
        if self.bootstrap_host is None:
            self.bootstrap_host = [None] * num
        elif not isinstance(self.bootstrap_host, list):
            self.bootstrap_host = [self.bootstrap_host] * num
        elif isinstance(self.bootstrap_host, list):
            self.bootstrap_host = self.bootstrap_host * self.parallel_sample_num

        # Normalize bootstrap_port
        if self.bootstrap_port is None:
            self.bootstrap_port = [None] * num
        elif not isinstance(self.bootstrap_port, list):
            self.bootstrap_port = [self.bootstrap_port] * num
        elif isinstance(self.bootstrap_port, list):
            self.bootstrap_port = self.bootstrap_port * self.parallel_sample_num

        # Normalize bootstrap_room
        if self.bootstrap_room is None:
            self.bootstrap_room = [None] * num
        elif not isinstance(self.bootstrap_room, list):
            self.bootstrap_room = [self.bootstrap_room + i for i in range(num)]
        elif isinstance(self.bootstrap_room, list):
            self.bootstrap_room = self.bootstrap_room * self.parallel_sample_num

        # Normalize bootstrap_pair_key
        if self.bootstrap_pair_key is None:
            self.bootstrap_pair_key = [None] * num
        elif not isinstance(self.bootstrap_pair_key, list):
            self.bootstrap_pair_key = [self.bootstrap_pair_key] * num
        elif isinstance(self.bootstrap_pair_key, list):
            self.bootstrap_pair_key = self.bootstrap_pair_key * self.parallel_sample_num

    def _validate_session_params(self):
        """Validate that session parameters are properly formatted."""
        if self.session_params is not None:
            if not isinstance(self.session_params, dict) and not isinstance(
                self.session_params[0], dict
            ):
                raise ValueError("Session params must be a dict or a list of dicts.")

    def _get_positional_embed_overrides_item(
        self, i: int
    ) -> Optional[PositionalEmbeds]:
        """Extract the i-th item from positional_embed_overrides."""
        if self.positional_embed_overrides is None:
            return None
        if isinstance(self.positional_embed_overrides, PositionalEmbeds):
            return self.positional_embed_overrides
        return self.positional_embed_overrides[i]

    def __getitem__(self, i):
        # Cache sub-objects so that repeated obj[i] calls return the same instance.
        # This avoids subtle bugs where different call sites get divergent objects.
        cache = self.__dict__.setdefault("_sub_obj_cache", {})
        if i in cache:
            return cache[i]
        sub = GenerateReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            input_embeds=(
                self.input_embeds[i] if self.input_embeds is not None else None
            ),
            positional_embed_overrides=self._get_positional_embed_overrides_item(i),
            image_data=self.image_data[i],
            video_data=self.video_data[i],
            audio_data=self.audio_data[i],
            sampling_params=self.sampling_params[i],
            rid=self.rid[i],
            return_logprob=self.return_logprob[i],
            logprob_start_len=self.logprob_start_len[i],
            top_logprobs_num=self.top_logprobs_num[i],
            token_ids_logprob=self.token_ids_logprob[i],
            return_text_in_logprobs=self.return_text_in_logprobs,
            stream=self.stream,
            log_metrics=self.log_metrics,
            return_hidden_states=(
                self.return_hidden_states[i]
                if isinstance(self.return_hidden_states, list)
                else self.return_hidden_states
            ),
            return_routed_experts=self.return_routed_experts,
            modalities=self.modalities[i] if self.modalities else None,
            session_params=self.session_params,
            lora_path=self.lora_path[i] if self.lora_path is not None else None,
            lora_id=self.lora_id[i] if self.lora_id is not None else None,
            custom_logit_processor=(
                self.custom_logit_processor[i]
                if self.custom_logit_processor is not None
                else None
            ),
            # if `__getitem__` is called, the bootstrap_host, bootstrap_port, bootstrap_room must be a list
            bootstrap_host=(
                self.bootstrap_host[i] if self.bootstrap_host is not None else None
            ),
            bootstrap_port=(
                self.bootstrap_port[i] if self.bootstrap_port is not None else None
            ),
            bootstrap_room=(
                self.bootstrap_room[i] if self.bootstrap_room is not None else None
            ),
            bootstrap_pair_key=(
                self.bootstrap_pair_key[i]
                if self.bootstrap_pair_key is not None
                else None
            ),
            decode_tp_size=(
                self.decode_tp_size[i] if self.decode_tp_size is not None else None
            ),
            routed_dp_rank=self.routed_dp_rank,
            disagg_prefill_dp_rank=self.disagg_prefill_dp_rank,
            conversation_id=self.conversation_id,
            priority=self.priority,
            extra_key=self.extra_key,
            no_logs=self.no_logs,
            custom_labels=self.custom_labels,
            return_bytes=self.return_bytes,
            return_entropy=self.return_entropy,
            external_trace_header=self.external_trace_header,
            http_worker_ipc=self.http_worker_ipc,
            received_time=self.received_time,
        )
        cache[i] = sub
        return sub


@dataclass
class TokenizedGenerateReqInput(BaseReq):
    # The input text
    input_text: str
    # The input token ids
    input_ids: List[int]
    # The multimodal inputs
    mm_inputs: object
    # The sampling parameters
    sampling_params: SamplingParams
    # Whether to return the logprobs
    return_logprob: bool
    # If return logprobs, the start location in the prompt for returning logprobs.
    logprob_start_len: int
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: int
    # If return logprobs, the token id to return logprob for
    token_ids_logprob: List[int]
    # Whether to stream output
    stream: bool

    # Whether to return hidden states
    return_hidden_states: bool = False

    # Whether to return captured routed experts
    return_routed_experts: bool = False
    # The start location in the prompt for returning routed experts.
    routed_experts_start_len: int = 0

    # The input embeds
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None

    # Embedding overrides to place at specific token positions.
    positional_embed_overrides: Optional[PositionalEmbeds] = None

    # Session info for continual prompting
    session_params: Optional[SessionParams] = None

    # LoRA related
    lora_id: Optional[str] = None  # None means just use the base model

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: Optional[str] = None

    # For disaggregated inference
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None
    bootstrap_room: Optional[int] = None
    bootstrap_pair_key: Optional[str] = None
    decode_tp_size: Optional[int] = None

    # Require reasoning for the request (hybrid reasoning model only)
    require_reasoning: bool = False

    # For DP routing
    routed_dp_rank: Optional[int] = None
    # For PD disagg — hint telling decode which prefill DP worker has the KV cache
    disagg_prefill_dp_rank: Optional[int] = None

    # Priority for the request
    priority: Optional[int] = None

    # Extra key for classifying the request (e.g. cache_salt)
    extra_key: Optional[str] = None

    # Routing key for routing-key schedule policy
    routing_key: Optional[str] = None

    # Whether to disallow logging for this request (e.g. due to ZDR)
    no_logs: bool = False

    # (Internal) Whether to return bytes for image generation
    return_bytes: bool = False

    # Whether to return entropy
    return_entropy: bool = False

    token_type_ids: Optional[List[int]] = None

    need_wait_for_mm_inputs: bool = False
    num_items_assigned: Optional[Dict[Modality, List[int]]] = None

    # For observability
    time_stats: Optional[Union[APIServerReqTimeStats, DPControllerReqTimeStats]] = None


@dataclass
class BatchTokenizedGenerateReqInput(BaseBatchReq):
    # The batch of tokenized requests
    batch: List[TokenizedGenerateReqInput]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


@dataclass
class EmbeddingReqInput(BaseReq):
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[List[str]], List[str], str]] = None
    # The image input. It can be an image instance, file name, URL, or base64 encoded string.
    # Can be formatted as:
    # - Single image for a single request
    # - List of images (one per request in a batch)
    # - List of lists of images (multiple images per request)
    # See also python/sglang/srt/utils.py:load_image for more details.
    image_data: Optional[MultimodalDataInputFormat] = None
    # The video input. Like image data, it can be a file name, a url, or base64 encoded string.
    video_data: Optional[MultimodalDataInputFormat] = None
    # The audio input. Like image data, it can be a file name, a url, or base64 encoded string.
    audio_data: Optional[MultimodalDataInputFormat] = None
    # The token ids for text; one can either specify text or input_ids.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # Placeholder token ID used to locate embedding override positions in input token IDs.
    embed_override_token_id: Optional[int] = None
    # Unresolved embedding overrides: per-input list of tensors.
    # Position resolution happens in the tokenizer manager after tokenization.
    # Shape: [num_inputs][num_replacements] where each entry is a torch.Tensor of [hidden_size].
    # Per-input entry may be None when only some inputs in a batch need overrides.
    # Runtime type: Optional[List[Optional[List[torch.Tensor]]]]
    # Typed as Any to avoid Pydantic/FastAPI schema errors (contains torch.Tensor).
    embed_overrides: Any = None
    # Resolved embedding overrides with positions (set by tokenizer manager or score mixin).
    # Runtime type: Optional[Union[PositionalEmbeds, List[Optional[PositionalEmbeds]]]]
    positional_embed_overrides: Any = None
    # Dummy sampling params for compatibility
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    # Dummy input embeds for compatibility
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
    # Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    log_metrics: bool = True
    # The modalities of the image data [image, multi-images, video]
    modalities: Optional[List[str]] = None
    # For cross-encoder requests
    is_cross_encoder_request: bool = False
    # Priority for the request
    priority: Optional[int] = None
    # Routing key for routing-key schedule policy
    routing_key: Optional[str] = None

    # For background responses (OpenAI responses API)
    background: bool = False

    # Propagates trace context via Engine.encode/async_encode
    external_trace_header: Optional[Dict] = None
    received_time: Optional[float] = None

    # The number of dimensions the resulting output embeddings should have. It is applicable for Matryoshka Embeddings.
    dimensions: Optional[int] = None

    # The path to the LoRA adaptors
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
    # The uid of LoRA adaptors, should be initialized by tokenizer manager
    lora_id: Optional[Union[List[Optional[str]], Optional[str]]] = None

    def normalize_batch_and_arguments(self):
        # at least one of text, input_ids, or image should be provided
        if self.text is None and self.input_ids is None and self.image_data is None:
            raise ValueError(
                "At least one of text, input_ids, or image should be provided"
            )

        # text and input_ids cannot be provided at the same time
        if self.text is not None and self.input_ids is not None:
            raise ValueError("text and input_ids cannot be provided at the same time")

        # Derive the batch size
        self.batch_size = 0
        self.is_single = True

        # check the batch size of text
        if self.text is not None:
            if isinstance(self.text, list):
                self.batch_size += len(self.text)
                self.is_single = False
            else:
                self.batch_size += 1

        # check the batch size of input_ids
        if self.input_ids is not None:
            if isinstance(self.input_ids[0], list):
                self.batch_size += len(self.input_ids)
                self.is_single = False
            else:
                self.batch_size += 1

        # Fill in default arguments
        if self.is_single:
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.sampling_params is None:
                self.sampling_params = {}
            self.sampling_params["max_new_tokens"] = 0
        else:
            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(self.batch_size)]
            else:
                assert isinstance(self.rid, list), "The rid should be a list."

            if self.sampling_params is None:
                self.sampling_params = [{}] * self.batch_size
            elif isinstance(self.sampling_params, dict):
                self.sampling_params = [self.sampling_params] * self.batch_size
            for i in range(self.batch_size):
                self.sampling_params[i]["max_new_tokens"] = 0

            self._normalize_lora_paths(self.batch_size)

        self._validate_rid_uniqueness()

    def _normalize_lora_paths(self, num):
        """Normalize LoRA paths for batch processing."""
        if self.lora_path is not None:
            if isinstance(self.lora_path, str):
                self.lora_path = [self.lora_path] * num
            elif isinstance(self.lora_path, list):
                if len(self.lora_path) != num:
                    raise ValueError(
                        f"lora_path list length ({len(self.lora_path)}) must match batch size ({num})"
                    )
            else:
                raise ValueError("lora_path should be a list or a string.")

    def contains_mm_input(self) -> bool:
        return (
            has_valid_data(self.image_data)
            or has_valid_data(self.video_data)
            or has_valid_data(self.audio_data)
        )

    def _get_positional_embed_overrides_item(
        self, i: int
    ) -> Optional[PositionalEmbeds]:
        """Extract the i-th item from positional_embed_overrides."""
        if self.positional_embed_overrides is None:
            return None
        if isinstance(self.positional_embed_overrides, PositionalEmbeds):
            return self.positional_embed_overrides
        return self.positional_embed_overrides[i]

    def __getitem__(self, i):
        # Cache sub-objects so that repeated obj[i] calls return the same instance.
        cache = self.__dict__.setdefault("_sub_obj_cache", {})
        if i in cache:
            return cache[i]

        if self.is_cross_encoder_request:
            sub = EmbeddingReqInput(
                text=[self.text[i]] if self.text is not None else None,
                positional_embed_overrides=self._get_positional_embed_overrides_item(i),
                sampling_params=self.sampling_params[i],
                rid=self.rid[i],
                lora_path=self.lora_path[i] if self.lora_path is not None else None,
                lora_id=self.lora_id[i] if self.lora_id is not None else None,
                is_cross_encoder_request=True,
                http_worker_ipc=self.http_worker_ipc,
            )
        else:
            sub = EmbeddingReqInput(
                text=self.text[i] if self.text is not None else None,
                input_ids=self.input_ids[i] if self.input_ids is not None else None,
                embed_override_token_id=self.embed_override_token_id,
                embed_overrides=(
                    self.embed_overrides[i]
                    if self.embed_overrides is not None
                    else None
                ),
                positional_embed_overrides=self._get_positional_embed_overrides_item(i),
                image_data=self.image_data[i] if self.image_data is not None else None,
                audio_data=self.audio_data[i] if self.audio_data is not None else None,
                video_data=self.video_data[i] if self.video_data is not None else None,
                sampling_params=self.sampling_params[i],
                rid=self.rid[i],
                lora_path=self.lora_path[i] if self.lora_path is not None else None,
                lora_id=self.lora_id[i] if self.lora_id is not None else None,
                external_trace_header=self.external_trace_header,
                dimensions=self.dimensions,
                http_worker_ipc=self.http_worker_ipc,
                received_time=self.received_time,
            )
        cache[i] = sub
        return sub


@dataclass
class TokenizedEmbeddingReqInput(BaseReq):
    # The input text
    input_text: str
    # The input token ids
    input_ids: List[int]
    # The image inputs
    image_inputs: dict
    # The token type ids
    token_type_ids: List[int]
    # Dummy sampling params for compatibility
    sampling_params: SamplingParams
    # Embedding overrides to place at specific token positions.
    positional_embed_overrides: Optional[PositionalEmbeds] = None
    # For DP routing
    routed_dp_rank: Optional[int] = None
    # Priority for the request
    priority: Optional[int] = None
    # The number of dimensions the resulting output embeddings should have. It is applicable for Matryoshka Embeddings.
    dimensions: Optional[int] = None

    # LoRA related
    lora_id: Optional[str] = None  # None means just use the base model
    # For observability
    time_stats: Optional[Union[APIServerReqTimeStats, DPControllerReqTimeStats]] = None


@dataclass
class BatchTokenizedEmbeddingReqInput(BaseBatchReq):
    # The batch of tokenized embedding requests
    batch: List[TokenizedEmbeddingReqInput]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


class BatchTokenIDOutput(msgspec.Struct, tag=True):
    # BaseBatchReq fields (inlined)
    rids: Optional[List[str]] = None
    http_worker_ipcs: Optional[List[Optional[str]]] = None

    # Speculative decoding metrics (inlined from SpeculativeDecodingMetricsMixin)
    spec_verify_ct: Optional[List[int]] = None
    spec_accepted_tokens: Optional[List[int]] = None
    spec_acceptance_histogram: Optional[List[List[int]]] = None

    # The finish reason (already dict at runtime via .to_json())
    finished_reasons: Optional[List[Optional[dict]]] = None
    # For incremental decoding
    decoded_texts: Optional[List[str]] = None
    decode_ids: Optional[List[List[int]]] = None
    read_offsets: Optional[List[int]] = None
    # Only used when `--skip-tokenizer-init` is on
    output_ids: Optional[List[List[int]]] = None
    # Detokenization configs
    skip_special_tokens: Optional[List[bool]] = None
    spaces_between_special_tokens: Optional[List[bool]] = None
    no_stop_trim: Optional[List[bool]] = None

    # Token counts
    prompt_tokens: Optional[List[int]] = None
    reasoning_tokens: Optional[List[int]] = None
    completion_tokens: Optional[List[int]] = None
    cached_tokens: Optional[List[int]] = None

    # Logprobs (each is a list-per-request, elements can be None)
    input_token_logprobs_val: Optional[list] = None
    input_token_logprobs_idx: Optional[list] = None
    output_token_logprobs_val: Optional[list] = None
    output_token_logprobs_idx: Optional[list] = None
    input_top_logprobs_val: Optional[list] = None
    input_top_logprobs_idx: Optional[list] = None
    output_top_logprobs_val: Optional[list] = None
    output_top_logprobs_idx: Optional[list] = None
    input_token_ids_logprobs_val: Optional[list] = None
    input_token_ids_logprobs_idx: Optional[list] = None
    output_token_ids_logprobs_val: Optional[list] = None
    output_token_ids_logprobs_idx: Optional[list] = None
    output_token_entropy_val: Optional[list] = None

    # Hidden states
    output_hidden_states: Optional[list] = None

    # Pre-serialized routed_experts (pickle bytes of List[Optional[torch.Tensor]])
    routed_experts: Optional[bytes] = None

    # The information of placeholder tokens (e.g., image token)
    placeholder_tokens_idx: Optional[List[Optional[List[int]]]] = None
    placeholder_tokens_val: Optional[List[Optional[List[int]]]] = None

    # Number of times each request was retracted.
    retraction_counts: Optional[List[int]] = None

    # The trainer step id. Used to know which step's weights are used for sampling.
    token_steps: Optional[List[List[int]]] = None

    # Load for DP balance
    load: Optional[GetLoadReqOutput] = None
    # Customized info (pre-serialized as pickle bytes)
    customized_info: Optional[bytes] = None
    # Detailed breakdown of cached tokens by source (device/host/storage)
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None
    # DP rank of the scheduler that processed each request
    dp_ranks: Optional[List[Optional[int]]] = None

    # For observability
    time_stats: Optional[List[SchedulerReqTimeStatsIPC]] = None


class BatchStrOutput(msgspec.Struct, tag=True):
    # BaseBatchReq fields (inlined)
    rids: Optional[List[str]] = None
    http_worker_ipcs: Optional[List[Optional[str]]] = None

    # Speculative decoding metrics (inlined)
    spec_verify_ct: Optional[List[int]] = None
    spec_accepted_tokens: Optional[List[int]] = None
    spec_acceptance_histogram: Optional[List[List[int]]] = None

    # The finish reason
    finished_reasons: Optional[List[Optional[dict]]] = None
    # The output decoded strings
    output_strs: Optional[List[str]] = None
    # The token ids
    output_ids: Optional[List[List[int]]] = None

    # Token counts
    prompt_tokens: Optional[List[int]] = None
    completion_tokens: Optional[List[int]] = None
    reasoning_tokens: Optional[List[int]] = None
    cached_tokens: Optional[List[int]] = None

    # Logprobs (each is a list-per-request, elements can be None)
    input_token_logprobs_val: Optional[list] = None
    input_token_logprobs_idx: Optional[list] = None
    output_token_logprobs_val: Optional[list] = None
    output_token_logprobs_idx: Optional[list] = None
    input_top_logprobs_val: Optional[list] = None
    input_top_logprobs_idx: Optional[list] = None
    output_top_logprobs_val: Optional[list] = None
    output_top_logprobs_idx: Optional[list] = None
    input_token_ids_logprobs_val: Optional[list] = None
    input_token_ids_logprobs_idx: Optional[list] = None
    output_token_ids_logprobs_val: Optional[list] = None
    output_token_ids_logprobs_idx: Optional[list] = None
    output_token_entropy_val: Optional[list] = None

    # Hidden states
    output_hidden_states: Optional[list] = None

    # Pre-serialized routed_experts (pickle bytes of List[Optional[torch.Tensor]])
    routed_experts: Optional[bytes] = None

    # The information of placeholder tokens (e.g., image token)
    placeholder_tokens_idx: Optional[List[Optional[List[int]]]] = None
    placeholder_tokens_val: Optional[List[Optional[List[int]]]] = None

    # Number of times each request was retracted.
    retraction_counts: Optional[List[int]] = None

    # The trainer step id. Used to know which step's weights are used for sampling.
    token_steps: Optional[List[List[int]]] = None

    # Load for DP balance
    load: Optional[GetLoadReqOutput] = None

    # Customized info (pre-serialized as pickle bytes)
    customized_info: Optional[bytes] = None
    # Detailed breakdown of cached tokens by source (device/host/storage)
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None
    # DP rank of the scheduler that processed each request
    dp_ranks: Optional[List[Optional[int]]] = None

    # For observability
    time_stats: Optional[List[SchedulerReqTimeStatsIPC]] = None


class BatchEmbeddingOutput(msgspec.Struct, tag=True):
    # BaseBatchReq fields (inlined)
    rids: Optional[List[str]] = None
    http_worker_ipcs: Optional[List[Optional[str]]] = None

    # The finish reason (already dict at runtime via .to_json())
    finished_reasons: Optional[List[Optional[dict]]] = None
    # The output embedding (List[List[float]] or List[Dict[int, float]])
    embeddings: Optional[list] = None
    # Token counts
    prompt_tokens: Optional[List[int]] = None
    cached_tokens: Optional[List[int]] = None
    # Placeholder token info
    placeholder_tokens_idx: Optional[List[Optional[List[int]]]] = None
    placeholder_tokens_val: Optional[List[Optional[List[int]]]] = None

    # Number of times each request was retracted.
    retraction_counts: Optional[List[int]] = None
    # Detailed breakdown of cached tokens by source (device/host/storage)
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None

    # For observability
    time_stats: Optional[List[SchedulerReqTimeStatsIPC]] = None


class ClearHiCacheReqInput(msgspec.Struct, tag=True):
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class ClearHiCacheReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class FlushCacheReqInput(msgspec.Struct, tag=True):
    timeout_s: Optional[float] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class FlushCacheReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    message: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class AddExternalCorpusReqInput(msgspec.Struct, tag=True):
    corpus_id: Optional[str] = None
    file_path: Optional[str] = None
    documents: Optional[List[str]] = None
    token_chunks: Optional[List[List[int]]] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class AddExternalCorpusReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    corpus_id: str = ""
    message: str = ""
    loaded_token_count: int = 0
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class RemoveExternalCorpusReqInput(msgspec.Struct, tag=True):
    corpus_id: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class RemoveExternalCorpusReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    message: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class ListExternalCorporaReqInput(msgspec.Struct, tag=True):
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class ListExternalCorporaReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    corpus_token_counts: Dict[str, int] = {}
    message: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class AttachHiCacheStorageReqInput(msgspec.Struct, tag=True):
    """Dynamically attach (enable) HiCache storage backend at runtime.

    Note: `hicache_storage_backend_extra_config_json` is a JSON string. It may contain both:
    - backend-specific configs (e.g., mooncake master address)
    - prefetch-related knobs (prefetch_threshold, prefetch_timeout_*, hicache_storage_pass_prefix_keys)
    """

    hicache_storage_backend: str = ""
    hicache_storage_backend_extra_config_json: Optional[str] = None
    hicache_storage_prefetch_policy: Optional[str] = None
    hicache_write_policy: Optional[str] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None

    def __post_init__(self):
        if self.hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if self.hicache_storage_prefetch_policy not in allowed:
                raise ValueError(
                    f"Invalid hicache_storage_prefetch_policy: {self.hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}."
                )
        if self.hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if self.hicache_write_policy not in allowed:
                raise ValueError(
                    f"Invalid hicache_write_policy: {self.hicache_write_policy!r}. "
                    f"Expected one of {allowed}."
                )


class AttachHiCacheStorageReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    message: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class DetachHiCacheStorageReqInput(msgspec.Struct, tag=True):
    """Dynamically detach (disable) HiCache storage backend at runtime."""

    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class DetachHiCacheStorageReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    message: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class PauseGenerationReqInput(msgspec.Struct, tag=True):
    """
    Note that the PauseGenerationRequests is only supported in SGLang Server.
    abort: Abort and return all requests currently being processed.

    in_place: Pause the scheduler's event_loop from performing inference;
            only non-inference requests (e.g., control commands) will be handled.
            The requests in the engine will be paused and stay in the event_loop,
            then continue generation after continue_generation with the old kv cache.
            Note: In 'inplace' mode, flush_cache will fail if there are any requests
            in the running_batch.

    retract: Pause the scheduler's event loop from performing inference;
            only non-inference requests will be handled, and all currently running
            requests will be retracted back to the waiting_queue.
            Note: The KV cache can be flushed in this mode and will be automatically
            recomputed after continue_generation.
    """

    mode: str = "abort"
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None

    def __post_init__(self):
        allowed = ["abort", "retract", "in_place"]
        if self.mode not in allowed:
            raise ValueError(
                f"Invalid mode: {self.mode!r}. " f"Expected one of {allowed}."
            )


class ContinueGenerationReqInput(msgspec.Struct, tag=True):
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class UpdateWeightFromDiskReqInput(msgspec.Struct, tag=True):
    # The model path with the new weights
    model_path: str = ""
    # The format to load the weights
    load_format: Optional[str] = None
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Whether to update weights asynchronously
    is_async: bool = False
    # Whether to empty torch cache
    torch_empty_cache: bool = False
    # Whether to keep the scheduler paused after weight update
    keep_pause: bool = False
    # Whether to recapture cuda graph after weight update
    recapture_cuda_graph: bool = False
    # The trainer step id. Used to know which step's weights are used for sampling.
    token_step: int = 0
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Tensor metadata
    manifest: Optional[Dict[str, Any]] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class UpdateWeightFromDiskReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    message: str = ""
    # Number of paused requests during weight sync.
    num_paused_requests: Optional[int] = 0
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class UpdateWeightsFromDistributedReqInput(msgspec.Struct, tag=True):
    names: List[str] = []
    dtypes: List[str] = []
    shapes: List[List[int]] = []
    # The group name
    group_name: str = "weight_update_group"
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Optional format specification for loading
    load_format: Optional[str] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class UpdateWeightsFromDistributedReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    message: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class UpdateWeightsFromTensorReqInput(msgspec.Struct, tag=True):
    """Update model weights from tensor input.

    - Tensors are serialized for transmission
    - Data is structured in JSON for easy transmission over HTTP
    """

    serialized_named_tensors: List[Union[str, bytes]] = []
    # Optional format specification for loading
    load_format: Optional[str] = None
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Optional: Determine whether to disable updating the draft model
    disable_draft_model: Optional[bool] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class UpdateWeightsFromTensorReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    message: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class InitWeightsSendGroupForRemoteInstanceReqInput(msgspec.Struct, tag=True):
    # The master address
    master_address: str = ""
    # The ports for each rank's communication group
    ports: str = ""
    # The rank in the communication group
    group_rank: int = 0
    # The world size
    world_size: int = 1
    # The group name
    group_name: str = "weight_send_group"
    # The backend
    backend: str = "nccl"
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


# Now UpdateWeightsFromIPCReqInput and UpdateWeightsFromIPCReqOutput
# are only used by Checkpoint Engine (https://github.com/MoonshotAI/checkpoint-engine)
class UpdateWeightsFromIPCReqInput(msgspec.Struct, tag=True):
    # ZMQ socket paths for each device UUID
    zmq_handles: Dict[str, str] = {}
    # Whether to flush cache after weight update
    flush_cache: bool = True
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class UpdateWeightsFromIPCReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    message: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class InitWeightsSendGroupForRemoteInstanceReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    message: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class SendWeightsToRemoteInstanceReqInput(msgspec.Struct, tag=True):
    # The master address
    master_address: str = ""
    # The ports for each rank's communication group
    ports: str = ""
    # The group name
    group_name: str = "weight_send_group"
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class SendWeightsToRemoteInstanceReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    message: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class UpdateExpertBackupReq(msgspec.Struct, tag=True):
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class BackupDramReq(msgspec.Struct, tag=True):
    rank: int = 0
    weight_pointer_map: Dict[str, Any] = {}
    session_id: str = ""
    buffer_size: int = 0
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class InitWeightsUpdateGroupReqInput(msgspec.Struct, tag=True):
    # The master address
    master_address: str = ""
    # The master port
    master_port: int = 0
    # The rank offset
    rank_offset: int = 0
    # The world size
    world_size: int = 1
    # The group name
    group_name: str = "weight_update_group"
    # The backend
    backend: str = "nccl"
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class InitWeightsUpdateGroupReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    message: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class DestroyWeightsUpdateGroupReqInput(msgspec.Struct, tag=True):
    group_name: str = "weight_update_group"
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class DestroyWeightsUpdateGroupReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    message: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class UpdateWeightVersionReqInput(msgspec.Struct, tag=True):
    # The new weight version
    new_version: str = ""
    # Whether to abort all running requests before updating
    abort_all_requests: bool = True
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class GetWeightsByNameReqInput(msgspec.Struct, tag=True):
    name: str = ""
    truncate_size: int = 100
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class GetWeightsByNameReqOutput(msgspec.Struct, tag=True):
    parameter: list = []
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class ReleaseMemoryOccupationReqInput(msgspec.Struct, tag=True):
    # Optional tags to identify the memory region, which is primarily used for RL
    # Currently we only support `weights` and `kv_cache`
    tags: Optional[List[str]] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class ReleaseMemoryOccupationReqOutput(msgspec.Struct, tag=True):
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class ResumeMemoryOccupationReqInput(msgspec.Struct, tag=True):
    # Optional tags to identify the memory region, which is primarily used for RL
    # Currently we only support `weights` and `kv_cache`
    tags: Optional[List[str]] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class ResumeMemoryOccupationReqOutput(msgspec.Struct, tag=True):
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class CheckWeightsReqInput(msgspec.Struct, tag=True):
    action: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class CheckWeightsReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    message: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class SlowDownReqInput(msgspec.Struct, tag=True):
    forward_sleep_time: Optional[float] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class SlowDownReqOutput(msgspec.Struct, tag=True):
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class AbortReq(msgspec.Struct, tag=True):
    # Whether to abort all requests
    abort_all: bool = False
    # The finished reason data
    finished_reason: Optional[Dict[str, Any]] = None
    abort_message: Optional[str] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None

    def __post_init__(self):
        # FIXME: This is a hack to keep the same with the old code
        if self.rid is None:
            self.rid = ""


class ActiveRanksOutput(msgspec.Struct, tag=True):
    status: List[bool] = []
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class GetInternalStateReq(msgspec.Struct, tag=True):
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class GetInternalStateReqOutput(msgspec.Struct, tag=True):
    internal_state: Dict[Any, Any] = {}
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class SetInternalStateReq(msgspec.Struct, tag=True):
    server_args: Dict[str, Any] = {}
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class SetInternalStateReqOutput(msgspec.Struct, tag=True):
    updated: bool = False
    server_args: Dict[str, Any] = {}
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class ProfileReqInput(msgspec.Struct, tag=True):
    # The output directory
    output_dir: Optional[str] = None
    # Specify the steps to start the profiling
    start_step: Optional[int] = None
    # If set, it profile as many as this number of steps.
    # If it is set, profiling is automatically stopped after this step, and
    # the caller doesn’t need to run stop_profile.
    num_steps: Optional[int] = None
    # The activities to record. The choices are ["CPU", "GPU", "MEM", "RPD"]
    activities: Optional[List[str]] = None
    # Whether profile by stages (e.g., prefill and decode) separately
    profile_by_stage: bool = False
    # Whether to record source information (file and line number) for the ops.
    with_stack: Optional[bool] = None
    # Whether to save information about operator’s input shapes.
    record_shapes: Optional[bool] = None
    # Merge profiles from all ranks into a single trace
    merge_profiles: bool = False
    # The prefix of the profile filenames
    profile_prefix: Optional[str] = None
    # Only profile these stages and ignore others
    profile_stages: Optional[List[str]] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class ProfileReqType(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


class ProfileReq(msgspec.Struct, tag=True):
    type: ProfileReqType = ProfileReqType.START_PROFILE
    output_dir: Optional[str] = None
    start_step: Optional[int] = None
    num_steps: Optional[int] = None
    activities: Optional[List[str]] = None
    profile_by_stage: bool = False
    with_stack: Optional[bool] = None
    record_shapes: Optional[bool] = None
    profile_id: Optional[str] = None
    merge_profiles: bool = False
    profile_prefix: Optional[str] = None
    profile_stages: Optional[List[str]] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class ProfileReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    message: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class FreezeGCReq(msgspec.Struct, tag=True):
    pass


class ConfigureLoggingReq(msgspec.Struct, tag=True):
    log_requests: Optional[bool] = None
    log_requests_level: Optional[int] = None
    log_requests_format: Optional[str] = None
    dump_requests_folder: Optional[str] = None
    dump_requests_threshold: Optional[int] = None
    crash_dump_folder: Optional[str] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class OpenSessionReqInput(msgspec.Struct, tag=True):
    capacity_of_str_len: int = 0
    session_id: Optional[str] = None
    streaming: Optional[bool] = None
    timeout: Optional[float] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class CloseSessionReqInput(msgspec.Struct, tag=True):
    session_id: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class OpenSessionReqOutput(msgspec.Struct, tag=True):
    session_id: Optional[str] = None
    success: bool = False
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class HealthCheckOutput(msgspec.Struct, tag=True):
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class ExpertDistributionReqType(Enum):
    START_RECORD = 1
    STOP_RECORD = 2
    DUMP_RECORD = 3


class ExpertDistributionReq(msgspec.Struct, tag=True):
    action: ExpertDistributionReqType = ExpertDistributionReqType.START_RECORD
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class ExpertDistributionReqOutput(msgspec.Struct, tag=True):
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class Function(msgspec.Struct):
    description: Optional[str] = None
    name: Optional[str] = None
    parameters: Optional[Any] = None


class Tool(msgspec.Struct):
    function: Function = msgspec.field(default_factory=Function)
    type: Optional[str] = "function"


class ParseFunctionCallReq(msgspec.Struct, tag=True):
    text: str = ""  # The text to parse.
    tools: List[Tool] = []  # A list of available function tools (name, parameters, etc.).
    tool_call_parser: Optional[str] = None  # Specify the parser type, e.g. 'llama3', 'qwen25', or 'mistral'. If not specified, tries all.
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class SeparateReasoningReqInput(msgspec.Struct, tag=True):
    text: str = ""  # The text to parse.
    reasoning_parser: str = ""  # Specify the parser type, e.g., "deepseek-r1".
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class VertexGenerateReqInput(msgspec.Struct, tag=True):
    instances: List[Dict[str, Any]] = []
    parameters: Optional[Dict[str, Any]] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class RpcReqInput(msgspec.Struct, tag=True):
    method: str = ""
    parameters: Optional[Dict[str, Any]] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class RpcReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    message: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class LoadLoRAAdapterReqInput(msgspec.Struct, tag=True):
    # The name of the lora module to newly loaded.
    lora_name: str = ""
    # The path of loading.
    lora_path: str = ""
    # Whether to pin the LoRA adapter in memory.
    pinned: bool = False
    # The unique identifier for the LoRA adapter, which automatically generated in the `TokenizerManager`.
    lora_id: Optional[str] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
            lora_path=self.lora_path,
            pinned=self.pinned,
        )


class UnloadLoRAAdapterReqInput(msgspec.Struct, tag=True):
    # The name of lora module to unload.
    lora_name: str = ""
    # The unique identifier for the LoRA adapter, which automatically generated in the `TokenizerManager`.
    lora_id: Optional[str] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
        )


class LoadLoRAAdapterFromTensorsReqInput(msgspec.Struct, tag=True):
    lora_name: str = ""
    config_dict: Dict[str, Any] = {}
    serialized_tensors: str = ""
    pinned: bool = False
    added_tokens_config: Optional[Dict[str, Any]] = None
    lora_id: Optional[str] = None
    load_format: Optional[str] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
            lora_path="__tensor__",
            pinned=self.pinned,
        )


class LoRAUpdateOutput(msgspec.Struct, tag=True):
    success: bool = False
    error_message: Optional[str] = None
    loaded_adapters: Optional[Dict[str, LoRARef]] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


LoadLoRAAdapterReqOutput = UnloadLoRAAdapterReqOutput = (
    LoadLoRAAdapterFromTensorsReqOutput
) = LoRAUpdateOutput


class BlockReqType(Enum):
    BLOCK = 1
    UNBLOCK = 2


class BlockReqInput(msgspec.Struct, tag=True):
    type: BlockReqType = BlockReqType.BLOCK
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class GetLoadReqInput(msgspec.Struct, tag=True):
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class GetLoadReqOutput(msgspec.Struct, tag=True):
    dp_rank: Optional[int] = None
    num_reqs: int = 0
    num_waiting_reqs: int = 0
    num_tokens: int = 0
    num_pending_tokens: int = 0
    ts_tic: float = 0.0


@dataclass
class MemoryMetrics:
    """Memory breakdown metrics."""

    weight_gb: float = field(
        metadata={"metric": ("gauge", "Model weight memory in GB")}
    )
    kv_cache_gb: float = field(metadata={"metric": ("gauge", "KV cache memory in GB")})
    graph_gb: float = field(metadata={"metric": ("gauge", "CUDA graph memory in GB")})
    token_capacity: int = field(
        metadata={"metric": ("gauge", "Max tokens in KV cache")}
    )


@dataclass
class SpeculativeMetrics:
    """Speculative decoding metrics."""

    accept_length: float = field(
        metadata={"metric": ("gauge", "Avg accepted tokens per step")}
    )
    accept_rate: float = field(
        metadata={"metric": ("gauge", "Speculative acceptance rate")}
    )


@dataclass
class LoRAMetrics:
    """LoRA adapter pool metrics."""

    slots_used: int = field(metadata={"metric": ("gauge", "LoRA adapter slots in use")})
    slots_total: int = field(metadata={"metric": ("gauge", "Total LoRA adapter slots")})
    utilization: float = field(
        metadata={"metric": ("gauge", "LoRA pool utilization ratio")}
    )


@dataclass
class DisaggregationMetrics:
    """PD disaggregation metrics."""

    mode: str  # "prefill", "decode", or "null" - not a metric
    prefill_prealloc_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Prefill prealloc queue requests")}
    )
    prefill_inflight_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Prefill inflight queue requests")}
    )
    decode_prealloc_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Decode prealloc queue requests")}
    )
    decode_transfer_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Decode transfer queue requests")}
    )
    decode_retracted_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Decode retracted queue requests")}
    )
    kv_transfer_speed_gb_s: float = field(
        default=0.0, metadata={"metric": ("gauge", "KV transfer speed in GB/s")}
    )
    kv_transfer_latency_ms: float = field(
        default=0.0, metadata={"metric": ("gauge", "KV transfer latency in ms")}
    )


@dataclass
class QueueMetrics:
    """Detailed queue breakdown."""

    waiting: int = field(metadata={"metric": ("gauge", "Main waiting queue size")})
    grammar: int = field(
        metadata={"metric": ("gauge", "Grammar compilation queue size")}
    )
    paused: int = field(
        metadata={"metric": ("gauge", "Requests paused by weight sync")}
    )
    retracted: int = field(metadata={"metric": ("gauge", "Retracted requests count")})


class GetLoadsReqInput(msgspec.Struct, tag=True):
    """Request for /v1/loads endpoint."""

    VALID_SECTIONS: ClassVar[frozenset] = frozenset(
        {"core", "memory", "spec", "lora", "disagg", "queues", "all"}
    )

    include: List[str] = ["all"]
    dp_rank: Optional[int] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None

    def __post_init__(self):
        """Validate include sections."""
        if self.include:
            invalid = set(self.include) - self.VALID_SECTIONS
            if invalid:
                raise ValueError(
                    f"Invalid include sections: {invalid}. "
                    f"Valid options: {sorted(self.VALID_SECTIONS)}"
                )


class GetLoadsReqOutput(msgspec.Struct, tag=True):
    """Per-DP-rank load metrics for /v1/loads endpoint."""

    dp_rank: int = 0
    timestamp: float = 0.0

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    num_used_tokens: int = 0
    max_total_num_tokens: int = 0
    # FIXME: token_usage is actually max usage across all pools (KV, SWA, mamba),
    # not just KV token usage. Rename requires API deprecation.
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    cache_hit_rate: float = 0.0
    utilization: float = 0.0
    max_running_requests: int = 0

    memory: Optional[MemoryMetrics] = None
    speculative: Optional[SpeculativeMetrics] = None
    lora: Optional[LoRAMetrics] = None
    disaggregation: Optional[DisaggregationMetrics] = None
    queues: Optional[QueueMetrics] = None
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class WatchLoadUpdateReq(msgspec.Struct, tag=True):
    loads: List[GetLoadReqOutput] = []
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class SetInjectDumpMetadataReqInput(msgspec.Struct, tag=True):
    dump_metadata: Dict[str, Any] = {}
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class SetInjectDumpMetadataReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class LazyDumpTensorsReqInput(msgspec.Struct, tag=True):
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class LazyDumpTensorsReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class DumperControlReqInput(msgspec.Struct, tag=True):
    method: str = ""
    body: Dict[str, Any] = {}
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None


class DumperControlReqOutput(msgspec.Struct, tag=True):
    success: bool = False
    response: List[Dict[str, Any]] = []
    error: str = ""
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None

class SamplingParamsIPC(msgspec.Struct, tag=True):
    """Msgpack-serializable IPC version of SamplingParams.

    Fields containing torch tensors or non-msgspec objects are stored as
    pickle-encoded bytes and reconstructed on the receiving side.
    """

    max_new_tokens: int = 128
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    stop_regex: Optional[Union[str, List[str]]] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_new_tokens: int = 0
    n: int = 1
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    structural_tag: Optional[str] = None
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    no_stop_trim: bool = False
    custom_params: Optional[Dict[str, Any]] = None
    stream_interval: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    sampling_seed: Optional[int] = None


    @classmethod
    def from_sampling_params(cls, params: SamplingParams) -> "SamplingParamsIPC":
        return cls(
            max_new_tokens=params.max_new_tokens,
            stop = params.stop,
            stop_token_ids = params.stop_token_ids,
            stop_regex = params.stop_regex,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            min_p=params.min_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            repetition_penalty=params.repetition_penalty,
            min_new_tokens=params.min_new_tokens,
            n=params.n,
            json_schema=params.json_schema,
            regex=params.regex,
            ebnf=params.ebnf,
            structural_tag=params.structural_tag,
            ignore_eos=params.ignore_eos,
            skip_special_tokens=params.skip_special_tokens,
            spaces_between_special_tokens=params.spaces_between_special_tokens,
            no_stop_trim=params.no_stop_trim,
            custom_params=params.custom_params,
            stream_interval=params.stream_interval,
            logit_bias=params.logit_bias,
            sampling_seed=params.sampling_seed,
        )

    @classmethod
    def to_sampling_params(cls, ipc_params: "SamplingParamsIPC") -> SamplingParams:
        return SamplingParams(
            max_new_tokens=ipc_params.max_new_tokens,
            stop = ipc_params.stop,
            stop_token_ids = ipc_params.stop_token_ids,
            stop_regex = ipc_params.stop_regex,
            temperature=ipc_params.temperature,
            top_p=ipc_params.top_p,
            top_k=ipc_params.top_k,
            min_p=ipc_params.min_p,
            frequency_penalty=ipc_params.frequency_penalty,
            presence_penalty=ipc_params.presence_penalty,
            repetition_penalty=ipc_params.repetition_penalty,
            min_new_tokens=ipc_params.min_new_tokens,
            n=ipc_params.n,
            json_schema=ipc_params.json_schema,
            regex=ipc_params.regex,
            ebnf=ipc_params.ebnf,
            structural_tag=ipc_params.structural_tag,
            ignore_eos=ipc_params.ignore_eos,
            skip_special_tokens=ipc_params.skip_special_tokens,
            spaces_between_special_tokens=ipc_params.spaces_between_special_tokens,
            no_stop_trim=ipc_params.no_stop_trim,
            custom_params=ipc_params.custom_params,
            stream_interval=ipc_params.stream_interval,
            logit_bias=ipc_params.logit_bias,
            sampling_seed=ipc_params.sampling_seed,
        )

class APIServerReqTimeStatsIPC(msgspec.Struct, tag=True):
    """Structured time stats for API server requests, to be sent over IPC as bytes."""

    created_time: float = 0.0
    finished_time: float = 0.0
    first_token_time: float = 0.0
    last_time: float = 0.0
    tokenize_finish_time: float = 0.0
    api_server_dispatch_time: float = 0.0
    api_server_dispatch_finish_time: float = 0.0
    response_sent_to_client_time: float = 0.0

    @classmethod
    def from_req_time_stats(cls, stats: APIServerReqTimeStats) -> "APIServerReqTimeStatsIPC":
        return cls(
            created_time=stats.created_time,
            finished_time=stats.finished_time,
            first_token_time=stats.first_token_time,
            last_time=stats.last_time,
            tokenize_finish_time=stats.tokenize_finish_time,
            api_server_dispatch_time=stats.api_server_dispatch_time,
            api_server_dispatch_finish_time=stats.api_server_dispatch_finish_time,
            response_sent_to_client_time=stats.response_sent_to_client_time,
        )

    @classmethod
    def to_req_time_stats(cls, ipc_stats: "APIServerReqTimeStatsIPC") -> APIServerReqTimeStats:
        return APIServerReqTimeStats(
            created_time=ipc_stats.created_time,
            finished_time=ipc_stats.finished_time,
            first_token_time=ipc_stats.first_token_time,
            last_time=ipc_stats.last_time,
            tokenize_finish_time=ipc_stats.tokenize_finish_time,
            api_server_dispatch_time=ipc_stats.api_server_dispatch_time,
            api_server_dispatch_finish_time=ipc_stats.api_server_dispatch_finish_time,
            response_sent_to_client_time=ipc_stats.response_sent_to_client_time,
        )

# ---------------------------------------------------------------------------
# IPC versions of complex request types (tensor-containing fields as bytes)
# ---------------------------------------------------------------------------


class TokenizedGenerateReqInputIPC(msgspec.Struct, tag=True):
    """Msgpack-serializable IPC version of TokenizedGenerateReqInput.

    Fields containing torch tensors or non-msgspec objects are stored as
    pickle-encoded bytes and reconstructed on the receiving side.
    """

    input_text: str = ""
    input_ids: List[int] = []
    return_logprob: bool = False
    logprob_start_len: int = 0
    top_logprobs_num: int = 0
    token_ids_logprob: List[int] = []
    stream: bool = False
    return_hidden_states: bool = False
    return_routed_experts: bool = False
    routed_experts_start_len: int = 0
    input_embeds: Optional[Any] = None
    session_params: Optional[SessionParams] = None
    lora_id: Optional[str] = None
    custom_logit_processor: Optional[str] = None
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None
    bootstrap_room: Optional[int] = None
    bootstrap_pair_key: Optional[str] = None
    decode_tp_size: Optional[int] = None
    require_reasoning: bool = False
    routed_dp_rank: Optional[int] = None
    disagg_prefill_dp_rank: Optional[int] = None
    priority: Optional[int] = None
    extra_key: Optional[str] = None
    routing_key: Optional[str] = None
    no_logs: bool = False
    return_bytes: bool = False
    return_entropy: bool = False
    token_type_ids: Optional[List[int]] = None
    need_wait_for_mm_inputs: bool = False
    num_items_assigned: Optional[Dict[str, List[int]]] = None
    # Complex fields as pickle bytes
    mm_inputs: Optional[bytes] = None
    sampling_params: SamplingParamsIPC
    positional_embed_overrides: Optional[bytes] = None
    time_stats: Optional[bytes] = None
    # BaseReq fields
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None

    @classmethod
    def from_tokenized(cls, req: TokenizedGenerateReqInput) -> "TokenizedGenerateReqInputIPC":
        import pickle as _pickle

        return cls(
            input_text=req.input_text,
            input_ids=req.input_ids,
            return_logprob=req.return_logprob,
            logprob_start_len=req.logprob_start_len,
            top_logprobs_num=req.top_logprobs_num,
            token_ids_logprob=req.token_ids_logprob,
            stream=req.stream,
            return_hidden_states=req.return_hidden_states,
            return_routed_experts=req.return_routed_experts,
            routed_experts_start_len=req.routed_experts_start_len,
            input_embeds=req.input_embeds,
            session_params=req.session_params,
            lora_id=req.lora_id,
            custom_logit_processor=req.custom_logit_processor,
            bootstrap_host=req.bootstrap_host,
            bootstrap_port=req.bootstrap_port,
            bootstrap_room=req.bootstrap_room,
            bootstrap_pair_key=req.bootstrap_pair_key,
            decode_tp_size=req.decode_tp_size,
            require_reasoning=req.require_reasoning,
            routed_dp_rank=req.routed_dp_rank,
            disagg_prefill_dp_rank=req.disagg_prefill_dp_rank,
            priority=req.priority,
            extra_key=req.extra_key,
            routing_key=req.routing_key,
            no_logs=req.no_logs,
            return_bytes=req.return_bytes,
            return_entropy=req.return_entropy,
            token_type_ids=req.token_type_ids,
            need_wait_for_mm_inputs=req.need_wait_for_mm_inputs,
            num_items_assigned=req.num_items_assigned,
            mm_inputs=_pickle.dumps(req.mm_inputs) if req.mm_inputs is not None else None,
            sampling_params=SamplingParamsIPC.from_sampling_params(req.sampling_params),
            positional_embed_overrides=(
                _pickle.dumps(req.positional_embed_overrides)
                if req.positional_embed_overrides is not None
                else None
            ),
            time_stats=_pickle.dumps(req.time_stats) if req.time_stats is not None else None,
            rid=req.rid,
            http_worker_ipc=req.http_worker_ipc,
        )

    def to_tokenized(self):
        import pickle as _pickle
        from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

        return TokenizedGenerateReqInput(
            input_text=self.input_text,
            input_ids=self.input_ids,
            mm_inputs=_pickle.loads(self.mm_inputs) if self.mm_inputs is not None else None,
            sampling_params=SamplingParamsIPC.to_sampling_params(self.sampling_params),
            return_logprob=self.return_logprob,
            logprob_start_len=self.logprob_start_len,
            top_logprobs_num=self.top_logprobs_num,
            token_ids_logprob=self.token_ids_logprob,
            stream=self.stream,
            return_hidden_states=self.return_hidden_states,
            return_routed_experts=self.return_routed_experts,
            routed_experts_start_len=self.routed_experts_start_len,
            input_embeds=self.input_embeds,
            positional_embed_overrides=(
                _pickle.loads(self.positional_embed_overrides)
                if self.positional_embed_overrides is not None
                else None
            ),
            session_params=self.session_params,
            lora_id=self.lora_id,
            custom_logit_processor=self.custom_logit_processor,
            bootstrap_host=self.bootstrap_host,
            bootstrap_port=self.bootstrap_port,
            bootstrap_room=self.bootstrap_room,
            bootstrap_pair_key=self.bootstrap_pair_key,
            decode_tp_size=self.decode_tp_size,
            require_reasoning=self.require_reasoning,
            routed_dp_rank=self.routed_dp_rank,
            disagg_prefill_dp_rank=self.disagg_prefill_dp_rank,
            priority=self.priority,
            extra_key=self.extra_key,
            routing_key=self.routing_key,
            no_logs=self.no_logs,
            return_bytes=self.return_bytes,
            return_entropy=self.return_entropy,
            token_type_ids=self.token_type_ids,
            need_wait_for_mm_inputs=self.need_wait_for_mm_inputs,
            num_items_assigned=self.num_items_assigned,
            time_stats=(
                _pickle.loads(self.time_stats) if self.time_stats is not None else None
            ),
            rid=self.rid,
            http_worker_ipc=self.http_worker_ipc,
        )


class BatchTokenizedGenerateReqInputIPC(msgspec.Struct, tag=True):
    """Msgpack-serializable IPC version of BatchTokenizedGenerateReqInput."""

    batch: List[TokenizedGenerateReqInputIPC] = []
    rids: Optional[List[str]] = None
    http_worker_ipcs: Optional[List[Optional[str]]] = None

    @classmethod
    def from_batch(cls, batch_req) -> "BatchTokenizedGenerateReqInputIPC":
        return cls(
            batch=[TokenizedGenerateReqInputIPC.from_tokenized(r) for r in batch_req.batch],
            rids=batch_req.rids,
            http_worker_ipcs=batch_req.http_worker_ipcs,
        )

    def to_batch(self):
        from sglang.srt.managers.io_struct import BatchTokenizedGenerateReqInput

        return BatchTokenizedGenerateReqInput(
            batch=[r.to_tokenized() for r in self.batch],
            rids=self.rids,
            http_worker_ipcs=self.http_worker_ipcs,
        )


class TokenizedEmbeddingReqInputIPC(msgspec.Struct, tag=True):
    """Msgpack-serializable IPC version of TokenizedEmbeddingReqInput."""

    input_text: str = ""
    input_ids: List[int] = []
    image_inputs: Dict[str, Any] = {}
    token_type_ids: List[int] = []
    lora_id: Optional[str] = None
    routed_dp_rank: Optional[int] = None
    priority: Optional[int] = None
    dimensions: Optional[int] = None
    # Complex fields as pickle bytes
    sampling_params: bytes = b""
    positional_embed_overrides: Optional[bytes] = None
    time_stats: Optional[bytes] = None
    # BaseReq fields
    rid: Optional[Union[str, List[str]]] = None
    http_worker_ipc: Optional[str] = None

    @classmethod
    def from_tokenized(cls, req) -> "TokenizedEmbeddingReqInputIPC":
        import pickle as _pickle

        return cls(
            input_text=req.input_text,
            input_ids=req.input_ids,
            image_inputs=req.image_inputs,
            token_type_ids=req.token_type_ids,
            lora_id=req.lora_id,
            routed_dp_rank=req.routed_dp_rank,
            priority=req.priority,
            dimensions=req.dimensions,
            sampling_params=_pickle.dumps(req.sampling_params),
            positional_embed_overrides=(
                _pickle.dumps(req.positional_embed_overrides)
                if req.positional_embed_overrides is not None
                else None
            ),
            time_stats=_pickle.dumps(req.time_stats) if req.time_stats is not None else None,
            rid=req.rid,
            http_worker_ipc=req.http_worker_ipc,
        )

    def to_tokenized(self):
        import pickle as _pickle

        from sglang.srt.managers.io_struct import TokenizedEmbeddingReqInput

        return TokenizedEmbeddingReqInput(
            input_text=self.input_text,
            input_ids=self.input_ids,
            image_inputs=self.image_inputs,
            token_type_ids=self.token_type_ids,
            sampling_params=_pickle.loads(self.sampling_params),
            positional_embed_overrides=(
                _pickle.loads(self.positional_embed_overrides)
                if self.positional_embed_overrides is not None
                else None
            ),
            lora_id=self.lora_id,
            routed_dp_rank=self.routed_dp_rank,
            priority=self.priority,
            dimensions=self.dimensions,
            time_stats=(
                _pickle.loads(self.time_stats) if self.time_stats is not None else None
            ),
            rid=self.rid,
            http_worker_ipc=self.http_worker_ipc,
        )


class BatchTokenizedEmbeddingReqInputIPC(msgspec.Struct, tag=True):
    """Msgpack-serializable IPC version of BatchTokenizedEmbeddingReqInput."""

    batch: List[TokenizedEmbeddingReqInputIPC] = []
    rids: Optional[List[str]] = None
    http_worker_ipcs: Optional[List[Optional[str]]] = None

    @classmethod
    def from_batch(cls, batch_req) -> "BatchTokenizedEmbeddingReqInputIPC":
        return cls(
            batch=[TokenizedEmbeddingReqInputIPC.from_tokenized(r) for r in batch_req.batch],
            rids=batch_req.rids,
            http_worker_ipcs=batch_req.http_worker_ipcs,
        )

    def to_batch(self):
        from sglang.srt.managers.io_struct import BatchTokenizedEmbeddingReqInput

        return BatchTokenizedEmbeddingReqInput(
            batch=[r.to_tokenized() for r in self.batch],
            rids=self.rids,
            http_worker_ipcs=self.http_worker_ipcs,
        )


# ---------------------------------------------------------------------------
# Msgpack IPC helpers for all channels
# ---------------------------------------------------------------------------

# Scheduler → Detokenizer
SchedulerToDetokenizerMsg = Union[BatchTokenIDOutput, BatchEmbeddingOutput, FreezeGCReq]

# Detokenizer/Scheduler → Tokenizer (both use the same tokenizer socket)
# Includes all types the scheduler sends directly to the tokenizer.
AllToTokenizerMsg = Union[
    # Detokenizer outputs (already migrated)
    BatchStrOutput,
    BatchEmbeddingOutput,
    BatchTokenIDOutput,
    # Scheduler control responses
    AbortReq,
    ActiveRanksOutput,
    HealthCheckOutput,
    FlushCacheReqOutput,
    ClearHiCacheReqOutput,
    AttachHiCacheStorageReqOutput,
    DetachHiCacheStorageReqOutput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromTensorReqOutput,
    UpdateWeightsFromIPCReqOutput,
    InitWeightsUpdateGroupReqOutput,
    DestroyWeightsUpdateGroupReqOutput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    SendWeightsToRemoteInstanceReqOutput,
    GetWeightsByNameReqOutput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqOutput,
    CheckWeightsReqOutput,
    SlowDownReqOutput,
    ProfileReqOutput,
    GetInternalStateReqOutput,
    SetInternalStateReqOutput,
    OpenSessionReqOutput,
    ExpertDistributionReqOutput,
    LoRAUpdateOutput,
    GetLoadReqOutput,
    GetLoadsReqOutput,
    DumperControlReqOutput,
    AddExternalCorpusReqOutput,
    RemoveExternalCorpusReqOutput,
    ListExternalCorporaReqOutput,
    SetInjectDumpMetadataReqOutput,
    LazyDumpTensorsReqOutput,
]

# Tokenizer → Scheduler
TokenizerToSchedulerMsg = Union[
    # Main request types (IPC versions, deserialized to dataclass on scheduler side)
    TokenizedGenerateReqInputIPC,
    BatchTokenizedGenerateReqInputIPC,
    TokenizedEmbeddingReqInputIPC,
    BatchTokenizedEmbeddingReqInputIPC,
    # Control messages
    AbortReq,
    FreezeGCReq,
    PauseGenerationReqInput,
    ContinueGenerationReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromIPCReqInput,
    InitWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqInput,
    GetWeightsByNameReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    CheckWeightsReqInput,
    SlowDownReqInput,
    ProfileReq,
    GetInternalStateReq,
    SetInternalStateReq,
    RpcReqInput,
    ExpertDistributionReq,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    UnloadLoRAAdapterReqInput,
    GetLoadReqInput,
    GetLoadsReqInput,
    FlushCacheReqInput,
    ClearHiCacheReqInput,
    AttachHiCacheStorageReqInput,
    DetachHiCacheStorageReqInput,
    OpenSessionReqInput,
    CloseSessionReqInput,
    ActiveRanksOutput,
    WatchLoadUpdateReq,
    BlockReqInput,
    UpdateWeightVersionReqInput,
    ConfigureLoggingReq,
    ParseFunctionCallReq,
    SeparateReasoningReqInput,
    VertexGenerateReqInput,
    DumperControlReqInput,
    AddExternalCorpusReqInput,
    RemoveExternalCorpusReqInput,
    ListExternalCorporaReqInput,
    UpdateExpertBackupReq,
    BackupDramReq,
    SetInjectDumpMetadataReqInput,
    LazyDumpTensorsReqInput,
]

_msgpack_encoder = msgspec.msgpack.Encoder()
_s2d_decoder = msgspec.msgpack.Decoder(SchedulerToDetokenizerMsg)
_a2t_decoder = msgspec.msgpack.Decoder(AllToTokenizerMsg)
_t2s_decoder = msgspec.msgpack.Decoder(TokenizerToSchedulerMsg)
_rpc_req_decoder = msgspec.msgpack.Decoder(RpcReqInput)
_rpc_res_decoder = msgspec.msgpack.Decoder(RpcReqOutput)


def send_msgpack(socket, obj):
    """Send a msgspec.Struct via zmq using msgpack serialization."""
    socket.send(_msgpack_encoder.encode(obj))


async def async_send_msgpack(socket, obj):
    """Async send a msgspec.Struct via zmq.asyncio using msgpack serialization."""
    await socket.send(_msgpack_encoder.encode(obj))


def recv_msgpack_s2d(socket):
    """Receive scheduler→detokenizer message via msgpack."""
    return _s2d_decoder.decode(socket.recv())


async def async_recv_msgpack_d2t(socket):
    """Receive detokenizer/scheduler→tokenizer message via async zmq msgpack."""
    data = await socket.recv()
    return _a2t_decoder.decode(data)


def recv_msgpack_d2t(socket):
    """Receive detokenizer/scheduler→tokenizer message via zmq msgpack (sync)."""
    return _a2t_decoder.decode(socket.recv())


def recv_msgpack_t2s(socket, flags=0):
    """Receive tokenizer→scheduler message via zmq msgpack."""
    return _t2s_decoder.decode(socket.recv(flags))


def recv_msgpack_rpc_req(socket, flags=0):
    """Receive engine→scheduler RPC request via zmq msgpack."""
    return _rpc_req_decoder.decode(socket.recv(flags))


def recv_msgpack_rpc_res(socket):
    """Receive scheduler→engine RPC response via zmq msgpack."""
    return _rpc_res_decoder.decode(socket.recv())


async def async_recv_msgpack_t2s(socket):
    """Receive tokenizer→scheduler message via async zmq msgpack."""
    return _t2s_decoder.decode(await socket.recv())


def _unwrap_t2s_msg(msg):
    """Convert IPC request types to their dataclass counterparts for scheduler use."""
    if isinstance(msg, TokenizedGenerateReqInputIPC):
        return msg.to_tokenized()
    if isinstance(msg, BatchTokenizedGenerateReqInputIPC):
        return msg.to_batch()
    if isinstance(msg, TokenizedEmbeddingReqInputIPC):
        return msg.to_tokenized()
    if isinstance(msg, BatchTokenizedEmbeddingReqInputIPC):
        return msg.to_batch()
    return msg


def _check_all_req_types():
    """A helper function to check all request types are defined in this file."""
    import inspect
    import sys

    all_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    for class_type in all_classes:
        # check its name
        name = class_type[0]
        is_io_struct = (
            name.endswith("Req") or name.endswith("Input") or name.endswith("Output")
        )
        is_base_req = (
            issubclass(class_type[1], BaseReq)
            or issubclass(class_type[1], BaseBatchReq)
            or issubclass(class_type[1], msgspec.Struct)
        )
        if is_io_struct and not is_base_req:
            raise ValueError(f"{name} is not a subclass of BaseReq or BaseBatchReq.")
        if (
            is_base_req
            and not is_io_struct
            and not issubclass(class_type[1], msgspec.Struct)
        ):
            raise ValueError(
                f"{name} is a subclass of BaseReq but not follow the naming convention."
            )


_check_all_req_types()
