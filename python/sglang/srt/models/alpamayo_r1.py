
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
"""Inference-only AlpamayoR1 model (Dummy Implementation)."""

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

class AlpamayoR1Config(PretrainedConfig):
    model_type = "alpamayo_r1"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

class AlpamayoR1(nn.Module):
    """
    Dummy implementation of AlpamayoR1 for SGLang.
    AlpamayoR1 wraps Qwen3VLForConditionalGeneration as its language model.
    """
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config = None,
    ):
        super().__init__()
        # Initialize internal Qwen3-VL model
        self.language_model = Qwen3VLForConditionalGeneration(
            config, 
            quant_config=quant_config, 
            cache_config=cache_config
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        forward_batch: "ForwardBatch",
        **kwargs,
    ):
        # Delegate forward to the language model
        return self.language_model(input_ids, positions, forward_batch, **kwargs)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Delegate loading to the inner model
        return self.language_model.load_weights(weights)

# Entry point for SGLang model registry
EntryClass = AlpamayoR1
