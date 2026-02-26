
# Copyright 2026 SGLang Team
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
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from sglang.srt.configs.qwen3_vl import Qwen3VLConfig

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.utils import logger
from sglang.srt.managers.schedule_batch import MultimodalInputs


# class AlpamayoR1Config(PretrainedConfig):
#     """Minimal config for AlpamayoR1 that wraps Qwen3-VL."""
#     model_type = "alpamayo_r1"
    
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         Store the vlm_name_or_path if provided
#         self.vlm_name_or_path = "Qwen/Qwen3-VL-8B-Instruct"
#         self.vlm_backend = "qwenvl3"
#         self.vocab_size = kwargs.get("vocab_size", 155697)  # Default vocab size for AlpamayoR1


class AlpamayoR1LogitsProcessor(LogitsProcessor):
    """Masks out Alpamayo trajectory token logits."""

    def __init__(self, config, traj_token_start_idx, traj_vocab_size):
        super().__init__(config)
        self.traj_mask_start = traj_token_start_idx
        self.traj_mask_end = traj_token_start_idx + traj_vocab_size
            
    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head,
        logits_metadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = super()._get_logits(
            hidden_states, lm_head, logits_metadata, embedding_bias
        )
        logits[:, self.traj_mask_start : self.traj_mask_end] = float("-inf")
        return logits
    

class AlpamayoR1(nn.Module):
    """
    Dummy implementation of AlpamayoR1 for SGLang.
    AlpamayoR1 wraps Qwen3VLForConditionalGeneration as its language model (vlm).
    
    This implementation bypasses the standard HF config loading since Alpamayo
    uses a custom config.json format with training-specific fields.
    """
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        
        logger.info(f"AlpamayoR1 initialized")

        # Store config for later use
        self.config = config

        qwen_config = config

        # we increaset vocab size to match Alpamayo's tokenizer, which may have additional special tokens compared to the base Qwen3-VL config
        qwen_config.text_config.vocab_size = config.vocab_size

        # Initialize internal Qwen3-VL model as 'vlm' (matching alpamayo naming)
        self.vlm = Qwen3VLForConditionalGeneration(
            qwen_config, 
            quant_config=quant_config, 
        )
    
        # override the logits processor to mask out trajectory tokens during generation
        self.vlm.logits_processor = AlpamayoR1LogitsProcessor(self.config, 
                                                                traj_token_start_idx=config.traj_token_start_idx, 
                                                                traj_vocab_size=config.traj_vocab_size)

        logger.info(f"AlpamayoR1: Successfully initialized Qwen3-VL as self.vlm")


        self.expert = None
        self.action_space = None
        self.action_in_proj = None
        self.action_out_proj = None
        self.traj_future_start_token_id = 155681 # <|traj_future_start|>
        self.traj_force_stop_token_id = 151645 #<|im_end|>
        # convert action-related parameters

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        forward_batch: "ForwardBatch",
        **kwargs,
    ):
        ret = self.vlm(input_ids, positions, forward_batch, **kwargs)

        if forward_batch.forward_mode.is_decode():
            bstar = int(input_ids.shape[0]) # number of batch items
            for i in range(bstar):
                if input_ids[i] == self.traj_future_start_token_id:
                    # stop generation immediately
                    ret.next_token_logits[i, :] = float("-inf")
                    ret.next_token_logits[i, self.traj_force_stop_token_id] = 0.0
                    # self._run_flow_matching()

        return ret

    
    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.vlm.pad_input_ids(input_ids, mm_inputs)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights into the model.
        
        The weights from Alpamayo checkpoint may have keys prefixed with 'vlm.'
        We strip this prefix and load into self.vlm.
        We skip expert.*, action_space.*, and other non-VLM weights since
        they are not implemented in this inference-only version.
        """
        weights_list = []
        for name, tensor in weights:
            # Skip weights for unimplemented components
            if name.startswith(("expert.", "action_space.", "action_in_proj.", "action_out_proj.", "diffusion.")):
                logger.debug(f"Skipping weight: {name} (not implemented in inference-only mode)")
                continue
            
            # Strip 'vlm.' prefix if present
            if name.startswith("vlm."):
                name = name[4:]  # Remove "vlm."
            weights_list.append((name, tensor))
        
        # Delegate to internal VLM's load_weights
        return self.vlm.load_weights(iter(weights_list))

# Entry point for SGLang model registry
EntryClass = AlpamayoR1
