
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
import copy
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.utils import logger
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.managers.schedule_batch import MultimodalInputs
from .action_in_proj import PerWaypointActionInProjV2


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


        # Build expert from text_config only (same as AutoModel.from_config(text_config)).
        expert_config = copy.deepcopy(self.vlm.config.text_config)
        if getattr(config, "expert_cfg", None) is not None:
            for key, value in config.expert_cfg.items():
                setattr(expert_config, key, value)
        self.expert = Qwen3Model(expert_config, quant_config=quant_config)
        # Expert branch consumes continuous action embeddings, so token embedding is not needed.
        if hasattr(self.expert, "embed_tokens"):
            del self.expert.embed_tokens

        self.action_space = None

        # Build action projection modules from Alpamayo config to match checkpoint shapes.
        action_in_proj_cfg = config.action_in_proj_cfg
        action_space_cfg = config.action_space_cfg
        traj_tokenizer_cfg = config.traj_tokenizer_cfg

        n_waypoints = action_space_cfg["n_waypoints"]
        action_dim = len(traj_tokenizer_cfg["dims_max"])

        self.action_in_proj = PerWaypointActionInProjV2(
            in_dims=[n_waypoints, action_dim],
            out_dim=expert_config.hidden_size,
            hidden_size=action_in_proj_cfg["hidden_size"],
            num_enc_layers=action_in_proj_cfg["num_enc_layers"],
            max_freq=action_in_proj_cfg["max_freq"],
            num_fourier_feats=action_in_proj_cfg["num_fourier_feats"],
        )
        self.action_out_proj = torch.nn.Linear(expert_config.hidden_size, action_dim)
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
            bstar = int(input_ids.shape[0]) # batch size
            for i in range(bstar):
                if input_ids[i] == self.traj_future_start_token_id:
                    # stop generation immediately
                    ret.next_token_logits[i, :] = float("-inf")
                    ret.next_token_logits[i, self.traj_force_stop_token_id] = 0.0
                    # self._run_flow_matching()
                    delta  = forward_batch.mm_inputs[i].mrope_positions
                    # postions

        return ret

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.vlm.pad_input_ids(input_ids, mm_inputs)

    def _load_expert_weights(self, expert_weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load expert (Qwen3 text backbone) weights."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.expert.named_parameters())
        loaded_cnt = 0

        for name, loaded_weight in expert_weights:
            # Keep compatibility with checkpoints that include an extra "model." prefix.
            if name.startswith("model."):
                name = name[len("model.") :]

            # embed_tokens is intentionally removed for expert branch.
            if name.startswith("embed_tokens."):
                continue

            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and (
                    layer_id < self.expert.start_layer
                    or layer_id >= self.expert.end_layer
                )
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Expert parameter {name} not found; skipping")
                    break

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_cnt += 1
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Expert parameter {name} not found; skipping")
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_cnt += 1

        logger.info(f"AlpamayoR1: loaded {loaded_cnt} expert tensors")

    def _load_plain_module_weights(
        self,
        module: nn.Module,
        module_name: str,
        module_weights: Iterable[Tuple[str, torch.Tensor]],
    ):
        state_dict = {name: tensor for name, tensor in module_weights}
        if not state_dict:
            logger.info(f"AlpamayoR1: no weights found for {module_name}")
            return

        incompatible = module.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.warning(
                f"AlpamayoR1: {module_name} missing keys: {incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            logger.warning(
                f"AlpamayoR1: {module_name} unexpected keys: {incompatible.unexpected_keys}"
            )
        logger.info(f"AlpamayoR1: loaded {len(state_dict)} tensors into {module_name}")

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights into the model.
        
        The weights from Alpamayo checkpoint may have keys prefixed with 'vlm.'
        We split and load weights into:
        - self.vlm
        - self.expert
        - self.action_in_proj
        - self.action_out_proj
        We skip unrelated modules (e.g. diffusion/action_space).
        """
        vlm_weights = []
        expert_weights = []
        action_in_proj_weights = []
        action_out_proj_weights = []

        for name, tensor in weights:
            if name.startswith("vlm."):
                vlm_weights.append((name[len("vlm.") :], tensor))
                continue

            if name.startswith("expert."):
                expert_weights.append((name[len("expert.") :], tensor))
                continue

            if name.startswith("action_in_proj."):
                action_in_proj_weights.append(
                    (name[len("action_in_proj.") :], tensor)
                )
                continue

            if name.startswith("action_out_proj."):
                action_out_proj_weights.append(
                    (name[len("action_out_proj.") :], tensor)
                )
                continue

            if name.startswith(("action_space.", "diffusion.")):
                logger.debug(f"Skipping weight: {name} (module not implemented)")
                continue

            # Keep compatibility for checkpoints without explicit "vlm." prefix.
            vlm_weights.append((name, tensor))

        # 1) Load VLM weights.
        self.vlm.load_weights(iter(vlm_weights))
        logger.info(f"AlpamayoR1: loaded {len(vlm_weights)} vlm tensors")

        # 2) Load expert weights.
        self._load_expert_weights(expert_weights)

        # 3) Load action projection modules.
        self._load_plain_module_weights(
            self.action_in_proj, "action_in_proj", action_in_proj_weights
        )
        self._load_plain_module_weights(
            self.action_out_proj, "action_out_proj", action_out_proj_weights
        )

# Entry point for SGLang model registry
EntryClass = AlpamayoR1
