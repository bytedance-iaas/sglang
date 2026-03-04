
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
import os
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

import einops
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.utils import logger
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.managers.schedule_batch import MultimodalInputs
from .action_in_proj import PerWaypointActionInProjV2
from .unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace


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


        # Build action projection modules from Alpamayo config to match checkpoint shapes.
        action_in_proj_cfg = config.action_in_proj_cfg
        traj_tokenizer_cfg = config.traj_tokenizer_cfg
        action_space_cfg = config.traj_tokenizer_cfg["action_space_cfg"]
        n_waypoints = action_space_cfg["n_waypoints"]
        action_dim = len(traj_tokenizer_cfg["dims_max"])


        # Instantiate action space (UnicycleAccelCurvatureActionSpace)
        action_space_kwargs = {
            k: v for k, v in action_space_cfg.items()
            if k not in ("_target_", "_recursive_", "n_waypoints")
        }
        self.action_space = UnicycleAccelCurvatureActionSpace(
            **action_space_kwargs,
        )

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

        # Set expert attention layers to ENCODER_ONLY:
        #   - Bidirectional (non-causal) attention among action tokens
        #   - Reads VLM's KV cache via shared layer_ids (both 0..N-1)
        #   - Does NOT write to KV cache (FlashInfer auto-skips set_kv_buffer for ENCODER_ONLY)
        for layer in self.expert.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "attn"):
                layer.self_attn.attn.attn_type = AttentionType.ENCODER_ONLY

        # Flow matching parameters
        self.n_diffusion_tokens = n_waypoints
        self.action_dims = [n_waypoints, action_dim]
        diffusion_cfg = getattr(config, "diffusion_cfg", {}) or {}
        self.num_inference_steps = diffusion_cfg.get("num_inference_steps", 10)

        # Flow matching debug switches.
        # Can be enabled by config.flow_matching_debug=True or env var:
        #   SGLANG_ALPAMAYO_FM_DEBUG=1
        self.flow_matching_debug = bool(
            getattr(config, "flow_matching_debug", False)
            or os.getenv("SGLANG_ALPAMAYO_FM_DEBUG", "0") == "1"
        )
        # Max number of Euler steps to log in detail (+ always logs the last step).
        self.flow_matching_debug_max_steps = int(
            os.getenv("SGLANG_ALPAMAYO_FM_DEBUG_MAX_STEPS", "3")
        )
        # Debug option: do not read VLM KV cache in expert flow-matching branch.
        # This is useful to isolate whether run-to-run drift comes from VLM KV states.
        self.flow_matching_disable_vlm_kv = (
            os.getenv("SGLANG_ALPAMAYO_FM_DISABLE_VLM_KV", "0") == "1"
        )

    def _fm_should_log_step(self, step_i: int) -> bool:
        if not self.flow_matching_debug:
            return False
        return (
            step_i < self.flow_matching_debug_max_steps
            or step_i == self.num_inference_steps - 1
        )

    def _fm_tensor_fingerprint(self, tag: str, tensor: torch.Tensor, sample_size: int = 4096):
        """Print compact numeric fingerprint for drift debugging."""
        if not self.flow_matching_debug:
            return

        with torch.no_grad():
            x = tensor.detach().float().reshape(-1)
            if x.numel() == 0:
                logger.info(
                    f"FM_DEBUG {tag}: empty shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                )
                return

            n = min(sample_size, x.numel())
            s = x[:n]
            idx = torch.arange(1, n + 1, device=s.device, dtype=s.dtype)

            mean = s.mean().item()
            std = s.std(unbiased=False).item()
            min_v = s.min().item()
            max_v = s.max().item()
            l2 = torch.linalg.vector_norm(s).item()
            checksum = (s * idx).sum().item()

        logger.info(
            "FM_DEBUG %s: shape=%s dtype=%s n=%d mean=%.8f std=%.8f min=%.8f max=%.8f l2=%.8f checksum=%.8f",
            tag,
            tuple(tensor.shape),
            tensor.dtype,
            n,
            mean,
            std,
            min_v,
            max_v,
            l2,
            checksum,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        forward_batch: "ForwardBatch",
        **kwargs,
    ):
        ret = self.vlm(input_ids, positions, forward_batch, **kwargs)

        if forward_batch.forward_mode.is_decode():
            bstar = int(input_ids.shape[0])
            active_indices = []
            reqs = getattr(forward_batch, "reqs", None)
            if self.flow_matching_debug:
                # Log every decoded token ID to detect VLM non-determinism across runs.
                # If the same request produces different token IDs between runs,
                # that is the root cause of expert_hidden_flat divergence.
                logger.info(
                    "FM_DEBUG decode_token: input_ids=%s seq_lens=%s",
                    input_ids.tolist(),
                    forward_batch.seq_lens.tolist(),
                )
            for i in range(bstar):
                has_history_traj = False
                if reqs is not None and i < len(reqs):
                    has_history_traj = getattr(reqs[i], "history_traj", None) is not None

                should_trigger_flow_matching = (
                    input_ids[i] == self.traj_future_start_token_id
                    or (
                        has_history_traj
                        and input_ids[i] == self.traj_force_stop_token_id
                    )
                )

                if should_trigger_flow_matching:
                    # Avoid recomputing if already attached once for this request.
                    # if (
                    #     reqs is not None
                    #     and i < len(reqs)
                    #     and getattr(reqs[i], "customized_info", None) is not None
                    #     and "traj_xyz" in reqs[i].customized_info
                    # ):
                    #     continue

                    # Force generation to stop immediately
                    ret.next_token_logits[i, :] = float("-inf")
                    ret.next_token_logits[i, self.traj_force_stop_token_id] = 0.0
                    active_indices.append(i)

            if active_indices:
                sampled_actions = self._run_flow_matching(
                    active_indices, forward_batch
                )
                # Convert sampled actions → trajectories and write to req.customized_info
                self._attach_traj_to_reqs(
                    sampled_actions, active_indices, forward_batch
                )

        return ret

    def _attach_traj_to_reqs(
        self,
        sampled_actions: torch.Tensor,
        active_indices: List[int],
        forward_batch: "ForwardBatch",
    ) -> None:
        """Convert sampled actions to trajectories and write to req.customized_info.

        Args:
            sampled_actions: (bstar, n_waypoints, action_dim) on GPU.
            active_indices: batch-slot indices of active requests.
            forward_batch: ForwardBatch carrying per-req objects.
        """
        reqs = getattr(forward_batch, "reqs", None)
        if reqs is None:
            logger.warning("_attach_traj_to_reqs: forward_batch.reqs is None; skipping action_to_traj")
            return

        device = sampled_actions.device

        for j, slot_i in enumerate(active_indices):
            req = reqs[slot_i]
            history_traj = getattr(req, "history_traj", None) or {}

            hist_xyz_raw = history_traj.get("ego_history_xyz")
            hist_rot_raw = history_traj.get("ego_history_rot")

            if hist_xyz_raw is None or hist_rot_raw is None:
                logger.warning(
                    f"_attach_traj_to_reqs: req {req.rid} missing history_traj; "
                    "skipping action_to_traj for this request"
                )
                continue

            # Convert to tensor if needed and move to GPU.
            # Use float32 to match action_space precision (not bfloat16).
            if not isinstance(hist_xyz_raw, torch.Tensor):
                hist_xyz = torch.tensor(hist_xyz_raw, dtype=torch.float32, device=device)
            else:
                hist_xyz = hist_xyz_raw.to(dtype=torch.float32, device=device)

            if not isinstance(hist_rot_raw, torch.Tensor):
                hist_rot = torch.tensor(hist_rot_raw, dtype=torch.float32, device=device)
            else:
                hist_rot = hist_rot_raw.to(dtype=torch.float32, device=device)

            # Ensure shape: (T, 3) and (T, 3, 3) – add batch dim for action_to_traj
            if hist_xyz.dim() == 2:     # (T, 3)
                hist_xyz = hist_xyz.unsqueeze(0)   # (1, T, 3)
            if hist_rot.dim() == 3:     # (T, 3, 3)
                hist_rot = hist_rot.unsqueeze(0)   # (1, T, 3, 3)

            # sampled_actions[j]: (n_waypoints, action_dim) → unsqueeze batch
            action_j = sampled_actions[j].unsqueeze(0).float()  # (1, n_waypoints, 2)

            with torch.no_grad():
                pred_xyz, pred_rot = self.action_space.action_to_traj(
                    action_j, hist_xyz, hist_rot
                )  # (1, n_waypoints, 3), (1, n_waypoints, 3, 3)

            if req.customized_info is None:
                req.customized_info = {}
            req.customized_info["traj_xyz"] = pred_xyz[0].cpu().tolist()
            req.customized_info["traj_rot"] = pred_rot[0].cpu().tolist()

        logger.info(
            f"_attach_traj_to_reqs: converted {len(active_indices)} sampled_actions to trajectories"
        )

    def _build_expert_forward_batch(
        self,
        active_indices: List[int],
        forward_batch: ForwardBatch,
        mrope_positions: torch.Tensor,
    ) -> ForwardBatch:
        """Build an EXTEND-mode ForwardBatch for the expert model.

        The expert reads the VLM's KV cache (shared layer_ids) and processes
        n_diffusion_tokens new action embeddings per request, using bidirectional
        attention (ENCODER_ONLY) without writing to KV cache.

        Args:
            active_indices: Indices of requests that need flow matching.
            forward_batch: The original decode-mode ForwardBatch from VLM.
            mrope_positions: Multimodal RoPE positions for action tokens,
                shape (3, bstar * n_diff).

        Returns:
            A ForwardBatch configured for EXTEND mode with the expert.
        """
        device = forward_batch.seq_lens.device
        bstar = len(active_indices)
        n_diff = self.n_diffusion_tokens
        idx_tensor = torch.tensor(active_indices, device=device, dtype=torch.long)

        # VLM's current seq_lens for active requests (tokens already in KV cache)
        vlm_seq_lens = forward_batch.seq_lens[idx_tensor]

        # Expert sees: prefix (VLM cached) + new action tokens
        # In debug mode, we can disable VLM KV reading for reproducibility checks.
        if self.flow_matching_disable_vlm_kv:
            expert_seq_lens = torch.full_like(vlm_seq_lens, n_diff)
            extend_prefix_lens = torch.zeros_like(vlm_seq_lens, dtype=torch.int32)
        else:
            expert_seq_lens = vlm_seq_lens + n_diff
            extend_prefix_lens = vlm_seq_lens.to(torch.int32)
        extend_seq_lens = torch.full((bstar,), n_diff, dtype=torch.int32, device=device)

        # Cumulative start locations for the new tokens (each request contributes n_diff)
        extend_start_loc = torch.arange(
            0, bstar * n_diff, n_diff, dtype=torch.int32, device=device
        )

        # Dummy out_cache_loc — ENCODER_ONLY skips KV cache writes,
        # but the tensor must exist with the correct shape.
        out_cache_loc = torch.zeros(
            bstar * n_diff, dtype=torch.int32, device=device
        )

        expert_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=bstar,
            input_ids=torch.zeros(bstar * n_diff, dtype=torch.long, device=device),
            req_pool_indices=forward_batch.req_pool_indices[idx_tensor],
            seq_lens=expert_seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(expert_seq_lens.sum()),
            extend_num_tokens=bstar * n_diff,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_seq_lens_cpu=extend_seq_lens.cpu().tolist(),
            extend_prefix_lens_cpu=extend_prefix_lens.cpu().tolist(),
            seq_lens_cpu=expert_seq_lens.cpu(),
            # Share the VLM's KV cache pool and attention backend
            req_to_token_pool=forward_batch.req_to_token_pool,
            token_to_kv_pool=forward_batch.token_to_kv_pool,
            attn_backend=forward_batch.attn_backend,
            # Multimodal RoPE positions for the action tokens
            mrope_positions=mrope_positions,
        )
        return expert_batch

    @torch.no_grad()
    def _run_flow_matching(
        self,
        active_indices: List[int],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Run flow matching (Euler integration) for requests that generated
        <traj_future_start>.

        This implements the same logic as the original alpamayo step_fn:
        1. Sample Gaussian noise x ~ N(0, I) in action space
        2. For each Euler step t_i -> t_{i+1}:
            a. action_in_proj(x, t) -> token embeddings [bstar, n_diff, hidden]
            b. expert forward (bidirectional, reads VLM KV cache) -> hidden states
            c. action_out_proj -> predicted velocity field v
            d. x = x + dt * v
        3. Return final x as the sampled action

        Args:
            active_indices: Indices of requests needing flow matching.
            forward_batch: The decode-mode ForwardBatch.

        Returns:
            Sampled actions of shape (bstar, n_waypoints, action_dim).
        """
        device = forward_batch.seq_lens.device
        bstar = len(active_indices)
        n_diff = self.n_diffusion_tokens

        # --- 1. Compute mRoPE positions for the action tokens ---
        # During decode, mrope_positions has shape (3, total_batch_size)
        # For each active request, action tokens get positions starting after
        # the current decode position (which is <traj_future_start>).
        positions_list = []
        for idx in active_indices:
            # Current mrope position of the <traj_future_start> token: shape (3,)
            current_mrope = forward_batch.mrope_positions[:, idx]  # (3,)
            # Action tokens: pos+1, pos+2, ..., pos+n_diff
            action_pos = (
                current_mrope.unsqueeze(1)
                + 1
                + torch.arange(n_diff, device=device).unsqueeze(0)
            )  # (3, n_diff)
            positions_list.append(action_pos)
        mrope_positions = torch.cat(positions_list, dim=1)  # (3, bstar * n_diff)

        # --- 2. Build the expert ForwardBatch (EXTEND mode) ---
        expert_batch = self._build_expert_forward_batch(
            active_indices, forward_batch, mrope_positions
        )

        if self.flow_matching_debug:
            logger.info(
                "FM_DEBUG meta: bstar=%d n_diff=%d action_dims=%s num_steps=%d disable_vlm_kv=%s",
                bstar,
                n_diff,
                self.action_dims,
                self.num_inference_steps,
                self.flow_matching_disable_vlm_kv,
            )
            self._fm_tensor_fingerprint("mrope_positions", mrope_positions)
            self._fm_tensor_fingerprint("forward_batch.seq_lens", forward_batch.seq_lens)
            self._fm_tensor_fingerprint("expert_batch.seq_lens", expert_batch.seq_lens)

        # Initialize attention metadata (FlashInfer wrapper plans / Triton metadata)
        # This is safe because the VLM's decode is already finished;
        # the runtime will re-init metadata for the next batch.
        #
        # BUG FIX: The expert uses ENCODER_ONLY attention (bidirectional self-attention
        # among action tokens + cross-attention to VLM prefix KV).  FlashInfer's EXTEND
        # mode has two code paths:
        #
        #   use_ragged=True  (correct for ENCODER_ONLY):
        #     - Ragged wrapper:  Q(64) × local K,V(64) computed in-memory, non-causal
        #     - Paged wrapper:   Q(64) × VLM prefix KV(3190), non-causal
        #     - Merge with log-sum-exp
        #     - ENCODER_ONLY suppresses KV cache write and forces causal=False
        #     - paged_kernel_lens = prefix_lens = 3190  (no garbage)
        #
        #   use_ragged=False (buggy for ENCODER_ONLY, forced by is_multimodal + enable_deterministic):
        #     - Paged wrapper only, causal=True (wrong!), save_kv_cache=True (wrong!)
        #     - out_cache_loc=zeros → writes all 64 action-token KVs to physical slot 0,
        #       corrupting VLM token 0's KV stored there
        #     - paged_kernel_lens = seq_lens = 3254 → req_to_token[3190..3253] are
        #       uninitialized, all pointing to slot 0 which holds corrupted KV
        #     - Between runs, VLM tokens land on different physical slots, so the
        #       corruption propagates differently → non-deterministic output
        #
        # Fix: temporarily clear is_multimodal and enable_deterministic so
        # init_forward_metadata chooses use_ragged=True for this expert batch.
        # The expert's paged attention (3190 prefix tokens < tile_size 4096) has no
        # split-K and processes the same KV values in the same logical order each run
        # → fully deterministic.

        backend = forward_batch.attn_backend
        logger.info(f"attention_backend={backend}")
        backend.init_forward_metadata(expert_batch)

        # _orig_is_multimodal = getattr(backend, "is_multimodal", False)
        # _orig_enable_deterministic = getattr(backend, "enable_deterministic", False)
        # backend.is_multimodal = False
        # backend.enable_deterministic = False
        # try:
        #     backend.init_forward_metadata(expert_batch)
        # finally:
        #     backend.is_multimodal = _orig_is_multimodal
        #     backend.enable_deterministic = _orig_enable_deterministic

        # --- 3. Euler integration loop ---
        # Match reference FlowMatching._euler: x is fp32 (default dtype)
        # so Euler accumulation x = x + dt * v stays in fp32 throughout.

        torch.cuda.manual_seed_all(42)  # for reproducibility in testing
        x = torch.randn(
            bstar, *self.action_dims, device=device,
        )

        logger.info(f"start flowmatching {x[0,:2]}")
        self._fm_tensor_fingerprint("x_init", x)
                    
        time_steps = torch.linspace(
            0.0, 1.0, self.num_inference_steps + 1, device=device
        )

        for step_i in range(self.num_inference_steps):
            dt = time_steps[step_i + 1] - time_steps[step_i]
            t = time_steps[step_i].view(1, 1, 1).expand(bstar, 1, 1)

            # --- step_fn start ---
            # a. Project noisy action + timestep -> expert token embeddings
            #    x: (bstar, n_waypoints, action_dim)
            #    t: (bstar, 1, 1)
            #    output: (bstar, n_diff, hidden_size)
            future_token_embeds = self.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(
                    bstar, n_diff, -1
                )

            # b. Flatten for sglang's model forward: (bstar * n_diff, hidden_size)
            input_embeds_flat = future_token_embeds.reshape(
                bstar * n_diff, -1
            )

            if self._fm_should_log_step(step_i):
                self._fm_tensor_fingerprint(f"step{step_i}.x_in", x)
                self._fm_tensor_fingerprint(
                    f"step{step_i}.future_token_embeds", future_token_embeds
                )
                self._fm_tensor_fingerprint(
                    f"step{step_i}.input_embeds_flat", input_embeds_flat
                )

            # c. Run expert: bidirectional attention over action tokens,
            #    attending to VLM's cached prefix KV via shared layer_ids.
            #    ENCODER_ONLY ensures:
            #      - Ragged attn: action tokens attend bidirectionally to each other
            #      - Paged attn:  action tokens attend to VLM's cached KV
            #      - No KV is written to the cache pool
            expert_hidden = self.expert(
                input_ids=None,
                positions=mrope_positions,
                forward_batch=expert_batch,
                input_embeds=input_embeds_flat,
            )  # (bstar * n_diff, hidden_size)

            if self._fm_should_log_step(step_i):
                self._fm_tensor_fingerprint(
                    f"step{step_i}.expert_hidden_flat", expert_hidden
                )

            # d. Project to action space
            expert_hidden = expert_hidden.view(
                bstar, n_diff, -1
            )  # (bstar, n_diff, hidden_size)
            pred = self.action_out_proj(expert_hidden)  # (bstar, n_diff, action_dim)
            pred = pred.view(bstar, *self.action_dims)

            if self._fm_should_log_step(step_i):
                self._fm_tensor_fingerprint(f"step{step_i}.pred", pred)
            # --- step_fn end ---

            # Euler update: x_{i+1} = x_i + dt * v(x_i, t_i)
            x = x + dt * pred
            if self._fm_should_log_step(step_i):
                self._fm_tensor_fingerprint(f"step{step_i}.x_out", x)

        # ---- Diagnostic: raw sampled actions (ALWAYS logged) ----
        with torch.no_grad():
            accel_ch = x[..., 0]  # (bstar, n_waypoints)
            kappa_ch = x[..., 1]  # (bstar, n_waypoints)
            logger.info(
                "FM_DIAG sampled_action: accel  mean=%.6f std=%.6f min=%.6f max=%.6f",
                accel_ch.mean().item(), accel_ch.std().item(),
                accel_ch.min().item(), accel_ch.max().item(),
            )
            logger.info(
                "FM_DIAG sampled_action: kappa  mean=%.6f std=%.6f min=%.6f max=%.6f",
                kappa_ch.mean().item(), kappa_ch.std().item(),
                kappa_ch.min().item(), kappa_ch.max().item(),
            )
            logger.info(
                "FM_DIAG action_space buffers: accel_mean=%.8f accel_std=%.8f "
                "curvature_mean=%.8f curvature_std=%.8f",
                self.action_space.accel_mean.item(),
                self.action_space.accel_std.item(),
                self.action_space.curvature_mean.item(),
                self.action_space.curvature_std.item(),
            )
        return x  # (bstar, n_waypoints, action_dim)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.vlm.pad_input_ids(input_ids, mm_inputs)


    
    def _load_expert_weights(
        self,
        expert_weights: Iterable[Tuple[str, torch.Tensor]],
        strict: bool = True,
    ):
        """Load expert (Qwen3 text backbone) weights.

        Returns:
            A tuple ``(loaded_cnt, missing_params, unexpected_ckpt_keys)``.
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.expert.named_parameters())
        expected_param_names = set(params_dict.keys())
        loaded_full_params = set()
        loaded_stacked_shards = defaultdict(set)
        unexpected_ckpt_keys = []
        loaded_cnt = 0
        seen_any_expert_weight = False

        for name, loaded_weight in expert_weights:
            seen_any_expert_weight = True
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
                    unexpected_ckpt_keys.append(name)
                    break

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_stacked_shards[name].add(shard_id)
                loaded_cnt += 1
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Expert parameter {name} not found; skipping")
                    unexpected_ckpt_keys.append(name)
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_full_params.add(name)
                loaded_cnt += 1

        missing_params = []
        for pname in expected_param_names:
            if "qkv_proj" in pname:
                if loaded_stacked_shards.get(pname, set()) != {"q", "k", "v"}:
                    missing_params.append(pname)
                continue
            if "gate_up_proj" in pname:
                if loaded_stacked_shards.get(pname, set()) != {0, 1}:
                    missing_params.append(pname)
                continue
            if pname not in loaded_full_params:
                missing_params.append(pname)

        logger.info(f"AlpamayoR1: loaded {loaded_cnt} expert tensors")
        if strict:
            if not seen_any_expert_weight:
                raise RuntimeError(
                    "AlpamayoR1 strict load failed: no checkpoint weights were routed "
                    "to expert."
                )
            if missing_params or unexpected_ckpt_keys:
                raise RuntimeError(
                    "AlpamayoR1 strict expert load failed: "
                    f"missing={len(missing_params)}, "
                    f"unexpected={len(unexpected_ckpt_keys)}. "
                    f"Sample missing={missing_params[:8]}, "
                    f"sample unexpected={unexpected_ckpt_keys[:8]}"
                )
        return loaded_cnt, missing_params, unexpected_ckpt_keys

    def _load_plain_module_weights(
        self,
        module: nn.Module,
        module_name: str,
        module_weights: Iterable[Tuple[str, torch.Tensor]],
        strict: bool = True,
    ):
        state_dict = {name: tensor for name, tensor in module_weights}
        if not state_dict:
            msg = f"AlpamayoR1: no weights found for {module_name}"
            if strict:
                raise RuntimeError(f"AlpamayoR1 strict load failed: {msg}")
            logger.info(msg)
            return

        incompatible = module.load_state_dict(state_dict, strict=strict)
        if incompatible.missing_keys:
            logger.warning(
                f"AlpamayoR1: {module_name} missing keys: {incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            logger.warning(
                f"AlpamayoR1: {module_name} unexpected keys: {incompatible.unexpected_keys}"
            )
        logger.info(f"AlpamayoR1: loaded {len(state_dict)} tensors into {module_name}")

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        strict: bool = True,
    ):
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
        action_space_weights = []
        skipped_weights = []

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

            if name.startswith("action_space."):
                action_space_weights.append(
                    (name[len("action_space."):], tensor)
                )
                continue

            # Keep compatibility for checkpoints without explicit "vlm." prefix.
            vlm_weights.append((name, tensor))

        # 1) Load VLM weights.
        if strict and not vlm_weights:
            raise RuntimeError(
                "AlpamayoR1 strict load failed: no checkpoint weights were routed to vlm."
            )
        self.vlm.load_weights(iter(vlm_weights))
        logger.info(f"AlpamayoR1: loaded {len(vlm_weights)} vlm tensors")

        # 2) Load expert weights.
        expert_loaded_cnt, expert_missing, expert_unexpected = self._load_expert_weights(
            expert_weights, strict=strict
        )

        # 3) Load action space buffers (accel_mean, accel_std, etc.).
        # IMPORTANT: convert action_space to float32 BEFORE loading weights,
        # so that the float32 checkpoint values are not truncated to bfloat16.
        # The original alpamayo code keeps action_space in float32 (keep_same_dtype
        # only applies to diffusion, action_in_proj, action_out_proj).
        self.action_space = self.action_space.float()
        if action_space_weights:
            self._load_plain_module_weights(
                self.action_space,
                "action_space",
                action_space_weights,
                strict=False,  # scalar hyperparams are not in state_dict
            )

        # 4) Load action projection modules.
        self._load_plain_module_weights(
            self.action_in_proj,
            "action_in_proj",
            action_in_proj_weights,
            strict=strict,
        )
        self._load_plain_module_weights(
            self.action_out_proj,
            "action_out_proj",
            action_out_proj_weights,
            strict=strict,
        )

        logger.info(
            "AlpamayoR1 load summary: "
            f"strict={strict}, "
            f"vlm_ckpt_tensors={len(vlm_weights)}, "
            f"expert_ckpt_tensors={len(expert_weights)}, "
            f"expert_loaded_tensors={expert_loaded_cnt}, "
            f"expert_missing_params={len(expert_missing)}, "
            f"expert_unexpected_ckpt_keys={len(expert_unexpected)}, "
            f"action_in_proj_ckpt_tensors={len(action_in_proj_weights)}, "
            f"action_out_proj_ckpt_tensors={len(action_out_proj_weights)}, "
            f"action_space_ckpt_tensors={len(action_space_weights)}, "
            f"skipped_tensors={len(skipped_weights)}"
        )

# Entry point for SGLang model registry
EntryClass = AlpamayoR1
