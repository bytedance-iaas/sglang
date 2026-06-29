# Adapted from the DSpark reference implementation (HF) but implemented with
# SGLang primitives (RadixAttention + SGLang KV cache). This model intentionally
# does not include token embeddings or an LM head; DSpark uses the target model's
# embedding/lm_head.

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import _is_npu
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer, DeepseekV4ForCausalLM
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.speculative.dspark_utils import (
    can_dspark_slice_qkv_weight,
    parse_dspark_draft_config,
)
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class DSparkAttention(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        tp_size = int(get_tensor_model_parallel_world_size())
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", total_num_heads)
        )
        head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))

        self.hidden_size = hidden_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        assert self.total_num_heads % tp_size == 0, (
            f"DSparkAttention requires total_num_heads divisible by tp_size. "
            f"total_num_heads={self.total_num_heads}, tp_size={tp_size}."
        )
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0, (
                f"DSparkAttention requires total_num_kv_heads divisible by tp_size when >= tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        else:
            assert tp_size % self.total_num_kv_heads == 0, (
                f"DSparkAttention requires tp_size divisible by total_num_kv_heads when total_num_kv_heads < tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=attention_bias,
            prefix="qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            prefix="o_proj",
        )

        # Per-head Q/K RMSNorm, matching HF Qwen3.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        rope_theta = float(getattr(config, "rope_theta", 1000000))
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_is_neox_style = bool(
            getattr(
                config, "rope_is_neox_style", getattr(config, "is_neox_style", True)
            )
        )
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )

        self.scaling = head_dim**-0.5
        # DSpark uses non-causal attention over the draft block.
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project hidden_states to K/V only (skip Q).

        This is used by DSpark to materialize ctx tokens into the draft KV cache:
        we only need K/V for the cached tokens; Q is never consumed.
        """
        # Fast path for unquantized weights: slice the fused QKV weight and run one GEMM.
        can_slice_qkv_weight, _ = can_dspark_slice_qkv_weight(self.qkv_proj)
        if can_slice_qkv_weight:
            kv_slice = slice(self.q_size, self.q_size + 2 * self.kv_size)
            weight = self.qkv_proj.weight[kv_slice]
            bias = (
                self.qkv_proj.bias[kv_slice] if self.qkv_proj.bias is not None else None
            )
            kv = F.linear(hidden_states, weight, bias)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            return k, v

        # Fallback: compute full QKV and discard Q (keeps compatibility with quantized weights).
        qkv, _ = self.qkv_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Match K shape so RoPE kernel head-count check passes on all backends.
        dummy_q = k.new_empty(k.shape)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class DSparkMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(getattr(config, "intermediate_size", 0))
        if intermediate_size <= 0:
            raise ValueError(
                f"Invalid intermediate_size={intermediate_size} for DSpark MLP."
            )

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix="gate_up_proj" if not prefix else f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix="down_proj" if not prefix else f"{prefix}.down_proj",
        )
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported DSpark activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DSparkDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = DSparkAttention(config=config, layer_id=layer_id)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DSparkMLP(config=config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            # Keep return types consistent for upstream callers.
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        # Pre-norm attention with fused residual+norm when possible (Qwen3-style).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DSparkDraftModel(nn.Module):
    """SGLang DSpark draft model (no embedding / lm_head weights).

    The checkpoint provides:
      - transformer weights for `layers.*`
      - `fc.weight`, `hidden_norm.weight` for projecting target context features
      - `norm.weight` for final normalization
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.layers = nn.ModuleList(
            [DSparkDecoderLayer(config=config, layer_id=i) for i in range(num_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Project per-token target context features:
        # concat(K * hidden_size) -> hidden_size, where K is the number of target-layer
        # feature tensors concatenated per token (not necessarily equal to num_layers).
        draft_config = parse_dspark_draft_config(draft_hf_config=config)
        target_num_layers = (
            int(draft_config.num_target_layers)
            if draft_config.num_target_layers is not None
            else num_layers
        )
        target_layer_ids = draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_layers, draft_num_layers=num_layers
        )
        num_context_features = len(target_layer_ids)
        dspark_cfg = getattr(config, "dspark_config", {}) or {}
        if not isinstance(dspark_cfg, dict):
            try:
                dspark_cfg = dict(dspark_cfg)
            except Exception:
                dspark_cfg = {}
        target_hidden_size = int(
            getattr(
                config,
                "target_hidden_size",
                dspark_cfg.get(
                    "target_hidden_size",
                    getattr(
                        config,
                        "context_hidden_size",
                        dspark_cfg.get("context_hidden_size", hidden_size),
                    ),
                ),
            )
        )

        self.num_context_features = int(num_context_features)
        self.target_hidden_size = int(target_hidden_size)
        self.fc = nn.Linear(
            self.num_context_features * self.target_hidden_size,
            hidden_size,
            bias=False,
        )
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.block_size = draft_config.resolve_block_size(default=16)
        self.vocab_size = int(getattr(config, "vocab_size"))

        self.markov_rank = int(getattr(config, "markov_rank", 0) or 0)
        self.markov_head_type = str(getattr(config, "markov_head_type", "vanilla"))
        if self.markov_rank > 0:
            self.markov_w1 = VocabParallelEmbedding(
                self.vocab_size,
                self.markov_rank,
                prefix="markov_w1",
            )
            self.markov_w2 = ParallelLMHead(
                self.vocab_size,
                self.markov_rank,
                quant_config=None,
                prefix="markov_w2",
            )
            if self.markov_head_type == "gated":
                self.gate_proj = nn.Linear(hidden_size + self.markov_rank, self.markov_rank)
            elif self.markov_head_type == "rnn":
                self.joint_proj = nn.Linear(
                    2 * self.markov_rank + hidden_size,
                    3 * self.markov_rank,
                )
            elif self.markov_head_type != "vanilla":
                raise ValueError(
                    f"Unsupported DSPARK markov_head_type={self.markov_head_type!r}."
                )
        else:
            self.markov_w1 = None
            self.markov_w2 = None

        self.enable_confidence_head = bool(
            getattr(config, "enable_confidence_head", False)
        )
        self.confidence_head_with_markov = bool(
            getattr(config, "confidence_head_with_markov", False)
        )
        self.confidence_head = None
        if self.enable_confidence_head:
            confidence_in = hidden_size
            if self.confidence_head_with_markov:
                if self.markov_rank <= 0:
                    raise ValueError(
                        "DSPARK confidence_head_with_markov requires markov_rank > 0."
                    )
                confidence_in += self.markov_rank
            self.confidence_head = nn.Linear(confidence_in, 1)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hidden states into draft hidden_size."""
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DSPARK target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, target_hidden_size={self.target_hidden_size}), "
                f"but got shape={tuple(target_hidden.shape)}. "
                "This usually means the target model is capturing a different number of layer features than "
                "the draft checkpoint/config expects."
            )
        return self.hidden_norm(self.fc(target_hidden))

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.markov_w1 is None:
            raise RuntimeError("DSPARK markov head is disabled.")
        return self.markov_w1(token_ids.long())

    def compute_markov_bias_local(
        self,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.markov_rank <= 0:
            weight = self.markov_w2.weight  # type: ignore[union-attr]
            return hidden_states.new_zeros((hidden_states.shape[0], weight.shape[0])), state

        prev_embeddings = self.get_prev_embeddings(prev_token_ids).to(
            dtype=hidden_states.dtype
        )
        latent = prev_embeddings
        new_state = state
        if self.markov_head_type == "gated":
            gate = torch.sigmoid(
                self.gate_proj(torch.cat([hidden_states, prev_embeddings], dim=-1))
            ).to(dtype=prev_embeddings.dtype)
            latent = gate * prev_embeddings
        elif self.markov_head_type == "rnn":
            if state is None:
                state = torch.zeros_like(prev_embeddings)
            z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
            gate_raw, candidate_raw, output_raw = self.joint_proj(z).chunk(3, dim=-1)
            gate = torch.sigmoid(gate_raw)
            candidate = torch.tanh(candidate_raw)
            new_state = gate * state + (1.0 - gate) * candidate
            latent = torch.tanh(output_raw)

        weight = self.markov_w2.weight  # type: ignore[union-attr]
        bias = torch.matmul(latent.to(dtype=weight.dtype), weight.t())
        return bias, new_state

    def predict_confidence(
        self,
        *,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        features = hidden_states
        if self.confidence_head_with_markov:
            prev_embeddings = self.get_prev_embeddings(prev_token_ids).to(
                dtype=hidden_states.dtype
            )
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return self.confidence_head(features).squeeze(-1).float()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            raise ValueError(
                "DSparkDraftModel requires `input_embeds` (use the target embedding)."
            )
        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if hidden_states.numel() != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped_name = name[len("model.") :]
                if stripped_name in params_dict:
                    return stripped_name
            else:
                prefixed_name = f"model.{name}"
                if prefixed_name in params_dict:
                    return prefixed_name
            return None

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                resolved_name = resolve_param_name(mapped_name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    # Ignore unexpected weights (e.g., HF rotary caches).
                    continue
                param = params_dict[resolved_name]
                if resolved_name.endswith("fc.weight") and tuple(
                    loaded_weight.shape
                ) != tuple(param.shape):
                    raise ValueError(
                        "DSPARK fc.weight shape mismatch. This usually means the draft checkpoint's "
                        "number of context features (K) does not match this config. "
                        f"Expected fc.weight.shape={tuple(param.shape)} "
                        f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                        f"but got {tuple(loaded_weight.shape)} for weight '{name}'."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


class DeepseekV4DSparkCore(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.hc_mult = int(config.hc_mult)
        self.rms_norm_eps = float(config.rms_norm_eps)

        draft_config = parse_dspark_draft_config(draft_hf_config=config)
        self.block_size = int(draft_config.resolve_block_size(default=5))
        self.noise_token_id = int(draft_config.mask_token_id)
        target_num_hidden_layers = int(
            getattr(
                config,
                "dspark_target_num_hidden_layers",
                config.num_hidden_layers,
            )
        )
        self.target_layer_ids = draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_hidden_layers,
            draft_num_layers=len(getattr(config, "dspark_target_layer_ids", []) or []),
        )
        self.num_hidden_layers = len(self.target_layer_ids)
        if self.num_hidden_layers <= 0:
            self.num_hidden_layers = int(getattr(config, "num_nextn_predict_layers", 1))

        self.layers = nn.ModuleList()
        for i in range(self.num_hidden_layers):
            layer = DeepseekV4DecoderLayer(
                config,
                layer_id=i,
                quant_config=quant_config,
                is_nextn=True,
                prefix=add_prefix(f"layers.{i}", prefix),
                alt_streams=None,
                compress_ratio_override=0,
            )
            self.layers.append(layer)

        self.layers[0].main_proj = ReplicatedLinear(
            self.hidden_size * len(self.target_layer_ids),
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("layers.0.main_proj", prefix),
        )
        self.layers[0].main_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        last_layer = self.layers[-1]
        last_layer.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        hc_dim = self.hc_mult * self.hidden_size
        last_layer.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32)
        )
        last_layer.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32)
        )
        last_layer.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        markov_rank = int(
            getattr(config, "dspark_markov_rank", getattr(config, "markov_rank", 0))
            or 0
        )
        last_layer.markov_head = nn.Module()
        last_layer.markov_head.markov_w1 = VocabParallelEmbedding(
            int(config.vocab_size),
            markov_rank,
            prefix=add_prefix(
                f"layers.{self.num_hidden_layers - 1}.markov_head.markov_w1",
                prefix,
            ),
        )
        last_layer.markov_head.markov_w2 = ParallelLMHead(
            int(config.vocab_size),
            markov_rank,
            quant_config=None,
            prefix=add_prefix(
                f"layers.{self.num_hidden_layers - 1}.markov_head.markov_w2",
                prefix,
            ),
        )
        last_layer.confidence_head = nn.Module()
        last_layer.confidence_head.proj = nn.Linear(
            self.hidden_size + markov_rank, 1, bias=False
        )

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.rms_norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + float(self.config.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        out, _ = self.layers[0].main_proj(target_hidden)
        return self.layers[0].main_norm(out)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            raise ValueError("DeepseekV4DSparkModel requires input_embeds.")
        hidden_states = input_embeds.unsqueeze(1).repeat(1, self.hc_mult, 1)
        for layer in self.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                input_ids=input_ids,
                input_ids_global=input_ids,
            )
        last_layer = self.layers[-1]
        hidden_states = self.hc_head(
            hidden_states,
            last_layer.hc_head_fn,
            last_layer.hc_head_scale,
            last_layer.hc_head_base,
        )
        return hidden_states


class DeepseekV4DSparkModel(DeepseekV4ForCausalLM):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.num_fused_shared_experts = 0
        self.model = DeepseekV4DSparkCore(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.layers = self.model.layers
        self.markov_rank = int(
            getattr(config, "dspark_markov_rank", getattr(config, "markov_rank", 0))
            or 0
        )
        self.confidence_head = self.model.layers[-1].confidence_head

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden_states)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        return self.model.project_target_hidden(target_hidden)

    def get_logits_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.layers[-1].norm(hidden_states)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.layers[-1].markov_head.markov_w1(token_ids.long())

    def compute_markov_bias_local(
        self,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del hidden_states
        embed = self.get_prev_embeddings(prev_token_ids)
        weight = self.model.layers[-1].markov_head.markov_w2.weight
        return torch.matmul(embed.to(dtype=weight.dtype), weight.t()), state

    def predict_confidence(
        self,
        *,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        embed = self.get_prev_embeddings(prev_token_ids).to(dtype=hidden_states.dtype)
        features = torch.cat([hidden_states, embed], dim=-1)
        return self.model.layers[-1].confidence_head.proj(features.float()).squeeze(-1)

    def _remap_dspark_weight_name(self, name: str) -> Optional[str]:
        if not name.startswith("mtp."):
            return None
        parts = name.split(".", 2)
        if len(parts) != 3:
            return None
        layer_idx = int(parts[1])
        rest = parts[2]
        name = f"model.layers.{layer_idx}.{rest}"
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
        name = name.replace(".gate.tid2eid", ".topk.tid2eid")
        name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
        name = name.replace(".w1.", ".gate_proj.")
        name = name.replace(".w2.", ".down_proj.")
        name = name.replace(".w3.", ".up_proj.")
        if (
            ".self_attn." in name
            or ".mlp." in name
            or ".main_proj." in name
        ):
            name = name.replace(".scale", ".weight_scale_inv")
        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        for raw_name, loaded_weight in weights:
            name = self._remap_dspark_weight_name(raw_name)
            if name is None:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                break
            else:
                loaded = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    if _is_npu:
                        name = name.replace("weight_packed", "weight")
                    mapped_name = name.replace(weight_name, param_name)
                    if mapped_name not in params_dict:
                        continue
                    param = params_dict[mapped_name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        mapped_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(mapped_name)
                    loaded = True
                    break
                if loaded:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        unloaded_params = {
            p
            for p in (params_dict.keys() - loaded_params)
            if "attn_mqa.k_scale" not in p and "attn_mqa.v_scale" not in p
        }
        if unloaded_params:
            logger.warning(
                "Some DSPARK weights are not initialized from checkpoints: %s",
                unloaded_params,
            )
        # DeepSeek V4 target post_load assumes a single `model.decoder` for nextn.
        # DSpark has multiple `model.layers.*` stages, so the generic nextn hook is
        # intentionally skipped here.


class DeepSeekV4DSparkModel(DeepseekV4DSparkModel):
    pass


class Qwen3DSparkModel(DSparkDraftModel):
    pass


EntryClass = [
    DSparkDraftModel,
    DeepseekV4DSparkModel,
    DeepSeekV4DSparkModel,
    Qwen3DSparkModel,
]
