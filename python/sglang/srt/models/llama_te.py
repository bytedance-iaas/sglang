import os
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import transformer_engine as te
import transformers
from torch import nn
from transformers import LlamaConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather
)
from vllm.model_executor.layers.rotary_embedding import get_rope

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import make_layers
from sglang.utils import get_exception_traceback


import transformer_engine.pytorch.cpp_extensions as tex
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
if not tex.device_supports_multicast():
    os.environ["UB_SKIPMC"] = "1"

MAX_SEQ_LEN = 4096

@contextmanager
def replace_decoder(te_decoder_cls):
    """
    Replace `LlamaDecoderLayer` with custom `TELlamaDecoderLayer`.
    """
    original_llama_decoder_cls = (
        transformers.models.llama.modeling_llama.LlamaDecoderLayer
    )
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = te_decoder_cls
    try:
        yield
    finally:
        transformers.models.llama.modeling_llama.LlamaDecoderLayer = (
            original_llama_decoder_cls
        )

@contextmanager
def sequence_parallel_context(layernorm_mlp, seq_len):
    """
    Context manager to temporarily modify LayerNormMLP's parallel configuration
    based on sequence length.
    
    When the sequence length is 1 (decode mode), disable sequence parallel and
    related optimizations; restore original settings when exiting the context.
    """
    # save original settings
    original_sequence_parallel = layernorm_mlp.sequence_parallel
    original_ub_overlap_ag = layernorm_mlp.ub_overlap_ag
    original_ub_overlap_rs = layernorm_mlp.ub_overlap_rs
    original_ub_bulk_wgrad = layernorm_mlp.ub_bulk_wgrad
    original_ub_bulk_dgrad = layernorm_mlp.ub_bulk_dgrad

    if seq_len == 1:  # decode mode
        # disable SP and SP overlap settings
        layernorm_mlp.sequence_parallel = False
        layernorm_mlp.ub_overlap_ag = False
        layernorm_mlp.ub_overlap_rs = False
        layernorm_mlp.ub_bulk_wgrad = False
        layernorm_mlp.ub_bulk_dgrad = False

    try:
        yield
    finally:
        # restore original settings
        layernorm_mlp.sequence_parallel = original_sequence_parallel
        layernorm_mlp.ub_overlap_ag = original_ub_overlap_ag
        layernorm_mlp.ub_overlap_rs = original_ub_overlap_rs
        layernorm_mlp.ub_bulk_wgrad = original_ub_bulk_wgrad
        layernorm_mlp.ub_bulk_dgrad = original_ub_bulk_dgrad

class TELlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_is_neox_style: bool = True,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class TELlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        # tp_group = get_tp_group()
        ranks = list(range(tp_size))
        tp_group = torch.distributed.new_group(ranks)
        te.initialize_ub(
            shape=[1, self.hidden_size],
            tp_size=tp_size,
            use_fp8=False,
            dtype=torch.bfloat16,
            bootstrap_backend=opts.bootstrap_backend,
        )
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = TELlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        # TE's LayerNormMLP (combines post_attention_layernorm and MLP)
        # overlap configuration adapted from
        # https://github.com/NVIDIA/TransformerEngine/blob/main/examples/pytorch/comm_gemm_overlap/te_layer_with_overlap.py#L120
        kwargs = {
            "params_dtype": torch.bfloat16,
            "device": "cuda",
            "tp_group": get_tp_group().device_group,
            "tp_size": tp_size,
            "sequence_parallel": True,
            "ub_overlap_ag": True,
            "ub_bulk_wgrad": True,
            "ub_bulk_dgrad": True,
            "set_parallel_mode": True,
            "ub_overlap_rs": True,
            "seq_length": MAX_SEQ_LEN,
        }
        # We don't need to return_layernorm_output and return_layernorm_output_gathered
        self.layernorm_mlp = te.pytorch.LayerNormMLP(
            hidden_size=self.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            eps=config.rms_norm_eps,
            bias=False,
            return_layernorm_output=False,
            return_layernorm_output_gathered=False,
            normalization="RMSNorm",
            activation="swiglu",
            **kwargs
        )
        self.layernorm_mlp.set_tensor_parallel_group(tp_group)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        # Fully Connected with TE
        tp_group = get_tp_group().device_group
        self.layernorm_mlp.set_tensor_parallel_group(tp_group)

        # This code is from rmsnorm forward_native in sglang
        if residual is not None:
            hidden_states = hidden_states + residual
            residual = hidden_states

        seq_len = hidden_states.shape[0]
        if seq_len > 1: # For prefill phase, split the sequence dimension
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
            # Padding sequence
            if seq_len % MAX_SEQ_LEN != 0:
                pad_len = MAX_SEQ_LEN - seq_len % MAX_SEQ_LEN
                pad_tensor = torch.zeros(pad_len, hidden_states.shape[1], device=hidden_states.device,
                                         dtype=hidden_states.dtype)
                hidden_states = torch.cat([hidden_states, pad_tensor], dim=0).contiguous()
            hidden_states_slices = torch.chunk(hidden_states, tp_size, dim=0)
            hidden_states = hidden_states_slices[tp_rank]

        with sequence_parallel_context(self.layernorm_mlp, seq_len):
            hidden_states = self.layernorm_mlp(hidden_states)

        if seq_len > 1:
            # TODO: remove manual all-gather 
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
            # Remove padding
            hidden_states = hidden_states[:seq_len].contiguous()

        return hidden_states, residual


class TELlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        tp_size = get_tensor_model_parallel_world_size()
        if MAX_SEQ_LEN % tp_size != 0:
            raise ValueError(f"To enable sequence parallel, TE requires MAX_SEQ_LEN {MAX_SEQ_LEN} to be divisible by tp_size {tp_size}")

        te.pytorch.initialize_ub(
            [MAX_SEQ_LEN, config.hidden_size],
            tp_size,
            use_fp8=False,
            dtype=torch.bfloat16,
            bootstrap_backend="nccl",
        )
        # token embedding
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )

        # transformer layers
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: TELlamaDecoderLayer(
                config=config, quant_config=quant_config, layer_id=idx, prefix=prefix
            ),
            prefix="model.layers",
        )

        # final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class TELlamaForCausalLM(nn.Module):
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj.weight": ("layernorm_mlp.fc1_weight", 0),
        "up_proj.weight": ("layernorm_mlp.fc1_weight", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()

        print("=" * 50)
        print("Initializing TE-LLaMA model!!!!!!")
        print("=" * 50)

        self.config = config
        self.quant_config = quant_config
        self.model = TELlamaModel(config, quant_config)
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, quant_config=quant_config
            )
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return self.pooler(hidden_states, forward_batch)

    def get_hidden_dim(self, module_name):
        # return input_dim, output_dim
        if module_name in ["q_proj", "o_proj", "qkv_proj"]:
            return self.config.hidden_size, self.config.hidden_size
        elif module_name in ["kv_proj"]:
            return self.config.hidden_size, self.config.hidden_size // (
                self.config.num_attention_heads // self.config.num_key_value_heads
            )
        else:
            raise NotImplementedError()

    def get_num_params(self):
        params_dict = dict(self.named_parameters())
        return len(params_dict)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if name.endswith(".bias") and name not in params_dict:
                continue
            if name.endswith(".kv_scale") and name not in params_dict:
                continue

            if "self_attn.q_proj.weight" in name:
                qkv_name = name.replace(
                    "self_attn.q_proj.weight", "self_attn.qkv_proj.weight"
                )
                assert qkv_name in params_dict
                param = params_dict[qkv_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, "q")

            elif "self_attn.k_proj.weight" in name:
                qkv_name = name.replace(
                    "self_attn.k_proj.weight", "self_attn.qkv_proj.weight"
                )
                assert qkv_name in params_dict
                param = params_dict[qkv_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, "k")

            elif "self_attn.v_proj.weight" in name:
                qkv_name = name.replace(
                    "self_attn.v_proj.weight", "self_attn.qkv_proj.weight"
                )
                assert qkv_name in params_dict
                param = params_dict[qkv_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, "v")

            elif "post_attention_layernorm.weight" in name:
                te_name = name.replace(
                    "post_attention_layernorm.weight", "layernorm_mlp.layer_norm_weight"
                )
                param = params_dict[te_name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)

            elif "mlp.up_proj.weight" in name:
                te_name = name.replace(
                    "mlp.up_proj.weight", "layernorm_mlp.fc1_weight"
                )
                assert te_name in params_dict
                param = params_dict[te_name]
                up_chunks = torch.chunk(loaded_weight, tp_size, dim=0)
                shard_size = param.size(0) // 2
                default_weight_loader(
                    param.narrow(0, shard_size, shard_size), 
                    up_chunks[tp_rank]
                )

            elif "mlp.gate_proj.weight" in name:
                te_name = name.replace(
                    "mlp.gate_proj.weight", "layernorm_mlp.fc1_weight"
                )
                assert te_name in params_dict
                param = params_dict[te_name]
                gate_chunks = torch.chunk(loaded_weight, tp_size, dim=0)
                shard_size = param.size(0) // 2
                default_weight_loader(
                    param.narrow(0, 0, shard_size), 
                    gate_chunks[tp_rank]
                )

            elif "mlp.down_proj.weight" in name:
                te_name = name.replace(
                    "mlp.down_proj.weight", "layernorm_mlp.fc2_weight"
                )
                assert te_name in params_dict
                param = params_dict[te_name]
                down_chunks = torch.chunk(loaded_weight, tp_size, dim=1)
                default_weight_loader(param, down_chunks[tp_rank])

            elif name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


    def get_weights_by_name(
        self, name: str, truncate_size: int = 100, tp_size: int = 1
    ) -> Optional[torch.Tensor]:
        try:
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                logger.info(
                    "word embedding is tied for this model, return embed_tokens.weight as lm_head.weight."
                )
                return (
                    self.model.embed_tokens.weight.cpu()
                    .to(torch.float32)
                    .numpy()
                    .tolist()[:truncate_size]
                )

            mapped_name = name
            mapped_shard_id = None
            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name in name:
                    mapped_name = name.replace(weight_name, param_name)
                    mapped_shard_id = shard_id
                    break
            params_dict = dict(self.named_parameters())
            param = params_dict[mapped_name]
            if mapped_shard_id is not None:
                if mapped_shard_id in ["q", "k", "v"]:
                    num_heads = self.config.num_attention_heads // tp_size
                    num_kv_heads = self.config.num_key_value_heads // tp_size
                    head_dim = (
                        self.config.hidden_size // self.config.num_attention_heads
                    )
                    if mapped_shard_id == "q":
                        offset = 0
                        size = num_heads * head_dim
                    elif mapped_shard_id == "k":
                        offset = num_heads * head_dim
                        size = num_kv_heads * head_dim
                    elif mapped_shard_id == "v":
                        offset = (num_heads + num_kv_heads) * head_dim
                        size = num_kv_heads * head_dim
                    weight = param.data.narrow(0, offset, size)
                elif mapped_shard_id in [0, 1]:
                    intermediate_size = self.config.intermediate_size
                    slice_size = intermediate_size // tp_size
                    if mapped_shard_id == 0:  # gate_proj
                        offset = 0
                        size = slice_size
                    elif mapped_shard_id == 1:  # up_proj
                        offset = slice_size
                        size = slice_size

                    weight = param.data.narrow(0, offset, size)
                else:
                    weight = param.data
            else:
                weight = param.data
            if tp_size > 1 and ("o_proj" in name or "down_proj" in name):
                gathered_weights = [torch.zeros_like(weight) for _ in range(tp_size)]
                torch.distributed.all_gather(gathered_weights, weight)
                weight = torch.cat(gathered_weights, dim=1)
            return weight.cpu().to(torch.float32).numpy().tolist()[:truncate_size]

        except Exception:
            logger.error(
                f"Error getting weights by name {name} in LlamaForCausalLM: {get_exception_traceback()}"
            )
            return None


EntryClass = [TELlamaForCausalLM]