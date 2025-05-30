
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/quantization/fp8.py

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

try:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
        prepare_fp8_layer_for_marlin,
    )

    MARLIN_FP8_AVAILABLE = True
except ImportError:
    MARLIN_FP8_AVAILABLE = False

    def dummy_func(*args, **kwargs):
        raise ImportError(
            "marlin FP8 requires some operators from vllm. Please install vllm."
        )

    apply_fp8_marlin_linear = prepare_fp8_layer_for_marlin = dummy_func


from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
    scaled_fp8_quant,
)
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    apply_w8a8_block_fp8_linear,
    cutlass_fp8_supported,
    input_to_float8,
    is_sm100_supported,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.utils import (
    all_close_1d,
    convert_to_channelwise,
    is_layer_skipped,
    per_tensor_dequantize,
    requantize_with_max_scale,
)
from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_hip,
    log_info_on_rank0,
    print_warning_once,
    set_weight_attrs,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod


ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = logging.getLogger(__name__)

class Fp8MoEInt4Config(QuantizationConfig):
    """Config class for ModelOpt W4A8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        is_moe_w4a8_serialized: bool = False,
        activation_scheme: str = "static",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: Optional[List[int]] = None,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.is_moe_w4a8_serialized = is_moe_w4a8_serialized
        if is_moe_w4a8_serialized:
            logger.warning(
                "Detected fp8 moe int4 checkpoint. Please note that"
                " the format is experimental and could change."
            )
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        self.weight_block_size = weight_block_size
        self.group_size = group_size

    @classmethod
    def get_name(cls) -> str:
        return "fp8_moe_int4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Fp8MoEInt4Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])

        is_checkpoint_fp8_serialized = "fp8" in quant_method
        is_moe_w4a8_serialized = "moe_int4" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        return cls(
            is_checkpoint_fp8_serialized,
            is_moe_w4a8_serialized,
            activation_scheme,
            ignored_layers,
            weight_block_size,
        )


    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return Fp8MoEInt4MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

class Fp8MoEInt4MoEMethod:

    def __init__(self, quant_config: Fp8MoEInt4Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoeWeightScaleSupported
        assert "weight_loader" in extra_weight_attrs

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.quant_config.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition //
                self.quant_config.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                        dtype=torch.float32),
                                             requires_grad=False)
        layer.register_parameter("w13_input_scale", w13_input_scale)
        # extra_weight_attrs.update(
        #     {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        # )
        # set_weight_attrs(w13_input_scale, {"scale_type": "input_scale"})
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                       dtype=torch.float32),
                                            requires_grad=False)
        # set_weight_attrs(w2_input_scale, {"scale_type": "input_scale"})
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

        # Pre-populate the strides
        k = layer.w2_weight.shape[1]
        n = layer.w13_weight.shape[1] / 2
        num_experts = layer.w2_weight.shape[0]
        device = layer.w13_weight.device

        self.a_strides1 = torch.empty((num_experts, 3), dtype=torch.int64, device=device)
        self.b_strides1 = torch.empty((num_experts, 3), dtype=torch.int64, device=device)
        self.c_strides1 = torch.empty((num_experts, 3), dtype=torch.int64, device=device)
        self.s_strides13 = torch.empty((num_experts, 3), dtype=torch.int64, device=device)

        self.a_strides2 = torch.empty((num_experts, 3), dtype=torch.int64, device=device)
        self.b_strides2 = torch.empty((num_experts, 3), dtype=torch.int64, device=device)
        self.c_strides2 = torch.empty((num_experts, 3), dtype=torch.int64, device=device)
        self.s_strides2 = torch.empty((num_experts, 3), dtype=torch.int64, device=device)
        # self.s_strides13 = self.c_strides1
        # self.s_strides2  = self.c_strides2
        self.b_strides1 = self.a_strides1
        self.b_strides2 = self.a_strides2
        self.a_strides1[:, 0].fill_(k)
        self.a_strides1[:, 1].fill_(1)
        self.a_strides1[:, 2].zero_()
        # self.b_strides1[:, 0].fill_(k)
        # self.b_strides1[:, 1].fill_(1)
        # self.b_strides1[:, 2].zero_()
        self.c_strides1[:, 0].fill_(1)
        self.c_strides1[:, 1].fill_(2 * n)
        self.c_strides1[:, 2].zero_()
        self.s_strides13[:, 0].fill_(2 * n)
        self.s_strides13[:, 1].fill_(1)
        self.s_strides13[:, 2].zero_()

        self.a_strides2[:, 0].fill_(n)
        self.a_strides2[:, 1].fill_(1)
        self.a_strides2[:, 2].zero_()
        # self.b_strides2[:, 0].fill_(n)
        # self.b_strides2[:, 1].fill_(1)
        # self.b_strides2[:, 2].zero_()
        self.c_strides2[:, 0].fill_(1)
        self.c_strides2[:, 1].fill_(k)
        self.c_strides2[:, 2].zero_()
        self.s_strides2[:, 0].fill_(k)
        self.s_strides2[:, 1].fill_(1)
        self.s_strides2[:, 2].zero_()

        return

    def _interleave_scales(self, scales: torch.Tensor) -> torch.Tensor:
        """Interleave scales in groups of 4 similar to TRT-LLM implementation."""
        s_shape = scales.shape
        # Reshape to separate groups of 4
        scales_interleaved = scales.reshape(s_shape[0], s_shape[1],
                                            (s_shape[2] // 4), 4)
        # Permute dimensions to interleave
        scales_interleaved = scales_interleaved.permute(0, 2, 1, 3)
        # Reshape back to original dimensions but with interleaved values
        scales_interleaved = scales_interleaved.reshape(
            s_shape[0], s_shape[2] // 4, s_shape[1] * 4)
        return scales_interleaved.contiguous()

    def process_weights_after_loading(self, layer: Module) -> None:
        num_experts = layer.w2_weight.shape[0]
        hidden_size = layer.w2_weight.shape[1]

        intermediate_size_per_partition = layer.w2_weight.shape[2] * 2
        dtype = torch.bfloat16
        device = layer.w2_weight.device

        # Interleave w13_weight_scale (gate_up_proj)
        w13_weight_scale = layer.w13_weight_scale_inv.to(dtype)
        w13_weight_scale = self._interleave_scales(w13_weight_scale)
        # layer.w13_weight_scale_inv = Parameter(w13_weight_scale.view(
        #     torch.quint4x2), requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(w13_weight_scale,
                                               requires_grad=False)

        # Interleave w2_weight_scale (down_proj)
        w2_weight_scale = layer.w2_weight_scale_inv.to(dtype)
        w2_weight_scale = self._interleave_scales(w2_weight_scale)
        # layer.w2_weight_scale_inv = Parameter(w2_weight_scale.view(
        #     torch.quint4x2), requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(w2_weight_scale,
                                              requires_grad=False)

        # Process input scales
        # w13_input_scale_scalar = layer.w13_input_scale.max().item()
        # w13_input_scale = Parameter(torch.ones(
        #     hidden_size,
        #     dtype=torch.bfloat16,
        #     device=layer.w13_input_scale.device),
        #                             requires_grad=False)
        # layer.w13_input_scale = Parameter(w13_input_scale /
        #                                   w13_input_scale_scalar,
        #                                   requires_grad=False)
        w13_input_scale_max = layer.w13_input_scale.max().to(dtype).item()
        new_w13_input_scale = torch.tensor(
            [w13_input_scale_max],  # Pass as a list to create a 1-D tensor with one element
            dtype=dtype,
            device=device
        )
        layer.w13_input_scale = Parameter(new_w13_input_scale, requires_grad=False)

        # w2_input_scale_scalar = layer.w2_input_scale.max().item()
        # w2_input_scale = Parameter(torch.ones(
        #     intermediate_size_per_partition,
        #     dtype=torch.float,
        #     device=layer.w2_input_scale.device),
        #                            requires_grad=False)
        # layer.w2_input_scale = Parameter(w2_input_scale /
        #                                  w2_input_scale_scalar,
        #                                  requires_grad=False)
        w2_input_scale_max = layer.w2_input_scale.max().to(dtype).item()
        new_w2_input_scale = torch.tensor(
            [w2_input_scale_max],
            dtype=dtype,
            device=device
        )
        layer.w2_input_scale = Parameter(new_w2_input_scale, requires_grad=False)

        # alpha
        # a1_alpha = torch.full(
        #     (num_experts, 1),
        #     fill_value=w13_input_scale_max,
        #     dtype=torch.float32,
        #     device=device
        # )
        # self.a1_alpha = Parameter(a1_alpha, requires_grad=False)

        # a2_alpha = torch.full(
        #     (num_experts, 1),
        #     fill_value=w2_input_scale_max,
        #     dtype=torch.float32,
        #     device=device
        # )
        # self.a2_alpha = Parameter(a2_alpha, requires_grad=False)



    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        # device = layer.w13_weight.device
        # device_id = device.index
        # save_dir = f"/nvme0n1/w4a8_debug_tensors/device_{device_id}"
        # import os
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir, exist_ok=True)
        #     tensors = {
        #         "x": x,
        #         "w13_weight": layer.w13_weight,
        #         "w2_weight": layer.w2_weight,
        #         "w13_weight_scale_inv": layer.w13_weight_scale_inv,
        #         "w2_weight_scale_inv": layer.w2_weight_scale_inv,
        #         "topk_weights": topk_weights,
        #         "topk_ids": topk_ids,
        #         "w13_input_scale": layer.w13_input_scale,
        #         "w2_input_scale": layer.w2_input_scale,
        #         "a_strides1": self.a_strides1,
        #         "b_strides1": self.b_strides1,
        #         "c_strides1": self.c_strides1,
        #         "a_strides2": self.a_strides2,
        #         "b_strides2": self.b_strides2,
        #         "c_strides2": self.c_strides2,
        #         "s_strides13": self.s_strides13,
        #         "s_strides2": self.s_strides2,
        #         "expert_map": expert_map,
        #     }

        #     with open(f"{save_dir}/shapes_and_dtypes.txt", "w") as f:
        #         for name, tensor in tensors.items():
        #             f.write(
        #                 f"{name}: {tensor.shape}, {tensor.dtype}, {tensor.device}\n"
        #             )
        #         f.write(
        #             f"apply_router_weight_on_input: {apply_router_weight_on_input}\n"
        #         )

        #     for name, tensor in tensors.items():
        #         torch.save(tensor, f"{save_dir}/{name}.pt")

        return x
