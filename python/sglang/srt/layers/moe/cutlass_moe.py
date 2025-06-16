# SPDX-License-Identifier: Apache-2.0
"""Fused MoE kernel."""
from typing import Optional  # TODO

import torch
from grouped_gemm.ops import permute, unpermute

# import _custom_ops as ops # TODO
from sgl_kernel import cutlass_moe_mm  # func name
from sgl_kernel import get_cutlass_moe_mm_data  # func name
from sgl_kernel import sgl_per_tensor_quant_fp8, silu_and_mul


# TODO make the grouped gemm kernel consistent with scaled gemm kernel
def cutlass_moe_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids_: torch.Tensor,
    ab_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.half,
    expert_map: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    moe_ws=None,
) -> torch.Tensor:
    """
    This function computes a a8w8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1_q (torch.Tensor): The first set of fp8-quantized expert weights.
        Shape: [num_experts, K, 2N] (the weights are passed transposed)
    - w2_q (torch.Tensor): The second set of fp8-quantized expert weights.
        Shape: [num_experts, N, K] (the weights are passed transposed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts] or [num_experts, 2N]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts] or [num_experts, K]
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - ab_strides1 (torch.Tensor): The input and weights strides of the first
        grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - ab_strides2 (torch.Tensor): The input and weights strides of the second
        grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [M]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [M]
    - out_dtype (torch.Tensor): The output tensor type.
    - expert_map (Optional[torch.Tensor]): In the case of Expert parallel,
        every Rank is responsible for a subset of experts. expert_map is a
        mapping from global expert-id to local expert-id. When expert_map[i]
        is -1, it means that this Rank is not responsible for global
        expert-id i.
    - apply_router_weight_on_input (bool): When true, the topk weights are
        applied directly on the inputs. This is only applicable when topk is 1.

    Returns:
    - torch.Tensor: The fp16 output tensor after applying the MoE layer.
    """

    assert topk_weights.shape == topk_ids_.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.float8_e4m3fn
    assert w2_q.dtype == torch.float8_e4m3fn
    assert a.shape[1] == w1_q.shape[1], "Hidden size mismatch w1"
    assert w1_q.shape[2] == w2_q.shape[1] * 2, "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert (
        a1_scale is None
        or a1_scale.dim() == 0
        or a1_scale.shape[0] == 1
        or a1_scale.shape[0] == a.shape[0]
    ), "Input scale shape mismatch"
    assert (
        w1_scale.dim() == 1
        or w1_scale.shape[1] == 1
        or w1_scale.shape[1] == w1_q.shape[2]
    ), "W1 scale shape mismatch"
    assert (
        w2_scale.dim() == 1
        or w2_scale.shape[1] == 1
        or w2_scale.shape[1] == w2_q.shape[2]
    ), "W2 scale shape mismatch"
    assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"
    assert (
        a2_scale is None or a1_scale is None or a2_scale.shape == a1_scale.shape
    ), "Intermediate scale shape mismatch"  # noqa: E501
    assert ab_strides1.shape[0] == w1_q.shape[0], "AB Strides 1 expert number mismatch"
    assert c_strides1.shape[0] == w1_q.shape[0], "C Strides 1 expert number mismatch"
    assert ab_strides2.shape[0] == w2_q.shape[0], "AB Strides 2 expert number  mismatch"
    assert c_strides2.shape[0] == w2_q.shape[0], "C Strides 2 expert number mismatch"
    assert out_dtype in [torch.half, torch.bfloat16], "Invalid output dtype"

    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(1)
    n = w2_q.size(1)

    assert m <= moe_ws.a_q_fp8.shape[0], "runtime shape exceed max workspace shape"

    local_topk_ids = topk_ids_
    if expert_map is not None:
        "Translate info from expert_map to topk_ids"
        local_topk_ids = torch.where(
            expert_map[topk_ids_] != -1, expert_map[topk_ids_], -1
        )

    topk = local_topk_ids.size(1)

    per_act_token = (
        a1_scale.numel() != 1
        if a1_scale is not None
        else (a2_scale.numel() != 1 if a2_scale is not None else False)
    )
    if apply_router_weight_on_input:
        assert topk == 1, "apply_router_weight_on_input is only implemented for topk=1"
        # TODO: this only works for topK=1, will need to update for topK>1
        a = a * topk_weights.to(out_dtype)

    # a_q, a1_scale = scaled_fp8_quant(
    #     hidden_states, a1_scale, use_per_token_if_dynamic=per_act_token)
    device = a.device
    a_q = moe_ws.a_q_fp8[0:m, :]

    a1_scale = moe_ws.act_scale
    a2_scale = a1_scale

    expert_offsets = moe_ws.expert_offsets
    problem_sizes1 = moe_ws.problem_sizes[0]
    problem_sizes2 = moe_ws.problem_sizes[1]

    intermediate = moe_ws.inter[0 : m * topk, :]
    a_map = moe_ws.permute_map[0][0 : m * topk]
    # a_map = torch.empty((m * topk), device = device, dtype = torch.int32)
    c_map = moe_ws.permute_map[1][0 : m * topk]

    c1 = moe_ws.c1[0 : m * topk, :]
    c2 = moe_ws.c2[0 : m * topk, :]
    intemediate_q = moe_ws.inter_q[0 : m * topk, :]

    # ops.get_cutlass_moe_mm_data(local_topk_ids, expert_offsets, problem_sizes1,
    #                             problem_sizes2, a_map, c_map, num_experts, n,
    #                             k)

    sgl_per_tensor_quant_fp8(a, a_q, a1_scale, False)
    get_cutlass_moe_mm_data(
        local_topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )

    rep_a_q, unpermute_map = permute(a_q, topk_ids_)
    rep_a1_scales = a1_scale  # a1_scale[a_map] if per_act_token else a1_scale

    cutlass_moe_mm(
        c1,
        rep_a_q,
        w1_q,
        rep_a1_scales,
        w1_scale,
        expert_offsets[:-1],
        problem_sizes1,
        ab_strides1,
        ab_strides1,
        c_strides1,
    )

    silu_and_mul(c1, intermediate)

    sgl_per_tensor_quant_fp8(intermediate, intemediate_q, a2_scale, False)

    cutlass_moe_mm(
        c2,
        intemediate_q,
        w2_q,
        a2_scale,
        w2_scale,
        expert_offsets[:-1],
        problem_sizes2,
        ab_strides2,
        ab_strides2,
        c_strides2,
    )

    return unpermute(c2, unpermute_map, topk_weights)
