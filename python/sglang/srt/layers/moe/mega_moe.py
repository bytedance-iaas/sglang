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
"""Mega-MoE forward path and expert-weight prep shared by Deepseek V2/V4."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.deepseek_v4 import (
    mega_moe_pre_dispatch,
    mega_moe_pre_dispatch_sm90,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.models.deepseek_common.utils import _device_sm

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


_MEGA_MOE_SYMM_BUFFER: dict = {}
_MEGA_MOE_DG_ENV_APPLIED = False


def _apply_mega_moe_dg_env() -> None:
    """Forward sglang's FP4/MXF4 opt-in flags to DeepGEMM via env vars.

    DeepGEMM reads `DG_USE_FP4_ACTS` (and `DG_USE_MXF4_KIND`) at host-function
    call time — both `get_symm_buffer_for_mega_moe` and `fp8_fp4_mega_moe`.
    Forwarding once at first use is sufficient (these are static config
    flags, not per-request state) and matches the `setdefault` pattern so
    explicit `DG_USE_*` overrides from outside still win.
    """
    global _MEGA_MOE_DG_ENV_APPLIED
    if _MEGA_MOE_DG_ENV_APPLIED:
        return
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get():
        os.environ.setdefault("DG_USE_FP4_ACTS", "1")
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND.get():
        os.environ.setdefault("DG_USE_MXF4_KIND", "1")
    _MEGA_MOE_DG_ENV_APPLIED = True


def _get_mega_moe_symm_buffer(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> SymmBuffer:
    import deep_gemm

    _apply_mega_moe_dg_env()

    key = (
        id(group),
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    buf = _MEGA_MOE_SYMM_BUFFER.get(key)
    if buf is None:
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        _MEGA_MOE_SYMM_BUFFER[key] = buf
    return buf


def should_use_mega_moe(moe: "DeepseekV2MoE", hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_megamoe():
        return False
    if not getattr(moe.experts, "_mega_moe_weights_built", False):
        return False
    if _device_sm is None or _device_sm < 90:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    if _device_sm == 90:
        # SM90 supports two paths:
        #   * `fp8_mega_moe`     — FP8 weights + per-128 float SF (legacy)
        #   * `fp8_fp4_mega_moe` — packed FP4 weights + per-32 UE8M0 SF (DSV4)
        # The arch dispatch happens inside the C++ binding (mega.hpp): both
        # entry points exist on SM90; we just have to pick the right one
        # based on which weight tensors `build_mega_moe_experts_weights`
        # produced for this layer.
        if getattr(moe.experts, "_mega_moe_sm90_fp4_weights", False):
            if not hasattr(deep_gemm, "fp8_fp4_mega_moe"):
                return False
        else:
            if not hasattr(deep_gemm, "fp8_mega_moe"):
                return False
            if not getattr(moe.experts, "_mega_moe_sm90_fp8_weights", False):
                return False
    elif _device_sm >= 100:
        if not hasattr(deep_gemm, "fp8_fp4_mega_moe"):
            return False
    else:
        return False
    if get_is_capture_mode():
        return True

    try:
        global_num_tokens = get_dp_global_num_tokens()
    except AttributeError:
        global_num_tokens = None
    if global_num_tokens:
        max_tokens_per_rank = max(global_num_tokens)
    else:
        max_tokens_per_rank = hidden_states.shape[0]
    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    if max_tokens_per_rank > cap:
        if getattr(moe.experts, "_mega_moe_sm90_fp4_memory_shared", False):
            raise RuntimeError(
                "SM90 FP4 MegaMOE memory sharing replaced the fallback FP4 expert "
                "layout, so requests cannot fall back to the non-MegaMOE path. "
                f"max_tokens_per_rank={max_tokens_per_rank} exceeds "
                "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
                f"{cap}; raise the env var or reduce chunked/cuda-graph batch size."
            )
        return False
    return True


def forward_mega_moe(
    moe: "DeepseekV2MoE",
    hidden_states: torch.Tensor,
    forward_batch: Optional["ForwardBatch"] = None,
    input_ids_global: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]

    sbo_overlap_flag = (
        moe.alt_stream is not None
        and moe.num_fused_shared_experts == 0
        and num_tokens > 0
        and get_is_capture_mode()
    )

    if sbo_overlap_flag:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = torch.cuda.stream(moe.alt_stream)
    else:
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = nullcontext()

    with mega_stream_ctx:
        y = _run_mega_routed(
            moe, hidden_states, forward_batch, input_ids_global, num_tokens
        )

    if sbo_overlap_flag:
        current_stream.wait_stream(moe.alt_stream)

    if shared_output is not None:
        y.add_(shared_output)
    return y


def _run_mega_routed(
    moe: "DeepseekV2MoE",
    hidden_states: torch.Tensor,
    forward_batch: Optional["ForwardBatch"],
    input_ids_global: Optional[torch.Tensor],
    num_tokens: int,
) -> torch.Tensor:
    import deep_gemm

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    hidden_size = moe.config.hidden_size

    if num_tokens > 0:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        topk_output = moe.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=(
                forward_batch.num_token_non_padded
                if forward_batch is not None
                else None
            ),
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=moe.layer_id,
            ),
            **topk_kwargs,
        )
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
    else:
        topk_ids = None
        topk_weights = None

    ep_group = get_moe_ep_group().device_group
    num_experts = moe.experts.num_experts
    top_k = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
    intermediate_size = moe.config.moe_intermediate_size
    num_max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    assert num_tokens <= num_max_tokens_per_rank, (
        f"mega MoE: num_tokens={num_tokens} exceeds cap "
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
        f"{num_max_tokens_per_rank}; raise the env var or shrink "
        f"cuda_graph_max_bs / chunked_prefill_size accordingly"
    )

    buf = _get_mega_moe_symm_buffer(
        ep_group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=top_k,
        hidden=hidden_size,
        intermediate_hidden=intermediate_size,
    )

    if num_tokens > 0:
        topk_ids_in = topk_ids.to(torch.int32)
        topk_weights_in = topk_weights.to(torch.float32)
    else:
        topk_ids_in = hidden_states.new_empty((0, top_k), dtype=torch.int32)
        topk_weights_in = hidden_states.new_empty((0, top_k), dtype=torch.float32)

    use_sm90_fp8_mega = _device_sm == 90 and getattr(
        moe.experts, "_mega_moe_sm90_fp8_weights", False
    )
    use_sm90_fp4_mega = _device_sm == 90 and getattr(
        moe.experts, "_mega_moe_sm90_fp4_weights", False
    )
    # SM90 FP8/FP4 路径都通过 `mega_moe_pre_dispatch_sm90` 写入 E4M3 激活
    # （per-128 FP32 SF）。如果用户错误地把 `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1`
    # 也打开，`SymmBuffer.x` 会被分配为 int8（packed FP4），buffer 字节大小
    # 偶然匹配但 SF 张量按 per-32 分配，会让下游 GEMM 读到错误的缩放因子，
    # 静默产出错误结果。这里显式拒绝该组合，强制用户走纯 SM100 路径。
    if (use_sm90_fp8_mega or use_sm90_fp4_mega) and \
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get():
        raise RuntimeError(
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS is incompatible with "
            "the SM90 mega-MoE paths (FP8 weights or FP4 weights). SM90 only "
            "supports FP8 activations with per-128 SF. Please disable "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS or run on SM100."
        )
    fused_routed_scaling = False
    if use_sm90_fp8_mega or use_sm90_fp4_mega:
        # SM90 paths (both FP8 weights and FP4 weights) use FP8 activations
        # with per-128 FP32 SF. The mega kernel itself dispatches on the
        # weight dtype; only the *weight*-side recipe changes for FP4.
        if moe.experts.should_fuse_routed_scaling_factor_in_topk:
            scale = 1.0
        else:
            scale = float(moe.routed_scaling_factor)
            fused_routed_scaling = True
        mega_moe_pre_dispatch_sm90(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            routed_scaling_factor=scale,
            quant_group_size=128,
        )
    elif envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get():
        # FP4 path goes through DeepGEMM's mega_moe_pre_dispatch which
        # handles the E2M1 packing variant. The jit implementation
        # only emits FP8.
        deep_gemm.mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=num_tokens,
            group_size=32,
            use_fp4_acts=True,
        )
    else:
        mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            quant_group_size=32,
        )

    # Allocate at least one row so y has a non-null CUDA data_ptr;
    # the DeepGEMM tvm-ffi binding rejects nullptr in convert_to_torch_tensor().
    y = torch.empty(
        (max(num_tokens, 1), hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    swiglu_limit = getattr(moe.config, "swiglu_limit", None)
    if use_sm90_fp8_mega:
        deep_gemm.fp8_mega_moe(
            y,
            moe.experts.mega_l1_weights,
            moe.experts.mega_l2_weights,
            buf,
            recipe=(128, 128, 128),
            activation="swiglu",
            activation_clamp=swiglu_limit,
            fast_math=True,
        )
    else:
        # SM90 FP4 + SM100 FP4/FP8 paths share the `fp8_fp4_mega_moe` entry;
        # the C++ binding (mega.hpp) dispatches on `arch_major` and the FP4
        # weight tensors carry their per-32 UE8M0 SF.
        deep_gemm.fp8_fp4_mega_moe(
            y,
            moe.experts.mega_l1_weights,
            moe.experts.mega_l2_weights,
            buf,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=swiglu_limit,
            fast_math=True,
        )
    y = y[:num_tokens]

    if (
        not moe.experts.should_fuse_routed_scaling_factor_in_topk
        and not fused_routed_scaling
    ):
        y.mul_(moe.routed_scaling_factor)
    return y


def _interleave_l1_weight_only(weight: torch.Tensor, gran: int = 8) -> torch.Tensor:
    num_groups, n, *rest = weight.shape
    half = n // 2
    gate = weight[:, :half].reshape(num_groups, half // gran, gran, *rest)
    up = weight[:, half:].reshape(num_groups, half // gran, gran, *rest)
    return torch.empty_like(weight).copy_(
        torch.stack([gate, up], dim=2).reshape(num_groups, n, *rest)
    )


def _interleave_l1_weights(
    weights: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    weight, scale = weights
    return _interleave_l1_weight_only(weight), _interleave_l1_weight_only(scale)


def build_mega_moe_experts_weights(experts) -> None:
    from deep_gemm import (
        transform_sf_into_required_layout,
        transform_weights_for_mega_moe,
    )
    from deep_gemm.mega import _transpose_sf_for_utccp

    if getattr(experts, "_mega_moe_weights_built", False):
        return

    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    num_groups, n1, half_k1 = w13.shape
    _, n2, half_k2 = w2.shape

    use_sm90_fp8_mega = (
        _device_sm == 90
        and w13.dtype == torch.float8_e4m3fn
        and w2.dtype == torch.float8_e4m3fn
    )
    # SM90 FP4: weights ship as packed E2M1 (two nibbles per byte stored as
    # int8/uint8) with per-32 UE8M0 SFB. The packed tensor has last dim K//2.
    use_sm90_fp4_mega = (
        _device_sm == 90
        and not use_sm90_fp8_mega
        and w13.dtype in (torch.int8, torch.uint8)
        and w2.dtype in (torch.int8, torch.uint8)
    )
    # FP4 weights are packed as int8 and have last dim K//2; FP8 weights use K.
    if use_sm90_fp8_mega:
        k1 = half_k1
        k2 = half_k2
    else:
        k1 = half_k1 * 2
        k2 = half_k2 * 2

    # Recipe / SF dtype selection:
    #   * SM90 FP8  : checkpoint-native block (128, 128) FP32 SF, no UE8M0 cast.
    #   * SM90 FP4  : per-32 UE8M0 SFB (matches DSV4); the SM100 FP4 transform
    #                 path is reused since both architectures feed UE8M0 SFB.
    #   * SM100 FP4 : per-32 UE8M0 SFB.
    if use_sm90_fp8_mega:
        recipe = (128, 128)
        disable_ue8m0_cast = True
    else:
        recipe = (1, 32)
        disable_ue8m0_cast = False

    scale_group_mn, scale_group_k = recipe
    assert k1 % scale_group_k == 0 and k2 % scale_group_k == 0, (
        f"invalid mega-moe K/group_size: k1={k1}, k2={k2}, "
        f"group_k={scale_group_k}"
    )
    expected_n_groups_1 = (n1 + scale_group_mn - 1) // scale_group_mn
    expected_n_groups_2 = (n2 + scale_group_mn - 1) // scale_group_mn
    expected_k_groups_1 = k1 // scale_group_k
    expected_k_groups_2 = k2 // scale_group_k
    assert w13_sf_fp32.shape[1] == expected_n_groups_1, (
        f"w13 scale N groups mismatch: got {w13_sf_fp32.shape[1]}, "
        f"expected {expected_n_groups_1} (n1={n1}, group_mn={scale_group_mn})"
    )
    assert w2_sf_fp32.shape[1] == expected_n_groups_2, (
        f"w2 scale N groups mismatch: got {w2_sf_fp32.shape[1]}, "
        f"expected {expected_n_groups_2} (n2={n2}, group_mn={scale_group_mn})"
    )
    assert w13_sf_fp32.shape[2] == expected_k_groups_1, (
        f"w13 scale K groups mismatch: got {w13_sf_fp32.shape[2]}, "
        f"expected {expected_k_groups_1} (k1={k1}, group_k={scale_group_k})"
    )
    assert w2_sf_fp32.shape[2] == expected_k_groups_2, (
        f"w2 scale K groups mismatch: got {w2_sf_fp32.shape[2]}, "
        f"expected {expected_k_groups_2} (k2={k2}, group_k={scale_group_k})"
    )

    fix_mega_moe_memory = envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get()
    if fix_mega_moe_memory and use_sm90_fp8_mega:
        # SM90 shares both fp8 weights and block-(128, 128) FP32 scales with the
        # DeepEP grouped-GEMM path. SM90 has no UTCCP scale transpose, and its
        # scale tensors stay in checkpoint layout.
        w13_interleaved = _interleave_l1_weight_only(w13)

        experts.w13_weight.data = w13_interleaved

        experts.mega_l1_weights = (
            experts.w13_weight.data,
            experts.w13_weight_scale_inv.data,
        )
        experts.mega_l2_weights = (
            experts.w2_weight.data,
            experts.w2_weight_scale_inv.data,
        )
    elif use_sm90_fp4_mega:
        # SM90 FP4 path: SFB stays as raw FP32 per-32 SF in checkpoint layout
        # (`[E, N, K//32]`). The transform packs it into the k-major UE8M0
        # uint32 layout the SM90 FP4 mega-MoE kernel directly ldgs. No
        # `transform_sf_into_required_layout` step is needed (and would in
        # fact fall through to `DG_HOST_UNREACHABLE` since the C++ helper has
        # no `arch_major == 9 && gran_k == 32` branch).
        from deep_gemm import transform_weights_for_mega_moe_sm90_fp4

        l1_pair, l2_pair = transform_weights_for_mega_moe_sm90_fp4(
            (w13, w13_sf_fp32), (w2, w2_sf_fp32)
        )
        if fix_mega_moe_memory:
            # The SM90 FP4 MegaMOE kernel consumes an interleaved L1 FP4 tensor
            # and packed UE8M0 scales. The non-MegaMOE FP4 runners consume the
            # checkpoint layout, so there is no single layout that supports both
            # paths. In memory-fix mode, prefer the intended MegaMOE path and
            # replace the original parameters so the old checkpoint-layout
            # tensors can be released before KV-pool sizing.
            experts.w13_weight.data = l1_pair[0]
            experts.w2_weight.data = l2_pair[0]
            experts.w13_weight_scale_inv.data = l1_pair[1]
            experts.w2_weight_scale_inv.data = l2_pair[1]
            experts.w13_weight_scale_inv.format_ue8m0 = True
            experts.w2_weight_scale_inv.format_ue8m0 = True

            experts.mega_l1_weights = (
                experts.w13_weight.data,
                experts.w13_weight_scale_inv.data,
            )
            experts.mega_l2_weights = (
                experts.w2_weight.data,
                experts.w2_weight_scale_inv.data,
            )
            experts._mega_moe_sm90_fp4_memory_shared = True
        else:
            experts.mega_l1_weights = l1_pair
            experts.mega_l2_weights = l2_pair
            experts._mega_moe_sm90_fp4_memory_shared = False
    else:
        w13_sf = transform_sf_into_required_layout(
            w13_sf_fp32,
            mn=n1,
            k=k1,
            recipe=recipe,
            num_groups=num_groups,
            disable_ue8m0_cast=disable_ue8m0_cast,
        )
        w2_sf = transform_sf_into_required_layout(
            w2_sf_fp32,
            mn=n2,
            k=k2,
            recipe=recipe,
            num_groups=num_groups,
            disable_ue8m0_cast=disable_ue8m0_cast,
        )

    if fix_mega_moe_memory and not use_sm90_fp8_mega and not use_sm90_fp4_mega:
        # Build the interleaved L1 weight + scale once; share the weight buffer
        # between `w13_weight.data` (normal deep-ep path) and `mega_l1_weights[0]`
        # (mega moe path). Mega moe additionally needs a UTCCP-transposed scale;
        # the deep-ep path consumes the non-transposed interleaved scale and a
        # swizzle-aware activation kernel. L2 weight is untouched by the mega
        # transform, so the existing `w2_weight.data` is shared directly.
        w13_interleaved, w13_sf_interleaved = _interleave_l1_weights((w13, w13_sf))
        w13_sf_utccp = _transpose_sf_for_utccp(w13_sf_interleaved)
        w2_sf_utccp = _transpose_sf_for_utccp(w2_sf)

        experts.w13_weight.data = w13_interleaved
        experts.w13_weight_scale_inv.data = w13_sf_interleaved
        experts.w2_weight_scale_inv.data = w2_sf
        experts.w13_weight_scale_inv.format_ue8m0 = True
        experts.w2_weight_scale_inv.format_ue8m0 = True

        experts.mega_l1_weights = (experts.w13_weight.data, w13_sf_utccp)
        experts.mega_l2_weights = (experts.w2_weight.data, w2_sf_utccp)
    elif not fix_mega_moe_memory and not use_sm90_fp4_mega:
        # SM90 FP4 already finalized `mega_l*_weights` in its own branch above.
        transform_fn = transform_weights_for_mega_moe
        if use_sm90_fp8_mega:
            from deep_gemm import transform_weights_for_mega_moe_sm90

            transform_fn = transform_weights_for_mega_moe_sm90

        l1_pair, l2_pair = transform_fn((w13, w13_sf), (w2, w2_sf))

        experts.mega_l1_weights = l1_pair
        experts.mega_l2_weights = l2_pair

    experts._mega_moe_sm90_fp8_weights = use_sm90_fp8_mega
    experts._mega_moe_sm90_fp4_weights = use_sm90_fp4_mega
    experts._mega_moe_weights_built = True
