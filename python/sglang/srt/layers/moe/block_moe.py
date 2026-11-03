"""Opt-in block routing and current-forward migration for SM90 FP8 MoE.

The upstream adapter is unchanged when the feature switches are disabled.
Physical tensor layout and routed scaling are checked at this boundary.
"""

import importlib
from dataclasses import dataclass
from functools import lru_cache

import torch

from sglang.srt.distributed import get_moe_ep_group
from sglang.srt.environ import envs
from sglang.srt.eplb import block_dispatch
from sglang.srt.eplb.block_options import options
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo

_buffers = {}
_providers = {}


@dataclass
class Prepared:
    moe: object
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    buf: object
    num_tokens: int
    ep_group: object
    implementation: object


@lru_cache(None)
def backend(name):
    module = importlib.import_module(
        {"native": "deep_gemm", "iaas": "iaas_kernels", "rdma": "iaas_kernels.rdma"}[
            name
        ]
    )
    for symbol in ("fp8_mega_moe", "mega_moe_pre_dispatch_sm90"):
        if not callable(getattr(module, symbol, None)):
            raise RuntimeError(f"{name} has no SM90 FP8 {symbol}")
    return module


def prepare(moe, hidden_states, forward_batch, input_ids_global):
    num_tokens = hidden_states.shape[0]
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

    cfg = options()
    group = get_moe_ep_group().device_group
    h, i = moe.config.hidden_size, moe.config.moe_intermediate_size
    topk = moe.config.num_experts_per_tok
    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    if num_tokens > cap:
        raise ValueError(f"MoE input {num_tokens} exceeds configured capacity {cap}")
    name = cfg.block_moe_backend
    provider_key = (name, id(group))
    if provider_key not in _providers:
        from sglang.srt.eplb.expert_transport import group_hosts

        hosts = group_hosts(get_moe_ep_group().cpu_group)
        if len(set(hosts)) > 1:
            if name != "iaas":
                raise ValueError("Multi-node Block MoE requires the IAAS RDMA backend")
            _providers[provider_key] = ("rdma", {"topology": hosts})
        else:
            _providers[provider_key] = (name, {})
    name, topology_args = _providers[provider_key]
    key = (name, id(group), moe.experts.num_experts, cap, topk, h, i)
    if key not in _buffers:
        if name == "native":
            from sglang.srt.layers.moe.mega_moe import _get_mega_moe_symm_buffer

            _buffers[key] = _get_mega_moe_symm_buffer(
                group, moe.experts.num_experts, cap, topk, h, i
            )
        else:
            _buffers[key] = backend(name).get_symm_buffer_for_sm90_mega_moe(
                group,
                moe.experts.num_experts,
                cap,
                topk,
                h,
                i,
                use_fp8_dispatch=True,
                activation="swiglu",
                **topology_args,
            )
    ids = (
        topk_ids.to(torch.int32)
        if num_tokens
        else hidden_states.new_empty((0, topk), dtype=torch.int32)
    )
    weights = (
        topk_weights.to(torch.float32)
        if num_tokens
        else hidden_states.new_empty((0, topk), dtype=torch.float32)
    )
    return Prepared(
        moe,
        hidden_states,
        ids,
        weights,
        _buffers[key],
        num_tokens,
        group,
        backend(name),
    )


def run_remaining(prepared, early=None):
    moe, buf = prepared.moe, prepared.buf
    impl = prepared.implementation
    factor = (
        1.0
        if moe.experts.should_fuse_routed_scaling_factor_in_topk
        else float(moe.routed_scaling_factor)
    )
    impl.mega_moe_pre_dispatch_sm90(
        prepared.hidden_states,
        prepared.topk_ids,
        prepared.topk_weights,
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
        num_tokens=prepared.num_tokens,
        group_size=128,
        routed_scaling_factor=factor,
    )
    output = torch.empty(
        (prepared.num_tokens, moe.config.hidden_size),
        dtype=torch.bfloat16,
        device=prepared.hidden_states.device,
    )
    extra = {} if early is None else early.combine_inputs()
    impl.fp8_mega_moe(
        output,
        moe.experts.mega_l1_weights,
        moe.experts.mega_l2_weights,
        buf,
        recipe=(128, 128, 128),
        activation="swiglu",
        activation_clamp=getattr(moe.config, "swiglu_limit", None),
        fast_math=True,
        **extra,
    )
    return output[: prepared.num_tokens]


def forward(moe, hidden_states, forward_batch=None, input_ids_global=None):
    if torch.cuda.get_device_capability(hidden_states.device) != (9, 0):
        raise ValueError("Block MoE execution currently requires SM90 FP8 weights")
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("Block MoE execution requires eager model execution")
    if (
        moe.num_fused_shared_experts
        or getattr(moe.experts, "fused_moe_backend", None) is not None
    ):
        raise ValueError(
            "Block MoE requires separate shared experts and native FP8 weights"
        )
    if not getattr(moe.experts, "_mega_moe_sm90_fp8_weights", False):
        raise ValueError("SM90 expert weights must be prepared before block execution")
    p = prepare(moe, hidden_states, forward_batch, input_ids_global)
    early = None
    if options().enable_early_expert_compute:
        from sglang.srt.eplb.early_compute import state_for as early_state_for

        if getattr(p.implementation, "EARLY_COMBINE_API_VERSION", 0) < 1:
            raise RuntimeError(
                "Early MoE requires a backend with slot-ordered early combine. "
                "Apply the supplied IAAS backend patch and rebuild it, or disable "
                "--enable-early-expert-compute. Rounded partial-sum merge is unsafe."
            )
        cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
        early = early_state_for(
            moe, get_global_expert_location_metadata(), get_moe_ep_group(), cap
        )
        p.topk_ids = early.run(p)
    p.topk_ids = block_dispatch.global_block_route(p.topk_ids, moe.layer_id, p.ep_group)
    state = None
    if options().enable_expert_migration:
        from sglang.srt.eplb.migration import state_for

        metadata = get_global_expert_location_metadata()
        state = state_for(moe, metadata, get_moe_ep_group())
        counts, quota = block_dispatch._LAST_CURRENT.pop(moe.layer_id)
        state.begin(metadata)
        # Submit independent shared work before a possible NCCL command-list D2H.
        shared = moe._forward_shared_experts(hidden_states)
        p.topk_ids = state.enqueue(metadata, counts, quota, p.topk_ids)
    else:
        shared = moe._forward_shared_experts(hidden_states)
    with torch.profiler.record_function("block_moe_remaining"):
        output = run_remaining(p, early)
    if state is not None:
        state.finish()
    if early is not None:
        early.finish()
    if shared is not None:
        output.add_(shared)
    return output
