from __future__ import annotations

import ctypes
import logging
import os
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
    DeepEPNormalCombineInput,
    DeepEPNormalDispatchOutput,
    DeepEPPDispatchHooks,
    _DeepEPDispatcherImplNormal,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import CombineOverlapArgs

try:
    from nccl_ep.nccl_wrapper import (
        NCCLLibrary,
        ncclDataTypeEnum,
        ncclEpAlgorithm_t,
        ncclEpGroupConfig_t,
        ncclEpTensorTag_t,
        ncclNDTensor_t,
        ncclEpDispatchConfig_t,
        get_nccl_comm_from_group,
    )

    use_nccl_ep = True
except ImportError:
    use_nccl_ep = False

logger = logging.getLogger(__name__)

NcclEPLLDispatchOutput = DeepEPLLDispatchOutput
NcclEPLLCombineInput = DeepEPLLCombineInput
NcclEPNormalDispatchOutput = DeepEPNormalDispatchOutput
NcclEPNormalCombineInput = DeepEPNormalCombineInput


def _torch_dtype_to_nccl(dtype: torch.dtype) -> int:
    _map = {
        torch.float16: ncclDataTypeEnum.ncclFloat16,
        torch.bfloat16: ncclDataTypeEnum.ncclBfloat16,
        torch.float32: ncclDataTypeEnum.ncclFloat32,
        torch.int32: ncclDataTypeEnum.ncclInt32,
        torch.int64: ncclDataTypeEnum.ncclInt64,
        torch.int8: ncclDataTypeEnum.ncclInt8,
        torch.uint8: ncclDataTypeEnum.ncclUint8,
    }
    if dtype == torch.float8_e4m3fn:
        return ncclDataTypeEnum.ncclUint8
    if dtype not in _map:
        raise ValueError(f"Unsupported dtype for NCCL-EP: {dtype}")
    return _map[dtype]


def _create_ep_tensor(
    lib: "NCCLLibrary",
    ep_group,
    tensor: torch.Tensor,
    tag: int,
) -> ncclNDTensor_t:
    """Create an opaque ncclNDTensor_t handle wrapping a PyTorch tensor via C API."""
    ndim = tensor.ndim
    dtype = _torch_dtype_to_nccl(tensor.dtype)
    data = ctypes.c_void_p(tensor.data_ptr())
    sizes = list(tensor.shape) + [1] * (5 - ndim)
    return lib.ncclEpTensorCreate(
        ep_group, ndim, dtype, tag, data,
        sizes[0], sizes[1], sizes[2], sizes[3], sizes[4],
    )


def _make_handle_array(handles: List[ncclNDTensor_t]):
    """Create a C array of opaque tensor handles for NCCL-EP APIs."""
    n = len(handles)
    if n == 0:
        return None, 0
    arr = (ctypes.c_void_p * n)()
    for i, h in enumerate(handles):
        arr[i] = h
    return arr, n


def _destroy_tensor_handles(lib, ep_group, handles):
    """Destroy a list of ncclNDTensor_t handles."""
    for h in handles:
        if h:
            lib.ncclEpTensorDestroy(ep_group, h)


class NcclEPBuffer:
    _lib: Optional[NCCLLibrary] = None
    _ep_group = None
    _comm = None
    _algorithm: Optional[int] = None
    _hidden_size: Optional[int] = None
    _num_max_dispatch_tokens_per_rank: Optional[int] = None
    _num_experts: Optional[int] = None
    _num_local_experts: Optional[int] = None
    _world_size: Optional[int] = None
    _rank: Optional[int] = None

    @classmethod
    def get_nccl_ep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int = -1,
        num_experts: int = -1,
        num_local_experts: int = -1,
        params_dtype: torch.dtype = torch.bfloat16,
    ):
        if cls._ep_group is not None:
            return cls

        if not use_nccl_ep:
            raise ImportError(
                "NCCL-EP is not available. Please ensure the NCCL-EP library "
                "(libnccl_ep.so) is built and NCCL_HOME is set."
            )

        cls._hidden_size = hidden_size
        cls._num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        cls._num_experts = num_experts
        cls._num_local_experts = num_local_experts
        cls._world_size = dist.get_world_size(group)
        cls._rank = dist.get_rank(group)

        nccl_ep_so = os.environ.get("NCCL_EP_SO", None)
        cls._lib = NCCLLibrary(nccl_ep_so)

        if not cls._lib.ep_available:
            raise RuntimeError(
                "NCCL library loaded but EP symbols not found. "
                "Ensure libnccl_ep.so is built with EP support."
            )

        cls._comm = get_nccl_comm_from_group(group, cls._lib)

        # NCCL-EP is only used for Low-Latency mode;
        # High-Throughput mode uses DeepEP.
        cls._algorithm = ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY

        token_size_bytes = hidden_size * torch.finfo(params_dtype).bits // 8

        config = ncclEpGroupConfig_t()
        config.version = 1
        config.algorithm = cls._algorithm
        config.num_experts = num_experts
        config.max_tokens_per_rank = num_max_dispatch_tokens_per_rank
        config.token_size_bytes = token_size_bytes
        config.rdma_buffer_size = 0
        config.num_qp_per_rank = 0
        config.num_channels = 0

        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        cls._ep_group = cls._lib.ncclEpCreateGroup(
            cls._comm, config, stream
        )

        logger.info(
            f"NCCL-EP group created (world_size={cls._world_size}, rank={cls._rank}, "
            f"algorithm={'LL' if cls._algorithm == ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY else 'HT'}, "
            f"num_experts={num_experts}, num_local_experts={num_local_experts})"
        )
        return cls

    @classmethod
    def destroy(cls):
        if cls._ep_group is not None:
            stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            cls._lib.ncclEpGroupDestroy(cls._ep_group, stream)
            cls._ep_group = None

    @classmethod
    def is_low_latency(cls):
        return cls._algorithm == ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY


class _NcclEPDispatcherImplBase:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        deepep_mode: DeepEPMode,
    ):
        if not use_nccl_ep:
            raise ImportError(
                "NCCL-EP is not available. Please ensure the NCCL-EP library "
                "(libnccl_ep.so) is built and NCCL_HOME is set."
            )

        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode
        self.world_size = dist.get_world_size(group)

        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_NCCL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        assert self.num_max_dispatch_tokens_per_rank <= 1024

        self.handle = None
        self.quant_config: Optional[dict] = None
        self.overlap_args = None
        self.meta_overlap_args = None

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config

    def set_overlap_args(self, combine_overlap_args, meta_overlap_args) -> None:
        self.overlap_args = combine_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        self.overlap_args = None
        self.meta_overlap_args = None

    def dispatch_a(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        raise NotImplementedError

    def dispatch_b(self, *args, **kwargs):
        raise NotImplementedError

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError

    def combine_b(self, *args, **kwargs):
        raise NotImplementedError

    def _get_buffer(self):
        return NcclEPBuffer.get_nccl_ep_buffer(
            self.group,
            self.hidden_size,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            self.num_local_experts,
            self.params_dtype,
        )


class _NcclEPDispatcherImplLowLatency(_NcclEPDispatcherImplBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device_module = torch.get_device_module()

    def dispatch_a(self, hidden_states: torch.Tensor, topk_output: TopKOutput, static_scale: torch.Tensor = None):
        buf = self._get_buffer()
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(torch.int64)
        expected_m = (
            hidden_states.shape[0] * buf._world_size * topk_ids.shape[1]
            + self.num_experts
        ) // self.num_experts

        recv_hidden, recv_count, use_fp8 = self._dispatch_core(
            hidden_states, topk_ids
        )
        return (
            recv_hidden,
            topk_ids,
            topk_weights,
            recv_count,
            expected_m,
            use_fp8,
        )

    def dispatch_b(
        self,
        recv_hidden,
        topk_ids,
        topk_weights,
        recv_count,
        expected_m,
        use_fp8,
    ):
        get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency(
            recv_count
        )

        if isinstance(recv_hidden, tuple):
            hidden_states, hidden_states_scale = recv_hidden
        else:
            hidden_states = recv_hidden
            hidden_states_scale = None

        return NcclEPLLDispatchOutput(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            recv_count,
            expected_m,
        )

    def _dispatch_core(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor):
        use_fp8 = not envs.SGLANG_NCCL_EP_BF16_DISPATCH.get()
        buf = self._get_buffer()
        lib = buf._lib
        ep_group = buf._ep_group
        stream_ptr = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

        handles_to_destroy = []
        try:
            topk_handle = _create_ep_tensor(
                lib, ep_group, topk_ids,
                ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_IDX,
            )
            handles_to_destroy.append(topk_handle)

            self.handle = lib.ncclEpCreateHandle(
                ep_group, topk_handle, None, stream_ptr, use_fp8=use_fp8
            )

            # LL dispatch: input must always be BF16; the C kernel handles
            # BF16->FP8 conversion internally and writes FP8 output + scales.
            dispatch_hidden = hidden_states

            in_token_handle = _create_ep_tensor(
                lib, ep_group, dispatch_hidden,
                ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            )
            handles_to_destroy.append(in_token_handle)

            num_recv = self.num_max_dispatch_tokens_per_rank * self.world_size
            # Output dtype is FP8 when use_fp8, BF16 otherwise
            out_dtype = torch.float8_e4m3fn if use_fp8 else dispatch_hidden.dtype
            out_shape = (self.num_local_experts, num_recv, dispatch_hidden.shape[-1])
            out_hidden = torch.empty(
                out_shape, dtype=out_dtype, device=hidden_states.device
            )
            out_token_handle = _create_ep_tensor(
                lib, ep_group, out_hidden,
                ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            )
            handles_to_destroy.append(out_token_handle)

            out_scale = None
            out_scale_handle = None
            if use_fp8:
                scale_shape = (
                    self.num_local_experts,
                    num_recv,
                    self.hidden_size // 128,
                )
                out_scale = torch.empty(
                    scale_shape, dtype=torch.float32, device=hidden_states.device
                )
                out_scale_handle = _create_ep_tensor(
                    lib, ep_group, out_scale,
                    ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_SCALES,
                )
                handles_to_destroy.append(out_scale_handle)

            num_tokens_per_expert = torch.zeros(
                self.num_local_experts, dtype=torch.int32, device=hidden_states.device
            )
            tpe_handle = _create_ep_tensor(
                lib, ep_group, num_tokens_per_expert,
                ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
            )
            handles_to_destroy.append(tpe_handle)

            in_handles = [in_token_handle]
            out_handles = [out_token_handle]
            if out_scale_handle is not None:
                out_handles.append(out_scale_handle)
            local_handles = [tpe_handle]

            in_ptr, n_in = _make_handle_array(in_handles)
            out_ptr, n_out = _make_handle_array(out_handles)
            local_ptr, n_local = _make_handle_array(local_handles)

            dispatch_config = ncclEpDispatchConfig_t()
            dispatch_config.round_scales = int(
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL
            )

            lib.ncclEpDispatch(
                self.handle,
                in_ptr, n_in,
                out_ptr, n_out,
                local_ptr, n_local,
                0,
                dispatch_config,
                stream_ptr,
            )
        finally:
            _destroy_tensor_handles(lib, ep_group, handles_to_destroy)

        if use_fp8:
            return (out_hidden, out_scale), num_tokens_per_expert, use_fp8
        return out_hidden, num_tokens_per_expert, use_fp8

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        combined = self._combine_core(hidden_states, topk_ids, topk_weights)
        return (combined,)

    def combine_b(self, combined):
        self.handle = None
        return combined

    def _combine_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        buf = self._get_buffer()
        lib = buf._lib
        ep_group = buf._ep_group
        stream_ptr = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

        handles_to_destroy = []
        try:
            in_token_handle = _create_ep_tensor(
                lib, ep_group, hidden_states,
                ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            )
            handles_to_destroy.append(in_token_handle)

            num_tokens = topk_ids.shape[0]
            out_hidden = torch.empty(
                (num_tokens, hidden_states.shape[-1]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            out_token_handle = _create_ep_tensor(
                lib, ep_group, out_hidden,
                ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            )
            handles_to_destroy.append(out_token_handle)

            topk_weights_handle = _create_ep_tensor(
                lib, ep_group, topk_weights,
                ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
            )
            handles_to_destroy.append(topk_weights_handle)

            in_ptr, n_in = _make_handle_array([in_token_handle])
            out_ptr, n_out = _make_handle_array([out_token_handle])
            local_ptr, n_local = _make_handle_array([topk_weights_handle])

            lib.ncclEpCombine(
                self.handle,
                in_ptr, n_in,
                out_ptr, n_out,
                local_ptr, n_local,
                0,
                None,
                stream_ptr,
            )
        finally:
            _destroy_tensor_handles(lib, ep_group, handles_to_destroy)

        lib.ncclEpHandleDestroy(self.handle)
        self.handle = None

        return out_hidden


class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class NcclEPDispatcher(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        super().__init__()
        self.deepep_mode = deepep_mode

        common_kwargs = dict(
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
        )

        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _NcclEPDispatcherImplLowLatency(
                **common_kwargs,
            )
        if self.deepep_mode.enable_normal():
            # HT/Normal mode uses DeepEP instead of NCCL-EP
            deepep_normal_kwargs = dict(common_kwargs)
            deepep_normal_kwargs["deepep_mode"] = DeepEPMode.NORMAL
            self._normal_dispatcher = _DeepEPDispatcherImplNormal(
                async_finish=async_finish,
                **deepep_normal_kwargs,
            )

        self._stage = _Stage.INITIAL
        self._deepep_dispatch_hooks = DeepEPPDispatchHooks()

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        static_scale: torch.Tensor = None,
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states=hidden_states, topk_output=topk_output, static_scale=static_scale)
        if self._deepep_dispatch_hooks is not None:
            self._deepep_dispatch_hooks(self)
        ret = self.dispatch_b()
        return ret

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        static_scale: torch.Tensor = None,
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        inner_state = self._get_impl().dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
            static_scale=static_scale,
        )
        self._dispatch_intermediate_state = inner_state

    def dispatch_b(self):
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl().dispatch_b(*inner_state)

    def combine(
        self,
        combine_input: CombineInput,
    ) -> torch.Tensor:
        self.combine_a(combine_input)
        ret = self.combine_b()
        return ret

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        hidden_states, topk_ids, topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl().combine_a(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = inner_state

    def combine_b(self):
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl().combine_b(*inner_state)

    def _get_impl(self):
        is_extend_in_batch = get_is_extend_in_batch()
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
        if resolved_deepep_mode == DeepEPMode.NORMAL:
            return self._normal_dispatcher
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage
        self._stage = new_stage

    def set_quant_config(self, quant_config: dict):
        super().set_quant_config(quant_config)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_quant_config(quant_config)
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_quant_config(quant_config)

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ):
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )

    def clear_overlap_args(self):
        super().clear_overlap_args()
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.clear_overlap_args()
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.clear_overlap_args()

    def register_deepep_dispatch_hook(self, hook):
        return self._deepep_dispatch_hooks.register_hook(hook)
