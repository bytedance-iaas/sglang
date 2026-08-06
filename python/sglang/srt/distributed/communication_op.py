# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch
import torch.distributed

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)

from .parallel_state import (
    get_attn_tp_group,
    get_moe_ep_group,
    get_moe_tp_group,
    get_tp_group,
)

_enable_cuda_graph_collective_break: ContextVar[bool] = ContextVar(
    "enable_cuda_graph_collective_break", default=False
)


@contextmanager
def cuda_graph_collective_break(enabled: bool = True) -> Iterator[None]:
    """Route model-parallel collectives through BCG eager break points.

    The context is intentionally scoped to the owning Breakable-CUDA-Graph
    capture session. Outside an active BCG capture, ``eager_on_graph`` calls
    the underlying collective directly, so eager execution keeps its existing
    behavior.
    """
    token = _enable_cuda_graph_collective_break.set(enabled)
    try:
        yield
    finally:
        _enable_cuda_graph_collective_break.reset(token)


def is_cuda_graph_collective_break_enabled() -> bool:
    """Whether the current execution is capturing BCG collective breaks."""
    return _enable_cuda_graph_collective_break.get()


def _tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_tp_group().all_reduce(input_)


bcg_tensor_model_parallel_all_reduce = eager_on_graph(True)(
    _tensor_model_parallel_all_reduce
)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    if _enable_cuda_graph_collective_break.get():
        return bcg_tensor_model_parallel_all_reduce(input_)
    return _tensor_model_parallel_all_reduce(input_)


def tensor_model_parallel_quant_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().quant_all_reduce(input_)


def tensor_model_parallel_fused_allreduce_rmsnorm(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fused TP all-reduce + RMSNorm.

    Policy and backend selection are owned by GroupCoordinator:
    it may dispatch to communicator-native fused APIs, custom fused kernels,
    or return None so callers can run generic fallback paths.
    """
    return get_tp_group().fused_allreduce_rmsnorm(input_, residual_inp_, weight_, eps)


def tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
    group_size: int = 128,
    emit_bf16: bool = False,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """Fused TP all-reduce + RMSNorm + per-group FP8 quant (ROCm/aiter).

    Returns ``(fp8_output, residual_out, per_group_scale)`` by default, or
    ``(fp8_output, residual_out, per_group_scale, bf16_output)`` when
    ``emit_bf16=True`` (kernel writes both fp8 and the pre-quantization bf16
    normed output — no extra kernel). ``None`` when the backend cannot
    service the request (non-AMD, custom AR disabled, shape unsupported).
    Callers MUST handle ``None`` by falling back to the separate
    fused-AR-RMSNorm + per-group-quant path.
    """
    return get_tp_group().fused_allreduce_rmsnorm_quant_per_group(
        input_, residual_inp_, weight_, eps, group_size, emit_bf16=emit_bf16
    )


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)


def _attention_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_attn_tp_group().all_reduce(input_)


bcg_attention_tensor_model_parallel_all_reduce = eager_on_graph(True)(
    _attention_tensor_model_parallel_all_reduce
)


def attention_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across attention parallel group."""
    if _enable_cuda_graph_collective_break.get():
        return bcg_attention_tensor_model_parallel_all_reduce(input_)
    return _attention_tensor_model_parallel_all_reduce(input_)


def attention_tensor_model_parallel_quant_all_reduce(
    input_: torch.Tensor,
) -> torch.Tensor:
    """All-reduce the input tensor across attention parallel group."""
    return get_attn_tp_group().quant_all_reduce(input_)


def _moe_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_moe_tp_group().all_reduce(input_)


bcg_moe_tensor_model_parallel_all_reduce = eager_on_graph(True)(
    _moe_tensor_model_parallel_all_reduce
)


def moe_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across moe parallel group."""
    if _enable_cuda_graph_collective_break.get():
        return bcg_moe_tensor_model_parallel_all_reduce(input_)
    return _moe_tensor_model_parallel_all_reduce(input_)


def _moe_expert_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_moe_ep_group().all_reduce(input_)


bcg_moe_expert_parallel_all_reduce = eager_on_graph(True)(
    _moe_expert_parallel_all_reduce
)


def moe_expert_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across moe expert parallel group."""
    if _enable_cuda_graph_collective_break.get():
        return bcg_moe_expert_parallel_all_reduce(input_)
    return _moe_expert_parallel_all_reduce(input_)
