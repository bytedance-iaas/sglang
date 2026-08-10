from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_SUPPORTED_DRAFT_BACKENDS = (
    "flashinfer",
    "fa3",
    "fa4",
    "triton",
    "ascend",
    "trtllm_mha",
    "dsv4",
)


class DraftWorkerBundle(msgspec.Struct, frozen=True):
    draft_worker: TpModelWorker
    draft_model_runner: ModelRunner
    draft_model: torch.nn.Module
    resolved_attention_backend: str


def _resolve_draft_attention_backend_fallback(
    *, server_args: ServerArgs, algo_label: str
) -> str:
    draft_backend = server_args.speculative_draft_attention_backend
    if draft_backend is None:
        draft_backend, _ = server_args.get_attention_backends()
    if draft_backend is None:
        return "triton" if torch.version.hip else "flashinfer"
    if draft_backend not in _SUPPORTED_DRAFT_BACKENDS:
        fallback = "triton" if torch.version.hip else "flashinfer"
        logger.warning(
            "%s draft worker only supports attention_backend in %s for now, "
            "but got %r. Falling back to '%s'.",
            algo_label,
            _SUPPORTED_DRAFT_BACKENDS,
            draft_backend,
            fallback,
        )
        return fallback
    return draft_backend


def build_draft_tp_worker(
    *,
    server_args: ServerArgs,
    gpu_id: int,
    ps: ParallelState,
    nccl_port: int,
    target_worker: TpModelWorker,
    algo_label: str,
    attention_backend_override: Optional[str] = None,
    defer_device_graph_init: bool = False,
) -> DraftWorkerBundle:
    """Build a draft worker against the fork's legacy TpModelWorker API."""

    draft_backend = attention_backend_override or (
        _resolve_draft_attention_backend_fallback(
            server_args=server_args, algo_label=algo_label
        )
    )
    draft_server_args = deepcopy(server_args)
    draft_server_args.skip_tokenizer_init = True
    draft_server_args.speculative_draft_attention_backend = None
    draft_server_args.prefill_attention_backend = None
    draft_server_args.decode_attention_backend = None
    draft_server_args.attention_backend = draft_backend
    draft_server_args.context_length = target_worker.model_runner.model_config.context_len

    req_to_token_pool, target_allocator = target_worker.get_memory_pool()
    saved_server_args = get_global_server_args()
    try:
        draft_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            tp_rank=ps.tp_rank,
            moe_ep_rank=ps.moe_ep_rank,
            pp_rank=ps.pp_rank,
            attn_cp_rank=ps.attn_cp_rank,
            moe_dp_rank=ps.moe_dp_rank,
            dp_rank=ps.dp_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=target_allocator,
            memory_pool_config=target_worker.model_runner.memory_pool_config,
            defer_device_graph_init=defer_device_graph_init,
        )
    finally:
        set_global_server_args_for_scheduler(saved_server_args)

    draft_model_runner = draft_worker.model_runner
    draft_worker.draft_runner = draft_model_runner
    return DraftWorkerBundle(
        draft_worker=draft_worker,
        draft_model_runner=draft_model_runner,
        draft_model=draft_model_runner.model,
        resolved_attention_backend=draft_backend,
    )


def make_draft_input_v2(
    *, bonus_tokens: torch.Tensor, new_seq_lens: torch.Tensor
) -> DFlashDraftInputV2:
    bs = int(new_seq_lens.numel())
    device = bonus_tokens.device
    return DFlashDraftInputV2(
        topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
        topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
        bonus_tokens=bonus_tokens.to(dtype=torch.int64),
        new_seq_lens=new_seq_lens.to(dtype=torch.int64),
        hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
    )


def make_draft_block_spec_info(
    *, draft_token_num: int, device: torch.device
) -> DFlashVerifyInput:
    return DFlashVerifyInput(
        draft_token=torch.empty((0,), dtype=torch.long, device=device),
        positions=torch.empty((0,), dtype=torch.int64, device=device),
        draft_token_num=int(draft_token_num),
        custom_mask=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


def make_draft_sampler_capture_hook(draft_sampler):
    def capture_hook(runner, out, forward_batch, num_tokens):
        del runner, num_tokens
        if not isinstance(out, LogitsProcessorOutput) or out.hidden_states is None:
            raise RuntimeError(
                "draft sampler set but the draft forward has no hidden_states."
            )
        draft_sampler(out.hidden_states, forward_batch.input_ids)

    return capture_hook


def build_block_pos_offsets(*, length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(int(length), device=device, dtype=torch.int64)
