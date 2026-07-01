from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import triton
import triton.language as tl
from tvm_ffi.module import Module

from sglang.jit_kernel.dsv4.utils import make_name
from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.environ import envs
from sglang.srt.managers.hisparse_accuracy_trace import trace_event

logger = logging.getLogger(__name__)


@triton.jit
def _online_c128_mtp_prepare_kernel(
    seq_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    main_state_ptr,
    req_to_token_stride_b: tl.constexpr,
    main_state_stride_b: tl.constexpr,
    bs,
    temp_state_slot_offset: tl.constexpr,
    state_width: tl.constexpr,
    block_d: tl.constexpr,
):
    bid = tl.program_id(0)
    if bid >= bs:
        return

    d = tl.arange(0, block_d)
    d_mask = d < state_width
    seq_len = tl.load(seq_lens_ptr + bid).to(tl.int64)
    has_partial = (seq_len > 0) & ((seq_len % 128) != 0)

    req_idx = tl.load(req_pool_indices_ptr + bid).to(tl.int64)
    slot = tl.where(has_partial, req_idx, 0)
    temp_slot = slot + temp_state_slot_offset
    value = tl.load(
        main_state_ptr + slot * main_state_stride_b + d,
        mask=d_mask & has_partial,
        other=0.0,
    )
    tl.store(
        main_state_ptr + temp_slot * main_state_stride_b + d,
        value,
        mask=d_mask & has_partial,
    )


@cache_once
def _jit_online_c128_mtp_module(
    head_dim: int, dtype_buffer: torch.dtype = torch.float32
) -> Module:
    args = make_cpp_args(head_dim, dtype_buffer)
    return load_jit(
        make_name(f"online_c128_mtp_{head_dim}"),
        *args,
        cuda_files=["deepseek_v4/online_c128_mtp.cuh"],
        cuda_wrappers=[
            ("write_prefix_states", f"OnlineC128MTPWritePrefixKernel<{args}>::run"),
            ("mark_pending", f"OnlineC128MTPMarkPendingKernel<{args}>::run"),
            ("lazy_commit", f"OnlineC128MTPLazyCommitKernel<{args}>::run"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


_COMMIT_STATUS_NO_PENDING = 0
_COMMIT_STATUS_COMMITTED = 1
_COMMIT_STATUS_BOUNDARY_NO_STATE = 2
_COMMIT_STATUS_INVALID_REQ = -1
_COMMIT_STATUS_TAIL_MISMATCH = -2
_COMMIT_STATUS_NO_PROGRESS = -3

_COMMIT_STATUS_NAMES = {
    _COMMIT_STATUS_NO_PENDING: "no_pending",
    _COMMIT_STATUS_COMMITTED: "committed",
    _COMMIT_STATUS_BOUNDARY_NO_STATE: "boundary_no_state",
    _COMMIT_STATUS_INVALID_REQ: "invalid_req",
    _COMMIT_STATUS_TAIL_MISMATCH: "tail_mismatch",
    _COMMIT_STATUS_NO_PROGRESS: "no_progress",
}


def online_c128_mtp_write_prefix_states(
    *,
    kv_score_input: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    ape: torch.Tensor,
    state: torch.Tensor,
    layer_bs: int,
    num_verify_tokens: int,
    state_slot_stride: int,
    head_dim: int,
) -> None:
    if layer_bs <= 0:
        return
    _jit_online_c128_mtp_module(head_dim, state.dtype).write_prefix_states(
        kv_score_input,
        seq_lens,
        req_pool_indices,
        req_to_token,
        ape,
        state,
        layer_bs,
        num_verify_tokens,
        state_slot_stride,
    )


def online_c128_mtp_mark_pending(
    *,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    pending_seq_lens: torch.Tensor,
    pending_tail_locs: torch.Tensor,
    bs: int,
    head_dim: int,
) -> None:
    if bs <= 0:
        return
    _jit_online_c128_mtp_module(head_dim, torch.float32).mark_pending(
        seq_lens,
        req_pool_indices,
        req_to_token,
        pending_seq_lens,
        pending_tail_locs,
        bs,
        pending_seq_lens.shape[0],
    )


def online_c128_mtp_lazy_commit(
    *,
    cur_seq_lens: torch.Tensor,
    cur_req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    pending_seq_lens: torch.Tensor,
    pending_tail_locs: torch.Tensor,
    commit_status: torch.Tensor,
    state: torch.Tensor,
    cur_bs: int,
    num_verify_tokens: int,
    state_slot_stride: int,
    head_dim: int,
    clear_pending: bool,
) -> None:
    if cur_bs <= 0:
        return
    _jit_online_c128_mtp_module(head_dim, state.dtype).lazy_commit(
        cur_seq_lens,
        cur_req_pool_indices,
        req_to_token,
        pending_seq_lens,
        pending_tail_locs,
        commit_status,
        state,
        cur_bs,
        num_verify_tokens,
        state_slot_stride,
        pending_seq_lens.shape[0],
        clear_pending,
    )


def online_c128_mtp_prepare(
    *,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    main_state: torch.Tensor,
    bs: int,
    temp_state_slot_offset: int,
    state_width: int,
) -> None:
    if bs <= 0:
        return
    assert state_width <= 2048
    block_d = triton.next_power_of_2(state_width)
    _online_c128_mtp_prepare_kernel[(bs,)](
        seq_lens,
        req_pool_indices,
        req_to_token,
        main_state,
        req_to_token.stride(0),
        main_state.stride(0),
        bs,
        temp_state_slot_offset,
        state_width,
        block_d,
    )


@dataclass
class _OnlineC128LayerRuntime:
    head_dim: int
    main_state: torch.Tensor
    state_slot_offset: int
    state_width: int


@dataclass
class _OnlineC128VerifyContext:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    tail_locs: torch.Tensor


class OnlineC128MTPController:
    def __init__(self, backend: Any):
        self.backend = backend
        self._verify_ctx: Optional[_OnlineC128VerifyContext] = None
        self._layer_runtimes: Optional[List[_OnlineC128LayerRuntime]] = None
        self._state_debug_enabled = envs.SGLANG_DSV4_HISPARSE_STATE_DEBUG.get()
        self._accuracy_trace_enabled = (
            envs.SGLANG_DSV4_HISPARSE_ACCURACY_TRACE.get()
        )

    def _debug_enabled(self) -> bool:
        return self._state_debug_enabled

    def _ctx_shape(self, ctx: Optional[_OnlineC128VerifyContext]) -> str:
        if ctx is None:
            return "None"
        return (
            f"req_shape={tuple(ctx.req_pool_indices.shape)}, "
            f"seq_shape={tuple(ctx.seq_lens.shape)}, "
            f"tail_shape={tuple(ctx.tail_locs.shape)}"
        )

    def enabled(self) -> bool:
        if getattr(self.backend.model_runner, "is_draft_worker", False):
            return False
        return (
            envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
            and envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get()
            and self.backend.mtp_enabled
        )

    def state_slot_offset(self) -> int:
        if not self.enabled():
            return 0
        return self.backend.token_to_kv_pool.get_online_c128_mtp_state_slot_offset()

    def begin_verify(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        if not self.enabled():
            self.clear()
            return
        if self._verify_ctx is not None and self._debug_enabled():
            raise RuntimeError(
                "Online C128 MTP begin_verify found an uncommitted previous "
                f"verify context: {self._ctx_shape(self._verify_ctx)}."
            )

        bs = min(seq_lens.shape[0], req_pool_indices.shape[0])
        captured_req_pool_indices = req_pool_indices[:bs].detach().clone()
        captured_seq_lens = seq_lens[:bs].detach().clone()
        tail_locs = self._capture_tail_locs(
            captured_req_pool_indices, captured_seq_lens
        )
        self._verify_ctx = _OnlineC128VerifyContext(
            req_pool_indices=captured_req_pool_indices,
            seq_lens=captured_seq_lens,
            tail_locs=tail_locs.detach().clone(),
        )
        if self._accuracy_trace_enabled:
            trace_event(
                logger,
                "online_c128_begin_verify",
                bs=bs,
                num_verify_tokens=self._num_verify_tokens(),
                state_slot_offset=self.state_slot_offset(),
                req_pool_indices=captured_req_pool_indices,
                seq_lens=captured_seq_lens,
                tail_locs=tail_locs,
            )
        if captured_req_pool_indices.numel() == 0 or captured_seq_lens.numel() == 0:
            return
        if self._num_verify_tokens() == 0:
            return
        head_dim = self._head_dim()
        if head_dim is None:
            return
        token_to_kv_pool = self.backend.token_to_kv_pool
        online_c128_mtp_mark_pending(
            seq_lens=captured_seq_lens,
            req_pool_indices=captured_req_pool_indices,
            req_to_token=self.backend.req_to_token,
            pending_seq_lens=token_to_kv_pool.get_online_c128_mtp_pending_seq_lens(),
            pending_tail_locs=token_to_kv_pool.get_online_c128_mtp_pending_tail_locs(),
            bs=bs,
            head_dim=head_dim,
        )
        for runtime in self._iter_layer_runtimes():
            online_c128_mtp_prepare(
                seq_lens=captured_seq_lens,
                req_pool_indices=captured_req_pool_indices,
                req_to_token=self.backend.req_to_token,
                main_state=runtime.main_state,
                bs=bs,
                temp_state_slot_offset=runtime.state_slot_offset,
                state_width=runtime.state_width,
            )

    def clear(self) -> None:
        ctx = self._verify_ctx
        if ctx is not None and self._accuracy_trace_enabled:
            trace_event(
                logger,
                "online_c128_clear",
                bs=min(ctx.seq_lens.shape[0], ctx.req_pool_indices.shape[0]),
                num_verify_tokens=self._num_verify_tokens(),
                state_slot_offset=self.state_slot_offset(),
            )
        self._verify_ctx = None

    def prepare_forward(
        self,
        logical_forward_mode,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        verify_bs: Optional[int] = None,
    ) -> int:
        if not self.enabled():
            self.clear()
            return 0
        if logical_forward_mode.is_idle():
            self.clear()
            return 0

        active_req_pool_indices = req_pool_indices
        active_seq_lens = seq_lens
        if logical_forward_mode.is_target_verify():
            if verify_bs is None:
                verify_bs = req_pool_indices.shape[0]
            active_req_pool_indices = req_pool_indices[:verify_bs]
            active_seq_lens = seq_lens[:verify_bs]
            if verify_bs == 0:
                self.clear()
                return 0

        self.commit_pending(
            req_pool_indices=active_req_pool_indices,
            seq_lens=active_seq_lens,
        )
        if self._accuracy_trace_enabled:
            trace_event(
                logger,
                "online_c128_prepare_forward",
                mode=str(logical_forward_mode),
                verify_bs=verify_bs,
                active_bs=min(
                    active_req_pool_indices.shape[0],
                    active_seq_lens.shape[0],
                ),
                state_slot_offset=(
                    self.state_slot_offset()
                    if logical_forward_mode.is_target_verify()
                    else 0
                ),
                req_pool_indices=active_req_pool_indices,
                seq_lens=active_seq_lens,
            )
        if not logical_forward_mode.is_target_verify():
            return 0

        self.begin_verify(
            req_pool_indices=active_req_pool_indices,
            seq_lens=active_seq_lens,
        )
        return self.state_slot_offset()

    def write_prefix_states(
        self,
        layer_id: int,
        compressor: Any,
        kv_score_input: torch.Tensor,
        logical_forward_mode,
    ) -> None:
        if (
            not self.enabled()
            or not logical_forward_mode.is_target_verify()
            or compressor.is_in_indexer
            or compressor.ratio != 128
            or kv_score_input.numel() == 0
        ):
            return

        ctx = self._active_ctx()
        num_verify_tokens = self._num_verify_tokens()
        if ctx is None or num_verify_tokens == 0:
            return

        token_to_kv_pool = self.backend.token_to_kv_pool
        head_dim = compressor.head_dim
        state_pool = token_to_kv_pool.get_attention_compress_states(layer_id)
        total_bs = kv_score_input.numel() // (num_verify_tokens * head_dim * 2)
        layer_bs = min(ctx.seq_lens.shape[0], ctx.req_pool_indices.shape[0], total_bs)
        if layer_bs <= 0:
            return

        online_c128_mtp_write_prefix_states(
            kv_score_input=kv_score_input,
            seq_lens=ctx.seq_lens,
            req_pool_indices=ctx.req_pool_indices,
            req_to_token=self.backend.req_to_token,
            ape=compressor.ape.reshape(128, head_dim),
            state=state_pool.kv_score_buffer.kv_score,
            layer_bs=layer_bs,
            num_verify_tokens=num_verify_tokens,
            state_slot_stride=state_pool.online_mtp_state_slot_offset,
            head_dim=head_dim,
        )

    def commit_pending(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        ctx = self._verify_ctx
        if ctx is None:
            return
        if not self.enabled():
            self.clear()
            return
        if ctx.seq_lens.numel() == 0 or ctx.req_pool_indices.numel() == 0:
            self.clear()
            return
        if req_pool_indices.numel() == 0 or seq_lens.numel() == 0:
            return

        num_verify_tokens = self._num_verify_tokens()
        if num_verify_tokens == 0:
            self.clear()
            return

        pending_seq_lens = (
            self.backend.token_to_kv_pool.get_online_c128_mtp_pending_seq_lens()
        )
        cur_seq_lens = seq_lens.to(pending_seq_lens.device)
        cur_req_pool_indices = req_pool_indices.to(pending_seq_lens.device)
        cur_bs = min(cur_seq_lens.shape[0], cur_req_pool_indices.shape[0])
        if self._accuracy_trace_enabled:
            trace_event(
                logger,
                "online_c128_commit_pending",
                cur_bs=cur_bs,
                num_verify_tokens=num_verify_tokens,
                state_slot_offset=self.state_slot_offset(),
                old_req_pool_indices=ctx.req_pool_indices,
                cur_req_pool_indices=cur_req_pool_indices,
                old_seq_lens=ctx.seq_lens,
                cur_seq_lens=cur_seq_lens,
            )

        self._commit_current_pending(
            cur_req_pool_indices=cur_req_pool_indices,
            cur_seq_lens=cur_seq_lens,
            cur_bs=cur_bs,
            num_verify_tokens=num_verify_tokens,
            context="online_c128_commit_pending",
            allow_no_progress_clear=False,
        )
        self.clear()

    def flush_pending_for_reqs(
        self,
        reqs: List[Any],
        *,
        require_forward_progress: bool = False,
    ) -> None:
        """Commit pending target-verify C128 state before request release.

        The normal path commits pending state at the next forward. A request can
        finish immediately after target verify, however, and radix cache may
        snapshot C128 state before that next forward happens. Commit only the
        finishing request slots here and keep pending state for the rest of the
        batch.
        """
        if not self.enabled():
            return

        req_pool_indices = []
        seq_lens = []
        for req in reqs:
            req_pool_idx = getattr(req, "req_pool_idx", None)
            if req_pool_idx is None:
                continue
            req_pool_indices.append(int(req_pool_idx))
            seq_lens.append(int(req.seqlen))
        if not req_pool_indices:
            return

        pending_seq_lens = (
            self.backend.token_to_kv_pool.get_online_c128_mtp_pending_seq_lens()
        )
        cur_req_pool_indices = torch.tensor(
            req_pool_indices,
            dtype=torch.int64,
            device=pending_seq_lens.device,
        )
        cur_seq_lens = torch.tensor(
            seq_lens,
            dtype=torch.int64,
            device=pending_seq_lens.device,
        )

        num_verify_tokens = self._num_verify_tokens()
        if num_verify_tokens == 0:
            self._clear_pending_req_pool_indices(cur_req_pool_indices)
            return

        cur_bs = min(cur_seq_lens.shape[0], cur_req_pool_indices.shape[0])
        if cur_bs <= 0:
            return

        if self._accuracy_trace_enabled:
            trace_event(
                logger,
                "online_c128_flush_pending_for_reqs",
                cur_bs=cur_bs,
                num_verify_tokens=num_verify_tokens,
                state_slot_offset=self.state_slot_offset(),
                cur_req_pool_indices=cur_req_pool_indices,
                cur_seq_lens=cur_seq_lens,
            )

        self._commit_current_pending(
            cur_req_pool_indices=cur_req_pool_indices,
            cur_seq_lens=cur_seq_lens,
            cur_bs=cur_bs,
            num_verify_tokens=num_verify_tokens,
            context="online_c128_flush_pending_for_reqs",
            allow_no_progress_clear=not require_forward_progress,
        )
        self._drop_ctx_req_pool_indices(cur_req_pool_indices[:cur_bs])

    def _commit_current_pending(
        self,
        *,
        cur_req_pool_indices: torch.Tensor,
        cur_seq_lens: torch.Tensor,
        cur_bs: int,
        num_verify_tokens: int,
        context: str,
        allow_no_progress_clear: bool,
    ) -> None:
        if cur_bs <= 0:
            return
        runtimes = list(self._iter_layer_runtimes())
        if not runtimes:
            return

        backend = self.backend
        token_to_kv_pool = backend.token_to_kv_pool
        pending_seq_lens = token_to_kv_pool.get_online_c128_mtp_pending_seq_lens()
        pending_tail_locs = token_to_kv_pool.get_online_c128_mtp_pending_tail_locs()
        commit_status = torch.empty(
            (cur_bs,), dtype=torch.int32, device=pending_seq_lens.device
        )

        for i, runtime in enumerate(runtimes):
            commit_status.zero_()
            online_c128_mtp_lazy_commit(
                cur_seq_lens=cur_seq_lens,
                cur_req_pool_indices=cur_req_pool_indices,
                req_to_token=backend.req_to_token,
                pending_seq_lens=pending_seq_lens,
                pending_tail_locs=pending_tail_locs,
                commit_status=commit_status,
                state=runtime.main_state,
                cur_bs=cur_bs,
                num_verify_tokens=num_verify_tokens,
                state_slot_stride=runtime.state_slot_offset,
                head_dim=runtime.head_dim,
                clear_pending=(i == len(runtimes) - 1),
            )

        self._check_commit_status(
            commit_status=commit_status,
            cur_req_pool_indices=cur_req_pool_indices,
            cur_seq_lens=cur_seq_lens,
            cur_bs=cur_bs,
            context=context,
            allow_no_progress_clear=allow_no_progress_clear,
        )

    def _check_commit_status(
        self,
        *,
        commit_status: torch.Tensor,
        cur_req_pool_indices: torch.Tensor,
        cur_seq_lens: torch.Tensor,
        cur_bs: int,
        context: str,
        allow_no_progress_clear: bool,
    ) -> None:
        status_cpu = commit_status[:cur_bs].detach().cpu()
        bad = status_cpu < 0
        if not bool(bad.any().item()):
            return

        req_cpu = cur_req_pool_indices[:cur_bs].detach().cpu()
        seq_cpu = cur_seq_lens[:cur_bs].detach().cpu()
        bad_indices = bad.nonzero(as_tuple=False).flatten()
        bad_entries = []
        no_progress_req_indices = []
        for pos in bad_indices.tolist():
            status = int(status_cpu[pos].item())
            req_idx = int(req_cpu[pos].item())
            bad_entries.append(
                {
                    "pos": pos,
                    "req_pool_idx": req_idx,
                    "seq_len": int(seq_cpu[pos].item()),
                    "status": _COMMIT_STATUS_NAMES.get(status, str(status)),
                }
            )
            if status == _COMMIT_STATUS_NO_PROGRESS:
                no_progress_req_indices.append(req_idx)

        all_no_progress = all(
            int(status_cpu[pos].item()) == _COMMIT_STATUS_NO_PROGRESS
            for pos in bad_indices.tolist()
        )
        if allow_no_progress_clear and all_no_progress:
            self._clear_pending_req_pool_indices(
                torch.tensor(
                    no_progress_req_indices,
                    dtype=torch.int64,
                    device=cur_req_pool_indices.device,
                )
            )
            return

        raise RuntimeError(
            "Online C128 MTP commit failed; refusing to clear pending state. "
            f"context={context}, entries={bad_entries}"
        )

    def _clear_pending_req_pool_indices(self, req_pool_indices: torch.Tensor) -> None:
        if req_pool_indices.numel() == 0:
            return
        token_to_kv_pool = self.backend.token_to_kv_pool
        pending_seq_lens = token_to_kv_pool.get_online_c128_mtp_pending_seq_lens()
        pending_tail_locs = token_to_kv_pool.get_online_c128_mtp_pending_tail_locs()
        reqs = req_pool_indices.to(device=pending_seq_lens.device, dtype=torch.long)
        valid = (reqs >= 0) & (reqs < pending_seq_lens.shape[0])
        reqs = reqs[valid]
        if reqs.numel() == 0:
            return
        pending_seq_lens[reqs] = -1
        pending_tail_locs[reqs] = 0

    def _drop_ctx_req_pool_indices(self, req_pool_indices: torch.Tensor) -> None:
        ctx = self._verify_ctx
        if ctx is None or req_pool_indices.numel() == 0:
            return
        old_bs = min(ctx.seq_lens.shape[0], ctx.req_pool_indices.shape[0])
        if old_bs <= 0:
            self.clear()
            return
        old_req_pool_indices = ctx.req_pool_indices[:old_bs]
        remove = (
            old_req_pool_indices.reshape(-1, 1)
            == req_pool_indices.to(old_req_pool_indices.device).reshape(1, -1)
        ).any(dim=1)
        keep = ~remove
        if not bool(keep.any().item()):
            self.clear()
            return
        self._verify_ctx = _OnlineC128VerifyContext(
            req_pool_indices=ctx.req_pool_indices[:old_bs][keep].detach().clone(),
            seq_lens=ctx.seq_lens[:old_bs][keep].detach().clone(),
            tail_locs=ctx.tail_locs[:old_bs][keep].detach().clone(),
        )

    def _num_verify_tokens(self) -> int:
        if not self.enabled():
            return 0
        num_verify_tokens = int(self.backend.speculative_num_draft_tokens)
        max_draft_tokens = (
            self.backend.token_to_kv_pool.get_online_c128_mtp_max_draft_tokens()
        )
        return num_verify_tokens if 0 < num_verify_tokens <= max_draft_tokens else 0

    def _capture_tail_locs(
        self, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor
    ) -> torch.Tensor:
        if req_pool_indices.numel() == 0 or seq_lens.numel() == 0:
            return torch.empty(0, dtype=torch.int32, device=req_pool_indices.device)
        bs = min(req_pool_indices.shape[0], seq_lens.shape[0])
        reqs = req_pool_indices[:bs].to(dtype=torch.long)
        lens = seq_lens[:bs].to(dtype=torch.long)
        positions = torch.clamp(lens - 1, min=0)
        tail_locs = self.backend.req_to_token[reqs, positions]
        return torch.where(lens > 0, tail_locs, torch.zeros_like(tail_locs)).to(
            dtype=torch.int32
        )

    def _active_ctx(self) -> Optional[_OnlineC128VerifyContext]:
        ctx = self._verify_ctx
        if (
            ctx is None
            or ctx.seq_lens.numel() == 0
            or ctx.req_pool_indices.numel() == 0
        ):
            return None
        return ctx

    def _head_dim(self) -> Optional[int]:
        for runtime in self._iter_layer_runtimes():
            return runtime.head_dim
        return None

    def _iter_layer_runtimes(self):
        if self._layer_runtimes is None:
            runtimes = []
            token_to_kv_pool = self.backend.token_to_kv_pool
            for layer in self.backend.model_runner.model.model.layers:
                attn = getattr(layer, "self_attn", None)
                compressor = getattr(attn, "compressor", None)
                if compressor is None or compressor.ratio != 128:
                    continue
                state_pool = token_to_kv_pool.get_attention_compress_states(
                    compressor.layer_id
                )
                runtimes.append(
                    _OnlineC128LayerRuntime(
                        head_dim=compressor.head_dim,
                        main_state=state_pool.kv_score_buffer.kv_score,
                        state_slot_offset=state_pool.online_mtp_state_slot_offset,
                        state_width=compressor.head_dim * 3,
                    )
                )
            self._layer_runtimes = runtimes
        return iter(self._layer_runtimes)
