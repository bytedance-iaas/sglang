"""Minimal synchronous spec-v2 draft state used by bundled DSpark.

This fork forces DSpark overlap scheduling off.  Keep only the request-state and
decode KV reservation behavior needed by that synchronous path; do not import
the newer main-branch FutureMap/runtime-context stack.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func


@dataclass
class DFlashDraftInputV2(SpecInput):
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    bonus_tokens: torch.Tensor
    new_seq_lens: torch.Tensor
    hidden_states: torch.Tensor
    max_top_k: int = 1
    uniform_top_k_value: Optional[int] = None
    reserved_seq_lens_cpu: Optional[torch.Tensor] = None
    reserved_seq_lens_sum: Optional[int] = None
    future_indices: Optional[torch.Tensor] = None
    verify_token_budget: Optional[int] = None

    def __post_init__(self) -> None:
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return (1, 1)

    @classmethod
    def create_idle_input(cls, device: torch.device) -> "DFlashDraftInputV2":
        return cls(
            topk_p=torch.empty((0, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, 0), device=device, dtype=torch.int64),
            bonus_tokens=torch.empty((0,), device=device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int64),
            hidden_states=torch.empty((0, 0), device=device, dtype=torch.float16),
        )

    def prepare_for_decode(self, batch: ScheduleBatch) -> None:
        """Reserve the fixed DSpark verify window in the shared target mapping."""

        bs = batch.batch_size()
        if bs == 0:
            return

        # DFLASH/DSpark bypasses ScheduleBatch's ordinary decode preparation,
        # so it owns the per-step SWA maintenance clock. Evict before ticking
        # to preserve the first-iteration safety gate used by overlap-capable
        # speculative paths.
        batch.maybe_evict_swa()
        for req in batch.reqs:
            req.decode_batch_idx += 1

        # Result-based speculative decoding bypasses ScheduleBatch's ordinary
        # decode preparation.  Accumulate the tokens committed by the previous
        # verify step here so frequency/presence/repetition penalties observe
        # the same request history as non-speculative decoding.
        if batch.sampling_info.penalizer_orchestrator.is_required:
            output_ids = torch.tensor(
                [
                    (
                        req.output_ids[-1]
                        if req.output_ids
                        else req.origin_input_ids[-1]
                    )
                    for req in batch.reqs
                ],
                dtype=torch.int64,
                device=batch.device,
            )
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                output_ids
            )

        block_size = int(get_global_server_args().speculative_num_draft_tokens)
        if block_size <= 0:
            raise ValueError(
                f"DSpark invalid speculative_num_draft_tokens={block_size}."
            )

        committed_lens = torch.empty((bs,), dtype=torch.int64, device="cpu")
        current_lens = torch.empty((bs,), dtype=torch.int32, device="cpu")
        reserved_lens = torch.empty((bs,), dtype=torch.int32, device="cpu")
        committed_sum = 0
        reserved_sum = 0
        needed = 0
        max_top_k = 1
        uniform_top_k = None
        is_uniform = True
        max_reserved_len = 0
        max_reserved_req = None
        for index, req in enumerate(batch.reqs):
            committed = int(req.kv_committed_len)
            current = int(req.kv_allocated_len)
            reserved = max(current, committed + 2 * block_size)
            top_k = int(req.sampling_params.top_k)
            committed_lens[index] = committed
            current_lens[index] = current
            reserved_lens[index] = reserved
            if reserved > max_reserved_len:
                max_reserved_len = reserved
                max_reserved_req = (
                    getattr(req, "rid", None),
                    committed,
                    current,
                    reserved,
                )
            committed_sum += committed
            reserved_sum += reserved
            needed += reserved - current
            max_top_k = max(max_top_k, top_k)
            if index == 0:
                uniform_top_k = top_k
            elif top_k != uniform_top_k:
                is_uniform = False

        self.max_top_k = max_top_k
        self.uniform_top_k_value = uniform_top_k if is_uniform else None

        row_width = int(batch.req_to_token_pool.req_to_token.shape[1])
        if max_reserved_len > row_width:
            rid, committed, allocated, reserved = max_reserved_req
            raise RuntimeError(
                "DFLASH decode reservation exceeds the req_to_token row before "
                "KV allocation: "
                f"rid={rid!r}, committed={committed}, allocated={allocated}, "
                f"reserved={reserved}, row_width={row_width}, "
                f"page_size={batch.token_to_kv_pool_allocator.page_size}, "
                f"max_draft_tokens={block_size}."
            )

        current_gpu = current_lens.to(batch.device, non_blocking=True)
        reserved_gpu = reserved_lens.to(batch.device, non_blocking=True)
        if needed > 0:
            page_size = batch.token_to_kv_pool_allocator.page_size
            if page_size == 1:
                out_cache_loc = alloc_token_slots(batch.tree_cache, needed)
            else:
                last_loc = get_last_loc(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    current_gpu,
                )
                out_cache_loc = alloc_paged_token_slots_extend(
                    batch.tree_cache,
                    current_gpu,
                    current_lens,
                    reserved_gpu,
                    reserved_lens,
                    last_loc,
                    needed,
                )
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                current_gpu,
                reserved_gpu,
                out_cache_loc,
                bs,
            )

        for index, req in enumerate(batch.reqs):
            req.kv_allocated_len = max(
                req.kv_allocated_len, int(reserved_lens[index])
            )

        batch.seq_lens_cpu = committed_lens
        batch.seq_lens_sum = committed_sum
        self.reserved_seq_lens_cpu = reserved_lens
        self.reserved_seq_lens_sum = reserved_sum

    def filter_batch(
        self, new_indices: torch.Tensor, has_been_filtered: bool = True
    ) -> None:
        del has_been_filtered
        if self.reserved_seq_lens_cpu is not None:
            self.reserved_seq_lens_cpu = self.reserved_seq_lens_cpu[
                new_indices.cpu()
            ]
            self.reserved_seq_lens_sum = int(self.reserved_seq_lens_cpu.sum().item())
        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
            return
        self.topk_p = self.topk_p[new_indices]
        self.topk_index = self.topk_index[new_indices]
        self.bonus_tokens = self.bonus_tokens[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]
        self.hidden_states = self.hidden_states[new_indices]

    def merge_batch(self, spec_info: "DFlashDraftInputV2") -> None:
        if self.reserved_seq_lens_cpu is not None:
            if spec_info.reserved_seq_lens_cpu is None:
                raise RuntimeError("Cannot merge DSpark draft state without reservation")
            self.reserved_seq_lens_cpu = torch.cat(
                [self.reserved_seq_lens_cpu, spec_info.reserved_seq_lens_cpu]
            )
            self.reserved_seq_lens_sum = int(self.reserved_seq_lens_cpu.sum().item())
        elif spec_info.reserved_seq_lens_cpu is not None:
            self.reserved_seq_lens_cpu = spec_info.reserved_seq_lens_cpu
            self.reserved_seq_lens_sum = spec_info.reserved_seq_lens_sum

        if self.future_indices is not None:
            if spec_info.future_indices is None:
                raise RuntimeError("Cannot merge DSpark draft states with mixed futures")
            self.future_indices = torch.cat(
                [self.future_indices, spec_info.future_indices]
            )
            return
        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p], dim=0)
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index], dim=0)
        self.bonus_tokens = torch.cat(
            [self.bonus_tokens, spec_info.bonus_tokens], dim=0
        )
        self.new_seq_lens = torch.cat(
            [self.new_seq_lens, spec_info.new_seq_lens], dim=0
        )
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], dim=0
        )
