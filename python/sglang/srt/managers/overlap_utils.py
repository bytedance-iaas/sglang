from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.speculative.spec_utils import spec_need_hidden_states
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_is_cuda = is_cuda()
_is_hip = is_hip()


def _resolve_future_token_ids_native(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


if _is_cuda or _is_hip:
    from sglang.jit_kernel.resolve_future_token_ids import (
        resolve_future_token_ids_cuda,
    )

    _resolve_future_token_ids = resolve_future_token_ids_cuda
else:
    _resolve_future_token_ids = _resolve_future_token_ids_native


@dataclass
class FutureIndices:
    indices: torch.Tensor
    interval: Optional[slice] = None
    slots: Optional[torch.Tensor] = None
    slot_interval: Optional[slice] = None


class FutureMap:
    def __init__(
        self,
        max_running_requests: int,
        chunked_prefill_size: int,
        context_len: int,
        device: torch.device,
        spec_algo: SpeculativeAlgorithm,
    ):
        # FIXME: the calculation of future_limit and future_buffer_len maybe too conservative
        self.future_ct = 0

        # Circular buffer layout (wraps in this order):
        # Running decode batch -> Prefill chunk 1 -> ... -> Prefill chunk N
        # A running decode batch's result will be resolved after all prefill chunks are done.
        # reserve `max_num_chunks` extra future slots on top of `max_running_requests * 3`.
        max_num_chunks = (
            (context_len + chunked_prefill_size - 1) // chunked_prefill_size
            if chunked_prefill_size
            else 0
        )
        self.future_limit = max_running_requests * (3 + max_num_chunks)
        # Adding 2 * max_running_requests to future_limit ensures the buffer is sufficiently large.
        self.future_buffer_len = self.future_limit + 2 * max_running_requests
        self.device = device
        self.spec_algo = spec_algo

        if self.spec_algo.is_none():
            # For non-speculative decoding, we only need to store the token ids.
            self.buf_initialized = True
            self.token_ids_buf = torch.empty(
                (self.future_buffer_len,), dtype=torch.int64, device=self.device
            )
        else:
            storage_max_running_requests = max_running_requests
            try:
                from sglang.srt.layers.dp_attention import (
                    get_attention_dp_size,
                    is_dp_attention_enabled,
                )

                if is_dp_attention_enabled():
                    dp_size = get_attention_dp_size()
                    storage_max_running_requests = max(
                        min(max_running_requests, 128),
                        (max_running_requests + dp_size - 1) // dp_size,
                        1,
                    )
            except Exception:
                pass
            self.storage_buffer_len = min(
                max(storage_max_running_requests * 5, 1),
                self.future_buffer_len,
            )
            self.future_to_slot = torch.empty(
                (self.future_buffer_len,), dtype=torch.int64, device=self.device
            )
            self.future_to_slot.fill_(-1)
            self.free_slots = list(range(self.storage_buffer_len))
            # For speculative decoding, we lazily initialize the buffers
            # This is to make the shape derivation easier.
            self.buf_initialized = False

    def _lazy_init_buf(self, draft_input: EagleDraftInput):
        self.buf_initialized = True

        # Get a reference for each tensor
        topk_p0 = draft_input.topk_p[0]
        topk_index0 = draft_input.topk_index[0]
        bonus_token0 = draft_input.bonus_tokens[0]
        new_seq_lens0 = draft_input.new_seq_lens[0]

        self.topk_p_buf = torch.empty(
            (self.storage_buffer_len, *topk_p0.shape),
            dtype=topk_p0.dtype,
            device=self.device,
        )
        self.topk_index_buf = torch.empty(
            (self.storage_buffer_len, *topk_index0.shape),
            dtype=topk_index0.dtype,
            device=self.device,
        )
        self.bonus_tokens_buf = torch.empty(
            (self.storage_buffer_len, *bonus_token0.shape),
            dtype=bonus_token0.dtype,
            device=self.device,
        )
        self.new_seq_lens_buf = torch.empty(
            (self.storage_buffer_len, *new_seq_lens0.shape),
            dtype=new_seq_lens0.dtype,
            device=self.device,
        )

        if spec_need_hidden_states():
            hidden_states0 = draft_input.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.storage_buffer_len, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

    def _grow_storage(self, min_free_slots: int):
        if self.storage_buffer_len >= self.future_buffer_len:
            return
        old_len = self.storage_buffer_len
        new_len = min(
            max(self.storage_buffer_len * 2, self.storage_buffer_len + min_free_slots),
            self.future_buffer_len,
        )

        def grow_tensor(buf: torch.Tensor):
            new_buf = torch.empty(
                (new_len, *buf.shape[1:]), dtype=buf.dtype, device=buf.device
            )
            new_buf[:old_len] = buf
            return new_buf

        self.topk_p_buf = grow_tensor(self.topk_p_buf)
        self.topk_index_buf = grow_tensor(self.topk_index_buf)
        self.bonus_tokens_buf = grow_tensor(self.bonus_tokens_buf)
        self.new_seq_lens_buf = grow_tensor(self.new_seq_lens_buf)
        if spec_need_hidden_states():
            self.hidden_states_buf = grow_tensor(self.hidden_states_buf)

        self.free_slots.extend(range(old_len, new_len))
        self.storage_buffer_len = new_len

    def _alloc_storage_slots(self, bs: int) -> torch.Tensor:
        if len(self.free_slots) < bs:
            self._grow_storage(bs - len(self.free_slots))
        if len(self.free_slots) < bs:
            raise RuntimeError(
                "FutureMap compact storage is exhausted: "
                f"need {bs} slots, free {len(self.free_slots)}, "
                f"storage_buffer_len={self.storage_buffer_len}, "
                f"future_buffer_len={self.future_buffer_len}"
            )
        slot_list = self.free_slots[-bs:]
        del self.free_slots[-bs:]
        return torch.tensor(slot_list, dtype=torch.int64, device=self.device)

    def alloc_future_indices(self, bs: int) -> FutureIndices:
        """Update the circular buffer pointer and allocate future indices."""
        cur_future_ct = self.future_ct
        self.future_ct = (cur_future_ct + bs) % self.future_limit
        start = cur_future_ct + 1
        end = cur_future_ct + 1 + bs
        indices = torch.arange(start, end, dtype=torch.int64, device=self.device)
        if self.spec_algo.is_none():
            return FutureIndices(indices=indices, interval=slice(start, end))

        return FutureIndices(
            indices=indices,
            interval=slice(start, end),
        )

    def resolve_future(self, batch: ScheduleBatch):
        if self.spec_algo.is_none():
            _resolve_future_token_ids(batch.input_ids, self.token_ids_buf)
        else:
            # TODO(lsyin): write future indices into spec_info.future_indices
            draft_input: EagleDraftInput = batch.spec_info
            if draft_input is None:
                # FIXME(lsyin): No future exists, only for prefill batch, not compatible with mixed mode
                return
            indices = draft_input.future_indices.indices
            # The indices tensor was allocated on the default stream but is
            # used here on the forward stream. Meanwhile, the old spec_info
            # holding this tensor will lose all Python references (replaced at
            # batch.spec_info), so the caching allocator (torch GC) could
            # reclaim the memory before the GPU finishes reading it.
            indices.record_stream(torch.get_device_module(self.device).current_stream())
            slots = self.future_to_slot[indices]
            if torch.any(slots < 0):
                raise RuntimeError(
                    "FutureMap resolved a future id without a compact storage slot."
                )
            draft_input.topk_p = self.topk_p_buf[slots]
            draft_input.topk_index = self.topk_index_buf[slots]
            draft_input.bonus_tokens = self.bonus_tokens_buf[slots]
            draft_input.new_seq_lens = self.new_seq_lens_buf[slots]
            if spec_need_hidden_states():
                draft_input.hidden_states = self.hidden_states_buf[slots]
            valid_slots = slots[slots >= 0]
            if len(valid_slots) > 0:
                self.future_to_slot[indices] = -1
                self.free_slots.extend(valid_slots.detach().cpu().tolist())

    def is_empty_slice(self, s: slice) -> bool:
        start, stop, step = s.indices(self.future_buffer_len)
        if step > 0:
            return start >= stop
        else:
            return start <= stop

    def store_to_map(
        self, future_indices: FutureIndices, batch_result: GenerationBatchResult
    ):
        if self.spec_algo.is_none():
            intv = future_indices.interval
            if self.is_empty_slice(intv):
                # idle indices in dp attention do not need store info
                return
            self.token_ids_buf[intv] = batch_result.next_token_ids
        else:
            draft_input: EagleDraftInput = batch_result.next_draft_input
            self.store_to_map_for_new_batch(future_indices, draft_input)

    def store_to_map_for_new_batch(
        self, future_indices: FutureIndices, draft_input: EagleDraftInput
    ):
        intv = future_indices.interval
        if self.is_empty_slice(intv):
            # idle indices in dp attention do not need store info
            return

        if not self.buf_initialized:
            self._lazy_init_buf(draft_input)

        slots = future_indices.slots
        if slots is None:
            slots = self._alloc_storage_slots(len(future_indices.indices))
            future_indices.slots = slots
        else:
            slots = slots.to(device=self.device, dtype=torch.int64)
        self.future_to_slot[future_indices.indices] = slots

        self.topk_p_buf[slots] = draft_input.topk_p.to(dtype=self.topk_p_buf.dtype)
        self.topk_index_buf[slots] = draft_input.topk_index.to(
            dtype=self.topk_index_buf.dtype
        )
        self.bonus_tokens_buf[slots] = draft_input.bonus_tokens.to(
            dtype=self.bonus_tokens_buf.dtype
        )
        self.new_seq_lens_buf[slots] = draft_input.new_seq_lens.to(
            dtype=self.new_seq_lens_buf.dtype
        )
        if spec_need_hidden_states():
            self.hidden_states_buf[slots] = draft_input.hidden_states
