from __future__ import annotations

import os
import random
from collections import deque
from contextlib import nullcontext
from enum import Enum
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Type, overload

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.disaggregation.base.conn import KVArgs, StateType
    from sglang.srt.disaggregation.common.conn import (
        CommonKVBootstrapServer,
        CommonKVManager,
        CommonKVReceiver,
        CommonKVSender,
    )
    from sglang.srt.managers.schedule_batch import Req

#########################
# Constants & Enums
#########################
FAKE_BOOTSTRAP_HOST = "2.2.2.2"


def get_dsv4_full_indexed_c128_state_indices(
    req_to_token: torch.Tensor,
    req_pool_idx: int,
    seq_len: int,
) -> np.ndarray:
    """Return the C128 state page used by this fork's Full-indexed planner.

    A completed 128-token chunk has no running state to transfer.  Otherwise
    the compressor reads/writes the state page derived from the Full physical
    slot at the beginning of the current chunk.
    """
    if seq_len <= 0 or seq_len % 128 == 0:
        return np.empty((0,), dtype=np.int32)
    chunk_start = ((seq_len - 1) // 128) * 128
    full_loc = int(req_to_token[int(req_pool_idx), chunk_start].item())
    # KV slot/page 0 is reserved for padded writes by both token and paged
    # allocators.  A zero here therefore means that req_to_token has not been
    # populated for this logical position; it is not a transferable state row.
    if full_loc <= 0:
        raise RuntimeError(
            "DSV4 C128 state payload references an unallocated Full KV slot: "
            f"req_pool_idx={req_pool_idx}, chunk_start={chunk_start}, "
            f"full_loc={full_loc}"
        )
    return np.array([full_loc // 128], dtype=np.int32)


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

    @staticmethod
    def to_engine_type(mode: str) -> str:
        if mode == DisaggregationMode.PREFILL.value:
            return "prefill"
        elif mode == DisaggregationMode.DECODE.value:
            return "decode"
        return "unified"


#########################
# Synchronization
#########################

# env var for testing failure, convert to float explicitly
FAILURE_PROB = float(os.getenv("DISAGGREGATION_TEST_FAILURE_PROB", 0))


def poll_and_all_reduce(pollers, gloo_group: dist.ProcessGroup):
    # at a certain prob, the poll is failed to simulate failure
    if FAILURE_PROB > 0:
        from sglang.srt.disaggregation.base import KVPoll

        polls = [
            int(KVPoll.Failed) if random.random() < FAILURE_PROB else int(poller.poll())
            for poller in pollers
        ]
    else:
        polls = [int(poller.poll()) for poller in pollers]
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()


def poll_and_all_reduce_attn_cp_tp_group(
    pollers,
    attn_cp_cpu_group: dist.ProcessGroup,
    attn_tp_cpu_group: dist.ProcessGroup,
):
    # First sync across attn-tp ranks so all TP participants for a given (dp, cp)
    # shard observe the same status transitions.
    polls = poll_and_all_reduce(pollers, attn_tp_cpu_group)

    # Then sync across attn-cp ranks, so all TPxCP participants in one DP shard
    # converge to the same global status.
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(
        tensor_to_reduce,
        op=dist.ReduceOp.MIN,
        group=attn_cp_cpu_group,
    )
    return tensor_to_reduce.tolist()


def poll_and_all_reduce_with_staging(
    decode_reqs, staging_handler, gloo_group: dist.ProcessGroup
):
    """Staging-aware polling: advance scatter, demote incomplete transfers, all_reduce."""
    from sglang.srt.disaggregation.base import KVPoll

    for decode_req in decode_reqs:
        if decode_req.kv_receiver.require_staging and not staging_handler.is_done(
            decode_req
        ):
            staging_handler.advance_scatter(decode_req)

    raw_polls = [int(dr.kv_receiver.poll()) for dr in decode_reqs]
    for i, decode_req in enumerate(decode_reqs):
        if raw_polls[i] == int(KVPoll.Success):
            if decode_req.kv_receiver.require_staging and not staging_handler.is_done(
                decode_req
            ):
                raw_polls[i] = int(KVPoll.Transferring)
    poll_tensor = torch.tensor(raw_polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(poll_tensor, op=dist.ReduceOp.MIN, group=gloo_group)
    return poll_tensor.tolist()


#########################
# Metadata Buffers
#########################


class ReqToMetadataIdxAllocator:
    """A memory pool that maps a request to its first output token location."""

    def __init__(
        self,
        size: int,
    ):
        self.size = size
        self.free_slots = deque(list(range(size)))

    def available_size(self):
        return len(self.free_slots)

    def alloc(self) -> Optional[int]:
        if len(self.free_slots) == 0:
            return None

        return self.free_slots.popleft()

    def free(self, free_index: int):
        self.free_slots.append(free_index)


class MetadataBuffers:
    def __init__(
        self,
        size: int,
        hidden_size: int,
        hidden_states_dtype: torch.dtype,
        max_top_logprobs_num: int = 128,
        custom_mem_pool: torch.cuda.MemPool = None,
    ):
        self.custom_mem_pool = custom_mem_pool
        bootstrap_room_dtype = torch.uint64
        device = "cpu"
        if is_npu():
            # For ascend backend, output tokens are placed in the NPU and will be transferred by D2D channel.
            device = "npu"
            # TODO: Fix me when npu backend supports torch.uint64
            bootstrap_room_dtype = torch.int64
        elif self.custom_mem_pool:
            # TODO(shangming): Fix me (use 'cuda') when nvlink_transport of Mooncake is bug-free
            device = "cpu"
        elif envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() == "INTRA_NODE_NVLINK":
            device = "cpu"
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            # TODO: abort top_logprobs_num > 128 in PD

            # We transfer the metadata of first output token to decode
            # The minimal size for RDMA is 64Bytes, so we pad it to > 64Bytes
            self.output_ids = torch.zeros((size, 16), dtype=torch.int32, device=device)
            self.cached_tokens = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_token_logprobs_val = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_token_logprobs_idx = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_top_logprobs_val = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.float32, device=device
            )
            self.output_top_logprobs_idx = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.int32, device=device
            )
            # For PD + spec decode
            self.output_topk_p = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_topk_index = torch.zeros(
                (size, 16), dtype=torch.int64, device=device
            )
            self.output_hidden_states = torch.zeros(
                (size, hidden_size), dtype=hidden_states_dtype, device=device
            )
            # Request validation: store bootstrap_room to detect metadata corruption
            self.bootstrap_room = torch.zeros(
                (size, 8), dtype=bootstrap_room_dtype, device=device
            )

    def get_buf_infos(self):
        ptrs = [
            self.output_ids.data_ptr(),
            self.cached_tokens.data_ptr(),
            self.output_token_logprobs_val.data_ptr(),
            self.output_token_logprobs_idx.data_ptr(),
            self.output_top_logprobs_val.data_ptr(),
            self.output_top_logprobs_idx.data_ptr(),
            self.output_topk_p.data_ptr(),
            self.output_topk_index.data_ptr(),
            self.output_hidden_states.data_ptr(),
            self.bootstrap_room.data_ptr(),
        ]
        data_lens = [
            self.output_ids.nbytes,
            self.cached_tokens.nbytes,
            self.output_token_logprobs_val.nbytes,
            self.output_token_logprobs_idx.nbytes,
            self.output_top_logprobs_val.nbytes,
            self.output_top_logprobs_idx.nbytes,
            self.output_topk_p.nbytes,
            self.output_topk_index.nbytes,
            self.output_hidden_states.nbytes,
            self.bootstrap_room.nbytes,
        ]
        item_lens = [
            self.output_ids[0].nbytes,
            self.cached_tokens[0].nbytes,
            self.output_token_logprobs_val[0].nbytes,
            self.output_token_logprobs_idx[0].nbytes,
            self.output_top_logprobs_val[0].nbytes,
            self.output_top_logprobs_idx[0].nbytes,
            self.output_topk_p[0].nbytes,
            self.output_topk_index[0].nbytes,
            self.output_hidden_states[0].nbytes,
            self.bootstrap_room[0].nbytes,
        ]
        return ptrs, data_lens, item_lens

    def get_buf(self, idx: int):
        return (
            self.output_ids[idx].clone(),
            self.cached_tokens[idx].clone(),
            self.output_token_logprobs_val[idx].clone(),
            self.output_token_logprobs_idx[idx].clone(),
            self.output_top_logprobs_val[idx].clone(),
            self.output_top_logprobs_idx[idx].clone(),
            self.output_topk_p[idx].clone(),
            self.output_topk_index[idx].clone(),
            self.output_hidden_states[idx].clone(),
            self.bootstrap_room[idx].clone(),
        )

    def set_buf(self, req: Req):

        self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
        self.cached_tokens[req.metadata_buffer_index][0] = req.cached_tokens
        self.cached_tokens[req.metadata_buffer_index][1] = req.cached_tokens_device
        self.cached_tokens[req.metadata_buffer_index][2] = req.cached_tokens_host
        self.cached_tokens[req.metadata_buffer_index][3] = req.cached_tokens_storage
        if req.return_logprob:
            if req.logprob.output_token_logprobs_val:  # not none or empty list
                self.output_token_logprobs_val[req.metadata_buffer_index][0] = (
                    req.logprob.output_token_logprobs_val[0]
                )
            if req.logprob.output_token_logprobs_idx:  # not none or empty list
                self.output_token_logprobs_idx[req.metadata_buffer_index][0] = (
                    req.logprob.output_token_logprobs_idx[0]
                )

            if req.logprob.output_top_logprobs_val:  # not none or empty list
                self.output_top_logprobs_val[req.metadata_buffer_index][
                    : len(req.logprob.output_top_logprobs_val[0])
                ] = torch.tensor(
                    req.logprob.output_top_logprobs_val[0],
                    dtype=torch.float32,
                    device="cpu",
                )
            if req.logprob.output_top_logprobs_idx:  # not none or empty list
                self.output_top_logprobs_idx[req.metadata_buffer_index][
                    : len(req.logprob.output_top_logprobs_idx[0])
                ] = torch.tensor(
                    req.logprob.output_top_logprobs_idx[0],
                    dtype=torch.int32,
                    device="cpu",
                )
        # For PD + spec decode
        if req.hidden_states_tensor is not None:
            # speculative_eagle_topk should not be greater than 16 currently
            topk = req.output_topk_p.size(0)

            self.output_topk_p[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_p
            )
            self.output_topk_index[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_index
            )
            self.output_hidden_states[req.metadata_buffer_index].copy_(
                req.hidden_states_tensor
            )
        # Store bootstrap_room for validation on decode side
        self.bootstrap_room[req.metadata_buffer_index, 0] = (
            req.bootstrap_room if req.bootstrap_room is not None else 0
        )


#########################
# Transfer Backend
#########################


class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    MORI = "mori"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"


class KVClassType(Enum):
    KVARGS = "kvargs"
    MANAGER = "manager"
    SENDER = "sender"
    RECEIVER = "receiver"
    BOOTSTRAP_SERVER = "bootstrap_server"


@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.KVARGS]
) -> Type[KVArgs]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.MANAGER]
) -> Type[CommonKVManager]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.SENDER]
) -> Type[CommonKVSender]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.RECEIVER]
) -> Type[CommonKVReceiver]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.BOOTSTRAP_SERVER]
) -> Type[CommonKVBootstrapServer]: ...


def get_kv_class(
    transfer_backend: TransferBackend, class_type: KVClassType
) -> Optional[Type]:
    from sglang.srt.disaggregation.fake import FakeKVReceiver, FakeKVSender

    if transfer_backend == TransferBackend.MOONCAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mooncake import (
            MooncakeKVBootstrapServer,
            MooncakeKVManager,
            MooncakeKVReceiver,
            MooncakeKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: MooncakeKVManager,
            KVClassType.SENDER: MooncakeKVSender,
            KVClassType.RECEIVER: (MooncakeKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.MORI:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mori import (
            MoriKVBootstrapServer,
            MoriKVManager,
            MoriKVReceiver,
            MoriKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: MoriKVManager,
            KVClassType.SENDER: MoriKVSender,
            KVClassType.RECEIVER: (MoriKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MoriKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.ASCEND:
        from sglang.srt.disaggregation.ascend import (
            AscendKVBootstrapServer,
            AscendKVManager,
            AscendKVReceiver,
            AscendKVSender,
        )
        from sglang.srt.disaggregation.base import KVArgs

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: AscendKVManager,
            KVClassType.SENDER: AscendKVSender,
            KVClassType.RECEIVER: (AscendKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: AscendKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.NIXL:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.nixl import (
            NixlKVBootstrapServer,
            NixlKVManager,
            NixlKVReceiver,
            NixlKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: NixlKVManager,
            KVClassType.SENDER: NixlKVSender,
            KVClassType.RECEIVER: (NixlKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: NixlKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.FAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.fake import (
            FakeKVManager,
            FakeKVReceiver,
            FakeKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: FakeKVManager,
            KVClassType.SENDER: FakeKVSender,
            KVClassType.RECEIVER: (FakeKVReceiver),
        }
        return class_mapping.get(class_type)

    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


def page_indices_to_cp_rank_page_indices(
    page_indices: np.ndarray,
    total_pages: int,
    cp_rank: int,
    cp_size: int,
) -> np.ndarray:
    """
    Filter page_indices (which are *global* page ids in the KV pool) to those
    belonging to the given CP rank for this request.

    For a single request, its pages occupy a contiguous global range
    [first_page, first_page + total_pages). We first compute the local
    split [0, total_pages) across cp_size ranks, then shift that local
    range by first_page back into the global page id space and take
    the intersection with page_indices.

    Returns:
        Subset of page_indices that fall in this rank's global
        [start_page, end_page) slice for the given CP rank.
    """
    if cp_size <= 1:
        return page_indices

    if page_indices.size == 0:
        return np.asarray(page_indices)

    first_page = int(page_indices.min())
    base = total_pages // cp_size
    rem = total_pages % cp_size

    if rem == 0:
        local_start = cp_rank * base
        local_end = local_start + base
    else:
        local_start = cp_rank * base + min(cp_rank, rem)
        n_pages = base + (1 if cp_rank < rem else 0)
        local_end = local_start + n_pages

    # Map back to global page ids.
    start_page = first_page + local_start
    end_page = first_page + local_end

    mask = (page_indices >= start_page) & (page_indices < end_page)
    return np.asarray(page_indices)[mask]


def filter_kv_indices_for_cp_rank(
    kv_mgr: CommonKVManager, kv_indices: np.ndarray, index_slice: slice
) -> Tuple[np.ndarray, slice]:
    """Filters kv_indices and index_slice for the current CP rank."""
    total_pages = len(kv_indices)
    cp_rank = kv_mgr.attn_cp_rank
    cp_size = kv_mgr.attn_cp_size

    rank_page_indices = page_indices_to_cp_rank_page_indices(
        page_indices=kv_indices,
        total_pages=total_pages,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )

    if rank_page_indices.size == 0:
        new_kv_indices = kv_indices[:0]
        new_index_slice = slice(index_slice.start, index_slice.start)
    else:
        mask = np.isin(kv_indices, rank_page_indices)
        if not mask.any():
            new_kv_indices = kv_indices[:0]
            new_index_slice = slice(index_slice.start, index_slice.start)
        else:
            first_pos = int(mask.argmax())
            last_pos = len(mask) - int(mask[::-1].argmax())

            new_kv_indices = kv_indices[first_pos:last_pos]
            new_index_slice = slice(
                index_slice.start + first_pos,
                index_slice.start + last_pos,
            )
    return new_kv_indices, new_index_slice


#########################
# Misc
#########################


def is_mla_backend(target_kv_pool) -> bool:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return isinstance(target_kv_pool, (MLATokenToKVPool, DeepSeekV4TokenToKVPool))


def build_transfer_entry_pairs(
    src_layer_ids: List[int],
    dst_layer_ids: List[int],
    n_src: int,
    n_dst: int,
    allow_positional_fallback: bool = False,
) -> List[Tuple[int, int]]:
    """Pair PP-local transfer entries with decode entries by stable layer id."""
    if n_src == 0:
        return []
    if bool(src_layer_ids) != bool(dst_layer_ids):
        if not allow_positional_fallback:
            raise RuntimeError(
                "Layer metadata must be provided by both PD peers or neither"
            )
        src_layer_ids = []
        dst_layer_ids = []
    if src_layer_ids:
        if len(src_layer_ids) != n_src or len(dst_layer_ids) != n_dst:
            raise RuntimeError(
                "Layer metadata length must match transfer entries: "
                f"src metadata={len(src_layer_ids)} entries={n_src}, "
                f"dst metadata={len(dst_layer_ids)} entries={n_dst}"
            )
        dst_positions = {}
        for dst_index, layer_id in enumerate(dst_layer_ids):
            dst_positions.setdefault(layer_id, deque()).append(dst_index)
        pairs = []
        for src_index, layer_id in enumerate(src_layer_ids):
            if not dst_positions.get(layer_id):
                raise RuntimeError(
                    "Decode peer is missing a transfer entry for model layer "
                    f"{layer_id}"
                )
            pairs.append((src_index, dst_positions[layer_id].popleft()))
        return pairs
    if n_dst < n_src or (n_src != n_dst and not allow_positional_fallback):
        raise RuntimeError(
            "PP-heterogeneous transfer requires layer ids on both peers; "
            f"got src={n_src} dst={n_dst} entries"
        )
    return [(index, index) for index in range(n_src)]


_DRAFT_KV_LAYER_ID_BASE = 1_000_000


def get_transfer_kv_layer_ids(kv_pool, num_entries: int) -> List[int]:
    """Return global layer ids aligned with get_contiguous_buf_infos()."""
    if kv_pool is None or num_entries <= 0:
        return []
    if hasattr(kv_pool, "get_kv_layer_ids"):
        layer_ids = list(kv_pool.get_kv_layer_ids())
        if len(layer_ids) == num_entries:
            return layer_ids
    start_layer = int(getattr(kv_pool, "start_layer", 0) or 0)
    end_layer = getattr(kv_pool, "end_layer", None)
    if end_layer is not None:
        layer_ids = list(range(start_layer, int(end_layer)))
        if len(layer_ids) == num_entries:
            return layer_ids
        if len(layer_ids) * 2 == num_entries:
            return layer_ids * 2
    return []


def get_transfer_draft_kv_layer_ids(num_entries: int) -> List[int]:
    if num_entries <= 0:
        return []
    return [_DRAFT_KV_LAYER_ID_BASE + index for index in range(num_entries)]


def pack_state_types(state_types) -> bytes:
    return ",".join(
        state_type.value if hasattr(state_type, "value") else str(state_type)
        for state_type in (state_types or [])
    ).encode("ascii")


def unpack_state_types(data: bytes):
    from sglang.srt.disaggregation.base.conn import StateType

    if not data:
        return []
    return [StateType(value) for value in data.decode("ascii").split(",") if value]


def resolve_state_component_dst_index(
    src_state_types,
    dst_state_types,
    src_index: int,
    *,
    require_metadata: bool = False,
) -> int:
    """Match state components by ``(StateType, occurrence)``.

    Registrations from older peers omit state types and retain positional
    behavior for wire compatibility.
    """
    if not dst_state_types:
        if require_metadata:
            raise RuntimeError(
                "Destination state_types metadata is required for this transfer."
            )
        return src_index
    if not src_state_types:
        raise RuntimeError(
            "Destination state_types are present but source state_types are empty."
        )
    if src_index >= len(src_state_types):
        raise RuntimeError(
            f"Source state component index {src_index} exceeds "
            f"state_types length {len(src_state_types)}."
        )
    state_type = src_state_types[src_index]
    occurrence = sum(
        item == state_type for item in src_state_types[: src_index + 1]
    )
    seen = 0
    for dst_index, dst_state_type in enumerate(dst_state_types):
        if dst_state_type == state_type:
            seen += 1
            if seen == occurrence:
                return dst_index
    raise RuntimeError(
        f"Decode peer is missing state component {state_type!s} "
        f"occurrence {occurrence}."
    )


def append_state_component(
    kv_args: KVArgs,
    state_type: StateType,
    data_ptrs: List[int],
    data_lens: List[int],
    item_lens: List[int],
    dim_per_tensor: Optional[List[int]] = None,
    layer_ids: Optional[List[int]] = None,
) -> None:
    """Append one state component. Caller orders state_types consistently
    on prefill and decode sides."""
    kv_args.state_types.append(state_type)
    kv_args.state_data_ptrs.append(data_ptrs)
    kv_args.state_data_lens.append(data_lens)
    kv_args.state_item_lens.append(item_lens)
    kv_args.state_dim_per_tensor.append(dim_per_tensor or [])
    kv_args.state_layer_ids.append(layer_ids or [])


def setup_state_kv_args(
    kv_args: KVArgs,
    token_to_kv_pool,
    draft_token_to_kv_pool=None,
    total_kv_layers: int = None,
    req_to_token_pool=None,
) -> None:
    """Populate ``kv_args`` state-buffer fields from the given pool.
    Shared by prefill and decode bootstrap paths so the state_type dispatch
    lives in one place.
    """
    from sglang.srt.disaggregation.base.conn import StateType
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, NSATokenToKVPool

    kv_args.state_types = []
    kv_args.state_data_ptrs = []
    kv_args.state_data_lens = []
    kv_args.state_item_lens = []
    kv_args.state_dim_per_tensor = []
    kv_args.state_layer_ids = []

    from sglang.srt.server_args import get_global_server_args

    is_dsv4_dspark = (
        get_global_server_args().speculative_algorithm == "DSPARK"
        and not is_npu()
        and isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
    )

    if hasattr(token_to_kv_pool, "get_state_buf_infos"):
        if is_dsv4_dspark:
            data_ptrs, data_lens, item_lens = (
                token_to_kv_pool.get_dspark_pd_state_buf_infos()
            )
        else:
            data_ptrs, data_lens, item_lens = token_to_kv_pool.get_state_buf_infos()

        # DeepSeekV4TokenToKVPool inherits BaseSWAKVPool; its heterogeneous
        # state list is described per-entry via get_state_buf_infos.
        if isinstance(token_to_kv_pool, BaseSWAKVPool):
            layer_ids = (
                token_to_kv_pool.get_dspark_pd_state_layer_ids()
                if is_dsv4_dspark
                else None
            )
            append_state_component(
                kv_args,
                StateType.SWA,
                data_ptrs,
                data_lens,
                item_lens,
                layer_ids=layer_ids,
            )
            if is_dsv4_dspark:
                c128_ptrs, c128_lens, c128_item_lens = (
                    token_to_kv_pool.get_c128_state_buf_infos()
                )
                if c128_ptrs:
                    append_state_component(
                        kv_args,
                        StateType.C128_STATE,
                        c128_ptrs,
                        c128_lens,
                        c128_item_lens,
                        layer_ids=token_to_kv_pool.get_c128_state_layer_ids(),
                    )
        elif isinstance(token_to_kv_pool, HybridLinearKVPool):
            dim = (
                token_to_kv_pool.get_state_dim_per_tensor()
                if hasattr(token_to_kv_pool, "get_state_dim_per_tensor")
                else None
            )
            append_state_component(
                kv_args, StateType.MAMBA, data_ptrs, data_lens, item_lens, dim
            )
        elif isinstance(token_to_kv_pool, (NSATokenToKVPool, NPUMLATokenToKVPool)):
            if draft_token_to_kv_pool is not None and isinstance(
                draft_token_to_kv_pool, NSATokenToKVPool
            ):
                (
                    draft_data_ptrs,
                    draft_data_lens,
                    draft_item_lens,
                ) = draft_token_to_kv_pool.get_state_buf_infos()
                data_ptrs = data_ptrs + draft_data_ptrs
                data_lens = data_lens + draft_data_lens
                item_lens = item_lens + draft_item_lens
            if isinstance(token_to_kv_pool, NPUMLATokenToKVPool):
                kv_args.kv_buf_groups = (
                    len(kv_args.kv_data_ptrs) // token_to_kv_pool.layer_num
                )
                kv_args.total_kv_layers = total_kv_layers
            else:
                append_state_component(
                    kv_args, StateType.NSA, data_ptrs, data_lens, item_lens
                )

    # Bundled DSV4 DSpark stores draft KV in a SWA-only DSV4 pool that shares
    # the target allocator and Full -> SWA mapping. Keep it as a second SWA
    # component so heterogeneous target state cannot shift the draft payload.
    if is_dsv4_dspark and draft_token_to_kv_pool is not None:
        if not isinstance(draft_token_to_kv_pool, DeepSeekV4TokenToKVPool):
            raise RuntimeError(
                "DSV4 DSpark draft state transfer requires a "
                "DeepSeekV4TokenToKVPool draft pool"
            )
        if not draft_token_to_kv_pool.compression_ratios or not all(
            ratio == 0 for ratio in draft_token_to_kv_pool.compression_ratios
        ):
            raise RuntimeError(
                "DSV4 DSpark draft state transfer expects SWA-only draft layers"
            )
        if (
            token_to_kv_pool.full_to_swa_index_mapping
            is not draft_token_to_kv_pool.full_to_swa_index_mapping
        ):
            raise RuntimeError(
                "DSV4 target and DSpark draft pools must share the SWA index mapping"
            )
        target_geometry = (
            token_to_kv_pool.page_size,
            token_to_kv_pool.swa_page_size,
            token_to_kv_pool.swa_window_size,
        )
        draft_geometry = (
            draft_token_to_kv_pool.page_size,
            draft_token_to_kv_pool.swa_page_size,
            draft_token_to_kv_pool.swa_window_size,
        )
        if target_geometry != draft_geometry:
            raise RuntimeError(
                "DSV4 target and DSpark draft pools must share paged SWA "
                f"geometry: target={target_geometry}, draft={draft_geometry}"
            )
        draft_ptrs, draft_lens, draft_item_lens = (
            draft_token_to_kv_pool.get_state_buf_infos()
        )
        if draft_ptrs:
            append_state_component(
                kv_args,
                StateType.SWA,
                draft_ptrs,
                draft_lens,
                draft_item_lens,
                layer_ids=get_transfer_draft_kv_layer_ids(len(draft_ptrs)),
            )

    if (
        StateType.MAMBA not in kv_args.state_types
        and req_to_token_pool is not None
        and hasattr(req_to_token_pool, "get_state_buf_infos")
    ):
        data_ptrs, data_lens, item_lens = req_to_token_pool.get_state_buf_infos()
        if data_ptrs:
            dim = (
                req_to_token_pool.get_state_dim_per_tensor()
                if hasattr(req_to_token_pool, "get_state_dim_per_tensor")
                else None
            )
            append_state_component(
                kv_args, StateType.MAMBA, data_ptrs, data_lens, item_lens, dim
            )


def prepare_abort(req: Req, error_message: str, status_code=None):
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    # populate finish metadata and stream output
    req.finished_reason = FINISH_ABORT(error_message, status_code)

    if req.return_logprob:
        req.logprob.input_token_logprobs_val = []
        req.logprob.input_token_logprobs_idx = []
        req.logprob.input_top_logprobs_val = []
        req.logprob.input_top_logprobs_idx = []
        req.logprob.input_token_ids_logprobs_val = []
        req.logprob.input_token_ids_logprobs_idx = []
