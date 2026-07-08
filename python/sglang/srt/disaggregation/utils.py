from __future__ import annotations

import os
import random
import json
from collections import deque
from contextlib import nullcontext
from enum import Enum
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Type, overload

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip, is_npu

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
_IS_HIP = is_hip()
DSV4_HISPARSE_PROTOCOL_VERSION = 3


def is_dsv4_c128_online_enabled() -> bool:
    """Return whether DSV4 C128 uses request-scoped online state."""
    return not _IS_HIP and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()


def build_dsv4_hisparse_capability_signature(
    server_args,
    model_config,
    token_to_kv_pool=None,
    hisparse_coordinator=None,
) -> Optional[str]:
    """Stable P/D signature for DSV4 HiSparse capability compatibility.

    The signature is emitted by DSV4 prefill workers and DSV4 HiSparse decode
    workers. It captures configuration that changes the meaning of transferred
    C4/C128 state, not ephemeral pointers or pool sizes. Decode-only HiSparse
    fields are included only when the local worker actually enables HiSparse.
    """
    from sglang.srt.configs.model_config import is_deepseek_v4
    from sglang.srt.mem_cache.sparsity import (
        parse_hisparse_config,
        resolve_hisparse_top_k,
    )

    hf_config = getattr(model_config, "hf_config", model_config)
    if not is_deepseek_v4(hf_config):
        return None

    disaggregation_mode = getattr(server_args, "disaggregation_mode", None)
    is_prefill = disaggregation_mode == "prefill" or (
        getattr(disaggregation_mode, "value", None) == "prefill"
    )
    enable_hisparse = bool(getattr(server_args, "enable_hisparse", False))
    if not enable_hisparse and not is_prefill:
        return None

    def hisparse_config_value(config, key: str):
        if hasattr(config, key):
            return getattr(config, key)
        get = getattr(config, "get", None)
        if get is not None:
            return get(key)
        raise AttributeError(f"HiSparse config does not expose {key!r}")

    payload = {
        "dsv4_hisparse_protocol_version": DSV4_HISPARSE_PROTOCOL_VERSION,
        "compressor_v2": bool(envs.SGLANG_OPT_USE_COMPRESSOR_V2.get()),
        "deepgemm_hc_prenorm": bool(envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()),
        "dsv4_compress_state_dtype": envs.SGLANG_DSV4_COMPRESS_STATE_DTYPE.get()
        .strip()
        .lower(),
        "dsv4_fp4_experts": bool(
            getattr(model_config, "is_fp4_experts", envs.SGLANG_DSV4_FP4_EXPERTS.get())
        ),
        "experimental_online_c128_mtp": bool(
            envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get()
        ),
        "online_c128": bool(is_dsv4_c128_online_enabled()),
        "tilelang_mhc_post": bool(envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get()),
        "tilelang_mhc_pre": bool(envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get()),
        "tilelang_mhc_split_sinkhorn": bool(
            envs.SGLANG_OPT_USE_TILELANG_MHC_SPLIT_SINKHORN.get()
        ),
    }
    if enable_hisparse:
        config = parse_hisparse_config(server_args)
        text_config = getattr(hf_config, "text_config", hf_config)
        top_k = int(resolve_hisparse_top_k(server_args, text_config))
        device_buffer_size = int(hisparse_config_value(config, "device_buffer_size"))
        min_device_buffer_size = int(
            hisparse_config_value(config, "min_device_buffer_size")
        )
        topk_mode = (
            "legacy_raw_index"
            if top_k in (512, 1024)
            else "flexible_topk_v2_raw_index"
        )

        c4_host_mirror = True
        if hisparse_coordinator is not None:
            c4_host_mirror = (
                bool(getattr(hisparse_coordinator, "is_dsv4_hisparse", False))
                and getattr(hisparse_coordinator, "host_radix_cache", None) is None
            )
        payload.update(
            {
                "c4_host_mirror": bool(c4_host_mirror),
                "device_buffer_size": device_buffer_size,
                "min_device_buffer_size": min_device_buffer_size,
                "top_k": top_k,
                "topk_mode": topk_mode,
                "use_topk_v2": bool(envs.SGLANG_OPT_USE_TOPK_V2.get()),
            }
        )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def get_dsv4_c128_state_indices(
    req_pool_idx: int,
    seq_len: int,
    *,
    online: bool,
    ring_size: int,
) -> np.ndarray:
    """Return PD transfer page indices for request-scoped DSV4 C128 state."""
    if seq_len == 0 or seq_len % 128 == 0:
        return np.empty((0,), dtype=np.int32)
    if online:
        return np.array([int(req_pool_idx)], dtype=np.int32)

    assert ring_size % 128 == 0, f"C128 ring_size must be 128-aligned, got {ring_size}"
    pages_per_req = ring_size // 128
    page = int(req_pool_idx) * pages_per_req + ((seq_len - 1) % ring_size) // 128
    return np.array([page], dtype=np.int32)


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


def append_state_component(
    kv_args: KVArgs,
    state_type: StateType,
    data_ptrs: List[int],
    data_lens: List[int],
    item_lens: List[int],
    dim_per_tensor: Optional[List[int]] = None,
) -> None:
    """Append one state component. Caller orders state_types consistently
    on prefill and decode sides."""
    kv_args.state_types.append(state_type)
    kv_args.state_data_ptrs.append(data_ptrs)
    kv_args.state_data_lens.append(data_lens)
    kv_args.state_item_lens.append(item_lens)
    kv_args.state_dim_per_tensor.append(dim_per_tensor or [])


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

    if hasattr(token_to_kv_pool, "get_state_buf_infos"):
        data_ptrs, data_lens, item_lens = token_to_kv_pool.get_state_buf_infos()

        # DeepSeekV4TokenToKVPool inherits BaseSWAKVPool; its heterogeneous
        # state list is described per-entry via get_state_buf_infos.
        if isinstance(token_to_kv_pool, BaseSWAKVPool):
            append_state_component(
                kv_args, StateType.SWA, data_ptrs, data_lens, item_lens
            )
            if hasattr(token_to_kv_pool, "get_c128_state_buf_infos"):
                (
                    c128_data_ptrs,
                    c128_data_lens,
                    c128_item_lens,
                ) = token_to_kv_pool.get_c128_state_buf_infos()
                if c128_data_ptrs:
                    append_state_component(
                        kv_args,
                        StateType.C128_STATE,
                        c128_data_ptrs,
                        c128_data_lens,
                        c128_item_lens,
                    )
            if (
                envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
                and StateType.C128_STATE not in kv_args.state_types
            ):
                raise RuntimeError(
                    "DSV4 online C128 compression requires P/D transfer of "
                    "StateType.C128_STATE, but setup_state_kv_args did not "
                    "register a C128 state component. This would let decode "
                    "reuse a radix prefix without the request-scoped C128 state."
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

    # DSV4 NextN shares the target allocator, so target and draft use the same
    # local SWA indices. Keep draft buffers in a separate positional component
    # to avoid mixing them into the target's heterogeneous state layout, while
    # reusing the existing SWA transport dispatch. NPU has a different paged
    # state layout and is intentionally left unchanged.
    if (
        not is_npu()
        and isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        and isinstance(draft_token_to_kv_pool, DeepSeekV4TokenToKVPool)
    ):
        if not draft_token_to_kv_pool.compression_ratios or not all(
            ratio == 0 for ratio in draft_token_to_kv_pool.compression_ratios
        ):
            raise RuntimeError("DSV4 draft state transfer expects SWA-only NextN layers")
        target_unified_kv = bool(getattr(token_to_kv_pool, "_unified_kv", False))
        draft_unified_kv = bool(
            getattr(draft_token_to_kv_pool, "_unified_kv", False)
        )
        if target_unified_kv != draft_unified_kv:
            raise RuntimeError(
                "DSV4 target and draft pools must use the same unified-KV mode"
            )

        if target_unified_kv:
            target_geometry = (
                token_to_kv_pool.unified_swa_window,
                token_to_kv_pool.unified_swa_ring_size,
                token_to_kv_pool.unified_swa_pages,
            )
            draft_geometry = (
                draft_token_to_kv_pool.unified_swa_window,
                draft_token_to_kv_pool.unified_swa_ring_size,
                draft_token_to_kv_pool.unified_swa_pages,
            )
            if target_geometry != draft_geometry:
                raise RuntimeError(
                    "DSV4 target and draft pools must share SWA ring geometry: "
                    f"target={target_geometry}, draft={draft_geometry}"
                )
            draft_ptrs, draft_lens, draft_item_lens = (
                draft_token_to_kv_pool.get_unified_swa_ring_buf_infos()
            )
            draft_state_type = StateType.SWA_RING
        else:
            if (
                token_to_kv_pool.full_to_swa_index_mapping
                is not draft_token_to_kv_pool.full_to_swa_index_mapping
            ):
                raise RuntimeError(
                    "DSV4 target and draft pools must share the SWA index mapping"
                )
            target_geometry = (
                token_to_kv_pool.page_size,
                getattr(
                    token_to_kv_pool,
                    "swa_window_size",
                    getattr(token_to_kv_pool, "sliding_window", None),
                ),
            )
            draft_geometry = (
                draft_token_to_kv_pool.page_size,
                getattr(
                    draft_token_to_kv_pool,
                    "swa_window_size",
                    getattr(draft_token_to_kv_pool, "sliding_window", None),
                ),
            )
            if target_geometry != draft_geometry:
                raise RuntimeError(
                    "DSV4 target and draft pools must share paged SWA geometry: "
                    f"target={target_geometry}, draft={draft_geometry}"
                )
            draft_ptrs, draft_lens, draft_item_lens = (
                draft_token_to_kv_pool.get_state_buf_infos()
            )
            draft_state_type = StateType.SWA

        if draft_ptrs:
            append_state_component(
                kv_args,
                draft_state_type,
                draft_ptrs,
                draft_lens,
                draft_item_lens,
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
