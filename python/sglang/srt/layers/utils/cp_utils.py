from dataclasses import dataclass
from itertools import accumulate
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.dp_attention import get_attention_cp_group, get_attention_cp_size, get_attention_cp_rank
from sglang.srt.server_args import get_global_server_args


@dataclass
class ContextParallelMetadata:
    split_list: List[int] = None
    max_rank_len: List[int] = None
    zigzag_index: List[int] = None
    per_rank_actual_token: List[int] = None
    reverse_split_len: List[int] = None
    cp_reverse_index: List[int] = None
    rebuild_index_tensor: torch.Tensor = None
    restore_index_tensor: torch.Tensor = None
    gathered_index_tensor: torch.Tensor = None

    # metadata for attention
    kv_len_prev: int = -1
    kv_len_next: int = -1
    actual_seq_q_prev: int = -1
    actual_seq_q_next: int = -1
    kv_len_prev_tensor: torch.Tensor = None
    kv_len_next_tensor: torch.Tensor = None
    actual_seq_q_prev_tensor: torch.Tensor = None
    actual_seq_q_next_tensor: torch.Tensor = None
    cu_seqlens_q_prev_tensor: torch.Tensor = None
    cu_seqlens_q_next_tensor: torch.Tensor = None

    total_seq_lens: torch.Tensor = None


def is_prefill_context_parallel_enabled():
    """Generic CP gating with NSA compatibility fallback."""
    args = get_global_server_args()
    return bool(
        getattr(args, "enable_prefill_context_parallel", False)
        or getattr(args, "enable_nsa_prefill_context_parallel", False)
    )


def get_prefill_cp_mode() -> str:
    """Return the CP split mode with backward compatibility to NSA flags."""
    args = get_global_server_args()
    mode = getattr(args, "prefill_cp_mode", None)
    if mode:
        return mode
    # fallback to NSA mode if generic mode not set
    return getattr(args, "nsa_prefill_cp_mode", "in-seq-split")


def is_prefill_cp_in_seq_split():
    return (
        is_prefill_context_parallel_enabled()
        and get_prefill_cp_mode() == "in-seq-split"
    )


def is_prefill_cp_round_robin_split():
    return (
        is_prefill_context_parallel_enabled()
        and get_prefill_cp_mode() == "round-robin-split"
    )


def can_prefill_cp_round_robin_split(forward_batch) -> bool:
    """Token-level round-robin split feasibility check."""
    if not forward_batch.forward_mode.is_context_parallel_extend():
        return False
    cp_size = get_attention_cp_size()
    seq_len = sum(forward_batch.extend_seq_lens_cpu) if forward_batch.extend_seq_lens_cpu is not None else 0
    return (
        is_prefill_cp_round_robin_split()
        and seq_len > 0
        and seq_len >= cp_size
        and cp_size > 1
    )


def cp_use_prefill(forward_batch) -> bool:
    return (
        forward_batch.attn_cp_metadata is not None
        and is_prefill_context_parallel_enabled()
        and forward_batch.forward_mode.is_context_parallel_extend()
    )


def can_cp_split(seq_len: int, cp_size: int, forward_batch):
    # TODO current just support prefill batch=1 and len(input_ids) > self.cp_size * 2
    # Note: (self.cp_size * 2) To achieve load balancing for seq computation,
    # the seq data needs to be divided and recombined at twice the size of cp_size.
    cur_cp_seq_len = seq_len // (cp_size * 2)
    # print("DEBUG: can_cp_split", cur_cp_seq_len, cp_size, forward_batch.forward_mode.is_context_parallel_extend(), is_prefill_context_parallel_enabled(), flush=True)
    if (
        cur_cp_seq_len != 0
        and cp_size > 1
        and forward_batch.forward_mode.is_context_parallel_extend()
        and is_prefill_context_parallel_enabled()
    ):
        return True
    else:
        return False


def cp_split_and_rebuild_data(forward_batch, input_: torch.Tensor):
    if is_prefill_cp_round_robin_split():
        return cp_round_robin_split_data(input_)
    return input_.index_select(0, forward_batch.attn_cp_metadata.rebuild_index_tensor)


def cp_split_and_rebuild_position(forward_batch, positions: torch.Tensor):
    if is_prefill_cp_round_robin_split():
        return cp_round_robin_split_data(positions)
    return positions.index_select(
        positions.dim() - 1, forward_batch.attn_cp_metadata.rebuild_index_tensor
    )


def cp_all_gather_reorganized_into_tensor(
    input_tensor, total_len, cp_size, forward_batch, stream
):
    """
    Allgather communication for context_parallel hidden_states.
    """
    # The input tensor should already be padded to the same length for allgather communication.
    # No need to pad again.
    # step1
    max_len = (total_len + cp_size - 1) // cp_size
    pad_size = max_len - input_tensor.shape[0]
    if pad_size > 0:
        input_tensor = F.pad(
            input_tensor, (0, 0, 0, pad_size), mode="constant", value=0
        )
    input_tensor_full = torch.empty(
        max_len * cp_size,
        input_tensor.shape[1],
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )

    get_attention_cp_group().cp_all_gather_into_tensor_async(
        input_tensor_full, input_tensor, stream
    )
    return input_tensor_full.index_select(
        0, forward_batch.attn_cp_metadata.gathered_index_tensor
    )


def cp_all_gather_reorganized_into_tensor_kv_cache(
    input_tensor, total_len, cp_size, forward_batch, stream
):
    """
    Allgather communication for context_parallel KV cache.
    Handles multi-dimensional tensors (e.g., [seq_len, num_heads, head_dim]).
    """
    max_len = (total_len + cp_size - 1) // cp_size
    pad_size = max_len - input_tensor.shape[0]
    if pad_size > 0:
        # Pad the first dimension (seq_len). F.pad expects padding in reverse dimension order.
        # For n dimensional tensor, we need 2*n values: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
        # To pad only the first dimension: [0, 0] * (ndim - 1) + [0, pad_size]
        padding = [0, 0] * (input_tensor.ndim - 1) + [0, pad_size]
        input_tensor = F.pad(input_tensor, padding, mode="constant", value=0)

    # Create output tensor with proper shape for all dimensions
    input_tensor_full = torch.empty(
        max_len * cp_size,
        *input_tensor.shape[1:],
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )

    get_attention_cp_group().cp_all_gather_into_tensor_async(
        input_tensor_full, input_tensor, stream
    )
    return input_tensor_full.index_select(
        0, forward_batch.attn_cp_metadata.gathered_index_tensor
    )


def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):
    """
    # for in-seq-split
    |   +-----------before allgather------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------before rerange---------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------result-------------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+
    """

    if is_prefill_cp_round_robin_split():
        # equal-width token interleave/disinterleave across cp ranks
        # shape-preserving: gather then transpose/rearrange to original token order
        output_tensor = input_tensor.new_empty(
            (input_tensor.shape[0] * cp_size, *input_tensor.shape[1:])
        )
        get_attention_cp_group().cp_all_gather_into_tensor_async(output_tensor, input_tensor, stream)
        out_shape = output_tensor.shape
        output_tensor = output_tensor.view(cp_size, -1, *out_shape[1:]).transpose(0, 1).reshape(out_shape)
        return output_tensor
    # in-seq-split path (zigzag)
    hidden_size = input_tensor.shape[1] if input_tensor.dim() == 2 else input_tensor.shape[-1]
    output_tensor = cp_all_gather_reorganized_into_tensor(
        input_tensor, forward_batch.attn_cp_metadata.total_seq_lens, cp_size, forward_batch, stream
    )
    output_tensor = output_tensor.index_select(
        0, forward_batch.attn_cp_metadata.restore_index_tensor
    )
    output_tensor = output_tensor.view(-1, hidden_size)
    return output_tensor


def cp_all_gather_rerange_kv_cache(input_tensor, cp_size, forward_batch, stream):
    """
    Allgather and reorganize KV cache from all ranks in context parallel group.

    # for in-seq-split
    |   +-----------before allgather------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------before rerange---------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------result-------------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+
    """
    if is_prefill_cp_round_robin_split():
        output_tensor = input_tensor.new_empty(
            (input_tensor.shape[0] * cp_size, *input_tensor.shape[1:])
        )
        get_attention_cp_group().cp_all_gather_into_tensor_async(
            output_tensor, input_tensor, stream
        )
        out_shape = output_tensor.shape
        return output_tensor.view(cp_size, -1, *out_shape[1:]).transpose(0, 1).reshape(
            out_shape
        )
    output_tensor = cp_all_gather_reorganized_into_tensor_kv_cache(
        input_tensor,
        forward_batch.attn_cp_metadata.total_seq_lens,
        cp_size,
        forward_batch,
        stream,
    )
    output_tensor = output_tensor.index_select(
        0, forward_batch.attn_cp_metadata.restore_index_tensor
    )
    # No need to reshape - output_tensor already has the correct shape [seq_len, ...]
    return output_tensor


def cp_allgather_and_save_kv_cache(forward_batch, layer, k, v, cp_size):
    """
    Allgather KV cache from all CP ranks and write the full result
    into each rank's local memory pool.
    """
    cache_loc = (
        forward_batch.out_cache_loc
        if not layer.is_cross_attention
        else forward_batch.encoder_out_cache_loc
    )

    k = k.contiguous()
    v = v.contiguous()

    key_cache_full = cp_all_gather_rerange_kv_cache(
        k, cp_size, forward_batch, torch.cuda.current_stream()
    )
    value_cache_full = cp_all_gather_rerange_kv_cache(
        v, cp_size, forward_batch, torch.cuda.current_stream()
    )

    forward_batch.token_to_kv_pool.set_kv_buffer(
        layer,
        cache_loc,
        key_cache_full,
        value_cache_full,
        layer.k_scale,
        layer.v_scale,
    )


def cp_allgather_and_save_mla_kv_cache(forward_batch, layer, k_nope, k_rope, cp_size):
    """
    Allgather MLA KV (k_nope, k_rope) from all CP ranks and write the full result
    into each rank's local memory pool via set_mla_kv_buffer.
    """
    cache_loc = (
        forward_batch.out_cache_loc
        if not layer.is_cross_attention
        else forward_batch.encoder_out_cache_loc
    )
    k_nope = k_nope.contiguous()
    k_rope = k_rope.contiguous()

    k_nope_full = cp_all_gather_rerange_kv_cache(
        k_nope, cp_size, forward_batch, torch.cuda.current_stream()
    )
    k_rope_full = cp_all_gather_rerange_kv_cache(
        k_rope, cp_size, forward_batch, torch.cuda.current_stream()
    )

    forward_batch.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
        layer,
        cache_loc,
        k_nope_full,
        k_rope_full,
    )


def cp_attn_forward_extend(
    forward_batch,
    q: torch.Tensor,
    device: torch.device,
    attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """
    Split q into prev/next zigzag halves based on CP metadata, call the
    backend-specific attention function twice with appropriate per-half
    metadata, and concatenate the results.

    attn_fn signature:
        attn_fn(q, cu_seqlens_q, cache_seqlens, max_seqlen_q) -> result
    where only these four CP-varying parameters differ between halves.
    All other backend-specific args should be captured in the closure.
    """
    cp_meta = forward_batch.attn_cp_metadata

    q_prev, q_next = torch.chunk(q, 2, dim=0)

    result_prev = attn_fn(
        q_prev,
        cp_meta.cu_seqlens_q_prev_tensor,
        cp_meta.kv_len_prev_tensor,
        cp_meta.actual_seq_q_prev,
    )

    result_next = attn_fn(
        q_next,
        cp_meta.cu_seqlens_q_next_tensor,
        cp_meta.kv_len_next_tensor,
        cp_meta.actual_seq_q_next,
    )

    return torch.concat([result_prev, result_next], dim=0)


def prepare_context_parallel_metadata(
    kv_len,
    cp_rank,
    cp_size,
    seqs_len,
):
    """prepare_input_dp_with_cp_dsa-zigzag index（in-seq-split）
    Example (DP_ATTENT_TP == CP_SIZE == 4):
    Description:
    1. Start with a full-length request.
    2. Split the request into multiple blocks (block0 to block7).
    3. Rearrange these blocks to balance computational
        load across different DP ranks.
    4. Assign the rearranged blocks to different DP attention
        time points (dp_atten_tp0 to dp_atten_tp3).
    +---------------------------------+
    |        cp_split_tokens         |
    +---------------------------------+
    |                                 |
    |   request_with_full_length     |
    |             | split (cp_size * 2) |
    |   +-------------------------+  |
    |   | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+  |
    |             | rerange          |
    |   +---------------------------------+
    |   | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |   +---------------------------------+
    |             |
    |   +-------------------------+
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |   +-------------------------+

    Why zigzag rearrange?
    - Attention calculations must follow causal attention principles.
    - Simply slicing by rank order can lead to computational load imbalance:
        * First rank may focus on fewer historical key-value tokens (less computation)
        * Last rank may focus on more tokens (more computation)
    - To mitigate uneven load, the input hidden states needs to be sliced by cp_size*2 and rearranged.
    """
    # just support batch = 1
    # kv_len: the number of tokens *computed in this extend pass* (i.e. the
    # "new" tokens). When radix/prefix cache hits, the effective KV length
    # visible to attention is: prefix_len + kv_len. CP attention must use the
    # full visible KV length, otherwise queries won't attend to cached prefix.
    kv_len = torch.tensor(kv_len)
    bs_per_cp_group = 1
    kv_len_origin = kv_len

    # Derive prefix offset from the full sequence length on CPU.
    # NOTE: forward_batch.seq_lens_cpu includes cached prefix + extend tokens.
    # In CP we only split the extend tokens, but cache_seqlens passed to FA must
    # include the cached prefix.
    prefix_len = 0
    try:
        if seqs_len is not None and len(seqs_len) == 1:
            prefix_len = int(seqs_len[0]) - int(kv_len_origin.item())
            if prefix_len < 0:
                prefix_len = 0
    except Exception:
        prefix_len = 0
    # get zigzag index
    cp_segment_num = cp_size * 2
    seq_per_batch = kv_len // cp_segment_num  # seq_len for each batch and segment
    split_list = seq_per_batch.repeat_interleave(cp_segment_num).int().tolist()
    remainder = kv_len % (cp_segment_num)
    if remainder > 0:
        split_list[:remainder] = [x + 1 for x in split_list[:remainder]]

    seq_max_rank_len_tensor = (kv_len + cp_size - 1) // cp_size
    seq_max_rank_len = int(seq_max_rank_len_tensor.item())
    max_rank_len = seq_max_rank_len_tensor.repeat_interleave(cp_size).int().tolist()
    zigzag_index = list(
        range(cp_rank, cp_rank + bs_per_cp_group * cp_segment_num, cp_segment_num)
    ) + list(
        range(
            cp_segment_num - cp_rank - 1,
            bs_per_cp_group * cp_segment_num,
            cp_segment_num,
        )
    )

    per_rank_actual_token = list(
        split_list[i] + split_list[cp_size * 2 - i - 1] for i in range(cp_size)
    )
    reverse_split_len = [
        element
        for i in range(cp_size)
        for element in (split_list[i], split_list[cp_size * 2 - i - 1])
    ]
    # get zigzag reverse index
    cp_reverse_index = []
    for batch_id in range(bs_per_cp_group):
        cp_reverse_index.extend(
            list(range(batch_id, cp_segment_num * bs_per_cp_group, 2 * bs_per_cp_group))
            + list(
                range(
                    (cp_segment_num - 1) * bs_per_cp_group + batch_id,
                    0,
                    -2 * bs_per_cp_group,
                )
            )
        )
    prefix_sum_list = list(accumulate(split_list))

    # TODO Support multi-batch-cp-split, multi-batch-cp support has accuracy issues
    # Prefix offset is critical when radix cache hits (prefix_len > 0).
    # These "cache_seqlens" values represent how many KV tokens are visible to
    # each query segment during CP attention.
    kv_len_prev = prefix_len + prefix_sum_list[cp_rank]
    kv_len_next = prefix_len + prefix_sum_list[cp_size * 2 - cp_rank - 1]
    actual_seq_q_prev = split_list[cp_rank]
    actual_seq_q_next = split_list[cp_size * 2 - cp_rank - 1]
    # Flash Attention expects cache_seqlens to have shape (batch_size,), not scalar
    device = torch.device("cuda")
    kv_len_prev_tensor = torch.tensor([kv_len_prev], device=device, dtype=torch.int32)
    kv_len_next_tensor = torch.tensor([kv_len_next], device=device, dtype=torch.int32)
    actual_seq_q_prev_tensor = torch.tensor(
        [actual_seq_q_prev], device=device, dtype=torch.int32
    )
    actual_seq_q_next_tensor = torch.tensor(
        [actual_seq_q_next], device=device, dtype=torch.int32
    )
    cu_seqlens_q_prev_tensor = torch.tensor(
        [0, actual_seq_q_prev], device=device, dtype=torch.int32
    )
    cu_seqlens_q_next_tensor = torch.tensor(
        [0, actual_seq_q_next], device=device, dtype=torch.int32
    )

    split_offsets = [0] + list(accumulate(split_list))
    rebuild_indices = []
    for split_idx in zigzag_index:
        rebuild_indices.extend(
            range(split_offsets[split_idx], split_offsets[split_idx + 1])
        )
    rebuild_index_tensor = torch.tensor(
        rebuild_indices, device=device, dtype=torch.long
    )
    restore_index_tensor = torch.empty_like(rebuild_index_tensor)
    restore_index_tensor[rebuild_index_tensor] = torch.arange(
        rebuild_index_tensor.numel(), device=device, dtype=torch.long
    )
    gathered_indices = []
    for rank_idx, token_count in enumerate(per_rank_actual_token):
        gathered_indices.extend(range(rank_idx * seq_max_rank_len, rank_idx * seq_max_rank_len + token_count))
    gathered_index_tensor = torch.tensor(
        gathered_indices, device=device, dtype=torch.long
    )

    attn_cp_metadata = ContextParallelMetadata(
        split_list=split_list,
        max_rank_len=max_rank_len,
        zigzag_index=zigzag_index,
        per_rank_actual_token=per_rank_actual_token,
        reverse_split_len=reverse_split_len,
        cp_reverse_index=cp_reverse_index,
        rebuild_index_tensor=rebuild_index_tensor,
        restore_index_tensor=restore_index_tensor,
        gathered_index_tensor=gathered_index_tensor,
        kv_len_prev=kv_len_prev,
        kv_len_next=kv_len_next,
        actual_seq_q_prev=actual_seq_q_prev,
        actual_seq_q_next=actual_seq_q_next,
        kv_len_prev_tensor=kv_len_prev_tensor,
        kv_len_next_tensor=kv_len_next_tensor,
        actual_seq_q_prev_tensor=actual_seq_q_prev_tensor,
        actual_seq_q_next_tensor=actual_seq_q_next_tensor,
        cu_seqlens_q_prev_tensor=cu_seqlens_q_prev_tensor,
        cu_seqlens_q_next_tensor=cu_seqlens_q_next_tensor,
        total_seq_lens=kv_len_origin,
    )
    return attn_cp_metadata


# ========== Round-robin split (generic) ==========
def cp_round_robin_split_data(input_: torch.Tensor):
    """Token-level round-robin split: token_i -> cp_rank = i % cp_size."""
    cp_size = get_attention_cp_size()
    cp_rank = get_attention_cp_rank()
    if input_.shape[0] % cp_size != 0:
        cur_len = input_.shape[0] // cp_size + (input_.shape[0] % cp_size > cp_rank)
        if cur_len == 0:
            return input_.new_empty(0, *input_.shape[1:])
        indices = torch.arange(cp_rank, input_.shape[0], cp_size, device=input_.device)
        return input_[indices]
    return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()


def cp_round_robin_split_q_seqs_cpu(extend_seqs_cpu: List[int]) -> Tuple[List[int], List[int]]:
    """CPU 版 round-robin 分配后的每条序列分段长度与被选中的序列下标（长度>0）。"""
    cp_size = get_attention_cp_size()
    cp_rank = get_attention_cp_rank()
    extra_seq = 0
    q_seqs = []
    for cur_len in extend_seqs_cpu:
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)
        q_seqs.append(cur_seq)
        extra_seq = cur_len - cur_seq * cp_size
    bs_idx = [i for i, x in enumerate(q_seqs) if x > 0]
    q_seqs = [q_len for q_len in q_seqs if q_len > 0]
    return q_seqs, bs_idx


def prepare_round_robin_context_parallel_metadata(kv_len, _cp_rank, _cp_size, _seqs_len):
    """
    Round-robin mode does not need zigzag metadata. We still return a metadata object
    so callers can use a unified `attn_cp_metadata is not None` contract.
    """
    _ = (_cp_rank, _cp_size, _seqs_len)
    return ContextParallelMetadata(total_seq_lens=torch.tensor(kv_len))
