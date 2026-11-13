from __future__ import annotations

import torch
import triton
import triton.language as tl

_QWEN4_MAX_SHORT_CONV_STATE_LEN = 16


@triton.jit
def _qwen4_varlen_conv_kernel(
    x_ptr,
    weight_ptr,
    state_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    req_indices_ptr,
    token_offsets_ptr,
    output_ptr,
    num_tokens,
    CHANNELS: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    DILATION: tl.constexpr,
    STATE_LEN: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    mask = (token < num_tokens) & (channel < CHANNELS)
    req = tl.load(req_indices_ptr + token, mask=token < num_tokens, other=0)
    token_offset = tl.load(token_offsets_ptr + token, mask=token < num_tokens, other=0)
    seq_start = tl.load(query_start_loc_ptr + req, mask=token < num_tokens, other=0)
    state_index = tl.load(state_indices_ptr + req, mask=token < num_tokens, other=0)
    acc = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    for weight_col in tl.static_range(0, KERNEL_SIZE):
        source_col = token_offset - (KERNEL_SIZE - 1 - weight_col) * DILATION
        from_tokens = source_col >= 0
        token_value = tl.load(
            x_ptr + (seq_start + source_col) * CHANNELS + channel,
            mask=mask & from_tokens,
            other=0.0,
        ).to(tl.float32)
        state_value = tl.load(
            state_ptr
            + state_index * CHANNELS * STATE_LEN
            + channel * STATE_LEN
            + STATE_LEN
            + source_col,
            mask=mask & ~from_tokens,
            other=0.0,
        ).to(x_ptr.dtype.element_ty)
        state_value = state_value.to(tl.float32)
        weight = tl.load(
            weight_ptr + channel * KERNEL_SIZE + weight_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += tl.where(from_tokens, token_value, state_value) * weight
    tl.store(output_ptr + token * CHANNELS + channel, acc, mask=mask)


@triton.jit
def _qwen4_varlen_state_writeback_kernel(
    x_ptr,
    state_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    track_indices_ptr,
    track_offsets_ptr,
    CHANNELS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    HAS_TRACK: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_STATE_LEN: tl.constexpr,
):
    req = tl.program_id(0)
    channel_1d = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    channel = channel_1d[:, None]
    state_col = tl.arange(0, BLOCK_STATE_LEN)[None, :]
    channel_mask_1d = channel_1d < CHANNELS
    channel_mask = channel_mask_1d[:, None]
    state_mask = channel_mask & (state_col < STATE_LEN)
    state_index = tl.load(state_indices_ptr + req)
    seq_start = tl.load(query_start_loc_ptr + req)
    length = tl.load(query_start_loc_ptr + req + 1) - seq_start
    state_base = state_index * CHANNELS * STATE_LEN

    # A row can shift by less than STATE_LEN, so every old value must be
    # materialized before this program starts overwriting the in-place slot.
    old_state = tl.load(
        state_ptr + state_base + channel * STATE_LEN + state_col,
        mask=state_mask,
        other=0.0,
    ).to(x_ptr.dtype.element_ty)
    tl.debug_barrier()

    for output_col in tl.static_range(0, STATE_LEN):
        source_col = length - STATE_LEN + output_col
        old_col = STATE_LEN + source_col
        old_value = tl.sum(tl.where(state_col == old_col, old_state, 0.0), axis=1)
        token_value = tl.load(
            x_ptr + (seq_start + source_col) * CHANNELS + channel_1d,
            mask=channel_mask_1d & (source_col >= 0),
            other=0.0,
        )
        value = tl.where(source_col >= 0, token_value, old_value)
        tl.store(
            state_ptr + state_base + channel_1d * STATE_LEN + output_col,
            value,
            mask=channel_mask_1d & (length > 0) & (state_index != 0),
        )

    if HAS_TRACK:
        track_index = tl.load(track_indices_ptr + req)
        track_offset = tl.load(track_offsets_ptr + req)
        track_base = track_index * CHANNELS * STATE_LEN
        for output_col in tl.static_range(0, STATE_LEN):
            source_col = track_offset - STATE_LEN + output_col
            old_col = STATE_LEN + source_col
            old_value = tl.sum(tl.where(state_col == old_col, old_state, 0.0), axis=1)
            token_value = tl.load(
                x_ptr + (seq_start + source_col) * CHANNELS + channel_1d,
                mask=channel_mask_1d & (source_col >= 0),
                other=0.0,
            )
            value = tl.where(source_col >= 0, token_value, old_value)
            tl.store(
                state_ptr + track_base + channel_1d * STATE_LEN + output_col,
                value,
                mask=channel_mask_1d & (length > 0) & (track_index != 0),
            )


def can_fuse_qwen4_varlen_conv(
    x,
    weight,
    state,
    state_indices,
    query_start_loc,
    req_indices,
    token_offsets,
    dilation,
):
    """Return whether ordinary PLE prefill can use the packed CUDA path."""

    requests = state_indices.shape[0] if state_indices.ndim == 1 else -1
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and weight.device == x.device
        and weight.dtype == torch.bfloat16
        and weight.ndim == 3
        and weight.shape[:2] == (x.shape[1], 1)
        and weight.is_contiguous()
        and state.device == x.device
        and state.dtype in (torch.bfloat16, torch.float32)
        and state.ndim == 3
        and state.shape[1] == x.shape[1]
        and 0 < state.shape[2] <= _QWEN4_MAX_SHORT_CONV_STATE_LEN
        and state.is_contiguous()
        and isinstance(dilation, int)
        and dilation > 0
        and state.shape[2] == (weight.shape[2] - 1) * dilation
        and state_indices.device == x.device
        and state_indices.dtype == torch.int64
        and state_indices.ndim == 1
        and state_indices.is_contiguous()
        and query_start_loc.device == x.device
        and query_start_loc.dtype == torch.int64
        and query_start_loc.shape == (requests + 1,)
        and query_start_loc.is_contiguous()
        and req_indices.device == x.device
        and req_indices.dtype == torch.int64
        and req_indices.shape == (x.shape[0],)
        and req_indices.is_contiguous()
        and token_offsets.device == x.device
        and token_offsets.dtype == torch.int64
        and token_offsets.shape == (x.shape[0],)
        and token_offsets.is_contiguous()
    )


def fused_qwen4_varlen_conv(
    x,
    weight,
    state,
    state_indices,
    query_start_loc,
    req_indices,
    token_offsets,
    dilation,
    *,
    track_indices=None,
    track_offsets=None,
):
    """Compute packed PLE convolution and update main/radix state slots.

    This raw API intentionally validates only tensor metadata that is available
    without a device-to-host synchronization. The caller must guarantee that
    ``query_start_loc`` is a nondecreasing prefix sum starting at zero and
    ending at ``x.shape[0]``; every ``req_indices[t]`` is in
    ``[0, state_indices.shape[0])``; and every ``token_offsets[t]`` is in
    ``[0, query_start_loc[req + 1] - query_start_loc[req])`` for
    ``req = req_indices[t]``.

    The caller must also guarantee that nonzero main slots in
    ``state_indices`` are unique, nonzero radix slots in ``track_indices`` are
    unique, and the two sets of nonzero slots are disjoint. Slot zero is
    reserved for dummy or empty rows and is never written. These value-level
    contracts are not checked here because doing so would introduce host
    synchronization.
    """

    if not can_fuse_qwen4_varlen_conv(
        x,
        weight,
        state,
        state_indices,
        query_start_loc,
        req_indices,
        token_offsets,
        dilation,
    ):
        raise ValueError("unsupported input for Qwen4 packed-varlen convolution")
    has_track = track_indices is not None or track_offsets is not None
    if has_track and (
        track_indices is None
        or track_offsets is None
        or track_indices.device != x.device
        or track_offsets.device != x.device
        or track_indices.dtype != torch.int64
        or track_offsets.dtype != torch.int64
        or track_indices.shape != state_indices.shape
        or track_offsets.shape != state_indices.shape
        or not track_indices.is_contiguous()
        or not track_offsets.is_contiguous()
    ):
        raise ValueError("invalid Qwen4 packed-varlen track metadata")

    output = torch.empty_like(x)
    block_channels = 128
    if x.shape[0]:
        _qwen4_varlen_conv_kernel[
            (x.shape[0], triton.cdiv(x.shape[1], block_channels))
        ](
            x,
            weight,
            state,
            state_indices,
            query_start_loc,
            req_indices,
            token_offsets,
            output,
            x.shape[0],
            CHANNELS=x.shape[1],
            KERNEL_SIZE=weight.shape[2],
            DILATION=dilation,
            STATE_LEN=state.shape[2],
            BLOCK_CHANNELS=block_channels,
        )

    requests = state_indices.shape[0]
    if requests:
        writeback_channels = 64
        _qwen4_varlen_state_writeback_kernel[
            (requests, triton.cdiv(x.shape[1], writeback_channels))
        ](
            x,
            state,
            state_indices,
            query_start_loc,
            track_indices if has_track else state_indices,
            track_offsets if has_track else state_indices,
            CHANNELS=x.shape[1],
            STATE_LEN=state.shape[2],
            HAS_TRACK=has_track,
            BLOCK_CHANNELS=writeback_channels,
            BLOCK_STATE_LEN=triton.next_power_of_2(state.shape[2]),
        )
    return output
