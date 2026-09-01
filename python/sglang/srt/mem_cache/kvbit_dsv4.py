"""Built-in DeepSeek V4 BU4 KV-cache primitives.

This module intentionally has no dependency on the external ``kvbit`` package.
It owns the fixed DSV4 layout, a CPU reference codec, and the capability gate
used before the target worker may allocate packed SWA storage.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import NamedTuple

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import KVCache

DSV4_KVBIT_NOPE_DIM = 448
DSV4_KVBIT_ROPE_DIM = 64
DSV4_KVBIT_GROUP_SIZE = 64
DSV4_KVBIT_BITS = 4
DSV4_KVBIT_CODE_BYTES = 224
DSV4_KVBIT_HEADER_BYTES = 28
DSV4_KVBIT_ROPE_BYTES = 128
DSV4_KVBIT_ROW_BYTES = 380
DSV4_NATIVE_SWA_ROW_BYTES = 584
DSV4_KVBIT_HADAMARD_DIM = 256


class DSV4KVBitLayout(NamedTuple):
    nope_dim: int
    rope_dim: int
    group_size: int
    bits: int
    code_bytes: int
    header_bytes: int
    rope_bytes: int
    row_bytes: int

    @property
    def header_offset(self) -> int:
        return self.code_bytes

    @property
    def rope_offset(self) -> int:
        return self.code_bytes + self.header_bytes

    def offsets(self) -> dict[str, tuple[int, int]]:
        return {
            "codes": (0, self.code_bytes),
            "header": (self.header_offset, self.rope_offset),
            "rope": (self.rope_offset, self.row_bytes),
        }


DSV4_BU4_LAYOUT = DSV4KVBitLayout(
    nope_dim=DSV4_KVBIT_NOPE_DIM,
    rope_dim=DSV4_KVBIT_ROPE_DIM,
    group_size=DSV4_KVBIT_GROUP_SIZE,
    bits=DSV4_KVBIT_BITS,
    code_bytes=DSV4_KVBIT_CODE_BYTES,
    header_bytes=DSV4_KVBIT_HEADER_BYTES,
    rope_bytes=DSV4_KVBIT_ROPE_BYTES,
    row_bytes=DSV4_KVBIT_ROW_BYTES,
)


class DSV4KVBitRuntimeCapability(NamedTuple):
    direct_packed_write: bool
    direct_packed_decode: bool


DSV4_KVBIT_RUNTIME_CAPABILITY = DSV4KVBitRuntimeCapability(
    direct_packed_write=_HAS_TRITON,
    direct_packed_decode=_HAS_TRITON,
)


def require_dsv4_kvbit_runtime_capability(
    capability: DSV4KVBitRuntimeCapability = DSV4_KVBIT_RUNTIME_CAPABILITY,
) -> None:
    missing = []
    if not capability.direct_packed_write:
        missing.append("direct packed write")
    if not capability.direct_packed_decode:
        missing.append("direct packed decode")
    if missing:
        raise RuntimeError(
            "SGLANG_ENABLE_KVBIT requires DSV4 "
            + " and ".join(missing)
            + " capability; native/scratch fallback is disabled."
        )


def validate_dsv4_bu4_geometry(nope_dim: int, rope_dim: int) -> None:
    if (nope_dim, rope_dim) != (
        DSV4_BU4_LAYOUT.nope_dim,
        DSV4_BU4_LAYOUT.rope_dim,
    ):
        raise ValueError(
            "DSV4 KVBit supports only the built-in 448-nope/64-rope BU4 "
            f"layout, got {nope_dim}-nope/{rope_dim}-rope."
        )


def dsv4_kvbit_enabled_for_worker(*, enabled: bool, is_draft_worker: bool) -> bool:
    return enabled and not is_draft_worker


def dsv4_kvbit_target_swa_savings(*, swa_ratio: float, num_target_layers: int) -> float:
    return (
        swa_ratio
        * (DSV4_NATIVE_SWA_ROW_BYTES - DSV4_KVBIT_ROW_BYTES)
        * num_target_layers
    )


def _require_cpu_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")


def _hadamard_h256(x: torch.Tensor) -> torch.Tensor:
    """Apply the normalized, self-inverse H256 transform to a tensor prefix."""
    if x.shape[-1] < DSV4_KVBIT_HADAMARD_DIM:
        raise ValueError(
            f"x last dimension must be at least {DSV4_KVBIT_HADAMARD_DIM}, "
            f"got {x.shape}"
        )
    if x.device.type == "cuda":
        from sglang.kernels.ops.quantization.hadamard import _jit_hadamard_module

        out = x.contiguous().clone()
        rows = out.view(-1, out.shape[-1])
        prefix = rows.as_strided(
            (rows.shape[0], DSV4_KVBIT_HADAMARD_DIM),
            (rows.stride(0), rows.stride(1)),
        )
        _jit_hadamard_module(x.dtype).hadamard_transform(
            prefix,
            prefix,
            DSV4_KVBIT_HADAMARD_DIM**-0.5,
        )
        return out

    prefix = x[..., :DSV4_KVBIT_HADAMARD_DIM].float().clone()
    leading = prefix.numel() // DSV4_KVBIT_HADAMARD_DIM
    step = 1
    while step < DSV4_KVBIT_HADAMARD_DIM:
        view = prefix.view(
            leading,
            DSV4_KVBIT_HADAMARD_DIM // (2 * step),
            2,
            step,
        )
        left = view[..., 0, :].clone()
        right = view[..., 1, :].clone()
        view[..., 0, :] = left + right
        view[..., 1, :] = left - right
        step *= 2
    prefix.mul_(DSV4_KVBIT_HADAMARD_DIM**-0.5)
    out = x.clone()
    out[..., :DSV4_KVBIT_HADAMARD_DIM] = prefix.to(x.dtype)
    return out


def fold_dsv4_h256(x: torch.Tensor) -> torch.Tensor:
    """Fold the leading H256 query/value domain; leave dimensions 256:512 intact."""
    return _hadamard_h256(x)


def restore_dsv4_h256(x: torch.Tensor) -> torch.Tensor:
    """Restore the leading H256 output domain (H256 is its own inverse)."""
    return _hadamard_h256(x)


def encode_dsv4_bu4_reference(kv: torch.Tensor) -> torch.Tensor:
    """Encode post-norm/post-RoPE DSV4 KV rows into the fixed BU4 layout."""
    _require_cpu_tensor(kv, name="kv")
    expected_dim = DSV4_BU4_LAYOUT.nope_dim + DSV4_BU4_LAYOUT.rope_dim
    if kv.ndim < 1 or kv.shape[-1] != expected_dim:
        raise ValueError(f"kv last dimension must be {expected_dim}, got {kv.shape}")
    if not kv.is_floating_point():
        raise TypeError(f"kv must be floating point, got {kv.dtype}")

    leading_shape = kv.shape[:-1]
    rows = fold_dsv4_h256(kv.reshape(-1, expected_dim))
    nope = (
        rows[:, : DSV4_BU4_LAYOUT.nope_dim]
        .float()
        .reshape(
            -1,
            DSV4_BU4_LAYOUT.nope_dim // DSV4_BU4_LAYOUT.group_size,
            DSV4_BU4_LAYOUT.group_size,
        )
    )
    group_min = nope.amin(dim=-1)
    group_range = nope.amax(dim=-1) - group_min
    scale = group_range / ((1 << DSV4_BU4_LAYOUT.bits) - 1)
    safe_scale = torch.where(group_range > 0, scale, torch.ones_like(scale))
    codes = torch.round((nope - group_min.unsqueeze(-1)) / safe_scale.unsqueeze(-1))
    codes = torch.where(
        group_range.unsqueeze(-1) > 0, codes, torch.zeros_like(codes)
    ).clamp_(0, 15)
    codes = codes.to(torch.uint8).reshape(-1, DSV4_BU4_LAYOUT.nope_dim)
    packed_codes = codes[:, 0::2] | (codes[:, 1::2] << 4)

    headers = (
        torch.stack((group_min, group_range), dim=-1)
        .to(torch.float16)
        .contiguous()
        .view(torch.uint8)
        .reshape(-1, DSV4_BU4_LAYOUT.header_bytes)
    )
    rope = (
        rows[:, DSV4_BU4_LAYOUT.nope_dim :]
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .reshape(-1, DSV4_BU4_LAYOUT.rope_bytes)
    )
    packed = torch.cat((packed_codes, headers, rope), dim=-1)
    return packed.reshape(*leading_shape, DSV4_BU4_LAYOUT.row_bytes)


def _decode_dsv4_bu4_stored_domain(packed: torch.Tensor) -> torch.Tensor:
    _require_cpu_tensor(packed, name="packed")
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed must have dtype torch.uint8, got {packed.dtype}")
    if packed.ndim < 1 or packed.shape[-1] != DSV4_BU4_LAYOUT.row_bytes:
        raise ValueError(
            f"packed last dimension must be {DSV4_BU4_LAYOUT.row_bytes}, "
            f"got {packed.shape}"
        )

    leading_shape = packed.shape[:-1]
    rows = packed.reshape(-1, DSV4_BU4_LAYOUT.row_bytes)
    packed_codes = rows[:, : DSV4_BU4_LAYOUT.code_bytes]
    codes = torch.stack(
        (packed_codes & 0x0F, packed_codes >> 4),
        dim=-1,
    ).reshape(-1, DSV4_BU4_LAYOUT.nope_dim)

    headers = (
        rows[
            :,
            DSV4_BU4_LAYOUT.header_offset : DSV4_BU4_LAYOUT.rope_offset,
        ]
        .contiguous()
        .view(torch.float16)
        .reshape(-1, DSV4_BU4_LAYOUT.nope_dim // DSV4_BU4_LAYOUT.group_size, 2)
        .float()
    )
    group_min = headers[..., 0]
    group_range = headers[..., 1]
    scale = group_range / ((1 << DSV4_BU4_LAYOUT.bits) - 1)
    nope = group_min.unsqueeze(-1) + codes.reshape(
        -1,
        DSV4_BU4_LAYOUT.nope_dim // DSV4_BU4_LAYOUT.group_size,
        DSV4_BU4_LAYOUT.group_size,
    ).float() * scale.unsqueeze(-1)
    nope = nope.reshape(-1, DSV4_BU4_LAYOUT.nope_dim).to(torch.bfloat16)

    rope = (
        rows[:, DSV4_BU4_LAYOUT.rope_offset :]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(-1, DSV4_BU4_LAYOUT.rope_dim)
    )
    decoded = torch.cat((nope, rope), dim=-1)
    return decoded.reshape(
        *leading_shape, DSV4_BU4_LAYOUT.nope_dim + DSV4_BU4_LAYOUT.rope_dim
    )


def decode_dsv4_bu4_reference(packed: torch.Tensor) -> torch.Tensor:
    """Decode fixed-layout BU4 rows and restore their H256 prefix on CPU."""
    return restore_dsv4_h256(_decode_dsv4_bu4_stored_domain(packed))


def merge_attention_states_natural_log(
    left_output: torch.Tensor,
    left_lse: torch.Tensor,
    right_output: torch.Tensor,
    right_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge independently normalized attention states using natural-log LSE."""
    if left_output.shape != right_output.shape:
        raise ValueError(
            f"attention output shapes must match, got "
            f"{left_output.shape} and {right_output.shape}"
        )
    expected_lse_shape = (
        left_output.shape[0],
        left_output.shape[2],
        left_output.shape[1],
    )
    if left_lse.shape != expected_lse_shape or right_lse.shape != expected_lse_shape:
        raise ValueError(
            f"LSE shape must be {expected_lse_shape} for output "
            f"{left_output.shape}, got {left_lse.shape} and {right_lse.shape}"
        )

    left_lse = torch.where(torch.isposinf(left_lse), -torch.inf, left_lse.float())
    right_lse = torch.where(torch.isposinf(right_lse), -torch.inf, right_lse.float())
    merged_lse = torch.logaddexp(left_lse, right_lse)
    left_weight = torch.exp(left_lse - merged_lse)
    right_weight = torch.exp(right_lse - merged_lse)
    left_weight = torch.where(
        torch.isfinite(left_lse), left_weight, torch.zeros_like(left_weight)
    )
    right_weight = torch.where(
        torch.isfinite(right_lse), right_weight, torch.zeros_like(right_weight)
    )
    left_weight = left_weight.transpose(1, 2).unsqueeze(-1)
    right_weight = right_weight.transpose(1, 2).unsqueeze(-1)
    output = left_output.float() * left_weight + right_output.float() * right_weight
    return output.to(left_output.dtype), merged_lse


if _HAS_TRITON:

    @triton.jit
    def _dsv4_bu4_pack_scatter_kernel(
        kv_ptr,
        loc_ptr,
        packed_ptr,
        stride_kv_row,
        stride_page,
        num_pages,
        PAGE_SIZE: tl.constexpr,
        ROW_BYTES: tl.constexpr,
    ):
        row = tl.program_id(0)
        loc = tl.load(loc_ptr + row)
        valid_loc = (loc >= 0) & (loc < num_pages * PAGE_SIZE)
        page = loc // PAGE_SIZE
        page_offset = loc % PAGE_SIZE
        dst = packed_ptr + page * stride_page + page_offset * ROW_BYTES

        offs = tl.arange(0, 512)
        values = tl.load(kv_ptr + row * stride_kv_row + offs).to(tl.float32)

        prefix = values
        partner = tl.gather(prefix, offs ^ 1, axis=0)
        prefix = tl.where((offs & 1) == 0, prefix + partner, partner - prefix)
        partner = tl.gather(prefix, offs ^ 2, axis=0)
        prefix = tl.where((offs & 2) == 0, prefix + partner, partner - prefix)
        partner = tl.gather(prefix, offs ^ 4, axis=0)
        prefix = tl.where((offs & 4) == 0, prefix + partner, partner - prefix)
        partner = tl.gather(prefix, offs ^ 8, axis=0)
        prefix = tl.where((offs & 8) == 0, prefix + partner, partner - prefix)
        partner = tl.gather(prefix, offs ^ 16, axis=0)
        prefix = tl.where((offs & 16) == 0, prefix + partner, partner - prefix)
        partner = tl.gather(prefix, offs ^ 32, axis=0)
        prefix = tl.where((offs & 32) == 0, prefix + partner, partner - prefix)
        partner = tl.gather(prefix, offs ^ 64, axis=0)
        prefix = tl.where((offs & 64) == 0, prefix + partner, partner - prefix)
        partner = tl.gather(prefix, offs ^ 128, axis=0)
        prefix = tl.where((offs & 128) == 0, prefix + partner, partner - prefix)
        values = tl.where(offs < 256, prefix * 0.0625, values)

        grouped = tl.reshape(values, [8, 64])
        group_min = tl.min(grouped, axis=1)
        group_max = tl.max(grouped, axis=1)
        group_range = group_max - group_min
        safe_step = tl.where(group_range > 0, group_range / 15.0, 1.0)
        codes = tl.floor((grouped - group_min[:, None]) / safe_step[:, None] + 0.5)
        codes = tl.where(group_range[:, None] > 0, codes, 0.0)
        codes = tl.minimum(tl.maximum(codes, 0.0), 15.0).to(tl.uint8)
        paired = tl.reshape(codes, [8, 32, 2])
        low, high = tl.split(paired)
        packed_codes = tl.reshape((low | (high << 4)), [256])
        code_offsets = tl.arange(0, 256)
        tl.store(
            dst + code_offsets,
            packed_codes,
            mask=valid_loc & (code_offsets < 224),
        )

        group_offsets = tl.arange(0, 8)
        header_ptr = dst + 224 + group_offsets * 4
        tl.store(
            header_ptr.to(tl.pointer_type(tl.float16)),
            group_min.to(tl.float16),
            mask=valid_loc & (group_offsets < 7),
        )
        tl.store(
            (header_ptr + 2).to(tl.pointer_type(tl.float16)),
            group_range.to(tl.float16),
            mask=valid_loc & (group_offsets < 7),
        )

        rope_offsets = tl.arange(0, 64)
        rope = tl.load(kv_ptr + row * stride_kv_row + 448 + rope_offsets).to(
            tl.bfloat16
        )
        tl.store(
            (dst + 252).to(tl.pointer_type(tl.bfloat16)) + rope_offsets,
            rope,
            mask=valid_loc,
        )

    @triton.jit
    def _dsv4_bu4_sparse_decode_kernel(
        q_ptr,
        packed_ptr,
        indices_ptr,
        lengths_ptr,
        sink_ptr,
        output_ptr,
        lse_ptr,
        stride_q_row,
        stride_q_head,
        stride_page,
        stride_indices_row,
        stride_output_row,
        stride_output_head,
        stride_lse_row,
        num_pages,
        softmax_scale,
        num_heads: tl.constexpr,
        index_width: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        ROW_BYTES: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        query_row = tl.program_id(0)
        head = tl.program_id(1)
        length = tl.load(lengths_ptr + query_row)

        q_offsets = tl.arange(0, 512)
        q = tl.load(
            q_ptr + query_row * stride_q_row + head * stride_q_head + q_offsets
        ).to(tl.float32)
        max_score = tl.load(sink_ptr + head).to(tl.float32)
        normalizer = 1.0
        accumulator = tl.zeros([512], dtype=tl.float32)

        start = 0
        while start < length:
            token_offsets = start + tl.arange(0, BLOCK_N)
            valid_token = token_offsets < length
            loc = tl.load(
                indices_ptr + query_row * stride_indices_row + token_offsets,
                mask=valid_token & (token_offsets < index_width),
                other=-1,
            )
            valid_token = valid_token & (loc >= 0) & (loc < num_pages * PAGE_SIZE)
            page = loc // PAGE_SIZE
            page_offset = loc % PAGE_SIZE
            row_ptr = (
                packed_ptr
                + page[:, None] * stride_page
                + page_offset[:, None] * ROW_BYTES
            )

            score = tl.zeros([BLOCK_N], dtype=tl.float32)
            for group in range(7):
                byte_offsets = tl.arange(0, 32)
                packed_codes = tl.load(
                    row_ptr + group * 32 + byte_offsets[None, :],
                    mask=valid_token[:, None],
                    other=0,
                )
                codes = tl.interleave(
                    (packed_codes & 0x0F).to(tl.float32),
                    ((packed_codes >> 4) & 0x0F).to(tl.float32),
                )
                header = row_ptr + 224 + group * 4
                minimum_bits = tl.load(
                    header.to(tl.pointer_type(tl.uint16)),
                    mask=valid_token[:, None],
                    other=0,
                )
                range_bits = tl.load(
                    (header + 2).to(tl.pointer_type(tl.uint16)),
                    mask=valid_token[:, None],
                    other=0,
                )
                minimum = tl.cast(
                    tl.reshape(minimum_bits, [BLOCK_N]), tl.float16, bitcast=True
                ).to(tl.float32)
                value_range = tl.cast(
                    tl.reshape(range_bits, [BLOCK_N]), tl.float16, bitcast=True
                ).to(tl.float32)
                values = codes * (value_range[:, None] / 15.0) + minimum[:, None]
                dim_offsets = group * 64 + tl.arange(0, 64)
                q_group = tl.gather(q, dim_offsets, axis=0)
                score += tl.sum(q_group[None, :] * values, axis=1)

            rope_offsets = tl.arange(0, 64)
            rope = tl.load(
                (row_ptr + 252).to(tl.pointer_type(tl.bfloat16))
                + rope_offsets[None, :],
                mask=valid_token[:, None],
                other=0.0,
            ).to(tl.float32)
            q_rope = tl.gather(q, 448 + rope_offsets, axis=0)
            score += tl.sum(q_rope[None, :] * rope, axis=1)
            score = tl.where(valid_token, score * softmax_scale, -float("inf"))

            next_max = tl.maximum(max_score, tl.max(score, axis=0))
            rescale = tl.exp(max_score - next_max)
            probabilities = tl.exp(score - next_max)
            probabilities = tl.where(valid_token, probabilities, 0.0)
            accumulator *= rescale

            for group in range(7):
                byte_offsets = tl.arange(0, 32)
                packed_codes = tl.load(
                    row_ptr + group * 32 + byte_offsets[None, :],
                    mask=valid_token[:, None],
                    other=0,
                )
                codes = tl.interleave(
                    (packed_codes & 0x0F).to(tl.float32),
                    ((packed_codes >> 4) & 0x0F).to(tl.float32),
                )
                header = row_ptr + 224 + group * 4
                minimum_bits = tl.load(
                    header.to(tl.pointer_type(tl.uint16)),
                    mask=valid_token[:, None],
                    other=0,
                )
                range_bits = tl.load(
                    (header + 2).to(tl.pointer_type(tl.uint16)),
                    mask=valid_token[:, None],
                    other=0,
                )
                minimum = tl.cast(
                    tl.reshape(minimum_bits, [BLOCK_N]), tl.float16, bitcast=True
                ).to(tl.float32)
                value_range = tl.cast(
                    tl.reshape(range_bits, [BLOCK_N]), tl.float16, bitcast=True
                ).to(tl.float32)
                values = codes * (value_range[:, None] / 15.0) + minimum[:, None]
                partial = tl.sum(probabilities[:, None] * values, axis=0)
                relative = tl.maximum(tl.minimum(q_offsets - group * 64, 63), 0)
                accumulator += tl.where(
                    (q_offsets >= group * 64) & (q_offsets < (group + 1) * 64),
                    tl.gather(partial, relative, axis=0),
                    0.0,
                )

            rope_partial = tl.sum(probabilities[:, None] * rope, axis=0)
            rope_relative = tl.maximum(tl.minimum(q_offsets - 448, 63), 0)
            accumulator += tl.where(
                q_offsets >= 448,
                tl.gather(rope_partial, rope_relative, axis=0),
                0.0,
            )
            normalizer = normalizer * rescale + tl.sum(probabilities, axis=0)
            max_score = next_max
            start += BLOCK_N

        output_offsets = tl.arange(0, 512)
        tl.store(
            output_ptr
            + query_row * stride_output_row
            + head * stride_output_head
            + output_offsets,
            accumulator / normalizer,
        )
        tl.store(
            lse_ptr + query_row * stride_lse_row + head,
            max_score + tl.log(normalizer),
        )


def _reshape_packed_rows(packed: torch.Tensor, *, page_size: int) -> torch.Tensor:
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("packed cache must be a rank-2 torch.uint8 tensor")
    expected_page_bytes = page_size * DSV4_BU4_LAYOUT.row_bytes
    if packed.shape[1] != expected_page_bytes:
        raise ValueError(
            f"packed page width must be {expected_page_bytes}, got {packed.shape[1]}"
        )
    return packed.reshape(-1, DSV4_BU4_LAYOUT.row_bytes)


def write_dsv4_bu4_packed(
    kv: torch.Tensor,
    loc: torch.Tensor,
    packed: torch.Tensor,
    *,
    page_size: int,
) -> None:
    """H256-fold, BU4-encode, and scatter DSV4 rows without FP8 scratch."""
    if kv.device.type != "cuda" or not _HAS_TRITON:
        raise RuntimeError("DSV4 KVBit packed writes require CUDA and Triton")
    if kv.ndim != 2 or kv.shape[-1] != 512:
        raise ValueError(f"kv must have shape (tokens, 512), got {kv.shape}")
    if loc.ndim != 1 or loc.shape[0] != kv.shape[0]:
        raise ValueError(f"loc must have shape ({kv.shape[0]},), got {loc.shape}")
    if kv.device != loc.device or kv.device != packed.device:
        raise ValueError("kv, loc, and packed cache must be on the same device")
    _reshape_packed_rows(packed, page_size=page_size)
    _dsv4_bu4_pack_scatter_kernel[(kv.shape[0],)](
        kv,
        loc,
        packed,
        kv.stride(0),
        packed.stride(0),
        packed.shape[0],
        PAGE_SIZE=page_size,
        ROW_BYTES=DSV4_BU4_LAYOUT.row_bytes,
        num_warps=4,
        num_stages=1,
    )


def _dsv4_sparse_decode_reference(
    q: torch.Tensor,
    packed: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    attn_sink: torch.Tensor,
    *,
    page_size: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = _reshape_packed_rows(packed, page_size=page_size)
    q_folded = fold_dsv4_h256(q)
    output = torch.empty_like(q)
    lse = torch.empty(
        (q.shape[0], q.shape[2], q.shape[1]),
        dtype=torch.float32,
        device=q.device,
    )
    for row in range(q.shape[0]):
        count = int(lengths[row])
        selected = indices[row, 0, :count].to(torch.long)
        selected = selected[(selected >= 0) & (selected < rows.shape[0])]
        stored = _decode_dsv4_bu4_stored_domain(rows[selected].cpu()).to(q.device)
        scores = torch.einsum("qhd,kd->qhk", q_folded[row].float(), stored.float())
        scores.mul_(softmax_scale)
        sink = attn_sink.float().view(1, -1, 1)
        scores = torch.cat((scores, sink), dim=-1)
        probabilities = torch.softmax(scores, dim=-1)[..., :-1]
        output[row] = torch.einsum("qhk,kd->qhd", probabilities, stored.float()).to(
            q.dtype
        )
        lse[row] = torch.logsumexp(scores, dim=-1).transpose(0, 1)
    return restore_dsv4_h256(output), lse


def dsv4_kvbit_sparse_decode(
    q: torch.Tensor,
    packed: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    attn_sink: torch.Tensor,
    *,
    page_size: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct sparse attention over page_size=256 packed DSV4 Q/K/V=512 rows."""
    if page_size != 256:
        raise ValueError(f"DSV4 KVBit requires page_size=256, got {page_size}")
    if q.ndim != 4 or q.shape[1] != 1 or q.shape[-1] != 512:
        raise ValueError(f"q must have shape (tokens, 1, heads, 512), got {q.shape}")
    if indices.ndim != 3 or indices.shape[:2] != q.shape[:2]:
        raise ValueError(
            f"indices must have shape ({q.shape[0]}, 1, width), got {indices.shape}"
        )
    _reshape_packed_rows(packed, page_size=page_size)
    if q.device.type != "cuda" or not _HAS_TRITON:
        return _dsv4_sparse_decode_reference(
            q,
            packed,
            indices,
            lengths,
            attn_sink,
            page_size=page_size,
            softmax_scale=softmax_scale,
        )

    q_folded = fold_dsv4_h256(q).contiguous()
    output_folded = torch.empty_like(q_folded)
    lse = torch.empty(
        (q.shape[0], q.shape[2], 1),
        dtype=torch.float32,
        device=q.device,
    )
    _dsv4_bu4_sparse_decode_kernel[(q.shape[0], q.shape[2])](
        q_folded,
        packed,
        indices,
        lengths,
        attn_sink,
        output_folded,
        lse,
        q_folded.stride(0),
        q_folded.stride(2),
        packed.stride(0),
        indices.stride(0),
        output_folded.stride(0),
        output_folded.stride(2),
        lse.stride(0),
        packed.shape[0],
        softmax_scale,
        num_heads=q.shape[2],
        index_width=indices.shape[-1],
        PAGE_SIZE=page_size,
        ROW_BYTES=DSV4_BU4_LAYOUT.row_bytes,
        BLOCK_N=16,
        num_warps=4,
        num_stages=1,
    )
    return restore_dsv4_h256(output_folded), lse


class DSV4KVBitPackedSWAPool(KVCache):
    """Persistent 380-byte target SWA rows with no native shadow or scratch."""

    is_dsv4_kvbit_packed_swa = True

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        validate_dsv4_bu4_geometry(qk_nope_head_dim, qk_rope_head_dim)
        if page_size != 256:
            raise ValueError(f"DSV4 KVBit requires page_size=256, got {page_size}")
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.store_dtype = torch.uint8
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_cache_total_dim = DSV4_BU4_LAYOUT.row_bytes
        self.bytes_per_page_padded = self.page_size * DSV4_BU4_LAYOUT.row_bytes
        self.num_pages = (self.size + self.page_size + 1) // self.page_size
        with (
            self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ),
        ):
            self.kv_buffer = [
                torch.zeros(
                    self.num_pages,
                    self.page_size * DSV4_BU4_LAYOUT.row_bytes,
                    dtype=torch.uint8,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

    def get_bytes_per_token(self) -> int:
        return DSV4_BU4_LAYOUT.row_bytes

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_buffer[layer_id - self.start_layer]

    def set_key_buffer(self, *args, **kwargs) -> None:
        raise RuntimeError("DSV4 KVBit accepts only fused BF16 direct writes")

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        write_dsv4_bu4_packed(
            cache_k,
            loc,
            self.kv_buffer[layer_id - self.start_layer],
            page_size=self.page_size,
        )

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("DSV4 KVBit uses a single packed K/V buffer")

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("DSV4 KVBit uses a single packed K/V buffer")

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError("DSV4 KVBit uses direct packed writes")
