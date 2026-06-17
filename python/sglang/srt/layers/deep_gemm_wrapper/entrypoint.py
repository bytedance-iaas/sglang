import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.deep_gemm_wrapper.configurer import (  # noqa: F401
    DEEPGEMM_BLACKWELL,
    DEEPGEMM_NEED_TMA_ALIGNED_SCALES,
    DEEPGEMM_SCALE_UE8M0,
    ENABLE_JIT_DEEPGEMM,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_cuda, is_sm90_supported, is_sm100_supported

logger = logging.getLogger(__name__)

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor  # noqa: F401

_SANITY_CHECK = envs.SGLANG_DEEPGEMM_SANITY_CHECK.get()
_SM90_MXFP8_LOCAL_DIFF_EVENTS = 0


def supports_sm90_mxfp8_fp8_grouped_gemm() -> bool:
    if not ENABLE_JIT_DEEPGEMM:
        return False
    return hasattr(deep_gemm, "m_grouped_mxfp8_fp8_gemm_nt_contiguous") and hasattr(
        deep_gemm, "m_grouped_mxfp8_fp8_gemm_nt_masked"
    )


def is_sm90_mxfp8_deepgemm_enabled() -> bool:
    return (
        is_cuda()
        and is_sm90_supported()
        and not is_sm100_supported()
        and supports_sm90_mxfp8_fp8_grouped_gemm()
    )


def supports_mxfp8_deepgemm() -> bool:
    return DEEPGEMM_BLACKWELL or is_sm90_mxfp8_deepgemm_enabled()


def _ceil_align(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def _sm90_mxfp8_format_debug_enabled() -> bool:
    if os.environ.get("SGLANG_SM90_MXFP8_DEBUG") != "1":
        return False
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank() == 0
    except Exception:
        return False


def _sm90_mxfp8_tensor_meta(t: torch.Tensor) -> dict:
    return {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype),
        "is_contiguous": t.is_contiguous(),
        "device": str(t.device),
        "numel": t.numel(),
    }


def _sm90_mxfp8_scale_expected_last_dim(
    scale: torch.Tensor,
    k: int,
    recipe: Optional[Tuple[int, int]],
) -> Optional[int]:
    if recipe is None:
        return None
    pack_factor = 4 if scale.dtype in (torch.int, torch.int32) else 1
    return (k + recipe[1] * pack_factor - 1) // (recipe[1] * pack_factor)


def _assert_sm90_mxfp8_scale_matches_recipe(
    scale: torch.Tensor,
    k: int,
    recipe: Optional[Tuple[int, int]],
    *,
    role: str,
) -> None:
    expected_last_dim = _sm90_mxfp8_scale_expected_last_dim(scale, k, recipe)
    if expected_last_dim is None:
        return
    if scale.shape[-1] != expected_last_dim:
        pack_factor = 4 if scale.dtype in (torch.int, torch.int32) else 1
        raise RuntimeError(
            f"SM90 MXFP8 {role} scale shape does not match recipe: "
            f"K={k}, recipe={recipe}, scale_dtype={scale.dtype}, "
            f"scale_shape={tuple(scale.shape)}, pack_factor={pack_factor}, "
            f"expected_last_dim={expected_last_dim}."
        )


def _sm90_mxfp8_local_diff_enabled() -> bool:
    return (
        _sm90_mxfp8_format_debug_enabled()
        and os.environ.get("SGLANG_SM90_MXFP8_DEBUG_STATS") == "1"
    )


def _unpack_sm90_mxfp8_scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype in (torch.int, torch.int32):
        scale_i32 = scale.contiguous().to(torch.int32)
        return torch.stack(
            [
                torch.bitwise_and(torch.bitwise_right_shift(scale_i32, shift), 0xFF)
                for shift in (0, 8, 16, 24)
            ],
            dim=-1,
        ).reshape(-1).to(torch.uint8)
    if scale.dtype == torch.uint8:
        return scale.contiguous()
    if scale.dtype == torch.float32:
        return _e8m0_fp32_to_u8(scale)
    raise RuntimeError(f"Unsupported SM90 MXFP8 scale dtype for debug: {scale.dtype}")


def _decode_e8m0_scale_bytes(scale_bytes: torch.Tensor) -> torch.Tensor:
    scale_i32 = torch.bitwise_left_shift(scale_bytes.to(torch.int32), 23).contiguous()
    return scale_i32.view(torch.float32)


def _dequant_sm90_mxfp8_vector(
    x: torch.Tensor,
    scale: torch.Tensor,
    *,
    gran_k: int,
    k: int,
) -> torch.Tensor:
    scale_bytes = _unpack_sm90_mxfp8_scale_bytes(scale)
    k_scale_idx = torch.arange(k, device=x.device, dtype=torch.long) // gran_k
    scale_per_k = _decode_e8m0_scale_bytes(scale_bytes[k_scale_idx])
    return x[:k].to(torch.float32) * scale_per_k


def _sm90_mxfp8_numeric_stats(t: torch.Tensor) -> dict:
    flat = t.reshape(-1)
    finite = torch.isfinite(flat)
    finite_count = int(finite.sum().item())
    stats = {
        "numel": flat.numel(),
        "finite_count": finite_count,
        "nan_count": int(torch.isnan(flat).sum().item()),
        "inf_count": int(torch.isinf(flat).sum().item()),
        "isfinite": finite_count == flat.numel(),
    }
    if finite_count > 0:
        stats["finite_absmax"] = float(flat[finite].abs().max().item())
    return stats


def _sm90_mxfp8_scale_byte_stats(scale: torch.Tensor) -> dict:
    scale_bytes = _unpack_sm90_mxfp8_scale_bytes(scale)
    return {
        "numel": scale_bytes.numel(),
        "min": int(scale_bytes.min().item()) if scale_bytes.numel() > 0 else None,
        "max": int(scale_bytes.max().item()) if scale_bytes.numel() > 0 else None,
        "zero_count": int((scale_bytes == 0).sum().item()),
        "ff_count": int((scale_bytes == 0xFF).sum().item()),
    }


def _sm90_mxfp8_diff_entry(
    *,
    group: int,
    row: int,
    sample_cols: int,
    diff: torch.Tensor,
    ref: torch.Tensor,
    actual: torch.Tensor,
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
    a_deq: torch.Tensor,
    b_deq: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> dict:
    return {
        "group": group,
        "row": row,
        "cols": sample_cols,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "ref_absmax": float(ref.abs().max().item()),
        "actual_absmax": float(actual.abs().max().item()),
        "ref_stats": _sm90_mxfp8_numeric_stats(ref),
        "actual_stats": _sm90_mxfp8_numeric_stats(actual),
        "a_raw_fp32_stats": _sm90_mxfp8_numeric_stats(a_raw.to(torch.float32)),
        "b_raw_fp32_stats": _sm90_mxfp8_numeric_stats(b_raw.to(torch.float32)),
        "a_deq_stats": _sm90_mxfp8_numeric_stats(a_deq),
        "b_deq_stats": _sm90_mxfp8_numeric_stats(b_deq),
        "a_scale_u8_stats": _sm90_mxfp8_scale_byte_stats(a_scale),
        "b_scale_u8_stats": _sm90_mxfp8_scale_byte_stats(b_scale),
    }


def _sm90_mxfp8_local_diff_report(
    location: str,
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    *,
    recipe_a: Optional[Tuple[int, int]],
    recipe_b: Optional[Tuple[int, int]],
    m_indices: Optional[torch.Tensor] = None,
    masked_m: Optional[torch.Tensor] = None,
) -> None:
    global _SM90_MXFP8_LOCAL_DIFF_EVENTS
    if not _sm90_mxfp8_local_diff_enabled():
        return
    max_events = int(
        os.environ.get("SGLANG_SM90_MXFP8_DEBUG_LOCAL_DIFF_MAX_EVENTS", "4")
    )
    if _SM90_MXFP8_LOCAL_DIFF_EVENTS >= max_events:
        return
    if recipe_a is None or recipe_b is None:
        return
    if out.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    _SM90_MXFP8_LOCAL_DIFF_EVENTS += 1

    lhs_x, lhs_sf = lhs
    rhs_x, rhs_sf = rhs
    k = lhs_x.shape[-1]
    gran_k_a = recipe_a[1]
    gran_k_b = recipe_b[1]
    sample_cols = min(
        rhs_x.shape[-2],
        int(os.environ.get("SGLANG_SM90_MXFP8_DEBUG_LOCAL_DIFF_N", "16")),
    )
    sample_rows_per_group = int(
        os.environ.get("SGLANG_SM90_MXFP8_DEBUG_LOCAL_DIFF_M", "2")
    )
    sample_groups = min(
        rhs_x.shape[0],
        int(os.environ.get("SGLANG_SM90_MXFP8_DEBUG_LOCAL_DIFF_GROUPS", "2")),
    )

    diffs = []
    with torch.no_grad():
        rhs_cols = torch.arange(sample_cols, device=rhs_x.device, dtype=torch.long)
        if m_indices is not None:
            row_count = min(lhs_x.shape[0], out.shape[0], sample_rows_per_group * 2)
            for row in range(row_count):
                group = int(m_indices[row].item())
                if group < 0 or group >= rhs_x.shape[0]:
                    continue
                a_deq = _dequant_sm90_mxfp8_vector(
                    lhs_x[row],
                    lhs_sf[row],
                    gran_k=gran_k_a,
                    k=k,
                )
                b_deq = torch.stack(
                    [
                        _dequant_sm90_mxfp8_vector(
                            rhs_x[group, col],
                            rhs_sf[group, col],
                            gran_k=gran_k_b,
                            k=k,
                        )
                        for col in rhs_cols.tolist()
                    ],
                    dim=0,
                )
                ref = torch.matmul(b_deq, a_deq).to(torch.float32)
                actual = out[row, :sample_cols].to(torch.float32)
                diff = (actual - ref).abs()
                diffs.append(
                    _sm90_mxfp8_diff_entry(
                        group=group,
                        row=row,
                        sample_cols=sample_cols,
                        diff=diff,
                        ref=ref,
                        actual=actual,
                        a_raw=lhs_x[row],
                        b_raw=rhs_x[group, rhs_cols],
                        a_deq=a_deq,
                        b_deq=b_deq,
                        a_scale=lhs_sf[row],
                        b_scale=rhs_sf[group, rhs_cols],
                    )
                )
        else:
            if masked_m is None:
                group_candidates = list(range(sample_groups))
            else:
                group_candidates = [
                    group
                    for group in range(rhs_x.shape[0])
                    if int(masked_m[group].item()) > 0
                ][:sample_groups]
            for group in group_candidates:
                valid_m = out.shape[-2]
                if masked_m is not None:
                    valid_m = min(valid_m, int(masked_m[group].item()))
                row_count = min(valid_m, sample_rows_per_group)
                for row in range(row_count):
                    a_deq = _dequant_sm90_mxfp8_vector(
                        lhs_x[group, row],
                        lhs_sf[group, row],
                        gran_k=gran_k_a,
                        k=k,
                    )
                    b_deq = torch.stack(
                        [
                            _dequant_sm90_mxfp8_vector(
                                rhs_x[group, col],
                                rhs_sf[group, col],
                                gran_k=gran_k_b,
                                k=k,
                            )
                            for col in rhs_cols.tolist()
                        ],
                        dim=0,
                    )
                    ref = torch.matmul(b_deq, a_deq).to(torch.float32)
                    actual = out[group, row, :sample_cols].to(torch.float32)
                    diff = (actual - ref).abs()
                    diffs.append(
                        _sm90_mxfp8_diff_entry(
                            group=group,
                            row=row,
                            sample_cols=sample_cols,
                            diff=diff,
                            ref=ref,
                            actual=actual,
                            a_raw=lhs_x[group, row],
                            b_raw=rhs_x[group, rhs_cols],
                            a_deq=a_deq,
                            b_deq=b_deq,
                            a_scale=lhs_sf[group, row],
                            b_scale=rhs_sf[group, rhs_cols],
                        )
                    )

    payload = {
        "sessionId": "sm90-mxfp8-precision",
        "runId": os.environ.get("SGLANG_SM90_MXFP8_DEBUG_RUN_ID", "pre-fix"),
        "hypothesisId": "H8",
        "location": location,
        "msg": "[DEBUG] SM90 MXFP8 grouped GEMM local PyTorch reference diff",
        "data": {
            "recipe_a": list(recipe_a),
            "recipe_b": list(recipe_b),
            "lhs": _sm90_mxfp8_tensor_meta(lhs_x),
            "lhs_scale": _sm90_mxfp8_tensor_meta(lhs_sf),
            "rhs": _sm90_mxfp8_tensor_meta(rhs_x),
            "rhs_scale": _sm90_mxfp8_tensor_meta(rhs_sf),
            "out": _sm90_mxfp8_tensor_meta(out),
            "diffs": diffs,
        },
    }
    logger.warning("[SM90_MXFP8_DEBUG] %s", json.dumps(payload, ensure_ascii=False))


def _e8m0_fp32_to_u8(sf: torch.Tensor) -> torch.Tensor:
    sf_i32 = sf.to(torch.float32).view(torch.int32)
    exp = torch.bitwise_right_shift(sf_i32, 23)
    mant = torch.bitwise_and(sf_i32, 0x7FFFFF)
    round_up = torch.logical_and(
        torch.logical_and(mant > 0, exp != 0xFE),
        ~torch.logical_and(exp == 0, mant <= 0x400000),
    )
    return torch.where(round_up, exp + 1, exp).to(torch.uint8).contiguous()


def _normalize_sm90_mxfp8_pair(
    pair: Tuple[torch.Tensor, torch.Tensor],
    *,
    scale_role: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, sf = pair
    k = x.shape[-1]
    k32 = _ceil_align(k, 32) // 32

    if sf.dtype == torch.uint8 and sf.shape[-1] == k32 and x.dtype == torch.float8_e4m3fn:
        return x, sf.contiguous()

    if sf.dtype == torch.int32:
        if x.dtype == torch.float8_e4m3fn:
            return x, sf
        raise RuntimeError(
            f"SM90 MXFP8 {scale_role} scale received packed UE8M0 scales "
            f"but activation dtype is {x.dtype}; expected torch.float8_e4m3fn."
        )

    if sf.dtype == torch.float32 and sf.shape[-1] == k32 and x.dtype == torch.float8_e4m3fn:
        return x, _e8m0_fp32_to_u8(sf)

    raise RuntimeError(
        f"SM90 MXFP8 {scale_role} wrapper only performs lossless scale layout adaptation. "
        f"Got activation dtype={x.dtype}, scale dtype={sf.dtype}, "
        f"scale_last_dim={sf.shape[-1]}, expected K/32={k32}."
    )


def _pad_sm90_mxfp8_lhs(
    lhs: Tuple[torch.Tensor, torch.Tensor], expected_m: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, sf = lhs
    if x.shape[-2] >= expected_m:
        return lhs
    padded_x = torch.empty(
        (*x.shape[:-2], expected_m, x.shape[-1]), device=x.device, dtype=x.dtype
    )
    padded_sf = torch.empty(
        (*sf.shape[:-2], expected_m, sf.shape[-1]), device=sf.device, dtype=sf.dtype
    )
    if sf.dtype == torch.int32 and not sf.is_contiguous() and sf.dim() >= 2:
        padded_sf_storage = torch.empty(
            (*sf.shape[:-2], sf.shape[-1], expected_m),
            device=sf.device,
            dtype=sf.dtype,
        )
        padded_sf = padded_sf_storage.transpose(-1, -2)
    padded_x[..., : x.shape[-2], :] = x
    padded_sf[..., : sf.shape[-2], :] = sf
    return padded_x, padded_sf


# TODO maybe rename these functions
def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
):
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    lhs = _ensure_cuda(lhs)
    rhs = _ensure_cuda(rhs)

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        with configure_deep_gemm_num_sms(
            overlap_args.num_sms if overlap_args is not None else None
        ):

            fp4_kwargs = {}
            if recipe_a is not None:
                fp4_kwargs["recipe_a"] = recipe_a
            if recipe_b is not None:
                fp4_kwargs["recipe_b"] = recipe_b

            return deep_gemm.fp8_m_grouped_gemm_nt_masked(
                lhs,
                rhs,
                out,
                masked_m,
                expected_m,
                **fp4_kwargs,
                **(
                    dict(
                        enable_overlap=True,
                        max_block_n=max_block_n,
                        signal=overlap_args.signal,
                    )
                    if overlap_args is not None
                    else {}
                ),
            )


def grouped_gemm_nt_mxfp8_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
):
    if not supports_sm90_mxfp8_fp8_grouped_gemm():
        raise RuntimeError(
            "The installed deep_gemm does not expose SM90 MXFP8 grouped GEMM APIs."
        )

    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = (
        compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_MXFP8_F8BF16_MASKED
    )

    lhs = _normalize_sm90_mxfp8_pair(_ensure_cuda(lhs), scale_role="lhs")
    rhs = _normalize_sm90_mxfp8_pair(_ensure_cuda(rhs), scale_role="rhs")
    _assert_sm90_mxfp8_scale_matches_recipe(
        lhs[1], k, recipe_a, role="lhs"
    )
    _assert_sm90_mxfp8_scale_matches_recipe(
        rhs[1], k, recipe_b, role="rhs"
    )

    padded_expected_m = _ceil_align(max(lhs[0].shape[-2], expected_m), 128)
    lhs = _pad_sm90_mxfp8_lhs(lhs, padded_expected_m)
    kernel_out = out
    if out.shape[-2] < padded_expected_m:
        kernel_out = torch.empty(
            (*out.shape[:-2], padded_expected_m, out.shape[-1]),
            device=out.device,
            dtype=out.dtype,
        )

    with compile_utils.deep_gemm_execution_hook(
        padded_expected_m, n, k, num_groups, kernel_type
    ):
        ret = deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_masked(
            lhs,
            rhs,
            kernel_out,
            masked_m,
            padded_expected_m,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
    if kernel_out is not out:
        out.copy_(kernel_out[..., : out.shape[-2], :])
    _sm90_mxfp8_local_diff_report(
        "entrypoint.py:grouped_gemm_nt_mxfp8_f8f8bf16_masked:local_diff",
        lhs,
        rhs,
        out,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        masked_m=masked_m,
    )
    return ret


def _ensure_cuda(
    pair: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        pair[0].cuda() if not pair[0].is_cuda else pair[0],
        pair[1].cuda() if not pair[1].is_cuda else pair[1],
    )


def grouped_gemm_nt_bf16_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
):
    num_groups, _, k = a.shape
    _, n, _ = b.shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        return deep_gemm.m_grouped_bf16_gemm_nt_masked(
            a,
            b,
            d,
            masked_m,
            expected_m,
        )


def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
):
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG

    if m == 0:
        return

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    fp4_kwargs = {}
    if recipe_a is not None:
        fp4_kwargs["recipe_a"] = recipe_a
    if recipe_b is not None:
        fp4_kwargs["recipe_b"] = recipe_b

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            lhs, rhs, out, m_indices, **fp4_kwargs
        )


def grouped_gemm_nt_mxfp8_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
):
    if not supports_sm90_mxfp8_fp8_grouped_gemm():
        raise RuntimeError(
            "The installed deep_gemm does not expose SM90 MXFP8 grouped GEMM APIs."
        )

    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = (
        compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_MXFP8_F8BF16_CONTIG
    )

    if m == 0:
        return

    lhs = _normalize_sm90_mxfp8_pair(_ensure_cuda(lhs), scale_role="lhs")
    rhs = _normalize_sm90_mxfp8_pair(_ensure_cuda(rhs), scale_role="rhs")
    _assert_sm90_mxfp8_scale_matches_recipe(
        lhs[1], k, recipe_a, role="lhs"
    )
    _assert_sm90_mxfp8_scale_matches_recipe(
        rhs[1], k, recipe_b, role="rhs"
    )

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_contiguous(
            lhs,
            rhs,
            out,
            m_indices,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
    _sm90_mxfp8_local_diff_report(
        "entrypoint.py:grouped_gemm_nt_mxfp8_f8f8bf16_contig:local_diff",
        lhs,
        rhs,
        out,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        m_indices=m_indices,
    )


def grouped_gemm_nt_bf16_contig(
    a: torch.Tensor, b: torch.Tensor, d: torch.Tensor, m_indices: torch.Tensor
):
    m, k = a.shape
    num_groups, n, _ = b.shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices)


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.fp8_gemm_nt(
            lhs,
            rhs,
            out,
        )


def gemm_nt_mxfp8_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    """Dense MXFP8 GEMM using fp8_fp4_gemm_nt with recipe_a=(1,32), recipe_b=(1,32)."""
    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    # If both scales are pre-packed int32, skip internal UE8M0 layout transform
    disable_cast = lhs[1].dtype == torch.int and rhs[1].dtype == torch.int

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.fp8_fp4_gemm_nt(
            lhs,
            rhs,
            out,
            recipe_a=(1, 32),
            recipe_b=(1, 32),
            disable_ue8m0_cast=disable_cast,
        )


def gemm_nt_bf16bf16f32(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
):
    m, k = lhs.shape
    n, _ = rhs.shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_BF16BF16F32

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.bf16_gemm_nt(lhs, rhs, out)


def tf32_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_splits: Optional[int],
):
    if x.shape[0] == 0:
        return
    deep_gemm.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_splits=num_splits)


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    # deep_gemm.set_pdl can initialize CUDA state, so run it only after the
    # scheduler/TP worker has been forked and assigned a GPU.
    if envs.SGLANG_DEEPGEMM_PDL.get() and hasattr(deep_gemm, "set_pdl"):
        deep_gemm.set_pdl(True)

    compile_utils.update_deep_gemm_config(gpu_id, server_args)


@contextmanager
def configure_deep_gemm_num_sms(num_sms):
    if num_sms is None or not ENABLE_JIT_DEEPGEMM:
        yield
    else:
        original_num_sms = deep_gemm.get_num_sms()
        deep_gemm.set_num_sms(num_sms)
        try:
            yield
        finally:
            deep_gemm.set_num_sms(original_num_sms)


def _sanity_check_input(x_fp8: Tuple[torch.Tensor, torch.Tensor]):
    if not _SANITY_CHECK:
        return

    x, x_scale = x_fp8

    if x_scale.dtype == torch.int:
        return

    from sglang.srt.layers.quantization.fp8_utils import ceil_to_ue8m0

    x_scale_ceil = ceil_to_ue8m0(x_scale)
    assert torch.all(x_scale == x_scale_ceil), f"{x_scale=} {x_scale_ceil=}"
