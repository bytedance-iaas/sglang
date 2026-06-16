import pytest
import torch

from sglang.srt.layers import deep_gemm_wrapper


def _require_sm90_mxfp8_grouped_gemm() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for SM90 MXFP8 grouped GEMM tests")
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        pytest.skip(f"SM90 MXFP8 grouped GEMM tests require sm_90, got sm_{major}x")
    if not deep_gemm_wrapper.supports_sm90_mxfp8_fp8_grouped_gemm():
        pytest.skip("deep_gemm does not expose SM90 MXFP8 grouped GEMM APIs")


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _cast_back_from_fp8_1d(
    x: torch.Tensor, sf: torch.Tensor, gran_k: int
) -> torch.Tensor:
    group_idx = torch.arange(x.size(-1), device=x.device) // gran_k
    return x.float() * sf[..., group_idx]


def _e8m0_from_fp32_pow2(sf: torch.Tensor) -> torch.Tensor:
    return ((sf.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def test_sm90_mxfp8_grouped_contiguous_wrapper_accuracy():
    _require_sm90_mxfp8_grouped_gemm()

    from deep_gemm.utils.math import per_token_cast_to_fp8

    groups, m_per_group, n, k = 2, 128, 48, 128
    m = groups * m_per_group
    a_ref = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a = per_token_cast_to_fp8(a_ref, use_ue8m0=False, gran_k=128)
    b_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf_fp32 = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    for group_id in range(groups):
        b_data[group_id], b_sf_fp32[group_id] = per_token_cast_to_fp8(
            b_ref[group_id], use_ue8m0=True, gran_k=32
        )

    grouped_layout = torch.arange(groups, device="cuda", dtype=torch.int32)
    grouped_layout = grouped_layout.repeat_interleave(m_per_group)
    d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    deep_gemm_wrapper.grouped_gemm_nt_mxfp8_f8f8bf16_contig(
        a, (b_data, _e8m0_from_fp32_pow2(b_sf_fp32)), d, grouped_layout
    )

    a_dequant = _cast_back_from_fp8_1d(a[0], a[1], gran_k=128)
    ref = torch.empty_like(d)
    for group_id in range(groups):
        start = group_id * m_per_group
        end = start + m_per_group
        b_dequant = _cast_back_from_fp8_1d(
            b_data[group_id], b_sf_fp32[group_id], gran_k=32
        )
        ref[start:end] = (a_dequant[start:end] @ b_dequant.t()).to(torch.bfloat16)

    assert _calc_diff(d, ref) < 0.03


def test_sm90_mxfp8_grouped_masked_wrapper_accuracy():
    _require_sm90_mxfp8_grouped_gemm()

    from deep_gemm.utils.math import per_token_cast_to_fp8

    groups, max_m, n, k = 2, 32, 48, 128
    masked_m = torch.tensor([7, 19], device="cuda", dtype=torch.int32)
    a_ref = torch.randn((groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a_data = torch.empty((groups, max_m, k), device="cuda", dtype=torch.float8_e4m3fn)
    a_sf = torch.empty((groups, max_m, 1), device="cuda", dtype=torch.float32)
    b_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf_fp32 = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    for group_id in range(groups):
        a_data[group_id], a_sf[group_id] = per_token_cast_to_fp8(
            a_ref[group_id], use_ue8m0=False, gran_k=128
        )
        b_data[group_id], b_sf_fp32[group_id] = per_token_cast_to_fp8(
            b_ref[group_id], use_ue8m0=True, gran_k=32
        )

    d = torch.empty((groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    deep_gemm_wrapper.grouped_gemm_nt_mxfp8_f8f8bf16_masked(
        (a_data, a_sf),
        (b_data, _e8m0_from_fp32_pow2(b_sf_fp32)),
        d,
        masked_m,
        expected_m=max_m,
    )

    a_dequant = _cast_back_from_fp8_1d(a_data, a_sf, gran_k=128)
    ref = torch.zeros_like(d)
    for group_id, valid_m in enumerate(masked_m.tolist()):
        b_dequant = _cast_back_from_fp8_1d(
            b_data[group_id], b_sf_fp32[group_id], gran_k=32
        )
        ref[group_id, :valid_m] = (
            a_dequant[group_id, :valid_m] @ b_dequant.t()
        ).to(torch.bfloat16)

    diff = max(
        _calc_diff(d[group_id, :valid_m], ref[group_id, :valid_m])
        for group_id, valid_m in enumerate(masked_m.tolist())
    )
    assert diff < 0.03
