"""Compatibility between sgl-deep-gemm and upstream DeepGEMM APIs."""


def get_masked_fp8_gemm(deep_gemm, *, require_overlap=False):
    # The packaged fork also supports the overlap signal used by TBO. Upstream
    # 2.8 renamed the ordinary API but does not implement that signal contract.
    legacy = getattr(deep_gemm, "fp8_m_grouped_gemm_nt_masked", None)
    if legacy is not None:
        return legacy
    if require_overlap:
        raise RuntimeError(
            "DeepGEMM masked GEMM overlap requires the sgl-deep-gemm signal API"
        )
    return deep_gemm.m_grouped_fp8_gemm_nt_masked
