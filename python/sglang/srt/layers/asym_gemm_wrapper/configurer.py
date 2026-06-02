import logging
import os

from sglang.srt.environ import envs
from sglang.srt.utils import (
    get_device_sm,
    is_blackwell_supported,
    is_sm89_supported,
    is_sm90_supported,
)

logger = logging.getLogger(__name__)


def _compute_enable_asym_gemm():
    sm_version = get_device_sm()
    if sm_version < 89:
        return False

    try:
        import asym_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_ASYMGEMM.get()


ENABLE_JIT_ASYMGEMM = _compute_enable_asym_gemm()

ASYMGEMM_BLACKWELL = ENABLE_JIT_ASYMGEMM and is_blackwell_supported()
ASYMGEMM_SCALE_UE8M0 = ASYMGEMM_BLACKWELL
ASYMGEMM_SM89 = ENABLE_JIT_ASYMGEMM and (
    (is_sm89_supported() and not ASYMGEMM_BLACKWELL)
    or os.environ.get("SGLANG_ASYMGEMM_SM89", "0") == "1"
)
# SM90 (Hopper / H20). The TMA/UMMA `*_asym_gemm_nt_*` kernels are compiled for
# SM100 (Blackwell) only in the asym_gemm build, so Hopper cannot use them and
# instead shares the native FP8 grouped-GEMM kernels with Ada (SM89).
ASYMGEMM_SM90 = ENABLE_JIT_ASYMGEMM and (
    (is_sm90_supported() and not ASYMGEMM_BLACKWELL and not ASYMGEMM_SM89)
    or os.environ.get("SGLANG_ASYMGEMM_SM90", "0") == "1"
)

# Native FP8 grouped-GEMM path (`m_grouped_fp8_asym_gemm_sm89[_masked]`). These
# kernels are JIT-compiled for the running architecture, so they run on both Ada
# (SM89) and Hopper (SM90/H20). Blackwell (SM100) instead uses the TMA/UMMA
# `*_asym_gemm_nt_*` kernels. The MoE runner detects the GPU architecture through
# this flag and chooses the matching asym_gemm API.
ASYMGEMM_NATIVE_FP8 = ENABLE_JIT_ASYMGEMM and (ASYMGEMM_SM89 or ASYMGEMM_SM90)
