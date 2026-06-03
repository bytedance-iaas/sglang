import logging
import os

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm

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


def _compute_blackwell() -> bool:
    """Ask the AsymGEMM library whether the running GPU is Blackwell.

    The arch -> kernel decision lives in the library (see ``asym_gemm.dispatch``),
    so SGLang never inspects raw SM numbers itself. The env knobs force the
    non-Blackwell (SM89/SM90) FP8 path when validating on a different GPU.
    """
    if not ENABLE_JIT_ASYMGEMM:
        return False
    if os.environ.get("SGLANG_ASYMGEMM_SM89", "0") == "1":
        return False
    if os.environ.get("SGLANG_ASYMGEMM_SM90", "0") == "1":
        return False
    import asym_gemm

    return asym_gemm.is_blackwell()


# Blackwell (SM100+) uses the TMA/UMMA `*_asym_gemm_nt_*` kernels with packed
# UE8M0 block scales; Ada (SM89) and Hopper (SM90/H20) use the `*_sm89` FP8
# grouped-GEMM kernels. This is the single capability flag the MoE backend
# branches on — matching DeepGEMM's `DEEPGEMM_BLACKWELL` / `DEEPGEMM_SCALE_UE8M0`.
ASYMGEMM_BLACKWELL = _compute_blackwell()
ASYMGEMM_SCALE_UE8M0 = ASYMGEMM_BLACKWELL
