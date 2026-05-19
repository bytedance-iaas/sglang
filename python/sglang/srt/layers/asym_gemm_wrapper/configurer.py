import logging
import os

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, is_blackwell_supported, is_sm89_supported

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
