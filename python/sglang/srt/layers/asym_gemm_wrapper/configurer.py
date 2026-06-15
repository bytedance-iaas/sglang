import logging

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, is_blackwell_supported

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

# The only architecture distinction sglang needs: Blackwell packs scales as
# UE8M0. Kernel selection (SM89/SM90 vs Blackwell) happens inside AsymGEMM's
# dispatch based on the actual GPU; all archs consume 1x128 / 128x128 block
# scales.
ASYMGEMM_BLACKWELL = ENABLE_JIT_ASYMGEMM and is_blackwell_supported()
ASYMGEMM_SCALE_UE8M0 = ASYMGEMM_BLACKWELL
