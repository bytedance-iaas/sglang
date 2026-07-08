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


def _compute_unified_moe_available() -> bool:
    """Whether the unified CPU(AMX INT8)+GPU(SM90 INT8) MoE kernel can be used.

    Requires SGLANG_ASYMGEMM_UNIFIED_MOE=1 plus an asym_gemm build that ships
    the `unified_moe` sub-package (CPU `_cpu_C` extension), an AMX-INT8 host,
    and an SM90 GPU (the INT8 grouped kernel is Hopper-only). When the env var
    is set but a prerequisite is missing, log why and fall back to the
    existing asym_gemm paths.
    """
    if not envs.SGLANG_ASYMGEMM_UNIFIED_MOE.get():
        return False

    def _fallback(reason: str) -> bool:
        logger.warning(
            "SGLANG_ASYMGEMM_UNIFIED_MOE=1 but %s — "
            "falling back to the existing asym_gemm MoE path.",
            reason,
        )
        return False

    if not ENABLE_JIT_ASYMGEMM:
        return _fallback("the asym_gemm library is not enabled/importable")
    if ASYMGEMM_BLACKWELL or get_device_sm() != 90:
        return _fallback(
            "the unified MoE kernel requires an SM90 (Hopper) GPU "
            f"(detected SM{get_device_sm()})"
        )

    import asym_gemm

    if getattr(asym_gemm, "unified_moe", None) is None:
        return _fallback(
            "asym_gemm.unified_moe is unavailable "
            "(the _cpu_C extension did not build)"
        )
    if not asym_gemm.unified_moe._C.caps().get("has_amx_int8", False):
        return _fallback("this host has no AMX-INT8 support")

    logger.info("AsymGEMM unified CPU+GPU MoE kernel enabled.")
    return True


ASYMGEMM_UNIFIED_MOE = _compute_unified_moe_available()
