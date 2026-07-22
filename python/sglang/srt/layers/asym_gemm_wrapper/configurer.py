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


def _compute_unified_moe_available():
    """Whether the unified CPU(INT8)+GPU(INT8) MoE kernel can be used.

    Requires SGLANG_ASYMGEMM_UNIFIED_MOE=1 plus an asym_gemm build that ships
    the `unified_moe` sub-package (CPU `_cpu_C` extension) and an SM90
    (Hopper) or SM80 (A100) GPU — the INT8 grouped kernels exist for exactly
    these two architectures (`Layer.gpu_backend`). Deliberately independent
    of ENABLE_JIT_ASYMGEMM: that flag gates the FP8/FP4 JIT paths (SM89+),
    while the unified INT8 stack runs on SM80 where those paths never will.

    The CPU bucket needs an INT8 row-major kernel (AMX, AVX512-VNNI, or
    AVX2, as reported by caps()["int8_rm_backend"]). A host without one still gets the
    unified path in GPU-only mode — the runner forces m_cpu=0 and a zero
    prefill fraction (see asym_gemm_unified._gpu_only_host).

    Returns (available, cpu_int8_backend). When the env var is set but a
    prerequisite is missing, logs why and falls back to the existing
    asym_gemm paths.
    """
    if not envs.SGLANG_ASYMGEMM_UNIFIED_MOE.get():
        return False, "none"

    def _fallback(reason: str):
        logger.warning(
            "SGLANG_ASYMGEMM_UNIFIED_MOE=1 but %s — "
            "falling back to the existing asym_gemm MoE path.",
            reason,
        )
        return False, "none"

    try:
        import asym_gemm
    except ImportError as e:
        return _fallback(f"the asym_gemm library is not importable ({e})")

    sm_version = get_device_sm()
    if sm_version not in (80, 90):
        return _fallback(
            "the unified MoE kernel requires an SM90 (Hopper) or "
            f"SM80 (A100) GPU (detected SM{sm_version})"
        )

    if getattr(asym_gemm, "unified_moe", None) is None:
        return _fallback(
            "asym_gemm.unified_moe is unavailable "
            "(the _cpu_C extension did not build)"
        )

    cpu_backend = asym_gemm.unified_moe._C.caps().get("int8_rm_backend", "none")
    if cpu_backend == "none":
        logger.warning(
            "AsymGEMM unified MoE: this host has no CPU INT8 kernel "
            "(no AMX-INT8, AVX512-VNNI, or AVX2) — the CPU bucket is "
            "disabled and all experts run on the GPU."
        )

    logger.info(
        "AsymGEMM unified CPU+GPU MoE kernel enabled "
        "(GPU SM%d, CPU INT8 backend: %s).",
        sm_version,
        cpu_backend,
    )
    return True, cpu_backend


ASYMGEMM_UNIFIED_MOE, ASYMGEMM_UNIFIED_CPU_INT8_BACKEND = (
    _compute_unified_moe_available()
)
