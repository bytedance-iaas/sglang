import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# Backend/dtype pairing: flashmla_sparse only takes BF16 KV;
# flashmla_kv only supports FP8 (it always reads KV as FP8 via
# is_fp8_kvcache=True, inline-quantizing BF16 would defeat HiSparse).
_HISPARSE_ALLOWED_BACKENDS_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv"},
}


def _hisparse_default_backend(kv_cache_dtype: str) -> str:
    return "flashmla_kv" if kv_cache_dtype == "fp8_e4m3" else "flashmla_sparse"


def apply_hisparse_nsa_backend_defaults(
    server_args: "ServerArgs",
    user_set_prefill: bool,
    user_set_decode: bool,
    kv_cache_dtype: str,
) -> bool:
    """Pick NSA backends for --enable-hisparse based on KV dtype.

    BF16 KV -> flashmla_sparse, FP8 KV -> flashmla_kv. Returns True if hisparse
    handled backend selection (caller should skip its own default logic).
    """
    if not server_args.enable_hisparse:
        return False

    backend = _hisparse_default_backend(kv_cache_dtype)
    if not user_set_prefill:
        server_args.nsa_prefill_backend = backend
    if not user_set_decode:
        server_args.nsa_decode_backend = backend
    logger.warning(
        f"HiSparse enabled ({kv_cache_dtype}): using NSA backends "
        f"prefill={server_args.nsa_prefill_backend}, decode={server_args.nsa_decode_backend}."
    )
    return True


def _validate_dsv4_hisparse_megamoe(server_args: "ServerArgs") -> None:
    """Validate the DSV4 HiSparse + MegaMoE combination before graph capture."""
    if server_args.moe_a2a_backend != "megamoe":
        return

    from sglang.srt.environ import envs

    if server_args.moe_runner_backend != "deep_gemm":
        raise ValueError(
            "DSV4 HiSparse MegaMoE requires --moe-runner-backend deep_gemm. "
            f"Got {server_args.moe_runner_backend!r}."
        )

    if not envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get():
        logger.warning(
            "DSV4 HiSparse is using --moe-a2a-backend megamoe without "
            "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1. This is allowed, but the env "
            "flag is the preferred way to keep MegaMoE defaults explicit."
        )

    if not envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
        raise ValueError(
            "DSV4 HiSparse MegaMoE requires SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1 "
            "to avoid the high-pressure DeepGEMM masked-GEMM scratch path."
        )

    if server_args.speculative_algorithm == "EAGLE":
        if server_args.speculative_moe_a2a_backend != "megamoe":
            logger.warning(
                "DSV4 HiSparse target is using MegaMoE, but the EAGLE draft "
                "MoE A2A backend is %r. This is a fallback configuration, not "
                "full MegaMoE draft coverage.",
                server_args.speculative_moe_a2a_backend,
            )

        cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
        draft_tokens = max(int(server_args.speculative_num_draft_tokens or 1), 1)
        graph_bs = (
            0
            if server_args.disable_cuda_graph
            else int(server_args.cuda_graph_max_bs or 0)
        )
        required_cap = graph_bs * draft_tokens
        if required_cap > cap:
            logger.warning(
                "DSV4 HiSparse MegaMoE graph capture may exceed "
                "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK >= "
                f"cuda_graph_max_bs * speculative_num_draft_tokens "
                f"({graph_bs} * {draft_tokens} = {required_cap}), but got {cap}. "
                "MegaMoE will fall back at runtime when the per-rank token cap "
                "is exceeded; raise the env var for full MegaMoE coverage."
            )


def _validate_dsv4_hisparse_online_c128_mtp(server_args: "ServerArgs") -> None:
    """Validate the DSV4 HiSparse + online C128 MTP combination early."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
        return

    if server_args.speculative_algorithm is None:
        return

    if not envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
        raise ValueError(
            "DSV4 HiSparse online C128 with speculative decode requires "
            "SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=1."
        )

    if server_args.speculative_algorithm != "EAGLE":
        raise ValueError(
            "DSV4 HiSparse online C128 MTP is currently only validated for "
            f"EAGLE, got {server_args.speculative_algorithm!r}."
        )

    if server_args.speculative_eagle_topk not in (None, 1):
        raise ValueError(
            "DSV4 HiSparse online C128 MTP requires "
            f"--speculative-eagle-topk 1, got {server_args.speculative_eagle_topk}."
        )

    speculative_num_steps = int(server_args.speculative_num_steps or 0)
    if speculative_num_steps > 2:
        raise ValueError(
            "DSV4 HiSparse online C128 MTP is currently validated only for "
            f"EAGLE step1/step2, got speculative_num_steps={speculative_num_steps}. "
            "Keep step3 disabled until SWA/C4 host/logical admission is validated "
            "under pressure."
        )

    if not envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
        raise ValueError(
            "DSV4 HiSparse online C128 MTP requires "
            "SGLANG_OPT_USE_COMPRESSOR_V2=1 because the v2 compressor carries "
            "online state-slot metadata."
        )

    logger.warning(
        "DSV4 HiSparse online C128 MTP enabled: C4 stays on HiSparse host "
        "mirror; C128 uses online EAGLE state banks; eagle_steps=%d, "
        "draft_tokens=%s.",
        speculative_num_steps,
        server_args.speculative_num_draft_tokens,
    )


def validate_hisparse(server_args: "ServerArgs") -> None:
    """Validate --enable-hisparse constraints (model class, radix cache, NSA backend)."""
    if not server_args.enable_hisparse:
        return

    from sglang.srt.configs.model_config import (
        is_deepseek_nsa,
        is_deepseek_v4,
    )

    hf_config = server_args.get_model_config().hf_config
    is_v4_hisparse = is_deepseek_v4(hf_config)
    assert is_deepseek_nsa(hf_config) or is_v4_hisparse, (
        "--enable-hisparse is only supported for DSA (DeepSeek Sparse Attention) "
        "models (e.g., DeepSeek V3.2, GLM-5) and DeepSeek V4 now. "
    )

    # DSv4 hisparse handles its own dtype/backend pairing elsewhere; the dtype-
    # aware checks below only apply to the DSA hisparse path.  Normal scheduler
    # radix is still not used for DSV4 HiSparse; C4 prefix reuse is handled by
    # HiSparseCoordinator.
    if is_v4_hisparse:
        _validate_dsv4_hisparse_online_c128_mtp(server_args)
        _validate_dsv4_hisparse_megamoe(server_args)
        return

    if server_args.kv_cache_dtype not in ("bfloat16", "auto", "fp8_e4m3"):
        raise ValueError(
            f"HiSparse requires bfloat16 or fp8_e4m3 KV cache, "
            f"but got --kv-cache-dtype={server_args.kv_cache_dtype}. "
            f"Please use --kv-cache-dtype=bfloat16 or fp8_e4m3."
        )

    allowed_backends = _HISPARSE_ALLOWED_BACKENDS_BY_DTYPE.get(
        server_args.kv_cache_dtype, {"flashmla_sparse", "flashmla_kv"}
    )
    for attr, label in [
        ("nsa_prefill_backend", "prefill"),
        ("nsa_decode_backend", "decode"),
    ]:
        backend = getattr(server_args, attr)
        if backend is not None and backend not in allowed_backends:
            raise ValueError(
                f"HiSparse with --kv-cache-dtype={server_args.kv_cache_dtype} requires "
                f"--nsa-{label}-backend in {sorted(allowed_backends)}, "
                f"but got {backend}."
            )
