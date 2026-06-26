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

    strict = envs.SGLANG_REQUIRE_MEGAMOE.get()

    def warn_or_raise(message: str, *args) -> None:
        if strict:
            raise ValueError(message % args if args else message)
        logger.warning(message, *args)

    if server_args.moe_runner_backend != "deep_gemm":
        warn_or_raise(
            "DSV4 HiSparse MegaMoE requires --moe-runner-backend deep_gemm. "
            "Got %r. MegaMoE will not be required unless "
            "SGLANG_REQUIRE_MEGAMOE=1.",
            server_args.moe_runner_backend,
        )

    if not envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get():
        warn_or_raise(
            "DSV4 HiSparse is using --moe-a2a-backend megamoe without "
            "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1. This is allowed in fallback "
            "mode, but strict MegaMoE requires the env flag."
        )

    if not envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
        warn_or_raise(
            "DSV4 HiSparse MegaMoE requires SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1 "
            "to avoid the high-pressure DeepGEMM masked-GEMM scratch path. "
            "MegaMoE will not be required unless SGLANG_REQUIRE_MEGAMOE=1."
        )

    if not envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.get():
        warn_or_raise(
            "DSV4 HiSparse MegaMoE requires SGLANG_OPT_USE_JIT_EP_ACTIVATION=1 "
            "when SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1."
        )

    if not envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION.get():
        warn_or_raise(
            "DSV4 HiSparse MegaMoE requires SGLANG_OPT_SWIGLU_CLAMP_FUSION=1 "
            "for the DeepSeek V4 swiglu_limit DeepGEMM path."
        )

    if server_args.speculative_algorithm == "EAGLE":
        if server_args.speculative_moe_a2a_backend != "megamoe":
            warn_or_raise(
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
            warn_or_raise(
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


def _validate_dsv4_hisparse_top_k(server_args: "ServerArgs") -> None:
    """Fail fast for DSV4 HiSparse C4 top-k modes.

    Fixed 512/1024 shapes keep the legacy raw-index path. Other supported
    values require the flexible topk_v2 raw-index path and are capped at 1024.
    """
    from sglang.srt.environ import envs
    from sglang.srt.mem_cache.sparsity import (
        parse_hisparse_config,
        resolve_hisparse_top_k,
    )

    hf_config = server_args.get_model_config().hf_config
    hf_text_config = getattr(hf_config, "text_config", hf_config)
    top_k = int(resolve_hisparse_top_k(server_args, hf_text_config))
    hisparse_config = parse_hisparse_config(server_args)
    model_index_topk = int(getattr(hf_text_config, "index_topk", top_k))

    if top_k <= 0:
        raise ValueError(f"DSV4 HiSparse requires positive top_k, got {top_k}.")
    if (
        getattr(hisparse_config, "top_k_explicit", False)
        and top_k != model_index_topk
        and not envs.SGLANG_DSV4_HISPARSE_ALLOW_TOPK_OVERRIDE.get()
    ):
        raise ValueError(
            "DSV4 HiSparse top_k must match the model index_topk for "
            "precision-equivalent C4 sparse attention. "
            f"hisparse_config.top_k={top_k}, model index_topk={model_index_topk}. "
            "Remove top_k from --hisparse-config or set it to the model value. "
            "Set SGLANG_DSV4_HISPARSE_ALLOW_TOPK_OVERRIDE=1 only for explicit "
            "accuracy/performance experiments."
        )
    if top_k > 1024:
        raise ValueError(
            "DSV4 HiSparse C4 sparse attention currently supports top_k <= 1024; "
            f"got top_k={top_k}. Use top_k=512/1024 for the legacy raw-index path "
            "or a flexible top_k <= 1024 with SGLANG_OPT_USE_TOPK_V2=1."
        )
    if top_k in (512, 1024):
        return
    if not envs.SGLANG_OPT_USE_TOPK_V2.get():
        raise ValueError(
            "DSV4 HiSparse flexible top_k values require "
            f"SGLANG_OPT_USE_TOPK_V2=1, got top_k={top_k}."
        )
    logger.warning(
        "DSV4 HiSparse flexible top_k=%d will use the topk_v2 raw-index path. "
        "Fixed top_k=512/1024 continue to use the legacy raw-index path.",
        top_k,
    )


def _validate_dsv4_hisparse_c4_verify_hot_buffer(server_args: "ServerArgs") -> None:
    """Ensure EAGLE target-verify C4 rows can stay resident until attention.

    Target verify runs several draft-token rows for the same request in one
    forward. HiSparse C4 swap-in returns device slots into a per-request hot
    buffer, so that buffer must be able to hold the union of all C4 top-k rows
    consumed by the subsequent attention call. Otherwise later rows can evict
    slots referenced by earlier rows and produce silent correctness loss.
    """
    if server_args.speculative_algorithm != "EAGLE":
        return

    from sglang.srt.mem_cache.sparsity import (
        parse_hisparse_config,
        resolve_hisparse_top_k,
    )

    hf_config = server_args.get_model_config().hf_config
    hf_text_config = getattr(hf_config, "text_config", hf_config)
    hisparse_cfg = parse_hisparse_config(server_args)
    top_k = resolve_hisparse_top_k(server_args, hf_text_config)
    draft_tokens = max(int(server_args.speculative_num_draft_tokens or 1), 1)
    required_device_buffer_size = top_k * draft_tokens

    if hisparse_cfg.device_buffer_size >= required_device_buffer_size:
        return

    recommended_device_buffer_size = 1 << (
        required_device_buffer_size - 1
    ).bit_length()
    raise ValueError(
        "DSV4 HiSparse + EAGLE target verify requires the C4 hot buffer to "
        "hold every draft-token top-k row for one request. Current "
        f"device_buffer_size={hisparse_cfg.device_buffer_size}, top_k={top_k}, "
        f"speculative_num_draft_tokens={draft_tokens}, required minimum is "
        f"{required_device_buffer_size}. Use "
        f"--hisparse-config '{{\"top_k\": {top_k}, "
        f"\"device_buffer_size\": {recommended_device_buffer_size}, "
        "\"host_to_device_ratio\": ...}' or reduce EAGLE draft tokens. "
        "Running with a smaller hot buffer can silently evict C4 slots needed "
        "by earlier target-verify rows and degrade answer quality."
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
        _validate_dsv4_hisparse_top_k(server_args)
        _validate_dsv4_hisparse_c4_verify_hot_buffer(server_args)
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
