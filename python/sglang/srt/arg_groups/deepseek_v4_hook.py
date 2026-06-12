import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def apply_deepseek_v4_defaults(server_args: "ServerArgs", model_arch: str) -> None:
    """Apply DeepSeek V4 model-specific server arg defaults and constraints."""
    from sglang.srt.environ import envs
    from sglang.srt.server_args import ServerArgs

    server_args.attention_backend = "dsv4"
    server_args.page_size = 256
    logger.info(
        f"Use dsv4 attention backend for {model_arch}, setting page_size to 256."
    )

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 256
        logger.warning(
            f"Setting max_running_requests to {server_args.max_running_requests} for {model_arch}."
        )

    if server_args.kv_cache_dtype == "auto":
        server_args.kv_cache_dtype = "fp8_e4m3"
        logger.warning(
            f"Setting KV cache dtype to {server_args.kv_cache_dtype} for {model_arch}."
        )
    assert server_args.kv_cache_dtype in [
        "fp8_e4m3"
    ], f"{server_args.kv_cache_dtype} is not supported for {model_arch}"

    if server_args.speculative_algorithm is not None:
        assert (
            server_args.speculative_algorithm == "EAGLE"
        ), f"Only EAGLE speculative algorithm is supported for {model_arch}"
        assert (
            server_args.speculative_eagle_topk == 1
        ), f"Only EAGLE speculative algorithm with topk == 1 is supported for {model_arch}"

        if not envs.SGLANG_ENABLE_SPEC_V2.get():
            envs.SGLANG_ENABLE_SPEC_V2.set(True)
            logger.warning("Spec v2 is enabled for EAGLE speculative decoding.")

    if server_args.swa_full_tokens_ratio == ServerArgs.swa_full_tokens_ratio:
        server_args.swa_full_tokens_ratio = 0.1
        logger.info(
            f"Setting swa_full_tokens_ratio to {server_args.swa_full_tokens_ratio} for {model_arch}."
        )

    validate_deepseek_v4_dcp(server_args)


def validate_deepseek_v4_dcp(server_args: "ServerArgs") -> None:
    """Validate DeepSeek V4 DCP (decode context parallel) compatibility.

    server_args.py already enforces ``tp_size % dcp_size == 0`` and disables
    piecewise cuda graph for ``dcp_size > 1``. This hook adds the DSV4-specific
    finer-grained checks: attn_tp divisibility and combo warnings for
    HiSparse / online c128.
    """
    if server_args.dcp_size <= 1:
        return

    # attn_tp = tp_size // dp_size; dcp must divide attn_tp so each dcp group
    # consists of attn_tp / dcp_size consecutive tp ranks within a dp shard.
    attn_tp = server_args.tp_size // max(server_args.dp_size, 1)
    if attn_tp % server_args.dcp_size != 0:
        raise ValueError(
            f"DeepSeekV4 DCP requires attn_tp ({attn_tp}) % dcp_size "
            f"({server_args.dcp_size}) == 0; configure tp/dp_size accordingly."
        )

    # Online c128 + DCP: untested combination; warn but allow.
    from sglang.srt.environ import envs

    if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
        logger.warning(
            "DeepSeekV4 DCP + SGLANG_OPT_USE_ONLINE_COMPRESS combo is "
            "experimental; numerical correctness has not been validated."
        )

    # HiSparse + DCP: HiSparse C4 device pool uses index translation that is
    # not yet DCP-aware in the fused C++ kernels; only the Triton fallback
    # write path supports DCP.
    if server_args.enable_hisparse:
        logger.warning(
            "DeepSeekV4 DCP + enable_hisparse falls back to Triton write "
            "path for KV writes; expect throughput regression vs. fused C++."
        )

    logger.info(
        f"DeepSeekV4 DCP enabled: dcp_size={server_args.dcp_size}, "
        f"attn_tp={attn_tp}"
    )


def validate_deepseek_v4_cp(server_args: "ServerArgs") -> None:
    """Validate DeepSeek V4 context-parallel configuration."""
    if not server_args.enable_nsa_prefill_context_parallel:
        return

    if server_args.nsa_prefill_cp_mode != "round-robin-split":
        raise ValueError(
            f"DeepSeekV4 only supports round-robin-split CP mode, "
            f"got {server_args.nsa_prefill_cp_mode}"
        )

    server_args.enable_dp_attention = True
    server_args.moe_dense_tp_size = 1
    server_args.attn_cp_size = server_args.tp_size // server_args.dp_size
    assert (
        server_args.dp_size == 1
    ), "For round-robin split mode, dp attention is not supported."
    assert (
        server_args.tp_size <= 8
    ), "Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues."
    logger.warning(
        f"Enable Context Parallel for DeepSeekV4, "
        f"dp_size={server_args.dp_size}, moe_dense_tp_size={server_args.moe_dense_tp_size}, "
        f"attn_cp_size={server_args.attn_cp_size}, ep_size={server_args.ep_size}, tp_size={server_args.tp_size}"
    )
