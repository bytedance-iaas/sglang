import json
import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _resolve_speculative_algorithm_alias(
    speculative_algorithm: Optional[str],
    speculative_draft_model_path: Optional[str],
    trust_remote_code: bool = False,
) -> Optional[str]:
    """Resolve CLI speculative algorithm; NEXTN/EAGLE may become FROZEN_KV_MTP for Gemma4 assistant drafts."""

    is_gemma4_draft = False
    if speculative_draft_model_path:
        from sglang.srt.utils.hf_transformers_utils import get_config

        cfg = get_config(
            speculative_draft_model_path, trust_remote_code=trust_remote_code
        )
        is_gemma4_draft = "Gemma4AssistantForCausalLM" in (
            getattr(cfg, "architectures", None) or []
        )

    if speculative_algorithm == "EAGLE3" and is_gemma4_draft:
        raise ValueError(
            "Gemma4AssistantForCausalLM draft requires "
            "--speculative-algorithm NEXTN or EAGLE; EAGLE3 is "
            "not supported for this draft architecture."
        )

    if speculative_algorithm == "NEXTN" or speculative_algorithm == "EAGLE":
        if is_gemma4_draft:
            logger.info(
                "Detected Gemma4AssistantForCausalLM draft; "
                f"promoting --speculative-algorithm {speculative_algorithm} to FROZEN_KV_MTP."
            )
            return "FROZEN_KV_MTP"
        return "EAGLE"

    return speculative_algorithm


def handle_speculative_decoding(server_args: "ServerArgs") -> None:
    if (
        server_args.speculative_draft_model_path is not None
        and server_args.speculative_draft_model_revision is None
    ):
        server_args.speculative_draft_model_revision = "main"

    if server_args.speculative_moe_runner_backend is None:
        server_args.speculative_moe_runner_backend = server_args.moe_runner_backend

    if server_args.speculative_algorithm is not None:
        server_args.speculative_algorithm = server_args.speculative_algorithm.upper()

    server_args.speculative_algorithm = _resolve_speculative_algorithm_alias(
        server_args.speculative_algorithm,
        server_args.speculative_draft_model_path,
        trust_remote_code=server_args.trust_remote_code,
    )
    effective_algorithm = server_args.effective_speculative_algorithm

    # Validate --speculative-draft-window-size once, regardless of algorithm.
    # Consumed by DFLASH (compact draft KV cache) and Llama EAGLE-3 (drafter attention SWA).
    if server_args.speculative_draft_window_size is not None:
        window_size = int(server_args.speculative_draft_window_size)
        if window_size <= 0:
            raise ValueError(
                f"--speculative-draft-window-size must be positive, got {window_size}."
            )
        server_args.speculative_draft_window_size = window_size
        if effective_algorithm not in ("EAGLE3", "DFLASH"):
            logger.warning(
                "--speculative-draft-window-size has no effect with "
                "speculative_algorithm=%s (honored by Llama EAGLE-3 and DFLASH only).",
                effective_algorithm,
            )

    if effective_algorithm is not None:
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.speculative.spec_registry import CustomSpecAlgo

        algo = SpeculativeAlgorithm.from_string(effective_algorithm)

        # TODO: move the per-algorithm validation below into spec module hooks.
        if isinstance(algo, CustomSpecAlgo) and algo.validate_server_args is not None:
            algo.validate_server_args(server_args)

    if server_args.speculative_skip_dp_mlp_sync:
        assert server_args.speculative_algorithm == "EAGLE", (
            "--speculative-skip-dp-mlp-sync is only supported with "
            f"speculative_algorithm == EAGLE, got {server_args.speculative_algorithm}."
        )

    if effective_algorithm == "DFLASH":
        _handle_dflash(server_args)
    elif effective_algorithm == "DSPARK":
        _handle_dspark(server_args)
    elif effective_algorithm == "FROZEN_KV_MTP":
        _handle_frozen_kv_mtp(server_args)
    elif effective_algorithm in ("EAGLE", "EAGLE3", "STANDALONE"):
        _handle_eagle_family(server_args)
    elif effective_algorithm == "NGRAM":
        _handle_ngram(server_args)

    if server_args.speculative_adaptive:
        _maybe_disable_adaptive(server_args)


def _handle_dflash(server_args: "ServerArgs") -> None:
    if server_args.enable_dp_attention:
        raise ValueError(
            "Currently DFLASH speculative decoding does not support dp attention."
        )

    if server_args.pp_size != 1:
        raise ValueError(
            "Currently DFLASH speculative decoding only supports pp_size == 1."
        )

    if server_args.speculative_draft_model_path is None:
        raise ValueError(
            "DFLASH speculative decoding requires setting --speculative-draft-model-path."
        )

    # DFLASH does not use EAGLE-style `num_steps`/`topk`, but those fields still
    # affect generic scheduler/KV-cache accounting (buffer sizing, KV freeing,
    # RoPE reservation). Force them to 1 to avoid surprising memory behavior.
    #
    # For DFlash, the natural unit is `block_size` (verify window length).
    if server_args.speculative_num_steps is None:
        server_args.speculative_num_steps = 1
    elif int(server_args.speculative_num_steps) != 1:
        logger.warning(
            "DFLASH only supports speculative_num_steps == 1; overriding speculative_num_steps=%s to 1.",
            server_args.speculative_num_steps,
        )
        server_args.speculative_num_steps = 1

    if server_args.speculative_eagle_topk is None:
        server_args.speculative_eagle_topk = 1
    elif int(server_args.speculative_eagle_topk) != 1:
        logger.warning(
            "DFLASH only supports speculative_eagle_topk == 1; overriding speculative_eagle_topk=%s to 1.",
            server_args.speculative_eagle_topk,
        )
        server_args.speculative_eagle_topk = 1

    if server_args.speculative_dflash_block_size is not None:
        if int(server_args.speculative_dflash_block_size) <= 0:
            raise ValueError(
                "DFLASH requires --speculative-dflash-block-size to be positive, "
                f"got {server_args.speculative_dflash_block_size}."
            )
        if server_args.speculative_num_draft_tokens is not None and int(
            server_args.speculative_num_draft_tokens
        ) != int(server_args.speculative_dflash_block_size):
            raise ValueError(
                "Both --speculative-num-draft-tokens and --speculative-dflash-block-size are set "
                "but they differ. For DFLASH they must match. "
                f"speculative_num_draft_tokens={server_args.speculative_num_draft_tokens}, "
                f"speculative_dflash_block_size={server_args.speculative_dflash_block_size}."
            )
        server_args.speculative_num_draft_tokens = int(
            server_args.speculative_dflash_block_size
        )

    if server_args.speculative_num_draft_tokens is None:
        from sglang.srt.speculative.dflash_utils import (
            parse_dflash_draft_config,
        )

        model_override_args = json.loads(server_args.json_model_override_args)
        inferred_block_size = None
        try:
            from sglang.srt.utils.hf_transformers_utils import get_config

            draft_hf_config = get_config(
                server_args.speculative_draft_model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.speculative_draft_model_revision,
                model_override_args=model_override_args,
            )
            inferred_block_size = parse_dflash_draft_config(
                draft_hf_config=draft_hf_config
            ).resolve_block_size(default=None)
        except Exception as e:
            logger.warning(
                "Failed to infer DFLASH block_size from draft model config; "
                "defaulting speculative_num_draft_tokens to 16. Error: %s",
                e,
            )

        if inferred_block_size is None:
            inferred_block_size = 16
            logger.warning(
                "speculative_num_draft_tokens is not set; defaulting to %d for DFLASH.",
                inferred_block_size,
            )
        server_args.speculative_num_draft_tokens = inferred_block_size

    if server_args.speculative_draft_window_size is not None:
        draft_tokens = int(server_args.speculative_num_draft_tokens)
        if server_args.speculative_draft_window_size < draft_tokens:
            raise ValueError(
                "--speculative-draft-window-size must be >= "
                "--speculative-num-draft-tokens (block_size). "
                f"window_size={server_args.speculative_draft_window_size}, block_size={draft_tokens}."
            )

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    server_args.disable_overlap_schedule = True
    logger.warning(
        "Overlap scheduler is disabled when using DFLASH speculative decoding (spec v2 is not supported yet)."
    )

    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled because of using dflash speculative decoding."
        )


def _target_checkpoint_bundles_dspark_draft(server_args: "ServerArgs") -> bool:
    from sglang.srt.speculative.dspark_components.dspark_config import (
        checkpoint_bundles_dspark_draft,
    )

    return checkpoint_bundles_dspark_draft(server_args.get_model_config().hf_config)


def _handle_dspark(server_args: "ServerArgs") -> None:
    if not server_args.device.startswith("cuda"):
        raise ValueError("DSpark speculative decoding only supports CUDA device.")

    if server_args.disaggregation_mode not in ("prefill", "decode"):
        raise ValueError(
            "This DeepSeek-V4 DSpark backport supports PD disaggregation only."
        )
    if server_args.disaggregation_transfer_backend != "mooncake":
        raise ValueError(
            "DeepSeek-V4 DSpark PD currently supports Mooncake only; "
            f"got {server_args.disaggregation_transfer_backend!r}."
        )
    if server_args.attn_cp_size != 1:
        raise ValueError(
            "DeepSeek-V4 DSpark does not support context parallel in this "
            f"backport (attn_cp_size={server_args.attn_cp_size})."
        )
    if server_args.disaggregation_mode == "decode" and server_args.pp_size != 1:
        raise ValueError("DeepSeek-V4 DSpark decode requires pp_size == 1.")
    if server_args.moe_a2a_backend != "none":
        raise ValueError(
            "DeepSeek-V4 DSpark currently requires --moe-a2a-backend none."
        )
    if server_args.speculative_moe_a2a_backend not in (None, "none"):
        raise ValueError(
            "DeepSeek-V4 DSpark does not support a separate MoE A2A backend."
        )
    if server_args.enable_eic_cache:
        raise ValueError("DeepSeek-V4 DSpark does not support EIC in this backport.")
    if server_args.enable_hierarchical_cache:
        if server_args.disaggregation_mode != "prefill":
            raise ValueError("DSpark HiCache L2 is supported on the prefill node only.")
        if server_args.hicache_storage_backend is not None:
            raise ValueError("DeepSeek-V4 DSpark supports HiCache L2 only, not L3.")
        if server_args.hicache_write_policy != "write_through":
            raise ValueError(
                "DeepSeek-V4 DSpark HiCache requires write_through policy."
            )

    if not server_args.disable_overlap_schedule:
        server_args.disable_overlap_schedule = True
        logger.warning("Overlap scheduling is disabled for DSpark.")

    if server_args.enable_dp_attention:
        if not server_args.enable_dp_lm_head:
            raise ValueError("DSpark with dp attention requires --enable-dp-lm-head.")
        if server_args.moe_a2a_backend != "none":
            raise ValueError(
                "DSpark with dp attention only supports the built-in TP MoE "
                f"(moe_a2a_backend='none'), got {server_args.moe_a2a_backend!r}."
            )
        if (
            server_args.speculative_moe_a2a_backend is not None
            and server_args.speculative_moe_a2a_backend != server_args.moe_a2a_backend
        ):
            raise ValueError(
                "DSpark ignores --speculative-moe-a2a-backend; with dp attention it "
                f"must match the target moe_a2a_backend={server_args.moe_a2a_backend!r} "
                f"(got {server_args.speculative_moe_a2a_backend!r})."
            )

    bundles_dspark = _target_checkpoint_bundles_dspark_draft(server_args)
    if not bundles_dspark:
        raise ValueError(
            "This backport supports only the DSpark draft bundled in the "
            "DeepSeek-V4-Flash-0731 checkpoint."
        )
    if server_args.speculative_draft_model_path is None:
        if bundles_dspark:
            server_args.speculative_draft_model_path = server_args.model_path
            server_args.speculative_draft_model_revision = server_args.revision
            logger.info(
                "DSpark draft weights are bundled in the target checkpoint; "
                "defaulting --speculative-draft-model-path to --model-path (%s).",
                server_args.model_path,
            )
    elif server_args.speculative_draft_model_path != server_args.model_path:
        raise ValueError(
            "Bundled DeepSeek-V4 DSpark must use the target checkpoint; do not "
            "set a different --speculative-draft-model-path."
        )
    elif server_args.speculative_draft_model_revision not in (
        None,
        server_args.revision,
    ):
        raise ValueError(
            "Bundled DeepSeek-V4 DSpark draft and target revisions must match."
        )

    if server_args.speculative_num_steps is None:
        server_args.speculative_num_steps = 1
    elif int(server_args.speculative_num_steps) != 1:
        logger.warning(
            "DSpark only supports speculative_num_steps == 1; overriding speculative_num_steps=%s to 1.",
            server_args.speculative_num_steps,
        )
        server_args.speculative_num_steps = 1

    if server_args.speculative_eagle_topk is None:
        server_args.speculative_eagle_topk = 1
    elif int(server_args.speculative_eagle_topk) != 1:
        logger.warning(
            "DSpark only supports speculative_eagle_topk == 1; overriding speculative_eagle_topk=%s to 1.",
            server_args.speculative_eagle_topk,
        )
        server_args.speculative_eagle_topk = 1

    gamma: Optional[int] = None
    if server_args.speculative_dspark_block_size is not None:
        if int(server_args.speculative_dspark_block_size) <= 0:
            raise ValueError(
                "DSpark requires --speculative-dspark-block-size to be positive, "
                f"got {server_args.speculative_dspark_block_size}."
            )
        gamma = int(server_args.speculative_dspark_block_size)
    else:
        from sglang.srt.speculative.dspark_components.dspark_config import (
            DEFAULT_DSPARK_GAMMA,
            read_draft_checkpoint_gamma,
        )

        try:
            gamma = read_draft_checkpoint_gamma(server_args=server_args)
        except Exception as e:
            logger.warning(
                "Failed to read DSpark gamma from draft model config; "
                "cannot cross-check --speculative-num-draft-tokens. Error: %s",
                e,
            )
        if gamma is None and server_args.speculative_num_draft_tokens is None:
            gamma = DEFAULT_DSPARK_GAMMA
            logger.warning(
                "DSpark gamma is not set; defaulting to %d.",
                gamma,
            )

    if gamma is not None:
        verify_window = int(gamma) + 1
        if (
            server_args.speculative_num_draft_tokens is not None
            and int(server_args.speculative_num_draft_tokens) != verify_window
        ):
            raise ValueError(
                "DSpark speculative_num_draft_tokens must equal gamma + 1 "
                f"(= {verify_window} for gamma={gamma}), but got "
                f"speculative_num_draft_tokens={server_args.speculative_num_draft_tokens}."
            )
        server_args.speculative_num_draft_tokens = verify_window

    if server_args.speculative_num_draft_tokens is None:
        raise ValueError(
            "DSpark could not resolve speculative_num_draft_tokens; set "
            "--speculative-dspark-block-size (= gamma)."
        )
    if int(server_args.speculative_num_draft_tokens) < 2:
        raise ValueError(
            "DSpark speculative_num_draft_tokens must be >= 2 (= gamma + 1), "
            f"got {server_args.speculative_num_draft_tokens}."
        )

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled because of using dspark speculative decoding."
        )

    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )

    ragged_mode = read_ragged_verify_mode()
    if ragged_mode is not RaggedVerifyMode.STATIC:
        raise ValueError(
            "DeepSeek-V4 DSpark currently requires "
            "SGLANG_RAGGED_VERIFY_MODE=static."
        )
    if (
        server_args.speculative_dspark_align_verify_tokens_to_graph_tier
        and ragged_mode is not RaggedVerifyMode.COMPACT
    ):
        logger.warning(
            "--speculative-dspark-align-verify-tokens-to-graph-tier only takes "
            "effect with SGLANG_RAGGED_VERIFY_MODE=compact (got %r); it will be "
            "a no-op.",
            ragged_mode.value,
        )
    if (
        server_args.speculative_dspark_sps_table_path
        and ragged_mode is RaggedVerifyMode.STATIC
    ):
        logger.warning(
            "--speculative-dspark-sps-table-path feeds the ragged-verify budget "
            "scheduler, which is off under SGLANG_RAGGED_VERIFY_MODE=static; it "
            "will be a no-op."
        )


def _handle_frozen_kv_mtp(server_args: "ServerArgs") -> None:
    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    server_args.disable_overlap_schedule = True
    logger.warning(
        "Overlap scheduler is disabled when using Frozen-KV MTP speculative decoding (spec v2 is not supported yet)."
    )

    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled because of using "
            "Frozen-KV MTP speculative decoding."
        )


def _handle_eagle_family(server_args: "ServerArgs") -> None:
    if (
        server_args.speculative_algorithm == "STANDALONE"
        and server_args.enable_dp_attention
    ):
        # TODO: support dp attention for standalone speculative decoding
        raise ValueError(
            "Currently standalone speculative decoding does not support dp attention."
        )

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    spec_v1_reason = None
    if (
        server_args.speculative_eagle_topk is not None
        and server_args.speculative_eagle_topk > 1
        and not server_args.disable_overlap_schedule
    ):
        server_args.disable_overlap_schedule = True
        spec_v1_reason = "spec v2 currently only supports topk = 1"
    elif (
        not envs.SGLANG_ENABLE_SPEC_V2.get()
        and not server_args.disable_overlap_schedule
    ):
        server_args.disable_overlap_schedule = True
        spec_v1_reason = "SGLANG_ENABLE_SPEC_V2=False"

    if server_args.disable_overlap_schedule:
        logger.warning(
            "Spec v1 is used for eagle/eagle3/standalone speculative decoding because %s.",
            spec_v1_reason or "overlap schedule is disabled",
        )
    else:
        logger.warning(
            "Spec v2 is enabled by default for eagle/eagle3/standalone speculative decoding."
        )

    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled because of using "
            "eagle speculative decoding."
        )

    model_arch = server_args.get_model_config().hf_config.architectures[0]
    if model_arch in [
        "DeepseekV32ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV4ForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "GlmMoeDsaForCausalLM",
        "BailingMoeForCausalLM",
        "BailingMoeV2ForCausalLM",
        "BailingMoeV2_5ForCausalLM",
        "MistralLarge3ForCausalLM",
        "PixtralForConditionalGeneration",
        "HYV3ForCausalLM",
    ]:
        if server_args.speculative_draft_model_path is None:
            server_args.speculative_draft_model_path = server_args.model_path
            server_args.speculative_draft_model_revision = server_args.revision
        else:
            if model_arch not in [
                "MistralLarge3ForCausalLM",
                "PixtralForConditionalGeneration",
            ]:
                logger.warning(
                    "DeepSeek MTP does not require setting speculative_draft_model_path."
                )

    if server_args.speculative_num_steps is None:
        assert (
            server_args.speculative_eagle_topk is None
            and server_args.speculative_num_draft_tokens is None
        )
        from sglang.srt.server_args import auto_choose_speculative_params

        (
            server_args.speculative_num_steps,
            server_args.speculative_eagle_topk,
            server_args.speculative_num_draft_tokens,
        ) = auto_choose_speculative_params(server_args)

    if (
        server_args.attention_backend == "trtllm_mha"
        or server_args.decode_attention_backend == "trtllm_mha"
        or server_args.prefill_attention_backend == "trtllm_mha"
    ):
        if server_args.speculative_eagle_topk > 1:
            raise ValueError(
                "trtllm_mha backend only supports topk = 1 for speculative decoding."
            )

    if (
        server_args.speculative_eagle_topk == 1
        and server_args.speculative_num_draft_tokens
        != server_args.speculative_num_steps + 1
    ):
        logger.warning(
            "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 when speculative_eagle_topk == 1"
        )
        server_args.speculative_num_draft_tokens = server_args.speculative_num_steps + 1

    if (
        server_args.speculative_eagle_topk > 1
        and server_args.page_size > 1
        and server_args.attention_backend not in ["flashinfer", "fa3"]
    ):
        raise ValueError(
            "speculative_eagle_topk > 1 with page_size > 1 is unstable and produces incorrect results for paged attention backends. This combination is only supported for the 'flashinfer' backend."
        )


def _handle_ngram(server_args: "ServerArgs") -> None:
    if not server_args.device.startswith("cuda"):
        raise ValueError("Ngram speculative decoding only supports CUDA device.")

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    server_args.disable_overlap_schedule = True
    server_args.enable_mixed_chunk = False
    server_args.speculative_eagle_topk = server_args.speculative_ngram_max_bfs_breadth
    if server_args.speculative_num_draft_tokens is None:
        server_args.speculative_num_draft_tokens = 12
        logger.warning(
            "speculative_num_draft_tokens is set to 12 by default for ngram speculative decoding. "
            "You can override this by explicitly setting --speculative-num-draft-tokens."
        )
    if server_args.speculative_ngram_external_corpus_path is not None:
        if server_args.speculative_ngram_external_sam_budget <= 0:
            raise ValueError(
                "--speculative-ngram-external-sam-budget must be positive when "
                "--speculative-ngram-external-corpus-path is set."
            )
        if server_args.speculative_ngram_external_corpus_max_tokens <= 0:
            raise ValueError(
                "--speculative-ngram-external-corpus-max-tokens must be positive when "
                "--speculative-ngram-external-corpus-path is set."
            )
        if (
            server_args.speculative_ngram_external_sam_budget
            > server_args.speculative_num_draft_tokens - 1
        ):
            raise ValueError(
                "speculative_ngram_external_sam_budget must be less than or equal to "
                f"speculative_num_draft_tokens - 1 ({server_args.speculative_num_draft_tokens - 1})."
            )
    logger.warning(
        "The overlap scheduler and mixed chunked prefill are disabled because of "
        "using ngram speculative decoding."
    )

    if (
        server_args.speculative_eagle_topk > 1
        and server_args.page_size > 1
        and server_args.attention_backend != "flashinfer"
    ):
        raise ValueError(
            f"speculative_eagle_topk({server_args.speculative_eagle_topk}) > 1 "
            f"with page_size({server_args.page_size}) > 1 is unstable "
            "and produces incorrect results for paged attention backends. "
            "This combination is only supported for the 'flashinfer' backend."
        )
    if server_args.enable_dp_attention:
        # TODO: support dp attention for ngram speculative decoding
        raise ValueError(
            "Currently ngram speculative decoding does not support dp attention."
        )


def _maybe_disable_adaptive(server_args: "ServerArgs") -> None:
    from sglang.srt.speculative.adaptive_spec_params import (
        adaptive_unsupported_reason,
    )

    reason = adaptive_unsupported_reason(server_args)
    if reason is not None:
        logger.warning(
            f"speculative_adaptive disabled: {reason}. "
            "Falling back to static speculative params."
        )
        server_args.speculative_adaptive = False
