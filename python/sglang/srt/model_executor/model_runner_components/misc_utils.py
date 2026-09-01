from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.configs.model_config import (
    dsa_layer_skips_topk,
    is_deepseek_dsa,
    is_kimi_k3,
)
from sglang.srt.runtime_context import (
    attention_backends,
    get_context,
    get_observability,
    get_schedule,
)
from sglang.srt.server_args import CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


def _resolve_pp_transformer(model):
    """Find the PP-partitioned transformer behind common model wrappers."""
    pending = [model]
    visited = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if (
            getattr(candidate, "layers", None) is not None
            and getattr(candidate, "start_layer", None) is not None
            and getattr(candidate, "end_layer", None) is not None
        ):
            return candidate
        pending.extend(
            getattr(candidate, attr, None) for attr in ("model", "language_model")
        )
    return None


def _pp_output_boundary_is_scattered(model) -> bool:
    transformer = _resolve_pp_transformer(model)
    if transformer is None or transformer.start_layer >= transformer.end_layer:
        return False
    layer_id = transformer.end_layer - 1
    modes = getattr(transformer.layers[layer_id], "layer_scatter_modes", None)
    mode = getattr(modes, "layer_output_mode", None)
    return getattr(mode, "name", None) == "SCATTERED"


def get_pp_proxy_tensor_ownership(model) -> frozenset[str]:
    """Return keys owned by this PP lane at its outgoing boundary.

    A model can explicitly declare auxiliary lane-local proxy tensors via
    ``pp_proxy_tensors_all_gather_exclude``. LayerScatterModes additionally
    determines whether the standard hidden states and residual are lane-local.
    """
    send_whole_keys = set(
        getattr(model, "pp_proxy_tensors_all_gather_exclude", None) or ()
    )
    if _pp_output_boundary_is_scattered(model):
        send_whole_keys.update(("hidden_states", "residual"))
    return frozenset(send_whole_keys)


def maybe_disable_chunked_prefix_cache(
    *, use_mla_backend: bool, is_draft_worker: bool
) -> None:
    # Chunked prefix caching requires an MLA model on a backend whose
    # kernels read that layout. This is a load-time gate, not a
    # resolution-time one: out-of-tree platforms register their supported
    # backends in init_backend(), which runs when this module is imported
    # — after ServerArgs.__post_init__. Target runner only: a draft
    # model's (often non-MLA) config must not flip the shared setting.
    if is_draft_worker:
        return
    # Chunked prefix cache is a prefill feature: the prefill half decides.
    prefill_backend, _ = attention_backends()
    if (
        not use_mla_backend
        or prefill_backend not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
    ):
        if not get_schedule().disable_chunked_prefix_cache:
            get_context().override(
                "model_runner.chunked_prefix_cache_gate",
                disable_chunked_prefix_cache=True,
            )
    if not get_schedule().disable_chunked_prefix_cache:
        logger.info("Chunked prefix cache is turned on.")


def create_msprobe_debugger() -> Optional[Any]:
    if get_observability().msprobe_dump_config is None:
        return None

    try:
        from msprobe.pytorch import PrecisionDebugger, seed_all
    except ImportError:
        logger.warning(
            "Please install msprobe for tensor data dump: pip install mindstudio-probe --pre, "
            "see https://gitcode.com/Ascend/msprobe for details."
        )
        return None

    seed_all(mode=True)
    return PrecisionDebugger(config_path=get_observability().msprobe_dump_config)


def resolve_pp_proxy_topk_size(
    *, model_config: ModelConfig, pp_size: int, pp_rank: int, start_layer: int
) -> Optional[int]:
    hf_config = model_config.hf_text_config
    if (
        pp_size <= 1
        or pp_rank == 0
        or not is_deepseek_dsa(hf_config)
        or not dsa_layer_skips_topk(hf_config, start_layer)
    ):
        return None
    return getattr(hf_config, "index_topk", None)


def resolve_pp_proxy_residual_num_blocks(
    *, model_config: ModelConfig, pp_size: int, pp_rank: int, start_layer: int
) -> Optional[int]:
    """Return the inherited Kimi K3 attention-residual bank width."""
    if pp_size <= 1 or pp_rank == 0 or not is_kimi_k3(model_config.hf_config):
        return None

    block_size = getattr(model_config.hf_text_config, "attn_res_block_size", None)
    if block_size is None:
        return None
    return (start_layer + block_size - 1) // block_size
