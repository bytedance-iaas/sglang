import json
import logging
from typing import Optional

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_dsa import DeepSeekDSAAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    DSABackendAdaptor,
    FlashAttentionAdaptor,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
    SparseConfig,
    SparseCoordinator,
)

logger = logging.getLogger(__name__)

_global_sparse_coordinator: Optional[SparseCoordinator] = None

_ALGORITHM_REGISTRY = {
    "quest": lambda config, device, **kw: QuestAlgorithm(config, device, **kw),
    "deepseek_dsa": lambda config, device, **kw: DeepSeekDSAAlgorithm(
        config, device, **kw
    ),
}


def _create_sparse_algorithm(
    config: SparseConfig,
    device: torch.device,
    **kwargs,
) -> BaseSparseAlgorithm:
    algorithm_name = config.algorithm.lower()
    factory = _ALGORITHM_REGISTRY.get(algorithm_name)

    if factory is None:
        raise ValueError(f"Unknown sparse algorithm: {algorithm_name}")

    return factory(config, device, **kwargs)


def _create_backend_adaptor(
    backend: str,
    device: torch.device,
    sparse_algorithm: BaseSparseAlgorithm,
    req_to_token_pool,
):
    """Create backend adaptor."""
    if isinstance(sparse_algorithm, DeepSeekDSAAlgorithm):
        return DSABackendAdaptor(device, req_to_token_pool)

    if backend in ["fa3", "flashattention"]:
        return FlashAttentionAdaptor(device)

    raise ValueError(f"Unknown attention backend: {backend}")


def _parse_sparse_config(server_args) -> SparseConfig:
    """Parse hierarchical sparse config from JSON string.

    Required fields with defaults: top_k (2048), device_buffer_size (2*top_k),
    host_to_device_ratio (2), swap_in_block_size (960).
    Optional fields (default None): algorithm, backend, min_sparse_prompt_len,
    page_size. Dynamic residency is opt-in and uses conservative defaults.
    All remaining fields go to sparse_extra_config.
    """
    extra_config_str = server_args.hisparse_config
    if extra_config_str is not None:
        try:
            extra_config = json.loads(extra_config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse hisparse_config: {e}") from e
    else:
        extra_config = {}

    top_k = extra_config.pop("top_k", 2048)
    device_buffer_size = extra_config.pop("device_buffer_size", 2 * top_k)
    host_to_device_ratio = extra_config.pop("host_to_device_ratio", 2)
    swap_in_block_size = extra_config.pop("swap_in_block_size", 960)

    if device_buffer_size < top_k:
        raise ValueError(
            f"device_buffer_size ({device_buffer_size}) must be no smaller than top_k ({top_k})"
        )
    if not isinstance(swap_in_block_size, int) or isinstance(swap_in_block_size, bool):
        raise ValueError(
            f"swap_in_block_size must be an integer, got {swap_in_block_size!r}"
        )
    if swap_in_block_size <= 0 or swap_in_block_size > 1024:
        raise ValueError(
            f"swap_in_block_size ({swap_in_block_size}) must be in the range [1, 1024]"
        )

    algorithm = extra_config.pop("algorithm", None)
    backend = extra_config.pop("backend", None)
    min_sparse_prompt_len = extra_config.pop("min_sparse_prompt_len", None)
    page_size = extra_config.pop("page_size", None)
    dynamic_residency = extra_config.pop("dynamic_residency", False)
    dynamic_residency_mode = extra_config.pop(
        "dynamic_residency_mode", "adaptive"
    )
    dynamic_residency_max_tokens = extra_config.pop(
        "dynamic_residency_max_tokens", 32768
    )
    dynamic_residency_max_requests = extra_config.pop(
        "dynamic_residency_max_requests", 1
    )
    dynamic_residency_min_remaining_tokens = extra_config.pop(
        "dynamic_residency_min_remaining_tokens", 256
    )
    dynamic_residency_promote_watermark = extra_config.pop(
        "dynamic_residency_promote_watermark", 0.20
    )
    dynamic_residency_demote_watermark = extra_config.pop(
        "dynamic_residency_demote_watermark", 0.10
    )
    dynamic_residency_cooldown_steps = extra_config.pop(
        "dynamic_residency_cooldown_steps", 16
    )
    dynamic_residency_admission_window_seconds = extra_config.pop(
        "dynamic_residency_admission_window_seconds", 1800
    )

    if not isinstance(dynamic_residency, bool):
        raise ValueError(
            f"dynamic_residency must be a boolean, got {dynamic_residency!r}"
        )
    valid_dynamic_residency_modes = {
        "adaptive",
        "forced_resident",
        "forced_host_backed",
        "admission_once",
        "admission_window",
    }
    if dynamic_residency_mode not in valid_dynamic_residency_modes:
        raise ValueError(
            "dynamic_residency_mode must be one of "
            f"{sorted(valid_dynamic_residency_modes)}, got "
            f"{dynamic_residency_mode!r}"
        )
    integer_fields = {
        "dynamic_residency_max_tokens": dynamic_residency_max_tokens,
        "dynamic_residency_max_requests": dynamic_residency_max_requests,
        "dynamic_residency_min_remaining_tokens": (
            dynamic_residency_min_remaining_tokens
        ),
        "dynamic_residency_cooldown_steps": dynamic_residency_cooldown_steps,
        "dynamic_residency_admission_window_seconds": (
            dynamic_residency_admission_window_seconds
        ),
    }
    for field_name, value in integer_fields.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"{field_name} must be a non-negative integer, got {value!r}"
            )
    if dynamic_residency_max_tokens == 0:
        raise ValueError("dynamic_residency_max_tokens must be greater than zero")
    if dynamic_residency_max_requests == 0:
        raise ValueError("dynamic_residency_max_requests must be greater than zero")

    watermark_fields = {
        "dynamic_residency_promote_watermark": dynamic_residency_promote_watermark,
        "dynamic_residency_demote_watermark": dynamic_residency_demote_watermark,
    }
    for field_name, value in watermark_fields.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{field_name} must be in [0, 1], got {value!r}")
    if dynamic_residency_demote_watermark >= dynamic_residency_promote_watermark:
        raise ValueError(
            "dynamic_residency_demote_watermark must be smaller than "
            "dynamic_residency_promote_watermark"
        )

    return SparseConfig(
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        host_to_device_ratio=host_to_device_ratio,
        swap_in_block_size=swap_in_block_size,
        algorithm=algorithm,
        backend=backend,
        page_size=page_size,
        min_sparse_prompt_len=min_sparse_prompt_len,
        dynamic_residency=dynamic_residency,
        dynamic_residency_mode=dynamic_residency_mode,
        dynamic_residency_max_tokens=dynamic_residency_max_tokens,
        dynamic_residency_max_requests=dynamic_residency_max_requests,
        dynamic_residency_min_remaining_tokens=(dynamic_residency_min_remaining_tokens),
        dynamic_residency_promote_watermark=float(dynamic_residency_promote_watermark),
        dynamic_residency_demote_watermark=float(dynamic_residency_demote_watermark),
        dynamic_residency_cooldown_steps=dynamic_residency_cooldown_steps,
        dynamic_residency_admission_window_seconds=(
            dynamic_residency_admission_window_seconds
        ),
        sparse_extra_config=extra_config,
    )


def parse_hisparse_config(server_args) -> SparseConfig:
    """Parse hisparse config from server_args, returning defaults if no config provided."""
    return _parse_sparse_config(server_args)


def create_sparse_coordinator(
    device: torch.device,
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    server_args,
    **kwargs,
) -> SparseCoordinator:
    config = _parse_sparse_config(server_args)
    algorithm = _create_sparse_algorithm(config, device, **kwargs)
    backend_adaptor = _create_backend_adaptor(
        config.backend, device, algorithm, req_to_token_pool
    )

    coordinator = SparseCoordinator(
        config=config,
        algorithm=algorithm,
        backend_adaptor=backend_adaptor,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        start_layer=start_layer,
        end_layer=end_layer,
        device=device,
    )
    register_sparse_coordinator(coordinator)
    return coordinator


def register_sparse_coordinator(coordinator: SparseCoordinator) -> None:
    global _global_sparse_coordinator
    _global_sparse_coordinator = coordinator


def get_sparse_coordinator() -> Optional[SparseCoordinator]:
    return _global_sparse_coordinator
