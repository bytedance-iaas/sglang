"""Memory pool configurators for profiling and sizing KV cache pools.

Each model architecture has its own configurator that computes pool sizes
from available GPU memory using a unified coeff+bias model:

    available_bytes = max_tokens * coeff + bias
    max_tokens = (available_bytes - bias) / coeff

Two entry points, same core computation:
- calculate_pool_sizes(available_bytes, page_size): profiling path
- calculate_pool_sizes_from_max_tokens(max_tokens, page_size): constraint path
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import (
    get_nsa_index_head_dim,
    is_deepseek_nsa,
    is_deepseek_v4,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.deepseek_v4_memory_pool import get_compress_state_ring_size
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.utils.common import is_float4_e2m1fn_x2


@dataclass
class MemoryPoolConfig:
    """Resolved memory pool config, shared between target and draft workers."""

    max_total_num_tokens: int
    max_running_requests: Optional[int] = None
    full_max_total_num_tokens: Optional[int] = None
    swa_max_total_num_tokens: Optional[int] = None

    # DSV4 compressed-attention pool sizes (target only; draft workers leave at 0).
    c4_max_total_num_tokens: int = 0
    c128_max_total_num_tokens: int = 0
    c4_state_pool_size: int = 0
    c128_state_pool_size: int = 0

    mem_fraction_static: Optional[float] = None

    def __post_init__(self):
        if self.max_total_num_tokens <= 0:
            msg = "Not enough memory. Please try to increase --mem-fraction-static."
            if self.mem_fraction_static is not None:
                msg += f" Current value: mem_fraction_static={self.mem_fraction_static}"
            raise RuntimeError(msg)


if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _get_dsv4_compress_state_dtype_sizes() -> tuple[int, int]:
    dtype_name = envs.SGLANG_DSV4_COMPRESS_STATE_DTYPE.get().strip().lower()
    if dtype_name in ("float32", "fp32"):
        return 4, 4
    if dtype_name in ("bfloat16", "bf16"):
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            raise ValueError(
                "SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16 is not supported when "
                "SGLANG_OPT_USE_ONLINE_COMPRESS=1; online c128 state must stay float32."
            )
        return 2, 2
    raise ValueError(
        "Unsupported SGLANG_DSV4_COMPRESS_STATE_DTYPE="
        f"{dtype_name!r}. Expected one of: float32, fp32, bfloat16, bf16."
    )


class MemoryPoolConfigurator:
    """Base class for memory pool configurators.

    Subclasses compute pool sizes for their architecture via coeff+bias model.
    Both entry points return MemoryPoolConfig (with max_running_requests=None,
    to be filled by the consumer).
    """

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        """Profiling path: compute pool sizes from available bytes."""
        raise NotImplementedError

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        """Constraint path: recalculate pool sizes from a constrained max_tokens."""
        raise NotImplementedError


class DefaultPoolConfigurator(MemoryPoolConfigurator):
    """Configurator for standard models: MHA, MLA, NSA, FP4.

    coeff = cell_size (bytes per token across all layers)
    bias = 0
    """

    def __init__(self, mr: ModelRunner):
        # Determine effective number of layers for KV cache
        if mambaish := mr.mambaish_config:
            effective_layer_ids = [
                i
                for i in mambaish.full_attention_layer_ids
                if mr.start_layer <= i < mr.end_layer
            ]
            num_layers = len(effective_layer_ids)
        else:
            num_layers = mr.num_effective_layers

        self._cell_size = self._compute_cell_size(mr, num_layers)

        # DFLASH: scale cell_size to account for draft model KV cache
        if mr.spec_algorithm.is_dflash() and not mr.is_draft_worker:
            from sglang.srt.speculative.dflash_utils import (
                scale_kv_cell_size_per_token_for_dflash,
            )

            draft_num_layers = mr.dflash_draft_num_layers
            if (
                draft_num_layers is not None
                and int(draft_num_layers) > 0
                and int(num_layers) > 0
            ):
                self._cell_size = scale_kv_cell_size_per_token_for_dflash(
                    target_cell_size_per_token=self._cell_size,
                    target_num_layers=int(num_layers),
                    draft_num_layers=int(draft_num_layers),
                )

    def _compute_cell_size(self, mr: ModelRunner, num_layers: int) -> int:
        """Compute per-token KV cache cost in bytes. Subclasses can override."""
        # args to config cell size
        model_config = mr.model_config
        kv_cache_dtype = mr.kv_cache_dtype

        kv_size = torch._utils._element_size(kv_cache_dtype)
        tp_size = get_attention_tp_size()

        if mr.use_mla_backend:
            cell_size = (
                (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
                * num_layers
                * kv_size
            )
            if is_float4_e2m1fn_x2(kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16
                cell_size = (cell_size // 2) + (
                    (
                        (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
                        // scale_block_size
                    )
                    * num_layers
                    * kv_size
                )

            # Add indexer KV cache overhead for NSA models (DeepSeek V3.2)
            if is_deepseek_nsa(model_config.hf_config):
                index_head_dim = get_nsa_index_head_dim(model_config.hf_config)
                indexer_size_per_token = (
                    index_head_dim
                    + index_head_dim // NSATokenToKVPool.quant_block_size * 4
                )
                element_size = torch._utils._element_size(
                    NSATokenToKVPool.index_k_with_scale_buffer_dtype
                )
                cell_size += indexer_size_per_token * num_layers * element_size
        else:
            cell_size = (
                model_config.get_num_kv_heads(tp_size)
                * (model_config.head_dim + model_config.v_head_dim)
                * num_layers
                * kv_size
            )

            if is_float4_e2m1fn_x2(kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16
                n = model_config.get_num_kv_heads(tp_size)
                k = model_config.head_dim
                cell_size = (cell_size // 2) + (
                    (n * k * num_layers * 2 * kv_size) // scale_block_size
                )

        return cell_size

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        max_total_num_tokens = available_bytes // self._cell_size
        max_total_num_tokens = max_total_num_tokens // page_size * page_size
        return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        max_total_num_tokens = max_total_num_tokens // page_size * page_size
        return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)


class HybridSWAPoolConfigurator(MemoryPoolConfigurator):
    """Configurator for hybrid sliding window attention models (Gemma2, Command-R, MiMo).

    Splits available memory between full attention and SWA pools.
    Does NOT inherit DefaultPoolConfigurator — different coeff model.
    """

    def __init__(self, mr: ModelRunner):
        model_config = mr.model_config
        kv_cache_dtype = mr.kv_cache_dtype
        kv_size = torch._utils._element_size(kv_cache_dtype)
        tp_size = get_attention_tp_size()

        self._full_layers_num = len(model_config.full_attention_layer_ids)
        self._swa_layers_num = len(model_config.swa_attention_layer_ids)
        assert (
            self._swa_layers_num > 0
        ), "Hybrid SWA model must have at least one SWA layer"

        self._swa_full_tokens_ratio = mr.server_args.swa_full_tokens_ratio

        # Full layer per-token memory (bytes)
        self._full_per_token = (
            model_config.get_num_kv_heads(tp_size)
            * (model_config.head_dim + model_config.v_head_dim)
            * kv_size
        )

        # SWA layer per-token memory (bytes)
        self._swa_per_token = (
            model_config.get_swa_num_kv_heads(tp_size)
            * (model_config.swa_head_dim + model_config.swa_v_head_dim)
            * kv_size
        )

        # EAGLE/STANDALONE draft KV pool inherits max_total tokens with its
        # full-attention layers; budget it into the full term.
        self._draft_full_layers_num = 0
        if (
            mr.spec_algorithm.is_eagle() or mr.spec_algorithm.is_standalone()
        ) and not mr.is_draft_worker:
            draft_layers = getattr(mr, "eagle_draft_num_layers", None)
            if draft_layers is not None and int(draft_layers) > 0:
                self._draft_full_layers_num = int(draft_layers)

        # Bytes per token of max_total_num_tokens.
        #
        # Hybrid (full_layers > 0): max_total = full_tokens, so cell_size accounts
        # for both pools: F*nf + r*S*ns (where swa_tokens = full_tokens * r).
        #
        # All-SWA (full_layers == 0): max_total = swa_tokens directly. The ratio
        # is meaningless here -- there is no full pool to relate to, and every
        # token beyond the sliding window can be evicted. So cell_size = S*ns,
        # with no ratio factor applied.
        if self._full_layers_num == 0:
            self._cell_size = (
                self._swa_per_token * self._swa_layers_num
                + self._full_per_token * self._draft_full_layers_num
            )
        else:
            self._cell_size = (
                self._full_per_token
                * (self._full_layers_num + self._draft_full_layers_num)
                + self._swa_full_tokens_ratio
                * self._swa_per_token
                * self._swa_layers_num
            )

    def _solve_pool_sizes(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        """Core computation: split max_total_num_tokens into full/swa pool sizes."""

        def align_page_size(x: int) -> int:
            return (x // page_size) * page_size

        if self._full_layers_num == 0:
            # All-SWA: no full pool, max_total = actual SWA pool size.
            # Ratio is not applied -- see __init__ comment.
            swa_tokens = align_page_size(max_total_num_tokens)
            logger.info(
                f"Use sliding window memory pool (all SWA). "
                f"swa_layer_tokens={swa_tokens}"
            )
            return MemoryPoolConfig(
                max_total_num_tokens=swa_tokens,
                full_max_total_num_tokens=0,
                swa_max_total_num_tokens=swa_tokens,
            )

        # Hybrid: full_tokens = max_total_num_tokens, swa_tokens = full_tokens * ratio
        full_tokens = align_page_size(max_total_num_tokens)
        swa_tokens = align_page_size(int(full_tokens * self._swa_full_tokens_ratio))

        logger.info(
            f"Use sliding window memory pool. "
            f"full_layer_tokens={full_tokens}, swa_layer_tokens={swa_tokens}"
        )

        return MemoryPoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=swa_tokens,
        )

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        max_total_num_tokens = int(available_bytes // self._cell_size)
        return self._solve_pool_sizes(max_total_num_tokens, page_size)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        return self._solve_pool_sizes(max_total_num_tokens, page_size)


@dataclass
class _DSV4PoolSizes:
    full_max_total_num_tokens: int
    swa_max_total_num_tokens: int
    c4_max_total_num_tokens: int
    c128_max_total_num_tokens: int
    c4_state_pool_size: int
    c128_state_pool_size: int


class DSV4PoolConfigurator(MemoryPoolConfigurator):
    """Configurator for DSV4 compressed-attention models.

    Splits available memory across full / swa / c4 / c128 + c4_state / c128_state
    pools. coeff is bytes_per_full_token (inflated by (T+D)/T when speculative
    decode reserves a draft worker, mirroring dflash's cell_size scaling); bias = 0.
    """

    def __init__(self, mr: ModelRunner):
        cfg = mr.model_config
        server_args = mr.server_args
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.indexer_head_dim = cfg.index_head_dim
        self.compression_ratios = cfg.compress_ratios
        self.swa_page_size = cfg.window_size
        self.swa_ratio = server_args.swa_full_tokens_ratio
        self.is_speculative = server_args.speculative_algorithm is not None
        self.online_c128_mtp_max_draft_tokens = (
            getattr(server_args, "max_speculative_num_draft_tokens", None)
            or server_args.speculative_num_draft_tokens
            or 0
        )
        self.enable_hisparse = mr.enable_hisparse
        self.context_len = cfg.context_len
        self.extra_context_len = 4 + int(server_args.speculative_num_draft_tokens or 0)
        self.dp_size = mr.dp_size
        self.disaggregation_mode = server_args.disaggregation_mode
        self.max_running_requests = server_args.max_running_requests
        self.hisparse_top_k = 0
        self.hisparse_device_buffer_size = 0
        if mr.enable_hisparse:
            from sglang.srt.mem_cache.sparsity import (
                parse_hisparse_config,
                resolve_hisparse_top_k,
            )

            hisparse_cfg = parse_hisparse_config(server_args)
            self.c4_shrink_factor = hisparse_cfg.host_to_device_ratio
            self.hisparse_top_k = resolve_hisparse_top_k(
                server_args, cfg.hf_text_config
            )
            self.hisparse_device_buffer_size = hisparse_cfg.device_buffer_size
        else:
            self.c4_shrink_factor = 1
        assert self.c4_shrink_factor >= 1
        if self.c4_shrink_factor > 1:
            logger.info(f"HiSparse c4 host-to-device ratio = {self.c4_shrink_factor}")

        self.c4_ring_size = get_compress_state_ring_size(4, self.is_speculative)
        self.c128_ring_size = get_compress_state_ring_size(128, self.is_speculative)

        self.num_layers_total = len(self.compression_ratios)
        self.num_layers_ca4 = sum(1 for r in self.compression_ratios if r == 4)
        self.num_layers_ca128 = sum(1 for r in self.compression_ratios if r == 128)

        self.bytes_per_full_token = self._get_bytes_per_full_token()
        if self.is_speculative:
            # Reserve memory for the speculative draft worker by inflating
            # per-token bytes by (target+draft)/target. Equivalent to dflash's
            # scale_kv_cell_size_per_token_for_dflash but applied to
            # bytes_per_full_token: tokens = avail / (bpft * (T+D)/T).
            draft_layers = 1
            target_layers = self.num_layers_total
            self.bytes_per_full_token *= (target_layers + draft_layers) / target_layers

        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            allow_experimental_online_c128_mtp = (
                envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get()
                and mr.spec_algorithm.is_eagle()
            )
            assert mr.spec_algorithm.is_none() or allow_experimental_online_c128_mtp, (
                "Online C128 speculative decode requires the experimental "
                "EAGLE topk=1 path gated by "
                "SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=1"
            )
            if allow_experimental_online_c128_mtp:
                assert self.online_c128_mtp_max_draft_tokens > 0, (
                    "SGLANG_EXPERIMENTAL_ONLINE_C128_MTP requires "
                    "speculative_num_draft_tokens to be set."
                )
                logger.warning(
                    "DSV4 compressed attention: experimental online c128 + MTP "
                    "enabled (EAGLE topk=1 only, draft_banks=%d). "
                    "Validate correctness carefully.",
                    self.online_c128_mtp_max_draft_tokens,
                )
            else:
                logger.info(
                    "DSV4 compressed attention: online c128 enabled (ring_size=1)"
                )

    def _get_bytes_per_full_token(self) -> float:
        kv_bytes = self.qk_nope_head_dim + self.qk_rope_head_dim * 2 + 8

        quant_block_size = 128
        indexer_bytes = (
            self.indexer_head_dim + self.indexer_head_dim // quant_block_size * 4
        )

        attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        c4_state_dtype_size, _ = _get_dsv4_compress_state_dtype_sizes()
        c4_state_bytes = 2 * 2 * attn_head_dim * c4_state_dtype_size
        c4_indexer_state_bytes = 2 * 2 * self.indexer_head_dim * c4_state_dtype_size

        c4_state_ratio = self.c4_ring_size / self.swa_page_size
        c4_frac = 1 / (4 * self.c4_shrink_factor)
        bytes_per_full_token = (
            self.swa_ratio * kv_bytes * self.num_layers_total
            + c4_frac * kv_bytes * self.num_layers_ca4
            + 1 / 128 * kv_bytes * self.num_layers_ca128
            + 1 / 4 * indexer_bytes * self.num_layers_ca4
            + self.swa_ratio * c4_state_ratio * c4_state_bytes * self.num_layers_ca4
            + self.swa_ratio
            * c4_state_ratio
            * c4_indexer_state_bytes
            * self.num_layers_ca4
        )
        if self.enable_hisparse:
            # HiSparse maps every C4 logical slot back to a hot-buffer device
            # slot.  That mapping scales with full-token capacity and is
            # allocated after the base DSV4 pools, so include it in the token
            # coefficient instead of over-allocating the token pool and OOMing
            # during coordinator/allocator setup.
            bytes_per_full_token += 8 / 4
            bytes_per_full_token += 1 / max(1, self.swa_page_size)
        return bytes_per_full_token

    def _estimate_hisparse_req_slots(self) -> int:
        max_num_reqs = self.max_running_requests
        if max_num_reqs is not None:
            max_num_reqs = max(1, int(max_num_reqs) // max(1, int(self.dp_size)))
        else:
            # Matches the lower bound used by ModelRunnerKVCacheMixin when the
            # final token capacity is not known yet.
            max_num_reqs = 2048

        pre_alloc_size = 0
        if self.disaggregation_mode == "decode":
            pre_alloc_size = envs.SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS.get()
            if max_num_reqs <= 32:
                pre_alloc_size = max_num_reqs * 2
            elif pre_alloc_size == 0 and self.enable_hisparse:
                pre_alloc_size = max_num_reqs

        return max_num_reqs + int(pre_alloc_size) + 1

    def _estimate_c128_state_pool_size(self) -> int:
        req_slots = self._estimate_hisparse_req_slots()
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            return req_slots
        return req_slots * self.c128_ring_size

    def _estimate_c128_state_fixed_overhead_bytes(self) -> int:
        if self.num_layers_ca128 == 0:
            return 0

        _, c128_state_dtype_size = _get_dsv4_compress_state_dtype_sizes()
        attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        c128_online = envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
        c128_state_bytes = (
            (3 if c128_online else 2) * attn_head_dim * c128_state_dtype_size
        )
        banks = 1
        if (
            c128_online
            and envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get()
            and self.online_c128_mtp_max_draft_tokens > 0
        ):
            banks += self.online_c128_mtp_max_draft_tokens

        return int(
            self._estimate_c128_state_pool_size()
            * banks
            * c128_state_bytes
            * self.num_layers_ca128
        )

    def _estimate_hisparse_fixed_overhead_bytes(self, page_size: int) -> int:
        if not self.enable_hisparse:
            return 0

        req_slots = self._estimate_hisparse_req_slots()
        c4_page_size = max(1, page_size // 4)
        padded_buffer_size = self.hisparse_device_buffer_size + c4_page_size
        max_context_len = self.context_len + self.extra_context_len
        max_compressed_context_len = (max_context_len + 3) // 4
        c4_layers = self.num_layers_ca4

        int64_bytes = 8
        int32_bytes = 4
        int16_bytes = 2
        uint64_bytes = 8

        overhead = 0
        overhead += req_slots * padded_buffer_size * int64_bytes
        overhead += req_slots * (max_compressed_context_len + c4_page_size) * int64_bytes
        overhead += 2 * c4_layers * req_slots * padded_buffer_size * int32_bytes
        overhead += c4_layers * req_slots * self.hisparse_device_buffer_size * int16_bytes
        overhead += self.hisparse_device_buffer_size * int16_bytes
        overhead += padded_buffer_size * int32_bytes
        overhead += 2 * req_slots * self.hisparse_top_k * int32_bytes
        overhead += c4_layers * uint64_bytes
        overhead += (c4_page_size + 1) * int64_bytes
        return int(overhead)

    def _compute_dsv4_sizes(self, full_token: int, page_size: int) -> _DSV4PoolSizes:
        full_token = full_token // page_size * page_size
        swa_tokens = int(full_token * self.swa_ratio) // page_size * page_size
        return _DSV4PoolSizes(
            full_max_total_num_tokens=full_token,
            swa_max_total_num_tokens=swa_tokens,
            c4_max_total_num_tokens=full_token // (4 * self.c4_shrink_factor),
            c128_max_total_num_tokens=full_token // 128,
            c4_state_pool_size=swa_tokens // self.swa_page_size * self.c4_ring_size,
            c128_state_pool_size=self._estimate_c128_state_pool_size(),
        )

    def _to_config(self, sizes: _DSV4PoolSizes) -> MemoryPoolConfig:
        full = sizes.full_max_total_num_tokens
        swa = sizes.swa_max_total_num_tokens
        logger.info(
            f"DSV4 pool sizes: full={full}, swa={swa}, "
            f"c4={sizes.c4_max_total_num_tokens}, "
            f"c128={sizes.c128_max_total_num_tokens}, "
            f"c4_state={sizes.c4_state_pool_size}, "
            f"c128_state={sizes.c128_state_pool_size}"
        )
        return MemoryPoolConfig(
            max_total_num_tokens=full,
            full_max_total_num_tokens=full,
            swa_max_total_num_tokens=swa,
            c4_max_total_num_tokens=sizes.c4_max_total_num_tokens,
            c128_max_total_num_tokens=sizes.c128_max_total_num_tokens,
            c4_state_pool_size=sizes.c4_state_pool_size,
            c128_state_pool_size=sizes.c128_state_pool_size,
        )

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        assert (
            page_size % 128 == 0
        ), "page_size must be multiple of 128 for compressed attention"

        hisparse_fixed_overhead_bytes = self._estimate_hisparse_fixed_overhead_bytes(
            page_size
        )
        c128_state_fixed_overhead_bytes = (
            self._estimate_c128_state_fixed_overhead_bytes()
        )
        fixed_overhead_bytes = (
            hisparse_fixed_overhead_bytes + c128_state_fixed_overhead_bytes
        )
        effective_available_bytes = available_bytes - fixed_overhead_bytes
        if effective_available_bytes <= 0:
            logger.warning(
                "DSV4 fixed overhead exceeds available KV pool bytes: "
                "raw_available_bytes=%.2f GB, hisparse_fixed_overhead=%.2f GB, "
                "c128_state_fixed_overhead=%.2f GB. "
                "Clamping effective pool bytes to zero; lower mem_fraction_static, "
                "max_running_requests, top_k, or device_buffer_size.",
                available_bytes / (1 << 30),
                hisparse_fixed_overhead_bytes / (1 << 30),
                c128_state_fixed_overhead_bytes / (1 << 30),
            )
            effective_available_bytes = 0
        full_token = int(effective_available_bytes / self.bytes_per_full_token)
        sizes = self._compute_dsv4_sizes(full_token, page_size)
        if fixed_overhead_bytes > 0:
            logger.info(
                "DSV4 memory calculation: "
                "bytes_per_full_token=%.2f, raw_available_bytes=%.2f GB, "
                "hisparse_fixed_overhead=%.2f GB, c128_state_fixed_overhead=%.2f GB, "
                "effective_available_bytes=%.2f GB, "
                "full_token=%d, c4_shrink_factor=%s, req_slots=%d, "
                "padded_buffer_size=%d, top_k=%d, device_buffer_size=%d, "
                "online_c128_mtp_max_draft_tokens=%d",
                self.bytes_per_full_token,
                available_bytes / (1 << 30),
                hisparse_fixed_overhead_bytes / (1 << 30),
                c128_state_fixed_overhead_bytes / (1 << 30),
                effective_available_bytes / (1 << 30),
                sizes.full_max_total_num_tokens,
                self.c4_shrink_factor,
                self._estimate_hisparse_req_slots(),
                self.hisparse_device_buffer_size + max(1, page_size // 4),
                self.hisparse_top_k,
                self.hisparse_device_buffer_size,
                self.online_c128_mtp_max_draft_tokens,
            )
        else:
            logger.info(
                f"DSV4 memory calculation: "
                f"bytes_per_full_token={self.bytes_per_full_token:.2f}, "
                f"available_bytes={available_bytes / (1 << 30):.2f} GB, "
                f"full_token={sizes.full_max_total_num_tokens}"
            )
        return self._to_config(sizes)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        assert (
            page_size % 128 == 0
        ), "page_size must be multiple of 128 for compressed attention"
        sizes = self._compute_dsv4_sizes(max_total_num_tokens, page_size)
        return self._to_config(sizes)


def create_memory_pool_configurator(
    mr: ModelRunner,
) -> MemoryPoolConfigurator:
    """Factory: select the right configurator for the model architecture."""
    if is_deepseek_v4(mr.model_config.hf_config) and mr.is_hybrid_swa:
        return DSV4PoolConfigurator(mr)
    if mr.is_hybrid_swa:
        return HybridSWAPoolConfigurator(mr)
    # Future: MambaPoolConfigurator
    return DefaultPoolConfigurator(mr)
