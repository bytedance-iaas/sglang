from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Literal, NamedTuple, Optional, Tuple

import torch

from sglang.jit_kernel.deepseek_v4 import fused_k_norm_rope_flashmla, fused_store_cache
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4 import (
    index_buf_accessor as dsv4_index_buf_accessor,
)
from sglang.srt.layers.attention.dsv4.index_buf_accessor import NopeFp8RopeBf16Pack
from sglang.srt.layers.attention.nsa import index_buf_accessor
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
    KVAndScore,
)
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.platforms import current_platform
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import ceil_div, is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

ONLINE_C128 = not _is_hip and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()


def get_compress_state_ring_size(
    compress_ratio: int, is_speculative: bool = False
) -> int:
    assert compress_ratio in [4, 128], f"Unsupported {compress_ratio = }"
    # Online c128 keeps a single (max, sum, kv) state per index instead of a
    # 128-slot ring buffer of raw tokens, so ring_size collapses to 1. Online
    # MTP uses extra pending state banks guarded by
    # SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.
    if compress_ratio == 128 and ONLINE_C128:
        if is_speculative and not envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
            raise AssertionError(
                "Online C128 speculative decode requires "
                "SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=1"
            )
        return 1
    if is_speculative:
        return 16 if compress_ratio == 4 else 256
    else:
        return 8 if compress_ratio == 4 else 128


class DeepSeekV4SingleKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim

        self.scale_pad = 1
        self.quantize_block_size = 64
        self.rope_storage_dtype = torch.bfloat16
        self.k_with_scale_buffer_dtype = torch.int8
        self._create_buffers()

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.kv_buffer = [
                    self.create_buffer(
                        num_pages=(self.size + self.page_size + 1) // self.page_size,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_bytes_per_token(self) -> int:
        dim_per_token = (
            self.qk_nope_head_dim
            + self.qk_rope_head_dim * self.rope_storage_dtype.itemsize
            + self.qk_nope_head_dim // self.quantize_block_size
            + self.scale_pad
        )
        return dim_per_token

    def create_buffer(self, *, num_pages: int):
        bytes_per_token = self.get_bytes_per_token()
        self.kv_cache_total_dim = bytes_per_token
        bytes_per_page_non_padded = self.page_size * bytes_per_token
        self.bytes_per_page_padded = ceil_div(bytes_per_page_non_padded, 576) * 576

        assert bytes_per_token == 448 + 64 * 2 + 8, (
            "DSV4 KV layout: qk_nope_head_dim FP8 (448) + qk_rope_head_dim BF16 "
            "(64*2) + nope FP8 scales + scale_pad = 584 bytes/token"
        )
        assert self.store_dtype == torch.uint8

        return torch.zeros(
            num_pages,
            self.bytes_per_page_padded,
            dtype=self.store_dtype,
            device=self.device,
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ):
        dsv4_index_buf_accessor.SetKAndS.execute(
            pool=self,
            buf=self.kv_buffer[layer_id],
            loc=loc,
            nope_fp8_rope_bf16_pack=cache_nope_fp8_rope_bf16_pack,
        )

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache(
            input=cache_k,
            cache=self.kv_buffer[layer_id],
            indices=loc,
            page_size=self.page_size,
            type="flashmla",
        )

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id]

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("Use get_key_buffer instead.")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Use get_key_buffer instead.")

    def _token_byte_indices(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(device=self.device, dtype=torch.int64)
        bytes_per_token = self.get_bytes_per_token()
        byte_offsets = torch.arange(bytes_per_token, device=self.device)
        pages = indices // self.page_size
        offsets = (indices % self.page_size) * bytes_per_token
        return (
            pages[:, None] * self.bytes_per_page_padded
            + offsets[:, None]
            + byte_offsets[None, :]
        ).reshape(-1)

    def get_cpu_copy(self, indices, mamba_indices=None):
        current_platform.synchronize()
        indices = indices.to(device=self.device, dtype=torch.int64)
        bytes_per_token = self.get_bytes_per_token()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            flat_buffer = self.kv_buffer[layer_id].reshape(-1)
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                byte_indices = self._token_byte_indices(chunk_indices)
                kv_cache_cpu[-1].append(
                    flat_buffer[byte_indices]
                    .reshape(len(chunk_indices), bytes_per_token)
                    .to("cpu", non_blocking=True)
                )
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        current_platform.synchronize()
        indices = indices.to(device=self.device, dtype=torch.int64)
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            flat_buffer = self.kv_buffer[layer_id].reshape(-1)
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                cpu_data = kv_cache_cpu[layer_id][i // chunk_size]
                assert cpu_data.shape[0] == len(chunk_indices)
                byte_indices = self._token_byte_indices(chunk_indices)
                flat_buffer[byte_indices] = cpu_data.to(
                    self.device, non_blocking=True
                ).reshape(-1)
        current_platform.synchronize()


class HiSparseC4DevicePool(DeepSeekV4SingleKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.compress_ratio = 4

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        mask = (full_indices + 1) % self.compress_ratio == 0
        compressed_indices = full_indices[mask] // self.compress_ratio
        return compressed_indices

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32
        )

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_to_hisparse_device(
            self.translate_loc_from_full_to_compressed(full_indices)
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().set_key_buffer_fused(layer_id, loc, cache_k)

    def transfer_values_on_device(
        self, dst_indices: torch.Tensor, src_indices: torch.Tensor
    ) -> None:
        if dst_indices.numel() == 0:
            return
        if dst_indices.numel() != src_indices.numel():
            raise RuntimeError(
                "HiSparseC4DevicePool device transfer mismatch: "
                f"{dst_indices.numel()} dst indices vs {src_indices.numel()} src indices."
            )

        dst_indices = dst_indices.to(device=self.device, dtype=torch.int64)
        src_indices = src_indices.to(device=self.device, dtype=torch.int64)

        bytes_per_token = self.get_bytes_per_token()
        byte_offsets = torch.arange(bytes_per_token, device=self.device)

        src_pages = src_indices // self.page_size
        dst_pages = dst_indices // self.page_size
        src_offsets = (src_indices % self.page_size) * bytes_per_token
        dst_offsets = (dst_indices % self.page_size) * bytes_per_token

        src_flat_indices = (
            src_pages[:, None] * self.bytes_per_page_padded
            + src_offsets[:, None]
            + byte_offsets[None, :]
        ).reshape(-1)
        dst_flat_indices = (
            dst_pages[:, None] * self.bytes_per_page_padded
            + dst_offsets[:, None]
            + byte_offsets[None, :]
        ).reshape(-1)

        for buf in self.kv_buffer:
            flat = buf.reshape(-1)
            flat[dst_flat_indices] = flat[src_flat_indices].clone()

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseC4DevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseC4DevicePool does not support load_cpu_copy")


class DeepSeekV4IndexerPool(KVCache):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.index_head_dim = index_head_dim

        self._create_buffer()

    def _create_buffer(self):
        num_scales_per_token = self.index_head_dim // self.quant_block_size
        page_bytes = self.page_size * self.index_head_dim
        page_bytes += self.page_size * num_scales_per_token * 4
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.index_k_with_scale_buffer = [
                    torch.zeros(
                        (self.size + self.page_size + 1) // self.page_size,
                        page_bytes,
                        dtype=self.index_k_with_scale_buffer_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_k_with_scale_buffer[layer_id]

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        buf = self.index_k_with_scale_buffer[layer_id]
        return index_buf_accessor.GetKAndS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def set_index_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache(
            input=cache_k,
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],
            indices=loc,
            page_size=self.page_size,
            type="indexer",
        )

    def _bytes_per_token(self) -> int:
        page_bytes = self.index_k_with_scale_buffer[0].shape[1]
        assert page_bytes % self.page_size == 0
        return page_bytes // self.page_size

    def _token_byte_indices(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(device=self.device, dtype=torch.int64)
        bytes_per_token = self._bytes_per_token()
        page_bytes = self.index_k_with_scale_buffer[0].shape[1]
        byte_offsets = torch.arange(bytes_per_token, device=self.device)
        pages = indices // self.page_size
        offsets = (indices % self.page_size) * bytes_per_token
        return (
            pages[:, None] * page_bytes
            + offsets[:, None]
            + byte_offsets[None, :]
        ).reshape(-1)

    def get_cpu_copy(self, indices, mamba_indices=None):
        current_platform.synchronize()
        indices = indices.to(device=self.device, dtype=torch.int64)
        bytes_per_token = self._bytes_per_token()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            flat_buffer = self.index_k_with_scale_buffer[layer_id].reshape(-1)
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                byte_indices = self._token_byte_indices(chunk_indices)
                kv_cache_cpu[-1].append(
                    flat_buffer[byte_indices]
                    .reshape(len(chunk_indices), bytes_per_token)
                    .to("cpu", non_blocking=True)
                )
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        current_platform.synchronize()
        indices = indices.to(device=self.device, dtype=torch.int64)
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            flat_buffer = self.index_k_with_scale_buffer[layer_id].reshape(-1)
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                cpu_data = kv_cache_cpu[layer_id][i // chunk_size]
                assert cpu_data.shape[0] == len(chunk_indices)
                byte_indices = self._token_byte_indices(chunk_indices)
                flat_buffer[byte_indices] = cpu_data.to(
                    self.device, non_blocking=True
                ).reshape(-1)
        current_platform.synchronize()


class DeepSeekV4LayerItem(NamedTuple):
    compress_ratio: Literal[0, 4, 128]
    compress_layer_id: int
    compress_kv_pool: Optional[DeepSeekV4SingleKVPool] = None


class DeepSeekV4TokenToKVPool(BaseSWAKVPool):

    def __init__(
        self,
        max_num_reqs: int,
        swa_size: int,
        c4_size: int,
        c128_size: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        page_size: int,
        swa_page_size: int,
        dtype: torch.dtype,
        c4_state_dtype: torch.dtype,
        c128_state_dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        indexer_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        compression_ratios: List[int],
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_hisparse: bool = False,
        online_mtp_max_draft_tokens: int = 0,
    ):
        super().__init__(
            swa_size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        c4_logical_size = c128_size * 32

        logger.info(
            "Initialize DeepSeekV4TokenToKVPool with "
            f"{max_num_reqs=} {swa_size=} {c4_size=} "
            f"{c4_logical_size=} {c128_size=} "
            f"{c4_state_pool_size=} {c128_state_pool_size=}"
        )

        self.max_num_reqs = max_num_reqs
        self.c4_size = c4_size
        self.c4_logical_size = c4_logical_size
        self.c128_size = c128_size
        self.c4_state_pool_size = c4_state_pool_size
        self.c128_state_pool_size = c128_state_pool_size
        self.c4_state_dtype = c4_state_dtype
        self.c128_state_dtype = c128_state_dtype
        self.compression_ratios = compression_ratios
        self.online_mtp_max_draft_tokens = online_mtp_max_draft_tokens
        self.online_c128_mtp_pending_seq_lens: Optional[torch.Tensor] = None
        if ONLINE_C128 and envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
            self.online_c128_mtp_pending_seq_lens = torch.empty(
                max_num_reqs, dtype=torch.int64, device=device
            )

        # Determine this PP stage's absolute layer range
        if (
            start_layer is not None
            and end_layer is not None
            and len(compression_ratios) >= end_layer
        ):
            self._stage_start = start_layer
            self._stage_end = end_layer
        else:
            self._stage_start = 0
            self._stage_end = len(compression_ratios)
        stage_ratios = compression_ratios[self._stage_start : self._stage_end]

        assert page_size % swa_page_size == 0

        self.swa_size = swa_size
        self.swa_window_size = swa_page_size
        self.swa_page_size = swa_page_size
        self.scale_pad = 1

        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.indexer_head_dim = indexer_head_dim

        c4_layer_num = sum(1 for r in stage_ratios if r == 4)
        c128_layer_num = sum(1 for r in stage_ratios if r == 128)
        c4_page_size = page_size // 4
        c128_page_size = page_size // 128
        self.swa_kv_pool = DeepSeekV4SingleKVPool(
            swa_size,
            swa_page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
        )

        c4_kv_pool_type = DeepSeekV4SingleKVPool
        if enable_hisparse:
            c4_kv_pool_type = HiSparseC4DevicePool
        self.c4_kv_pool = c4_kv_pool_type(
            c4_size,
            c4_page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            c4_layer_num,
            device,
            enable_memory_saver,
        )

        self.c128_kv_pool = DeepSeekV4SingleKVPool(
            c128_size,
            c128_page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            c128_layer_num,
            device,
            enable_memory_saver,
        )

        self.c4_indexer_kv_pool = DeepSeekV4IndexerPool(
            self.c4_logical_size if not _is_hip else c4_size,
            c4_page_size,
            dtype,
            indexer_head_dim,
            c4_layer_num,
            device,
            enable_memory_saver,
        )

        self._init_compressed_layer_mapping()

        if _is_hip:
            self._init_paged_compress_states(False)
        else:
            self._init_paged_compress_states(enable_memory_saver)

        self._should_cache_swa = envs.SGLANG_OPT_CACHE_SWA_TRANSLATION.get()
        self.cached_loc = None

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping
        self.cached_loc = None  # mapping replaced; discard any cached translation

    def invalidate_loc_cache(self) -> None:
        self.cached_loc = None

    def get_ring_size(self, compress_ratio: int) -> int:
        server_args = get_global_server_args()
        is_speculative = server_args.speculative_algorithm is not None
        return get_compress_state_ring_size(compress_ratio, is_speculative)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None

        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def set_swa_loc(self, loc: torch.Tensor) -> None:
        # No-op: SWAKVPool's set_swa_loc precomputes SWA-translated loc once per
        # forward batch for set_kv_buffer to read via self.swa_loc. DSV4 has its
        # own equivalent cache via `_should_cache_swa + cached_loc` (in
        # set_swa_key_buffer_radix_fused), so we ignore main's precomputed loc.
        pass

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        for bufs in [
            self.c4_kv_pool.kv_buffer,
            self.c4_indexer_kv_pool.index_k_with_scale_buffer,
            self.c128_kv_pool.kv_buffer,
        ]:
            for buf in bufs:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)

        return data_ptrs, data_lens, item_lens

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        for buf in self.swa_kv_pool.kv_buffer:
            assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(buf[0].nbytes)

        for pools in [
            self.compress_state_pools,
            self.indexer_compress_state_pools,
        ]:
            for pool in pools:
                if pool is None:
                    continue
                t = pool.kv_score_buffer.kv_score
                assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
                data_ptrs.append(t.data_ptr())
                data_lens.append(t.nbytes)
                item_lens.append(t[0].nbytes * pool.ring_size)

        return data_ptrs, data_lens, item_lens

    def _init_paged_compress_states(self, enable_memory_saver: bool):
        c4_state_pool_size = self.c4_state_pool_size
        c128_state_pool_size = self.c128_state_pool_size
        total_L = len(self.compression_ratios)
        self.compress_state_pools: List[Optional[CompressStatePool]] = [None] * total_L
        self.indexer_compress_state_pools: List[Optional[CompressStatePool]] = [
            None
        ] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 0:
                continue
            overlap = ratio == 4
            size = c4_state_pool_size if ratio == 4 else c128_state_pool_size
            ring_size = self.get_ring_size(ratio)

            compress_state_kwargs = dict(
                size=size,
                ring_size=ring_size,
                overlap=overlap,
                head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
                dtype=self.c4_state_dtype if ratio == 4 else self.c128_state_dtype,
                device=self.device,
                enable_memory_saver=enable_memory_saver,
                ratio=ratio,
                online=(ratio == 128 and ONLINE_C128),
                swa_page_size=self.swa_page_size,
                online_mtp_max_draft_tokens=(
                    self.online_mtp_max_draft_tokens if ratio == 128 else 0
                ),
            )
            self.compress_state_pools[idx] = CompressStatePool(
                **compress_state_kwargs
            )

            if ratio == 4:
                self.indexer_compress_state_pools[idx] = CompressStatePool(
                    size=size,
                    ring_size=ring_size,
                    overlap=overlap,
                    head_dim=self.indexer_head_dim,
                    device=self.device,
                    dtype=self.c4_state_dtype,
                    enable_memory_saver=enable_memory_saver,
                    ratio=ratio,
                    swa_page_size=self.swa_page_size,
                )

    def _init_compressed_layer_mapping(self):
        c1_cnt = c4_cnt = c128_cnt = 0
        total_L = len(self.compression_ratios)
        self.layer_mapping: List[Optional[DeepSeekV4LayerItem]] = [None] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 0:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=0,
                    compress_layer_id=c1_cnt,
                )
                c1_cnt += 1
            elif ratio == 4:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=4,
                    compress_layer_id=c4_cnt,
                    compress_kv_pool=self.c4_kv_pool,
                )
                c4_cnt += 1
            elif ratio == 128:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=128,
                    compress_layer_id=c128_cnt,
                    compress_kv_pool=self.c128_kv_pool,
                )
                c128_cnt += 1
            else:
                raise ValueError(f"Unsupported compression ratio: {ratio}")

    def wait_layer_transfer(self, layer_id: int) -> None:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_attention_compress_states(self, layer_id: int) -> CompressStatePool:
        self.wait_layer_transfer(layer_id)
        compress_state_pool = self.compress_state_pools[layer_id]
        assert (
            compress_state_pool is not None
        ), "Only c4/c128 layers have attention states."
        return compress_state_pool

    def get_online_c128_mtp_state_slot_offset(self) -> int:
        for pool in self.compress_state_pools:
            if pool is not None and pool.ratio == 128:
                return int(pool.online_mtp_state_slot_offset)
        return 0

    def get_online_c128_mtp_max_draft_tokens(self) -> int:
        return int(self.online_mtp_max_draft_tokens)

    def get_online_c128_mtp_pending_seq_lens(self) -> torch.Tensor:
        if self.online_c128_mtp_pending_seq_lens is None:
            raise RuntimeError(
                "Online C128 MTP pending seq_lens buffer is not initialized. "
                "Set SGLANG_OPT_USE_ONLINE_COMPRESS=1 and "
                "SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=1 for EAGLE online C128 MTP."
            )
        return self.online_c128_mtp_pending_seq_lens

    def get_indexer_compress_states(self, layer_id: int) -> CompressStatePool:
        self.wait_layer_transfer(layer_id)
        indexer_compress_state_pool = self.indexer_compress_state_pools[layer_id]
        assert (
            indexer_compress_state_pool is not None
        ), "Only c4 layers have indexer states."
        return indexer_compress_state_pool

    def _swa_local_layer_id(self, layer_id: int) -> int:
        """Convert absolute model layer_id to SWA-pool-local (PP-stage-local) index."""
        return layer_id - self._stage_start

    def get_swa_key_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))

    def set_swa_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        self.swa_kv_pool.set_key_buffer(
            self._swa_local_layer_id(layer_id), loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_extra_key_page_size(self, layer_id: int) -> int:
        _, _, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.page_size

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:
        self.wait_layer_transfer(layer_id)
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.get_key_buffer(compress_layer_id)

    def set_extra_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        compress_kv_pool.set_key_buffer(
            compress_layer_id, loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_index_k_page_size(self) -> int:
        return self.c4_indexer_kv_pool.page_size

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_with_scale_buffer(compress_layer_id)

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_scale_buffer(
            compress_layer_id, seq_len, page_indices
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        self.c4_indexer_kv_pool.set_index_k_scale_buffer(
            compress_layer_id, loc, index_k, index_k_scale
        )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def _compressed_indices_from_full(
        self, indices: torch.Tensor, compress_ratio: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = (indices + 1) % compress_ratio == 0
        return mask, (indices[mask] // compress_ratio).to(torch.int64)

    def _compressed_indices_from_mask(
        self,
        indices: torch.Tensor,
        compress_ratio: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.to(device=indices.device)
        target_mask = (indices + 1) % compress_ratio == 0
        if not torch.equal(mask, target_mask):
            raise RuntimeError(
                "DSV4 KV offload/load compression boundary mismatch: "
                f"ratio={compress_ratio}, saved={int(mask.sum().item())}, "
                f"target={int(target_mask.sum().item())}."
            )
        return (indices[mask] // compress_ratio).to(torch.int64)

    def _compress_state_locs_from_full(
        self, indices: torch.Tensor, compress_ratio: int
    ) -> torch.Tensor:
        swa_loc = self.translate_loc_from_full_to_swa(indices).to(torch.int64)
        swa_loc = swa_loc[swa_loc > 0]
        return self._compress_state_locs_from_swa(swa_loc, compress_ratio)

    def _compress_state_locs_from_swa(
        self, swa_loc: torch.Tensor, compress_ratio: int
    ) -> torch.Tensor:
        state_locs = self._raw_compress_state_locs_from_swa(swa_loc, compress_ratio)
        return torch.unique(state_locs.to(torch.int64), sorted=True)

    def _raw_compress_state_locs_from_swa(
        self, swa_loc: torch.Tensor, compress_ratio: int
    ) -> torch.Tensor:
        if swa_loc.numel() == 0:
            return swa_loc

        ring_size = self.get_ring_size(compress_ratio)
        swa_pages = swa_loc // self.swa_page_size
        state_locs = (
            swa_pages * ring_size + (swa_loc % ring_size)
        ) // compress_ratio
        return state_locs.to(torch.int64)

    def _copy_compress_state_pools(
        self,
        state_locs: torch.Tensor,
        state_pools: List[Optional[CompressStatePool]],
        compress_ratio: int,
    ) -> List[Optional[torch.Tensor]]:
        copied_states: List[Optional[torch.Tensor]] = []
        for layer_id, pool in enumerate(state_pools):
            if (
                pool is None
                or self.compression_ratios[layer_id] != compress_ratio
                or state_locs.numel() == 0
            ):
                copied_states.append(None)
                continue
            copied_states.append(
                pool.kv_score_buffer.kv_score[state_locs].to(
                    "cpu", non_blocking=True
                )
            )
        return copied_states

    def _load_compress_state_pools(
        self,
        copied_states: List[Optional[torch.Tensor]],
        state_locs: torch.Tensor,
        state_pools: List[Optional[CompressStatePool]],
        compress_ratio: int,
    ) -> None:
        for layer_id, (copied_state, pool) in enumerate(zip(copied_states, state_pools)):
            if (
                copied_state is None
                or pool is None
                or self.compression_ratios[layer_id] != compress_ratio
            ):
                continue
            if state_locs.numel() != copied_state.shape[0]:
                raise RuntimeError(
                    "DSV4 KV offload/load state location mismatch: "
                    f"saved={copied_state.shape[0]}, target={state_locs.numel()}."
                )
            pool.set_state_by_state_loc(
                state_locs, KVAndScore(copied_state.to(self.device, non_blocking=True))
            )

    def _filter_layer_chunks(self, kv_cache_cpu, row_mask: torch.Tensor, pool):
        if kv_cache_cpu is None:
            return None
        if row_mask is None or bool(torch.all(row_mask).item()):
            return kv_cache_cpu

        chunk_size = getattr(pool, "cpu_offloading_chunk_size", len(row_mask))
        filtered = []
        for layer_chunks in kv_cache_cpu:
            if len(layer_chunks) == 0:
                filtered.append([])
                continue

            filtered_layer = []
            first_chunk = layer_chunks[0]
            if isinstance(first_chunk, (list, tuple)):
                k_cpu = torch.cat([chunk[0] for chunk in layer_chunks], dim=0)
                v_cpu = torch.cat([chunk[1] for chunk in layer_chunks], dim=0)
                k_cpu = k_cpu[row_mask]
                v_cpu = v_cpu[row_mask]
                for i in range(0, len(k_cpu), chunk_size):
                    filtered_layer.append(
                        [k_cpu[i : i + chunk_size], v_cpu[i : i + chunk_size]]
                    )
            else:
                data_cpu = torch.cat(layer_chunks, dim=0)
                data_cpu = data_cpu[row_mask]
                for i in range(0, len(data_cpu), chunk_size):
                    filtered_layer.append(data_cpu[i : i + chunk_size])
            filtered.append(filtered_layer)
        return filtered

    def _load_remapped_compress_state_pools(
        self,
        copied_states: List[Optional[torch.Tensor]],
        saved_state_locs_cpu: torch.Tensor,
        old_swa_locs: torch.Tensor,
        new_swa_locs: torch.Tensor,
        state_pools: List[Optional[CompressStatePool]],
        compress_ratio: int,
    ) -> None:
        if old_swa_locs.numel() == 0 or new_swa_locs.numel() == 0:
            return

        old_state_locs = self._raw_compress_state_locs_from_swa(
            old_swa_locs.to(device=self.device, dtype=torch.int64), compress_ratio
        )
        new_state_locs = self._raw_compress_state_locs_from_swa(
            new_swa_locs.to(device=self.device, dtype=torch.int64), compress_ratio
        )
        if old_state_locs.numel() == 0 or new_state_locs.numel() == 0:
            return
        old_unique_state_locs = torch.unique(old_state_locs.to(torch.int64), sorted=True)
        new_unique_state_locs = torch.unique(new_state_locs.to(torch.int64), sorted=True)
        if torch.equal(old_unique_state_locs, new_unique_state_locs):
            self._load_compress_state_pools(
                copied_states, new_unique_state_locs, state_pools, compress_ratio
            )
            return

        saved_state_locs = saved_state_locs_cpu.to(device="cpu", dtype=torch.int64)
        saved_rows = {int(loc): row for row, loc in enumerate(saved_state_locs.tolist())}
        new_to_old = {}
        for old_loc, new_loc in zip(
            old_state_locs.to("cpu").tolist(), new_state_locs.to("cpu").tolist()
        ):
            if old_loc in saved_rows and new_loc not in new_to_old:
                new_to_old[new_loc] = old_loc

        if not new_to_old:
            return

        target_locs_cpu = torch.tensor(
            sorted(new_to_old), dtype=torch.int64, device="cpu"
        )
        source_rows = torch.tensor(
            [saved_rows[new_to_old[int(loc)]] for loc in target_locs_cpu.tolist()],
            dtype=torch.int64,
            device="cpu",
        )
        target_locs = target_locs_cpu.to(device=self.device)

        for layer_id, (copied_state, pool) in enumerate(zip(copied_states, state_pools)):
            if (
                copied_state is None
                or pool is None
                or self.compression_ratios[layer_id] != compress_ratio
            ):
                continue
            pool.set_state_by_state_loc(
                target_locs,
                KVAndScore(
                    copied_state[source_rows].to(self.device, non_blocking=True)
                ),
            )

    def get_cpu_copy(self, indices, mamba_indices=None):
        current_platform.synchronize()
        indices = indices.to(device=self.device, dtype=torch.int64)

        swa_indices = self.translate_loc_from_full_to_swa(indices).to(torch.int64)
        swa_mask = swa_indices > 0
        swa_kv_cpu = (
            self.swa_kv_pool.get_cpu_copy(swa_indices[swa_mask])
            if torch.any(swa_mask)
            else None
        )

        c4_mask, c4_indices = self._compressed_indices_from_full(indices, 4)
        c128_mask, c128_indices = self._compressed_indices_from_full(indices, 128)

        c4_kv_cpu = None
        if not isinstance(self.c4_kv_pool, HiSparseC4DevicePool):
            c4_kv_cpu = (
                self.c4_kv_pool.get_cpu_copy(c4_indices)
                if c4_indices.numel() > 0
                else None
            )
        c4_indexer_cpu = (
            self.c4_indexer_kv_pool.get_cpu_copy(c4_indices)
            if c4_indices.numel() > 0
            else None
        )
        c128_kv_cpu = (
            self.c128_kv_pool.get_cpu_copy(c128_indices)
            if c128_indices.numel() > 0
            else None
        )

        c4_state_locs = self._compress_state_locs_from_full(indices, 4)
        c128_state_locs = self._compress_state_locs_from_full(indices, 128)
        c4_attention_states = self._copy_compress_state_pools(
            c4_state_locs, self.compress_state_pools, 4
        )
        c4_indexer_states = self._copy_compress_state_pools(
            c4_state_locs, self.indexer_compress_state_pools, 4
        )
        c128_attention_states = self._copy_compress_state_pools(
            c128_state_locs, self.compress_state_pools, 128
        )

        current_platform.synchronize()
        return {
            "length": len(indices),
            "swa": swa_kv_cpu,
            "swa_mask": swa_mask.cpu(),
            "swa_indices": swa_indices[swa_mask].cpu(),
            "c4": c4_kv_cpu,
            "c4_indexer": c4_indexer_cpu,
            "c4_mask": c4_mask.cpu(),
            "c4_state_locs": c4_state_locs.cpu(),
            "c4_attention_states": c4_attention_states,
            "c4_indexer_states": c4_indexer_states,
            "c128": c128_kv_cpu,
            "c128_mask": c128_mask.cpu(),
            "c128_state_locs": c128_state_locs.cpu(),
            "c128_attention_states": c128_attention_states,
        }

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        current_platform.synchronize()
        indices = indices.to(device=self.device, dtype=torch.int64)
        if len(indices) != kv_cache_cpu["length"]:
            raise RuntimeError(
                "DSV4 KV offload/load length mismatch: "
                f"saved={kv_cache_cpu['length']}, target={len(indices)}."
            )

        swa_kv_cpu = kv_cache_cpu["swa"]
        swa_indices = self.translate_loc_from_full_to_swa(indices).to(torch.int64)
        old_swa_mask = kv_cache_cpu["swa_mask"].to(device=indices.device)
        new_swa_mask = swa_indices > 0
        row_mask = torch.empty((0,), dtype=torch.bool, device="cpu")
        load_swa_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        old_swa_locs = torch.empty((0,), dtype=torch.int64, device=self.device)
        if torch.any(old_swa_mask):
            saved_swa_indices = kv_cache_cpu["swa_indices"].to(
                device=self.device, dtype=torch.int64
            )
            row_mask = new_swa_mask[old_swa_mask].cpu()
            load_swa_indices = swa_indices[old_swa_mask][
                row_mask.to(device=indices.device)
            ]
            old_swa_locs = saved_swa_indices[row_mask.to(device=self.device)]
        if swa_kv_cpu is not None:
            if load_swa_indices.numel() > 0:
                swa_kv_cpu = self._filter_layer_chunks(
                    swa_kv_cpu, row_mask, self.swa_kv_pool
                )
                self.swa_kv_pool.load_cpu_copy(swa_kv_cpu, load_swa_indices)

        c4_indices = self._compressed_indices_from_mask(
            indices, 4, kv_cache_cpu["c4_mask"]
        )
        c128_indices = self._compressed_indices_from_mask(
            indices, 128, kv_cache_cpu["c128_mask"]
        )

        if kv_cache_cpu["c4"] is not None:
            self.c4_kv_pool.load_cpu_copy(kv_cache_cpu["c4"], c4_indices)
        if kv_cache_cpu["c4_indexer"] is not None:
            self.c4_indexer_kv_pool.load_cpu_copy(
                kv_cache_cpu["c4_indexer"], c4_indices
            )
        if kv_cache_cpu["c128"] is not None:
            self.c128_kv_pool.load_cpu_copy(kv_cache_cpu["c128"], c128_indices)

        self._load_remapped_compress_state_pools(
            kv_cache_cpu["c4_attention_states"],
            kv_cache_cpu["c4_state_locs"],
            old_swa_locs,
            load_swa_indices,
            self.compress_state_pools,
            4,
        )
        self._load_remapped_compress_state_pools(
            kv_cache_cpu["c4_indexer_states"],
            kv_cache_cpu["c4_state_locs"],
            old_swa_locs,
            load_swa_indices,
            self.indexer_compress_state_pools,
            4,
        )
        self._load_remapped_compress_state_pools(
            kv_cache_cpu["c128_attention_states"],
            kv_cache_cpu["c128_state_locs"],
            old_swa_locs,
            load_swa_indices,
            self.compress_state_pools,
            128,
        )
        current_platform.synchronize()

    def set_swa_key_buffer_radix(
        self,
        layer_id: int,
        raw_loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
        self.swa_kv_pool.set_key_buffer(
            self._swa_local_layer_id(layer_id), swa_loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))

    def set_swa_key_buffer_radix_fused(
        self,
        layer_id: int,
        raw_loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        if self._should_cache_swa:
            if layer_id == self.start_layer or self.cached_loc is None:
                self.cached_loc = self.translate_loc_from_full_to_swa(raw_loc)
            swa_loc = self.cached_loc
        else:
            swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
        return self.swa_kv_pool.set_key_buffer_fused(
            self._swa_local_layer_id(layer_id), swa_loc, cache_k
        )

    def set_swa_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        raw_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if self._should_cache_swa:
            if layer_id == self.start_layer or self.cached_loc is None:
                self.cached_loc = self.translate_loc_from_full_to_swa(raw_loc)
            swa_loc = self.cached_loc
        else:
            swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
        fused_k_norm_rope_flashmla(
            kv=kv,
            kv_weight=kv_weight,
            eps=eps,
            freqs_cis=freqs_cis,
            positions=positions,
            out_loc=swa_loc,
            kvcache=self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)],
            page_size=self.swa_kv_pool.page_size,
        )

    def set_extra_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.set_key_buffer_fused(compress_layer_id, loc, cache_k)

    def set_index_k_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.set_index_fused(compress_layer_id, loc, cache_k)
