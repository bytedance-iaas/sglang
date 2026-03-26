"""
TurboQuant memory pool for KV cache compression.

Implements Google's TurboQuant (ICLR 2026) KV cache quantization.
Stores bit-packed centroid indices + L2 norms per head per token.
On read, entries are dequantized back to the model's working dtype.

Follows the same pattern as MHATokenToKVPoolFP4 for buffer management.
"""

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.quantization.turboquant_kernels import (
    HadamardTransform,
    _next_power_of_2,
    compute_packed_dim,
    compute_packed_dim_mixed,
    parse_bits,
    turboquant_dequantize,
    turboquant_dequantize_mixed,
    turboquant_quantize,
    turboquant_quantize_mixed,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    get_tensor_size_bytes,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)

# Target peak memory for float32 intermediates during chunked dequantization.
# The Hadamard inverse allocates (chunk_tokens * num_heads * padded_dim * 4)
# bytes of float32 temporaries.  We pick the chunk size dynamically so this
# stays under the budget below.
_DEQUANT_CHUNK_MEMORY_BUDGET = 256 * 1024 * 1024  # 256 MB


class MHATokenToKVPoolTurboQuant(MHATokenToKVPool):
    """Memory pool that stores KV cache compressed via TurboQuant.

    Storage per token per head per layer:
      - Bit-packed centroid indices (uint8, packed at b bits/coord)
      - L2 norm (float32, 1 per token-head)

    Two shared workspace buffers (one K, one V) of shape
    (max_tokens, head_num, head_dim) in the working dtype are pre-allocated
    and reused across layers.  On _get_key_buffer / _get_value_buffer the
    compressed data is dequantized *in chunks* into the workspace to limit
    peak float32 memory usage.

    On set_kv_buffer: quantize via TurboQuant, store compressed.
    On get_key/value_buffer: dequantize to working dtype via workspace.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        bits: float = 4,
        mode: str = "mse",
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        self.bits = bits
        self.mode = mode
        self.is_mixed, self.bits_hi, self.bits_lo = parse_bits(bits)
        self.mse_bits = int(bits) - 1 if mode == "prod" and not self.is_mixed else int(bits)

        # Cache padded dimensions
        self.padded_head_dim = _next_power_of_2(head_dim)
        effective_v = v_head_dim if v_head_dim is not None else head_dim
        self.v_padded_head_dim = _next_power_of_2(effective_v)

        # Initialize Hadamard transforms (shared across layers, deterministic seeds)
        torch_device = torch.device(device)
        self.k_hadamard = HadamardTransform(head_dim, seed=42, device=torch_device)
        self.v_hadamard = HadamardTransform(effective_v, seed=137, device=torch_device)

        # Compute chunk size for dequantization based on memory budget.
        # float32 temporaries: chunk_tokens * head_num * max_padded_dim * 4 bytes
        max_padded = max(self.padded_head_dim, self.v_padded_head_dim)
        bytes_per_token = head_num * max_padded * 4  # float32
        self._dequant_chunk_tokens = max(
            1, _DEQUANT_CHUNK_MEMORY_BUDGET // bytes_per_token
        )

        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            v_head_dim=v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
        )

    def _create_buffers(self):
        """Allocate bit-packed compressed storage buffers + shared workspace."""
        # Set store_dtype here (not in __init__) because KVCache.__init__
        # overwrites it.  Matches the FP4 pool pattern.
        self.store_dtype = torch.uint8

        m = self.size + self.page_size
        k_packed_dim = compute_packed_dim_mixed(self.padded_head_dim, self.bits)
        v_packed_dim = compute_packed_dim_mixed(self.v_padded_head_dim, self.bits)

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # Bit-packed centroid indices — per layer
                self.k_buffer = [
                    torch.zeros(
                        (m, self.head_num, k_packed_dim),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (m, self.head_num, v_packed_dim),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                # L2 norms per token per head — per layer
                self.k_norms_buffer = [
                    torch.zeros(
                        (m, self.head_num),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_norms_buffer = [
                    torch.zeros(
                        (m, self.head_num),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                # QJL sign bits — only for "prod" mode
                if self.mode == "prod":
                    k_qjl_dim = compute_packed_dim(self.padded_head_dim, 1)
                    v_qjl_dim = compute_packed_dim(self.v_padded_head_dim, 1)
                    self.k_qjl_buffer = [
                        torch.zeros(
                            (m, self.head_num, k_qjl_dim),
                            dtype=torch.uint8,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_qjl_buffer = [
                        torch.zeros(
                            (m, self.head_num, v_qjl_dim),
                            dtype=torch.uint8,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.k_residual_norms_buffer = [
                        torch.zeros(
                            (m, self.head_num),
                            dtype=torch.float32,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_residual_norms_buffer = [
                        torch.zeros(
                            (m, self.head_num),
                            dtype=torch.float32,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

                # Shared workspace buffers for dequantized data — reused across
                # layers.  Only one layer's attention runs at a time, so a single
                # pair suffices.
                self._k_workspace = torch.zeros(
                    (m, self.head_num, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                self._v_workspace = torch.zeros(
                    (m, self.head_num, self.v_head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_norms_buffer
        del self.v_norms_buffer
        del self._k_workspace
        del self._v_workspace
        if self.mode == "prod":
            del self.k_qjl_buffer
            del self.v_qjl_buffer
            del self.k_residual_norms_buffer
            del self.v_residual_norms_buffer

    def get_kv_size_bytes(self):
        k_size = sum(get_tensor_size_bytes(b) for b in self.k_buffer)
        k_size += sum(get_tensor_size_bytes(b) for b in self.k_norms_buffer)
        k_size += get_tensor_size_bytes(self._k_workspace)
        v_size = sum(get_tensor_size_bytes(b) for b in self.v_buffer)
        v_size += sum(get_tensor_size_bytes(b) for b in self.v_norms_buffer)
        v_size += get_tensor_size_bytes(self._v_workspace)
        if self.mode == "prod":
            k_size += sum(get_tensor_size_bytes(b) for b in self.k_qjl_buffer)
            k_size += sum(
                get_tensor_size_bytes(b) for b in self.k_residual_norms_buffer
            )
            v_size += sum(get_tensor_size_bytes(b) for b in self.v_qjl_buffer)
            v_size += sum(
                get_tensor_size_bytes(b) for b in self.v_residual_norms_buffer
            )
        return k_size, v_size

    def _dequant_layer_chunked(
        self,
        packed: torch.Tensor,
        norms: torch.Tensor,
        workspace: torch.Tensor,
        hadamard: HadamardTransform,
        padded_dim: int,
        out_dim: int,
        qjl_buf: Optional[torch.Tensor] = None,
        residual_norms_buf: Optional[torch.Tensor] = None,
    ):
        """Dequantize one layer's buffer in chunks into *workspace*."""
        total_tokens = packed.shape[0]  # m  (pool size)
        num_heads = packed.shape[1]
        chunk = self._dequant_chunk_tokens

        for start in range(0, total_tokens, chunk):
            end = min(start + chunk, total_tokens)
            c_packed = packed[start:end]  # (c, H, packed_dim)
            c_norms = norms[start:end]  # (c, H)

            if self.is_mixed:
                # Compute split_packed_offset for this padded_dim
                split = padded_dim // 2
                offset = compute_packed_dim(split, self.bits_hi)
                quantized = {
                    "packed_indices": c_packed.reshape(-1, c_packed.shape[-1]),
                    "norms": c_norms.reshape(-1),
                    "padded_dim": padded_dim,
                    "split_packed_offset": offset,
                    "bits_hi": self.bits_hi,
                    "bits_lo": self.bits_lo,
                }
                result = turboquant_dequantize_mixed(
                    quantized, hadamard, self.dtype
                )
            else:
                quantized = {
                    "packed_indices": c_packed.reshape(-1, c_packed.shape[-1]),
                    "norms": c_norms.reshape(-1),
                    "padded_dim": padded_dim,
                }
                if self.mode == "prod" and qjl_buf is not None:
                    c_qjl = qjl_buf[start:end]
                    quantized["qjl_signs"] = c_qjl.reshape(
                        -1, c_qjl.shape[-1]
                    )
                    quantized["residual_norms"] = residual_norms_buf[
                        start:end
                    ].reshape(-1)
                result = turboquant_dequantize(
                    quantized, hadamard, int(self.bits), self.mode, self.dtype
                )

            workspace[start:end] = result[:, :out_dim].reshape(
                end - start, num_heads, out_dim
            )

    def _get_key_buffer(self, layer_id: int):
        """Dequantize and return full key buffer for a layer."""
        idx = layer_id - self.start_layer
        qjl_buf = self.k_qjl_buffer[idx] if self.mode == "prod" else None
        res_buf = (
            self.k_residual_norms_buffer[idx] if self.mode == "prod" else None
        )
        self._dequant_layer_chunked(
            self.k_buffer[idx],
            self.k_norms_buffer[idx],
            self._k_workspace,
            self.k_hadamard,
            self.padded_head_dim,
            self.head_dim,
            qjl_buf,
            res_buf,
        )
        return self._k_workspace

    def _get_value_buffer(self, layer_id: int):
        """Dequantize and return full value buffer for a layer."""
        idx = layer_id - self.start_layer
        qjl_buf = self.v_qjl_buffer[idx] if self.mode == "prod" else None
        res_buf = (
            self.v_residual_norms_buffer[idx] if self.mode == "prod" else None
        )
        self._dequant_layer_chunked(
            self.v_buffer[idx],
            self.v_norms_buffer[idx],
            self._v_workspace,
            self.v_hadamard,
            self.v_padded_head_dim,
            self.v_head_dim,
            qjl_buf,
            res_buf,
        )
        return self._v_workspace

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        """Quantize and store K/V cache entries via TurboQuant."""
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id

        idx = layer_id - self.start_layer
        num_tokens = cache_k.shape[0]

        k_flat = cache_k.reshape(-1, self.head_dim)
        v_flat = cache_v.reshape(-1, self.v_head_dim)

        if self.is_mixed:
            k_q = turboquant_quantize_mixed(
                k_flat, self.k_hadamard, self.bits_hi, self.bits_lo
            )
            v_q = turboquant_quantize_mixed(
                v_flat, self.v_hadamard, self.bits_hi, self.bits_lo
            )
        else:
            k_q = turboquant_quantize(
                k_flat, self.k_hadamard, int(self.bits), self.mode
            )
            v_q = turboquant_quantize(
                v_flat, self.v_hadamard, int(self.bits), self.mode
            )

        self.k_buffer[idx][loc] = k_q["packed_indices"].reshape(
            num_tokens, self.head_num, -1
        )
        self.v_buffer[idx][loc] = v_q["packed_indices"].reshape(
            num_tokens, self.head_num, -1
        )
        self.k_norms_buffer[idx][loc] = k_q["norms"].reshape(
            num_tokens, self.head_num
        )
        self.v_norms_buffer[idx][loc] = v_q["norms"].reshape(
            num_tokens, self.head_num
        )

        if not self.is_mixed and self.mode == "prod":
            self.k_qjl_buffer[idx][loc] = k_q["qjl_signs"].reshape(
                num_tokens, self.head_num, -1
            )
            self.v_qjl_buffer[idx][loc] = v_q["qjl_signs"].reshape(
                num_tokens, self.head_num, -1
            )
            self.k_residual_norms_buffer[idx][loc] = k_q[
                "residual_norms"
            ].reshape(num_tokens, self.head_num)
            self.v_residual_norms_buffer[idx][loc] = v_q[
                "residual_norms"
            ].reshape(num_tokens, self.head_num)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Copy KV cache entries between locations."""
        if tgt_loc.numel() == 0:
            return
        for i in range(self.layer_num):
            self.k_buffer[i][tgt_loc] = self.k_buffer[i][src_loc]
            self.v_buffer[i][tgt_loc] = self.v_buffer[i][src_loc]
            self.k_norms_buffer[i][tgt_loc] = self.k_norms_buffer[i][src_loc]
            self.v_norms_buffer[i][tgt_loc] = self.v_norms_buffer[i][src_loc]
            if self.mode == "prod":
                self.k_qjl_buffer[i][tgt_loc] = self.k_qjl_buffer[i][src_loc]
                self.v_qjl_buffer[i][tgt_loc] = self.v_qjl_buffer[i][src_loc]
                self.k_residual_norms_buffer[i][tgt_loc] = (
                    self.k_residual_norms_buffer[i][src_loc]
                )
                self.v_residual_norms_buffer[i][tgt_loc] = (
                    self.v_residual_norms_buffer[i][src_loc]
                )
