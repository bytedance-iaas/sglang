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
    turboquant_dequantize,
    turboquant_quantize,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    get_tensor_size_bytes,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)


class MHATokenToKVPoolTurboQuant(MHATokenToKVPool):
    """Memory pool that stores KV cache compressed via TurboQuant.

    Storage per token per head per layer:
      - Bit-packed centroid indices (uint8, packed at b bits/coord)
      - L2 norm (float32, 1 per token-head)
      - [prod mode only] QJL sign bits (uint8, packed at 1 bit/coord)
      - [prod mode only] Residual norm (float32, 1 per token-head)

    On set_kv_buffer: quantize via TurboQuant, store compressed.
    On get_key/value_buffer: dequantize to working dtype (full buffer, same as FP4).
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
        bits: int = 4,
        mode: str = "mse",
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        self.bits = bits
        self.mode = mode
        self.mse_bits = bits - 1 if mode == "prod" else bits

        # Cache padded dimensions
        self.padded_head_dim = _next_power_of_2(head_dim)
        effective_v = v_head_dim if v_head_dim is not None else head_dim
        self.v_padded_head_dim = _next_power_of_2(effective_v)

        # Initialize Hadamard transforms (shared across layers, deterministic seeds)
        torch_device = torch.device(device)
        self.k_hadamard = HadamardTransform(head_dim, seed=42, device=torch_device)
        self.v_hadamard = HadamardTransform(effective_v, seed=137, device=torch_device)

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
        """Allocate bit-packed compressed storage buffers."""
        m = self.size + self.page_size
        k_packed_dim = compute_packed_dim(self.padded_head_dim, self.mse_bits)
        v_packed_dim = compute_packed_dim(self.v_padded_head_dim, self.mse_bits)

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # Bit-packed centroid indices
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

                # L2 norms per token per head
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

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_norms_buffer
        del self.v_norms_buffer
        if self.mode == "prod":
            del self.k_qjl_buffer
            del self.v_qjl_buffer
            del self.k_residual_norms_buffer
            del self.v_residual_norms_buffer

    def get_kv_size_bytes(self):
        k_size = sum(get_tensor_size_bytes(b) for b in self.k_buffer)
        k_size += sum(get_tensor_size_bytes(b) for b in self.k_norms_buffer)
        v_size = sum(get_tensor_size_bytes(b) for b in self.v_buffer)
        v_size += sum(get_tensor_size_bytes(b) for b in self.v_norms_buffer)
        if self.mode == "prod":
            k_size += sum(get_tensor_size_bytes(b) for b in self.k_qjl_buffer)
            k_size += sum(get_tensor_size_bytes(b) for b in self.k_residual_norms_buffer)
            v_size += sum(get_tensor_size_bytes(b) for b in self.v_qjl_buffer)
            v_size += sum(get_tensor_size_bytes(b) for b in self.v_residual_norms_buffer)
        return k_size, v_size

    def _get_key_buffer(self, layer_id: int):
        """Dequantize and return full key buffer for a layer."""
        idx = layer_id - self.start_layer
        packed = self.k_buffer[idx]
        norms = self.k_norms_buffer[idx]

        quantized = {
            "packed_indices": packed.reshape(-1, packed.shape[-1]),
            "norms": norms.reshape(-1),
            "padded_dim": self.padded_head_dim,
        }
        if self.mode == "prod":
            quantized["qjl_signs"] = self.k_qjl_buffer[idx].reshape(
                -1, self.k_qjl_buffer[idx].shape[-1]
            )
            quantized["residual_norms"] = self.k_residual_norms_buffer[idx].reshape(-1)

        result = turboquant_dequantize(
            quantized, self.k_hadamard, self.bits, self.mode, self.dtype
        )
        return result[:, : self.head_dim].reshape(
            packed.shape[0], self.head_num, self.head_dim
        )

    def _get_value_buffer(self, layer_id: int):
        """Dequantize and return full value buffer for a layer."""
        idx = layer_id - self.start_layer
        packed = self.v_buffer[idx]
        norms = self.v_norms_buffer[idx]

        quantized = {
            "packed_indices": packed.reshape(-1, packed.shape[-1]),
            "norms": norms.reshape(-1),
            "padded_dim": self.v_padded_head_dim,
        }
        if self.mode == "prod":
            quantized["qjl_signs"] = self.v_qjl_buffer[idx].reshape(
                -1, self.v_qjl_buffer[idx].shape[-1]
            )
            quantized["residual_norms"] = self.v_residual_norms_buffer[idx].reshape(-1)

        result = turboquant_dequantize(
            quantized, self.v_hadamard, self.bits, self.mode, self.dtype
        )
        return result[:, : self.v_head_dim].reshape(
            packed.shape[0], self.head_num, self.v_head_dim
        )

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

        k_q = turboquant_quantize(k_flat, self.k_hadamard, self.bits, self.mode)
        v_q = turboquant_quantize(v_flat, self.v_hadamard, self.bits, self.mode)

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

        if self.mode == "prod":
            self.k_qjl_buffer[idx][loc] = k_q["qjl_signs"].reshape(
                num_tokens, self.head_num, -1
            )
            self.v_qjl_buffer[idx][loc] = v_q["qjl_signs"].reshape(
                num_tokens, self.head_num, -1
            )
            self.k_residual_norms_buffer[idx][loc] = k_q["residual_norms"].reshape(
                num_tokens, self.head_num
            )
            self.v_residual_norms_buffer[idx][loc] = v_q["residual_norms"].reshape(
                num_tokens, self.head_num
            )

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
