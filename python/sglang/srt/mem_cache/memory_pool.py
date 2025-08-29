"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter

"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

import abc
import logging
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.func import vmap, functionalize
import triton
import triton.language as tl

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import get_bool_env_var, is_cuda, next_power_of_2

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()

from sglang.jack_utils import hcdprint
from sglang.srt.utils import is_flashinfer_available
if is_flashinfer_available():
    print(f"[horenc] LOAD FLASHINFER FUNCTION")
    # from sglang.srt.layers.quantization.nvfp4_tensor import NVFP4QuantizeUtil # need to delayed import
    # from flashinfer import fp4_quantize
    # from flashinfer import jack_test
    # from flashinfer import fp4_batched_quantize
    #from flashinfer import nvfp4_batched_quantize # -> NVFP4QuantizeUtil.batched_quantize 

# from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil
USE_KV_MXFP4 = 1
# 1.
# EXP_BIAS = 1

# 2.
# # ===== FP4 E2M1 參數（默認使用 bias=2）=====
# BIAS = 1  # 若規格是 bias=1，改成 1 即可
# EXP_BITS = 2
# MAN_BITS = 1
# S_MAX = 1 << (EXP_BITS + MAN_BITS + 1)  # 不直接使用，僅保留

# # 預先計算常數（float32）
# _MIN_NORMAL = float(2.0 ** (1 - BIAS))      # 最小正規數，例如 bias=2 -> 0.5
# _SUB_STEP   = float(2.0 ** (-BIAS))         # 唯一非零次正規數的量級，bias=2 -> 0.25
# _SUB_HALF   = float(2.0 ** (-BIAS - 1))     # 0 與次正規值的中點，bias=2 -> 0.125
# _MAX_VALUE  = float((2.0 ** (3 - BIAS)) * 1.5)  # 最大有限值，bias=2 -> 3.0
# _NORM_THR   = 1.25  # 正規化 mantissa 1 與 1.5 的臨界點


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (size, max_context_len), dtype=torch.int32, device=device
            )

        self.free_slots = list(range(size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))


class KVCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter):
        self.layer_transfer_counter = layer_transfer_counter

    def get_cpu_copy(self, indices):
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError()

class MHATokenToKVPool(KVCache):
    # ===== FP4 E2M1 配置 =====
    BIAS = 1  # 或 2
    EVEN_INDEX_IN_HIGH_NIBBLE = True

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
        self.head_num = head_num
        self.head_dim = head_dim

        # for disagg with nvlink
        self.enable_custom_mem_pool = get_bool_env_var(
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "false"
        )
        if self.enable_custom_mem_pool:
            # TODO(shangming): abstract custom allocator class for more backends
            from mooncake.allocator import NVLinkAllocator

            allocator = NVLinkAllocator.get_allocator(self.device)
            self.custom_mem_pool = torch.cuda.MemPool(allocator.allocator())
        else:
            self.custom_mem_pool = None

        self._create_buffers()

        self.layer_transfer_counter = None
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if _is_cuda else None

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB"
        )
        self.mem_usage = (k_size + v_size) / GB


        if not USE_KV_MXFP4:
            # [horenc]
            # 預建 FP4 lookup table # FP4 值（已排序）與 thresholds（相鄰中點）
            hcdprint(f"[horenc] init FP4_VALUES table in class MHATokenToKVPool:__init__()")

            self._max_elems = 10_000_000_00  # 這裡你可以改成你模型的最大 tensor 大小

            self.FP4_VALUES = torch.tensor([
                    -6.0, -3.0, -2.0, -1.0,
                    -0.5, -0.25, -0.125, -0.0,
                    0.0,  0.125, 0.25, 0.5,
                    1.0, 2.0, 3.0, 6.0
            ], dtype=torch.float32, device=self.device) # set device for CUDA graph
            # # Perf pre-do: thresholds 長度為 15：介於相鄰兩個 quant level 的中點
            # fp = self.FP4_VALUES
            # self.FP4_THRESHOLDS = ((fp[:-1] + fp[1:]) / 2.0).to(self.device)  # shape [15]
            
            # self.FP4_THRESHOLDS = ((self.FP4_VALUES[:-1] + self.FP4_VALUES[1:]) / 2.0)
            self.FP4_VALUES_expand = self.FP4_VALUES.unsqueeze(0)  # [1,16]
            
            # # 預分配 buffer
            # self._indices_buf = torch.empty(self._max_elems, dtype=torch.uint8, device=device)
            # self._packed_buf = torch.empty((self._max_elems + 1) // 2, dtype=torch.uint8, device=device)
            # self._dequant_buf = torch.empty(self._max_elems, dtype=torch.float32, device=device)

            # 建立 FP4 E2M1 的查表 (float32)
            # self._FP4_LUT = self._build_fp4_e2m1_lut(self.BIAS)

        # torch cimpile init
        self._FP4_LUT = self._build_fp4_e2m1_lut(self.BIAS)
        self._quantize_fn = torch.compile(self._quantize_bf16_to_fp4_e2m1_impl)
        self._dequantize_fn = torch.compile(self._dequantize_fp4_e2m1_impl)

        # 用來記錄歷史最小/最大值，保護 scale
        self._max_seen_val = 0.0
        self._min_seen_val = float('inf')
        self._fixed_scale = None

    @staticmethod
    def _build_fp4_e2m1_lut(bias: int):
        codes = torch.arange(16, dtype=torch.uint8)
        sign = ((codes >> 3) & 0x1).type(torch.float32)
        exp  = ((codes >> 1) & 0x3).type(torch.float32)
        mant = (codes & 0x1).type(torch.float32)
        base = torch.pow(2.0, exp - float(bias))
        val_normal = base * (1.0 + 0.5 * mant)
        val_sub = mant * (2.0 ** (-bias))
        is_sub = (exp == 0)
        val = torch.where(is_sub, val_sub, val_normal)
        val = torch.where((exp == 0) & (mant == 0), torch.zeros_like(val), val)
        val = torch.where(sign.bool(), -val, val)
        return val

    def _update_scale(self, x_bf16):
        max_input = torch.max(torch.abs(x_bf16)).item()
        min_input = torch.min(torch.abs(x_bf16[x_bf16 != 0])).item() if torch.any(x_bf16 != 0) else 0.0

        # 更新歷史最大/最小值
        self._max_seen_val = max(self._max_seen_val, max_input)
        self._min_seen_val = min(self._min_seen_val, min_input) if min_input > 0 else self._min_seen_val

        # 固定 scale，用 max(abs) 保護，避免飽和
        max_fp4 = torch.max(torch.abs(self._FP4_LUT)).item()
        self._fixed_scale = self._max_seen_val / max_fp4 if max_fp4 > 0 else 1.0

    def _quantize_bf16_to_fp4_e2m1_impl(self, x_bf16, scale: float):
        x = x_bf16.type(torch.float32) / float(scale)
        lut = self._FP4_LUT.to(x.device)
        max_val = torch.max(torch.abs(lut))
        x = torch.clamp(x, min=-max_val, max=max_val)

        # 逐元素找最接近 LUT 的值
        diffs = torch.abs(x.unsqueeze(-1) - lut.view(1, 1, 1, -1))
        codes = torch.argmin(diffs, dim=-1).type(torch.uint8)

        # pack 每兩個 FP4 進 uint8
        codes_pair = codes.view(*codes.shape[:-1], 64, 2)
        if self.EVEN_INDEX_IN_HIGH_NIBBLE:
            high = codes_pair[..., 0]
            low  = codes_pair[..., 1]
        else:
            high = codes_pair[..., 1]
            low  = codes_pair[..., 0]
        packed = (high << 4) | low
        return packed

    def _dequantize_fp4_e2m1_impl(self, packed_u8, scale: float):
        high = ((packed_u8 >> 4) & 0xF).type(torch.long)
        low  = (packed_u8 & 0xF).type(torch.long)
        if self.EVEN_INDEX_IN_HIGH_NIBBLE:
            codes = torch.stack([high, low], dim=-1)
        else:
            codes = torch.stack([low, high], dim=-1)
        codes = codes.view(*packed_u8.shape[:-1], 128)
        lut = self._FP4_LUT.to(packed_u8.device)
        vals = lut[codes] * float(scale)
        return vals.type(torch.bfloat16)

    # ===== 外部 API =====
    def quantize_bf16_to_fp4_e2m1(self, x_bf16):
        self._update_scale(x_bf16)
        return self._quantize_fn(x_bf16, self._fixed_scale)

    def dequantize_fp4_e2m1_to_bf16(self, packed_u8):
        if self._fixed_scale is None:
            raise RuntimeError("Scale has not been initialized by quantization yet.")
        return self._dequantize_fn(packed_u8, self._fixed_scale)

    # @torch.inference_mode()
    # def quantize_bf16_to_fp4_e2m1(self, tensor_bf16: torch.Tensor) -> torch.Tensor:
    #     return self._quantize_impl(tensor_bf16, self.FP4_VALUES_expand)

    # @torch.inference_mode()
    # def dequantize_fp4_e2m1_to_bf16(self, tensor_uint8: torch.Tensor) -> torch.Tensor:
    #     return self._dequantize_impl(tensor_uint8, self.FP4_VALUES)

    # @torch.compile
    # def quantize_bf16_to_fp4_e2m1(self, tensor_bf16: torch.Tensor) -> torch.Tensor:
    #     """
    #     tensor_bf16: [m, 8, 128], bfloat16
    #     output: [m, 8, 64], uint8, 每個uint8包2個fp4_e2m1
    #     """

    #     # 轉成float32計算
    #     x = tensor_bf16.to(torch.float32)  # [m,8,128]

    #     # 先處理 sign bit
    #     sign = (x < 0).to(torch.uint8)
    #     abs_x = torch.abs(x)

    #     # 處理log2 (避免log2(0))
    #     eps = 1e-8
    #     log2_abs = torch.log2(abs_x + eps)

    #     # 計算 exponent bits (clamp在0~3)
    #     exp_val = torch.clamp(torch.floor(log2_abs).to(torch.int32) + EXP_BIAS, 0, 3)

    #     # 計算 mantissa bits
    #     base = 2.0 ** (exp_val.to(torch.float32) - EXP_BIAS)
    #     mantissa_val = abs_x / base - 1.0
    #     mantissa = (mantissa_val >= 0.5).to(torch.uint8)

    #     # 如果 x==0，全部設為0
    #     zero_mask = (abs_x < eps)
    #     sign[zero_mask] = 0
    #     exp_val[zero_mask] = 0
    #     mantissa[zero_mask] = 0

    #     # pack bits: s e1 e0 m
    #     fp4 = (sign << 3) | (exp_val << 1) | mantissa  # shape [m,8,128], uint8

    #     # 每2個4-bit合成1個byte
    #     fp4_reshaped = fp4.view(*fp4.shape[:-1], 64, 2)  # [m,8,64,2]

    #     high = fp4_reshaped[..., 0].to(torch.uint8)
    #     low = fp4_reshaped[..., 1].to(torch.uint8)

    #     packed = (high << 4) | low  # [m,8,64]

    #     return packed


    # @torch.compile
    # def dequantize_fp4_e2m1_to_bf16(self, tensor_uint8: torch.Tensor) -> torch.Tensor:
    #     """
    #     tensor_uint8: [m,8,64], uint8
    #     output: [m,8,128], bfloat16
    #     """

    #     # 拆成兩個4-bit
    #     high = (tensor_uint8 >> 4) & 0xF
    #     low = tensor_uint8 & 0xF

    #     fp4 = torch.stack([high, low], dim=-1)  # [m,8,64,2]

    #     # 轉回128維
    #     fp4 = fp4.view(*tensor_uint8.shape[:-1], 128)  # [m,8,128]

    #     # 解碼 fp4 e2m1
    #     sign = (fp4 >> 3) & 0x1
    #     exponent = (fp4 >> 1) & 0x3
    #     mantissa = fp4 & 0x1

    #     base = 2.0 ** (exponent.to(torch.float32) - EXP_BIAS)
    #     val = base * (1.0 + mantissa.to(torch.float32) * 0.5)

    #     val = val * (1 - 2 * sign.to(torch.float32))  # sign bit轉成正負號

    #     return val.to(torch.bfloat16)


    # @torch.compile
    # def quantize_bf16_to_fp4_e2m1(self, x_bf16: torch.Tensor) -> torch.Tensor:
    #     """
    #     輸入: [m, 8, 128]、bfloat16
    #     輸出: [m, 8, 64]、uint8；每個 uint8 依序封兩個 fp4（高4位=偶數索引，低4位=奇數索引）
    #     """
    #     assert x_bf16.dim() == 3 and x_bf16.shape[-1] == 128 and x_bf16.shape[-2] == 8
    #     x = x_bf16.to(torch.float32)

    #     # 基本元件
    #     absx = torch.abs(x)
    #     sign_bit = (x < 0).to(torch.uint8)
    #     # NaN 當作 0 處理
    #     is_nan = torch.isnan(absx)
    #     absx = torch.where(is_nan, torch.zeros_like(absx), absx)
    #     sign_bit = torch.where((absx == 0), torch.zeros_like(sign_bit), sign_bit)

    #     # 分區：0 / 次正規 / 正規（含飽和）
    #     zero_mask = absx < _SUB_HALF
    #     sub_mask  = (~zero_mask) & (absx < _MIN_NORMAL)
    #     norm_mask = absx >= _MIN_NORMAL

    #     # 正規數：求 unbiased exponent（避免 log2(0)）
    #     safe_abs = torch.clamp(absx, min=_MIN_NORMAL)
    #     e_unbiased = torch.floor(torch.log2(safe_abs)).to(torch.int32)          # e (可負)
    #     e_field    = torch.clamp(e_unbiased + BIAS, 1, 3).to(torch.int32)       # 指數欄位 1..3

    #     # mantissa 選擇（就近；臨界點 1.25 採 >= 往上）
    #     base      = torch.pow(2.0, e_unbiased.to(torch.float32))                # 2^e
    #     y_norm    = absx / base                                                 # ∈ [1, 2)
    #     m_norm    = (y_norm >= _NORM_THR).to(torch.uint8)                       # 0:1.0, 1:1.5

    #     # 飽和（非常大時 e_field 會被夾到 3，m_norm=1 即對應到最大值 3.0）
    #     # 次正規：只有 mantissa=1 的一個值（_SUB_STEP），四捨五入門檻 _SUB_HALF
    #     m_field = torch.zeros_like(m_norm, dtype=torch.uint8)
    #     exp_field = torch.zeros_like(e_field, dtype=torch.int32)

    #     # 正規填入
    #     exp_field = torch.where(norm_mask, e_field, exp_field)
    #     m_field   = torch.where(norm_mask, m_norm, m_field)
    #     # 次正規填入（mantissa=1）
    #     m_field   = torch.where(sub_mask, torch.ones_like(m_field), m_field)
    #     # 零保留 0

    #     # 組 4-bit： [sign | exp(2b) | mant(1b)]
    #     fp4_nibble = (sign_bit << 3) | (exp_field.to(torch.uint8) << 1) | m_field  # [m,8,128] uint8 (值 0..15)

    #     # 打包成 uint8：偶數索引 -> 高半 byte；奇數索引 -> 低半 byte
    #     fp4_pairs = fp4_nibble.reshape(*fp4_nibble.shape[:-1], 64, 2)  # [m,8,64,2]
    #     high = fp4_pairs[..., 0]
    #     low  = fp4_pairs[..., 1]
    #     packed = (high << 4) | low
    #     return packed.to(torch.uint8)


    # @torch.compile
    # def dequantize_fp4_e2m1_to_bf16(self, packed_u8: torch.Tensor) -> torch.Tensor:
    #     """
    #     輸入: [m, 8, 64]、uint8
    #     輸出: [m, 8, 128]、bfloat16
    #     """
    #     assert packed_u8.dim() == 3 and packed_u8.shape[-1] == 64 and packed_u8.shape[-2] == 8

    #     # 拆成兩個 4-bit（高半、低半）
    #     high = (packed_u8 >> 4) & 0xF
    #     low  = packed_u8 & 0xF
    #     fp4 = torch.stack([high, low], dim=-1).reshape(*packed_u8.shape[:-1], 128)  # [m,8,128], uint8 (0..15)

    #     sign = (fp4 >> 3) & 0x1
    #     exp  = (fp4 >> 1) & 0x3
    #     mant = fp4 & 0x1

    #     exp_f = exp.to(torch.float32)
    #     mant_f = mant.to(torch.float32)

    #     # 正規：2^(exp-BIAS) * (1 + mant*0.5)
    #     base = torch.pow(2.0, exp_f - float(BIAS))
    #     val_normal = base * (1.0 + 0.5 * mant_f)

    #     # 次正規/零：exp==0 -> value = (mant/2) * 2^(1-BIAS) = mant * 2^(-BIAS)
    #     val_sub = mant_f * float(2.0 ** (-BIAS))

    #     is_sub = (exp == 0)
    #     val = torch.where(is_sub, val_sub, val_normal)

    #     # 套用符號（零維持零）
    #     val = torch.where((exp == 0) & (mant == 0), torch.zeros_like(val), val)  # 確保 +0/-0 -> +0
    #     val = torch.where(sign.bool(), -val, val)

    #     return val.to(torch.bfloat16)


    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                hcdprint(f"[horenc] class MHATokenToKVPool:_create_buffers(): self.store_dtype = {self.store_dtype}")
                if self.store_dtype == torch.float4_e2m1fn_x2:
                    hcdprint(f"[horenc] class MHATokenToKVPool:_create_buffers(): Jack hack - "
                            f"force torch.uint8 + dividiveby2 for matching fp4 shape[42, 8, 64] (ori:[42, 8, 128])")  
                    m = self.size + self.page_size
                    n = self.head_num
                    k = self.head_dim
                    hcdprint(f"[horenc] class MHATokenToKVPool:_create_buffers(): Jack hack - "
                            f"Milestone1 layout: m={m}, n={n}, k={k}, per-layer buffer_size = {(m) * (n) * (k // 2)}")
                    self.k_buffer = [
                        torch.zeros( # TODO change this 3D to 1D
                            (self.size + self.page_size, self.head_num, self.head_dim // 2), # 3D
                            dtype=torch.uint8,
                            device=self.device,
                        )
                        # TODO: Change to something like this torch.zeros( MNK),  dtype=torch.uint8,device=self.device,)
                        for _ in range(self.layer_num)
                    ]
                    # loc now = 3D, TODO: change it to 1D
                    # [layer_id][loc] => [loc]==[ self.size + self.page_size, 2D(head), 3D(dim)]
                    # self.size + self.page_size = M
                    # self.head_num = N
                    # self.head_dim = K
                    # Use yichen's formula to calculate 1D's k/v buffer size
                    self.v_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.head_dim // 2),
                            dtype=torch.uint8,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

                    if USE_KV_MXFP4:
                        m = self.size + self.page_size
                        n = self.head_num
                        k = self.head_dim
                        scale_block_size = 32
                        assert (n * k) % scale_block_size == 0, f"n*k ({n*k}) must be divisible by scale_block_size ({scale_block_size})"

                        # debug
                        scale_buffer_size = m * n * (k // scale_block_size)
                        hcdprint(f"[horenc] class MHATokenToKVPool:_create_buffers(): Jack - "
                        f"Allocate KV scale buffer layout: m={m}, n={n}, k={k}, per-layer buffer_size = {scale_buffer_size}")
                        
                        self.k_scale_buffer = [
                            torch.zeros(
                                (m, n, k // scale_block_size),
                                dtype=torch.uint8,
                                device=self.device,
                            )
                            for _ in range(self.layer_num)
                        ]
                        self.v_scale_buffer = [
                            torch.zeros(
                                (m, n, k // scale_block_size),
                                dtype=torch.uint8,
                                device=self.device,
                            )
                            for _ in range(self.layer_num)
                        ]
                else: # origin
                    # [size, head_num, head_dim] for each layer
                    # The padded slot 0 is used for writing dummy outputs from padded tokens.
                    self.k_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool

    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        torch.cuda.synchronize()

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        # hcdprint(f"[horenc] going to hack MHATokenToKVPool get_key_buffer()")
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != torch.float4_e2m1fn_x2:
            # ori
            return self._get_key_buffer(layer_id)
        else:
            if USE_KV_MXFP4:
                from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil
                # cache_k = MXFP4QuantizeUtil.quantize_packed(self._get_key_buffer(layer_id), 32)
                hcdprint(f"\t [horenc] MHATokenToKVPool:get_key_buffer()  DQ layer_id = {layer_id}, self.dtype = {self.dtype}")
                hcdprint(f"\t [horenc]  DQ bf-Dequantized k shape: {self._get_key_buffer(layer_id).shape}, "
                        f"dtype: {self._get_key_buffer(layer_id).dtype}")
                dequantized_cache_k = MXFP4QuantizeUtil.dequantize_tokenwise(
                                            # q_bytes=self._get_key_buffer(layer_id),
                                            q_bytes=self.k_buffer[layer_id - self.start_layer].view(torch.uint8), # expect uint8
                                            scales=self.k_scale_buffer[layer_id - self.start_layer], # expect uint8, is uint8
                                            # e8m0_scale=self.k_scale_buffer[layer_id - self.start_layer], # expect uint8, is uint8
                                            # e8m0_scale=self.k_scale_buffer[layer_id - self.start_layer].view(self.dtype),
                                            block_size=32
                                        )
                # myfunc = vmap(lambda t: MXFP4QuantizeUtil.dequantize_packed(
                #                             quantized_data=t,
                #                             dtype=torch.bfloat16,
                #                             block_sizes=[32]
                #                         ), in_dims=0, out_dims=0)
                # out_list = []
                # for i in range(self._get_key_buffer(layer_id).size(0)):
                #     dequant = MXFP4QuantizeUtil.dequantize_packed(
                #                             quantized_data=self._get_key_buffer(layer_id)[i],
                #                             dtype=torch.bfloat16,
                #                             block_sizes=[32]
                #                         )
                #     if dequant == None:
                #         dequant = torch.zeros(self.head_num, self.head_dim, device=self.device)
                #     out_list.append(dequant)
                # dequantized_cache_k = torch.stack(out_list, dim = 0)
                

                # dequantized_cache_k = MXFP4QuantizeUtil.dequantize_packed(
                #                             quantized_data=self._get_key_buffer(layer_id),
                #                             dtype=torch.bfloat16,
                #                             block_sizes=[32]
                #                         )

                hcdprint(f"\t [horenc]  DQ af-Dequantized k shape: {dequantized_cache_k.shape}, "
                        f"dtype: {dequantized_cache_k.dtype}")
                return dequantized_cache_k
            else:
                # horenc
                hcdprint(f"\t [horenc] MHATokenToKVPool:get_key_buffer()  DQ layer_id = {layer_id}, self.dtype = {self.dtype}")
                hcdprint(f"\t [horenc]  DQ bf-Dequantized k shape: {self._get_key_buffer(layer_id).shape}")
                hcdprint(f"\t [horenc]  DQ bf-Dequantized k dtype: {self._get_key_buffer(layer_id).dtype}")
                dequantized_cache_k = self.dequantize_fp4_e2m1_to_bf16(self._get_key_buffer(layer_id))
                hcdprint(f"\t [horenc]  DQ af-Dequantized k shape {dequantized_cache_k.shape}")
                hcdprint(f"\t [horenc]  DQ af-Dequantized k dtype: {dequantized_cache_k.dtype}")
                return dequantized_cache_k

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        # hcdprint(f"[horenc] going to hack MHATokenToKVPool get_value_buffer()")
        hcdprint(f"\t [horenc] MHATokenToKVPool:get_value_buffer()   layer_id = {layer_id}, self.dtype = {self.dtype}")
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        if self.store_dtype != torch.float4_e2m1fn_x2:
            # ori
            return self._get_value_buffer(layer_id)
        else:
            if USE_KV_MXFP4:
                from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil
                # cache_k = MXFP4QuantizeUtil.quantize_packed(self._get_value_buffer(layer_id), 32)
                hcdprint(f"\t [horenc]   bf-Dequantized v shape: {self._get_value_buffer(layer_id).shape}, "
                            f"dtype: {self._get_value_buffer(layer_id).dtype}")
                # dequantized_cache_v = MXFP4QuantizeUtil.dequantize_packed(
                #                             quantized_data=self._get_value_buffer(layer_id),
                #                             dtype=torch.bfloat16,
                #                             block_sizes=[32]
                #                         )
                dequantized_cache_v = MXFP4QuantizeUtil.dequantize_tokenwise(
                                            # q_bytes=self._get_value_buffer(layer_id),
                                            q_bytes=self.v_buffer[layer_id - self.start_layer].view(torch.uint8),  # expect uint8
                                            scales=self.v_scale_buffer[layer_id - self.start_layer],  # expect uint8, is uint8
                                            # e8m0_scale=self.v_scale_buffer[layer_id - self.start_layer],  # expect uint8, is uint8
                                            # e8m0_scale=self.v_scale_buffer[layer_id - self.start_layer].view(self.dtype),
                                            block_size=32
                                        )
                hcdprint(f"\t [horenc]   af-Dequantized v shape: {dequantized_cache_v.shape}, "
                             f"dtype: {dequantized_cache_v.dtype}")
                return dequantized_cache_v
            else:
                # horenc
                hcdprint(f"\t [horenc]   layer_id = {layer_id}")
                hcdprint(f"\t [horenc]   bf-Dequantized v shape: {self._get_value_buffer(layer_id).shape}, "
                            f"dtype: {self._get_value_buffer(layer_id).dtype}")
                dequantized_cache_v = self.dequantize_fp4_e2m1_to_bf16(self._get_value_buffer(layer_id))
                hcdprint(f"\t [horenc]   af-Dequantized v shape: {dequantized_cache_v.shape}, "
                            f"dtype: {dequantized_cache_v.dtype}")
                return dequantized_cache_v

    def get_kv_buffer(self, layer_id: int):
        # hcdprint(f"[horenc] MHATokenToKVPool:get_kv_buffer() some attention do not enter here like torch_native_backend")
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)


    # # # ----------------------
    # # # Optimize - pytorch
    # # # ----------------------
    # # @torch.inference_mode() 
    # # def quantize_bf16_to_fp4_e2m1(self, tensor_bf16: torch.Tensor) -> torch.Tensor:
    # # def _quantize_impl(self, tensor_bf16: torch.Tensor) -> torch.Tensor:
    # @staticmethod
    # def _quantize_impl(tensor_bf16, FP4_VALUES_expand):
    #     shape = tensor_bf16.shape
    #     last_dim = shape[-1]
    #     assert last_dim % 2 == 0, "最後維度必須是偶數"

    #     tensor_f32 = tensor_bf16.float().clamp(-6.0, 6.0)
    #     tensor_2d = tensor_f32.view(-1, last_dim)  # [N, last_dim]

    #     # 距離計算
    #     # distances = torch.abs(tensor_2d.unsqueeze(-1) - self.FP4_VALUES_expand)  # [N, last_dim, 16]
    #     distances = torch.abs(tensor_2d.unsqueeze(-1) - FP4_VALUES_expand)  # [N, last_dim, 16]
    #     indices = distances.argmin(dim=-1).to(torch.uint8)  # [N, last_dim]

    #     # pack 兩個4bit成1byte, 避免 concat, 直接用切片操作
    #     # [N, last_dim//2], 直接取奇偶位組合
    #     packed_uint8 = (indices[:, 0::2] & 0x0F) | ((indices[:, 1::2] & 0x0F) << 4)

    #     new_shape = list(shape[:-1]) + [last_dim // 2]
    #     return packed_uint8.contiguous().view(*new_shape)

    # # @torch.inference_mode()
    # # def dequantize_fp4_e2m1_to_bf16(self, tensor_uint8: torch.Tensor) -> torch.Tensor:
    # # def _dequantize_impl(self, tensor_uint8: torch.Tensor) -> torch.Tensor:    
    # @staticmethod
    # def _dequantize_impl(tensor_uint8, FP4_VALUES):
    #     shape = tensor_uint8.shape
    #     last_dim = shape[-1]

    #     tensor_2d = tensor_uint8.view(-1, last_dim)  # [N, last_dim]

    #     low_nibble = tensor_2d & 0x0F  # [N, last_dim]
    #     high_nibble = (tensor_2d >> 4) & 0x0F  # [N, last_dim]

    #     # 利用拼接的新維度，交錯合併 (interleave) 兩個 nibble
    #     # 預先建立空 tensor, 並利用索引賦值避免 concat 和 stack 開銷
    #     N = tensor_2d.shape[0]
    #     out_len = last_dim * 2
    #     out_indices = torch.empty((N, out_len), dtype=torch.long, device=tensor_uint8.device)

    #     out_indices[:, 0::2] = low_nibble
    #     out_indices[:, 1::2] = high_nibble

    #     # values = self.FP4_VALUES[out_indices]  # [N, last_dim*2]
    #     values = FP4_VALUES[out_indices]  # [N, last_dim*2]

    #     new_shape = list(shape[:-1]) + [last_dim * 2]
    #     return values.contiguous().view(*new_shape).to(torch.bfloat16)


    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        hcdprint(f"[horenc] MHATokenToKVPool:set_kv_buffer() cache_k.dtype = {cache_k.dtype}")
        hcdprint(f"[horenc] MHATokenToKVPool:set_kv_buffer() self.dtype = {self.dtype}")
        hcdprint(f"[horenc] MHATokenToKVPool:set_kv_buffer() k_scale = {k_scale}")
        # hcdprint(f"\t [horenc]  Quantized k/v check")
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            if self.dtype != torch.float4_e2m1fn_x2:
                # origin
                cache_k = cache_k.to(self.dtype)
                cache_v = cache_v.to(self.dtype)
            else:
                # new - yichen's quant
                # hcdprint(f"[horenc] hack .to -> scaled_fp4_quant()")
                # hcdprint(f"[horenc] cache_k.shape: {list(cache_k.shape)}")
                hcdprint(f"\t [horenc]  Q layer_id ({layer_id}) - self.start_layer ({self.start_layer})  = {layer_id - self.start_layer}, loc = {loc}")
                hcdprint(f"\t [horenc]  Q bf-Quantize k/v shape: {cache_k.shape}, dtype: {cache_k.dtype}")
                # [7, 8, 128] = 7 * 8 * 128 = 7,168
                if USE_KV_MXFP4:
                    from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil
                    # wrapped = functionalize(lambda t: MXFP4QuantizeUtil.quantize_packed(t.clone(), 32))
                    # myfunc = vmap(wrapped, in_dims=0, out_dims=0)
                    # # myfunc = torch.compile(myfunc)
                    # cache_k = myfunc(cache_k)
                    # cache_v = myfunc(cache_v)

                    # cache_k = torch.stack([MXFP4QuantizeUtil.quantize_packed(cache_k[i], 32) for i in range(cache_k.size(0))], dim=0)
                    # cache_v = torch.stack([MXFP4QuantizeUtil.quantize_packed(cache_v[i], 32) for i in range(cache_v.size(0))], dim=0)

                    cache_k, cache_k_scale = MXFP4QuantizeUtil.quantize_tokenwise(cache_k, 32)
                    cache_v, cache_v_scale = MXFP4QuantizeUtil.quantize_tokenwise(cache_v, 32)

                    # TODO: Save scale
                    hcdprint(f"\t [horenc]  Q af-Quantized (new) k/v scale shape: {cache_k_scale.shape}, dtype: {cache_k_scale.dtype}")
                    # 看是不是/32, yes 7* 8 * 128 / 32= [224]
                    # [要把這些存成怎樣]
                    self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_scale
                    self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_scale
                else:
                    # new 250809 - working
                    cache_k = self.quantize_bf16_to_fp4_e2m1(cache_k)
                    cache_v = self.quantize_bf16_to_fp4_e2m1(cache_v)
                hcdprint(f"\t [horenc]  Q af-Quantized k/v shape: {cache_k.shape}, dtype: {cache_k.dtype}")
                # real:
                #   [224, 16] = [7,8,128]=>[7168]   /32 => [224, 42]
                # expect:
                #   [7, 8, 128/2] = 7 * 8 * 64 = 3584
                hcdprint(f"\t [horenc]  Q self.k_buffer[layer_id - self.start_layer].shape = {self.k_buffer[layer_id - self.start_layer].shape}")
                hcdprint(f"\t [horenc]  Q self.k_buffer[layer_id - self.start_layer][loc].shape = {self.k_buffer[layer_id - self.start_layer][loc].shape}")  

        # hcdprint("[debug] cache_k.shape =", cache_k.shape)
        # hcdprint("[debug] target shape =", self.k_buffer[layer_id - self.start_layer][loc].shape)
            
        hcdprint(f"[horenc] MHATokenToKVPool:set_kv_buffer() self.store_dtype = {self.store_dtype}")
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if get_is_capture_mode() and self.alt_stream is not None:
            # Overlap the copy of K and V cache for small batch size
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[layer_id - self.start_layer][loc] = cache_v
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v


    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        copy_all_layer_kv_cache[(len(self.data_ptrs),)](
            self.data_ptrs,
            self.data_strides,
            tgt_loc,
            src_loc,
            len(tgt_loc),
            next_power_of_2(len(tgt_loc)),
        )


class SWAKVPool(KVCache):
    """KV cache with separate pools for full and SWA attention layers."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
    ):
        self.size = size
        self.size_swa = size_swa
        self.dtype = dtype
        self.device = device
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = 1
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose
        TokenToKVPoolClass = MHATokenToKVPool
        self.swa_kv_pool = TokenToKVPoolClass(
            size=size_swa,
            page_size=self.page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=self.swa_layer_nums,
            device=device,
            enable_memory_saver=False,
        )
        self.full_kv_pool = TokenToKVPoolClass(
            size=size,
            page_size=self.page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=self.full_layer_nums,
            device=device,
            enable_memory_saver=False,
        )
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for full_attn_layer_id, global_layer_id in enumerate(full_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (full_attn_layer_id, False)
        for swa_layer_id, global_layer_id in enumerate(swa_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (swa_layer_id, True)
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        k_size, v_size = self.full_kv_pool.get_kv_size_bytes()
        k_size_swa, v_size_swa = self.swa_kv_pool.get_kv_size_bytes()
        return k_size + k_size_swa, v_size + v_size_swa

    def get_contiguous_buf_infos(self):
        full_kv_data_ptrs, full_kv_data_lens, full_kv_item_lens = (
            self.full_kv_pool.get_contiguous_buf_infos()
        )
        swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens = (
            self.swa_kv_pool.get_contiguous_buf_infos()
        )

        kv_data_ptrs = full_kv_data_ptrs + swa_kv_data_ptrs
        kv_data_lens = full_kv_data_lens + swa_kv_data_lens
        kv_item_lens = full_kv_item_lens + swa_kv_item_lens

        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_key_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_key_buffer(layer_id_pool)

    def get_value_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_value_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_value_buffer(layer_id_pool)

    def get_kv_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_kv_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_kv_buffer(layer_id_pool)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):

        layer_id = layer.layer_id
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            if self.full_to_swa_index_mapping is not None:
                loc = self.translate_loc_from_full_to_swa(loc)
            self.swa_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )
        else:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )


class AscendTokenToKVPool(MHATokenToKVPool):

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            # Continuous memory improves the efficiency of Ascend`s transmission backend,
            # while other backends remain unchanged.
            hcdprint(f"[horenc] AscendTokenToKVPool CREATE kv$ buffer XXXXXXX")
            self.kv_buffer = torch.zeros(
                (
                    2,
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.k_buffer = self.kv_buffer[0]
            self.v_buffer = self.kv_buffer[1]

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self.get_key_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        import torch_npu

        torch_npu._npu_reshape_and_cache(
            key=cache_k,
            value=cache_v,
            key_cache=self.k_buffer[layer_id].view(
                -1, self.page_size, self.head_num, self.head_dim
            ),
            value_cache=self.v_buffer[layer_id].view(
                -1, self.page_size, self.head_num, self.head_dim
            ),
            slot_indices=loc,
        )


@triton.jit
def set_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    loc = tl.load(loc_ptr + pid_loc)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
        )
    else:
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
        )

    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    set_mla_kv_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
    )


@triton.jit
def set_mla_kv_scale_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim  # Make sure don't cross the boundary

    loc = tl.load(loc_ptr + pid_loc)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    # Check each offs should read 'nope' or 'rope'
    is_nope = offs < nope_dim
    src_nope = tl.load(
        cache_k_nope_ptr + pid_loc * nope_stride + offs,
        mask=mask & is_nope,
        other=0.0
    )
    src_rope = tl.load(
        cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim),
        mask=mask & ~is_nope,
        other=0.0
    )

    # Combine nope + rope
    src = src_nope + src_rope
    tl.store(dst_ptr, src, mask=mask)

def set_mla_kv_scale_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128  # Keep origin, works for smaller total_dim as well.
    n_loc = loc.numel() 
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    set_mla_kv_scale_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
    )


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
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

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        # for disagg with nvlink
        self.enable_custom_mem_pool = get_bool_env_var(
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "false"
        )
        if self.enable_custom_mem_pool:
            # TODO(shangming): abstract custom allocator class for more backends
            from mooncake.allocator import NVLinkAllocator

            allocator = NVLinkAllocator.get_allocator(self.device)
            self.custom_mem_pool = torch.cuda.MemPool(allocator.allocator())
        else:
            self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # TODO
                hcdprint(f"[horenc] TODO:MLA TODO:MLA MLATokenToKVPool CREATE kv$ buffer 1")
                if self.dtype != torch.float4_e2m1fn_x2:
                    # The padded slot 0 is used for writing dummy outputs from padded tokens.
                    self.kv_buffer = [
                        torch.zeros(
                            (size + page_size, 1, kv_lora_rank + qk_rope_head_dim),
                            dtype=self.store_dtype,
                            device=device,
                        )
                        for _ in range(layer_num)
                    ]
                else:
                    hcdprint(f"[horenc] TODO:MLA TODO:MLA USE uint8 // 2 self.store_dtype = {self.store_dtype}")
                    m = size + page_size
                    n = 1 # head_num
                    # k = self.head_dim
                    k = kv_lora_rank + qk_rope_head_dim # head_dim # k = 576 (512 + 64)
                    hcdprint(f"[horenc] TODO:MLA __init__ create kv kv_scale buffer m = {m} ({size} + {page_size}) n = {n} k = {k} ({kv_lora_rank} + {qk_rope_head_dim})")

                    scale_block_size = 16
                    self.store_dtype = torch.uint8

                    self.kv_buffer = [
                        torch.zeros(
                            # (size + page_size, 1, kv_lora_rank + qk_rope_head_dim),
                            (m, n, k // 2),
                            dtype=self.store_dtype,
                            device=device,
                        )
                        for _ in range(layer_num)
                    ]

                    # from sglang.srt.utils import get_available_gpu_memory
                    # sigfault
                    # logger.info(
                    #     f"Memory pool middle1. "
                    #     f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
                    # )
                    import math
                    self.kv_scale_buffer = [
                        torch.zeros(
                            # (size + page_size, 1, kv_lora_rank + qk_rope_head_dim),
                            # (m, n, k // scale_block_size), # horenc25827 for debug [m, k] => [m, 1, k]
                            (m, k // scale_block_size), # horenc25827 match quant/dequant function [m, 1, k] => [m, k]
                            # bateched quant shape
                            # (m, math.ceil(n / 128) * 128 * math.ceil(k / 16 / 4) * 4),
                            # dtype=self.store_dtype, # store_dtype = fp4
                            dtype=torch.uint8,
                            device=device,
                        )
                        for _ in range(layer_num)
                    ]
                    # hcdprint(f"[horenc] TODO:MLA TODO:MLA TODO:MLA for SET2 3rd global_sf")
                    # hcdprint(f"learn from this way: set_mla_kv_buffer_triton(self.kv_buffer[layer_id], loc, cache_k_nope_fp4, cache_k_rope_fp4)")
                    # hcdprint(f"Not sure if need to + page_size (1) ")
                    # self.kv_globalsf_buffer = [
                    #     torch.zeros(
                    #         (2),
                    #         # 2 + page_size, # because of set_mla_kv_buffer_triton(..., k_nope, k_rope) # Need to check if how this func reads the buffer. 
                    #         dtype=torch.float32,
                    #         device=device,
                    #     )
                    #     for _ in range(layer_num)
                    # ]
                    # # set_mla_kv_buffer_triton(
                    # #     self.kv_globalsf_buffer[layer_id], loc, k_nope_global_sf, k_rope_global_sf
                    # # )

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.layer_transfer_counter = None

        kv_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, KV size: {kv_size / GB:.2f} GB"
        )
        self.mem_usage = kv_size / GB

    def get_kv_size_bytes(self):
        # if self.dtype != torch.float4_e2m1fn_x2:
        hcdprint(f"[horenc] TODO:MLA:CHECK MLATokenToKVPool CREATE kv$ buffer related - "
              f"get_kv_size_bytes()")
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += np.prod(kv_cache.shape) * kv_cache.dtype.itemsize
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        if self.dtype != torch.float4_e2m1fn_x2:
            hcdprint(f"kv4 doesn't support disagg yet")
        hcdprint(f"[horenc] TODO:MLA:CHECK MLATokenToKVPool CREATE kv$ buffer related - "
                f"get_contiguous_buf_infos()")
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        # Get kv together
        hcdprint(f"[horenc]({layer_id}) class MLATokenToKVPool:get_key_buffer(): GET key"
                f" {self.store_dtype} != {self.dtype}")
        # 16: bfloat16 != bfloat16
        # 8: torch.uint8 != torch.float8_e4m3fn
        if self.store_dtype != self.dtype:
            if self.dtype != torch.float4_e2m1fn_x2:
                return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)
            else:
                hcdprint(f"[horenc]({layer_id}) GET key but actually get kv at the same time")
                hcdprint(f"[horenc]({layer_id}) MLA GET kv - self.kv_buffer[layer_id - self.start_layer].dtype = {self.kv_buffer[layer_id - self.start_layer].dtype} .shape = {self.kv_buffer[layer_id - self.start_layer].shape}")
                # cache_k_nope_fp4 = self.kv_buffer[layer_id - self.start_layer].view(self.dtype) # k_nope + k_rope = cache_k_nope_fp4, cache_k_rope_fp4
                cache_k_nope_fp4 = self.kv_buffer[layer_id - self.start_layer].view(torch.uint8) # k_nope + k_rope = cache_k_nope_fp4, cache_k_rope_fp4
                # cache_k_nope_fp4_sf = self.kv_scale_buffer[layer_id - self.start_layer].view(self.dtype) # scale of (k_nope + k_rope) = cache_k_nope_fp4_sf, cache_k_rope_fp4_sf
                # cache_k_nope_fp4_sf = self.kv_scale_buffer[layer_id - self.start_layer].view(torch.uint8) # scale of (k_nope + k_rope) = cache_k_nope_fp4_sf, cache_k_rope_fp4_sf
                cache_k_nope_fp4_sf = self.kv_scale_buffer[layer_id - self.start_layer] # follow dequant and buffer init = uint8   hcdprint(f"[horenc]({layer_id}) TODO:MLA GET kv - cache_k_nope_fp4.dtype = {cache_k_nope_fp4.dtype} cache_k_nope_fp4.shape = {cache_k_nope_fp4.shape}")
                hcdprint(f"[horenc]({layer_id}) 1 MLA GET kv - cache_k_nope_fp4.dtype = {cache_k_nope_fp4.dtype} cache_k_nope_fp4.shape = {cache_k_nope_fp4.shape} type(cache_k_nope_fp4) = {type(cache_k_nope_fp4)}")
                hcdprint(f"[horenc]({layer_id}) 2 MLA GET kv - cache_k_nope_fp4_sf.dtype = {cache_k_nope_fp4_sf.dtype} cache_k_nope_fp4_sf.shape = {cache_k_nope_fp4_sf.shape} type(cache_k_nope_fp4_sf) = {type(cache_k_nope_fp4_sf)}")

                from sglang.srt.layers.quantization.nvfp4_tensor import KVFP4QuantizeUtil # need to delayed import
                # cache_k_nope_fp4_sf = cache_k_nope_fp4_sf.squeeze(1) # horenc25827 for debug [m, 1, k] -> [m, k]


                # DEBUG kv_buffer
                if layer_id == 0 or layer_id == 1:
                    hcdprint(f"{layer_id} Jack DEBUG GET - cache_k_nope_fp4[29] shape: {cache_k_nope_fp4[29].shape}, "
                            f"cache_k_nope_fp4[29] values: {cache_k_nope_fp4[29]}") # shape: [1, 36]
                    hcdprint(f"{layer_id} Jack DEBUG GET - cache_k_nope_fp4[30] shape: {cache_k_nope_fp4[30].shape}, "
                            f"cache_k_nope_fp4[30] values: {cache_k_nope_fp4[30]}") # shape: [1, 36]
                    hcdprint(f"{layer_id} Jack DEBUG GET - cache_k_nope_fp4[31] shape: {cache_k_nope_fp4[31].shape}, "
                            f"cache_k_nope_fp4[31] values: {cache_k_nope_fp4[31]}") # shape: [1, 36]

                # Debug e.g., (0) loc = tensor([29
                # Debug e.g., (0) loc = tensor([29
                # Debug e.g., (0) loc = tensor([29
                # if layer_id == 0 and loc.numel() == 1 and int(loc.item()) == 29:
                # if layer_id == 0:
                if layer_id == 0 or layer_id == 1:
                    # hcdprint(f"Jack DEBUG GET - cache_k_nope_fp4_sf = {cache_k_nope_fp4_sf}")
                    # hcdprint(f"Jack DEBUG GET - cache_k_nope_fp4_sf[29] = {cache_k_nope_fp4_sf[29]}")
                    # slot_tensor = cache_k_nope_fp4_sf[29]  # shape: [1, 36]
                    # hcdprint(f"{layer_id} Jack DEBUG GET - cache_k_nope_fp4_sf[1] shape: {cache_k_nope_fp4_sf[1].shape}" # shape: [1, 36]
                    #         f"cache_k_nope_fp4_sf[1] values: {cache_k_nope_fp4_sf[1]}")
                    # hcdprint(f"{layer_id} Jack DEBUG GET - cache_k_nope_fp4_sf[2] shape: {cache_k_nope_fp4_sf[2].shape}, "  # shape: [1, 36]
                    #         f"cache_k_nope_fp4_sf[2] values: {cache_k_nope_fp4_sf[2]}")
                    hcdprint(f"{layer_id} Jack DEBUG GET - cache_k_nope_fp4_sf[16] shape: {cache_k_nope_fp4_sf[16].shape}, "  # shape: [1, 36]
                            f"cache_k_nope_fp4_sf[16] values: {cache_k_nope_fp4_sf[16]}")
                    hcdprint(f"{layer_id} Jack DEBUG GET - cache_k_nope_fp4_sf[17] shape: {cache_k_nope_fp4_sf[17].shape}, "  # shape: [1, 36]
                            f"cache_k_nope_fp4_sf[17] values: {cache_k_nope_fp4_sf[17]}")
                    hcdprint(f"{layer_id} Jack DEBUG GET - cache_k_nope_fp4_sf[18] shape: {cache_k_nope_fp4_sf[18].shape}, "  # shape: [1, 36]
                            f"cache_k_nope_fp4_sf[18] values: {cache_k_nope_fp4_sf[18]}")
                    hcdprint(f"{layer_id} Jack DEBUG GET - cache_k_nope_fp4_sf[29] shape: {cache_k_nope_fp4_sf[29].shape}, "
                            f"cache_k_nope_fp4_sf[29] values: {cache_k_nope_fp4_sf[29]}") # shape: [1, 36]
                    hcdprint(f"{layer_id} Jack DEBUG GET - cache_k_nope_fp4_sf[30] shape: {cache_k_nope_fp4_sf[30].shape}, "
                            f"cache_k_nope_fp4_sf[30] values: {cache_k_nope_fp4_sf[30]}") # shape: [1, 36]
                    hcdprint(f"{layer_id} Jack DEBUG GET - cache_k_nope_fp4_sf[31] shape: {cache_k_nope_fp4_sf[31].shape}, "
                            f"cache_k_nope_fp4_sf[31] values: {cache_k_nope_fp4_sf[31]}") # shape: [1, 36]

                cache_k_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                    cache_k_nope_fp4, cache_k_nope_fp4_sf
                )
                hcdprint(f"[horenc]({layer_id}) GET done go attn - cache_k_nope_fp4_dequant.shape = {cache_k_nope_fp4_dequant.shape}")
                return cache_k_nope_fp4_dequant

        return self.kv_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        # XXXXX: Never come
        hcdprint(f"XXX [horenc]({layer_id}) class MLATokenToKVPool:get_key_buffer(): Jack - GET val XXXXX"
                f" {self.store_dtype} != {self.dtype} XXXXX")
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer][
                ..., : self.kv_lora_rank
            ].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if self.dtype != torch.float4_e2m1fn_x2:
                cache_k = cache_k.to(self.dtype)
            else:
                """
                # debug cache_k
                hcdprint("cache_k type:", type(cache_k))
                hcdprint("cache_k shape:", cache_k.shape)
                hcdprint("cache_k dtype:", cache_k.dtype)
                hcdprint("cache_k device:", cache_k.device)
                num_elements = cache_k.numel()
                hcdprint("Number of elements:", num_elements)
                =>
                    cache_k type: <class 'torch.Tensor'>
                    cache_k shape: torch.Size([7, 1, 576])
                    cache_k type: <class 'torch.Tensor'>
                    cache_k dtype: torch.bfloat16
                    cache_k shape: torch.Size([7, 1, 576])
                    cache_k device: cuda:1
                    cache_k dtype: torch.bfloat16
                    Number of elements: 4032
                    cache_k device: cuda:3
                    Number of elements: 4032
                """
                # cache_k = cache_k.to(self.dtype) # ori
                # hcdprint(f"[horenc] TODO:MLA SET1 fp4 - replace with NVFP4QuantizeUtil.batched_quantize()")

                hcdprint(f"[horenc] bf cache_k.shape = {cache_k.shape}, dtype = {cache_k.dtype}, cache_k.is_contiguous = {cache_k.is_contiguous()}")
                hcdprint(f"[horenc] cache_k.contiguous is needed, otherwise fail!!")
                # cache_k = cache_k.contiguous() # horenc this hack works: PASS
                hcdprint(f"[horenc] af cache_k.shape = {cache_k.shape}, dtype = {cache_k.dtype}, cache_k.is_contiguous = {cache_k.is_contiguous()}")

                from sglang.srt.layers.quantization.nvfp4_tensor import KVFP4QuantizeUtil # need to delayed import
                cache_k_fp4, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(
                        cache_k
                )
                # cache_k_fp4_sf = cache_k_fp4_sf.unsqueeze(1) # horenc25827 for debug [m, k] -> [m, 1, k]

        hcdprint(f"[horenc]({layer_id}) class MLATokenToKVPool:set_kv_buffer(): "
                f"Jack - SET1 - PREFILL (pytorch) loc = {loc}, "
                f"cache_k.dtype {cache_k.dtype} != self.dtype {self.dtype}, "
                f"self.store_dtype {self.store_dtype} != self.dtype {self.dtype} TODO:MLA")

        if self.store_dtype != self.dtype:
            #       self.store_dtype != self.dtype
            # fp16: bfloat16 != bfloat16
            # fp8:  torch.uint8 != torch.float8_e4m3fn
            # fp4:  torch.uint8 != torch.float4_e2m1fn_x2
            # if cache_k.dtype != self.dtype: # save kv4 scales
            if self.dtype == torch.float4_e2m1fn_x2: # save kv4 scales
                hcdprint(f"[horenc] cache_k_fp4.dtype = {cache_k_fp4.dtype}, cache_k_fp4.is_contiguous() = {cache_k_fp4.is_contiguous()}")
                hcdprint(f"[horenc] cache_k_fp4.shape = {cache_k_fp4.shape}")
                # self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view( # ori
                #     self.store_dtype
                # )
                self.kv_buffer[layer_id - self.start_layer][loc] = cache_k_fp4.view(
                    self.store_dtype
                )
                hcdprint(f"[horenc] cache_k_fp4_sf.dtype = {cache_k_fp4_sf.dtype}, cache_k_fp4_sf.is_contiguous() = {cache_k_fp4_sf.is_contiguous()}")
                hcdprint(f"[horenc] cache_k_fp4_sf.shape = {cache_k_fp4_sf.shape}")
                hcdprint(f"self.kv_scale_buffer[layer_id - self.start_layer][loc].shape = {self.kv_scale_buffer[layer_id - self.start_layer][loc].shape}") 
                """
                [horenc] TODO:MLA __init__ create kv kv_scale buffer m = 1206597 n = 1 k = 576 (512 + 64)
                cache_k_fp4.shape = torch.Size([7, 1, 288]) dtype? contiguous?
                cache_k_fp4_sf.shape = torch.Size([7, 4608]) dtype? contiguous?
                (old) self.kv_scale_buffer[layer_id - self.start_layer][loc].shape = torch.Size([7, 1, 36])
                """
                hcdprint(f"[horenc] self.store_dtype = {self.store_dtype}")
                self.kv_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf.view(
                    self.store_dtype
                )
                hcdprint(f"[horenc] MLA SET1 done")

                # Debug e.g., (0) loc = tensor([29
                # Debug e.g., (0) loc = tensor([29
                # Debug e.g., (0) loc = tensor([29
                # After warmup first loc = tensor([16, 17, 18, 19, 20, 21, 22, 23, 24] this is SET2
                # if layer_id == 0 and loc.numel() == 1 and int(loc.item()) == 29:
                # if layer_id == 0 and loc.numel() == 1: # only debug case = [1, n, k]
                if (layer_id == 0 or layer_id == 1) and loc.numel() == 1 and (int(loc.item()) == 1 or int(loc.item()) == 2): #loc < prompt# will not enter here
                    hcdprint(f"{layer_id} Jack DEBUG SET1 - cache_k_fp4_sf[{int(loc.item())}] = {cache_k_fp4_sf}")

            else: # fp8
                self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                    self.store_dtype
                )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id
        # hcdprint(f"[horenc]({layer_id}) class MLATokenToKVPool:set_mla_kv_buffer(): SET2 "
        #         f"-> set_mla_kv_buffer_triton() (USE TRITON writh kv$)")
        # hcdprint(f"[horenc]({layer_id}) loc = {loc}\t "
        #         f"{cache_k_nope.dtype} != {self.dtype}, "
        #         f"{self.store_dtype} != {self.dtype} TODO:MLA")
        if cache_k_nope.dtype != self.dtype:
            if self.dtype == torch.float4_e2m1fn_x2: # save kv4 scales
                # cache_k_nope = cache_k_nope.to(self.dtype) # ori
                # cache_k_rope = cache_k_rope.to(self.dtype) # ori

                # hcdprint(f"[horenc]({layer_id}) SET2 - "
                #         f"cache_k_nope.dtype = {cache_k_nope.dtype}, cache_k_nope.is_contiguous() = {cache_k_nope.is_contiguous()}")
                # hcdprint(f"[horenc]({layer_id}) SET2 - cache_k_nope.shape = {cache_k_nope.shape}")
                from sglang.srt.layers.quantization.nvfp4_tensor import KVFP4QuantizeUtil # need to delayed import
                cache_k_nope_fp4, cache_k_nope_fp4_sf = KVFP4QuantizeUtil.batched_quantize(
                        cache_k_nope
                )
                # cache_k_nope_fp4_sf = cache_k_nope_fp4_sf.unsqueeze(1) # horenc25827 for debug [m, k] -> [m, 1, k]

                # hcdprint(f"[horenc]({layer_id}) SET2 - "
                #         f"cache_k_nope_fp4.dtype = {cache_k_nope_fp4.dtype}, cache_k_nope_fp4.is_contiguous() = {cache_k_nope_fp4.is_contiguous()}")
                # hcdprint(f"[horenc]({layer_id}) SET2 - cache_k_nope_fp4.shape = {cache_k_nope_fp4.shape}")
                # hcdprint(f"[horenc]({layer_id}) SET2 - "
                #         f"cache_k_nope_fp4_sf.dtype = {cache_k_nope_fp4_sf.dtype}, cache_k_nope_fp4_sf.is_contiguous() = {cache_k_nope_fp4_sf.is_contiguous()}")
                # hcdprint(f"[horenc]({layer_id}) SET2 - cache_k_nope_fp4_sf.shape = {cache_k_nope_fp4_sf.shape}")
                """
                input:
                    [horenc](60) SET2 - cache_k_nope.dtype = torch.bfloat16, cache_k_nope.is_contiguous() = True
                    [horenc](60) SET2 - cache_k_nope.shape = torch.Size([1, 1, 512])
                output:
                    [horenc](60) SET2 - cache_k_nope_fp4.dtype = torch.uint8, cache_k_nope_fp4.is_contiguous() = True
                    [horenc](60) SET2 - cache_k_nope_fp4.shape = torch.Size([1, 1, 256])
                    [horenc](60) SET2 - cache_k_nope_fp4_sf.dtype = torch.uint8, cache_k_nope_fp4_sf.is_contiguous() = True
                    [horenc](60) SET2 - cache_k_nope_fp4_sf.shape = torch.Size([1, 32])
                """

                # hcdprint(f"[horenc]({layer_id}) SET2 - "
                #         f"cache_k_rope.dtype = {cache_k_rope.dtype}, cache_k_rope.is_contiguous() = {cache_k_rope.is_contiguous()}")
                # hcdprint(f"[horenc]({layer_id}) SET2 - cache_k_rope.shape = {cache_k_rope.shape}")
                # #                cache_k = cache_k.contiguous() # horenc this hack works: PASS
                # # Same as SET1's cache_k, cache_k_rope may not be contiguous and thus trt will report error
                # cache_k_rope = cache_k_rope.contiguous() # horenc this hack works: PASS
                cache_k_rope_fp4, cache_k_rope_fp4_sf = KVFP4QuantizeUtil.batched_quantize(
                        cache_k_rope
                )
                # cache_k_rope_fp4_sf = cache_k_rope_fp4_sf.unsqueeze(1) # horenc25827 for debug [m, k] -> [m, 1, k]
                # hcdprint(f"[horenc]({layer_id}) SET2 - "
                #         f"cache_k_rope_fp4.dtype = {cache_k_rope_fp4.dtype}, cache_k_rope_fp4.is_contiguous() = {cache_k_rope_fp4.is_contiguous()}")
                # hcdprint(f"[horenc]({layer_id}) SET2 - cache_k_rope_fp4.shape = {cache_k_rope_fp4.shape}")
                # hcdprint(f"[horenc]({layer_id}) SET2 - "
                #         f"cache_k_rope_fp4_sf.dtype = {cache_k_rope_fp4_sf.dtype}, cache_k_rope_fp4_sf.is_contiguous() = {cache_k_rope_fp4_sf.is_contiguous()}")
                # hcdprint(f"[horenc]({layer_id}) SET2 - cache_k_rope_fp4_sf.shape = {cache_k_rope_fp4_sf.shape}")
                """
                    input:
                    [horenc](60) SET2 - cache_k_rope.dtype = torch.bfloat16, cache_k_rope.is_contiguous() = True
                    [horenc](60) SET2 - cache_k_rope.shape = torch.Size([1, 1, 64])
                    output:
                    [horenc](60) SET2 - cache_k_rope_fp4.dtype = torch.uint8, cache_k_rope_fp4.is_contiguous() = True
                    [horenc](60) SET2 - cache_k_rope_fp4.shape = torch.Size([1, 1, 32])
                    [horenc](60) SET2 - cache_k_rope_fp4_sf.dtype = torch.uint8, cache_k_rope_fp4_sf.is_contiguous() = True
                    [horenc](60) SET2 - cache_k_rope_fp4_sf.shape = torch.Size([1, 4])
                """

                # Debug e.g., (0) loc = tensor([29
                # Debug e.g., (0) loc = tensor([29
                # Debug e.g., (0) loc = tensor([29
                # After warmup first loc = tensor([16, 17, 18, 19, 20, 21, 22, 23, 24] this is SET2
                # if layer_id == 0 and loc.numel() == 1 and int(loc.item()) == 29:
                # if layer_id == 0 and loc.numel() == 1 and int(loc.item()) == 29:
                # if layer_id == 0 and loc.numel() == 1: # only debug case = [1, n, k]
                # if (layer_id == 0 or layer_id == 1) and loc.numel() == 1 and (int(loc.item()) == 16 or int(loc.item()) == 17 or int(loc.item()) == 29): #loc < prompt# will not enter here



                # # DEBUG kv cache []
                # if (layer_id == 0 or layer_id == 1) and loc.numel() == 1 and (int(loc.item()) == 29 or int(loc.item()) == 30 or int(loc.item()) == 31): #loc < prompt# will not enter here
                #     hcdprint(f"{layer_id} Jack DEBUG SET2 DECODE - cache_k_nope_fp4[{int(loc.item())}] = {cache_k_nope_fp4}, cache_k_rope_fp4[{int(loc.item())}] = {cache_k_rope_fp4}") # 512/2 + 64/2 = 256 + 32


                # # # PREFILL: loc.numel() >= 1
                # # if (layer_id == 0 or layer_id == 1) and (int(loc.item()) >= 16): #loc < prompt# will not enter here
                # if (layer_id == 0 or layer_id == 1):
                #     if (torch.any(loc == 16) or torch.any(loc == 17) or torch.any(loc == 18)): #loc < prompt# will not enter here
                #         for i, idx in enumerate(loc):
                #             layer_loc = int(idx.item())
                #             hcdprint(f"{layer_id} Jack DEBUG SET2 PREFILL - i={i}, layer_loc[{layer_loc}]=quant[{i}], cache_k_nope_fp4_sf[{layer_loc}] = {cache_k_nope_fp4_sf[i]}, cache_k_rope_fp4_sf[{layer_loc}] = {cache_k_rope_fp4_sf[i]}")
                #         # for idx in loc:
                #         #     i = int(idx.item())
                #         #     hcdprint(f"{layer_id} Jack DEBUG SET2 PREFILL - cache_k_nope_fp4_sf[{i}] = {cache_k_nope_fp4_sf[i]}, cache_k_rope_fp4_sf[{i}] = {cache_k_rope_fp4_sf[i]}")

                # # DECODE: loc.numel() == 1 
                # if (layer_id == 0 or layer_id == 1) and loc.numel() == 1 and (int(loc.item()) == 29 or int(loc.item()) == 30 or int(loc.item()) == 31): #loc < prompt# will not enter here
                #     hcdprint(f"{layer_id} Jack DEBUG SET2 DECODE - cache_k_nope_fp4_sf[{int(loc.item())}] = {cache_k_nope_fp4_sf}, cache_k_rope_fp4_sf[{int(loc.item())}] = {cache_k_rope_fp4_sf}") # 512/16 + 64/16 = 32+4

            else:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k_nope = cache_k_nope.view(self.store_dtype)
            cache_k_rope = cache_k_rope.view(self.store_dtype)

        # fp16: bfloat16 != bfloat16
        # fp8: torch.uint8 != torch.float8_e4m3fn
        if self.dtype != torch.float4_e2m1fn_x2:
            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id], loc, cache_k_nope, cache_k_rope
            )
        else:
            hcdprint(f"[horenc]({layer_id}) SET2 - DECODE (MLA/triton) - REAL WRITE")
            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id], loc, cache_k_nope_fp4, cache_k_rope_fp4
            )
            # set_mla_kv_buffer_triton(
            set_mla_kv_scale_buffer_triton(
                self.kv_scale_buffer[layer_id], loc, cache_k_nope_fp4_sf, cache_k_rope_fp4_sf
            )


    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        torch.cuda.synchronize()


class AscendMLAPagedTokenToKVPool(MLATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super(MLATokenToKVPool, self).__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.kv_buffer = torch.zeros(
                (
                    layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )

        self.layer_transfer_counter = None

        kv_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, KV size: {kv_size / GB:.2f} GB"
        )
        self.mem_usage = kv_size / GB

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [self.kv_buffer[i][0].nbytes for i in range(self.layer_num)]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(store_dtype)

        import torch_npu

        torch_npu._npu_reshape_and_cache_siso(
            key=cache_k.view(-1, 1, self.kv_lora_rank + self.qk_rope_head_dim),
            key_cache=self.kv_buffer[layer_id - self.start_layer].view(
                -1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim
            ),
            slot_indices=loc,
        )


class DoubleSparseTokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        heavy_channel_num: int,
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

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            self.k_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]

            # [size, head_num, heavy_channel_num] for each layer
            self.label_buffer = [
                torch.zeros(
                    (size + 1, head_num, heavy_channel_num), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_label_buffer(self, layer_id: int):
        return self.label_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_label: torch.Tensor,
    ):
        # NOTE(Andy): ignore the dtype check
        layer_id = layer.layer_id
        self.k_buffer[layer_id - self.start_layer][loc] = cache_k
        self.v_buffer[layer_id - self.start_layer][loc] = cache_v
        self.label_buffer[layer_id - self.start_layer][loc] = cache_label


@triton.jit
def copy_all_layer_kv_cache(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128

    bid = tl.program_id(0)
    stride = tl.load(strides + bid)

    data_ptr = tl.load(data_ptrs + bid)
    data_ptr = tl.cast(data_ptr, tl.pointer_type(tl.uint8))

    num_locs_offset = tl.arange(0, num_locs_upper)
    tgt_locs = tl.load(tgt_loc_ptr + num_locs_offset, mask=num_locs_offset < num_locs)
    src_locs = tl.load(src_loc_ptr + num_locs_offset, mask=num_locs_offset < num_locs)

    # NOTE: we cannot parallelize over the tgt_loc_ptr dim with cuda blocks
    # because this copy is an inplace operation.

    num_loop = tl.cdiv(stride, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = (num_locs_offset < num_locs)[:, None] and (copy_offset < stride)[None, :]
        value = tl.load(
            data_ptr + src_locs[:, None] * stride + copy_offset[None, :], mask=mask
        )
        tl.store(
            data_ptr + tgt_locs[:, None] * stride + copy_offset[None, :],
            value,
            mask=mask,
        )
