# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import time

# import numpy as np
import torch
# import tensorrt_llm
# import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
# from flashinfer import nvfp4_quantize
from flashinfer import nvfp4_batched_quantize


# E2M1_MAX = 6.0
# E2M1_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
# E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32)

# class KVFP4QuantizeUtil:
#     """Utility class for NVFP4 quantization and dequantization operations."""

#     @staticmethod
#     @torch.compile
#     def batched_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Quantize tensor to KVFP4 format
#         Args:
#             tensor: Input tensor of shape [B, M, N]

#         Returns:
#             quant_tensor: Quantized tensor of shape [B, M, N/2]
#             scale_factors: Scale factors of shape [B, M*N/16]
#         """
#         b, m, n = tensor.shape
#         device = tensor.device

#         # Reshape to [B, M*N/16, 16] for block-wise quantization
#         reshaped = tensor.view(b, m * n // 16, 16)

#         # Compute scale factors per block
#         block_max = reshaped.abs().max(dim=-1, keepdim=True).values
#         scale_exp = torch.ceil(
#             torch.log2(torch.clamp(block_max / E2M1_MAX, min=1e-10))
#         )
#         scale_factors = (scale_exp + 127).squeeze(-1).to(torch.uint8)

#         # Apply scaling
#         scaled = reshaped / torch.exp2(scale_exp)

#         # Quantize to FP4
#         sign_bits = (scaled < 0).to(torch.uint8) << 3
#         abs_vals = scaled.abs()

#         # Vectorized quantization using searchsorted
#         bounds = E2M1_BOUNDS.to(device)
#         magnitude_bits = torch.searchsorted(bounds, abs_vals, right=False)

#         # Combine sign and magnitude
#         fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)

#         # Pack two FP4 values into one uint8
#         fp4_reshaped = fp4_vals.view(b, m, n)
#         packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]

#         return packed, scale_factors

#     @staticmethod
#     @torch.compile
#     def batched_dequantize(
#         quant_tensor: torch.Tensor,
#         scale_factors: torch.Tensor,
#         dtype: torch.dtype = torch.bfloat16,
#     ) -> torch.Tensor:
#         """
#         Dequantize KVFP4 tensor
#         Args:
#             quant_tensor: Quantized tensor of shape [B, M, N/2]
#             scale_factors: Scale factors of shape [B, M*N/16]
#             dtype: Target dtype for output

#         Returns:
#             Dequantized tensor of shape [B, M, N]
#         """
#         b, m, n_half = quant_tensor.shape
#         n = n_half * 2
#         device = quant_tensor.device

#         # More efficient unpacking using bit operations
#         fp4_vals = torch.empty(b, m, n, dtype=torch.uint8, device=device)
#         fp4_vals[..., 0::2] = quant_tensor & 0x0F
#         fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F

#         # Extract sign and magnitude
#         sign_mask = (fp4_vals & 0x08) != 0
#         magnitude_idx = fp4_vals & 0x07

#         # Convert to float values
#         values = E2M1_VALUES.to(device)
#         float_vals = values[magnitude_idx.long()]
#         float_vals = torch.where(sign_mask, -float_vals, float_vals)

#         # Reshape for block-wise scaling
#         reshaped = float_vals.view(b, m * n // 16, 16)

#         # Apply scale factors
#         scale_exp = scale_factors.float() - 127
#         scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))

#         return scaled.view(b, m, n).to(dtype)


# E2M1_MAX = 6.0
# E2M1_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device="cuda")
# E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32, device="cuda")

# class KVFP4QuantizeUtil:
#     """Utility class for NVFP4 quantization and dequantization operations."""

#     @staticmethod
#     def batched_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Quantize tensor to KVFP4 format
#         Args:
#             tensor: Input tensor of shape [B, M, N]

#         Returns:
#             quant_tensor: Quantized tensor of shape [B, M, N/2]
#             scale_factors: Scale factors of shape [B, M*N/16]
#         """
#         b, m, n = tensor.shape
#         device = tensor.device

#         # Reshape to [B, M*N/16, 16] for block-wise quantization
#         reshaped = tensor.view(b, m * n // 16, 16)

#         # Compute scale factors per block
#         block_max = reshaped.abs().max(dim=-1, keepdim=True).values
#         scale_exp = torch.ceil(
#             torch.log2(torch.clamp(block_max / E2M1_MAX, min=1e-10))
#         )
#         scale_factors = (scale_exp + 127).squeeze(-1).to(torch.uint8)

#         # Apply scaling
#         scaled = reshaped / torch.exp2(scale_exp)

#         # Quantize to FP4
#         sign_bits = (scaled < 0).to(torch.uint8) << 3
#         abs_vals = scaled.abs()

#         # Vectorized quantization using searchsorted
#         bounds = E2M1_BOUNDS.to(device)
#         magnitude_bits = torch.searchsorted(bounds, abs_vals, right=False)

#         # Combine sign and magnitude
#         fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)

#         # Pack two FP4 values into one uint8
#         fp4_reshaped = fp4_vals.view(b, m, n)
#         packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]

#         return packed, scale_factors

#     @staticmethod
#     def batched_dequantize(
#         quant_tensor: torch.Tensor,
#         scale_factors: torch.Tensor,
#         dtype: torch.dtype = torch.bfloat16,
#     ) -> torch.Tensor:
#         """
#         Dequantize KVFP4 tensor
#         Args:
#             quant_tensor: Quantized tensor of shape [B, M, N/2]
#             scale_factors: Scale factors of shape [B, M*N/16]
#             dtype: Target dtype for output

#         Returns:
#             Dequantized tensor of shape [B, M, N]
#         """
#         b, m, n_half = quant_tensor.shape
#         n = n_half * 2
#         device = quant_tensor.device

#         # More efficient unpacking using bit operations
#         fp4_vals = torch.empty(b, m, n, dtype=torch.uint8, device=device)
#         fp4_vals[..., 0::2] = quant_tensor & 0x0F
#         fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F

#         # Extract sign and magnitude
#         sign_mask = (fp4_vals & 0x08) != 0
#         magnitude_idx = fp4_vals & 0x07

#         # Convert to float values
#         values = E2M1_VALUES.to(device)
#         float_vals = values[magnitude_idx.long()]
#         float_vals = torch.where(sign_mask, -float_vals, float_vals)

#         # Reshape for block-wise scaling
#         reshaped = float_vals.view(b, m * n // 16, 16)

#         # Apply scale factors
#         scale_exp = scale_factors.float() - 127
#         scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))

#         return scaled.view(b, m, n).to(dtype)


E2M1_MAX = 6.0
# Put constants directly on CUDA if available
_device = "cuda" if torch.cuda.is_available() else "cpu"
E2M1_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=_device)
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32, device=_device)

class KVFP4QuantizeUtil:
    """Utility class for NVFP4 quantization and dequantization operations."""

    @staticmethod
    @torch.compile
    def batched_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to KVFP4 format
        Args:
            tensor: Input tensor of shape [B, M, N]

        Returns:
            quant_tensor: Quantized tensor of shape [B, M, N/2]
            scale_factors: Scale factors of shape [B, M*N/16]
        """
        b, m, n = tensor.shape

        # Reshape to [B, M*N/16, 16] for block-wise quantization
        reshaped = tensor.view(b, m * n // 16, 16)

        # Compute scale factors per block
        block_max = reshaped.abs().max(dim=-1, keepdim=True).values
        scale_exp = torch.ceil(
            torch.log2(torch.clamp(block_max / E2M1_MAX, min=1e-10))
        )
        scale_factors = (scale_exp + 127).squeeze(-1).to(torch.uint8)

        # Apply scaling
        scaled = reshaped / torch.exp2(scale_exp)

        # Quantize to FP4
        sign_bits = (scaled < 0).to(torch.uint8) << 3
        abs_vals = scaled.abs()

        # Pure tensor version (CUDA Graph safe)
        magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= E2M1_BOUNDS, dim=-1)

        # Combine sign and magnitude
        fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)

        # Pack two FP4 values into one uint8
        fp4_reshaped = fp4_vals.view(b, m, n)
        packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]

        return packed, scale_factors

    @staticmethod
    @torch.compile
    def batched_dequantize(
        quant_tensor: torch.Tensor,
        scale_factors: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize KVFP4 tensor
        Args:
            quant_tensor: Quantized tensor of shape [B, M, N/2]
            scale_factors: Scale factors of shape [B, M*N/16]
            dtype: Target dtype for output

        Returns:
            Dequantized tensor of shape [B, M, N]
        """
        b, m, n_half = quant_tensor.shape
        n = n_half * 2

        # More efficient unpacking using bit operations
        fp4_vals = torch.empty(b, m, n, dtype=torch.uint8, device=quant_tensor.device)
        fp4_vals[..., 0::2] = quant_tensor & 0x0F
        fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F

        # Extract sign and magnitude
        sign_mask = (fp4_vals & 0x08) != 0
        magnitude_idx = fp4_vals & 0x07

        # Convert to float values
        float_vals = E2M1_VALUES[magnitude_idx.long()]
        float_vals = torch.where(sign_mask, -float_vals, float_vals)

        # Reshape for block-wise scaling
        reshaped = float_vals.view(b, m * n // 16, 16)

        # Apply scale factors
        scale_exp = scale_factors.float() - 127
        scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))

        return scaled.view(b, m, n).to(dtype)




# @torch.compile
# def convert_swizzled_to_linear(
#     a_sf_swizzled: torch.Tensor, m: int, k: int, block_size: int
# ) -> torch.Tensor:
#     """Convert swizzled scale factor layout to linear layout."""
#     m_tiles = (m + 127) // 128
#     f = block_size * 4
#     k_tiles = (k + f - 1) // f
#     tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
#     tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
#     out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
#     return out[:m, : k // block_size]


# @torch.compile
# def break_fp4_bytes(a: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
#     """Vectorized FP4 byte unpacking to float values."""
#     assert a.dtype == torch.uint8
#     original_shape = a.shape

#     # Vectorized FP4 unpacking
#     a_flat = a.flatten()
#     high = (a_flat & 0xF0) >> 4
#     low = a_flat & 0x0F
#     combined = torch.stack((low, high), dim=1).flatten()

#     signs = (combined & 0x08).to(torch.bool)
#     abs_vals = (combined & 0x07).to(torch.long)

#     # FP4 E2M1 lookup table
#     e2m1_to_float = torch.tensor(
#         [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
#         dtype=torch.float32,
#         device=a.device,
#     )
#     values = e2m1_to_float[abs_vals] * torch.where(signs, -1.0, 1.0)

#     return values.view(*original_shape[:-1], original_shape[-1] * 2).to(dtype=dtype)

# class NVFP4QuantizeUtil:
#     """Utility class for NVFP4 quantization and dequantization operations."""

    # # @classmethod
    # # def batched_quantize(cls, tensor_bf16, global_sf, sf_vec_size):
    # #     tensor_fp4, scale_factors = torch.ops.trtllm.fp4_batched_quantize(
    # #         tensor_bf16, global_sf, sf_vec_size, False
    # #     )
    # #     return tensor_fp4, scale_factors
    # @classmethod
    # def batched_quantize(
    #     cls, tensor_bf16: torch.Tensor, global_sf: float, sf_vec_size: int
    # ) -> tuple[torch.Tensor, torch.Tensor]:

    #     # return torch.ops.trtllm.fp4_batched_quantize(
    #     return nvfp4_batched_quantize(
    #         tensor_bf16, global_sf, sf_vec_size
    #     )


    # @classmethod
    # @torch.compile
    # def batched_dequantize(
    #     cls,
    #     tensor_fp4: torch.Tensor,
    #     tensor_sf: torch.Tensor,
    #     global_scale: float,
    #     dtype: torch.dtype,
    #     block_size: int = 16,
    # ) -> torch.Tensor:

    #     # Optimize for tensors with middle dimensions <= 1024
    #     original_shape = tensor_fp4.shape
    #     # Use batched version for general case
    #     *batch_dims, m, packed_k = original_shape
    #     k = packed_k * 2

    #     tensor_f32 = break_fp4_bytes(tensor_fp4, torch.float32)
    #     sf_f32 = tensor_sf.view(torch.float8_e4m3fn).to(torch.float32) / global_scale

    #     batch_size = tensor_f32.view(-1, m, k).size(0)
    #     tensor_f32_flat = tensor_f32.view(batch_size, m, k)
    #     sf_f32_flat = sf_f32.view(batch_size, -1)

    #     def process_single_batch(fp4_batch: torch.Tensor, sf_batch: torch.Tensor) -> torch.Tensor:
    #         sf_linear = convert_swizzled_to_linear(sf_batch.unsqueeze(0), m, k, block_size)
    #         sf_expanded = sf_linear.unsqueeze(-1).expand(-1, -1, block_size)
    #         sf_broadcast = sf_expanded.contiguous().view(m, k)
    #         return fp4_batch * sf_broadcast

    #     result = torch.vmap(process_single_batch)(tensor_f32_flat, sf_f32_flat)
    #     return result.view(*batch_dims, m, k).to(dtype=dtype)


# @torch.compile
# def convert_swizzled_to_linear(
#     a_sf_swizzled: torch.Tensor, m: int, k: int, block_size: int
# ) -> torch.Tensor:
#     """Convert swizzled scale factor layout to linear layout."""
#     m_tiles = (m + 127) // 128
#     f = block_size * 4
#     k_tiles = (k + f - 1) // f
#     tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
#     tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
#     out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
#     return out[:m, : k // block_size]


# @torch.compile
# def break_fp4_bytes(a: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
#     """Vectorized FP4 byte unpacking to float values."""
#     assert a.dtype == torch.uint8
#     original_shape = a.shape

#     # Vectorized FP4 unpacking
#     a_flat = a.flatten()
#     high = (a_flat & 0xF0) >> 4
#     low = a_flat & 0x0F
#     combined = torch.stack((low, high), dim=1).flatten()

#     signs = (combined & 0x08).to(torch.bool)
#     abs_vals = (combined & 0x07).to(torch.long)

#     # FP4 E2M1 lookup table
#     e2m1_to_float = torch.tensor(
#         [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
#         dtype=torch.float32,
#         device=a.device,
#     )
#     values = e2m1_to_float[abs_vals] * torch.where(signs, -1.0, 1.0)

#     return values.view(*original_shape[:-1], original_shape[-1] * 2).to(dtype=dtype)


# class NVFP4QuantizeUtil:
#     """Utility class for NVFP4 quantization and dequantization operations."""

#     @classmethod
#     def batched_quantize(
#         cls, tensor_bf16: torch.Tensor, global_sf: float, sf_vec_size: int
#     ) -> tuple[torch.Tensor, torch.Tensor]:

#         # Optimize for tensors with middle dimensions <= 1024 by using 2D kernel
#         if len(tensor_bf16.shape) == 3 and tensor_bf16.shape[1] <= 1024:
#             original_shape = tensor_bf16.shape
#             # Flatten middle dimension into first dimension
#             tensor_2d = tensor_bf16.view(original_shape[0] * original_shape[1], original_shape[2])
#             # tensor_fp4, scale_factors = torch.ops.trtllm.fp4_quantize(
#             # from flashinfer import nvfp4_quantize
#             tensor_fp4, scale_factors = nvfp4_quantize(
#                 tensor_2d, global_sf, sf_vec_size, False
#             )
#             # Only reshape tensor_fp4, keep scale_factors as 2D for dequantize compatibility
#             tensor_fp4 = tensor_fp4.view(original_shape[0], original_shape[1], tensor_fp4.shape[-1])
#             return tensor_fp4, scale_factors
#         else:
#             # return torch.ops.trtllm.fp4_batched_quantize(
#             # from flashinfer import nvfp4_batched_quantize
#             return nvfp4_batched_quantize(
#                 tensor_bf16, global_sf, sf_vec_size, False
#             )

#     @classmethod
#     @torch.compile
#     def batched_dequantize(
#         cls,
#         tensor_fp4: torch.Tensor,
#         tensor_sf: torch.Tensor,
#         global_scale: float,
#         dtype: torch.dtype,
#         block_size: int = 16,
#     ) -> torch.Tensor:

#         # Optimize for tensors with middle dimensions <= 1024
#         original_shape = tensor_fp4.shape
#         if len(original_shape) == 3 and original_shape[1] <= 1024:
#             # Flatten middle dimension into first dimension
#             tensor_fp4_2d = tensor_fp4.view(original_shape[0] * original_shape[1], original_shape[2])
#             # scale_factors is already 2D from quantize, use directly
#             tensor_sf_2d = tensor_sf

#             m, packed_k = tensor_fp4_2d.shape
#             k = packed_k * 2

#             tensor_f32 = break_fp4_bytes(tensor_fp4_2d, torch.float32)
#             sf_f32 = tensor_sf_2d.view(torch.float8_e4m3fn).to(torch.float32) / global_scale

#             sf_linear = convert_swizzled_to_linear(sf_f32.unsqueeze(0), m, k, block_size)
#             sf_expanded = sf_linear.unsqueeze(-1).expand(-1, -1, block_size)
#             sf_broadcast = sf_expanded.contiguous().view(m, k)

#             result_2d = tensor_f32 * sf_broadcast
#             # Reshape back to original structure
#             return result_2d.view(original_shape[0], original_shape[1], k).to(dtype=dtype)
#         else:
#             # Use batched version for general case
#             *batch_dims, m, packed_k = original_shape
#             k = packed_k * 2

#             tensor_f32 = break_fp4_bytes(tensor_fp4, torch.float32)
#             sf_f32 = tensor_sf.view(torch.float8_e4m3fn).to(torch.float32) / global_scale

#             batch_size = tensor_f32.view(-1, m, k).size(0)
#             tensor_f32_flat = tensor_f32.view(batch_size, m, k)
#             sf_f32_flat = sf_f32.view(batch_size, -1)

#             def process_single_batch(fp4_batch: torch.Tensor, sf_batch: torch.Tensor) -> torch.Tensor:
#                 sf_linear = convert_swizzled_to_linear(sf_batch.unsqueeze(0), m, k, block_size)
#                 sf_expanded = sf_linear.unsqueeze(-1).expand(-1, -1, block_size)
#                 sf_broadcast = sf_expanded.contiguous().view(m, k)
#                 return fp4_batch * sf_broadcast

#             result = torch.vmap(process_single_batch)(tensor_f32_flat, sf_f32_flat)
#             return result.view(*batch_dims, m, k).to(dtype=dtype)

