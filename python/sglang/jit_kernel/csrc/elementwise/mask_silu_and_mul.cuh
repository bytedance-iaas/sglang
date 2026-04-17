#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <type_traits>

namespace {

template <typename T>
struct Vec2Traits;

template <>
struct Vec2Traits<fp16_t> {
  using PackedT = fp16x2_t;
  static __device__ __forceinline__ PackedT load(const fp16_t* ptr) {
    return *reinterpret_cast<const PackedT*>(ptr);
  }
  static __device__ __forceinline__ void store(fp16_t* ptr, PackedT value) {
    *reinterpret_cast<PackedT*>(ptr) = value;
  }
};

template <>
struct Vec2Traits<bf16_t> {
  using PackedT = bf16x2_t;
  static __device__ __forceinline__ PackedT load(const bf16_t* ptr) {
    return *reinterpret_cast<const PackedT*>(ptr);
  }
  static __device__ __forceinline__ void store(bf16_t* ptr, PackedT value) {
    *reinterpret_cast<PackedT*>(ptr) = value;
  }
};

template <bool USE_FP32_ACCUM>
__device__ __forceinline__ float silu_mul(float gate_val, float up_val) {
  if constexpr (USE_FP32_ACCUM) {
    return (gate_val / (1.0f + __expf(-gate_val))) * up_val;
  } else {
    return static_cast<float>(
        gate_val * (1.0f / (1.0f + __expf(-gate_val))) * up_val);
  }
}

template <typename T, int BLOCK_M, int THREADS_N, int VEC_ELEMS, bool USE_FP32_ACCUM>
__global__ void masked_silu_and_mul_kernel_2d(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int32_t* __restrict__ masked_m,
    const int64_t stride_input_0,
    const int64_t stride_input_1,
    const int64_t stride_output_0,
    const int64_t stride_output_1,
    const int size_n) {
  const int expert_id = blockIdx.z;
  const int token_index = blockIdx.y * BLOCK_M + threadIdx.y;
  const int n_offset = (blockIdx.x * THREADS_N + threadIdx.x) * VEC_ELEMS;
  const int token_num_cur_expert = masked_m[expert_id];
  if (token_index >= token_num_cur_expert || n_offset >= size_n) return;

  const T* input_base = input + expert_id * stride_input_0 + token_index * stride_input_1 + n_offset;
  T* output_base = output + expert_id * stride_output_0 + token_index * stride_output_1 + n_offset;
  const int remaining = size_n - n_offset;
  const int n_valid = remaining >= VEC_ELEMS ? VEC_ELEMS : remaining;
  const T* gate_ptr = input_base;
  const T* up_ptr = gate_ptr + size_n;
  T* out_ptr = output_base;

  if constexpr (VEC_ELEMS >= 2 && (VEC_ELEMS % 2 == 0)) {
    if (n_valid == VEC_ELEMS) {
      using PackedT = typename Vec2Traits<T>::PackedT;
      constexpr int kPackedIters = VEC_ELEMS / 2;
      PackedT result_vec[kPackedIters];

#pragma unroll
      for (int i = 0; i < kPackedIters; ++i) {
        const PackedT gate_vec = Vec2Traits<T>::load(gate_ptr + i * 2);
        const PackedT up_vec = Vec2Traits<T>::load(up_ptr + i * 2);
        const fp32x2_t gate_val = device::cast<fp32x2_t>(gate_vec);
        const fp32x2_t up_val = device::cast<fp32x2_t>(up_vec);

        fp32x2_t result_val;
        result_val.x = silu_mul<USE_FP32_ACCUM>(gate_val.x, up_val.x);
        result_val.y = silu_mul<USE_FP32_ACCUM>(gate_val.y, up_val.y);
        result_vec[i] = device::cast<PackedT>(result_val);
      }

#pragma unroll
      for (int i = 0; i < kPackedIters; ++i) {
        Vec2Traits<T>::store(out_ptr + i * 2, result_vec[i]);
      }
      return;
    }
  }

  for (int i = 0; i < n_valid; ++i) {
    out_ptr[i] = static_cast<T>(
        silu_mul<USE_FP32_ACCUM>(static_cast<float>(gate_ptr[i]), static_cast<float>(up_ptr[i])));
  }
}

}  // namespace

// Host-side launcher, exposed as a JIT kernel entry point.
// Template parameter T is the scalar type (fp16_t or bf16_t).
template <typename T, int BLOCK_M = 8, int THREADS_N = 16, int VEC_ELEMS = 8>
void masked_silu_and_mul(
    tvm::ffi::TensorView input_arr,
    tvm::ffi::TensorView output_arr,
    tvm::ffi::TensorView masked_m_arr,
    int64_t expert_num,
    int64_t logical_m,
    int64_t size_n,
    bool use_fp32_accum) {
  const T* input = static_cast<const T*>(input_arr.data_ptr());
  T* output = static_cast<T*>(output_arr.data_ptr());
  const int32_t* masked_m = static_cast<const int32_t*>(masked_m_arr.data_ptr());

  // Strides in elements (assumes stride_input_2 == stride_output_2 == 1, i.e. contiguous innermost)
  const int64_t stride_input_0 = input_arr.strides()[0];
  const int64_t stride_input_1 = input_arr.strides()[1];
  const int64_t stride_output_0 = output_arr.strides()[0];
  const int64_t stride_output_1 = output_arr.strides()[1];

  static_assert(BLOCK_M > 0, "BLOCK_M must be positive");
  static_assert(THREADS_N > 0, "THREADS_N must be positive");
  static_assert(VEC_ELEMS > 0, "VEC_ELEMS must be positive");
  static_assert(BLOCK_M * THREADS_N <= 1024, "CUDA block size exceeds 1024 threads");
  constexpr bool kUseVecPath =
      (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) && (VEC_ELEMS % 2 == 0);

  auto launch = [&](auto use_fp32_tag) {
    constexpr bool kUseFP32 = decltype(use_fp32_tag)::value;
    constexpr int kKernelVecElems = kUseVecPath ? VEC_ELEMS : 1;
    constexpr int kKernelTileN = THREADS_N * kKernelVecElems;
    const int grid_n = (static_cast<int>(size_n) + kKernelTileN - 1) / kKernelTileN;
    const int grid_m = (static_cast<int>(logical_m) + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(grid_n, grid_m, static_cast<int>(expert_num));
    dim3 block(THREADS_N, BLOCK_M);
    host::LaunchKernel(grid, block, input_arr.device())(
        masked_silu_and_mul_kernel_2d<T, BLOCK_M, THREADS_N, kKernelVecElems, kUseFP32>,
        input,
        output,
        masked_m,
        stride_input_0,
        stride_input_1,
        stride_output_0,
        stride_output_1,
        static_cast<int>(size_n));
  };

  if (use_fp32_accum) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}
