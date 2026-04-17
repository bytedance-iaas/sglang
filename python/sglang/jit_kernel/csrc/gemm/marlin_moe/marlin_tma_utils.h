#pragma once

#include <cstdint>
#include <dlpack/dlpack.h>
#include <utility>

#if !defined(SGLANG_MARLIN_DIRECT_ENABLE_B_TMA) && defined(__CUDACC__) && (CUDA_VERSION >= 12010) && \
    __has_include(<cute/arch/copy_sm90_tma.hpp>) && __has_include(<cutlass/arch/barrier.h>)
#define SGLANG_MARLIN_DIRECT_ENABLE_B_TMA 1
#endif

#if defined(SGLANG_MARLIN_DIRECT_ENABLE_B_TMA) && defined(__CUDACC__) && (CUDA_VERSION >= 12010)

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <dlfcn.h>

#if __has_include(<cute/arch/copy_sm90_tma.hpp>) && __has_include(<cutlass/arch/barrier.h>)
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>

namespace device {
namespace marlin_moe {
namespace tma {

using Barrier = cutlass::arch::ClusterTransactionBarrier;
using LocalTensorMapDesc = cute::TmaDescriptor;

static_assert(sizeof(LocalTensorMapDesc) == 128, "Unexpected TMA descriptor size");

template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, typename dtype_t>
constexpr uint32_t get_inner_block_atom_size() {
  return kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t);
}

template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t kSwizzleMode, typename dtype_t>
__device__ __forceinline__ void tma_copy_2d(
    void const* desc_ptr,
    Barrier* barrier_ptr,
    dtype_t* smem_ptr,
    const uint32_t& inner_idx,
    const uint32_t& outer_idx) {
  constexpr uint32_t BLOCK_INNER_ATOM = get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, dtype_t>();
  #pragma unroll
  for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++i) {
    cute::SM90_TMA_LOAD_2D::copy(
        desc_ptr,
        reinterpret_cast<uint64_t*>(barrier_ptr),
        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
        smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
        inner_idx + i * BLOCK_INNER_ATOM,
        outer_idx);
  }
}

__device__ __forceinline__ void prefetch_tma_descriptor(const LocalTensorMapDesc* desc_ptr) {
  cute::prefetch_tma_descriptor(desc_ptr);
}

inline void* get_driver_handle() {
  static void* handle = nullptr;
  if (handle == nullptr) {
    handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  }
  return handle;
}

template <typename... Args>
inline CUresult lazy_cuTensorMapEncodeTiled(Args&&... args) {
  using FuncType = decltype(&cuTensorMapEncodeTiled);
  static FuncType func = nullptr;
  if (func == nullptr) {
    auto* handle = get_driver_handle();
    if (handle == nullptr) {
      return CUDA_ERROR_NOT_INITIALIZED;
    }
    func = reinterpret_cast<FuncType>(dlsym(handle, "cuTensorMapEncodeTiled"));
    if (func == nullptr) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
  }
  return func(std::forward<Args>(args)...);
}

inline CUtensorMapDataType dl_dtype_to_tensor_map_dtype(DLDataType dtype) {
  if (dtype.code == DLDataTypeCode::kDLInt && dtype.bits == 32) {
    return CU_TENSOR_MAP_DATA_TYPE_INT32;
  }
  if (dtype.code == DLDataTypeCode::kDLFloat && dtype.bits == 32) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  }
  if (dtype.code == DLDataTypeCode::kDLBfloat && dtype.bits == 16) {
    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  }
  if (dtype.code == DLDataTypeCode::kDLUInt && dtype.bits == 8) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  }
  if (dtype.code == DLDataTypeCode::kDLUInt && dtype.bits == 4) {
    return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B;
  }
  return CU_TENSOR_MAP_DATA_TYPE_UINT8;
}

inline CUtensorMapSwizzle swizzle_mode_to_tensor_map_swizzle(int swizzle_mode) {
  switch (swizzle_mode) {
    case 0:
    case 16:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    default:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
  }
}

inline bool make_tensor_map_2d_desc(
    LocalTensorMapDesc* tensor_map_desc,
    const void* data_ptr,
    DLDataType dtype,
    int gmem_inner_dim,
    int gmem_outer_dim,
    int smem_inner_dim,
    int smem_outer_dim,
    int gmem_outer_stride,
    int swizzle_mode) {
  if (tensor_map_desc == nullptr) {
    return false;
  }
  auto* tensor_map = reinterpret_cast<CUtensorMap*>(tensor_map_desc);

  const int elem_size = static_cast<int>(dtype.bits / 8);
  if (elem_size <= 0) {
    return false;
  }

  if (swizzle_mode != 0) {
    smem_inner_dim = swizzle_mode / elem_size;
  }

  const cuuint64_t gmem_dims[2] = {
      static_cast<cuuint64_t>(gmem_inner_dim),
      static_cast<cuuint64_t>(gmem_outer_dim),
  };
  const cuuint32_t smem_dims[2] = {
      static_cast<cuuint32_t>(smem_inner_dim),
      static_cast<cuuint32_t>(smem_outer_dim),
  };
  const cuuint64_t gmem_strides[1] = {
      static_cast<cuuint64_t>(gmem_outer_stride * elem_size),
  };
  const cuuint32_t elem_strides[2] = {1, 1};

  auto result = lazy_cuTensorMapEncodeTiled(
      tensor_map,
      dl_dtype_to_tensor_map_dtype(dtype),
      2,
      const_cast<void*>(data_ptr),
      gmem_dims,
      gmem_strides,
      smem_dims,
      elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle_mode_to_tensor_map_swizzle(swizzle_mode),
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return result == CUDA_SUCCESS;
}

}  // namespace tma
}  // namespace marlin_moe
}  // namespace device

#define SGLANG_MARLIN_MOE_HAS_LOCAL_TMA 1

#else

namespace device {
namespace marlin_moe {
namespace tma {

struct alignas(128) LocalTensorMapDesc {
  std::uint8_t data[128] = {};
};

inline bool make_tensor_map_2d_desc(
    LocalTensorMapDesc*,
    const void*,
    DLDataType,
    int,
    int,
    int,
    int,
    int,
    int) {
  return false;
}

}  // namespace tma
}  // namespace marlin_moe
}  // namespace device

#define SGLANG_MARLIN_MOE_HAS_LOCAL_TMA 0

#endif

#else

namespace device {
namespace marlin_moe {
namespace tma {

struct alignas(128) LocalTensorMapDesc {
  std::uint8_t data[128] = {};
};

inline bool make_tensor_map_2d_desc(
    LocalTensorMapDesc*,
    const void*,
    DLDataType,
    int,
    int,
    int,
    int,
    int,
    int) {
  return false;
}

}  // namespace tma
}  // namespace marlin_moe
}  // namespace device

#define SGLANG_MARLIN_MOE_HAS_LOCAL_TMA 0

#endif
