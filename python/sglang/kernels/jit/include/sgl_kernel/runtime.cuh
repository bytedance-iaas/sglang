/// \file runtime.cuh
/// \brief Host-side CUDA runtime query helpers.
///
/// Thin wrappers around CUDA occupancy and device-property APIs with
/// automatic error checking via `CHECK_CUDA`.

#pragma once

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <utility>
#ifndef USE_ROCM
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#ifndef cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#endif
#ifndef cudaDeviceGetAttribute
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#endif
#ifndef cudaDevAttrMultiProcessorCount
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#endif
#ifndef cudaDevAttrComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#endif
#ifndef cudaDevAttrComputeCapabilityMinor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#endif
#ifndef cudaDevAttrMaxSharedMemoryPerBlock
#define cudaDevAttrMaxSharedMemoryPerBlock hipDeviceAttributeMaxSharedMemoryPerBlock
#endif
#ifndef cudaDevAttrMaxSharedMemoryPerBlockOptin
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeSharedMemPerBlockOptin
#endif
#ifndef cudaRuntimeGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#endif
#ifndef cudaOccupancyAvailableDynamicSMemPerBlock
inline hipError_t
cudaOccupancyAvailableDynamicSMemPerBlock(std::size_t* smem, const void* func, int num_blocks, int block_size) {
  // HIP does not expose this directly; return max shared mem as conservative estimate
  hipDeviceProp_t prop;
  int device;
  hipGetDevice(&device);
  hipGetDeviceProperties(&prop, device);
  *smem = prop.sharedMemPerBlock;
  return hipSuccess;
}
#endif
#endif

namespace sglang {

namespace host::runtime {

inline void* get_device_accessible_ptr(const tvm::ffi::TensorView& tensor) {
  void* ptr = tensor.data_ptr();
  const auto tensor_device_type = tensor.device().device_type;
  if (tensor_device_type != kDLCPU && tensor_device_type != kDLGPUHost) {
    return ptr;
  }

  void* device_ptr = nullptr;
#ifdef USE_ROCM
  RuntimeDeviceCheck(::hipHostGetDevicePointer(&device_ptr, ptr, 0));
#else
  RuntimeDeviceCheck(::cudaHostGetDevicePointer(&device_ptr, ptr, 0));
#endif
  return device_ptr;
}

/**
 * \brief Memo for a per-device driver query, keyed by device ordinal.
 *
 * \tparam T        Cached value type.
 * \tparam kDefault The "not queried yet" sentinel. A query that can legally
 *         return it would be re-run on every call, so pick one it cannot.
 */
template <typename T, T kDefault>
struct DeviceCacheMap {
 public:
  // Generous bound on the device ordinals one process can see; a larger ordinal
  // is not an error, it just falls through to the driver query uncached.
  static constexpr uint32_t kNumStaticMaxDevice = 72;

  constexpr DeviceCacheMap() {
    for (uint32_t i = 0; i < kNumStaticMaxDevice; ++i) {
      m_data[i] = kDefault;
    }
  }

  template <typename Fn>
  T get_cached(int32_t device_, bool use_cache, Fn&& fn) {
    const auto device = static_cast<uint32_t>(device_);
    if (use_cache && device < kNumStaticMaxDevice && m_data[device] != kDefault) {
      return m_data[device];
    }
    const auto value = static_cast<T>(std::forward<Fn>(fn)(device_));
    if (device < kNumStaticMaxDevice) {
      m_data[device] = value;
    }
    return value;
  }

 private:
  T m_data[kNumStaticMaxDevice];
};

// Return the maximum number of active blocks per SM for the given kernel
template <typename T>
inline auto get_blocks_per_sm(T&& kernel, int32_t block_dim, std::size_t dynamic_smem = 0) -> uint32_t {
  int num_blocks_per_sm = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, block_dim, dynamic_smem));
  return static_cast<uint32_t>(num_blocks_per_sm);
}

// Return the number of SMs for the given device
inline auto get_sm_count(int device_id, bool use_cache = true) -> uint32_t {
  static DeviceCacheMap<uint32_t, 0> sm_count_cache;
  return sm_count_cache.get_cached(device_id, use_cache, [](int32_t device_id) {
    int sm_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
    return sm_count;
  });
}

// Return the Major compute capability for the given device
inline auto get_cc_major(int device_id, bool use_cache = true) -> int {
  static DeviceCacheMap<int, -1> cc_major_cache;
  return cc_major_cache.get_cached(device_id, use_cache, [](int32_t device_id) {
    int cc_major;
    CHECK_CUDA(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device_id));
    return cc_major;
  });
}

// Return the Minor compute capability for the given device
inline auto get_cc_minor(int device_id, bool use_cache = true) -> int {
  static DeviceCacheMap<int, -1> cc_minor_cache;
  return cc_minor_cache.get_cached(device_id, use_cache, [](int32_t device_id) {
    int cc_minor;
    CHECK_CUDA(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device_id));
    return cc_minor;
  });
}

// Return the SM version (major * 10 + minor) for the given device
inline auto get_sm_version(int device_id, bool use_cache = true) -> int {
  return get_cc_major(device_id, use_cache) * 10 + get_cc_minor(device_id, use_cache);
}

// Return the runtime version
inline auto get_runtime_version() -> int {
  int runtime_version;
  CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));
  return runtime_version;
}

// Return the maximum dynamic shared memory per block for the given kernel
template <typename T>
inline auto get_available_dynamic_smem_per_block(T&& kernel, int num_blocks, int block_size) -> std::size_t {
  std::size_t smem_size;
  CHECK_CUDA(cudaOccupancyAvailableDynamicSMemPerBlock(&smem_size, kernel, num_blocks, block_size));
  return smem_size;
}

/**
 * \brief Shared memory one block may use on the given device, in bytes.
 *
 * \param device_id CUDA device ordinal.
 * \param opt_in    True for the larger limit a kernel reaches by setting
 *        `cudaFuncAttributeMaxDynamicSharedMemorySize`, false for the default
 *        cap that needs no opt-in.
 * \param use_cache False re-queries the driver instead of reading the memo.
 */
inline auto get_max_smem_per_block(int device_id, bool opt_in = true, bool use_cache = true) -> std::size_t {
  /// NOTE: we all know this is no larger than 1MB for all GPU, so use uint32_t
  static DeviceCacheMap<uint32_t, 0> smem_cache[2];
  const auto attr = opt_in ? cudaDevAttrMaxSharedMemoryPerBlockOptin : cudaDevAttrMaxSharedMemoryPerBlock;
  return smem_cache[opt_in].get_cached(device_id, use_cache, [attr](int32_t device_id) {
    int value = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&value, attr, device_id));
    return value;
  });
}

namespace details {

struct MaybeDevice {
  MaybeDevice(DLDevice device) : device_id(device.device_id) {}
  MaybeDevice(int32_t device_id_) : device_id(device_id_) {}
  uint32_t device_id;
};

}  // namespace details

inline constexpr uint32_t kNumStaticMaxDevice = DeviceCacheMap<uint32_t, 0>::kNumStaticMaxDevice;

/// \brief Run `fn` on the first call per device, keyed off the uniqueness of (`Tag`, `Fn`).
/// \note Relaxed on purpose, to keep the hot path a single atomic load: concurrent first callers
/// may all run `fn`, so it must be idempotent (e.g. setting a function attribute). A throwing `fn`
/// leaves the device uninitialized, and devices past `kNumStaticMaxDevice` rerun `fn` every call.
template <typename Tag = void, std::invocable Fn>
inline void init_per_device(int device_id, bool use_cache, Fn&& fn) {
  static_assert(kNumStaticMaxDevice % 8 == 0);
  constexpr uint32_t kNumBytes = kNumStaticMaxDevice / 8;
  static std::atomic_uint8_t s_initialized[kNumBytes]{};
  const auto device = static_cast<uint32_t>(device_id);
  const auto can_use_cache = use_cache && device < kNumStaticMaxDevice;
  const auto mask = static_cast<uint8_t>(1u << (device % 8));
  if (can_use_cache && s_initialized[device / 8].load() & mask) return;
  std::forward<Fn>(fn)();
  if (can_use_cache) s_initialized[device / 8].fetch_or(mask);
}

// Some wrappers of the above functions

template <typename Tag = void, std::invocable Fn>
inline void init_per_device(details::MaybeDevice device, Fn&& fn) {
  return init_per_device<Tag>(device.device_id, true, std::forward<Fn>(fn));
}

}  // namespace host::runtime

}  // namespace sglang
