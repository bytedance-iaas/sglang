#pragma once
#include <sgl_kernel/early_moe/vendor/common/math.cuh>
#include <sgl_kernel/early_moe/vendor/mma/sm90.cuh>
#include <sgl_kernel/early_moe/vendor/ptx/wgmma.cuh>

#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <early_moe/comm/gate_ready.cuh>

#include <cooperative_groups.h>
#include <cuda_fp8.h>

namespace early_moe {
using Barrier = cutlass::arch::ClusterTransactionBarrier;
using FP8 = cutlass::float_e4m3_t;
namespace ptx = iaas_kernels::ptx;
namespace mma = iaas_kernels::mma::sm90;
__device__ __forceinline__ void async_global_fence() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}
__device__ __forceinline__ void load_tma(const CUtensorMap* d, Barrier* b, void* dst, int x, int y) {
  cute::SM90_TMA_LOAD_2D::copy(
      d, reinterpret_cast<uint64_t*>(b), uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL), dst, x, y);
}
}  // namespace early_moe
