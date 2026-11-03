#pragma once
#include <early_moe/impls/sm90_early_iaas.cuh>

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace early_moe {
inline void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
inline CUtensorMap descriptor(
    void* ptr,
    uint64_t inner,
    uint64_t outer,
    uint32_t box_inner,
    uint32_t box_outer,
    bool fp32 = false,
    bool swizzle = true) {
  CUtensorMap d{};
  uint64_t dims[2] = {inner, outer}, strides[1] = {inner * (fp32 ? 4u : 1u)};
  uint32_t box[2] = {box_inner, box_outer}, elem[2] = {1, 1};
  auto e = cuTensorMapEncodeTiled(
      &d,
      fp32 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_UINT8,
      2,
      ptr,
      dims,
      strides,
      box,
      elem,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (e != CUDA_SUCCESS) throw std::runtime_error("cuTensorMapEncodeTiled failed: " + std::to_string(e));
  return d;
}
struct Runtime {
  KernelArgs args{};
  const void* kernel;
  int shared, capacity, workers, stages, tile_n, threads, groups;
  cudaEvent_t completion;
  cudaStream_t stream{};
  bool launched = false;
  Runtime(int workers_, int stages_, int tile_n_, int groups_)
      : workers(workers_), stages(stages_), tile_n(tile_n_), groups(groups_) {
    if (groups < 1 || groups > MAX_GROUPS || workers < 1 || tile_n != 256)
      throw std::runtime_error("requires BN256 and a positive worker count with 1..16 inflight groups");
    int dev;
    check(cudaGetDevice(&dev));
    cudaDeviceProp p;
    check(cudaGetDeviceProperties(&p, dev));
    if (p.major != 9 || !p.cooperativeLaunch) throw std::runtime_error("requires SM90 cooperative launch");
    if (stages != 4) throw std::runtime_error("BN256 requires four persistent stages");
    threads = 384;
    kernel = reinterpret_cast<const void*>(fused_early_iaas_sm90<4>);
    shared = sizeof(IAASShared<4>);
    check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    int blocks;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, shared));
    capacity = blocks * p.multiProcessorCount;
    if (workers < 1 || workers > MAX_WORKERS || workers + 1 > capacity)
      throw std::runtime_error("worker grid exceeds final kernel cooperative occupancy");
    check(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming));
  }
  ~Runtime() {
    cudaEventDestroy(completion);
  }
  bool busy() const {
    if (!launched) return false;
    auto e = cudaEventQuery(completion);
    if (e == cudaErrorNotReady) return true;
    check(e);
    return false;
  }
  void launch(const Call& c, cudaStream_t next_stream) {
    if (launched && stream != next_stream && busy())
      throw std::runtime_error("context is already in flight on another stream");
    args.c = c;
    if (c.workers != workers || c.groups != groups) throw std::runtime_error("worker/group count mismatch");
    void* params[] = {&args};
    check(cudaLaunchCooperativeKernel(kernel, dim3(workers + 1), dim3(threads), params, shared, next_stream));
    check(cudaEventRecord(completion, next_stream));
    stream = next_stream;
    launched = true;
  }
};
}  // namespace early_moe
