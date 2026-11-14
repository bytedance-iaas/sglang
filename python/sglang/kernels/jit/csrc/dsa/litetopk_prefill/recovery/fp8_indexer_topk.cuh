// SPDX-License-Identifier: Apache-2.0
//
// SGLang TVM-FFI wrapper for the FP8 LiteTopK producer, exact candidate
// selector and GPU-only recovery adapted from vLLM PR #65920 at
// 833266447780a74713c574e67f3b3bf9caa868eb.  The device sources in this
// directory retain their original provenance headers.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "fp8_indexer.cuh"
#include "topk_candidates_fp32.cuh"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>

namespace sglang {

namespace litetopk_host {

inline CUtensorMap make_tma_map_2d(
    void* ptr,
    CUtensorMapDataType dtype,
    std::array<cuuint64_t, 2> global_dims,
    cuuint64_t global_stride,
    std::array<cuuint32_t, 2> box_dims,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
  const cuuint64_t global_strides[1] = {global_stride};
  const cuuint32_t element_strides[2] = {1, 1};
  CUtensorMap map{};
  const CUresult result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &map,
      dtype,
      2,
      ptr,
      global_dims.data(),
      global_strides,
      box_dims.data(),
      element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK_HOST(result == CUDA_SUCCESS) << "cuTensorMapEncodeTiled(2D) failed: " << static_cast<int>(result);
  return map;
}

inline CUtensorMap make_tma_map_3d(
    void* ptr,
    CUtensorMapDataType dtype,
    std::array<cuuint64_t, 3> global_dims,
    std::array<cuuint64_t, 2> global_strides,
    std::array<cuuint32_t, 3> box_dims,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
  const cuuint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap map{};
  const CUresult result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &map,
      dtype,
      3,
      ptr,
      global_dims.data(),
      global_strides.data(),
      box_dims.data(),
      element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK_HOST(result == CUDA_SUCCESS) << "cuTensorMapEncodeTiled(3D) failed: " << static_cast<int>(result);
  return map;
}

template <auto Kernel>
inline void set_max_dynamic_smem_per_device(int32_t device_id, std::size_t bytes) {
  static std::atomic_uint64_t initialized{};
  const auto device = static_cast<uint32_t>(device_id);
  const uint64_t mask = device < 64 ? uint64_t{1} << device : 0;
  if (mask != 0 && (initialized.load(std::memory_order_relaxed) & mask) != 0) return;
  host::RuntimeDeviceCheck(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(Kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
  if (mask != 0) initialized.fetch_or(mask, std::memory_order_relaxed);
}

}  // namespace litetopk_host

template <uint32_t kNumHeads, uint32_t kTopK>
struct Fp8IndexerTopKRecoveryKernel {
  using Impl = fused_mqa_topk::device::fp8_indexer::Fp8Impl<kNumHeads, false>;
  using RecoveryImpl = fused_mqa_topk::device::fp8_indexer::Fp8Impl<kNumHeads, true>;
  using SelectImpl = fused_mqa_topk::TopKCandidatesFP32Impl<kTopK>;
  static_assert(Impl::kSmemBytes == RecoveryImpl::kSmemBytes);
  static_assert(Impl::kBlockQ == RecoveryImpl::kBlockQ);

  static void
  run(tvm::ffi::TensorView q,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView kv,
      tvm::ffi::TensorView kv_scale,
      tvm::ffi::TensorView ks,
      tvm::ffi::TensorView ke,
      tvm::ffi::TensorView counter,
      tvm::ffi::TensorView codes,
      tvm::ffi::TensorView values,
      tvm::ffi::TensorView indices,
      tvm::ffi::TensorView row_state,
      tvm::ffi::TensorView recovery,
      tvm::ffi::TensorView out,
      int64_t num_ctas,
      int64_t tail_chunks) {
    using namespace host;
    constexpr uint32_t kHeadDim = Impl::kHeadDim;
    auto R = SymbolicSize{"rows"};
    auto K = SymbolicSize{"keys"};
    auto CAP = SymbolicSize{"cap"};
    auto COUNTER = SymbolicSize{"counter"};
    auto CODES = SymbolicSize{"codes"};
    auto RECOVERY = SymbolicSize{"recovery"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({R, kNumHeads, kHeadDim}).with_dtype<fp8_e4m3_t>().with_device(device_).verify(q);
    TensorMatcher({R, kNumHeads}).with_dtype<fp32_t>().with_device(device_).verify(weights);
    TensorMatcher({K, kHeadDim}).with_dtype<fp8_e4m3_t>().with_device(device_).verify(kv);
    TensorMatcher({K}).with_dtype<fp32_t>().with_device(device_).verify(kv_scale);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(ks).verify(ke);
    TensorMatcher({COUNTER}).with_dtype<int32_t>().with_device(device_).verify(counter);
    TensorMatcher({CODES}).with_dtype<int32_t>().with_device(device_).verify(codes);
    TensorMatcher({R, CAP}).with_dtype<fp32_t>().with_device(device_).verify(values);
    TensorMatcher({R, CAP}).with_dtype<int32_t>().with_device(device_).verify(indices);
    TensorMatcher({3, R}).with_dtype<int32_t>().with_device(device_).verify(row_state);
    TensorMatcher({RECOVERY}).with_dtype<fp32_t>().with_device(device_).verify(recovery);
    TensorMatcher({R, kTopK}).with_dtype<int32_t>().with_device(device_).verify(out);

    const int64_t num_rows = R.unwrap();
    const int64_t num_keys = K.unwrap();
    const int64_t cap = CAP.unwrap();
    const uint32_t num_blocks = static_cast<uint32_t>(div_ceil(num_rows, int64_t{Impl::kBlockQ}));
    CHECK_HOST(num_rows > 0 && num_rows * kNumHeads <= INT32_MAX) << "rows out of range";
    CHECK_HOST(num_keys > 0 && num_keys <= (int64_t{1} << 20)) << "keys out of range";
    CHECK_HOST(cap >= 4096 && cap <= 12288 && cap % 4 == 0) << "candidate cap out of range";
    CHECK_HOST(COUNTER.unwrap() >= 2 * int64_t{num_blocks}) << "counter too small";
    CHECK_HOST(RECOVERY.unwrap() >= int64_t{Impl::kBlockQ} * num_keys)
        << "recovery must hold at least one query block x keys";
    CHECK_HOST(num_ctas > 0 && tail_chunks > 0) << "invalid launch geometry";
    CHECK_HOST(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0) << "q must be 16-byte aligned";
    CHECK_HOST(reinterpret_cast<uintptr_t>(weights.data_ptr()) % 16 == 0) << "weights must be 16-byte aligned";
    CHECK_HOST(reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0) << "kv must be 16-byte aligned";
    CHECK_HOST(reinterpret_cast<uintptr_t>(kv_scale.data_ptr()) % 16 == 0) << "kv_scale must be 16-byte aligned";
    CHECK_HOST(reinterpret_cast<uintptr_t>(codes.data_ptr()) % 16 == 0) << "codes must be 16-byte aligned";
    CHECK_HOST(reinterpret_cast<uintptr_t>(values.data_ptr()) % 16 == 0) << "values must be 16-byte aligned";
    CHECK_HOST(reinterpret_cast<uintptr_t>(indices.data_ptr()) % 16 == 0) << "indices must be 16-byte aligned";
    CHECK_HOST(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0) << "out must be 16-byte aligned";
    const DLDevice device = device_.unwrap();

    fused_mqa_topk::Fp8IndexerTensorMaps tmaps{};
    tmaps.q = litetopk_host::make_tma_map_2d(
        q.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        {kHeadDim, static_cast<uint64_t>(num_rows) * kNumHeads},
        kHeadDim,
        {kHeadDim, Impl::kQTmaRows},
        CU_TENSOR_MAP_SWIZZLE_128B);
    tmaps.weights = litetopk_host::make_tma_map_2d(
        weights.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        {kNumHeads, static_cast<uint64_t>(num_rows)},
        kNumHeads * 4ull,
        {kNumHeads, Impl::kBlockQ});
    tmaps.kv = litetopk_host::make_tma_map_2d(
        kv.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        {kHeadDim, static_cast<uint64_t>(num_keys)},
        kHeadDim,
        {kHeadDim, Impl::kSplitKV},
        CU_TENSOR_MAP_SWIZZLE_128B);
    for (uint32_t m = 0; m < 2; ++m) {
      const uint64_t stride = Impl::sample_stride(m);
      const uint64_t rows_s = std::max<uint64_t>(static_cast<uint64_t>(num_keys) / stride, 1);
      tmaps.kv_strided[m] = litetopk_host::make_tma_map_3d(
          kv.data_ptr(),
          CU_TENSOR_MAP_DATA_TYPE_UINT8,
          {kHeadDim, stride, rows_s},
          {kHeadDim, stride * kHeadDim},
          {kHeadDim, 1, Impl::kSplitKV},
          CU_TENSOR_MAP_SWIZZLE_128B);
    }
    tmaps.sf_kv = litetopk_host::make_tma_map_2d(
        kv_scale.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        {static_cast<uint64_t>(num_keys), 1},
        static_cast<uint64_t>(div_ceil(num_keys * 4, int64_t{16}) * 16),
        {Impl::kSplitKV, 1});

    auto* counts = static_cast<int32_t*>(row_state.data_ptr());
    auto* thresholds = reinterpret_cast<float*>(counts + num_rows);
    fused_mqa_topk::Fp8IndexerParams params{};
    params.ks = static_cast<const int32_t*>(ks.data_ptr());
    params.ke = static_cast<const int32_t*>(ke.data_ptr());
    params.kv_scale = static_cast<const float*>(kv_scale.data_ptr());
    params.block_ready = static_cast<uint32_t*>(counter.data_ptr());
    params.block_done = params.block_ready + num_blocks;
    params.sample_codes = static_cast<uint2*>(codes.data_ptr());
    params.cand_scores = static_cast<float*>(values.data_ptr());
    params.cand_indices = static_cast<int32_t*>(indices.data_ptr());
    params.cand_count = counts;
    params.thresholds = thresholds;
    params.overflow = counts + 2 * num_rows;
    params.num_rows = static_cast<uint32_t>(num_rows);
    params.num_blocks = num_blocks;
    params.num_keys = static_cast<uint32_t>(num_keys);
    for (uint32_t m = 0; m < 2; ++m) {
      params.sample_limit[m] = static_cast<uint32_t>(num_keys / Impl::sample_stride(m) * Impl::sample_stride(m));
    }
    params.cap = static_cast<uint32_t>(cap);
    params.topk = kTopK;
    params.dense_bound = kTopK;
    params.tail_chunks = static_cast<uint32_t>(tail_chunks);

    const auto producer = fused_mqa_topk::device::fp8_indexer::fp8_indexer_kernel<Impl>;
    litetopk_host::set_max_dynamic_smem_per_device<fused_mqa_topk::device::fp8_indexer::fp8_indexer_kernel<Impl>>(
        device.device_id, Impl::kSmemBytes);
    const auto sm_count = runtime::get_sm_count(device.device_id);
    const uint32_t grid =
        std::min({static_cast<uint32_t>(num_ctas), sm_count, num_blocks * static_cast<uint32_t>(tail_chunks)});
    CHECK_HOST(CODES.unwrap() >= 2 * int64_t{grid} * int64_t{Impl::kCodesPerCta}) << "sample codes too small";
    LaunchKernel(grid, Impl::kNumThreads, device, Impl::kSmemBytes)(producer, params, tmaps);

    const fused_mqa_topk::TopKCandidatesFP32Params select_params{
        .scores = static_cast<const float*>(values.data_ptr()),
        .indices = static_cast<const int32_t*>(indices.data_ptr()),
        .count = counts,
        .thr = thresholds,
        .out = static_cast<int32_t*>(out.data_ptr()),
        .cap = static_cast<uint32_t>(cap),
    };
    const auto selector = fused_mqa_topk::topk_candidates_fp32_kernel<SelectImpl>;
    litetopk_host::set_max_dynamic_smem_per_device<fused_mqa_topk::topk_candidates_fp32_kernel<SelectImpl>>(
        device.device_id, SelectImpl::kSmemBytes);
    LaunchKernel(static_cast<uint32_t>(num_rows), SelectImpl::kBlockSize, device, SelectImpl::kSmemBytes)
        .config({.use_pdl = true})(selector, select_params);

    params.recovery_scores = static_cast<float*>(recovery.data_ptr());
    params.recovery_out = static_cast<int32_t*>(out.data_ptr());
    params.recovery_weights = static_cast<const float*>(weights.data_ptr());
    params.recovery_stride = num_keys;
    params.recovery_slots =
        static_cast<uint32_t>(std::min<int64_t>(32, RECOVERY.unwrap() / (int64_t{Impl::kBlockQ} * num_keys)));
    CHECK_HOST(params.recovery_slots > 0) << "no recovery slot";
    const auto recover = fused_mqa_topk::device::fp8_indexer::fp8_recovery_kernel<RecoveryImpl>;
    litetopk_host::set_max_dynamic_smem_per_device<
        fused_mqa_topk::device::fp8_indexer::fp8_recovery_kernel<RecoveryImpl>>(
        device.device_id, RecoveryImpl::kSmemBytes);
    LaunchKernel(num_blocks, RecoveryImpl::kNumThreads, device, RecoveryImpl::kSmemBytes)(recover, params, tmaps);
  }
};

}  // namespace sglang
