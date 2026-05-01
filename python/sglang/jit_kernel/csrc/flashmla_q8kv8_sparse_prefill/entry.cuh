// entry.cuh -- JIT dispatch entry for Q8KV8 sparse prefill attention kernel (SM90).
// Bridges Python (tvm_ffi) → SparseAttnFwdQ8SM90NewParams → run_fwd_phase1_q8_sm90_new_kernel.
#pragma once

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "phase1_new.cuh"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

// Cache num_sm per device.  Querying it every call via
// cudaDeviceGetAttribute takes the CUDA driver's global lock and serializes
// kernel launches across 8-DP processes under high concurrency, which was
// observed to cause a cuLaunchKernel->sched_yield deadlock at 256+
// concurrent requests.  The original (prebuilt) FlashMLA interface uses
// at::cuda::getCurrentDeviceProperties() which has the same caching effect.
static int _q8kv8_num_sm_cache[16] = {0};

static inline int _get_num_sm_cached(int device_id) {
  if (device_id < 0 || device_id >= 16) {
    int n;
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device_id);
    return n;
  }
  int cached = _q8kv8_num_sm_cache[device_id];
  if (cached == 0) {
    cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, device_id);
    _q8kv8_num_sm_cache[device_id] = cached;
  }
  return cached;
}

// Main dispatch function: takes TVM TensorView arguments and scalars.
void sparse_prefill_q8kv8_dispatch(
    tvm::ffi::TensorView q,           // [s_q, h_q, d_qk], uint8 (float8_e4m3fn storage)
    tvm::ffi::TensorView kv,          // [s_kv, h_kv, d_qk], uint8 (float8_e4m3fn storage)
    tvm::ffi::TensorView indices,     // [s_q, h_kv, topk], int32
    tvm::ffi::TensorView q_scale,     // scalar tensor, float32
    tvm::ffi::TensorView kv_scale,    // scalar tensor, float32
    tvm::ffi::TensorView out,         // [s_q, h_q, d_v], bfloat16
    tvm::ffi::TensorView max_logits,  // [s_q, h_q], float32
    tvm::ffi::TensorView lse,         // [s_q, h_q], float32
    int64_t s_q_val,
    int64_t s_kv_val,
    int64_t h_q_val,
    int64_t h_kv_val,
    int64_t d_qk_val,
    int64_t d_v_val,
    int64_t topk_val,
    double sm_scale_val,
    int64_t has_attn_sink,    // boolean flag: 1 if attn_sink provided
    int64_t has_topk_length,  // boolean flag: 1 if topk_length provided
    int64_t cuda_stream       // cudaStream_t as int64
) {
  SparseAttnFwdQ8SM90NewParams params;
  params.s_q = (int)s_q_val;
  params.s_kv = (int)s_kv_val;
  params.h_q = (int)h_q_val;
  params.h_kv = (int)h_kv_val;
  params.d_qk = (int)d_qk_val;
  params.d_v = (int)d_v_val;
  params.topk = (int)topk_val;
  params.sm_scale = (float)sm_scale_val;
  params.sm_scale_div_log2 = (float)sm_scale_val * (float)M_LOG2E;

  params.q = reinterpret_cast<const uint8_t*>(q.data_ptr());
  params.kv = reinterpret_cast<const uint8_t*>(kv.data_ptr());
  params.indices = static_cast<int*>(indices.data_ptr());

  // q_scale and kv_scale are scalar tensors on GPU
  params.q_scale = 0.0f;   // will use ptr
  params.kv_scale = 0.0f;  // will use ptr
  params.q_scale_ptr = static_cast<const float*>(q_scale.data_ptr());
  params.kv_scale_ptr = static_cast<const float*>(kv_scale.data_ptr());

  params.attn_sink = nullptr;
  params.topk_length = nullptr;

  // Zero-overhead Q8 prefix scaffolding -- nullptrs => current path (per-tensor scale, contiguous fp8 kv).
  params.kv_page_table = nullptr;
  params.kv_group_scale_ptr = nullptr;
  params.kv_rope_bf16_ptr = nullptr;
  params.kv_group_scale_stride_token = 0;
  params.kv_rope_stride_token = 0;

  // Strides: layout is [s_q, h_q, d_qk] for Q, [s_kv, h_kv, d_qk] for KV
  params.stride_q_s_q = (int)h_q_val * (int)d_qk_val;
  params.stride_q_h_q = (int)d_qk_val;
  params.stride_kv_s_kv = (int64_t)h_kv_val * (int64_t)d_qk_val;
  params.stride_kv_h_kv = (int)d_qk_val;
  params.stride_indices_s_q = (int)h_kv_val * (int)topk_val;
  params.stride_indices_h_kv = (int)topk_val;

  params.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
  params.max_logits = static_cast<float*>(max_logits.data_ptr());
  params.lse = static_cast<float*>(lse.data_ptr());

  // Match prebuilt FlashMLA: set current device (like at::cuda::CUDAGuard)
  // and use caller-provided stream. num_sm is cached to avoid per-call
  // cudaDeviceGetAttribute (matches prebuilt's cached device_prop).
  DLDevice dev = q.device();
  cudaSetDevice(dev.device_id);
  params.stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  params.num_sm = _get_num_sm_cached(dev.device_id);

  // Dispatch based on d_qk and topk_length presence
  bool have_topk_length = (has_topk_length != 0);
  int d_qk = (int)d_qk_val;

  if (d_qk == 512) {
    if (have_topk_length) {
      sm90::fwd::run_fwd_phase1_q8_sm90_new_kernel<512, true>(params);
    } else {
      sm90::fwd::run_fwd_phase1_q8_sm90_new_kernel<512, false>(params);
    }
  } else if (d_qk == 576) {
    if (have_topk_length) {
      sm90::fwd::run_fwd_phase1_q8_sm90_new_kernel<576, true>(params);
    } else {
      sm90::fwd::run_fwd_phase1_q8_sm90_new_kernel<576, false>(params);
    }
  } else {
    fprintf(stderr, "sparse_prefill_q8kv8: unsupported d_qk=%d (must be 512 or 576)\n", d_qk);
    exit(1);
  }
}

// Variant with attn_sink and topk_length tensors
void sparse_prefill_q8kv8_dispatch_full(
    tvm::ffi::TensorView q,            // [s_q, h_q, d_qk], uint8
    tvm::ffi::TensorView kv,           // [s_kv, h_kv, d_qk], uint8
    tvm::ffi::TensorView indices,      // [s_q, h_kv, topk], int32
    tvm::ffi::TensorView q_scale,      // scalar tensor, float32
    tvm::ffi::TensorView kv_scale,     // scalar tensor, float32
    tvm::ffi::TensorView attn_sink,    // [h_q], float32
    tvm::ffi::TensorView topk_length,  // [s_q], int32
    tvm::ffi::TensorView out,          // [s_q, h_q, d_v], bfloat16
    tvm::ffi::TensorView max_logits,   // [s_q, h_q], float32
    tvm::ffi::TensorView lse,          // [s_q, h_q], float32
    int64_t s_q_val,
    int64_t s_kv_val,
    int64_t h_q_val,
    int64_t h_kv_val,
    int64_t d_qk_val,
    int64_t d_v_val,
    int64_t topk_val,
    double sm_scale_val,
    int64_t cuda_stream) {
  SparseAttnFwdQ8SM90NewParams params;
  params.s_q = (int)s_q_val;
  params.s_kv = (int)s_kv_val;
  params.h_q = (int)h_q_val;
  params.h_kv = (int)h_kv_val;
  params.d_qk = (int)d_qk_val;
  params.d_v = (int)d_v_val;
  params.topk = (int)topk_val;
  params.sm_scale = (float)sm_scale_val;
  params.sm_scale_div_log2 = (float)sm_scale_val * (float)M_LOG2E;

  params.q = reinterpret_cast<const uint8_t*>(q.data_ptr());
  params.kv = reinterpret_cast<const uint8_t*>(kv.data_ptr());
  params.indices = static_cast<int*>(indices.data_ptr());

  params.q_scale = 0.0f;
  params.kv_scale = 0.0f;
  params.q_scale_ptr = static_cast<const float*>(q_scale.data_ptr());
  params.kv_scale_ptr = static_cast<const float*>(kv_scale.data_ptr());

  params.attn_sink = static_cast<float*>(attn_sink.data_ptr());
  params.topk_length = static_cast<int*>(topk_length.data_ptr());

  // Zero-overhead Q8 prefix scaffolding -- nullptrs => current path (per-tensor scale, contiguous fp8 kv).
  params.kv_page_table = nullptr;
  params.kv_group_scale_ptr = nullptr;
  params.kv_rope_bf16_ptr = nullptr;
  params.kv_group_scale_stride_token = 0;
  params.kv_rope_stride_token = 0;

  params.stride_q_s_q = (int)h_q_val * (int)d_qk_val;
  params.stride_q_h_q = (int)d_qk_val;
  params.stride_kv_s_kv = (int64_t)h_kv_val * (int64_t)d_qk_val;
  params.stride_kv_h_kv = (int)d_qk_val;
  params.stride_indices_s_q = (int)h_kv_val * (int)topk_val;
  params.stride_indices_h_kv = (int)topk_val;

  params.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
  params.max_logits = static_cast<float*>(max_logits.data_ptr());
  params.lse = static_cast<float*>(lse.data_ptr());

  // Match prebuilt: set device + cached num_sm.
  DLDevice dev = q.device();
  cudaSetDevice(dev.device_id);
  params.stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  params.num_sm = _get_num_sm_cached(dev.device_id);

  int d_qk = (int)d_qk_val;

  if (d_qk == 512) {
    sm90::fwd::run_fwd_phase1_q8_sm90_new_kernel<512, true>(params);
  } else if (d_qk == 576) {
    sm90::fwd::run_fwd_phase1_q8_sm90_new_kernel<576, true>(params);
  } else {
    fprintf(stderr, "sparse_prefill_q8kv8: unsupported d_qk=%d (must be 512 or 576)\n", d_qk);
    exit(1);
  }
}


}  // anonymous namespace
