/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// SM90 FP8 Tensor Core index-logits kernel for request-major speculative rows.
// One CTA owns one request group and 64 compressed positions. The packed FP4 K
// tile is decoded once and reused by every query row in the group.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>

#include "params.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sglang {
namespace fp4_grouped_indexer_sm90 {

using namespace cute;
using bf16 = cutlass::bfloat16_t;
using fp8 = cutlass::float_e4m3_t;

#define FP4_INDEXER_CUDA_CHECK(call)                                                        \
  do {                                                                                      \
    cudaError_t err = (call);                                                               \
    if (err != cudaSuccess) {                                                               \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                              \
    }                                                                                       \
  } while (0)

__host__ __device__ __forceinline__ constexpr int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

__device__ __forceinline__ uint16_t f32x2_to_e4m3x2(float lo, float hi) {
  uint16_t out;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(out) : "f"(hi), "f"(lo));
  return out;
}

__device__ __forceinline__ float ue8m0_to_f32(uint8_t exponent) {
  return __uint_as_float(static_cast<uint32_t>(exponent) << 23);
}

__device__ __forceinline__ float fp4_query_scale(float amax) {
  float x = fmaxf(amax * (1.0f / 6.0f), 1.0e-4f);
  uint32_t bits = __float_as_uint(x);
  uint32_t exponent = (bits >> 23) & 0xff;
  exponent += (bits & 0x7fffff) != 0;
  exponent = min(max(exponent, 1u), 254u);
  return __uint_as_float(exponent << 23);
}

__device__ __forceinline__ uint8_t e2m1_to_e4m3(uint8_t code) {
  constexpr uint64_t lut = 0x4c4844403c383000ULL;
  uint8_t magnitude = static_cast<uint8_t>((lut >> ((code & 7) * 8)) & 0xff);
  return magnitude | ((code & 8) << 4);
}

template <typename Kernel>
__global__ void fp4_grouped_indexer_kernel(__grid_constant__ const Sm90Fp4GroupedIndexerParams params);

struct Sm90Fp4GroupedIndexerKernel {
  static constexpr int HEADS = 64;
  static constexpr int HEAD_DIM = 128;
  static constexpr int BLOCK_L = 64;
  static constexpr int SCALE_GROUPS = 4;
  static constexpr int SCALE_GROUP_SIZE = 32;
  static constexpr int NUM_WARPGROUPS = 2;
  static constexpr int NUM_THREADS = 128 * NUM_WARPGROUPS;

  using SmemLayout =
      decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8>{}, Shape<Int<64>, Int<64>>{}, Step<_1, _2>{}));
  using TiledMMA =
      decltype(make_tiled_mma(GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN<>{}, Layout<Shape<_1, _1, _1>>{}));

  struct SharedStorage {
    array_aligned<fp8, cosize_v<SmemLayout>, 128> q[NUM_WARPGROUPS];
    array_aligned<fp8, cosize_v<SmemLayout>, 128> k[SCALE_GROUPS];
    array_aligned<bf16, HEADS * BLOCK_L, 128> scores[NUM_WARPGROUPS];
    int32_t slots[BLOCK_L];
    float k_scales[SCALE_GROUPS][BLOCK_L];
    float q_scales[NUM_WARPGROUPS][HEADS];
  };

  template <typename TA, typename TB, typename TC>
  static __device__ __forceinline__ void gemm_one_k32(
      TiledMMA& mma, TA const& sQ, TB const& sK, TC& acc, int tid) {
    ThrMMA thr_mma = mma.get_slice(tid);
    Tensor q_frag = thr_mma.partition_fragment_A(sQ);
    Tensor k_frag = thr_mma.partition_fragment_B(sK);
    static_assert(size<2>(q_frag) == 2);
    static_assert(size<2>(k_frag) == 2);
    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    mma.accumulate_ = GMMA::ScaleOut::Zero;
    cute::gemm(mma, q_frag(_, _, 0), k_frag(_, _, 0), acc);
    warpgroup_fence_operand(acc);
  }

  static __device__ __forceinline__ void devfunc(const Sm90Fp4GroupedIndexerParams& p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
    const int tid = threadIdx.x;
    const int warpgroup = tid / 128;
    const int wg_tid = tid % 128;
    const int l0 = blockIdx.x * BLOCK_L;
    const int b0 = blockIdx.y * p.group_size;
    const int group_rows = min(p.group_size, p.batch_size - b0);

    extern __shared__ char smem_raw[];
    SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_raw);
    Tensor sQ = make_tensor(make_smem_ptr(ss.q[warpgroup].data()), SmemLayout{});
    Tensor sK0 = make_tensor(make_smem_ptr(ss.k[0].data()), SmemLayout{});
    Tensor sK1 = make_tensor(make_smem_ptr(ss.k[1].data()), SmemLayout{});
    Tensor sK2 = make_tensor(make_smem_ptr(ss.k[2].data()), SmemLayout{});
    Tensor sK3 = make_tensor(make_smem_ptr(ss.k[3].data()), SmemLayout{});

    const int64_t* req = reinterpret_cast<const int64_t*>(p.req);
    const int64_t* lens = reinterpret_cast<const int64_t*>(p.lens);
    const int32_t* req_to_token = reinterpret_cast<const int32_t*>(p.req_to_token);
    const uint8_t* table = reinterpret_cast<const uint8_t*>(p.table);
    const int64_t request = req[b0];

    if (tid < BLOCK_L) {
      const int position = l0 + tid;
      int32_t slot = 0;
      if (position < p.width) {
        slot = req_to_token[request * p.req_stride + static_cast<int64_t>(position) * p.ratio] / p.ratio;
      }
      ss.slots[tid] = slot;
    }
    __syncthreads();

    for (int idx = tid; idx < SCALE_GROUPS * BLOCK_L; idx += NUM_THREADS) {
      const int g = idx / BLOCK_L;
      const int col = idx % BLOCK_L;
      const int slot = ss.slots[col];
      const int page = slot / p.page_size;
      const int off = slot - page * p.page_size;
      const uint8_t exponent =
          table[static_cast<int64_t>(page) * p.table_stride + p.page_size * 64 + off * 4 + g];
      ss.k_scales[g][col] = ue8m0_to_f32(exponent);
    }

    for (int idx = tid; idx < SCALE_GROUPS * BLOCK_L * SCALE_GROUP_SIZE; idx += NUM_THREADS) {
      const int g = idx / (BLOCK_L * SCALE_GROUP_SIZE);
      const int rem = idx % (BLOCK_L * SCALE_GROUP_SIZE);
      const int col = rem / SCALE_GROUP_SIZE;
      const int d = rem % SCALE_GROUP_SIZE;
      const int slot = ss.slots[col];
      const int page = slot / p.page_size;
      const int off = slot - page * p.page_size;
      const int packed_col = g * 16 + d / 2;
      const uint8_t packed =
          table[static_cast<int64_t>(page) * p.table_stride + off * 64 + packed_col];
      const uint8_t code = (d & 1) ? (packed >> 4) : (packed & 0xf);
      fp8 value;
      *reinterpret_cast<uint8_t*>(&value) = e2m1_to_e4m3(code);
      switch (g) {
        case 0:
          sK0(col, d) = value;
          break;
        case 1:
          sK1(col, d) = value;
          break;
        case 2:
          sK2(col, d) = value;
          break;
        default:
          sK3(col, d) = value;
          break;
      }
    }
    __syncthreads();

    const bf16* q = reinterpret_cast<const bf16*>(p.q);
    const bf16* weights = reinterpret_cast<const bf16*>(p.weights);
    float* out = reinterpret_cast<float*>(p.out);
    TiledMMA mma;

    for (int row_in_group = warpgroup; row_in_group < group_rows; row_in_group += NUM_WARPGROUPS) {
      const int b = b0 + row_in_group;
      Tensor total = partition_fragment_C(mma, Shape<Int<HEADS>, Int<BLOCK_L>>{});
      cute::fill(total, 0.0f);

      CUTE_UNROLL
      for (int g = 0; g < SCALE_GROUPS; ++g) {
        if (wg_tid < HEADS) {
          float amax = 0.0f;
          const bf16* q_head = q + static_cast<int64_t>(b) * p.q_stride_b +
                               static_cast<int64_t>(wg_tid) * p.q_stride_h + g * SCALE_GROUP_SIZE;
          CUTE_UNROLL
          for (int d = 0; d < SCALE_GROUP_SIZE; ++d) {
            amax = fmaxf(amax, fabsf(static_cast<float>(q_head[d])));
          }
          ss.q_scales[warpgroup][wg_tid] = fp4_query_scale(amax);
        }
        NamedBarrier::arrive_and_wait(128, warpgroup);

        for (int pair = wg_tid; pair < HEADS * (SCALE_GROUP_SIZE / 2); pair += 128) {
          const int head = pair / (SCALE_GROUP_SIZE / 2);
          const int d = (pair % (SCALE_GROUP_SIZE / 2)) * 2;
          const bf16* src = q + static_cast<int64_t>(b) * p.q_stride_b +
                            static_cast<int64_t>(head) * p.q_stride_h + g * SCALE_GROUP_SIZE + d;
          const float inv_scale = 1.0f / ss.q_scales[warpgroup][head];
          const uint16_t packed = f32x2_to_e4m3x2(
              static_cast<float>(src[0]) * inv_scale, static_cast<float>(src[1]) * inv_scale);
          fp8 v0, v1;
          *reinterpret_cast<uint8_t*>(&v0) = packed & 0xff;
          *reinterpret_cast<uint8_t*>(&v1) = packed >> 8;
          sQ(head, d) = v0;
          sQ(head, d + 1) = v1;
        }
        NamedBarrier::arrive_and_wait(128, warpgroup);

        Tensor part = partition_fragment_C(mma, Shape<Int<HEADS>, Int<BLOCK_L>>{});
        if (g == 0) {
          gemm_one_k32(mma, sQ, sK0, part, wg_tid);
        } else if (g == 1) {
          gemm_one_k32(mma, sQ, sK1, part, wg_tid);
        } else if (g == 2) {
          gemm_one_k32(mma, sQ, sK2, part, wg_tid);
        } else {
          gemm_one_k32(mma, sQ, sK3, part, wg_tid);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(part);

        CUTE_UNROLL
        for (int rp = 0; rp < 2; ++rp) {
          const int head = (wg_tid / 32) * 16 + (wg_tid % 32) / 4 + 8 * rp;
          CUTE_UNROLL
          for (int j = 0; j < BLOCK_L / 8; ++j) {
            CUTE_UNROLL
            for (int cp = 0; cp < 2; ++cp) {
              const int col = (wg_tid % 4) * 2 + 8 * j + cp;
              const int i = j * 4 + rp * 2 + cp;
              total(i) +=
                  part(i) * ss.q_scales[warpgroup][head] * ss.k_scales[g][col];
            }
          }
        }
        NamedBarrier::arrive_and_wait(128, warpgroup);
      }

      CUTE_UNROLL
      for (int rp = 0; rp < 2; ++rp) {
        const int head = (wg_tid / 32) * 16 + (wg_tid % 32) / 4 + 8 * rp;
        CUTE_UNROLL
        for (int j = 0; j < BLOCK_L / 8; ++j) {
          CUTE_UNROLL
          for (int cp = 0; cp < 2; ++cp) {
            const int col = (wg_tid % 4) * 2 + 8 * j + cp;
            const int i = j * 4 + rp * 2 + cp;
            ss.scores[warpgroup][head * BLOCK_L + col] = bf16(total(i));
          }
        }
      }
      NamedBarrier::arrive_and_wait(128, warpgroup);

      if (wg_tid < BLOCK_L) {
        float sum = 0.0f;
        CUTE_UNROLL
        for (int head = 0; head < HEADS; ++head) {
          float score =
              fmaxf(static_cast<float>(ss.scores[warpgroup][head * BLOCK_L + wg_tid]), 0.0f);
          const float weight = static_cast<float>(weights[static_cast<int64_t>(b) * p.weight_stride_b + head]);
          sum += static_cast<float>(bf16(score * weight));
        }
        const int position = l0 + wg_tid;
        if (position < p.width) {
          const bool valid = position < lens[b];
          out[static_cast<int64_t>(b) * p.out_stride + position] =
              valid ? static_cast<float>(bf16(sum)) : -INFINITY;
        }
      }
      NamedBarrier::arrive_and_wait(128, warpgroup);
    }
#else
    if (cute::thread0()) {
      CUTE_INVALID_CONTROL_PATH("sm90_fp4_grouped_indexer only supports sm90");
    }
#endif
  }

  static void run(const Sm90Fp4GroupedIndexerParams& p) {
    auto kernel = &fp4_grouped_indexer_kernel<Sm90Fp4GroupedIndexerKernel>;
    constexpr size_t smem_size = sizeof(SharedStorage);
    static bool attr_set = [&]() {
      FP4_INDEXER_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      return true;
    }();
    (void)attr_set;
    dim3 grid(ceil_div(p.width, BLOCK_L), ceil_div(p.batch_size, p.group_size), 1);
    kernel<<<grid, NUM_THREADS, smem_size, p.stream>>>(p);
    FP4_INDEXER_CUDA_CHECK(cudaGetLastError());
  }
};

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS)
    fp4_grouped_indexer_kernel(__grid_constant__ const Sm90Fp4GroupedIndexerParams params) {
  Kernel::devfunc(params);
}

inline void run_sm90_fp4_grouped_indexer(const Sm90Fp4GroupedIndexerParams& params) {
  Sm90Fp4GroupedIndexerKernel::run(params);
}

}  // namespace fp4_grouped_indexer_sm90
}  // namespace sglang
