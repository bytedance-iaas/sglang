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
// One CTA owns one request group and a chunk of compressed positions. Small
// groups keep decoded Q resident across K tiles; each K tile serves all rows.

#pragma once

#include <sgl_kernel/utils.cuh>  // For LaunchKernel and RuntimeDeviceCheck

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
using fp8 = cutlass::float_e5m2_t;

__host__ __device__ __forceinline__ constexpr int ceil_div(int x, int y) {
  return x / y + (x % y != 0);
}

__device__ __forceinline__ float ue8m0_to_f32(uint8_t exponent) {
  return __uint_as_float(static_cast<uint32_t>(exponent) << 23);
}

__device__ __forceinline__ uint8_t scaled_e2m1_to_e5m2(uint8_t code, int exponent_delta) {
  // Positive E5M2 encodings of {0, .5, 1, 1.5, 2, 3, 4, 6}. Multiplication
  // by a power of two is an exact exponent-field adjustment. The common scale
  // is checked before this fast path; wide-scale inputs use BF16 instead.
  constexpr uint64_t lut = 0x464442403e3c3800ULL;
  const uint8_t magnitude = code & 7;
  if (magnitude == 0) {
    return 0;
  }
  const int base = static_cast<int>((lut >> (magnitude * 8)) & 0xff);
  return static_cast<uint8_t>(base + exponent_delta * 4) | ((code & 8) << 4);
}

__device__ __forceinline__ uint32_t scaled_e2m1x4_to_e5m2x4(uint32_t codes, int exponent_delta) {
  const uint32_t base = __byte_perm(0x3e3c3800u, 0x46444240u, codes & 0x7777u);
  const uint32_t delta = uint32_t(uint8_t(exponent_delta * 4)) * 0x01010101u;
  const uint32_t adjusted = __vadd4(base, delta);
  const uint32_t signs =
      ((codes & 0x8u) << 4) | ((codes & 0x80u) << 8) | ((codes & 0x800u) << 12) | ((codes & 0x8000u) << 16);
  return (adjusted | signs) & ~__vcmpeq4(base, 0u);
}

// Adding exponent bits is exact only while every nonzero E2M1 code stays
// in the normal E5M2 range. min code=.5 => delta>=-13; max code=6 => delta<=12.
__device__ __forceinline__ bool needs_bf16(uint32_t packed_scale) {
  int lo = 255, hi = 1;
  CUTE_UNROLL
  for (int g = 0; g < 4; ++g) {
    const int e = uint8_t(packed_scale >> (g * 8));
    lo = min(lo, e);
    hi = max(hi, e);
  }
  // Normal E5M2 representability alone is not sufficient: FP8 and BF16
  // WGMMA accumulate in a different order. With <=4 scale bits per operand,
  // an E2M1 dot has <=23 significant integer bits, so its FP32 sum is exact.
  // Keep absolute scales moderate to avoid intermediate rescale overflow/FTZ.
  return hi - lo > 4 || lo < 95 || hi > 159;
}

__device__ __forceinline__ bf16 decode_bf16(uint8_t code, uint8_t exponent) {
  const int magnitude = code & 7;
  if (magnitude == 0) return bf16::bitcast(0);
  // Construct the exact BF16 bits, including the smallest-scale subnormal.
  const int base = magnitude == 1 ? -1 : (magnitude / 2 - 1);
  const int significand = (magnitude > 1 && (magnitude & 1)) ? 192 : 128;
  const int e = int(exponent) + base;
  uint16_t bits = e >= 255 ? 0x7f80
                  : e > 0  ? uint16_t((e << 7) | (significand & 127))
                           : uint16_t(significand >> (1 - e));
  return bf16::bitcast(bits | ((code & 8) << 12));
}

template <typename Kernel>
__global__ void fp4_grouped_indexer_kernel(__grid_constant__ const Sm90Fp4GroupedIndexerParams params);

template <int TILES_PER_CTA = 1, int Q_ROWS = 4, bool ALIGNED = true, bool SPECIALIZED = true>
struct Sm90Fp4GroupedIndexerKernel {
  static constexpr int HEADS = 64;
  static constexpr int HEAD_DIM = 128;
  static constexpr int BLOCK_L = 64;
  static constexpr int SCALE_GROUPS = 4;
  static constexpr int SCALE_GROUP_SIZE = 32;
  static constexpr int NUM_WARPGROUPS = 3;
  static constexpr int WARPS_PER_WARPGROUP = 4;
  static constexpr int NUM_THREADS = 128 * NUM_WARPGROUPS;
  static constexpr bool RESIDENT_Q = TILES_PER_CTA > 1;
  static_assert(TILES_PER_CTA >= 1 && Q_ROWS >= NUM_WARPGROUPS);

  using SmemLayout =
      decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8>{}, Shape<Int<64>, Int<128>>{}, Step<_1, _2>{}));
  using TiledMMA = decltype(make_tiled_mma(GMMA::MMA_64x64x32_F32E5M2E5M2_SS_TN<>{}, Layout<Shape<_1, _1, _1>>{}));

  using Bf16Layout =
      decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<bf16>{}, Shape<Int<64>, Int<128>>{}, Step<_1, _2>{}));
  using Bf16MMA = decltype(make_tiled_mma(
      GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}, Layout<Shape<_1, _1, _1>>{}));
  static_assert(2 * sizeof(bf16) * cosize_v<Bf16Layout> <= Q_ROWS * sizeof(fp8) * cosize_v<SmemLayout>);

  struct SharedStorage {
    union {
      array_aligned<fp8, cosize_v<SmemLayout>, 128> q[Q_ROWS];
      struct {
        array_aligned<bf16, cosize_v<Bf16Layout>, 128> q;
        array_aligned<bf16, cosize_v<Bf16Layout>, 128> k;
      } wide;
    };
    array_aligned<fp8, cosize_v<SmemLayout>, 128> k;
    array_aligned<float, WARPS_PER_WARPGROUP * BLOCK_L, 128> warp_sums[NUM_WARPGROUPS];
    int32_t slots[BLOCK_L];
    uint8_t k_exponents[SCALE_GROUPS][BLOCK_L];
    float k_scales[BLOCK_L];
    float q_scales[Q_ROWS][HEADS];
    volatile int tile_starts[TILES_PER_CTA];
  };

  static __device__ __forceinline__ uint32_t load_word(const uint8_t* source) {
    if constexpr (ALIGNED) {
      return *reinterpret_cast<const uint32_t*>(source);
    } else {
      // Public byte-strided cache views need not be four-byte aligned.
      uint32_t word = 0;
      CUTE_UNROLL
      for (int byte = 0; byte < 4; ++byte)
        word |= uint32_t(source[byte]) << (8 * byte);
      return word;
    }
  }

  // Every CTA thread must call this helper, including inactive tail warpgroups.
  static __device__ __forceinline__ void
  decode_q(const Sm90Fp4GroupedIndexerParams& p, SharedStorage& ss, int b, int q_slot, int wg_tid, bool active = true) {
    const uint8_t* q = reinterpret_cast<const uint8_t*>(p.q);
    const uint32_t* q_scale = reinterpret_cast<const uint32_t*>(p.q_scale);
    Tensor sQ = make_tensor(make_smem_ptr(ss.q[q_slot].data()), SmemLayout{});
    if (active && wg_tid < HEADS) {
      const uint32_t packed_scale = q_scale[static_cast<int64_t>(b) * p.q_scale_stride_b + wg_tid];
      uint8_t max_exponent = 1;
      CUTE_UNROLL
      for (int g = 0; g < SCALE_GROUPS; ++g) {
        max_exponent = max(max_exponent, static_cast<uint8_t>(packed_scale >> (8 * g)));
      }
      const uint8_t common_exponent = max(static_cast<int>(max_exponent) - 12, 1);
      ss.q_scales[q_slot][wg_tid] = ue8m0_to_f32(common_exponent);
    }
    __syncthreads();
    if (!active) return;
    for (int word = wg_tid; word < HEADS * (HEAD_DIM / 8); word += 128) {
      const int head = word / (HEAD_DIM / 8);
      const int d = (word % (HEAD_DIM / 8)) * 8;
      const int g = d / SCALE_GROUP_SIZE;
      const uint32_t packed =
          load_word(q + static_cast<int64_t>(b) * p.q_stride_b + static_cast<int64_t>(head) * p.q_stride_h + d / 2);
      const uint32_t packed_scale = q_scale[static_cast<int64_t>(b) * p.q_scale_stride_b + head];
      const int exponent = static_cast<uint8_t>(packed_scale >> (8 * g));
      const int common_exponent = (__float_as_uint(ss.q_scales[q_slot][head]) >> 23) & 0xff;
      const int exponent_delta = exponent - common_exponent;
      CUTE_UNROLL
      for (int half = 0; half < 2; ++half) {
        const uint32_t expanded = scaled_e2m1x4_to_e5m2x4(packed >> (half * 16), exponent_delta);
        *reinterpret_cast<uint32_t*>(&sQ(head, d + half * 4)) = expanded;
      }
    }
    // Publish this thread's generic shared stores to the WGMMA async proxy.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }

  template <typename MMA, typename TA, typename TB, typename TC>
  static __device__ __forceinline__ void gemm_k128(MMA& mma, TA const& sQ, TB const& sK, TC& acc, int tid) {
    ThrMMA thr_mma = mma.get_slice(tid);
    Tensor q_frag = thr_mma.partition_fragment_A(sQ);
    Tensor k_frag = thr_mma.partition_fragment_B(sK);
    static_assert(size<2>(q_frag) == size<2>(k_frag));
    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    mma.accumulate_ = GMMA::ScaleOut::Zero;
    CUTE_UNROLL
    for (int k = 0; k < size<2>(q_frag); ++k) {
      cute::gemm(mma, q_frag(_, _, k), k_frag(_, _, k), acc);
      mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_fence_operand(acc);
  }

  // Uniform CTA slow path. Scratch aliases resident FP8 Q; caller must reload
  // resident Q before returning to the fast path on a later tile.
  static __device__ __forceinline__ void
  bf16_tile(const Sm90Fp4GroupedIndexerParams& p, SharedStorage& ss, int b0, int group_rows, int l0, int max_visible) {
    const int tid = threadIdx.x;
    const int page_size = SPECIALIZED ? 64 : p.page_size;
    const uint8_t* table = reinterpret_cast<const uint8_t*>(p.table);
    const uint8_t* q = reinterpret_cast<const uint8_t*>(p.q);
    const uint32_t* q_scale = reinterpret_cast<const uint32_t*>(p.q_scale);
    const bf16* weights = reinterpret_cast<const bf16*>(p.weights);
    const int64_t* lens = reinterpret_cast<const int64_t*>(p.lens);
    float* out = reinterpret_cast<float*>(p.out);
    Tensor sQ = make_tensor(make_smem_ptr(ss.wide.q.data()), Bf16Layout{});
    Tensor sK = make_tensor(make_smem_ptr(ss.wide.k.data()), Bf16Layout{});
    for (int word = tid; word < BLOCK_L * (HEAD_DIM / 8); word += NUM_THREADS) {
      const int col = word / (HEAD_DIM / 8);
      const int d = (word % (HEAD_DIM / 8)) * 8;
      const int slot = ss.slots[col];
      const int page = slot / page_size;
      const int off = slot - page * page_size;
      const uint32_t packed =
          l0 + col < max_visible ? load_word(table + int64_t(page) * p.table_stride + off * 64 + d / 2) : 0;
      const uint8_t e = ss.k_exponents[d / 32][col];
      CUTE_UNROLL
      for (int j = 0; j < 8; ++j)
        sK(col, (d + j) / 2 + ((d + j) & 1) * 64) = decode_bf16((packed >> (j * 4)) & 15, e);
    }
    for (int row = 0; row < group_rows; ++row) {
      const int b = b0 + row;
      for (int word = tid; word < HEADS * (HEAD_DIM / 8); word += NUM_THREADS) {
        const int head = word / (HEAD_DIM / 8);
        const int d = (word % (HEAD_DIM / 8)) * 8;
        const uint32_t packed = load_word(q + int64_t(b) * p.q_stride_b + int64_t(head) * p.q_stride_h + d / 2);
        const uint8_t e = q_scale[int64_t(b) * p.q_scale_stride_b + head] >> ((d / 32) * 8);
        CUTE_UNROLL
        // Match the original Triton WGMMA order: all even elements, then odd.
        for (int j = 0; j < 8; ++j)
          sQ(head, (d + j) / 2 + ((d + j) & 1) * 64) = decode_bf16((packed >> (j * 4)) & 15, e);
      }
      // Both cooperative K and Q writes must be visible to async MMA readers.
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncthreads();
      if (tid < 128) {
        Bf16MMA mma;
        Tensor acc = partition_fragment_C(mma, Shape<Int<HEADS>, Int<BLOCK_L>>{});
        gemm_k128(mma, sQ, sK, acc, tid);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc);
        const int warp = tid / 32;
        const int lane = tid % 32;
        const int head_in_warp = lane / 4;
        const int head0 = warp * 16 + head_in_warp;
        const int head1 = head0 + 8;
        const float weight0 = float(weights[int64_t(b) * p.weight_stride_b + head0]);
        const float weight1 = float(weights[int64_t(b) * p.weight_stride_b + head1]);
        CUTE_UNROLL
        for (int j = 0; j < BLOCK_L / 8; ++j) {
          CUTE_UNROLL
          for (int cp = 0; cp < 2; ++cp) {
            const int col = (lane % 4) * 2 + 8 * j + cp;
            const float score0 = fmaxf(float(bf16(acc(j * 4 + cp))), 0.0f);
            const float score1 = fmaxf(float(bf16(acc(j * 4 + 2 + cp))), 0.0f);
            float sum = float(bf16(score0 * weight0)) + float(bf16(score1 * weight1));
            // Match Triton's head-reduction tree, including its BF16 tie behavior.
            sum += __shfl_xor_sync(0xffffffffu, sum, 16);
            sum += __shfl_xor_sync(0xffffffffu, sum, 8);
            sum += __shfl_xor_sync(0xffffffffu, sum, 4);
            if (head_in_warp == 0) ss.warp_sums[0][warp * BLOCK_L + col] = sum;
          }
        }
      }
      __syncthreads();
      if (tid < BLOCK_L && l0 + tid < p.width) {
        const float sum = (ss.warp_sums[0][tid] + ss.warp_sums[0][2 * BLOCK_L + tid]) +
                          (ss.warp_sums[0][BLOCK_L + tid] + ss.warp_sums[0][3 * BLOCK_L + tid]);
        out[int64_t(b) * p.out_stride + l0 + tid] = l0 + tid < lens[b] ? float(bf16(sum)) : -INFINITY;
      }
      __syncthreads();
    }
  }

  static __device__ __forceinline__ void devfunc(const Sm90Fp4GroupedIndexerParams& p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
    const int page_size = SPECIALIZED ? 64 : p.page_size;
    const int ratio = SPECIALIZED ? 2 : p.ratio;
    const int tid = threadIdx.x;
    const int warpgroup = tid / 128;
    const int wg_tid = tid % 128;
    const int b0 = blockIdx.y * p.group_size;
    const int group_rows = min(p.group_size, p.batch_size - b0);

    extern __shared__ char smem_raw[];
    SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_raw);
    Tensor sK = make_tensor(make_smem_ptr(ss.k.data()), SmemLayout{});

    const int64_t* req = reinterpret_cast<const int64_t*>(p.req);
    const int64_t* lens = reinterpret_cast<const int64_t*>(p.lens);
    const int32_t* req_to_token = reinterpret_cast<const int32_t*>(p.req_to_token);
    const uint8_t* table = reinterpret_cast<const uint8_t*>(p.table);
    const int64_t request = req[b0];

    // Uniform CTA decision: do not decode Q/K or issue MMA for a fully masked chunk.
    int max_visible = 0;
    for (int row = 0; row < group_rows; ++row) {
      max_visible = max(max_visible, static_cast<int>(min(int64_t(p.width), lens[b0 + row])));
    }
    // The zero-length case needs only contiguous output stores.
    if constexpr (TILES_PER_CTA > 1) {
      if (max_visible == 0) {
        float* out = reinterpret_cast<float*>(p.out);
        for (int i = tid; i < group_rows * BLOCK_L * TILES_PER_CTA; i += NUM_THREADS) {
          const int row = i / (BLOCK_L * TILES_PER_CTA);
          const int position = blockIdx.x * BLOCK_L * TILES_PER_CTA + i % (BLOCK_L * TILES_PER_CTA);
          if (position < p.width) out[int64_t(b0 + row) * p.out_stride + position] = -INFINITY;
        }
        return;
      }
    }
    {
      // This is a bijection of the fixed capture grid's tiles. Materialize
      // the per-CTA schedule in shared memory so shifts and masks are not live
      // across conversion/MMA. The existing q_wide reduction publishes it.
      constexpr int base_shift = TILES_PER_CTA == 8 ? 3 : TILES_PER_CTA == 4 ? 2 : TILES_PER_CTA == 2 ? 1 : 0;
      int tile_shift = base_shift;
      if constexpr (TILES_PER_CTA > 1) {
        if (max_visible < p.width) {
          const int64_t visible_tiles = int64_t(ceil_div(max_visible, BLOCK_L)) * ceil_div(p.batch_size, p.group_size);
          const int target_shift = visible_tiles >= 1024 ? 3 : visible_tiles >= 512 ? 2 : visible_tiles >= 256 ? 1 : 0;
          tile_shift = min(base_shift, target_shift);
        }
      }
      const int subtiles = 1 << tile_shift;
      const int chunk_start = blockIdx.x * BLOCK_L * subtiles;
      const auto tile_start = [&](int tile) {
        return chunk_start + ((tile >> tile_shift) * gridDim.x * subtiles + (tile & (subtiles - 1))) * BLOCK_L;
      };
      if (chunk_start >= max_visible) {
        float* out = reinterpret_cast<float*>(p.out);
        for (int i = tid; i < group_rows * BLOCK_L * TILES_PER_CTA; i += NUM_THREADS) {
          const int row = i / (BLOCK_L * TILES_PER_CTA);
          const int within_row = i % (BLOCK_L * TILES_PER_CTA);
          const int position = tile_start(within_row / BLOCK_L) + within_row % BLOCK_L;
          if (position < p.width) out[int64_t(b0 + row) * p.out_stride + position] = -INFINITY;
        }
        return;
      }
      if constexpr (TILES_PER_CTA > 1) {
        if (tid < TILES_PER_CTA) ss.tile_starts[tid] = tile_start(tid);
      }
    }

    const uint32_t* query_scales = reinterpret_cast<const uint32_t*>(p.q_scale);
    bool q_wide = false;
    for (int i = tid; i < group_rows * HEADS; i += NUM_THREADS)
      q_wide |= needs_bf16(query_scales[int64_t(b0 + i / HEADS) * p.q_scale_stride_b + i % HEADS]);
    q_wide = __syncthreads_or(q_wide);
    bool resident_valid = false;

    for (int tile = 0; tile < TILES_PER_CTA; ++tile) {
      const int l0 = TILES_PER_CTA > 1 ? ss.tile_starts[tile] : blockIdx.x * BLOCK_L;
      if (l0 >= p.width) break;
      if (l0 >= max_visible) {
        float* out = reinterpret_cast<float*>(p.out);
        for (int i = tid; i < group_rows * BLOCK_L; i += NUM_THREADS) {
          const int row = i / BLOCK_L;
          const int position = l0 + i % BLOCK_L;
          if (position < p.width) out[int64_t(b0 + row) * p.out_stride + position] = -INFINITY;
        }
        continue;
      }
      bool k_wide = false;
      if (tid < BLOCK_L) {
        const int col = tid;
        const int position = l0 + col;
        int32_t slot = 0;
        if (position < max_visible) {
          slot = req_to_token[request * p.req_stride + static_cast<int64_t>(position) * ratio] / ratio;
        }
        // The same thread consumes its slot for scale lookup. The reduction
        // barrier below publishes slots/scales before cooperative Q/K decode.
        ss.slots[col] = slot;
        const int page = slot / page_size;
        const int off = slot - page * page_size;
        uint8_t max_exponent = 1;
        uint32_t packed_scale = 0;
        CUTE_UNROLL
        for (int g = 0; g < SCALE_GROUPS; ++g) {
          const uint8_t exponent =
              l0 + col < max_visible ? table[static_cast<int64_t>(page) * p.table_stride + page_size * 64 + off * 4 + g]
                                     : 1;
          packed_scale |= uint32_t(exponent) << (8 * g);
          ss.k_exponents[g][col] = exponent;
          max_exponent = max(max_exponent, exponent);
        }
        const uint8_t common_exponent = max(static_cast<int>(max_exponent) - 12, 1);
        ss.k_scales[col] = ue8m0_to_f32(common_exponent);
        k_wide = l0 + col < max_visible && needs_bf16(packed_scale);
      }
      k_wide = __syncthreads_or(k_wide);
      if (q_wide || k_wide) {
        bf16_tile(p, ss, b0, group_rows, l0, max_visible);
        resident_valid = false;
        continue;
      }

      if constexpr (RESIDENT_Q) {
        if (!resident_valid) {
          // Keep barrier arrivals identical across inactive tail warpgroups.
          for (int round = 0; round < ceil_div(group_rows, NUM_WARPGROUPS); ++round) {
            const int row = round * NUM_WARPGROUPS + warpgroup;
            const bool active = row < group_rows;
            decode_q(p, ss, b0 + row, active ? row : 0, wg_tid, active);
          }
          __syncthreads();
          resident_valid = true;
        }
      }

      for (int word = tid; word < BLOCK_L * (HEAD_DIM / 8); word += NUM_THREADS) {
        const int col = word / (HEAD_DIM / 8);
        const int d = (word % (HEAD_DIM / 8)) * 8;
        const int g = d / SCALE_GROUP_SIZE;
        const int slot = ss.slots[col];
        const int page = slot / page_size;
        const int off = slot - page * page_size;
        const int packed_col = d / 2;
        const uint32_t packed =
            l0 + col < max_visible
                ? load_word(table + static_cast<int64_t>(page) * p.table_stride + off * 64 + packed_col)
                : 0;
        const int common_exponent = (__float_as_uint(ss.k_scales[col]) >> 23) & 0xff;
        const int exponent_delta = static_cast<int>(ss.k_exponents[g][col]) - common_exponent;
        CUTE_UNROLL
        for (int half = 0; half < 2; ++half) {
          const uint32_t expanded = scaled_e2m1x4_to_e5m2x4(packed >> (half * 16), exponent_delta);
          // SW64 preserves contiguous aligned groups of four FP8 elements.
          *reinterpret_cast<uint32_t*>(&sK(col, d + half * 4)) = expanded;
        }
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncthreads();

      const bf16* weights = reinterpret_cast<const bf16*>(p.weights);
      float* out = reinterpret_cast<float*>(p.out);
      TiledMMA mma;

      const int rounds = ceil_div(group_rows, NUM_WARPGROUPS);
      for (int round = 0; round < rounds; ++round) {
        const int row_in_group = round * NUM_WARPGROUPS + warpgroup;
        const bool active = row_in_group < group_rows;
        const int b = b0 + row_in_group;

        const int q_slot = RESIDENT_Q ? (active ? row_in_group : 0) : warpgroup;
        Tensor sQ = make_tensor(make_smem_ptr(ss.q[q_slot].data()), SmemLayout{});
        if constexpr (!RESIDENT_Q) {
          decode_q(p, ss, b, q_slot, wg_tid, active);
          __syncthreads();
        }

        Tensor acc = partition_fragment_C(mma, Shape<Int<HEADS>, Int<BLOCK_L>>{});
        if (active) {
          gemm_k128(mma, sQ, sK, acc, wg_tid);
          warpgroup_commit_batch();
          warpgroup_wait<0>();
          warpgroup_fence_operand(acc);
        }

        if (active) {
          // Lanes with the same lane % 4 own the same 16 columns. Their two
          // accumulator rows cover one contiguous 16-head slice.
          const int warp = wg_tid / 32;
          const int lane = wg_tid % 32;
          const int head_in_warp = lane / 4;
          const int head0 = warp * 16 + head_in_warp;
          const int head1 = head0 + 8;
          const float q_scale0 = ss.q_scales[q_slot][head0];
          const float q_scale1 = ss.q_scales[q_slot][head1];
          const float weight0 = static_cast<float>(weights[static_cast<int64_t>(b) * p.weight_stride_b + head0]);
          const float weight1 = static_cast<float>(weights[static_cast<int64_t>(b) * p.weight_stride_b + head1]);

          CUTE_UNROLL
          for (int j = 0; j < BLOCK_L / 8; ++j) {
            CUTE_UNROLL
            for (int cp = 0; cp < 2; ++cp) {
              const int col = (lane % 4) * 2 + 8 * j + cp;
              const float k_scale = ss.k_scales[col];
              const float score0 = fmaxf(static_cast<float>(bf16(acc(j * 4 + cp) * q_scale0 * k_scale)), 0.0f);
              const float score1 = fmaxf(static_cast<float>(bf16(acc(j * 4 + 2 + cp) * q_scale1 * k_scale)), 0.0f);
              float sum = float(bf16(score0 * weight0)) + float(bf16(score1 * weight1));
              // Match Triton's head-reduction tree, including its BF16 tie behavior.
              sum += __shfl_xor_sync(0xffffffffu, sum, 16);
              sum += __shfl_xor_sync(0xffffffffu, sum, 8);
              sum += __shfl_xor_sync(0xffffffffu, sum, 4);
              if (head_in_warp == 0) ss.warp_sums[warpgroup][warp * BLOCK_L + col] = sum;
            }
          }
        }
        if constexpr (RESIDENT_Q) {
          asm volatile("barrier.sync %0, 128;" ::"r"(warpgroup + 1) : "memory");
        } else {
          __syncthreads();
        }

        if (active && wg_tid < BLOCK_L) {
          const float sum = (ss.warp_sums[warpgroup][wg_tid] + ss.warp_sums[warpgroup][2 * BLOCK_L + wg_tid]) +
                            (ss.warp_sums[warpgroup][BLOCK_L + wg_tid] + ss.warp_sums[warpgroup][3 * BLOCK_L + wg_tid]);
          const int position = l0 + wg_tid;
          if (position < p.width) {
            const bool valid = position < lens[b];
            out[static_cast<int64_t>(b) * p.out_stride + position] = valid ? static_cast<float>(bf16(sum)) : -INFINITY;
          }
        }
        if constexpr (RESIDENT_Q) {
          asm volatile("barrier.sync %0, 128;" ::"r"(warpgroup + 1) : "memory");
        } else {
          __syncthreads();
        }
      }
      if constexpr (RESIDENT_Q) {
        // Protect shared K from the next tile's cooperative writers.
        __syncthreads();
      }
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
      host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      return true;
    }();
    (void)attr_set;
    dim3 grid(ceil_div(p.width, BLOCK_L * TILES_PER_CTA), ceil_div(p.batch_size, p.group_size), 1);
    host::LaunchKernel(grid, NUM_THREADS, p.stream, smem_size)(kernel, p);
  }
};

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 2)
    fp4_grouped_indexer_kernel(__grid_constant__ const Sm90Fp4GroupedIndexerParams params) {
  Kernel::devfunc(params);
}

template <bool ALIGNED, bool SPECIALIZED>
inline void dispatch_layout(const Sm90Fp4GroupedIndexerParams& p, int tiles_per_cta) {
  switch (tiles_per_cta) {
    case 1:
      Sm90Fp4GroupedIndexerKernel<1, 4, ALIGNED, SPECIALIZED>::run(p);
      break;
    case 2:
      Sm90Fp4GroupedIndexerKernel<2, 6, ALIGNED, SPECIALIZED>::run(p);
      break;
    case 4:
      Sm90Fp4GroupedIndexerKernel<4, 6, ALIGNED, SPECIALIZED>::run(p);
      break;
    case 8:
      Sm90Fp4GroupedIndexerKernel<8, 6, ALIGNED, SPECIALIZED>::run(p);
      break;
    default:
      host::Panic("Unsupported tiles_per_cta: ", tiles_per_cta);
  }
}

inline void run_sm90_fp4_grouped_indexer_tpc(const Sm90Fp4GroupedIndexerParams& p, int tpc) {
  if (p.width == 0 || p.batch_size == 0) return;
  if (tpc == 0) {
    const int tiles_per_group = ceil_div(p.width, 64);
    const int64_t total_tiles = int64_t(tiles_per_group) * ceil_div(p.batch_size, p.group_size);
    tpc = 1;
    if (p.group_size <= 6 && tiles_per_group >= 2 && total_tiles >= 256) {
      tpc = tiles_per_group >= 8 && total_tiles >= 1024 ? 8 : tiles_per_group >= 4 && total_tiles >= 512 ? 4 : 2;
    }
  }
  host::RuntimeCheck(tpc == 1 || p.group_size <= 6, "Multi-tile kernel supports at most 6 query rows");
  const uintptr_t alignment = reinterpret_cast<uintptr_t>(p.q) | reinterpret_cast<uintptr_t>(p.table) |
                              uintptr_t(p.q_stride_b) | uintptr_t(p.q_stride_h) | uintptr_t(p.table_stride);
  if ((alignment & 3) != 0) {
    dispatch_layout<false, false>(p, tpc);
  } else if (p.page_size == 64 && p.ratio == 2) {
    dispatch_layout<true, true>(p, tpc);
  } else {
    dispatch_layout<true, false>(p, tpc);
  }
}

inline void run_sm90_fp4_grouped_indexer(const Sm90Fp4GroupedIndexerParams& p) {
  run_sm90_fp4_grouped_indexer_tpc(p, 0);
}

}  // namespace fp4_grouped_indexer_sm90
}  // namespace sglang
