/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#pragma once

#include <limits>

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/scalar_type.hpp>

#include "kernel.h"
#include "kernel_direct.h"
#include "marlin_tma_utils.h"
#include "marlin_template.h"
#include "marlin_direct_template.h"

namespace device::marlin_moe {

namespace detail {

}  // namespace detail

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};
__global__ void MarlinDirectDefault(MARLIN_DIRECT_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);
using MarlinDirectFuncPtr = void (*)(MARLIN_DIRECT_KERNEL_PARAMS);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

template <int moe_block_size>
__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr,
    int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    const int32_t* __restrict__ sorted_token_ids_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_past_padded_ptr,
    int size_m,
    int size_k,
    int top_k) {};

#else

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
template <int moe_block_size>
__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr,
    int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    const int32_t* __restrict__ sorted_token_ids_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_past_padded_ptr,
    int size_m,
    int size_k,
    int top_k) {
  int num_tokens_past_padded = num_tokens_past_padded_ptr[0];
  int num_moe_blocks = div_ceil(num_tokens_past_padded, moe_block_size);
  int32_t block_sorted_ids[moe_block_size];
  int block_num_valid_tokens = 0;
  int64_t old_expert_id = 0;
  int64_t expert_id = 0;
  int row_stride = size_k * sizeof(half) / 16;

  auto read_moe_block_data = [&](int block_id) {
    block_num_valid_tokens = moe_block_size;
    int4* tmp_block_sorted_ids = reinterpret_cast<int4*>(block_sorted_ids);
    for (int i = 0; i < moe_block_size / 4; i++) {
      tmp_block_sorted_ids[i] = ((int4*)sorted_token_ids_ptr)[block_id * moe_block_size / 4 + i];
    }
    for (int i = 0; i < moe_block_size; i++) {
      if (block_sorted_ids[i] >= size_m * top_k) {
        block_num_valid_tokens = i;
        break;
      };
    }
  };

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int in_offset = (row / top_k) * row_stride;
    int out_offset = row * row_stride;

    half const* a_row_half = reinterpret_cast<half const*>(a_int4_ptr + in_offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + out_offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      auto cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        auto cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int index = blockIdx.x; index < num_moe_blocks; index += gridDim.x) {
    old_expert_id = expert_id;
    int tmp_expert_id = expert_ids_ptr[index];
    if (tmp_expert_id == -1) continue;
    expert_id = tmp_expert_id;
    perm_int_ptr += (expert_id - old_expert_id) * size_k;
    read_moe_block_data(index);

    for (int i = 0; i < block_num_valid_tokens; i++)
      permute_row(block_sorted_ids[i]);
  }
}

__global__ void build_direct_sorted_token_ids_kernel(
    int32_t* __restrict__ sorted_token_ids_ptr,
    int size_m,
    int num_tokens_post_padded) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_tokens_post_padded) {
    return;
  }
  sorted_token_ids_ptr[idx] = idx < size_m ? idx : size_m;
}

template <typename scalar_t>
__global__ void silu_and_mul_kernel(
    const scalar_t* __restrict__ input_ptr,
    scalar_t* __restrict__ output_ptr,
    int size_m,
    int size_n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = size_m * size_n;
  if (idx >= total) {
    return;
  }
  int row = idx / size_n;
  int col = idx - row * size_n;
  int gate_offset = row * (2 * size_n) + col;
  int up_offset = gate_offset + size_n;
  float gate = static_cast<float>(input_ptr[gate_offset]);
  float up = static_cast<float>(input_ptr[up_offset]);
  float silu = gate / (1.0f + expf(-gate));
  output_ptr[idx] = static_cast<scalar_t>(silu * up);
}

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

int get_scales_cache_size(
    thread_config_t const& th_config,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups = tb_groups * pipe_stages * 2;  // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);             // We load at least 32 scale groups
    return load_groups * tb_n * 2;
  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * pipe_stages;
  }
}

int get_kernel_cache_size(
    thread_config_t const& th_config,
    bool m_block_size_8,
    int thread_m_blocks,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    int has_zp,
    int is_zp_float) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;

  // shm size for block_sorted_ids/rd_block_sorted_ids/block_topk_weights
  // both of them requires tb_m * 4 bytes (tb_m * int32 or tb_m * float32)
  int sh_block_meta_size = tb_m * 4;
  int sh_a_size = pipe_stages * (tb_m * tb_k) * 2;
  int sh_b_size = pipe_stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits, group_size, has_act_order, is_k_full);
  int sh_g_idx_size = has_act_order && !is_k_full ? pipe_stages * tb_k / 4 : 0;
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size + sh_block_meta_size;

  return total_size;
}

bool is_valid_config(
    thread_config_t const& th_config,
    bool m_block_size_8,
    int thread_m_blocks,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    int has_zp,
    int is_zp_float,
    int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 || th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  // Check that pipeline fits into cache
  int cache_size = get_kernel_cache_size(
      th_config,
      m_block_size_8,
      thread_m_blocks,
      prob_m,
      prob_n,
      prob_k,
      num_bits,
      group_size,
      has_act_order,
      is_k_full,
      has_zp,
      is_zp_float);
  return cache_size + 512 <= max_shared_mem;
}

#define _GET_IF(                                                                                                       \
    W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8, GROUP_BLOCKS, NUM_THREADS, IS_ZP_FLOAT) \
  else if (                                                                                                            \
      q_type == W_TYPE && thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS &&                  \
      thread_k_blocks == THREAD_K_BLOCKS && m_block_size_8 == M_BLOCK_SIZE_8 && group_blocks == GROUP_BLOCKS &&        \
      num_threads == NUM_THREADS && is_zp_float == IS_ZP_FLOAT) {                                                      \
    constexpr auto S_TYPE = W_TYPE == host::kFE2M1f                                                                    \
                                ? (GROUP_BLOCKS == 1 ? host::kFE4M3fn : host::kFE8M0fnu)                               \
                                : (std::is_same<scalar_t, half>::value ? host::kFloat16 : host::kBFloat16);            \
    kernel = Marlin<                                                                                                   \
        scalar_t,                                                                                                      \
        W_TYPE.id(),                                                                                                   \
        S_TYPE.id(),                                                                                                   \
        NUM_THREADS,                                                                                                   \
        THREAD_M_BLOCKS,                                                                                               \
        THREAD_N_BLOCKS,                                                                                               \
        THREAD_K_BLOCKS,                                                                                               \
        M_BLOCK_SIZE_8,                                                                                                \
        pipe_stages,                                                                                                   \
        GROUP_BLOCKS,                                                                                                  \
        IS_ZP_FLOAT>;                                                                                                  \
  }

// COMMON: cases for (group_blocks in [-1, 2, 4, 8] and is_zp_float == false)
//         this is the most common cases
// BIGGROUP: cases for big group size (group_blocks in [-1, 8])
// FZP: cases for float-zero-point (is_zp_float = true)
// ACT: cases for act order case (group_blocks == 0)
// NVFP4: cases for nvfp4(e2m1) (group_blocks == 1)
// MXFP4: cases for mxfp4(e2m1) (group_blocks == 2)
#define COMMON_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)       \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, -1, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 2, NUM_THREADS, false)   \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 4, NUM_THREADS, false)   \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 8, NUM_THREADS, false)   \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define COMMON_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
                                                                        \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
                                                                        \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define COMMON_GET_IF(W_TYPE)            \
  COMMON_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  COMMON_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  COMMON_GET_IF_M234(W_TYPE, 16, 4, 256) \
  COMMON_GET_IF_M234(W_TYPE, 8, 4, 128)

#define BIGGROUP_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, -1, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 8, NUM_THREADS, false)   \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define BIGGROUP_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)   \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define BIGGROUP_GET_IF(W_TYPE)            \
  BIGGROUP_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  BIGGROUP_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  BIGGROUP_GET_IF_M234(W_TYPE, 16, 4, 256) \
  BIGGROUP_GET_IF_M234(W_TYPE, 8, 4, 128)

#define NVFP4_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)      \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false)

#define NVFP4_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false)

#define NVFP4_GET_IF(W_TYPE)            \
  NVFP4_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  NVFP4_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  NVFP4_GET_IF_M234(W_TYPE, 16, 4, 256) \
  NVFP4_GET_IF_M234(W_TYPE, 8, 4, 128)

#define MXFP4_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)      \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 2, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)

#define MXFP4_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)

#define MXFP4_GET_IF(W_TYPE)            \
  MXFP4_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  MXFP4_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  MXFP4_GET_IF_M234(W_TYPE, 16, 4, 256) \
  MXFP4_GET_IF_M234(W_TYPE, 8, 4, 128)

// We currently have 4-bit models only with group_blocks == 4
#define FZP_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)       \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 4, NUM_THREADS, true) \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true)

#define FZP_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)      \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true)

#define FZP_GET_IF(W_TYPE)            \
  FZP_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  FZP_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  FZP_GET_IF_M234(W_TYPE, 16, 4, 256) \
  FZP_GET_IF_M234(W_TYPE, 8, 4, 128)

// We currently have 4-bit models only with group_blocks == 4
#define ACT_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)        \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 0, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false)

#define ACT_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)       \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false)

#define ACT_GET_IF(W_TYPE)            \
  ACT_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  ACT_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  ACT_GET_IF_M234(W_TYPE, 16, 4, 256) \
  ACT_GET_IF_M234(W_TYPE, 8, 4, 128)

#define _GET_DIRECT_IF(                                                                                                \
    W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8, GROUP_BLOCKS, NUM_THREADS, IS_ZP_FLOAT) \
  else if (                                                                                                             \
      q_type == W_TYPE && thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS &&                   \
      thread_k_blocks == THREAD_K_BLOCKS && m_block_size_8 == M_BLOCK_SIZE_8 && group_blocks == GROUP_BLOCKS &&         \
      num_threads == NUM_THREADS && is_zp_float == IS_ZP_FLOAT) {                                                       \
    constexpr auto S_TYPE = W_TYPE == host::kFE2M1f                                                                     \
                                ? (GROUP_BLOCKS == 1 ? host::kFE4M3fn : host::kFE8M0fnu)                                \
                                : (std::is_same<scalar_t, half>::value ? host::kFloat16 : host::kBFloat16);             \
    kernel = device::marlin_moe_direct::MarlinDirect<                                                                    \
        scalar_t,                                                                                                        \
        W_TYPE.id(),                                                                                                     \
        S_TYPE.id(),                                                                                                     \
        NUM_THREADS,                                                                                                     \
        THREAD_M_BLOCKS,                                                                                                 \
        THREAD_N_BLOCKS,                                                                                                 \
        THREAD_K_BLOCKS,                                                                                                 \
        M_BLOCK_SIZE_8,                                                                                                  \
        pipe_stages,                                                                                                     \
        GROUP_BLOCKS,                                                                                                    \
        IS_ZP_FLOAT>;                                                                                                    \
  }

#define COMMON_GET_DIRECT_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)       \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, -1, NUM_THREADS, false)  \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 2, NUM_THREADS, false)   \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 4, NUM_THREADS, false)   \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 8, NUM_THREADS, false)   \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define COMMON_GET_DIRECT_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \
  _GET_DIRECT_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_DIRECT_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_DIRECT_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_DIRECT_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
  _GET_DIRECT_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_DIRECT_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_DIRECT_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_DIRECT_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
  _GET_DIRECT_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_DIRECT_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_DIRECT_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_DIRECT_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define COMMON_GET_DIRECT_IF(W_TYPE)            \
  COMMON_GET_DIRECT_IF_M1(W_TYPE, 8, 8, 256)    \
  COMMON_GET_DIRECT_IF_M1(W_TYPE, 8, 4, 128)    \
  COMMON_GET_DIRECT_IF_M234(W_TYPE, 16, 4, 256) \
  COMMON_GET_DIRECT_IF_M234(W_TYPE, 8, 4, 128)

#define BIGGROUP_GET_DIRECT_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, -1, NUM_THREADS, false)  \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 8, NUM_THREADS, false)   \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define BIGGROUP_GET_DIRECT_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)   \
  _GET_DIRECT_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_DIRECT_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
  _GET_DIRECT_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_DIRECT_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
  _GET_DIRECT_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_DIRECT_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define BIGGROUP_GET_DIRECT_IF(W_TYPE)            \
  BIGGROUP_GET_DIRECT_IF_M1(W_TYPE, 8, 8, 256)    \
  BIGGROUP_GET_DIRECT_IF_M1(W_TYPE, 8, 4, 128)    \
  BIGGROUP_GET_DIRECT_IF_M234(W_TYPE, 16, 4, 256) \
  BIGGROUP_GET_DIRECT_IF_M234(W_TYPE, 8, 4, 128)

#define NVFP4_GET_DIRECT_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)      \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 1, NUM_THREADS, false) \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false)

#define NVFP4_GET_DIRECT_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \
  _GET_DIRECT_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false) \
  _GET_DIRECT_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false) \
  _GET_DIRECT_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false)

#define NVFP4_GET_DIRECT_IF(W_TYPE)            \
  NVFP4_GET_DIRECT_IF_M1(W_TYPE, 8, 8, 256)    \
  NVFP4_GET_DIRECT_IF_M1(W_TYPE, 8, 4, 128)    \
  NVFP4_GET_DIRECT_IF_M234(W_TYPE, 16, 4, 256) \
  NVFP4_GET_DIRECT_IF_M234(W_TYPE, 8, 4, 128)

#define MXFP4_GET_DIRECT_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)      \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 2, NUM_THREADS, false) \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)

#define MXFP4_GET_DIRECT_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \
  _GET_DIRECT_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \
  _GET_DIRECT_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \
  _GET_DIRECT_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)

#define MXFP4_GET_DIRECT_IF(W_TYPE)            \
  MXFP4_GET_DIRECT_IF_M1(W_TYPE, 8, 8, 256)    \
  MXFP4_GET_DIRECT_IF_M1(W_TYPE, 8, 4, 128)    \
  MXFP4_GET_DIRECT_IF_M234(W_TYPE, 16, 4, 256) \
  MXFP4_GET_DIRECT_IF_M234(W_TYPE, 8, 4, 128)

#define FZP_GET_DIRECT_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)       \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 4, NUM_THREADS, true) \
  _GET_DIRECT_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true)

#define FZP_GET_DIRECT_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)      \
  _GET_DIRECT_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \
  _GET_DIRECT_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \
  _GET_DIRECT_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true)

#define FZP_GET_DIRECT_IF(W_TYPE)            \
  FZP_GET_DIRECT_IF_M1(W_TYPE, 8, 8, 256)    \
  FZP_GET_DIRECT_IF_M1(W_TYPE, 8, 4, 128)    \
  FZP_GET_DIRECT_IF_M234(W_TYPE, 16, 4, 256) \
  FZP_GET_DIRECT_IF_M234(W_TYPE, 8, 4, 128)

template <typename scalar_t>
MarlinFuncPtr get_marlin_kernel(
    const host::ScalarType q_type,
    int thread_m_blocks,
    int thread_n_blocks,
    int thread_k_blocks,
    bool m_block_size_8,
    bool has_act_order,
    bool has_zp,
    int group_blocks,
    int num_threads,
    bool is_zp_float) {
  int num_bits = q_type.size_bits();
  auto kernel = MarlinDefault;
  if (false) {
  }

  COMMON_GET_IF(host::kU4)
  COMMON_GET_IF(host::kU4B8)
  COMMON_GET_IF(host::kU8B128)

  NVFP4_GET_IF(host::kFE2M1f)

  BIGGROUP_GET_IF(host::kFE4M3fn)

  ACT_GET_IF(host::kU4B8)
  ACT_GET_IF(host::kU8B128)
  if (std::is_same<scalar_t, nv_bfloat16>::value) {
    if (false) {
    }
    MXFP4_GET_IF(host::kFE2M1f)
  }

  return kernel;
}

template <typename scalar_t>
MarlinDirectFuncPtr get_marlin_direct_kernel(
    const host::ScalarType q_type,
    int thread_m_blocks,
    int thread_n_blocks,
    int thread_k_blocks,
    bool m_block_size_8,
    bool has_zp,
    int group_blocks,
    int num_threads,
    bool is_zp_float) {
  auto kernel = MarlinDirectDefault;
  if (false) {
  }

  COMMON_GET_DIRECT_IF(host::kU4)
  COMMON_GET_DIRECT_IF(host::kU4B8)
  COMMON_GET_DIRECT_IF(host::kU8B128)

  NVFP4_GET_DIRECT_IF(host::kFE2M1f)

  BIGGROUP_GET_DIRECT_IF(host::kFE4M3fn)

  if (std::is_same<scalar_t, nv_bfloat16>::value) {
    if (false) {
    }
    MXFP4_GET_DIRECT_IF(host::kFE2M1f)
  }

  // float zero-point variants
  FZP_GET_DIRECT_IF(host::kU4)
  FZP_GET_DIRECT_IF(host::kU8)

  return kernel;
}

template <typename scalar_t>
exec_config_t determine_exec_config(
    const host::ScalarType& q_type,
    int prob_m,
    int prob_n,
    int prob_k,
    int thread_m_blocks,
    bool m_block_size_8,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    bool has_zp,
    bool is_zp_float,
    int max_shared_mem) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1 ? large_batch_thread_configs : small_batch_thread_configs;
  int thread_configs_size = thread_m_blocks > 1 ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
                                                : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  int count = 0;
  constexpr int device_max_reg_size = 255 * 1024;
  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(
            th_config,
            m_block_size_8,
            thread_m_blocks,
            prob_m,
            prob_n,
            prob_k,
            num_bits,
            group_size,
            has_act_order,
            is_k_full,
            has_zp,
            is_zp_float,
            max_shared_mem)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config,
        m_block_size_8,
        thread_m_blocks,
        prob_m,
        prob_n,
        prob_k,
        num_bits,
        group_size,
        has_act_order,
        is_k_full,
        has_zp,
        is_zp_float);

    int group_blocks = 0;
    if (!has_act_order) {
      group_blocks = group_size == -1 ? -1 : (group_size / 16);
    }

    auto kernel = get_marlin_kernel<scalar_t>(
        q_type,
        thread_m_blocks,
        th_config.thread_n / 16,
        th_config.thread_k / 16,
        m_block_size_8,
        has_act_order,
        has_zp,
        group_blocks,
        th_config.num_threads,
        is_zp_float);

    if (kernel == MarlinDefault) continue;

    if (thread_m_blocks > 1) {
      exec_cfg = {1, th_config};
      break;
    } else {
      cudaFuncAttributes attr;
      cudaFuncGetAttributes(&attr, kernel);
      int reg_size = max(attr.numRegs, 1) * th_config.num_threads * 4;
      int allow_count = min(device_max_reg_size / reg_size, max_shared_mem / (cache_size + 1024));
      allow_count = max(min(allow_count, 4), 1);
      if (allow_count > count) {
        count = allow_count;
        exec_cfg = {count, th_config};
      };
    }
  }

  return exec_cfg;
}

// Like determine_exec_config but queries the *direct* kernel for register
// usage so that blocks_per_sm is accurate for the direct path.
template <typename scalar_t>
exec_config_t determine_direct_exec_config(
    const host::ScalarType& q_type,
    int prob_m,
    int prob_n,
    int prob_k,
    int thread_m_blocks,
    bool m_block_size_8,
    int num_bits,
    int group_size,
    bool is_k_full,
    bool has_zp,
    bool is_zp_float,
    int max_shared_mem) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1 ? large_batch_thread_configs : small_batch_thread_configs;
  int thread_configs_size = thread_m_blocks > 1 ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
                                                : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  int count = 0;
  constexpr int device_max_reg_size = 255 * 1024;
  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(
            th_config,
            m_block_size_8,
            thread_m_blocks,
            prob_m,
            prob_n,
            prob_k,
            num_bits,
            group_size,
            false,
            is_k_full,
            has_zp,
            is_zp_float,
            max_shared_mem)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config,
        m_block_size_8,
        thread_m_blocks,
        prob_m,
        prob_n,
        prob_k,
        num_bits,
        group_size,
        false,
        is_k_full,
        has_zp,
        is_zp_float);

    int group_blocks = group_size == -1 ? -1 : (group_size / 16);

    auto kernel = get_marlin_direct_kernel<scalar_t>(
        q_type,
        thread_m_blocks,
        th_config.thread_n / 16,
        th_config.thread_k / 16,
        m_block_size_8,
        has_zp,
        group_blocks,
        th_config.num_threads,
        is_zp_float);

    if (kernel == MarlinDirectDefault) continue;

    if (thread_m_blocks > 1) {
      exec_cfg = {1, th_config};
      break;
    } else {
      cudaFuncAttributes attr;
      cudaFuncGetAttributes(&attr, kernel);
      int reg_size = max(attr.numRegs, 1) * th_config.num_threads * 4;
      int allow_count = min(device_max_reg_size / reg_size, max_shared_mem / (cache_size + 1024));
      allow_count = max(min(allow_count, 4), 1);
      if (allow_count > count) {
        count = allow_count;
        exec_cfg = {count, th_config};
      };
    }
  }

  return exec_cfg;
}


inline int direct_config_split_block_excess(
    thread_config_t const& th_config,
    int blocks_per_sm,
    int logical_m,
    int prob_n,
    int moe_block_size,
    int sms) {
  if (blocks_per_sm <= 1 || logical_m <= 0) {
    return 0;
  }
  const int thread_n_blocks = th_config.thread_n / 16;
  const int n_tiles = prob_n / 16 / thread_n_blocks;
  const int parallel = div_ceil(logical_m, moe_block_size);
  const int max_no_split_blocks = max(parallel * n_tiles, 1);
  return max(sms * blocks_per_sm - max_no_split_blocks, 0);
}

inline bool direct_config_has_high_lock_risk(
    thread_config_t const& th_config,
    int blocks_per_sm,
    int logical_m,
    int prob_n,
    int moe_block_size,
    int sms) {
  return direct_config_split_block_excess(th_config, blocks_per_sm, logical_m, prob_n, moe_block_size, sms) > 0;
}

inline int pick_direct_blocks_per_sm_with_lock_avoidance(
    thread_config_t const& th_config,
    int allow_count,
    int logical_m,
    int prob_n,
    int moe_block_size,
    int sms) {
  int best_chosen = 1;
  int best_split_excess = std::numeric_limits<int>::max();
  for (int chosen = 1; chosen <= allow_count; ++chosen) {
    const int split_excess =
        direct_config_split_block_excess(th_config, chosen, logical_m, prob_n, moe_block_size, sms);
    const bool better_zero_split = split_excess == 0 && (best_split_excess != 0 || chosen > best_chosen);
    const bool better_split_case = split_excess > 0 && best_split_excess > 0 && chosen < best_chosen;
    if (split_excess < best_split_excess || better_zero_split || better_split_case) {
      best_split_excess = split_excess;
      best_chosen = chosen;
    }
  }
  return best_chosen;
}

inline int limit_direct_blocks_per_sm_for_lock_avoidance(
    int blocks_per_sm,
    int logical_m,
    int prob_n,
    int thread_n,
    int moe_block_size,
    int sms) {
  if (blocks_per_sm <= 1 || logical_m > 32) {
    return blocks_per_sm;
  }

  const int parallel = div_ceil(logical_m, moe_block_size);
  const int n_tiles = prob_n / thread_n;
  const int max_no_split_blocks = max(parallel * n_tiles, 1);
  int limited = div_ceil(max_no_split_blocks, sms);
  limited = max(limited, 1);
  return min(blocks_per_sm, limited);
}

template <typename scalar_t>
bool try_determine_small_m_direct_exec_config(
    const host::ScalarType& q_type,
    int logical_m,
    int prob_n,
    int prob_k,
    bool is_w13_stage,
    int thread_m_blocks,
    bool m_block_size_8,
    int num_bits,
    int group_size,
    bool is_k_full,
    bool has_zp,
    bool is_zp_float,
    int max_shared_mem,
    int sms,
    exec_config_t* exec_cfg_out) {
  if (thread_m_blocks != 1 || logical_m > 32) {
    return false;
  }

  const thread_config_t preferred_configs[2] = {
      is_w13_stage ? thread_config_t{128, 128, 256} : thread_config_t{64, 128, 128},
      is_w13_stage ? thread_config_t{64, 128, 128} : thread_config_t{128, 128, 256},
  };
  const int group_blocks = group_size == -1 ? -1 : (group_size / 16);
  constexpr int device_max_reg_size = 255 * 1024;
  exec_config_t best_cfg = exec_config_t{0, thread_config_t{-1, -1, -1}};
  int best_split_excess = std::numeric_limits<int>::max();
  bool found_cfg = false;
  const thread_config_t* candidate_configs = preferred_configs;
  int candidate_count = sizeof(preferred_configs) / sizeof(thread_config_t);
  if (logical_m > 16) {
    candidate_configs = small_batch_thread_configs;
    candidate_count = sizeof(small_batch_thread_configs) / sizeof(thread_config_t);
  }

  for (int cfg_idx = 0; cfg_idx < candidate_count; ++cfg_idx) {
    const auto& th_config = candidate_configs[cfg_idx];
    if (!is_valid_config(
            th_config,
            m_block_size_8,
            thread_m_blocks,
            logical_m,
            prob_n,
            prob_k,
            num_bits,
            group_size,
            false,
            is_k_full,
            has_zp,
            is_zp_float,
            max_shared_mem)) {
      continue;
    }

    auto kernel = get_marlin_direct_kernel<scalar_t>(
        q_type,
        thread_m_blocks,
        th_config.thread_n / 16,
        th_config.thread_k / 16,
        m_block_size_8,
        has_zp,
        group_blocks,
        th_config.num_threads,
        is_zp_float);
    if (kernel == MarlinDirectDefault) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config,
        m_block_size_8,
        thread_m_blocks,
        logical_m,
        prob_n,
        prob_k,
        num_bits,
        group_size,
        false,
        is_k_full,
        has_zp,
        is_zp_float);
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);
    int reg_size = max(attr.numRegs, 1) * th_config.num_threads * 4;
    int allow_count = min(device_max_reg_size / reg_size, max_shared_mem / (cache_size + 1024));
    allow_count = max(min(allow_count, 4), 1);
    int chosen_count = pick_direct_blocks_per_sm_with_lock_avoidance(
        th_config,
        allow_count,
        logical_m,
        prob_n,
        m_block_size_8 ? 8 : 16,
        sms);
    int split_excess = direct_config_split_block_excess(
        th_config,
        chosen_count,
        logical_m,
        prob_n,
        m_block_size_8 ? 8 : 16,
        sms);

    if (!found_cfg || split_excess < best_split_excess ||
        (split_excess == best_split_excess && chosen_count > best_cfg.blocks_per_sm)) {
      best_cfg = {chosen_count, th_config};
      best_split_excess = split_excess;
      found_cfg = true;
      if (split_excess == 0 && logical_m <= 16) {
        break;
      }
    }
  }

  if (found_cfg) {
    *exec_cfg_out = best_cfg;
    return true;
  }
  return false;
}
template <typename scalar_t>
void marlin_mm(
    const void* A,
    const void* B,
    void* C,
    void* C_tmp,
    void* b_bias,
    void* s,
    void* s2,
    void* zp,
    void* g_idx,
    void* perm,
    void* a_tmp,
    void* sorted_token_ids,
    void* expert_ids,
    void* num_tokens_past_padded,
    void* topk_weights,
    int moe_block_size,
    int top_k,
    bool mul_topk_weights,
    bool is_ep,
    int prob_m,
    int prob_n,
    int prob_k,
    void* workspace,
    host::ScalarType const& q_type,
    bool has_bias,
    bool has_act_order,
    bool is_k_full,
    bool has_zp,
    int num_groups,
    int group_size,
    int dev,
    cudaStream_t stream,
    int thread_k,
    int thread_n,
    int sms,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {
  int thread_m_blocks = div_ceil(moe_block_size, 16);
  bool m_block_size_8 = moe_block_size == 8;

  if (has_zp) {
    host::RuntimeCheck(
        q_type == host::kU4 || q_type == host::kU8, "q_type must be u4 or u8 when has_zp = True. Got = ", q_type.str());
  } else {
    host::RuntimeCheck(
        q_type == host::kU4B8 || q_type == host::kU8B128 || q_type == host::kFE4M3fn || q_type == host::kFE2M1f,
        "q_type must be uint4b8, uint8b128, float8_e4m3fn or float4_e2m1f when "
        "has_zp = False. Got = ",
        q_type.str());
  }

  host::RuntimeCheck(
      prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m, ", ", prob_n, ", ", prob_k, "]");

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      host::RuntimeCheck(group_size != -1);
      group_blocks = group_size / 16;
      host::RuntimeCheck(
          prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);
    } else {
      host::RuntimeCheck(group_size == 0);
      group_blocks = 0;
    }
  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      host::RuntimeCheck(
          prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);
    }
  }

  int num_bits = q_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;
  const int4* bias_ptr = (const int4*)b_bias;
  const int4* s_ptr = (const int4*)s;
  const uint16_t* s2_ptr = (const uint16_t*)s2;
  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;
  const int32_t* sorted_token_ids_ptr = (const int32_t*)sorted_token_ids;
  const int32_t* expert_ids_ptr = (const int32_t*)expert_ids;
  const int32_t* num_tokens_past_padded_ptr = (const int32_t*)num_tokens_past_padded;
  const float* topk_weights_ptr = (const float*)topk_weights;
  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    auto perm_kernel = permute_cols_kernel<8>;
    if (moe_block_size == 8) {
    } else if (moe_block_size == 16)
      perm_kernel = permute_cols_kernel<16>;
    else if (moe_block_size == 32)
      perm_kernel = permute_cols_kernel<32>;
    else if (moe_block_size == 48)
      perm_kernel = permute_cols_kernel<48>;
    else if (moe_block_size == 64)
      perm_kernel = permute_cols_kernel<64>;
    else
      host::Panic("unsupported moe_block_size ", moe_block_size);

    // clang-format off
    perm_kernel<<<sms, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, sorted_token_ids_ptr, expert_ids_ptr,
        num_tokens_past_padded_ptr, prob_m, prob_k, top_k);
    // clang-format on
    A_ptr = a_tmp_ptr;
    prob_m = prob_m * top_k;
    top_k = 1;

    // If we have a full K, then we can run the non-act-order version of Marlin
    // (since the weight rows are reordered by increasing group ids, and by
    // having a full K, we have full original groups)
    if (is_k_full) has_act_order = false;
  }

  int max_shared_mem = 0;
  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  host::RuntimeCheck(max_shared_mem > 0);

  // Set thread config
  exec_config_t exec_cfg;
  thread_config_t thread_tfg;
  if (thread_k != -1 && thread_n != -1) {
    thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
    exec_cfg = exec_config_t{1, thread_tfg};
    host::RuntimeCheck(prob_n % thread_n == 0, "prob_n = ", prob_n, " is not divisible by thread_n = ", thread_n);
    host::RuntimeCheck(prob_k % thread_k == 0, "prob_k = ", prob_k, " is not divisible by thread_k = ", thread_k);
  } else {
    // Auto config
    exec_cfg = determine_exec_config<scalar_t>(
        q_type,
        prob_m,
        prob_n,
        prob_k,
        thread_m_blocks,
        m_block_size_8,
        num_bits,
        group_size,
        has_act_order,
        is_k_full,
        has_zp,
        is_zp_float,
        max_shared_mem);
    thread_tfg = exec_cfg.tb_cfg;
  }

  int num_threads = thread_tfg.num_threads;
  thread_k = thread_tfg.thread_k;
  thread_n = thread_tfg.thread_n;
  int blocks = sms * exec_cfg.blocks_per_sm;
  if (exec_cfg.blocks_per_sm > 1) max_shared_mem = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;

  host::RuntimeCheck(
      is_valid_config(
          thread_tfg,
          m_block_size_8,
          thread_m_blocks,
          prob_m,
          prob_n,
          prob_k,
          num_bits,
          group_size,
          has_act_order,
          is_k_full,
          has_zp,
          is_zp_float,
          max_shared_mem),
      "Invalid thread config: thread_m_blocks = ",
      thread_m_blocks,
      ", thread_k = ",
      thread_tfg.thread_k,
      ", thread_n = ",
      thread_tfg.thread_n,
      ", num_threads = ",
      thread_tfg.num_threads,
      " for MKN = [",
      prob_m,
      ", ",
      prob_k,
      ", ",
      prob_n,
      "] and num_bits = ",
      num_bits,
      ", group_size = ",
      group_size,
      ", has_act_order = ",
      has_act_order,
      ", is_k_full = ",
      is_k_full,
      ", has_zp = ",
      has_zp,
      ", is_zp_float = ",
      is_zp_float,
      ", max_shared_mem = ",
      max_shared_mem);

  auto kernel = get_marlin_kernel<scalar_t>(
      q_type,
      thread_m_blocks,
      thread_n_blocks,
      thread_k_blocks,
      m_block_size_8,
      has_act_order,
      has_zp,
      group_blocks,
      num_threads,
      is_zp_float);

  if (kernel == MarlinDefault) {
    host::Panic(
        "Unsupported shapes: MNK = [",
        prob_m,
        ", ",
        prob_n,
        ", ",
        prob_k,
        "]",
        ", has_act_order = ",
        has_act_order,
        ", num_groups = ",
        num_groups,
        ", group_size = ",
        group_size,
        ", thread_m_blocks = ",
        thread_m_blocks,
        ", thread_n_blocks = ",
        thread_n_blocks,
        ", thread_k_blocks = ",
        thread_k_blocks,
        ", num_bits = ",
        num_bits);
  }

  host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem));
  // clang-format off
  kernel<<<blocks, num_threads, max_shared_mem, stream>>>(
      A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, s_ptr, s2_ptr, zp_ptr, g_idx_ptr,
      sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
      topk_weights_ptr, top_k, mul_topk_weights, is_ep, num_groups, prob_m,
      prob_n, prob_k, locks, has_bias, use_atomic_add, use_fp32_reduce, max_shared_mem);
  // clang-format on
}

template <typename scalar_t>
void marlin_direct_mm(
    const void* A,
    const void* B,
    void* C,
    void* C_tmp,
    void* b_bias,
    void* s,
    void* s2,
    void* zp,
    void* g_idx,
    void* masked_m,
    void* active_expert_ids,
    void* active_expert_count,
    int launch_num_experts,
    int expected_m,
    int a_expert_stride,
    int c_expert_stride,
    int num_experts,
    int moe_block_size,
    int prob_m,
    int prob_n,
    int prob_k,
    bool is_w13_stage,
    void* workspace,
    host::ScalarType const& q_type,
    bool has_bias,
    bool has_act_order,
    bool is_k_full,
    bool has_zp,
    int num_groups,
    int group_size,
    int dev,
    cudaStream_t stream,
    int thread_k,
    int thread_n,
    int sms,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {
  host::RuntimeCheck(!has_act_order, "marlin_direct_mm does not support act-order yet");
  host::RuntimeCheck(num_experts > 0, "num_experts must be positive");

  int thread_m_blocks = div_ceil(moe_block_size, 16);
  bool m_block_size_8 = moe_block_size == 8;

  if (has_zp) {
    host::RuntimeCheck(
        q_type == host::kU4 || q_type == host::kU8, "q_type must be u4 or u8 when has_zp = True. Got = ", q_type.str());
  } else {
    host::RuntimeCheck(
        q_type == host::kU4B8 || q_type == host::kU8B128 || q_type == host::kFE4M3fn || q_type == host::kFE2M1f,
        "q_type must be uint4b8, uint8b128, float8_e4m3fn or float4_e2m1f when has_zp = False. Got = ",
        q_type.str());
  }

  host::RuntimeCheck(
      prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m, ", ", prob_n, ", ", prob_k, "]");
  int logical_m = expected_m > 0 ? std::min(expected_m, prob_m) : prob_m;
  host::RuntimeCheck(logical_m > 0, "logical_m must be positive");

  int group_blocks = 0;
  if (group_size == -1) {
    group_blocks = -1;
  } else {
    group_blocks = group_size / 16;
    host::RuntimeCheck(
        prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);
  }

  int num_bits = q_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;
  const int4* bias_ptr = (const int4*)b_bias;
  const int4* s_ptr = (const int4*)s;
  const uint16_t* s2_ptr = (const uint16_t*)s2;
  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int32_t* masked_m_ptr = (const int32_t*)masked_m;
  const int32_t* active_expert_ids_ptr =
      active_expert_ids == nullptr ? nullptr : (const int32_t*)active_expert_ids;
  const int32_t* active_expert_count_ptr =
      active_expert_count == nullptr ? nullptr : (const int32_t*)active_expert_count;
  int* locks = (int*)workspace;

  int max_shared_mem = 0;
  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  host::RuntimeCheck(max_shared_mem > 0);

  exec_config_t exec_cfg;
  thread_config_t thread_tfg;
  if (thread_k != -1 && thread_n != -1) {
    thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
    exec_cfg = exec_config_t{1, thread_tfg};
    host::RuntimeCheck(prob_n % thread_n == 0, "prob_n = ", prob_n, " is not divisible by thread_n = ", thread_n);
    host::RuntimeCheck(prob_k % thread_k == 0, "prob_k = ", prob_k, " is not divisible by thread_k = ", thread_k);
  } else {
    if (!try_determine_small_m_direct_exec_config<scalar_t>(
            q_type,
            logical_m,
            prob_n,
            prob_k,
            is_w13_stage,
            thread_m_blocks,
            m_block_size_8,
            num_bits,
            group_size,
            is_k_full,
            has_zp,
            is_zp_float,
            max_shared_mem,
            sms,
            &exec_cfg)) {
      exec_cfg = determine_direct_exec_config<scalar_t>(
          q_type,
          logical_m,
          prob_n,
          prob_k,
          thread_m_blocks,
          m_block_size_8,
          num_bits,
          group_size,
          is_k_full,
          has_zp,
          is_zp_float,
          max_shared_mem);
    }
    thread_tfg = exec_cfg.tb_cfg;
  }

  int num_threads = thread_tfg.num_threads;
  thread_k = thread_tfg.thread_k;
  thread_n = thread_tfg.thread_n;
  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  const int y_launch_blocks =
      active_expert_ids_ptr == nullptr ? num_experts : max(min(launch_num_experts, num_experts), 1);
  if (thread_m_blocks == 1) {
    exec_cfg.blocks_per_sm = limit_direct_blocks_per_sm_for_lock_avoidance(
        exec_cfg.blocks_per_sm,
        logical_m,
        prob_n,
        thread_n,
        moe_block_size,
        sms);
  }

  int blocks = sms * exec_cfg.blocks_per_sm;
  if (exec_cfg.blocks_per_sm > 1) max_shared_mem = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

  bool tensor_map_b_ready = false;
  bool tensor_map_s_ready = false;
  device::marlin_moe::tma::LocalTensorMapDesc tensor_map_b_desc{};
  device::marlin_moe::tma::LocalTensorMapDesc tensor_map_s_desc{};
#if SGLANG_MARLIN_MOE_HAS_LOCAL_TMA
  const int pack_factor = 32 / num_bits;
  const int b_sh_stride = ((thread_n * 16 / pack_factor) / 4);
  const int b_tma_landing_inner = b_sh_stride * 4;
  const int b_tma_landing_outer = thread_k_blocks;
  const size_t b_tma_barrier_bytes = sizeof(device::marlin_moe::tma::Barrier) * device::marlin::pipe_stages;
  const bool b_tma_has_barrier_space =
      b_tma_barrier_bytes <= 2 * static_cast<size_t>(moe_block_size) * sizeof(int4);
  const int scale_tma_group_blocks = num_groups > 1 ? group_size / 16 : -1;
  if (thread_m_blocks == 1 && !has_act_order) {
    const DLDataType b_tensor_map_dtype = {.code = DLDataTypeCode::kDLInt, .bits = 32, .lanes = 1};
    const int b_gmem_inner_dim = prob_n * 16 / pack_factor;
    const int b_gmem_outer_dim = num_experts * (prob_k / 16);
    const int b_gmem_outer_stride = b_gmem_inner_dim;
    const int b_sh_stride = ((thread_n_blocks * 16) * 16 / pack_factor) / 4;
    const int b_smem_inner_dim = b_sh_stride * 4;
    const int b_smem_outer_dim = thread_k_blocks;
    const int b_inner_tile = b_smem_inner_dim;
    const int b_outer_tile = b_smem_outer_dim;
    const int expert_outer_dim = prob_k / 16;
    host::RuntimeCheck(
        b_inner_tile <= b_gmem_inner_dim,
        "B-TMA inner tile exceeds gmem inner dim: b_inner_tile=",
        b_inner_tile,
        ", b_gmem_inner_dim=",
        b_gmem_inner_dim,
        ", prob_n=",
        prob_n);
    host::RuntimeCheck(
        b_outer_tile <= expert_outer_dim,
        "B-TMA outer tile exceeds expert outer dim: b_outer_tile=",
        b_outer_tile,
        ", expert_outer_dim=",
        expert_outer_dim,
        ", thread_k=",
        thread_k,
        ", prob_k=",
        prob_k);
    tensor_map_b_ready = device::marlin_moe::tma::make_tensor_map_2d_desc(
        &tensor_map_b_desc,
        B,
        b_tensor_map_dtype,
        b_gmem_inner_dim,
        b_gmem_outer_dim,
        b_smem_inner_dim,
        b_smem_outer_dim,
        b_gmem_outer_stride,
        0);
    host::RuntimeCheck(
        tensor_map_b_ready,
        "B-TMA descriptor encode failed: prob_n=",
        prob_n,
        ", prob_k=",
        prob_k,
        ", thread_n=",
        thread_n,
        ", thread_k=",
        thread_k);
    if (scale_tma_group_blocks != -1) {
      const DLDataType s_tensor_map_dtype = {.code = DLDataTypeCode::kDLInt, .bits = 32, .lanes = 1};
      const int s_gl_stride = prob_n / 8;
      const int scales_expert_stride = prob_n * prob_k / group_size / (q_type == host::kFE2M1f ? 16 : 8);
      const int s_outer_dim_per_expert = scales_expert_stride / s_gl_stride;
      const int s_sh_stride = (16 * thread_n_blocks) / 8;
      const int s_tb_groups = scale_tma_group_blocks >= thread_k_blocks
                                  ? 1
                                  : thread_k_blocks / scale_tma_group_blocks / (q_type == host::kFE2M1f ? 2 : 1);
      const int s_gmem_inner_dim = s_gl_stride * 4;
      const int s_gmem_outer_dim = num_experts * s_outer_dim_per_expert;
      const int s_gmem_outer_stride = s_gmem_inner_dim;
      const int s_smem_inner_dim = s_sh_stride * 4;
      const int s_smem_outer_dim = s_tb_groups;
      host::RuntimeCheck(
          s_smem_outer_dim <= s_outer_dim_per_expert,
          "S-TMA outer tile exceeds expert outer dim: s_smem_outer_dim=",
          s_smem_outer_dim,
          ", s_outer_dim_per_expert=",
          s_outer_dim_per_expert,
          ", num_groups=",
          num_groups,
          ", group_size=",
          group_size);
      tensor_map_s_ready = device::marlin_moe::tma::make_tensor_map_2d_desc(
          &tensor_map_s_desc,
          s,
          s_tensor_map_dtype,
          s_gmem_inner_dim,
          s_gmem_outer_dim,
          s_smem_inner_dim,
          s_smem_outer_dim,
          s_gmem_outer_stride,
          0);
      host::RuntimeCheck(
          tensor_map_s_ready,
          "S-TMA descriptor encode failed: prob_n=",
          prob_n,
          ", prob_k=",
          prob_k,
          ", thread_n=",
          thread_n,
          ", thread_k=",
          thread_k,
          ", num_groups=",
          num_groups);
    }
  }
#endif

  const int parallel = div_ceil(logical_m, moe_block_size);
  const int max_no_split_blocks = max(parallel * n_tiles, 1);
  if (thread_m_blocks == 1 && logical_m <= 32) {
    blocks = min(blocks, max_no_split_blocks);
  }

  host::RuntimeCheck(
      is_valid_config(
          thread_tfg,
          m_block_size_8,
          thread_m_blocks,
          logical_m,
          prob_n,
          prob_k,
          num_bits,
          group_size,
          false,
          is_k_full,
          has_zp,
          is_zp_float,
          max_shared_mem),
      "Invalid direct thread config");

  auto kernel = get_marlin_direct_kernel<scalar_t>(
      q_type,
      thread_m_blocks,
      thread_n_blocks,
      thread_k_blocks,
      m_block_size_8,
      has_zp,
      group_blocks,
      num_threads,
      is_zp_float);

  if (kernel == MarlinDirectDefault) {
    host::Panic(
        "Unsupported direct shapes: MNK = [",
        logical_m,
        ", ",
        prob_n,
        ", ",
        prob_k,
        "]",
        ", num_groups = ",
        num_groups,
        ", group_size = ",
        group_size,
        ", thread_m_blocks = ",
        thread_m_blocks,
        ", thread_n_blocks = ",
        thread_n_blocks,
        ", thread_k_blocks = ",
        thread_k_blocks,
        ", num_bits = ",
        num_bits);
  }

  host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem));
  dim3 grid(blocks, y_launch_blocks);
  kernel<<<grid, num_threads, max_shared_mem, stream>>>(
      A_ptr,
      B_ptr,
      C_ptr,
      C_tmp_ptr,
      bias_ptr,
      s_ptr,
      s2_ptr,
      zp_ptr,
      g_idx_ptr,
      masked_m_ptr,
      active_expert_ids_ptr,
      active_expert_count_ptr,
      tensor_map_b_desc,
      tensor_map_b_ready,
      tensor_map_s_desc,
      tensor_map_s_ready,
      expected_m,
      a_expert_stride,
      c_expert_stride,
      num_groups,
      prob_m,
      prob_n,
      prob_k,
      locks,
      has_bias,
      use_atomic_add,
      use_fp32_reduce,
      max_shared_mem);
  host::RuntimeDeviceCheck(cudaPeekAtLastError());
}

#endif

}  // namespace device::marlin_moe

template <typename scalar_t>
void moe_wna16_marlin_gemm(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView c,
    tvm::ffi::TensorView b_q_weight,
    tvm::ffi::TensorView b_bias,
    tvm::ffi::TensorView b_scales,
    tvm::ffi::TensorView global_scale,
    tvm::ffi::TensorView b_zeros,
    tvm::ffi::TensorView g_idx,
    tvm::ffi::TensorView perm,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView sorted_token_ids,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView num_tokens_post_padded,
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView a_tmp,
    tvm::ffi::TensorView c_tmp,
    int64_t moe_block_size,
    int64_t top_k,
    bool mul_topk_weights,
    bool is_ep,
    int64_t b_q_type_id,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool has_act_order,
    bool has_bias,
    bool is_k_full,
    bool has_zp,
    int64_t num_groups,
    int64_t group_size,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {
  using namespace host;

  ScalarType const b_q_type = ScalarType::from_id(b_q_type_id);
  int pack_factor = 32 / b_q_type.size_bits();

  if (moe_block_size != 8) {
    RuntimeCheck(moe_block_size % 16 == 0, "unsupported moe_block_size=", moe_block_size);
    RuntimeCheck(moe_block_size >= 16 && moe_block_size <= 64, "unsupported moe_block_size=", moe_block_size);
  }

  // Verify A
  RuntimeCheck(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0), ", size_m = ", size_m);
  RuntimeCheck(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1), ", size_k = ", size_k);

  // Verify B
  RuntimeCheck(
      size_k % device::marlin::tile_size == 0,
      "size_k = ",
      size_k,
      " is not divisible by tile_size = ",
      device::marlin::tile_size);
  RuntimeCheck(
      (size_k / device::marlin::tile_size) == b_q_weight.size(1),
      "Shape mismatch: b_q_weight.size(1) = ",
      b_q_weight.size(1),
      ", size_k = ",
      size_k,
      ", tile_size = ",
      device::marlin::tile_size);
  RuntimeCheck(
      b_q_weight.size(2) % device::marlin::tile_size == 0,
      "b_q_weight.size(2) = ",
      b_q_weight.size(2),
      " is not divisible by tile_size = ",
      device::marlin::tile_size);
  int64_t actual_size_n = (b_q_weight.size(2) / device::marlin::tile_size) * pack_factor;
  RuntimeCheck(size_n == actual_size_n, "size_n = ", size_n, ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({-1, -1}).with_dtype<scalar_t>().with_device(device).verify(a);

  device.verify(b_q_weight.device());
  RuntimeCheck(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  device.verify(b_scales.device());
  RuntimeCheck(b_scales.is_contiguous(), "b_scales is not contiguous");

  // thread_k, thread_n, sms
  int thread_k = -1;
  int thread_n = -1;
  int sms = -1;
  DLDevice dl_device = device.unwrap();
  int dev = dl_device.device_id;
  cudaStream_t stream = LaunchKernel::resolve_device(dl_device);
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

  // Verify c (allocation done in Python)
  device.verify(c.device());
  RuntimeCheck(c.is_contiguous(), "c is not contiguous");
  RuntimeCheck(
      c.size(0) == size_m * top_k, "Shape mismatch: c.size(0) = ", c.size(0), ", size_m * topk = ", size_m * top_k);
  RuntimeCheck(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1), ", size_n = ", size_n);

  // Alloc c_tmp: SKIP, done in Python

  // Detect groupsize: b_scales rank and dims
  RuntimeCheck(b_scales.dim() == 3, "b_scales rank = ", b_scales.dim(), " is not 3");
  RuntimeCheck(b_scales.size(2) == size_n, "b_scales dim 2 = ", b_scales.size(2), " is not size_n = ", size_n);
  RuntimeCheck(
      b_scales.size(1) == num_groups, "b_scales dim 1 = ", b_scales.size(1), " is not num_groups = ", num_groups);

  // Validate g_idx, perm (Optional unwrap done in Python; empty tensors when absent)
  if (g_idx.size(g_idx.dim() - 1) > 0 && perm.size(perm.dim() - 1) > 0) {
    device.verify(g_idx.device());
    RuntimeCheck(g_idx.is_contiguous(), "g_idx is not contiguous");
    device.verify(perm.device());
    RuntimeCheck(perm.is_contiguous(), "perm is not contiguous");

    int64_t g_idx_last = g_idx.size(g_idx.dim() - 1);
    int64_t perm_last = perm.size(perm.dim() - 1);
    RuntimeCheck(
        (g_idx_last == 0 && perm_last == 0) || (g_idx_last == size_k && perm_last == size_k),
        "Unexpected g_idx.size(-1) = ",
        g_idx_last,
        " and perm.size(-1) = ",
        perm_last,
        ", where size_k = ",
        size_k);
  }
  // has_act_order derivation: SKIP (passed as param)

  // Verify group_size consistency
  if (has_act_order) {
    // SKIP: a_tmp allocation done in Python
    if (is_k_full) {
      RuntimeCheck(num_groups > 1, "For act_order, num_groups must be > 1");
      RuntimeCheck(size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by num_groups = ", num_groups);
    }
  } else {
    if (num_groups > 1) {
      RuntimeCheck(
          size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by b_scales.size(1) = ", num_groups);
    }
  }

  // Verify global_scale (Optional unwrap done in Python)
  int64_t global_scale_size = global_scale.size(0);
  if (global_scale_size > 0) {
    RuntimeCheck(b_q_type == kFE2M1f && group_size == 16, "global_scale can only be used for nvfp4 format.");
  } else {
    RuntimeCheck(
        !(b_q_type == kFE2M1f && group_size == 16), "the global_scale parameter must be passed for nvfp4 format.");
  }

  // Verify b_bias (Optional unwrap done in Python)
  if (has_bias) {
    device.verify(b_bias.device());
    RuntimeCheck(b_bias.is_contiguous(), "b_bias is not contiguous");
    RuntimeCheck(b_bias.size(1) == size_n, "b_bias.size(0) != size_n");
    RuntimeCheck(b_bias.stride(1) == 1, "b_bias.stride(1) != 1");
  }

  // b_zeros Optional unwrap + has_zp derivation: SKIP (done in Python)

  // Verify b_q_type vs has_zp
  if (has_zp) {
    device.verify(b_zeros.device());
    RuntimeCheck(b_zeros.is_contiguous(), "b_zeros is not contiguous");
    RuntimeCheck(
        b_q_type == kU4 || b_q_type == kU8, "b_q_type must be u4 or u8 when has_zp = True. Got = ", b_q_type.str());
  } else {
    RuntimeCheck(
        b_q_type == kU4B8 || b_q_type == kU8B128 || b_q_type == kFE4M3fn || b_q_type == kFE2M1f,
        "b_q_type must be uint4b8, uint8b128, float8_e4m3fn or "
        "float4_e2m1f when "
        "has_zp = False. Got = ",
        b_q_type.str());
  }

  if (has_zp && is_zp_float) {
    RuntimeCheck(
        std::is_same<scalar_t, fp16_t>::value,
        "Computation type must be float16 (half) when using float zero "
        "points.");
  }

  // Verify b_zeros
  if (has_zp) {
    RuntimeCheck(b_zeros.dim() == 3, "b_zeros rank = ", b_zeros.dim(), " is not 3");
    if (is_zp_float) {
      RuntimeCheck(b_zeros.size(2) == size_n, "b_zeros dim 2 = ", b_zeros.size(2), " is not size_n = ", size_n);
      RuntimeCheck(
          num_groups == b_zeros.size(1), "b_zeros dim 1 = ", b_zeros.size(1), " is not num_groups = ", num_groups);
      RuntimeCheck(num_groups != -1, "num_groups must be != -1");
    } else {
      RuntimeCheck(
          b_zeros.size(1) == num_groups, "b_zeros dim 1 = ", b_zeros.size(1), " is not num_groups = ", num_groups);
      RuntimeCheck(
          b_zeros.size(2) == size_n / pack_factor,
          "b_zeros dim 2 = ",
          b_zeros.size(2),
          " is not size_n / pack_factor = ",
          size_n / pack_factor);
    }
  }

  // Verify workspace size
  RuntimeCheck(
      size_n % device::marlin::min_thread_n == 0,
      "size_n = ",
      size_n,
      ", is not divisible by min_thread_n = ",
      device::marlin::min_thread_n);

  int64_t max_n_tiles = size_n / device::marlin::min_thread_n;
  int64_t min_workspace_size =
      std::min(max_n_tiles * (sorted_token_ids.size(0) / moe_block_size), static_cast<int64_t>(sms) * 4);
  RuntimeCheck(
      workspace.size(0) >= min_workspace_size,
      "workspace.numel = ",
      workspace.size(0),
      " is below min_workspace_size = ",
      min_workspace_size);

  // Early return for zero-size M (moved after all validation)
  if (size_m == 0) return;

  device::marlin_moe::marlin_mm<scalar_t>(
      a.data_ptr(),
      b_q_weight.data_ptr(),
      c.data_ptr(),
      c_tmp.data_ptr(),
      b_bias.data_ptr(),
      b_scales.data_ptr(),
      global_scale.data_ptr(),
      b_zeros.data_ptr(),
      g_idx.data_ptr(),
      perm.data_ptr(),
      a_tmp.data_ptr(),
      sorted_token_ids.data_ptr(),
      expert_ids.data_ptr(),
      num_tokens_post_padded.data_ptr(),
      topk_weights.data_ptr(),
      static_cast<int>(moe_block_size),
      static_cast<int>(top_k),
      mul_topk_weights,
      is_ep,
      static_cast<int>(size_m),
      static_cast<int>(size_n),
      static_cast<int>(size_k),
      workspace.data_ptr(),
      b_q_type,
      has_bias,
      has_act_order,
      is_k_full,
      has_zp,
      static_cast<int>(num_groups),
      static_cast<int>(group_size),
      dev,
      stream,
      thread_k,
      thread_n,
      sms,
      use_atomic_add,
      use_fp32_reduce,
      is_zp_float);
}

template <typename scalar_t>
void deepep_moe_wna16_marlin_batched_gemm(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView c,
    tvm::ffi::TensorView w13_q_weight,
    tvm::ffi::TensorView w2_q_weight,
    tvm::ffi::TensorView w13_scales,
    tvm::ffi::TensorView w2_scales,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView c_tmp,
    tvm::ffi::TensorView sorted_token_ids,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView num_tokens_post_padded,
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView intermediate1,
    tvm::ffi::TensorView intermediate2,
    tvm::ffi::TensorView masked_m,
    int64_t moe_block_size,
    int64_t b_q_type_id,
    int64_t hidden_size,
    int64_t intermediate_size,
    bool is_k_full,
    int64_t group_size,
    bool use_atomic_add_w13,
    bool use_atomic_add_w2,
    bool use_fp32_reduce) {
  using namespace host;

  ScalarType const b_q_type = ScalarType::from_id(b_q_type_id);
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  RuntimeCheck(a.dim() == 3, "a rank = ", a.dim(), " is not 3");
  RuntimeCheck(c.dim() == 3, "c rank = ", c.dim(), " is not 3");
  RuntimeCheck(masked_m.dim() == 1, "masked_m rank = ", masked_m.dim(), " is not 1");
  RuntimeCheck(a.size(0) == c.size(0), "expert dimension mismatch");
  RuntimeCheck(a.size(1) == c.size(1), "token dimension mismatch");
  RuntimeCheck(a.size(2) == hidden_size, "hidden_size mismatch");
  RuntimeCheck(c.size(2) == hidden_size, "output hidden_size mismatch");
  RuntimeCheck(w13_q_weight.size(0) == a.size(0), "w13 expert dimension mismatch");
  RuntimeCheck(w2_q_weight.size(0) == a.size(0), "w2 expert dimension mismatch");
  RuntimeCheck(w13_scales.size(0) == a.size(0), "w13 scales expert dimension mismatch");
  RuntimeCheck(w2_scales.size(0) == a.size(0), "w2 scales expert dimension mismatch");
  RuntimeCheck(intermediate1.size(0) == a.size(1), "intermediate1 token dimension mismatch");
  RuntimeCheck(intermediate1.size(1) == 2 * intermediate_size, "intermediate1 hidden dimension mismatch");
  RuntimeCheck(intermediate2.size(0) == a.size(1), "intermediate2 token dimension mismatch");
  RuntimeCheck(intermediate2.size(1) == intermediate_size, "intermediate2 hidden dimension mismatch");
  RuntimeCheck(topk_weights.size(0) == a.size(1), "topk_weights token dimension mismatch");
  RuntimeCheck(topk_weights.size(1) == 1, "topk_weights second dimension mismatch");
  RuntimeCheck(num_tokens_post_padded.size(0) == 1, "num_tokens_post_padded size mismatch");
  RuntimeCheck(masked_m.size(0) == a.size(0), "masked_m expert dimension mismatch");
  RuntimeCheck(group_size == -1 || group_size > 0, "group_size must be -1 or positive");

  TensorMatcher({-1, -1, -1}).with_dtype<scalar_t>().with_device(device).verify(a);
  TensorMatcher({-1, -1, -1}).with_dtype<scalar_t>().with_device(device).verify(c);
  TensorMatcher({-1, -1}).with_dtype<scalar_t>().with_device(device).verify(intermediate1);
  TensorMatcher({-1, -1}).with_dtype<scalar_t>().with_device(device).verify(intermediate2);
  device.verify(w13_q_weight.device());
  device.verify(w2_q_weight.device());
  device.verify(w13_scales.device());
  device.verify(w2_scales.device());
  device.verify(workspace.device());
  device.verify(c_tmp.device());
  device.verify(sorted_token_ids.device());
  device.verify(expert_ids.device());
  device.verify(num_tokens_post_padded.device());
  device.verify(topk_weights.device());
  RuntimeCheck(a.is_contiguous(), "a must be contiguous");
  RuntimeCheck(c.is_contiguous(), "c must be contiguous");
  RuntimeCheck(w13_q_weight.is_contiguous(), "w13_q_weight must be contiguous");
  RuntimeCheck(w2_q_weight.is_contiguous(), "w2_q_weight must be contiguous");
  RuntimeCheck(w13_scales.is_contiguous(), "w13_scales must be contiguous");
  RuntimeCheck(w2_scales.is_contiguous(), "w2_scales must be contiguous");
  RuntimeCheck(workspace.is_contiguous(), "workspace must be contiguous");
  RuntimeCheck(c_tmp.is_contiguous(), "c_tmp must be contiguous");
  RuntimeCheck(sorted_token_ids.is_contiguous(), "sorted_token_ids must be contiguous");
  RuntimeCheck(expert_ids.is_contiguous(), "expert_ids must be contiguous");
  RuntimeCheck(num_tokens_post_padded.is_contiguous(), "num_tokens_post_padded must be contiguous");
  RuntimeCheck(topk_weights.is_contiguous(), "topk_weights must be contiguous");
  RuntimeCheck(intermediate1.is_contiguous(), "intermediate1 must be contiguous");
  RuntimeCheck(intermediate2.is_contiguous(), "intermediate2 must be contiguous");
  RuntimeCheck(masked_m.is_contiguous(), "masked_m must be contiguous");

  DLDevice dl_device = device.unwrap();
  int dev = dl_device.device_id;
  cudaStream_t stream = LaunchKernel::resolve_device(dl_device);
  int sms = -1;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

  auto* a_base = reinterpret_cast<char*>(a.data_ptr());
  auto* c_base = reinterpret_cast<char*>(c.data_ptr());
  auto* w13_base = reinterpret_cast<char*>(w13_q_weight.data_ptr());
  auto* w2_base = reinterpret_cast<char*>(w2_q_weight.data_ptr());
  auto* w13_scales_base = reinterpret_cast<char*>(w13_scales.data_ptr());
  auto* w2_scales_base = reinterpret_cast<char*>(w2_scales.data_ptr());
  auto* masked_m_ptr = static_cast<const int32_t*>(masked_m.data_ptr());

  size_t a_stride0_bytes = a.stride(0) * sizeof(scalar_t);
  size_t c_stride0_bytes = c.stride(0) * sizeof(scalar_t);
  size_t w13_stride0_bytes = w13_q_weight.stride(0) * sizeof(int32_t);
  size_t w2_stride0_bytes = w2_q_weight.stride(0) * sizeof(int32_t);
  size_t w13_scales_stride0_bytes = w13_scales.stride(0) * sizeof(scalar_t);
  size_t w2_scales_stride0_bytes = w2_scales.stride(0) * sizeof(scalar_t);

  int num_experts = static_cast<int>(a.size(0));
  int padded_m = static_cast<int>(a.size(1));
  int max_num_tokens_padded = static_cast<int>(sorted_token_ids.size(0));
  int max_num_m_blocks = static_cast<int>(expert_ids.size(0));
  int num_groups_w13 = static_cast<int>(w13_scales.size(1));
  int num_groups_w2 = static_cast<int>(w2_scales.size(1));

  RuntimeCheck(max_num_tokens_padded >= padded_m, "sorted_token_ids capacity is too small");
  RuntimeCheck(max_num_m_blocks >= div_ceil(padded_m, static_cast<int>(moe_block_size)), "expert_ids capacity is too small");

  RuntimeDeviceCheck(cudaMemsetAsync(
      c.data_ptr(), 0, c.size(0) * c.size(1) * c.size(2) * sizeof(scalar_t), stream));
  RuntimeDeviceCheck(cudaMemsetAsync(
      expert_ids.data_ptr(), 0, expert_ids.size(0) * sizeof(int32_t), stream));

  for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    int size_m = masked_m_ptr[expert_idx];
    if (size_m <= 0) {
      continue;
    }

    RuntimeCheck(size_m <= padded_m, "masked_m exceeds padded_m");

    int num_tokens_padded =
        ((size_m + static_cast<int>(moe_block_size) - 1) / static_cast<int>(moe_block_size)) *
        static_cast<int>(moe_block_size);
    int blocks = div_ceil(num_tokens_padded, 256);
    device::marlin_moe::build_direct_sorted_token_ids_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<int32_t*>(sorted_token_ids.data_ptr()),
        size_m,
        num_tokens_padded);
    RuntimeDeviceCheck(cudaMemcpyAsync(
        num_tokens_post_padded.data_ptr(),
        &num_tokens_padded,
        sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream));

    device::marlin_moe::marlin_mm<scalar_t>(
        a_base + expert_idx * a_stride0_bytes,
        w13_base + expert_idx * w13_stride0_bytes,
        intermediate1.data_ptr(),
        c_tmp.data_ptr(),
        nullptr,
        w13_scales_base + expert_idx * w13_scales_stride0_bytes,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        sorted_token_ids.data_ptr(),
        expert_ids.data_ptr(),
        num_tokens_post_padded.data_ptr(),
        topk_weights.data_ptr(),
        static_cast<int>(moe_block_size),
        1,
        false,
        false,
        size_m,
        2 * static_cast<int>(intermediate_size),
        static_cast<int>(hidden_size),
        workspace.data_ptr(),
        b_q_type,
        false,
        false,
        is_k_full,
        false,
        num_groups_w13,
        static_cast<int>(group_size),
        dev,
        stream,
        -1,
        -1,
        sms,
        use_atomic_add_w13,
        use_fp32_reduce,
        false);

    int silu_blocks = div_ceil(size_m * static_cast<int>(intermediate_size), 256);
    device::marlin_moe::silu_and_mul_kernel<scalar_t><<<silu_blocks, 256, 0, stream>>>(
        static_cast<const scalar_t*>(intermediate1.data_ptr()),
        static_cast<scalar_t*>(intermediate2.data_ptr()),
        size_m,
        static_cast<int>(intermediate_size));

    device::marlin_moe::marlin_mm<scalar_t>(
        intermediate2.data_ptr(),
        w2_base + expert_idx * w2_stride0_bytes,
        c_base + expert_idx * c_stride0_bytes,
        c_tmp.data_ptr(),
        nullptr,
        w2_scales_base + expert_idx * w2_scales_stride0_bytes,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        sorted_token_ids.data_ptr(),
        expert_ids.data_ptr(),
        num_tokens_post_padded.data_ptr(),
        topk_weights.data_ptr(),
        static_cast<int>(moe_block_size),
        1,
        false,
        false,
        size_m,
        static_cast<int>(hidden_size),
        static_cast<int>(intermediate_size),
        workspace.data_ptr(),
        b_q_type,
        false,
        false,
        is_k_full,
        false,
        num_groups_w2,
        static_cast<int>(group_size),
        dev,
        stream,
        -1,
        -1,
        sms,
        use_atomic_add_w2,
        use_fp32_reduce,
        false);
  }
}

template <typename scalar_t>
void deepep_moe_wna16_marlin_direct_gemm(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView c,
    tvm::ffi::TensorView b_q_weight,
    tvm::ffi::TensorView b_bias,
    tvm::ffi::TensorView b_scales,
    tvm::ffi::TensorView global_scale,
    tvm::ffi::TensorView b_zeros,
    tvm::ffi::TensorView g_idx,
    tvm::ffi::TensorView active_expert_ids,
    tvm::ffi::TensorView active_expert_count,
    int64_t launch_num_experts,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView masked_m,
    int64_t expected_m,
    tvm::ffi::TensorView c_tmp,
    int64_t moe_block_size,
    int64_t b_q_type_id,
    int64_t num_experts,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool is_w13_stage,
    bool has_bias,
    bool is_k_full,
    bool has_zp,
    int64_t num_groups,
    int64_t group_size,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {
  using namespace host;

  ScalarType const b_q_type = ScalarType::from_id(b_q_type_id);
  int pack_factor = 32 / b_q_type.size_bits();

  RuntimeCheck(a.dim() == 3, "a rank = ", a.dim(), " is not 3");
  RuntimeCheck(c.dim() == 3, "c rank = ", c.dim(), " is not 3");
  RuntimeCheck(masked_m.dim() == 1, "masked_m rank = ", masked_m.dim(), " is not 1");
  RuntimeCheck(
      active_expert_ids.dim() == 1,
      "active_expert_ids rank = ",
      active_expert_ids.dim(),
      " is not 1");
  RuntimeCheck(
      active_expert_count.dim() == 1,
      "active_expert_count rank = ",
      active_expert_count.dim(),
      " is not 1");
  RuntimeCheck(expected_m == -1 || expected_m > 0, "expected_m must be -1 or positive");
  RuntimeCheck(a.size(0) == num_experts, "a.size(0) mismatch");
  RuntimeCheck(c.size(0) == num_experts, "c.size(0) mismatch");
  RuntimeCheck(masked_m.size(0) == num_experts, "masked_m.size(0) mismatch");
  RuntimeCheck(a.size(1) == size_m, "a.size(1) mismatch");
  RuntimeCheck(a.size(2) == size_k, "a.size(2) mismatch");
  RuntimeCheck(c.size(1) == size_m, "c.size(1) mismatch");
  RuntimeCheck(c.size(2) == size_n, "c.size(2) mismatch");
  RuntimeCheck(b_q_weight.size(0) == num_experts, "b_q_weight.size(0) mismatch");
  RuntimeCheck(
      (size_k / device::marlin::tile_size) == b_q_weight.size(1),
      "b_q_weight.size(1) mismatch");
  RuntimeCheck(
      b_q_weight.size(2) % device::marlin::tile_size == 0,
      "b_q_weight.size(2) must be divisible by tile_size");
  RuntimeCheck(
      size_n == (b_q_weight.size(2) / device::marlin::tile_size) * pack_factor,
      "size_n mismatch with b_q_weight");
  RuntimeCheck(b_scales.dim() == 3, "b_scales rank = ", b_scales.dim(), " is not 3");
  RuntimeCheck(b_scales.size(0) == num_experts, "b_scales.size(0) mismatch");
  RuntimeCheck(b_scales.size(1) == num_groups, "b_scales.size(1) mismatch");
  RuntimeCheck(b_scales.size(2) == size_n, "b_scales.size(2) mismatch");

  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  device.verify(a.device());
  device.verify(c.device());
  device.verify(b_q_weight.device());
  device.verify(b_scales.device());
  device.verify(workspace.device());
  device.verify(masked_m.device());
  device.verify(active_expert_ids.device());
  device.verify(active_expert_count.device());
  device.verify(c_tmp.device());
  RuntimeCheck(
      a.stride(2) == 1 && a.stride(1) == size_k,
      "a must be contiguous within each expert tile");
  RuntimeCheck(
      c.stride(2) == 1 && c.stride(1) == size_n,
      "c must be contiguous within each expert tile");
  RuntimeCheck(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");
  RuntimeCheck(b_scales.is_contiguous(), "b_scales is not contiguous");
  RuntimeCheck(workspace.is_contiguous(), "workspace is not contiguous");
  RuntimeCheck(masked_m.is_contiguous(), "masked_m is not contiguous");
  RuntimeCheck(active_expert_ids.is_contiguous(), "active_expert_ids is not contiguous");
  RuntimeCheck(active_expert_count.is_contiguous(), "active_expert_count is not contiguous");
  RuntimeCheck(c_tmp.is_contiguous(), "c_tmp is not contiguous");

  if (has_bias) {
    device.verify(b_bias.device());
    RuntimeCheck(b_bias.is_contiguous(), "b_bias is not contiguous");
    RuntimeCheck(b_bias.size(0) == num_experts, "b_bias.size(0) mismatch");
    RuntimeCheck(b_bias.size(1) == size_n, "b_bias.size(1) mismatch");
  }

  if (has_zp) {
    device.verify(b_zeros.device());
    RuntimeCheck(b_zeros.is_contiguous(), "b_zeros is not contiguous");
    RuntimeCheck(b_zeros.size(0) == num_experts, "b_zeros.size(0) mismatch");
  }

  if (g_idx.size(g_idx.dim() - 1) > 0) {
    device.verify(g_idx.device());
    RuntimeCheck(g_idx.is_contiguous(), "g_idx is not contiguous");
  }

  if (active_expert_ids.size(0) > 0) {
    RuntimeCheck(
        active_expert_ids.size(0) <= num_experts,
        "active_expert_ids.size(0) exceeds num_experts");
  }
  if (active_expert_count.size(0) > 0) {
    RuntimeCheck(active_expert_count.size(0) == 1, "active_expert_count.size(0) must be 1");
  }

  int thread_k = -1;
  int thread_n = -1;
  int sms = -1;
  DLDevice dl_device = device.unwrap();
  int dev = dl_device.device_id;
  cudaStream_t stream = LaunchKernel::resolve_device(dl_device);
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  int64_t min_direct_workspace_size =
      static_cast<int64_t>(num_experts) * (static_cast<int64_t>(sms) * 4 + 1);
  RuntimeCheck(
      workspace.size(0) >= min_direct_workspace_size,
      "workspace.numel = ",
      workspace.size(0),
      " is below min_direct_workspace_size = ",
      min_direct_workspace_size);

  constexpr int scalar_per_int4 = sizeof(int4) / sizeof(scalar_t);
  RuntimeCheck(
      a.stride(0) % scalar_per_int4 == 0,
      "a.stride(0) must align with int4 packing");
  int a_expert_stride = static_cast<int>(a.stride(0) / scalar_per_int4);
  RuntimeCheck(
      c.stride(0) % scalar_per_int4 == 0,
      "c.stride(0) must align with int4 packing");
  int c_expert_stride = static_cast<int>(c.stride(0) / scalar_per_int4);

  if (size_m == 0 || num_experts == 0) return;

  device::marlin_moe::marlin_direct_mm<scalar_t>(
      a.data_ptr(),
      b_q_weight.data_ptr(),
      c.data_ptr(),
      c_tmp.data_ptr(),
      b_bias.data_ptr(),
      b_scales.data_ptr(),
      global_scale.data_ptr(),
      b_zeros.data_ptr(),
      g_idx.data_ptr(),
      masked_m.data_ptr(),
      active_expert_ids.size(0) == 0 ? nullptr : active_expert_ids.data_ptr(),
      active_expert_count.size(0) == 0 ? nullptr : active_expert_count.data_ptr(),
      static_cast<int>(launch_num_experts),
      static_cast<int>(expected_m),
      a_expert_stride,
      c_expert_stride,
      static_cast<int>(num_experts),
      static_cast<int>(moe_block_size),
      static_cast<int>(size_m),
      static_cast<int>(size_n),
      static_cast<int>(size_k),
      is_w13_stage,
      workspace.data_ptr(),
      b_q_type,
      has_bias,
      false,
      is_k_full,
      has_zp,
      static_cast<int>(num_groups),
      static_cast<int>(group_size),
      dev,
      stream,
      thread_k,
      thread_n,
      sms,
      use_atomic_add,
      use_fp32_reduce,
      is_zp_float);
}
