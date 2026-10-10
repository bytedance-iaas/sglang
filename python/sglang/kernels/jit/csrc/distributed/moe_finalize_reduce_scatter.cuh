// Fused deferred-MoE finalize + shared add + NVLink reduce-scatter (BF16).
//
// The rank-local routed output is accumulated in registers and written
// directly to the destination rank's Lamport slot. It is never materialized
// as a [num_tokens, hidden] tensor in global memory.

#include "nvlink_comm.cuh"
#include <optional>
#include <type_traits>

namespace sglang {

using MoeRSStageVec = device::AlignedVector<bf16x2_t, 4>;
using MoeRSLamport = device::distributed::LamportTrait<bf16_t, 8, /*kAtom=*/4>;

template <uint32_t kWorldSize, typename WeightT>
struct MoeFinalizeReduceScatterParams {
  const bf16_t* gemm2;
  const int32_t* idx;
  const WeightT* weights;
  const bf16_t* shared;
  bf16_t* output;
  uint32_t rank;
  uint32_t num_push_vecs;
  uint32_t num_poll_vecs;
  uint32_t num_vecs_per_token;
  uint32_t gemm2_vecs_per_row;
  uint32_t num_gemm2_rows;
  uint32_t tokens_avg;
  uint32_t tokens_rem;
  fast_mod_div_u32_t vecs_per_token_div;
  PushWorkSpace<kWorldSize> ws;
};

template <uint32_t kTopK, bool kHasShared, typename WeightT, uint32_t kWorldSize>
SGL_DEVICE MoeRSStageVec
moe_finalize_rs_vec(const MoeFinalizeReduceScatterParams<kWorldSize, WeightT>& params, uint32_t token, uint32_t hvec) {
  using namespace device;
  const auto* idx = params.idx + static_cast<int64_t>(token) * kTopK;
  const auto* weights = params.weights + static_cast<int64_t>(token) * kTopK;
  int32_t rows[kTopK];
  WeightT weight[kTopK];
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    rows[k] = idx[k];
    weight[k] = weights[k];
  }

  MoeRSStageVec shared;
  if constexpr (kHasShared) {
    shared.load(params.shared + static_cast<int64_t>(token) * params.num_vecs_per_token * 8, hvec);
  }

  MoeRSStageVec input[kTopK];
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    if (rows[k] >= 0 && static_cast<uint32_t>(rows[k]) < params.num_gemm2_rows) {
      input[k].load(params.gemm2 + static_cast<int64_t>(rows[k]) * params.gemm2_vecs_per_row * 8, hvec);
    }
  }

  fp32x2_t acc[4] = {};
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    if (rows[k] < 0 || static_cast<uint32_t>(rows[k]) >= params.num_gemm2_rows) continue;
    const auto w = cast<fp32_t>(weight[k]);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const auto [x, y] = cast<fp32x2_t>(input[k][j]);
      acc[j].x = fmaf(x, w, acc[j].x);
      acc[j].y = fmaf(y, w, acc[j].y);
    }
  }

  MoeRSStageVec result;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    if constexpr (kHasShared) {
      // Preserve the unfused BF16 finalize, then BF16 shared-add rounding.
      const auto routed = cast<fp32x2_t>(cast<bf16x2_t>(acc[j]));
      const auto sh = cast<fp32x2_t>(shared[j]);
      result[j] = cast<bf16x2_t>(fp32x2_t{routed.x + sh.x, routed.y + sh.y});
    } else {
      result[j] = cast<bf16x2_t>(acc[j]);
    }
  }
  return result;
}

template <uint32_t kWorldSize, uint32_t kTopK, bool kUsePDL, bool kHasShared, typename WeightT>
PUSH_KERNEL void
moe_finalize_reduce_scatter_kernel(const __grid_constant__ MoeFinalizeReduceScatterParams<kWorldSize, WeightT> params) {
  using namespace device;
  constexpr uint32_t kGroup = get_poll_group<false>(kWorldSize);

  const auto warp_in_block = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto global_warp_id = blockIdx.x + gridDim.x * warp_in_block;
  const auto global_tid = global_warp_id * kWarpThreads + lane_id;
  const auto num_threads = blockDim.x * gridDim.x;

  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};
  const auto vpt = params.num_vecs_per_token;

  for (auto vid = global_tid; vid < params.num_push_vecs; vid += num_threads) {
    const auto token_id = vid / params.vecs_per_token_div;
    const auto hvec = vid % params.vecs_per_token_div;
    const auto dst_rank = token_id % kWorldSize;
    const auto dst_token_id = token_id / kWorldSize;
    const auto rank_prefix = dst_rank * params.tokens_avg + std::min(dst_rank, params.tokens_rem);
    const auto src_token = rank_prefix + dst_token_id;
    auto vec = moe_finalize_rs_vec<kTopK, kHasShared>(params, src_token, hvec);
    MoeRSLamport::clear_pos_zero(vec.data());
    ptx::st_relaxed_16B(vec, epoch.slot_ptr(dst_rank, params.rank), dst_token_id * vpt + hvec);
  }

  const auto poll_base = epoch.slot_ptr(params.rank);
  const auto slot_vecs = params.ws.slot_bytes / sizeof(MoeRSStageVec);
  MoeRSStageVec pos_zero;
  MoeRSLamport::fill_pos_zero(pos_zero.data());
  PDLTriggerSecondary<kUsePDL>();

  for (auto vid = global_tid; vid < params.num_poll_vecs; vid += num_threads) {
    MoeRSStageVec out;
    if constexpr (kGroup >= kWorldSize) {
      MoeRSStageVec vec[kWorldSize];
      do {
        bool has_zero = false;
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          ptx::ld_relaxed_16B(vec[i], poll_base, i * slot_vecs + vid);
        }
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          has_zero |= MoeRSLamport::has_pos_zero(vec[i].data());
        }
        if (!has_zero) break;
      } while (true);
      out = device::reduce_vec(vec);
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        ptx::st_global_16B(pos_zero, poll_base, i * slot_vecs + vid);
      }
    } else {
      constexpr uint32_t kNumPairs = 4;
      fp32x2_t acc[kNumPairs];
      constexpr uint32_t kNumGroups = div_ceil(kWorldSize, kGroup);
      MoeRSStageVec vec[kGroup];
#pragma unroll
      for (uint32_t g = 0; g < kNumGroups; ++g) {
        do {
          bool has_zero = false;
#pragma unroll
          for (uint32_t j = 0; j < kGroup; ++j) {
            const auto i = g * kGroup + j;
            if (i >= kWorldSize) continue;
            ptx::ld_relaxed_16B(vec[j], poll_base, i * slot_vecs + vid);
            has_zero |= MoeRSLamport::has_pos_zero(vec[j].data());
          }
          if (!has_zero) break;
        } while (true);
#pragma unroll
        for (uint32_t j = 0; j < kGroup; ++j) {
          const auto i = g * kGroup + j;
          if (i >= kWorldSize) continue;
#pragma unroll
          for (uint32_t k = 0; k < kNumPairs; ++k) {
            const auto [x, y] = cast<fp32x2_t>(vec[j][k]);
            acc[k].x = i == 0 ? x : acc[k].x + x;
            acc[k].y = i == 0 ? y : acc[k].y + y;
          }
          ptx::st_global_16B(pos_zero, poll_base, i * slot_vecs + vid);
        }
      }
#pragma unroll
      for (uint32_t k = 0; k < kNumPairs; ++k) {
        out[k] = cast<bf16x2_t>(acc[k]);
      }
    }
    out.store(params.output, vid);
  }

  __syncthreads();
  epoch.flip();
}

template <uint32_t kWorldSize, uint32_t kTopK, bool kUsePDL, typename WeightT>
struct MoeFinalizeReduceScatter {
  using TensorView = tvm::ffi::TensorView;
  using CommunicatorRef = host::distributed::CommunicatorRef;

  static void
  run(CommunicatorRef ref,
      TensorView output,
      TensorView gemm2,
      TensorView idx,
      TensorView weights,
      std::optional<TensorView> shared) {
    using namespace host;
    static_assert(std::is_same_v<WeightT, bf16_t> || std::is_same_v<WeightT, fp32_t>);
    const auto& push = ref->get_push_obj();
    CHECK_HOST(push.world_size == kWorldSize);

    auto tokens = SymbolicSize{"num_tokens"};
    auto rows = SymbolicSize{"num_permuted_rows"};
    auto hidden = SymbolicSize{"hidden"};
    auto hidden_padded = SymbolicSize{"hidden_padded"};
    auto expanded = SymbolicSize{"num_expanded"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({rows, hidden_padded})
        .with_strides({hidden_padded, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(gemm2);
    TensorMatcher({tokens, kTopK})
        .with_strides({kTopK, 1})
        .with_dtype<WeightT>()
        .template with_device<kDLCUDA>(device)
        .verify(weights);
    expanded.set_value(tokens.unwrap() * kTopK);
    TensorMatcher({expanded}).with_strides({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(idx);

    const auto total_tokens = static_cast<uint32_t>(tokens.unwrap());
    const auto rank = push.rank;
    const auto avg = total_tokens / kWorldSize;
    const auto rem = total_tokens % kWorldSize;
    const auto local_tokens = avg + (rank < rem ? 1 : 0);
    hidden.set_value(output.size(1));
    TensorMatcher({static_cast<int64_t>(local_tokens), hidden})
        .with_strides({hidden, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(output);
    if (shared.has_value()) {
      TensorMatcher({tokens, hidden})
          .with_strides({hidden, 1})
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(shared.value());
    }

    const auto hidden_size = static_cast<uint32_t>(hidden.unwrap());
    const auto hidden_padded_size = static_cast<uint32_t>(hidden_padded.unwrap());
    CHECK_HOST(total_tokens > 0);
    CHECK_HOST(hidden_size % 8 == 0);
    CHECK_HOST(hidden_padded_size >= hidden_size && hidden_padded_size % 8 == 0);
    CHECK_HOST(output.numel() * sizeof(bf16_t) <= push.slot_bytes);

    const auto vpt = hidden_size / 8;
    const auto num_push_vecs = total_tokens * vpt;
    const auto num_poll_vecs = local_tokens * vpt;
    const auto params = MoeFinalizeReduceScatterParams<kWorldSize, WeightT>{
        .gemm2 = static_cast<const bf16_t*>(gemm2.data_ptr()),
        .idx = static_cast<const int32_t*>(idx.data_ptr()),
        .weights = static_cast<const WeightT*>(weights.data_ptr()),
        .shared = shared.has_value() ? static_cast<const bf16_t*>(shared.value().data_ptr()) : nullptr,
        .output = static_cast<bf16_t*>(output.data_ptr()),
        .rank = rank,
        .num_push_vecs = num_push_vecs,
        .num_poll_vecs = num_poll_vecs,
        .num_vecs_per_token = vpt,
        .gemm2_vecs_per_row = hidden_padded_size / 8,
        .num_gemm2_rows = static_cast<uint32_t>(rows.unwrap()),
        .tokens_avg = avg,
        .tokens_rem = rem,
        .vecs_per_token_div = fast_mod_div_u32_t{vpt},
        .ws = push.get_workspace<kWorldSize>(0),
    };
    const auto kernel = shared.has_value()
                            ? moe_finalize_reduce_scatter_kernel<kWorldSize, kTopK, kUsePDL, true, WeightT>
                            : moe_finalize_reduce_scatter_kernel<kWorldSize, kTopK, kUsePDL, false, WeightT>;
    const auto block_size = choose_push_block_size(std::max(num_push_vecs, num_poll_vecs));
    LaunchKernel(push.num_blocks, block_size, output.device()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
