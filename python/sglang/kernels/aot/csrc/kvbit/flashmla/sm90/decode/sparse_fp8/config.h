#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

using namespace cute;

namespace sm90::decode::sparse_fp8 {

template <ModelType MODEL_TYPE, int NUM_HEADS>
class KernelTemplate {
 public:
  static_assert(NUM_HEADS == 64 || NUM_HEADS == 128);
  static constexpr bool PACKED_INT4 = true;
  static constexpr int NUM_M_BLOCKS = NUM_HEADS / 64;
  static constexpr int CLUSTER_SIZE = NUM_M_BLOCKS;

  static constexpr int HEAD_DIM_K = MODEL_TYPE == ModelType::V32 ? 576 : 512;
  static constexpr int HEAD_DIM_V = 512;
  static constexpr int HEAD_DIM_ROPE = 64;
  static constexpr int HEAD_DIM_NOPE = HEAD_DIM_K - HEAD_DIM_ROPE;

  static constexpr int QUANT_TILE_SIZE = MODEL_TYPE == ModelType::V32 ? 128 : 64;
  static constexpr int NUM_SCALES = MODEL_TYPE == ModelType::V32 ? 4 : 8;  // For MODEL1: 7 fp8_e4m3 + 1 padding

  static constexpr int NUM_THREADS = 128 * 3;
  static constexpr int BLOCK_M = 64;
  static constexpr int TOPK_BLOCK_SIZE = 64;
  static constexpr int NUM_K_BUFS = 2;

  using SmemLayoutQTile =
      decltype(tile_to_shape(GMMA::Layout_SW128_Atom<bf16, GMMA::Major::K>{}, Shape<Int<BLOCK_M>, Int<64>>{}));

  template <int NUM_TILES>
  using SmemLayoutQTiles =
      decltype(tile_to_shape(SmemLayoutQTile{}, Shape<Int<BLOCK_M>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}));

  using SmemLayoutQ = SmemLayoutQTiles<HEAD_DIM_K / 64>;

  using SmemLayoutKTile = decltype(tile_to_shape(
      GMMA::Layout_INTER_Atom<bf16, GMMA::Major::K>{}, Shape<Int<TOPK_BLOCK_SIZE>, _64>{}, Step<_1, _2>{}));

  using SmemLayoutXTile =
      decltype(tile_to_shape(GMMA::Layout_SW128_Atom<bf16, GMMA::Major::K>{}, Shape<Int<TOPK_BLOCK_SIZE>, _64>{}));

  template <int NUM_TILES>
  using SmemLayoutKTiles =
      decltype(tile_to_shape(SmemLayoutKTile{}, Shape<Int<TOPK_BLOCK_SIZE>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}));

  template <int NUM_TILES>
  using SmemLayoutKTilesTransposed = decltype(composition(
      SmemLayoutKTiles<NUM_TILES>{},
      Layout<Shape<Int<64 * NUM_TILES>, Int<TOPK_BLOCK_SIZE>>, Stride<Int<TOPK_BLOCK_SIZE>, _1>>{}));

  static constexpr int OBUF_SW = 64;
  using SmemLayoutOBufAtom = GMMA::Layout_K_SW128_Atom<bf16>;
  using SmemLayoutOBuf =
      decltype(tile_to_shape(SmemLayoutOBufAtom{}, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, Step<_1, _2>{}));

  using SmemLayoutOAccumBuf = Layout<
      Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>,
      Stride<Int<520>, _1>  // We use stride = 520 here to avoid bank conflict
      >;

  using SmemLayoutK = SmemLayoutKTiles<HEAD_DIM_K / 64>;
  using SmemLayoutV = SmemLayoutKTilesTransposed<HEAD_DIM_V / 64>;
  using SmemLayoutHalfV = SmemLayoutKTilesTransposed<HEAD_DIM_V / 64 / 2>;

  using SmemLayoutS =
      decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));

  struct SharedMemoryPlan {
    array_aligned<bf16, cosize_v<SmemLayoutQ>> q;
    union {
      array_aligned<bf16, cosize_v<SmemLayoutK>> k[NUM_K_BUFS];
      array_aligned<bf16, cosize_v<SmemLayoutOBuf>> oBuf;
      array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> oAccumBuf;
    } u;
    CUTE_ALIGNAS(1024) array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    bool is_kv_valid[NUM_K_BUFS][TOPK_BLOCK_SIZE];

    float sM[BLOCK_M], sL[BLOCK_M], sScale[BLOCK_M], sOScale[BLOCK_M];
    transac_bar_t bar_q, bar_k_local_ready[NUM_K_BUFS], bar_k_remote_ready[NUM_K_BUFS], bar_k_avail[NUM_K_BUFS];

    // [M3.c.4 Stage-2] Packed-FP8 nope staging buffer (one dim-block).
    // 64 tokens x 64 dims x bf16 = 8KB.
    // Outside the union so sK writes don't clobber it.
    //
    // Dual-use:
    //   - legacy variable-bit path (bu == 0): staging holds scalar R@X output
    //     that is then copied to GMMA-layout sK.
    //   - wgmma uniform-bit path (bu > 0, MODEL1, Route G step 4+5): during
    //     the wgmma issue phase, this buffer aliases sX_tile[0] (primary sX
    //     A operand). After the last wgmma retires, the same 8KB is
    //     overwritten with the bf16(rC) output (per-thread fragment scatter)
    //     so the existing staging->sK copy path stays byte-identical.
    CUTE_ALIGNAS(128) bf16 packed_nope_staging[64 * 64];

    // [M3.c.4 Stage-5 Route G step 4+5] wgmma R matrix K-tile in smem.
    //
    // packed_r_tile: [64 dims_out, 64 dims_in] bf16 K-major (SmemLayoutKTile).
    //   Loaded from R_base per (dim_block, K-tile). Consumed by wgmma as B
    //   in MMA_64x64x16_F32BF16BF16_SS<K, K>.
    //
    // [step3k smem-fit revert] The Route G step8/9 producer double-buffers
    //   (packed_x_alt_tile + packed_r_alt_tile, +16 KB) were REMOVED. Reason:
    //   (1) MODEL1 SharedMemoryPlan with both alt tiles = 241664 B, which
    //       exceeds the H20 SM90 opt-in dyn-smem cap of 232448 B, so the
    //       kernel could not even launch (cudaFuncSetAttribute -> invalid
    //       argument). The stale Jun-30 binary masked this; the current
    //       source never actually ran.
    //   (2) Route H step2a/3b independently PROVED the packed producer is NOT
    //       the decode bottleneck (zeroing the entire producer nope rebuild
    //       gave 0 tps change), so the producer wgmma-pipeline overlap those
    //       alt tiles enabled is a dead speculative optimization.
    //   The producer loop is reverted to a byte-correct single-buffer
    //   dim-block-by-1 schedule that reuses only packed_nope_staging (sX +
    //   staging) and packed_r_tile (sR). New MODEL1 plan = 225280 B (fits).
    CUTE_ALIGNAS(128) array_aligned<bf16, cosize_v<SmemLayoutKTile>> packed_r_tile;
  };

  template <typename Shape_Q, typename TMA_Q>
  struct TmaParams {
    Shape_Q shape_Q;
    TMA_Q tma_Q;
    CUtensorMap tensor_map_o;
  };

  using TiledMMA_QK = decltype(make_tiled_mma(
      GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}, Layout<Shape<_1, _1, _1>>{}));

  using TiledMMA_QK_rQ = decltype(make_tiled_mma(
      GMMA::MMA_64x64x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{}, Layout<Shape<_1, _1, _1>>{}));

  using TiledMMA_PV_LocalP = decltype(make_tiled_mma(
      GMMA::MMA_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}, Layout<Shape<_1, _1, _1>>{}));

  using TiledMMA_PV_RemoteP = decltype(make_tiled_mma(
      GMMA::MMA_64x256x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::MN>{}, Layout<Shape<_1, _1, _1>>{}));

  enum NamedBarriers : uint32_t {
    sScale_and_sS_ready = 0,
    sScale_and_sS_free = 1,
    oBuf_free_and_sL_ready = 2,
    epilogue_r2s_ready = 3,
    batch_loop_sync = 4,
    warpgroup0_sync = 5,
    packed_kv_producer_sync = 6
  };

  // Synchronize all threads within the cluster (which processes one q token)
  static __forceinline__ __device__ void sync_all_threads_in_cluster() {
    if constexpr (CLUSTER_SIZE == 1) {
      __syncthreads();
    } else {
      ku::barrier_cluster_arrive_relaxed();
      ku::barrier_cluster_wait_acquire();
    }
  }

  // Save rPb (64x64, bfloat16) to sP using the stmatrix instruction
  template <typename Tensor0, typename Tensor1>
  static __forceinline__ __device__ void save_rPb_to_sP(Tensor0 const& rPb, Tensor1 const& sP, int idx_in_warpgroup) {
    auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, bf16>{}, TiledMMA_QK{});
    ThrCopy thr_copy = r2s_copy.get_slice(idx_in_warpgroup);
    Tensor thr_copy_rPb = thr_copy.retile_S(rPb);
    Tensor thr_copy_sP = thr_copy.partition_D(sP);
    cute::copy(r2s_copy, thr_copy_rPb, thr_copy_sP);
  }

  template <
      bool IS_NO_SPLIT,
      typename TMAParams,
      typename Tensor0,
      typename Tensor1,
      typename Tensor2,
      typename Tensor3>
  static __forceinline__ __device__ void store_o(
      Tensor0& rO,         // ((2, 2, 32), 1, 1)
      Tensor1& gOorAccum,  // (BLOCK_SIZE_M, HEAD_DIM_V)
      Tensor2& sOutputBuf,
      Tensor3& sOutputAccumBuf,
      SharedMemoryPlan& plan,
      float o_scales[2],
      TMAParams& tma_params,
      int batch_idx,
      int s_q_idx,
      int head_block_idx,
      int num_valid_seq_q,
      int warpgroup_idx,
      int idx_in_warpgroup) {
    using cutlass::arch::NamedBarrier;
    if constexpr (IS_NO_SPLIT) {
      // Should convert the output to bfloat16 / float16, and save it to O
      // Here we don't pipeline STSM and tma store because it's slower
      Tensor sMyOutputBuf = local_tile(sOutputBuf, Shape<_64, _256>{}, make_coord(_0{}, warpgroup_idx));

      // Calculate "base" ptrs in advance
      // Each STSM fills a chunk of shape 16x16, while we are using SW-OBUF_SW, so we need OBUF_SW/16 base pointers
      constexpr int NUM_CHUNKS_IN_SW_ATOM = OBUF_SW / 16;
      bf16* base_output_buf_ptrs[NUM_CHUNKS_IN_SW_ATOM];
      CUTE_UNROLL
      for (int i = 0; i < NUM_CHUNKS_IN_SW_ATOM; ++i) {
        base_output_buf_ptrs[i] = &sMyOutputBuf(
            (idx_in_warpgroup / 32) * 16 + idx_in_warpgroup % 16, idx_in_warpgroup % 32 / 16 * 8 + i * 16);
      }

      CUTE_UNROLL
      for (int idx = 0; idx < (HEAD_DIM_V / 2) / 16; idx += 1) {
        // In each iteration we deal with a chunk of shape 16x16
        using bf16x2 = __nv_bfloat162;
        bf16x2 a01 = __float22bfloat162_rn(float2{rO(idx * 8 + 0) * o_scales[0], rO(idx * 8 + 1) * o_scales[0]});
        bf16x2 a23 = __float22bfloat162_rn(float2{rO(idx * 8 + 2) * o_scales[1], rO(idx * 8 + 3) * o_scales[1]});
        bf16x2 a45 = __float22bfloat162_rn(float2{rO(idx * 8 + 4) * o_scales[0], rO(idx * 8 + 5) * o_scales[0]});
        bf16x2 a67 = __float22bfloat162_rn(float2{rO(idx * 8 + 6) * o_scales[1], rO(idx * 8 + 7) * o_scales[1]});
        SM90_U32x4_STSM_N::copy(
            *reinterpret_cast<uint32_t*>(&a01),
            *reinterpret_cast<uint32_t*>(&a23),
            *reinterpret_cast<uint32_t*>(&a45),
            *reinterpret_cast<uint32_t*>(&a67),
            *reinterpret_cast<uint128_t*>(base_output_buf_ptrs[idx % 4] + (idx / 4 * 4) * 16 * 64));
      }

      cutlass::arch::fence_view_async_shared();
      NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_r2s_ready);

      if (threadIdx.x == 0) {
        SM90_TMA_STORE_5D::copy(
            &tma_params.tensor_map_o, plan.u.oBuf.data(), 0, head_block_idx * 64, 0, s_q_idx, batch_idx);
        cute::tma_store_arrive();
      }
    } else {
      // Should save the result to OAccum
      CUTLASS_PRAGMA_UNROLL
      for (int idx = 0; idx < size(rO); idx += 2) {
        int row = (idx_in_warpgroup / 32) * 16 + (idx_in_warpgroup % 32 / 4) + (idx % 4 >= 2 ? 8 : 0);
        int col = warpgroup_idx * 256 + (idx_in_warpgroup % 4) * 2 + idx / 4 * 8;
        *(float2*)(&(sOutputAccumBuf(row, col))) = float2{
            rO(idx) * o_scales[idx % 4 >= 2],
            rO(idx + 1) * o_scales[idx % 4 >= 2],
        };
      }
      cutlass::arch::fence_view_async_shared();

      NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_r2s_ready);

      if (elect_one_sync()) {
        CUTLASS_PRAGMA_UNROLL
        for (int local_row = 0; local_row < BLOCK_M / (256 / 32); ++local_row) {
          int row = local_row * (256 / 32) + (threadIdx.x / 32);
          if (row < num_valid_seq_q) {
            SM90_BULK_COPY_S2G::copy(&sOutputAccumBuf(row, _0{}), &gOorAccum(row, _0{}), HEAD_DIM_V * sizeof(float));
          }
        }
        cute::tma_store_arrive();
      }
    }
  }

  template <typename TMAParams>
  static __device__ __forceinline__ void devfunc(const SparseAttnDecodeParams& params, const TMAParams& tma_params);

  static void run_impl(const SparseAttnDecodeParams& params);

  static void run_int4(const SparseAttnDecodeParams& params);
};

}  // namespace sm90::decode::sparse_fp8
