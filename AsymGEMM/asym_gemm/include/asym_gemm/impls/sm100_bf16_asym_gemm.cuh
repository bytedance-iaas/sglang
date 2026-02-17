#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <asym_gemm/common/asymScheduler.cuh>
#include <asym_gemm/common/utils.cuh>
#include <asym_gemm/common/sm100_utils.cuh>

namespace asym_gemm {

using namespace asym_gemm::sm100;

__device__ __forceinline__ float bf16_to_float(cutlass::bfloat16_t v) {
    return static_cast<float>(v);
}

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages_,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation, typename cd_dtype_t,
          uint64_t kTensorCoreUtilControl>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_bf16_asym_gemm_impl(uint32_t* offsets, uint32_t* experts,
                     uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    // Enlarge `BLOCK_K` for some cases
    // NOTES: this is for reducing the `umma_arrive()` overhead
    constexpr bool kDoMergeStages =
        kNumStages_ >= 8 and kGemmType == GemmType::Normal and
        kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K;
    // Ensure there are at least `kNumMinStages` stages after merge
    constexpr uint32_t kNumMinStages = 8;
    constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
    constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;
    constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;
    DG_STATIC_ASSERT(kNumStages == 2, "This simplified setup requires kNumStages == 2");

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // if (threadIdx.x == 0 && blockIdx.y == 3)
    //     printf("blockIdx.x: %d, blockIdx.y: %d \n", blockIdx.x, blockIdx.y);           

    // if (blockIdx.y != 4)
    //     return;

    // GEMM with accumulation must have FP32 output
    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t WAVE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M;
    constexpr uint32_t kNumTMAStoreStages = 2;
    // DG_STATIC_ASSERT(BLOCK_K_ == 64, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0 and 2 % kNumMWaves == 0, "Invalid block M");
    DG_STATIC_ASSERT(sizeof(cutlass::bfloat16_t) * LAYOUT_AD_M % kSwizzleAMode == 0, "Invalid swizzle A mode");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Utils
    bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = get_lane_idx();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // 2-CTA MMA
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
    DG_STATIC_ASSERT(not kIsMulticastOnA or kNumMulticast == 1, "Invalid multicast");
    DG_STATIC_ASSERT(LOAD_BLOCK_M == BLOCK_M, "Only support tensor memory layout A/D");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_NUM_TILES = 1;
    constexpr uint32_t SMEM_B_SIZE = SMEM_B_SIZE_PER_STAGE * SMEM_B_NUM_TILES;
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
                     "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

    // NOTES: Make sure we have enough shared memory for UMMA padding
    static constexpr uint32_t UMMA_A_SIZE_PER_STAGE = constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(nv_bfloat16);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE, "Memory Out of bound for UMMA");

    // Automatically deduce the number of epilogue stages (1 or 2), according to the tensor memory size
    // TODO: test cases of `kNumMWaves == 2 and kNumEpilogueStages == 2`
    constexpr uint32_t kNumEpilogueStages = 2;

    // Real tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols>();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // D/A/B shared memory
    auto smem_cd = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a  = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE);
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE);
    auto full_barriers              = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers             = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto tmem_full_barriers         = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_empty_barriers        = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages + i); });
    
    auto full_barriers_b            = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages * 2); });
    auto empty_barriers_b           = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages * 2 + 1); });
    // auto tmem_full_barriers_b       = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages * 2 + 2); });
    // auto tmem_empty_barriers_b      = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages * 2 + 3); });
    
    // NOTE: Extra barriers for B (full/empty).
    int extend_barrier = 2;
    auto tensor_core_full_barrier   = barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2 + extend_barrier;

    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     printf("information about tensor memory, kNumStages: %d, kNumEpilogueStages: %d, kNumTMAStoreStages: %d, kNumAccumTmemCols: %d, kNumTmemCols: %d \n", kNumStages, kNumEpilogueStages, kNumTMAStoreStages, kNumAccumTmemCols, kNumTmemCols);
    // }

    // Fill the tensor memory pointer
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2 + extend_barrier + 1);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive only at the leader CTA
            full_barriers[i]->init(kNumMulticast);
            // Arrive at all CTAs
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            tmem_full_barriers[i]->init(1);
            // Arrive only at the leader CTA
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        if constexpr (kTensorCoreUtilControl < 100)
            tensor_core_full_barrier->init(1);

        // init barriers for B 
        // Arrive only at the leader CTA
        full_barriers_b[0]->init(kNumMulticast);
        // Arrive at all CTAs
        empty_barriers_b[0]->init(1);
        // tmem_full_barriers_b[0]->init(1);
        // tmem_empty_barriers_b[0]->init(kNumMulticast * kNumUMMAStoreThreads);

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = asymScheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(shape_m, shape_n, experts, offsets);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0, tensor_core_phase = 0, phase_b = 0;
    auto advance_pipeline = [&](uint32_t& block_idx) {
        ++ block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = (stage_idx + 1) % kNumStages;
        phase ^= stage_idx == 0;
    };

    uint32_t block_k = ceil_div_device(shape_k, BLOCK_K);
    // uint32_t block_k = 1;
    uint32_t n_idx = scheduler.n_idx;

    // if (threadIdx.x == 0) {
    //     printf("information blockIdx.x: %d, blockIdx.y: %d, Stages: %d, BLOCK_K: %d, BLOCK_N: %d, BLOCK_M: %d, block_k: %d, m_start: %d, m_end: %d \n", blockIdx.x, blockIdx.y, kNumStages, BLOCK_K, BLOCK_N, BLOCK_M, block_k, scheduler.m_start, scheduler.m_end);
    // }

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp
        // Persistently schedule over blocks
        for (int block_k_iter = 0; block_k_iter < block_k; ++block_k_iter) {
            constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
            uint32_t k_idx = block_k_iter * BLOCK_K;
            const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
            
            empty_barriers_b[0]->wait(phase_b ^ 1);
            phase_b ^= 1;
            if constexpr (kMajorB == cute::UMMA::Major::K)
            {
                tma_copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                    &tensor_map_b, full_barriers_b[0], smem_b[0], k_idx, n_idx, kNumMulticast, batch_idx);
                // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
                //     printf("block_k_iter: %d \n", block_k_iter);
                // }
            }
            if constexpr (kMajorB == cute::UMMA::Major::MN)
                tma_copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                    &tensor_map_b, full_barriers_b[0], smem_b[0], n_idx, k_idx, kNumMulticast, batch_idx);

            // if (lane_idx == 0 and block_k_iter == 0 and n_idx == 0 and blockIdx.y == 0) {
            //     printf("DBG_BF16_ASYM LOAD_B warp=%u lane=%u k_idx=%u n_idx=%u\n",
            //            (unsigned)warp_idx, (unsigned)lane_idx, (unsigned)k_idx, (unsigned)n_idx);
            // }

            if (is_leader_cta) {
                full_barriers_b[0]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
            } else {
                full_barriers_b[0]->arrive(0u);
            }

            const auto& num_total_k_blocks = ceil_div_device(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t block_m_iter = scheduler.m_start; block_m_iter < scheduler.m_end; advance_pipeline(block_m_iter)) {
                // Compute offsets
                // NOTES: the group is always concatenated with the outer dimension
                uint32_t m_idx = block_m_iter * BLOCK_M;
                // Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // NOTES: `k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major
                // And for all m-grouped GEMMs, A must be K-majored
                DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::Batched or
                                 kMajorA == cute::UMMA::Major::K, "Invalid major");
                // Add 2 CTA offsets
                if constexpr (kNumMulticast > 1) {
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * LOAD_BLOCK_M) : 0;
                    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
                //     auto bar_ptr  = full_barriers[stage_idx];
                //     auto smem_ptr = smem_a[stage_idx];

                //     printf("kNumMulticast=%u\n", (unsigned)kNumMulticast);

                //     printf("kIsMulticastOnA=%u phase=%u k_idx=%u m_idx=%u\n",
                //             (unsigned)stage_idx, (unsigned)phase, (unsigned)k_idx, (unsigned)m_idx);

                //     printf("TMA A: stage=%u phase=%u k_idx=%u m_idx=%u\n",
                //             (unsigned)stage_idx, (unsigned)phase, (unsigned)k_idx, (unsigned)m_idx);

                //     printf("  barrier=%p  smem_dst=%p\n", (void*)bar_ptr, (void*)smem_ptr);

                //     printf("  align: barrier%%16=%llu smem%%16=%llu\n",
                //             (unsigned long long)((uintptr_t)bar_ptr & 0xFULL),
                //             (unsigned long long)((uintptr_t)smem_ptr & 0xFULL));
                // }

                // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
                //     printf("Within TMA load warp, block_m_iter: %d, stage_idx: %d, phase: %d \n", block_m_iter, stage_idx, phase);
                // }

                // Issue TMAs
                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);                
                if constexpr (kMajorA == cute::UMMA::Major::K)
                    tma_copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorA == cute::UMMA::Major::MN)
                    tma_copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_idx, kNumMulticast, batch_idx);

                // if (lane_idx == 0 and block_k_iter == 0 and m_idx == 0 and n_idx == 0 and blockIdx.y == 0) {
                //     printf("DBG_BF16_ASYM LOAD_A warp=%u lane=%u stage=%u phase=%u m_idx=%u k_idx=%u\n",
                //            (unsigned)warp_idx, (unsigned)lane_idx, (unsigned)stage_idx, (unsigned)phase,
                //            (unsigned)m_idx, (unsigned)k_idx);
                // }
            
                // Arrive at full barriers
                constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE;
                if (is_leader_cta) {
                    full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
                } else {
                    full_barriers[stage_idx]->arrive(0u);
                }
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        // TODO: refactor `UMMA_M` calculation
        constexpr uint32_t UMMA_M = LAYOUT_AD_M * (kIsMulticastOnA ? 1 : kNumMulticast);
        constexpr uint32_t UMMA_N = BLOCK_N * (kIsMulticastOnA ? kNumMulticast : 1);
        constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::bfloat16_t);
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float, UMMA_M, UMMA_N, kMajorA, kMajorB>();
        
        // if (threadIdx.x == 32 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("the MMA warp, before iteraction 291: stage_idx: %d, phase: %d \n", stage_idx, phase);
        // }

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        // Merged stages only happens in NT normal GEMM cases
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
        auto a_desc = make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;
        
        // if (threadIdx.x == 32 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("the MMA warp, before iteraction 300: stage_idx: %d, phase: %d \n", stage_idx, phase);
        // }
        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // UMMA and empty barrier arrival alias
        auto umma_arrive = [](const uint64_t* barrier) {
            if constexpr (kNumMulticast == 1) {
                cutlass::arch::umma_arrive(barrier);
            } else {
                constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
            }
        };

        // if (threadIdx.x == 32 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("the MMA warp, before iteraction 327: stage_idx: %d, phase: %d \n", stage_idx, phase);
        // }

        auto empty_barrier_arrive = [&]() {
            umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));

            // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
            umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[stage_idx]));
        };

        uint32_t accum_stage_iter = 0, accum_stage_idx = 0, accum_phase_idx = 0;

        auto advance_accum_pipeline = [&]() {
            accum_stage_idx = (accum_stage_idx + 1) % kNumEpilogueStages;
            accum_phase_idx ^= accum_stage_idx == 0;
        };
        // if (threadIdx.x == 32 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("the MMA warp, before iteraction: stage_idx: %d, phase: %d \n", stage_idx, phase);
        // }
        // uint32_t accum_stage_iter = 0;
        // Persistently schedule over blocks
        for (int block_k_iter = 0; block_k_iter < block_k; ++block_k_iter) {
            full_barriers_b[0]->wait(phase_b);
            // if (threadIdx.x == 32 && block_k_iter == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            //     printf("smem_b[0] first4=%f,%f,%f,%f\n",
            //            bf16_to_float(smem_b[0][0]),
            //            bf16_to_float(smem_b[0][1]),
            //            bf16_to_float(smem_b[0][2]),
            //            bf16_to_float(smem_b[0][3]));
            // }
            phase_b ^= 1;
            tcgen05_after_thread_sync();
            // Launch MMAs
            const auto& num_total_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t block_m_iter = scheduler.m_start; block_m_iter < scheduler.m_end; advance_pipeline(block_m_iter), advance_accum_pipeline()) {
                // auto accum_stage_idx = accum_stage_iter % kNumEpilogueStages;
                // auto accum_phase_idx = (accum_stage_iter / kNumEpilogueStages) & 1;

                // Wait tensor memory empty barrier arrival
                // wait the output in tensor memory has been written to HBM
                // if (threadIdx.x == 32 && blockIdx.x == 0) {
                //     printf("the MMA warp, before wait: block_m_iter: %d, stage_idx: %d, phase: %d \n", block_m_iter, stage_idx, phase);
                // }
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
                // if (threadIdx.x == 32 && blockIdx.x == 0 && blockIdx.y == 0) {
                //     printf("the MMA warp 352, after tmem_empty_barriers wait: block_m_iter: %d, stage_idx: %d, phase: %d \n", block_m_iter, stage_idx, phase);
                // }
                tcgen05_after_thread_sync();
                // if (threadIdx.x == 32 && blockIdx.x == 0 && blockIdx.y == 0) {
                //     printf("the MMA warp 356, after tmem_empty_barriers wait: block_m_iter: %d, stage_idx: %d, phase: %d \n", block_m_iter, stage_idx, phase);
                // }
                // Wait TMA arrival
                full_barriers[stage_idx]->wait(phase);
                tcgen05_after_thread_sync();
                // if (lane_idx == 0 and block_m_iter == 0 and blockIdx.x == 0 and blockIdx.y == 0) {
                //     printf("DBG_BF16_ASYM MMA_LOAD stage=%u, BLOCK_K/UMMA_K=%u, A0..3=%f,%f,%f,%f B0..3=%f,%f,%f,%f\n",
                //            (unsigned)stage_idx, (unsigned)BLOCK_K/UMMA_K,
                //            bf16_to_float(smem_a[stage_idx][0]), bf16_to_float(smem_a[stage_idx][1]),
                //            bf16_to_float(smem_a[stage_idx][2]), bf16_to_float(smem_a[stage_idx][3]),
                //            bf16_to_float(smem_b[0][0]), bf16_to_float(smem_b[0][1]),
                //            bf16_to_float(smem_b[0][2]), bf16_to_float(smem_b[0][3]));
                //     for (int i = 0; i < 64; i += 8) {
                //         printf("DBG_BF16_ASYM MMA_LOAD64 stage=%u off=%d "
                //                "A=%f,%f,%f,%f,%f,%f,%f,%f "
                //                "B=%f,%f,%f,%f,%f,%f,%f,%f\n",
                //                (unsigned)stage_idx, i,
                //                bf16_to_float(smem_a[stage_idx][i + 0]), bf16_to_float(smem_a[stage_idx][i + 1]),
                //                bf16_to_float(smem_a[stage_idx][i + 2]), bf16_to_float(smem_a[stage_idx][i + 3]),
                //                bf16_to_float(smem_a[stage_idx][i + 4]), bf16_to_float(smem_a[stage_idx][i + 5]),
                //                bf16_to_float(smem_a[stage_idx][i + 6]), bf16_to_float(smem_a[stage_idx][i + 7]),
                //                bf16_to_float(smem_b[0][i + 0]), bf16_to_float(smem_b[0][i + 1]),
                //                bf16_to_float(smem_b[0][i + 2]), bf16_to_float(smem_b[0][i + 3]),
                //                bf16_to_float(smem_b[0][i + 4]), bf16_to_float(smem_b[0][i + 5]),
                //                bf16_to_float(smem_b[0][i + 6]), bf16_to_float(smem_b[0][i + 7]));
                //     }
                // }
                // ++accum_stage_iter;
                // if (threadIdx.x == 32 && blockIdx.x == 0 && blockIdx.y == 0) {
                //     printf("after wait the MMA warp, after sync: block_k_iter: %d, block_m_iter: %d, stage_idx: %d, phase: %d \n", block_k_iter, block_m_iter, stage_idx, phase);
                // }

                // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 32)
                //     printf("Asym before UMMA tmem_empty_barriers block_m_iter=%d \n", block_m_iter);

                // Issue UMMA in the leader CTA
                using mma_t = cute::conditional_t<kNumMulticast == 1, SM100_MMA_F16BF16_SS, SM100_MMA_F16BF16_2x1SM_SS>;
                const auto& runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                const auto& a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(0));
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                    // for (uint32_t k = 0; k < 4; ++ k) {
                        uint32_t atom_k_idx = k * UMMA_K / BLOCK_ATOM_K;
                        b_desc.lo = advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t>(b_desc_base_lo, atom_k_idx * LOAD_BLOCK_N * BLOCK_ATOM_K, k * UMMA_K % BLOCK_ATOM_K);
                        #pragma unroll
                        for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                        // for (uint32_t w = 0; w < 1; ++ w) {
                            DG_STATIC_ASSERT((WAVE_BLOCK_M * BLOCK_K) % 128 == 0, "Invalid swizzling offset");
                            a_desc.lo = advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t>(a_desc_base_lo, atom_k_idx * LOAD_BLOCK_M * BLOCK_ATOM_K + w * WAVE_BLOCK_M * BLOCK_ATOM_K, k * UMMA_K % BLOCK_ATOM_K);
                            mma_t::fma(a_desc, b_desc,
                                       accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
                                       k > 0, // start fresh (overwrite/init C) k > 0
                                       runtime_instr_desc);
                        }
                    }
                }

                // if (threadIdx.x == 32) {
                //     const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
                //     const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
                //     const uint32_t mb_first = 0;
                //     const uint32_t nb_first = 0;
                //     const uint32_t mb_mid = num_m_blocks / 2;
                //     const uint32_t nb_mid = num_n_blocks / 2;
                //     const uint32_t mb_last = num_m_blocks - 1;
                //     const uint32_t nb_last = num_n_blocks - 1;
                //     int sample_tag = -1;

                //     if (block_m_iter == mb_first && blockIdx.x == nb_first) sample_tag = 0;
                //     else if (block_m_iter == mb_mid && blockIdx.x == nb_mid) sample_tag = 1;
                //     else if (block_m_iter == mb_last && blockIdx.x == nb_last) sample_tag = 2;

                //     if (sample_tag >= 0) {
                //         uint32_t result_values[4];
                //         constexpr uint32_t kDebugResultOffset = 0;
                //         cute::SM100_TMEM_LOAD_32dp32b4x::copy(
                //             accum_stage_idx * kNumMWaves * BLOCK_N + kDebugResultOffset,
                //             result_values[0], result_values[1], result_values[2], result_values[3]);
                //         cutlass::arch::fence_view_async_tmem_load();
                //         printf("DBG_UMMA_RESULT asym tag=%d bk=%d bm=%u nb=%u accum_stage=%u "
                //                "RESULT=%f,%f,%f,%f\n",
                //                sample_tag, block_k_iter, block_m_iter, blockIdx.x, accum_stage_idx,
                //                __uint_as_float(result_values[0]), __uint_as_float(result_values[1]),
                //                __uint_as_float(result_values[2]), __uint_as_float(result_values[3]));
                //     }
                // }

                // Commit to the mbarrier object
                // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                // empty_barrier_arrive();

                // if (threadIdx.x == 32) {
                //     const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
                //     const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
                //     const uint32_t mb_first = 0;
                //     const uint32_t nb_first = 0;
                //     // const uint32_t mb_mid = num_m_blocks / 2;
                //     // const uint32_t nb_mid = num_n_blocks / 2;
                //     const uint32_t mb_mid = 1;
                //     const uint32_t nb_mid = 0;
                //     // const uint32_t mb_last = num_m_blocks - 1;
                //     // const uint32_t nb_last = num_n_blocks - 1;
                //     const uint32_t mb_last = 2;
                //     const uint32_t nb_last = 0;
                //     int sample_tag = -1;

                //     if (block_m_iter == mb_first && blockIdx.x == nb_first && blockIdx.y == 0) sample_tag = 0;
                //     else if (block_m_iter == mb_mid && blockIdx.x == nb_mid && blockIdx.y == 0) sample_tag = 1;
                //     else if (block_m_iter == mb_last && blockIdx.x == nb_last  && blockIdx.y == 0) sample_tag = 2;

                //     if (sample_tag >= 0) {
                //         auto a_debug = smem_a[stage_idx];
                //         auto b_debug = smem_b[0];
                //         printf("DBG_UMMA asym tag=%d bk=%d bm=%u nb=%u a_stage=%u "
                //             "A=%f,%f,%f,%f,%f,%f,%f,%f "
                //             "B=%f,%f,%f,%f,%f,%f,%f,%f\n",
                //             sample_tag, block_k_iter, block_m_iter, blockIdx.x, stage_idx,
                //             bf16_to_float(a_debug[0]), bf16_to_float(a_debug[1]), bf16_to_float(a_debug[2]), bf16_to_float(a_debug[3]),
                //             bf16_to_float(a_debug[4]), bf16_to_float(a_debug[5]), bf16_to_float(a_debug[6]), bf16_to_float(a_debug[7]),
                //             bf16_to_float(b_debug[0]), bf16_to_float(b_debug[1]), bf16_to_float(b_debug[2]), bf16_to_float(b_debug[3]),
                //             bf16_to_float(b_debug[4]), bf16_to_float(b_debug[5]), bf16_to_float(b_debug[6]), bf16_to_float(b_debug[7]));
                //     }
                // }

                // if (block_m_iter == 0 and blockIdx.x == 0 and blockIdx.y == 0) {
                //     uint32_t mma_values[4];
                //     cute::SM100_TMEM_LOAD_32dp32b4x::copy(
                //         accum_stage_idx * kNumMWaves * BLOCK_N,
                //         mma_values[0], mma_values[1], mma_values[2], mma_values[3]);
                //     cutlass::arch::fence_view_async_tmem_load();
                //     printf("DBG_BF16_ASYM MMA_OUT threadIDx=%u block_k_iter=%u accum_stage=%u TMEM0..3=%f,%f,%f,%f\n",
                //            (unsigned)threadIdx.x, (unsigned)block_k_iter, (unsigned)accum_stage_idx,
                //            __uint_as_float(mma_values[0]), __uint_as_float(mma_values[1]),
                //            __uint_as_float(mma_values[2]), __uint_as_float(mma_values[3]));
                // }

                // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 32)
                //     printf("Asym after UMMA tmem_empty_barriers block_m_iter=%d \n", block_m_iter);

                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));

                // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));

                // Let tensor cores relax for lower possibility of frequency drop
                DG_STATIC_ASSERT(kTensorCoreUtilControl > 0, "Invalid tensor utilization control");
                if constexpr (kTensorCoreUtilControl < 100) {
                    // For utilization control
                    umma_arrive(reinterpret_cast<uint64_t*>(tensor_core_full_barrier));

                    // Wait for last UMMA to be done
                    tensor_core_full_barrier->wait(tensor_core_phase);
                    tensor_core_phase ^= 1;

                    // Sleep for certain cycles
                    constexpr static uint64_t kNumUMMACycles = (2ull * LAYOUT_AD_M * kNumMWaves * BLOCK_N * BLOCK_K) / 8192ull;
                    constexpr static uint64_t kNumDummyCycles = (100ull - kTensorCoreUtilControl) * kNumUMMACycles / kTensorCoreUtilControl;
                    const auto& start_clock = clock64();
                    if (cute::elect_one_sync())
                        while (clock64() - start_clock < kNumDummyCycles) {}
                    __syncwarp();
                }
            }
            umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers_b[0]));
        }

        // To safely deconstruct barriers, we need another round of waits
        // const auto& iter_idx = scheduler.current_iter - 1;
        // if (kNumMulticast > 1 and iter_idx >= 0) {
        //     const auto& phase = (iter_idx / kNumEpilogueStages) & 1;
        //     tmem_empty_barriers[iter_idx % kNumEpilogueStages]->wait(phase);
        // }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {        
        // Epilogue warp groups
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        // if (blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("threadIdx.x: %d, epilogue_warp_idx: %d \n", threadIdx.x, epilogue_warp_idx);
        // }

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 128) {
        //     const uint32_t tmem_val = ld_shared(tmem_ptr_in_smem);
            // printf("tmem_ptr_in_smem=%p tmem_val=%u warp_idx=%u smem_buffer=%p\n",
            //        (void*)tmem_ptr_in_smem, tmem_val, (unsigned)warp_idx, (void*)smem_buffer);
        // }
        // DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

        // TMA checks
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
        DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
        DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");

        // Share store pipeline between blocks
        uint32_t accum_stage_iter = 0, accum_stage_idx = 0, accum_phase_idx = 0, tma_stage_idx = 0;
        auto advance_store_pipeline = [&]() {
            tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
        };

        auto advance_accum_pipeline = [&]() {
            accum_stage_idx = (accum_stage_idx + 1) % kNumEpilogueStages;
            accum_phase_idx ^= accum_stage_idx == 0;
        };
        // if (threadIdx.x == 256 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("Within the Epilogue warp, stage_idx: %d, phase: %d \n", stage_idx, phase);
        // }
        // uint32_t accum_stage_idx = -1, accum_phase_idx = -1;
        // Persistently schedule over blocks
        for (int block_k_iter = 0; block_k_iter < block_k; ++block_k_iter) {
            for (uint32_t block_m_iter = scheduler.m_start; block_m_iter < scheduler.m_end; block_m_iter++, advance_accum_pipeline()) {
                // auto accum_stage_idx = accum_stage_iter % kNumEpilogueStages;
                // auto accum_phase_idx = (accum_stage_iter / kNumEpilogueStages) & 1;

                // if (threadIdx.x == 128 && blockIdx.x == 0) {
                //     printf("Within the Epilogue warp, block_k_iter: %d, block_m_iter: %d, stage_idx: %d, phase: %d \n", block_k_iter, block_m_iter, stage_idx, phase);
                // }

                // Wait UMMA arrival
                tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
                tcgen05_after_thread_sync();
                // ++accum_stage_iter;

                // Load from tensor memory into registers, and write shared memory with STSM
                DG_STATIC_ASSERT(kNumEpilogueThreads == 128, "Epilogue threads not enough");
                DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

                // if (blockIdx.x == 0 && blockIdx.y == 0) {
                //     printf("Epilogue warp after barriers, threadIdx.x: %d, stage_idx: %d, phase: %d \n", threadIdx.x, stage_idx, phase);
                // }

                // Iterate over M waves
                #pragma unroll
                for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                    // Issue every swizzled atom and pipeline STSM and TMA store
                    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
                    #pragma unroll
                    for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
                        // Wait shared memory to be released
                        if (epilogue_warp_idx == 0)
                        {
                            // if (threadIdx.x == 128 && blockIdx.x == 0 && blockIdx.y == 0) {
                            //     printf("Epilogue warp, scheduler.n_idx: %d, kNumStores: %d, BLOCK_N: %d, STORE_BLOCK_N: %d \n", scheduler.n_idx, kNumStores, BLOCK_N, STORE_BLOCK_N);
                            // }
                            cute::tma_store_wait<kNumTMAStoreStages - 1>();
                        }
                        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                        // The pipeline stage
                        const auto m_idx = BLOCK_M * block_m_iter + w * WAVE_BLOCK_M;
                        const auto n_idx = blockIdx.x * BLOCK_N + s * STORE_BLOCK_N;

                        // Store into shared memory
                        #pragma unroll
                        for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                            // Calculate the index of the bank group to be written in the atom
                            auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);

                            // Reshape the atom in another view and swizzle
                            //  - original: `(LAYOUT_AD_M, kSwizzleCDMode / kNumBankGroupBytes)`
                            //  - new: `(LAYOUT_AD_M * kSwizzleCDMode / kNumBankGroupBytes / 8, 8)`
                            // NOTES: "8" is the number of bank groups, "16" is the swizzling pattern
                            constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                            auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                            auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                            col ^= row % (kSwizzleCDMode / 16);

                            // Source and destination memory address
                            uint32_t tmem_addr = accum_stage_idx * kNumMWaves * BLOCK_N +         // Accumulator offset
                                                w * BLOCK_N +                                          // Wave offset
                                                s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;         // In-block offset
                            auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +        // Base pointer
                                            epilogue_warp_idx * 32 * kSwizzleCDMode +                   // Warp offset
                                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;  // In-atom offset

                            // Load from tensor memory, store into shared memory
                            uint32_t values[kNumElemsPerBankGroup];
                            if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                                // For FP32 output, read and store
                                DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
                                cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr,
                                    values[0], values[1], values[2], values[3]);
                                cutlass::arch::fence_view_async_tmem_load();
                                // if (lane_idx == 0 and block_k_iter == 0 and block_m_iter == 0 and blockIdx.x == 0 and w == 0 and s == 0 and i == 0 and blockIdx.y == 0) {
                                //     printf("DBG_BF16_ASYM EPI_TMEM warp=%u epi_warp=%u lane=%u stage=%u tmem_addr=%u values=%f,%f,%f,%f\n",
                                //            (unsigned)warp_idx, (unsigned)epilogue_warp_idx, (unsigned)lane_idx, (unsigned)accum_stage_idx, (unsigned)tmem_addr,
                                //            __uint_as_float(values[0]), __uint_as_float(values[1]),
                                //            __uint_as_float(values[2]), __uint_as_float(values[3]));
                                // }
                                // if (epilogue_warp_idx == 0 && lane_idx == 0 && w == 0 && s == 0 && i == 0) {
                                //     const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
                                //     const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
                                //     const uint32_t mb_first = 0;
                                //     const uint32_t nb_first = 0;
                                //     // const uint32_t mb_mid = num_m_blocks / 2;
                                //     // const uint32_t nb_mid = num_n_blocks / 2;
                                //     const uint32_t mb_mid = 1;
                                //     const uint32_t nb_mid = 0;
                                //     // const uint32_t mb_last = num_m_blocks - 1;
                                //     // const uint32_t nb_last = num_n_blocks - 1;
                                //     const uint32_t mb_last = 2;
                                //     const uint32_t nb_last = 0;
                                //     int sample_tag = -1;
                                //     if (block_k_iter == 0 && block_m_iter == mb_first && blockIdx.x == nb_first && blockIdx.y == 0) sample_tag = 0;
                                //     else if (block_k_iter == 0 && block_m_iter == mb_mid && blockIdx.x == nb_mid && blockIdx.y == 0) sample_tag = 1;
                                //     else if (block_k_iter == 0 && block_m_iter == mb_last && blockIdx.x == nb_last && blockIdx.y == 0) sample_tag = 2;
                                //     if (sample_tag >= 0) {
                                //         printf("DBG_TMEM_BEFORE_STSM asym tag=%d bk=%d bm=%u nb=%u m=%u n=%u tmem_addr=%u "
                                //                "TMEM=%f,%f,%f,%f\n",
                                //                sample_tag, block_k_iter, block_m_iter, blockIdx.x, m_idx, n_idx, tmem_addr,
                                //                __uint_as_float(values[0]), __uint_as_float(values[1]),
                                //                __uint_as_float(values[2]), __uint_as_float(values[3]));
                                //     }
                                // }
                                st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
                            } else {
                                // For BF16 output, read, cast and store
                                DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8 and cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid type");
                                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                                    values[0], values[1], values[2], values[3],
                                    values[4], values[5], values[6], values[7]);
                                cutlass::arch::fence_view_async_tmem_load();
                                // if (lane_idx == 0 and block_k_iter == 0 and block_m_iter == 0 and blockIdx.x == 0 and w == 0 and s == 0 and i == 0 and blockIdx.y == 0) {
                                //     printf("DBG_BF16_ASYM EPI_TMEM warp=%u epi_warp=%u lane=%u stage=%u tmem_addr=%u values=%f,%f,%f,%f\n",
                                //            (unsigned)warp_idx, (unsigned)epilogue_warp_idx, (unsigned)lane_idx, (unsigned)accum_stage_idx, (unsigned)tmem_addr,
                                //            __uint_as_float(values[0]), __uint_as_float(values[1]),
                                //            __uint_as_float(values[2]), __uint_as_float(values[3]));
                                // }
                                // if (epilogue_warp_idx == 0 && lane_idx == 0 && w == 0 && s == 0 && i == 0) {
                                //     const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
                                //     const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
                                //     const uint32_t mb_first = 0;
                                //     const uint32_t nb_first = 0;
                                //     // const uint32_t mb_mid = num_m_blocks / 2;
                                //     // const uint32_t nb_mid = num_n_blocks / 2;
                                //     const uint32_t mb_mid = 1;
                                //     const uint32_t nb_mid = 0;
                                //     // const uint32_t mb_last = num_m_blocks - 1;
                                //     // const uint32_t nb_last = num_n_blocks - 1;
                                //     const uint32_t mb_last = 2;
                                //     const uint32_t nb_last = 0;
                                //     int sample_tag = -1;
                                //     if (block_k_iter == 0 && block_m_iter == mb_first && blockIdx.x == nb_first && blockIdx.y == 0) sample_tag = 0;
                                //     else if (block_k_iter == 0 && block_m_iter == mb_mid && blockIdx.x == nb_mid && blockIdx.y == 0) sample_tag = 1;
                                //     else if (block_k_iter == 0 && block_m_iter == mb_last && blockIdx.x == nb_last && blockIdx.y == 0) sample_tag = 2;
                                //     if (sample_tag >= 0) {
                                //         printf("DBG_TMEM_BEFORE_STSM asym tag=%d bk=%d bm=%u nb=%u m=%u n=%u tmem_addr=%u "
                                //                "TMEM=%f,%f,%f,%f,%f,%f,%f,%f\n",
                                //                sample_tag, block_k_iter, block_m_iter, blockIdx.x, m_idx, n_idx, tmem_addr,
                                //                __uint_as_float(values[0]), __uint_as_float(values[1]),
                                //                __uint_as_float(values[2]), __uint_as_float(values[3]),
                                //                __uint_as_float(values[4]), __uint_as_float(values[5]),
                                //                __uint_as_float(values[6]), __uint_as_float(values[7]));
                                //     }
                                // }
                                st_shared(smem_ptr,
                                        cast_into_bf16_and_pack(values[0], values[1]),
                                        cast_into_bf16_and_pack(values[2], values[3]),
                                        cast_into_bf16_and_pack(values[4], values[5]),
                                        cast_into_bf16_and_pack(values[6], values[7]));
                            }
                        }

                        // if (epilogue_warp_idx == 0 && lane_idx == 0 &&
                        //     w == 0 && s == 0) {
                        //     const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
                        //     const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
                        //     const uint32_t mb_first = 0;
                        //     const uint32_t nb_first = 0;
                        //     const uint32_t mb_mid = num_m_blocks / 2;
                        //     const uint32_t nb_mid = num_n_blocks / 2;
                        //     const uint32_t mb_last = num_m_blocks - 1;
                        //     const uint32_t nb_last = num_n_blocks - 1;
                        //     int sample_tag = -1;
                        //     if (block_m_iter == mb_first && blockIdx.x == nb_first) sample_tag = 0;
                        //     else if (block_m_iter == mb_mid && blockIdx.x == nb_mid) sample_tag = 1;
                        //     else if (block_m_iter == mb_last && blockIdx.x == nb_last) sample_tag = 2;

                        //     if (sample_tag >= 0) {
                        //         auto d_debug = smem_cd[tma_stage_idx];
                        //         printf("DBG_EPI_CD_AFTER_UMMA asym tag=%d bk=%d bm=%u nb=%u m=%u n=%u cd_stage=%u "
                        //                "CD=%f,%f,%f,%f,%f,%f,%f,%f\n",
                        //                sample_tag, block_k_iter, block_m_iter, blockIdx.x, m_idx, n_idx, tma_stage_idx,
                        //                static_cast<float>(d_debug[0]), static_cast<float>(d_debug[1]),
                        //                static_cast<float>(d_debug[2]), static_cast<float>(d_debug[3]),
                        //                static_cast<float>(d_debug[4]), static_cast<float>(d_debug[5]),
                        //                static_cast<float>(d_debug[6]), static_cast<float>(d_debug[7]));
                        //     }
                        // }

                        // Notify tensor memory empty (only at the leader CTA) arrival ASAP
                        // NOTES: only the last stage needs to do this
                        if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                            // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 128)
                            //     printf("Asym before release tmem_empty_barriers block_m_iter=%d \n", block_m_iter);
                            tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }

                        // tcgen05_before_thread_sync();
                        // tmem_empty_barriers[stage_idx]->arrive(0u);

                        __syncwarp();

                        // Synchronize all threads and issue TMA
                        cute::tma_store_fence();
                        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                        // if (epilogue_warp_idx == 0 && lane_idx == 0 &&
                        //     w == 0 && s == 0) {
                        //     const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
                        //     const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
                        //     const uint32_t mb_first = 0;
                        //     const uint32_t nb_first = 0;
                        //     const uint32_t mb_mid = num_m_blocks / 2;
                        //     const uint32_t nb_mid = num_n_blocks / 2;
                        //     const uint32_t mb_last = num_m_blocks - 1;
                        //     const uint32_t nb_last = num_n_blocks - 1;
                        //     int sample_tag = -1;
                        //     if (block_k_iter == 0 && block_m_iter == mb_first && scheduler.n_idx == nb_first) sample_tag = 0;
                        //     else if (block_k_iter == 0 && block_m_iter == mb_mid && scheduler.n_idx == nb_mid) sample_tag = 1;
                        //     else if (block_k_iter == 0 && block_m_iter == mb_last && scheduler.n_idx == nb_last) sample_tag = 2;
 
                        //     if (sample_tag >= 0) {
                        //         auto d_debug = smem_cd[tma_stage_idx];
                        //         printf("DBG_EPI_CD_PRE_TMA asym tag=%d bk=%d bm=%u nb=%u m=%u n=%u cd_stage=%u "
                        //                "CD=%f,%f,%f,%f,%f,%f,%f,%f\n",
                        //                sample_tag, block_k_iter, block_m_iter, scheduler.n_idx, m_idx, n_idx, tma_stage_idx,
                        //                static_cast<float>(d_debug[0]), static_cast<float>(d_debug[1]), static_cast<float>(d_debug[2]), static_cast<float>(d_debug[3]),
                        //                static_cast<float>(d_debug[4]), static_cast<float>(d_debug[5]), static_cast<float>(d_debug[6]), static_cast<float>(d_debug[7]));
                        //     }
                        // }

                        // if (blockIdx.x == 0 && blockIdx.y == 0) {
                        //     printf("Epilogue warp before store, threadIdx.x: %d, stage_idx: %d, phase: %d \n", threadIdx.x, stage_idx, phase);
                        // }

                        if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {

                            if constexpr (kGemmType == GemmType::Batched) {
                                // if (blockIdx.x == 0 && blockIdx.y == 0) {
                                //     printf("Epilogue warp within store 590, threadIdx.x: %d, stage_idx: %d, phase: %d \n", threadIdx.x, stage_idx, phase);
                                // }
                                using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                    cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                                cute_tma_t::copy(&tensor_map_cd, smem_cd[tma_stage_idx],
                                                n_idx, m_idx, scheduler.current_group_idx);
                            } else {
                          
                                if (block_k_iter == 0)
                                {
                                    // if (blockIdx.x == 0 && blockIdx.y == 0) {
                                    //     printf("Epilogue warp, kWithAccumulation: false, threadIdx.x: %d, tma_stage_idx: %d, n_idx: %d, m_idx: %d \n", threadIdx.x, tma_stage_idx, n_idx, m_idx);
                                    // }
                                    using cute_tma_t = cute::conditional_t<false,
                                        cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                                    cute_tma_t::copy(&tensor_map_cd, smem_cd[tma_stage_idx], n_idx, m_idx);
                                }
                                else
                                {
                                    // if (blockIdx.x == 0 && blockIdx.y == 0) {
                                    //     printf("Epilogue warp within store, threadIdx.x: %d, tma_stage_idx: %d, n_idx: %d, m_idx: %d \n", threadIdx.x, tma_stage_idx, n_idx, m_idx);
                                    // }
                                    // if (blockIdx.x == 0 && blockIdx.y == 0) {
                                    //     printf("Epilogue warp, kWithAccumulation: false, threadIdx.x: %d, tma_stage_idx: %d, n_idx: %d, m_idx: %d \n", threadIdx.x, tma_stage_idx, n_idx, m_idx);
                                    // }
                                    using cute_tma_t = cute::conditional_t<true,
                                        cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                                    cute_tma_t::copy(&tensor_map_cd, smem_cd[tma_stage_idx], n_idx, m_idx);
                                }
                             
                            }
                            cute::tma_store_arrive();
                        }
                    }
                }

                // if (threadIdx.x == 128 && blockIdx.x == 0 && blockIdx.y == 0) {
                //     printf("Within the Epilogue warp, after storing, block_k_iter: %d, block_m_iter: %d, stage_idx: %d, phase: %d \n", block_k_iter, block_m_iter, stage_idx, phase);
                // }
            }
        }

        // Deallocate tensor memory by the last UMMA store warp
        // NOTES: warp 0 is waiting TMA store
        if (epilogue_warp_idx == kNumUMMAStoreThreads / 32 - 1) {
            const auto tmem_ptr = ld_shared(tmem_ptr_in_smem);
            // if (blockIdx.x == 0 && blockIdx.y == 0) {
            //     printf("Within the Epilogue warp, Allocator().free, threadIdx.x: %d, stage_idx: %d, phase: %d \n", threadIdx.x, stage_idx, phase);
            // }
            Allocator().free(tmem_ptr, kNumTmemCols);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace asym_gemm

#pragma clang diagnostic pop
