#include <cuda_fp16.h>
#include <iostream>
#include <random>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <curand_kernel.h>
#include <c10/cuda/CUDAStream.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"
#include "rotary.h"

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <chrono>

#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "static_switch.h"
#include "hardware_info.h"
// #include "flash.h"
#include <cooperative_groups.h>

// #include "bandwidth_attention.h"

// #include "philox_unpack.cuh" // For at::cuda::philox::unpack
#include "namespace_config.h"

using namespace cooperative_groups;
using namespace cute;
using namespace FLASH_NAMESPACE;

// Error checking macro
#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

// kHeadDim_, kBlockM_, kBlockN_
// int SM_size = 128;
// using Kernel_traits = Flash_fwd_kernel_traits<32, 128, 32, 4, false, false, cutlass::half_t>;

// CUTLASS GEMM kernel configuration for Hopper (SM90)
using ElementGEMM = cutlass::half_t; // FP32
using LayoutGEMM = cutlass::layout::RowMajor;
using Gemm = cutlass::gemm::device::Gemm<
    ElementGEMM, LayoutGEMM, // Matrix A
    ElementGEMM, LayoutGEMM, // Matrix B
    ElementGEMM, LayoutGEMM, // Matrix C
    ElementGEMM,         // Accumulator
    cutlass::arch::OpClassSimt, // SIMT operation
    cutlass::arch::Sm90         // Hopper architecture
>;

// Updated Params struct to match block_info.h expectations
struct Params {
    cutlass::half_t* x_ptr;       // [B, S, H]
    cutlass::half_t* w_ptr;
    cutlass::half_t* o_ptr;                // Output polonger
    cutlass::half_t* bias_ptr;                // Output polonger

    int expert_size;   // Sequence length S
    int list_size;
    int* expert_list;
    int* index_list;
    long M;
    long N;
    long K;
};

template<typename Kernel_traits>
__global__ void cpuAwareTilingWithoutBias(Params params) {
    const int sliceID = blockIdx.x;
    const int expertID = blockIdx.y;

    const int tidx = threadIdx.x;
    const int tile_size = Kernel_traits::kBlockM; // configurable

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kBlockK = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    using Element = typename Kernel_traits::Element;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];

    // Load X tile: [BlockM, H]
    Tensor mW = make_tensor(make_gmem_ptr(params.w_ptr), make_shape(params.expert_size, params.N, params.K), make_stride(params.N * params.K, params.K, _1{}));

    Tensor sX = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutX{});
    Tensor sW = make_tensor(sX.data() + size(sX), typename Kernel_traits::SmemLayoutW{});
    // Tensor sWt = make_tensor(sW.data(), typename Kernel_traits::SmemLayoutWtransposed{});
    // Tensor sWtNoSwizzle = make_tensor(sW.data().get(), typename Kernel_traits::SmemLayoutWtransposedNoSwizzle{});
    Tensor sO = make_tensor(sW.data() + size(sW), typename Kernel_traits::SmemLayoutC{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tXsX = gmem_thr_copy_QKV.partition_D(sX);
    Tensor tWsW = gmem_thr_copy_QKV.partition_D(sW);
    Tensor tOsO_QKV = gmem_thr_copy_QKV.partition_S(sO);

    Tensor cX = make_identity_tensor(make_shape(size<0>(sX), size<1>(sX)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cW = make_identity_tensor(make_shape(size<0>(sW), size<1>(sW)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    Tensor tXcX = gmem_thr_copy_QKV.partition_S(cX);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tWcW = gmem_thr_copy_QKV.partition_S(cW);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tOcO = gmem_thr_copy_QKV.partition_S(cO);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    Tensor tXpX = make_tensor<bool>(make_shape(size<2>(tXsX)));
    Tensor tWpW = make_tensor<bool>(make_shape(size<2>(tWsW)));
    // Tensor tWpW = make_tensor<bool>(make_shape(Int<8>{}, Int<16>{}, Int<8>{}));

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrX  = thr_mma.partition_fragment_A(sX); 
    Tensor tOrW  = thr_mma.partition_fragment_B(sW);    // (MMA, MMA_K,MMA_N)
    Tensor tSrO  = thr_mma.partition_fragment_C(sO);    // (MMA, MMA_K,MMA_N)

    auto smem_tiled_copy_X = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_X = smem_tiled_copy_X.get_thread_slice(tidx);
    Tensor tSsX = smem_thr_copy_X.partition_S(sX);

    auto smem_tiled_copy_W = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_W = smem_tiled_copy_W.get_thread_slice(tidx);
    Tensor tOsW = smem_thr_copy_W.partition_S(sW);

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOsO)));

    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tSsO = smem_thr_copy_O.partition_S(sO);

    cp_async_fence();
    FLASH_NAMESPACE::cp_async_wait<0>();
    __syncthreads();
    int k_max = (params.K + kBlockK - 1) / kBlockK;
    int len_start = 0;

    int expert_id = params.expert_list[expertID];
    int len = params.index_list[expertID] - len_start;

    Tensor gW = local_tile(mW(expert_id, _, _), Shape<Int<kBlockN>, Int<kBlockK>>{}, make_coord(sliceID, _));
    Tensor tWgW = gmem_thr_copy_QKV.partition_S(gW);

    FLASH_NAMESPACE::copy</*Is_even_MN=*/true, true>(gmem_tiled_copy_QKV, tWgW(_, _, _, 0), tWsW, tWcW, tWpW);

    Tensor mO = make_tensor(make_gmem_ptr(params.o_ptr + len_start * params.N), make_shape(len, params.N), make_stride(params.N, _1{}));
    Tensor gO = local_tile(mO, Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(_, sliceID));

    Tensor mX = make_tensor(make_gmem_ptr(params.x_ptr + len_start * params.K), make_shape(len, params.K), make_stride(params.K, _1{}));
    Tensor gX = local_tile(mX, Shape<Int<kBlockM>, Int<kBlockK>>{}, make_coord(_, _));

    Tensor tXgX = gmem_thr_copy_QKV.partition_S(gX);
    Tensor tOgO_QKV = gmem_thr_copy_QKV.partition_D(gO);
    len_start = params.index_list[expertID];

    cp_async_fence();
    FLASH_NAMESPACE::cp_async_wait<0>();
    __syncthreads();

    int m_max = (len + kBlockM - 1) / kBlockM;
    int k = 0;

    for (int m = 0; m < m_max; m++) 
    {
        clear(tSrO);

        if (m * kBlockM + kBlockM <= len)
        {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, true>(gmem_tiled_copy_QKV, tXgX(_, _, _, m, k), tXsX, tXcX, tXpX);
        }
        else
        {
            int gap = len - kBlockM * m_max + kBlockM;
            FLASH_NAMESPACE::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(gmem_tiled_copy_QKV, tXgX(_, _, _, m, k), tXsX, tXcX, tXpX, gap);
        }
                
        cp_async_fence();
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        FLASH_NAMESPACE::gemm(tSrO, tSrX, tOrW, tSsX, tOsW, tiled_mma, smem_tiled_copy_X, smem_tiled_copy_W, smem_thr_copy_X, smem_thr_copy_W);

        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        Tensor rO = make_tensor_like<Element>(tSrO);
        flash::convert_type_out(tSrO, rO);

        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
        __syncthreads();

        Tensor tOgO = gmem_thr_copy_O.partition_D(gO(_, _, m));
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

        if (m * kBlockM + kBlockM <= len)
        {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO);
        }
        else
        {
            int gap = len - kBlockM * m_max + kBlockM;
            FLASH_NAMESPACE::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, gap);
        }

        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
    }

    for (k = 1; k < k_max; ++k) {
        FLASH_NAMESPACE::copy</*Is_even_MN=*/true, true>(
        gmem_tiled_copy_QKV, tWgW(_, _, _, k), tWsW, tWcW, tWpW);

        for (int m = 0; m < m_max; m++) 
        {
            clear(tSrO);

            if (m * kBlockM + kBlockM <= len)
            {
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, true>(gmem_tiled_copy_QKV, tXgX(_, _, _, m, k), tXsX, tXcX, tXpX);
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, true>(gmem_tiled_copy_QKV, tOgO_QKV(_, _, _, m), tOsO_QKV, tOcO, tOpO);
            }
            else
            {
                int gap = len - kBlockM * m_max + kBlockM;
                FLASH_NAMESPACE::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(gmem_tiled_copy_QKV, tXgX(_, _, _, m, k), tXsX, tXcX, tXpX, gap);
                FLASH_NAMESPACE::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(gmem_tiled_copy_QKV, tOgO_QKV(_, _, _, m), tOsO_QKV, tOcO, tOpO, gap);
            }

            cp_async_fence();
            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();

            Tensor tSrO_copy_view = smem_thr_copy_O.retile_D(tSrO);        // ((Atom,AtomNum), MMA_M, MMA_N)
            cute::copy(smem_tiled_copy_O, tSsO, tSrO_copy_view);
            // cp_async_fence();
            FLASH_NAMESPACE::cp_async_wait<0>();

            FLASH_NAMESPACE::gemm(tSrO, tSrX, tOrW, tSsX, tOsW, tiled_mma, smem_tiled_copy_X, smem_tiled_copy_W, smem_thr_copy_X, smem_thr_copy_W);

            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();

            Tensor rO = make_tensor_like<Element>(tSrO);
            flash::convert_type_out(tSrO, rO);

            Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
            Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

            cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
            __syncthreads();

            Tensor tOgO = gmem_thr_copy_O.partition_D(gO(_, _, m));
            Tensor tOrO = make_tensor<Element>(shape(tOgO));
            cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
            __syncthreads();

            if (m * kBlockM + kBlockM <= len)
            {
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO);
            }
            else
            {
                int gap = len - kBlockM * m_max + kBlockM;
                FLASH_NAMESPACE::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, gap);
            }

            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();
        }
    }
    // if (threadIdx.x == 0)
    //     print("expertID: %d, sliceID: %d\n", expertID, sliceID);
}

template<typename Kernel_traits>
void asymComputeWithoutBias(Params params, dim3 grid, cudaStream_t stream) {
    dim3 block(128);

    size_t smem_size = (Kernel_traits::kBlockM * Kernel_traits::kHeadDim +  Kernel_traits::kBlockN * Kernel_traits::kHeadDim +  Kernel_traits::kBlockN * Kernel_traits::kBlockM) * sizeof(half);

    cudaFuncSetAttribute(
        (void*)cpuAwareTilingWithoutBias<Kernel_traits>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

    cpuAwareTilingWithoutBias<Kernel_traits><<<grid, block, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Function to measure execution time
float measure_time(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds;
}

void launch_MoECompute(cutlass::half_t* A, cutlass::half_t* B, cutlass::half_t* C, int M, int N, int K, int expert_size, int list_size, int* expert_list, int* index_list, cudaStream_t stream) {
    // Initialize Params
    Params params;
    params.x_ptr = A;
    params.w_ptr = B;
    params.o_ptr = C;
    params.M = M;
    params.N = N;
    params.K = K;

    params.list_size = list_size;
    params.expert_size = expert_size;
    params.expert_list = expert_list;
    params.index_list = index_list;

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // print("M: %d, N: %d, K: %d, list_size: %d, expert_size: %d\n", M, N, K, list_size, expert_size);

    if (params.N == 4096)
    {
        static constexpr int kBlockN = 128;
        int block_per_expert = params.N / kBlockN;
        dim3 grid(block_per_expert, list_size);
        using KT = Flash_fwd_kernel_traits<128, 128, kBlockN, 4, false, false, cutlass::half_t>;
        asymComputeWithoutBias<KT>(params, grid, stream);
    }
    else if (params.N == 4096 * 2)
    {
        static constexpr int kBlockN = 128;
        int block_per_expert = params.N / kBlockN;
        dim3 grid(block_per_expert, list_size);
        using KT = Flash_fwd_kernel_traits<128, 128, kBlockN, 4, false, false, cutlass::half_t>;
        asymComputeWithoutBias<KT>(params, grid, stream);
    }
    else if (params.N == 7168)
    {
        static constexpr int kBlockN = 128;
        int block_per_expert = params.N / kBlockN;
        dim3 grid(block_per_expert, list_size);
        using KT = Flash_fwd_kernel_traits<128, 128, kBlockN, 4, false, false, cutlass::half_t>;
        asymComputeWithoutBias<KT>(params, grid, stream);
    }
    else
    {
        std::cout << "the dimension does not support: " << N << std::endl;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // std::cout << "cup-aware symCompute: " << measure_time(start, stop) << " ms" << std::endl;
}

// void launch_MoECompute(cutlass::half_t* A, cutlass::half_t* B, cutlass::half_t* C, int M, int N, int K, cudaStream_t stream);