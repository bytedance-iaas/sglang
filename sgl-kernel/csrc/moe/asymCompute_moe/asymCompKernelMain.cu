
// #include <torch/extension.h>
// #include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <chrono>

#define CUDA_CHECK(FN) { \
    auto cudaError = FN; \
    if (cudaError != cudaSuccess) { \
        std::cerr << "FATAL: " #FN " failed: " << cudaGetErrorString(cudaError) << std::endl; \
        return -1; \
    } \
}

void launch_MoECompute(cutlass::half_t* A, cutlass::half_t* B, cutlass::half_t* C, int M, int N, int K, int expert_size, int list_size, int* expert_list, int* index_list, cudaStream_t stream);

int main() {
    // make_shape(params.num_heads_q, params.hidden_dim, kHeadDim),
    int M = 500;
    int K = 7168;         
    int N = 4096;    
    int E = 64;
    std::vector<int> h_expert_ids = {0, 3, 2, 4, 6, 5, 9, 21};
    std::vector<int> h_offsets    = {20, 30, 50, 200, 300, 400, 450, 500}; // end offsets


    int G = (int)h_expert_ids.size();
    if ((int)h_offsets.size() != G) { /* error */ }

    int list_size = h_offsets.back(); // total routed entries L

    int *d_expert_ids = nullptr;
    int *d_offsets    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_expert_ids, G * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsets,    G * sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_expert_ids, h_expert_ids.data(), G * sizeof(int),
                            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(d_offsets, h_offsets.data(), G * sizeof(int),
                            cudaMemcpyHostToDevice));

    cutlass::half_t *A_ptr, *B_ptr, *C_ptr;

    size_t x_size = static_cast<size_t>(M) * K;
    size_t o_size = static_cast<size_t>(M) * N;
    size_t wk_size = static_cast<size_t>(N) * K * E;
    cudaMalloc(&C_ptr, o_size * sizeof(cutlass::half_t));
    cudaMalloc(&A_ptr, x_size * sizeof(cutlass::half_t));
    cudaMalloc(&B_ptr, wk_size * sizeof(cutlass::half_t));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaDeviceSynchronize();

    launch_MoECompute(A_ptr, B_ptr, C_ptr, M, N, K, E, list_size, d_expert_ids, d_offsets, stream);
    cudaDeviceSynchronize();

    cudaFree(A_ptr);
    cudaFree(B_ptr);
    cudaFree(C_ptr);

    return 0;
}