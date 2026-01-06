
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <chrono>

static inline void cudaCheck(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

void* host_ptr = nullptr;
float* arr;

// Frees cudaHostAlloc'ed memory (Mapped or Default)
struct CudaHostDeleter {
  void operator()(void* p) const noexcept { if (p) cudaFreeHost(p); }
};

void launch_MoECompute(cutlass::half_t* A, cutlass::half_t* B, cutlass::half_t* C, int M, int N, int K, int expert_size, int list_size, int* expert_list, int* index_list, cudaStream_t stream);

void fuseKernel(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor expert_list_tensor, torch::Tensor index_list_tensor) {
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t expert_size = B.size(0);
    int64_t list_size = expert_list_tensor.size(0);
    int64_t N = B.size(1);

    printf("A.size(0): %d, A.size(1): %d, B.size(0): %d,  B.size(1): %d, B.size(2): %d,  C.size(0): %d,  C.size(1): %d \n", A.size(0), A.size(1), B.size(0), B.size(1), B.size(2), C.size(0), C.size(1));
    printf("expert_list_tensor: %d, index_list_tensor: %d \n", expert_list_tensor.size(0), index_list_tensor.size(0));

    TORCH_CHECK(B.size(2) == K, "Dimension mismatch: A.shape[2] != B.shape[2]");

    auto* A_ptr = reinterpret_cast<const cutlass::half_t*>(A.data_ptr<at::Half>());
    auto* B_ptr = reinterpret_cast<const cutlass::half_t*>(B.data_ptr<at::Half>());
    auto* C_ptr = reinterpret_cast<const cutlass::half_t*>(C.data_ptr<at::Half>());
    auto* expert_list_ptr = expert_list_tensor.data_ptr<int>();
    auto* index_list_ptr = index_list_tensor.data_ptr<int>();

    auto stream = at::cuda::getCurrentCUDAStream();
    launch_MoECompute(A_ptr, B_ptr, C_ptr, M, N, K, expert_size, list_size, expert_list_ptr, index_list_ptr, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fuseKernel", &fuseKernel, "Matrix multiplication for MoE");
}