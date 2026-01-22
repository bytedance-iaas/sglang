#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <tuple>
#include <utility>
#include <string>

#include "../csrc/apis/asym_gemm.hpp"

// IMPORTANT: include the header that owns `prepare_init` / JIT globals
#include "../csrc/jit/compiler.hpp"   // adjust if your path is different

static torch::Tensor make_fp8_e4m3(const torch::Tensor& x) {
    return x.to(torch::kFloat8_e4m3fn);
}

static inline int64_t ceil_div_i64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

int main(int argc, char** argv) {
    // JIT cache + lineinfo (optional but useful)
    setenv("DG_JIT_CACHE_DIR", "/tmp/deepgemm_jit", 1);
    setenv("DG_JIT_WITH_LINEINFO", "1", 1);
    setenv("DG_JIT_DEBUG", "1", 1);

    // CRITICAL: initialize DeepGEMM JIT globals BEFORE any kernel build
    // NOTE: pass the directory whose child is "include/deep_gemm"
    // In this repo that is: <repo>/deep_gemm
    deep_gemm::Compiler::prepare_init(
        "/sgl-workspace/sglang/DeepGEMM/deep_gemm",
        "/usr/local/cuda-12.9"
    );

    deep_gemm::KernelRuntime::prepare_init("/usr/local/cuda-12.9");

    torch::NoGradGuard ng;
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA not available.\n";
        return 1;
    }


    auto dev = torch::Device(torch::kCUDA, 0);
    c10::cuda::CUDAGuard device_guard(dev);

    // -----------------------------
    // 3) Tensors (your current demo settings)
    // -----------------------------
    const int64_t m = 8192, n = 4096, k = 7168, num_groups = 4;

    int64_t sf_k = 56;
    int64_t sf_n  = 32;

    const bool disable_ue8m0_cast = true;
    const std::string compiled_dims = "mnk";
    std::optional<std::tuple<int,int,int>> recipe = std::nullopt;

    auto A_fp16 = torch::randn({m, k}, torch::TensorOptions().device(dev).dtype(torch::kFloat16));
    auto A = make_fp8_e4m3(A_fp16);

    auto B_fp16 = torch::randn({num_groups, n, k}, torch::TensorOptions().device(dev).dtype(torch::kFloat16));
    auto B = make_fp8_e4m3(B_fp16);

    auto D = torch::empty({m, n}, torch::TensorOptions().device(dev).dtype(torch::kBFloat16));

    auto m_indices = torch::randint(0, num_groups, {m},
        torch::TensorOptions().device(dev).dtype(torch::kInt32));
    m_indices.index_put_({torch::indexing::Slice(0, 128)}, -1);

    auto SFA = torch::ones({m, sf_k}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    auto SFB = torch::ones({num_groups, sf_n, sf_k},
        torch::TensorOptions().device(dev).dtype(torch::kFloat32));

    std::pair<torch::Tensor, torch::Tensor> a_pair{A, SFA};
    std::pair<torch::Tensor, torch::Tensor> b_pair{B, SFB};

    std::cout << "Calling deep_gemm::gemm::m_grouped_fp8_asym_gemm_nt_contiguous...\n";
    std::cout << "A=" << A.sizes() << " " << A.scalar_type() << "  SFA=" << SFA.sizes() << " " << SFA.scalar_type() << "\n";
    std::cout << "B=" << B.sizes() << " " << B.scalar_type() << "  SFB=" << SFB.sizes() << " " << SFB.scalar_type() << "\n";
    std::cout << "D=" << D.sizes() << " " << D.scalar_type() << "  m_indices=" << m_indices.sizes() << " " << m_indices.scalar_type() << "\n";

    deep_gemm::gemm::m_grouped_fp8_asym_gemm_nt_contiguous(
        a_pair, b_pair, D, m_indices, recipe, compiled_dims, disable_ue8m0_cast
    );

    cudaDeviceSynchronize();
    std::cout << "Done. D.mean=" << D.to(torch::kFloat32).mean().item<float>() << "\n";
    return 0;
}

// ipdb> pp a[0].shape
// torch.Size([35456, 7168])
// ipdb> pp a[1].shape
// torch.Size([35456, 56])
// ipdb> pp b[0].shape
// torch.Size([4, 4096, 7168])
// ipdb> pp b[1].shape
// torch.Size([4, 32, 56])