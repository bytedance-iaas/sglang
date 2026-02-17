#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <tuple>
#include <utility>
#include <string>
#include <random>
#include <vector>

#include "../csrc/apis/asym_gemm.hpp"
#include "../csrc/utils/layout.hpp"
#include "../csrc/utils/math.hpp"

// IMPORTANT: include the header that owns `prepare_init` / JIT globals
#include "../csrc/jit/compiler.hpp"   // adjust if your path is different

static torch::Tensor make_fp8_e4m3(const torch::Tensor& x) {
    return x.to(torch::kFloat8_e4m3fn);
}

static torch::Tensor build_m_indices_like_generators(
    int64_t expected_m_per_group,
    int64_t num_groups,
    int64_t* out_m,
    int64_t* out_active_m
) {
    const int64_t alignment = asym_gemm::get_mk_alignment_for_contiguous_layout();
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0.7f, 1.3f);

    std::vector<int64_t> actual_ms;
    std::vector<int64_t> aligned_ms;
    actual_ms.reserve(num_groups);
    aligned_ms.reserve(num_groups);

    int64_t total_m = 0;
    int64_t active_m = 0;
    for (int64_t i = 0; i < num_groups; ++i) {
        const int64_t actual_m = std::max<int64_t>(1, static_cast<int64_t>(expected_m_per_group * dist(rng)));
        const int64_t aligned_m = asym_gemm::align(actual_m, alignment);
        actual_ms.push_back(actual_m);
        aligned_ms.push_back(aligned_m);
        total_m += aligned_m;
        active_m += actual_m;
    }

    auto m_indices_cpu = torch::empty({total_m},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    auto* mi = m_indices_cpu.data_ptr<int32_t>();

    int64_t start = 0;
    for (int64_t i = 0; i < num_groups; ++i) {
        const int64_t actual_end = start + actual_ms[i];
        const int64_t aligned_end = start + aligned_ms[i];
        for (int64_t j = start; j < actual_end; ++j)
            mi[j] = static_cast<int32_t>(i);
        for (int64_t j = actual_end; j < aligned_end; ++j)
            mi[j] = -1;
        start = aligned_end;
    }

    *out_m = total_m;
    *out_active_m = active_m;
    return m_indices_cpu;
}

int main(int argc, char** argv) {
    // JIT cache + lineinfo (optional but useful)
    setenv("DG_JIT_CACHE_DIR", "/tmp/deepgemm_jit", 1);
    setenv("DG_JIT_WITH_LINEINFO", "1", 1);
    setenv("DG_JIT_DEBUG", "1", 1);

    // CRITICAL: initialize DeepGEMM JIT globals BEFORE any kernel build
    // NOTE: pass the directory whose child is "include/asym_gemm"
    // In this repo that is: <repo>/asym_gemm
    asym_gemm::Compiler::prepare_init(
        "/sgl-workspace/sglang/AsymGEMM/asym_gemm",
        "/usr/local/cuda-12.9"
    );

    asym_gemm::KernelRuntime::prepare_init("/usr/local/cuda-12.9");

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
    const int64_t n = 4096, k = 7168, num_groups = 4;
    const int64_t expected_m_per_group = 2048;
    int64_t m = 0, active_m = 0;

    int64_t sf_k = 56;
    int64_t sf_n  = 32;

    const bool disable_ue8m0_cast = true;
    const std::string compiled_dims = "mnk";
    std::optional<std::tuple<int,int,int>> recipe = std::nullopt;

    auto m_indices_cpu = build_m_indices_like_generators(expected_m_per_group, num_groups, &m, &active_m);
    auto m_indices = m_indices_cpu.to(dev);

    auto A_fp16 = torch::randn({m, k}, torch::TensorOptions().device(dev).dtype(torch::kFloat16));
    auto A = make_fp8_e4m3(A_fp16);

    auto B_fp16_cpu = torch::randn(
        {num_groups, n, k},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat16).pinned_memory(true));
    auto B_cpu = make_fp8_e4m3(B_fp16_cpu);
    auto B_gpu = B_cpu.to(dev, /*non_blocking=*/true);

    auto D = torch::empty({m, n}, torch::TensorOptions().device(dev).dtype(torch::kBFloat16));

    auto SFA = torch::ones({m, sf_k}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    auto SFB_cpu = torch::ones(
        {num_groups, sf_n, sf_k},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32).pinned_memory(true));
    auto SFB_gpu = SFB_cpu.to(dev, /*non_blocking=*/true);

    std::pair<torch::Tensor, torch::Tensor> a_pair{A, SFA};
    std::pair<torch::Tensor, torch::Tensor> b_pair_cpu{B_cpu, SFB_cpu};
    std::pair<torch::Tensor, torch::Tensor> b_pair_gpu{B_gpu, SFB_gpu};

    std::cout << "Calling asym_gemm::gemm::m_grouped_fp8_asym_gemm_nt_contiguous...\n";
    std::cout << "m=" << m << ", active_m=" << active_m << ", n=" << n << ", k=" << k << ", num_groups=" << num_groups << "\n";
    std::cout << "A=" << A.sizes() << " " << A.scalar_type() << "  SFA=" << SFA.sizes() << " " << SFA.scalar_type() << "\n";
    std::cout << "B_cpu=" << B_cpu.sizes() << " " << B_cpu.scalar_type()
              << "  SFB_cpu=" << SFB_cpu.sizes() << " " << SFB_cpu.scalar_type() << "\n";
    std::cout << "B_gpu=" << B_gpu.sizes() << " " << B_gpu.scalar_type()
              << "  SFB_gpu=" << SFB_gpu.sizes() << " " << SFB_gpu.scalar_type() << "\n";
    std::cout << "D=" << D.sizes() << " " << D.scalar_type() << "  m_indices=" << m_indices.sizes() << " " << m_indices.scalar_type() << "\n";

    asym_gemm::gemm::m_grouped_fp8_asym_gemm_nt_contiguous(
        a_pair, b_pair_cpu, D, m_indices, recipe, compiled_dims, disable_ue8m0_cast
    );

    // asym_gemm::gemm::m_grouped_fp8_gemm_nt_contiguous(
    //     a_pair, b_pair_gpu, D, m_indices, recipe, compiled_dims, disable_ue8m0_cast
    // );

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
