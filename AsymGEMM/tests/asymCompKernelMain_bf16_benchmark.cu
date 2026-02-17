#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <tuple>
#include <utility>
#include <string>
#include <random>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// #include "../csrc/apis/asym_gemm.hpp"
#include "../csrc/apis/gemm.hpp"
#include "../csrc/utils/layout.hpp"
#include "../csrc/utils/math.hpp"

// IMPORTANT: include the header that owns `prepare_init` / JIT globals
#include "../csrc/jit/compiler.hpp"   // adjust if your path is different

static torch::Tensor make_fp8_e4m3(const torch::Tensor& x) {
    return x.to(torch::kFloat8_e4m3fn);
}

static inline int64_t ceil_div_i64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static int fill_with_sentinel(
    int* m_indices, int M,
    int* offsets, int* experts, int capacity
) {
    if (!offsets || !experts || capacity <= 0) return 0;

    if (M <= 0 || !m_indices) {
        return 0;
    }

    int write = 0;

    auto maybe_emit = [&](int start_idx) {
        int e = m_indices[start_idx];
        if (e != -1) {
            if (write < capacity) {
                offsets[write] = start_idx;
                experts[write] = e;
            }
            ++write;
        }
    };

    maybe_emit(0);
    for (int i = 1; i < M; ++i) {
        if (m_indices[i] != m_indices[i - 1]) {
            maybe_emit(i);
        }
    }

    if (write < capacity) {
        offsets[write] = M;
        experts[write] = -1;
    }
    ++write;

    return std::min(write, capacity);
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
        for (int64_t j = start; j < actual_end; ++j) {
            mi[j] = static_cast<int32_t>(i);
        }
        for (int64_t j = actual_end; j < aligned_end; ++j) {
            mi[j] = -1;
        }
        start = aligned_end;
    }

    *out_m = total_m;
    *out_active_m = active_m;
    return m_indices_cpu;
}

static float time_kernel_ms(const std::function<void()>& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / static_cast<float>(iters);
}

static double tflops_from_flops_and_ms(double flops, float ms) {
    return flops / (static_cast<double>(ms) * 1e-3) / 1e12;
}

static void print_tensor_bytes(const char* name, const torch::Tensor& t) {
    const int64_t bytes = t.numel() * t.element_size();
    const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
    std::cout << name << " bytes=" << bytes << " (" << mib << " MiB)\n";
}

static void print_int_vector(const char* name, const std::vector<int>& v) {
    std::cout << name << "=[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << v[i];
    }
    std::cout << "]\n";
}

static void print_first_bf16_values_cpu(const char* name, const torch::Tensor& t_cpu, int count) {
    auto flat = t_cpu.contiguous().view({-1});
    const auto* ptr = flat.data_ptr<at::BFloat16>();
    const int64_t n = std::min<int64_t>(flat.numel(), count);
    std::cout << name << " first " << n << " values=[";
    for (int64_t i = 0; i < n; ++i) {
        if (i) std::cout << ", ";
        std::cout << static_cast<float>(ptr[i]);
    }
    std::cout << "]\n";
}

static void print_bf16_block_cpu(
    const char* name,
    const torch::Tensor& t_cpu,
    int row0,
    int col0,
    int rows,
    int cols
) {
    auto t = t_cpu.contiguous();
    if (t.dim() != 2) {
        std::cout << name << " is not 2D\n";
        return;
    }
    const auto* ptr = t.data_ptr<at::BFloat16>();
    const auto stride0 = t.stride(0);
    const auto stride1 = t.stride(1);
    const int64_t max_r = std::min<int64_t>(t.size(0), row0 + rows);
    const int64_t max_c = std::min<int64_t>(t.size(1), col0 + cols);
    std::cout << name << " block rows [" << row0 << "," << max_r - 1
              << "], cols [" << col0 << "," << max_c - 1 << "]\n";
    for (int64_t r = row0; r < max_r; ++r) {
        std::cout << "  [";
        for (int64_t c = col0; c < max_c; ++c) {
            if (c != col0) std::cout << ", ";
            const auto v = static_cast<float>(ptr[r * stride0 + c * stride1]);
            std::cout << v;
        }
        std::cout << "]\n";
    }
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
    // 3) Tensors (your current demo settings) 7168
    // ----------------------------- 
    const int64_t n = 4096, k = 64, num_groups = 4;
    const int64_t expected_m_per_group = 2048;

    int64_t sf_k = 56;
    int64_t sf_n  = 32;

    const bool disable_ue8m0_cast = true;
    const std::string compiled_dims = "mnk";
    std::optional<std::tuple<int,int,int>> recipe = std::nullopt;

    int64_t m;
    int64_t active_m = 0;
    auto m_indices_cpu = build_m_indices_like_generators(expected_m_per_group, num_groups, &m, &active_m);
    auto m_indices = m_indices_cpu.to(dev);

    auto A_bp16 = torch::randn({m, k}, torch::TensorOptions().device(dev).dtype(torch::kBFloat16));
    auto B_bp16_cpu = torch::randn(
        {num_groups, n, k},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kBFloat16).pinned_memory(true)
    );
    print_first_bf16_values_cpu("B_bp16_cpu", B_bp16_cpu, 4);

    auto B_bp16 = B_bp16_cpu.to(dev, /*non_blocking=*/true);

    const int max_len = static_cast<int>(B_bp16.size(0)) + 1;
    std::vector<int> offsets_h(max_len);
    std::vector<int> experts_h(max_len);
    auto opts_i32_cuda = torch::TensorOptions().device(dev).dtype(torch::kInt32);
    auto offsets_t = torch::empty({max_len}, opts_i32_cuda);
    auto experts_t = torch::empty({max_len}, opts_i32_cuda);
    int list_size = fill_with_sentinel(m_indices_cpu.data_ptr<int>(), m_indices_cpu.numel(),
                                       offsets_h.data(), experts_h.data(), max_len);
    print_int_vector("offsets_h", offsets_h);
    print_int_vector("experts_h", experts_h);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    cudaMemcpyAsync(offsets_t.data_ptr<int>(), offsets_h.data(),
                    max_len * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(experts_t.data_ptr<int>(), experts_h.data(),
                    max_len * sizeof(int), cudaMemcpyHostToDevice, stream);

    auto D = torch::empty({m, n}, torch::TensorOptions().device(dev).dtype(torch::kBFloat16));

    auto AsymD = torch::empty({m, n}, torch::TensorOptions().device(dev).dtype(torch::kBFloat16));
    AsymD.fill_(std::numeric_limits<float>::quiet_NaN());

    // auto offsets_i32 = torch::tensor({0, 256, 512, 768, 1024},
    //                                 torch::TensorOptions().device(dev).dtype(torch::kInt32));
    // auto offsets_i32 = torch::tensor({0, 9984, 19456, 27264, 34304},
    //                                  torch::TensorOptions().device(dev).dtype(torch::kInt32));
    // auto experts_i32 = torch::tensor({0, 1, 2, 3, -1},
    //                                  torch::TensorOptions().device(dev).dtype(torch::kInt32));
    // const int list_size = static_cast<int>(offsets_t.numel());

    std::cout << "m=" << m << ", n=" << n << ", k=" << k << ", num_groups=" << num_groups << ",list_size=" << list_size << "\n";

    std::cout << "Calling asym_gemm::gemm::m_grouped_fp8_asym_gemm_nt_contiguous...\n";
    std::cout << "A=" << A_bp16.sizes() << " " << A_bp16.scalar_type() << "\n";
    std::cout << "B_bp16_cpu=" << B_bp16_cpu.sizes() << " " << B_bp16_cpu.scalar_type() << "\n";
    std::cout << "D=" << D.sizes() << " " << D.scalar_type() << "\n";
    print_tensor_bytes("A_bp16", A_bp16);
    print_tensor_bytes("B_bp16", B_bp16);
    print_tensor_bytes("B_bp16_cpu", B_bp16_cpu);
    print_tensor_bytes("D", D);

    const double aligned_flops = 2.0 * static_cast<double>(m) *
        static_cast<double>(n) * static_cast<double>(k);
    const double active_flops = 2.0 * static_cast<double>(active_m) *
        static_cast<double>(n) * static_cast<double>(k);

    // asym_gemm::gemm::m_grouped_bf16_gemm_nt_contiguous(
    //     A_bp16, B_bp16, D, m_indices, compiled_dims
    // );

    const int warmup = 5;
    const int iters = 10;

    auto gemm_ms = time_kernel_ms([&] {
        asym_gemm::gemm::m_grouped_bf16_gemm_nt_contiguous(
            A_bp16, B_bp16, D, m_indices, compiled_dims
        );
    }, warmup, iters);

  
    // auto opts_i32_cuda = torch::TensorOptions().device(dev).dtype(torch::kInt32);
    // auto offsets_t = torch::empty({max_len}, opts_i32_cuda);
    // auto experts_t = torch::empty({max_len}, opts_i32_cuda);
    // cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    // cudaMemcpyAsync(offsets_t.data_ptr<int>(), offsets_h.data(),
    //                 max_len * sizeof(int), cudaMemcpyHostToDevice, stream);
    // cudaMemcpyAsync(experts_t.data_ptr<int>(), experts_h.data(),
    //                 max_len * sizeof(int), cudaMemcpyHostToDevice, stream);

    auto AsymD_before_cpu = AsymD.to(torch::kCPU);
    auto asym_gemm_ms = time_kernel_ms([&] {
        asym_gemm::gemm::m_grouped_bf16_asym_gemm_nt_contiguous(
            A_bp16, B_bp16_cpu, AsymD, offsets_t, experts_t, list_size, compiled_dims
        );
    }, warmup, iters);

    const double gemm_tflops_aligned_m = tflops_from_flops_and_ms(aligned_flops, gemm_ms);
    const double gemm_tflops_active_m = tflops_from_flops_and_ms(active_flops, gemm_ms);
    const double asym_tflops_aligned_m = tflops_from_flops_and_ms(aligned_flops, asym_gemm_ms);
    const double asym_tflops_active_m = tflops_from_flops_and_ms(active_flops, asym_gemm_ms);

    std::cout << "Active M=" << active_m << " (aligned M=" << m << "), N=" << n << ", K=" << k << "\n";
    std::cout << "m_grouped_bf16_gemm_nt_contiguous: " << gemm_ms << " ms"
              << ", " << gemm_tflops_aligned_m << " TFLOP/s (aligned M)"
              << ", " << gemm_tflops_active_m << " TFLOP/s (active M)\n";
    std::cout << "m_grouped_bf16_asym_gemm_nt_contiguous: " << asym_gemm_ms << " ms"
              << ", " << asym_tflops_aligned_m << " TFLOP/s (aligned M)"
              << ", " << asym_tflops_active_m << " TFLOP/s (active M)\n";

    cudaDeviceSynchronize();
    auto D_cpu = D.to(torch::kCPU);
    auto AsymD_cpu = AsymD.to(torch::kCPU);
    auto diff_cpu = (D_cpu - AsymD_cpu).abs();
    const float diff_max = diff_cpu.max().item<float>();
    const float diff_mean = diff_cpu.mean().item<float>();
    std::cout << "D vs AsymD abs diff: max=" << diff_max
              << ", mean=" << diff_mean << "\n";
    auto unchanged_mask = AsymD_cpu.eq(AsymD_before_cpu);
    auto both_nan_mask = AsymD_cpu.isnan().bitwise_and(AsymD_before_cpu.isnan());
    auto unchanged_nan_aware_mask = unchanged_mask.bitwise_or(both_nan_mask);
    const int64_t unchanged_count = unchanged_nan_aware_mask.sum().item<int64_t>();
    const int64_t still_nan_count = AsymD_cpu.isnan().sum().item<int64_t>();
    const int64_t total_count = AsymD_cpu.numel();
    std::cout << "AsymD unchanged elements (NaN-aware): "
              << unchanged_count << " / " << total_count << "\n";
    std::cout << "AsymD still NaN after kernel: "
              << still_nan_count << " / " << total_count << "\n";

    const int block_rows = 4;
    const int block_cols = 8;
    print_bf16_block_cpu("D", D_cpu, 0, 0, block_rows, block_cols);
    print_bf16_block_cpu("AsymD", AsymD_cpu, 0, 0, block_rows, block_cols);
    print_bf16_block_cpu("abs(D-AsymD)", diff_cpu, 0, 0, block_rows, block_cols);
    // std::cout << "Done. D.mean=" << D.to(torch::kFloat32).mean().item<float>() << "\n";
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
