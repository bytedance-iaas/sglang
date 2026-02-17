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
#include <cstring>

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

static std::optional<int64_t> parse_fixed_m_arg(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fixed-m") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after --fixed-m\n";
                return std::nullopt;
            }
            return std::stoll(argv[i + 1]);
        }
        constexpr const char* kPrefix = "--fixed-m=";
        if (std::strncmp(argv[i], kPrefix, std::strlen(kPrefix)) == 0) {
            return std::stoll(argv[i] + std::strlen(kPrefix));
        }
    }
    return std::nullopt;
}

int main(int argc, char** argv) {
    // JIT cache + lineinfo (optional but useful)
    // setenv("DG_JIT_CACHE_DIR", "/tmp/deepgemm_jit", 1);
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
    // const int64_t n = 4096, k = 7168, num_groups = 4;
    const int64_t n = 256, k = 64, num_groups = 4;
    const int64_t expected_m_per_group = 2048;

    int64_t sf_k = 56;
    int64_t sf_n  = 32;

    const bool disable_ue8m0_cast = true;
    const std::string compiled_dims = "mnk";
    std::optional<std::tuple<int,int,int>> recipe = std::nullopt;

    auto fixed_m = parse_fixed_m_arg(argc, argv);
    int64_t m = 0;
    int64_t active_m = 0;
    torch::Tensor m_indices_cpu;
    if (fixed_m.has_value()) {
        if (*fixed_m <= 0) {
            std::cerr << "--fixed-m must be > 0\n";
            return 1;
        }
        m = *fixed_m;
        active_m = m;
        m_indices_cpu = torch::zeros(
            {m},
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32)
        );
        std::cout << "Using fixed m=" << m << " (all rows mapped to expert 0)\n";
    } else {
        m_indices_cpu = build_m_indices_like_generators(expected_m_per_group, num_groups, &m, &active_m);
    }
    auto m_indices = m_indices_cpu.to(dev);

    constexpr float kABase = 0.1f;
    constexpr float kBBase = 0.2f;
    constexpr float kStep = 1e-2f;

    auto A_bp16 = (
        torch::arange(
            1,
            m * k + 1,
            torch::TensorOptions().device(dev).dtype(torch::kFloat32)
        ).view({m, k}) * kStep + kABase
    ).to(torch::kBFloat16);

    auto B_bp16_cpu = (
        torch::arange(
            1,
            num_groups * n * k + 1,
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)
        ).view({num_groups, n, k}) * kStep + kBBase
    ).to(torch::kBFloat16).pin_memory();
    auto B_bp16 = B_bp16_cpu.to(dev, /*non_blocking=*/true);

    auto A_bp16_cpu = A_bp16.to(torch::kCPU);
    print_bf16_block_cpu("A_bp16", A_bp16_cpu, 0, 0, 5, 5);
    print_bf16_block_cpu("B_bp16_cpu[group=0]", B_bp16_cpu.select(0, 0), 0, 0, 5, 5);

    const int max_len = static_cast<int>(B_bp16.size(0)) + 1;
    std::vector<int> offsets_h(max_len);
    std::vector<int> experts_h(max_len);
    auto opts_i32_cuda = torch::TensorOptions().device(dev).dtype(torch::kInt32);
    auto offsets_t = torch::empty({max_len}, opts_i32_cuda);
    auto experts_t = torch::empty({max_len}, opts_i32_cuda);
    int list_size = fill_with_sentinel(m_indices_cpu.data_ptr<int>(), m_indices_cpu.numel(),
                                       offsets_h.data(), experts_h.data(), max_len);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    cudaMemcpyAsync(offsets_t.data_ptr<int>(), offsets_h.data(),
                    max_len * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(experts_t.data_ptr<int>(), experts_h.data(),
                    max_len * sizeof(int), cudaMemcpyHostToDevice, stream);

    auto D = torch::empty({m, n}, torch::TensorOptions().device(dev).dtype(torch::kBFloat16));
    // auto offsets_i32 = torch::tensor({0, 256, 512, 768, 1024},
    //                                 torch::TensorOptions().device(dev).dtype(torch::kInt32));
    
    auto run_kernel = [&]() {
        asym_gemm::gemm::m_grouped_bf16_asym_gemm_nt_contiguous(
            A_bp16, B_bp16_cpu, D, offsets_t, experts_t, list_size, compiled_dims
        );
    };

   asym_gemm::gemm::m_grouped_bf16_asym_gemm_nt_contiguous(
        A_bp16, B_bp16_cpu, D, offsets_t, experts_t, list_size, compiled_dims
    );

    asym_gemm::gemm::m_grouped_bf16_gemm_nt_contiguous(
        A_bp16, B_bp16, D, m_indices, compiled_dims
    );
    // constexpr int warmup = 10;
    // constexpr int iters = 50;
    // const float avg_ms = time_kernel_ms(run_kernel, warmup, iters);

    // const double flops_active = 2.0 * static_cast<double>(active_m) * static_cast<double>(n) * static_cast<double>(k);
    // const double flops_padded = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    // const double tflops_active = tflops_from_flops_and_ms(flops_active, avg_ms);
    // const double tflops_padded = tflops_from_flops_and_ms(flops_padded, avg_ms);
    cudaDeviceSynchronize();

    // std::cout << "m=" << m << ", n=" << n << ", k=" << k << ", num_groups=" << num_groups << ",list_size=" << list_size << "\n";
    // std::cout << "active_m=" << active_m << " (active ratio=" << (100.0 * static_cast<double>(active_m) / static_cast<double>(m)) << "%)\n";
    // std::cout << "bench avg_ms=" << avg_ms << " (" << warmup << " warmup, " << iters << " iters)\n";
    // std::cout << "est TFLOPS(active_m)=" << tflops_active << "\n";
    // std::cout << "est TFLOPS(padded_m)=" << tflops_padded << "\n";

    // std::cout << "A=" << A_bp16.sizes() << " " << A_bp16.scalar_type() << "\n";
    // std::cout << "B_bp16_cpu=" << B_bp16_cpu.sizes() << " " << B_bp16_cpu.scalar_type() << "\n";
    // std::cout << "D=" << D.sizes() << " " << D.scalar_type() << "\n";
    // print_tensor_bytes("A_bp16", A_bp16);
    // print_tensor_bytes("B_bp16", B_bp16);
    // print_tensor_bytes("B_bp16_cpu", B_bp16_cpu);
    // print_tensor_bytes("D", D);

    return 0;
}
