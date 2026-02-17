#pragma once

#include <torch/python.h>
#include <vector>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"

#include "epilogue.hpp"
#include "runtime_utils.hpp"

namespace asym_gemm {

class SM100FP8AsymGemm1D1DRuntime final: public LaunchRuntime<SM100FP8AsymGemm1D1DRuntime> {
public:
    struct Args {
        int m, n, k, num_groups;
        const std::string& compiled_dims;
        const std::optional<std::string>& epilogue_type;

        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void* offsets;
        void* experts;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <asym_gemm/impls/sm100_fp8_asym_gemm_1d1d.cuh>

using namespace asym_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_asym_gemm_1d1d_impl<
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {}
    >);
}};
)",
        to_string(args.gemm_config.major_a), to_string(args.gemm_config.major_b),
        get_compiled_dim(args.m, 'm', args.compiled_dims), get_compiled_dim(args.n, 'n', args.compiled_dims), get_compiled_dim(args.k, 'k', args.compiled_dims),
        args.gemm_config.block_m, args.gemm_config.block_n, args.gemm_config.block_k,
        args.num_groups,
        args.gemm_config.smem_config.swizzle_a_mode, args.gemm_config.smem_config.swizzle_b_mode, args.gemm_config.smem_config.swizzle_cd_mode,
        args.gemm_config.num_stages,
        args.gemm_config.thread_config.num_non_epilogue_threads, args.gemm_config.thread_config.num_epilogue_threads,
        args.gemm_config.multicast_config.num_multicast, args.gemm_config.multicast_config.is_multicast_on_a,
        args.gemm_config.num_sms,
        to_string(args.gemm_config.gemm_type), args.gemm_config.with_accumulation, to_string(args.gemm_config.cd_dtype),
        get_default_epilogue_type(args.epilogue_type));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.offsets, args.experts, args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd));
    }
};

static int build_offsets_experts_from_indices(const int* m_indices, const int m,
                                              int* offsets, int* experts, const int capacity) {
    if (!offsets || !experts || capacity <= 0 || m <= 0 || !m_indices)
        return 0;

    int write = 0;
    auto maybe_emit = [&](int start_idx) {
        const int e = m_indices[start_idx];
        if (e != -1) {
            if (write < capacity) {
                offsets[write] = start_idx;
                experts[write] = e;
            }
            ++write;
        }
    };

    maybe_emit(0);
    for (int i = 1; i < m; ++i) {
        if (m_indices[i] != m_indices[i - 1])
            maybe_emit(i);
    }

    // Sentinel
    if (write < capacity) {
        offsets[write] = m;
        experts[write] = -1;
    }
    ++write;
    return std::min(write, capacity);
}

static void sm100_m_grouped_fp8_asym_gemm_contiguous_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                     const torch::Tensor& b, const torch::Tensor& sfb,
                                                     const torch::Tensor& d,
                                                     const torch::Tensor& m_indices,
                                                     const int& num_groups, const int& m, const int& n, const int& k,
                                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                     const std::string& compiled_dims) {
    const auto& aligned_k = align(k, 128);
    const auto& config = get_best_config<SM100ArchSpec>(
        GemmType::MGroupedContiguous, KernelType::Kernel1D1D,
        // NOTES: `num_groups` is 1, since the contiguous layout is seen as a whole
        m, n, k, 1, major_a, major_b,
        torch::kFloat8_e4m3fn, d.scalar_type(), false,
        device_runtime->get_num_sms());

    // Create tensor descriptors
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), 1,
                                                 config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, config.block_k, 1, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                  config.block_n, config.block_k, num_groups, 0);

    // Build offsets/experts from m_indices for B-centric scheduling.
    const auto m_indices_cpu = m_indices.to(torch::kCPU);
    auto* mi = m_indices_cpu.data_ptr<int>();
    const int max_len = static_cast<int>(m_indices_cpu.numel()) + 1;
    std::vector<int> offsets_h(max_len);
    std::vector<int> experts_h(max_len);
    const int list_size = build_offsets_experts_from_indices(mi, m, offsets_h.data(), experts_h.data(), max_len);
    if (list_size <= 1)
        return;

    auto opts_i32_cuda = torch::TensorOptions().device(a.device()).dtype(torch::kInt32);
    auto offsets_t = torch::empty({max_len}, opts_i32_cuda);
    auto experts_t = torch::empty({max_len}, opts_i32_cuda);
    auto offsets_cpu_t = torch::from_blob(offsets_h.data(), {max_len}, torch::TensorOptions().dtype(torch::kInt32)).clone();
    auto experts_cpu_t = torch::from_blob(experts_h.data(), {max_len}, torch::TensorOptions().dtype(torch::kInt32)).clone();
    offsets_t.copy_(offsets_cpu_t, /*non_blocking=*/false);
    experts_t.copy_(experts_cpu_t, /*non_blocking=*/false);

    // Launch kernel
    const SM100FP8AsymGemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .epilogue_type = std::nullopt,
        .gemm_config = config,
        .launch_args = LaunchArgs({ceil_div(n, config.block_n), list_size - 1}, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .offsets = offsets_t.data_ptr<int>(),
        .experts = experts_t.data_ptr<int>(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };
    const auto& code = SM100FP8AsymGemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_m_grouped_fp8_asym_gemm_contiguous_1d1d", code);
    SM100FP8AsymGemm1D1DRuntime::launch(runtime, args);
}

} // namespace asym_gemm
