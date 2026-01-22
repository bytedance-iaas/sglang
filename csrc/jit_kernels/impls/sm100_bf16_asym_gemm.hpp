#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM100BF16GemmRuntime final: public LaunchRuntime<SM100BF16GemmRuntime> {
public:
    struct Args {
        int m, n, k, num_groups;
        const std::string& compiled_dims;

        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void* offsets;
        void* experts;
        int list_size;

        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_asym_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_asym_gemm_impl<
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
        args.gemm_config.tc_util);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.offsets, args.experts, args.list_size, 
            args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_cd));
    }
};

static void sm100_m_grouped_bf16_asym_gemm_contiguous(const torch::Tensor& a,
                                                 const torch::Tensor& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& m_indices,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                 const std::string& compiled_dims) {
    const auto& aligned_k = align(k, 64);
    const auto& config = get_best_config<SM100ArchSpec>(
        GemmType::MGroupedContiguous, KernelType::KernelNoSF,
        // NOTES: `num_groups` is 1, since the contiguous layout is seen as a whole
        m, n, k, 1, major_a, major_b,
        torch::kBFloat16, d.scalar_type(), false,
        device_runtime->get_num_sms());

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

    int max_len = (int)b.size(0);

    // 1) read m_indices on CPU
    auto m_indices_cpu = m_indices.to(torch::kCPU);
    auto* mi = m_indices_cpu.data_ptr<int>();
    int M = (int)m_indices_cpu.numel();

    // 2) build offsets/experts on CPU
    std::vector<int> offsets_h(max_len);
    std::vector<int> experts_h(max_len);
    int list_size = fill_with_sentinel(mi, m_indices.size(0), offsets_h.data(), experts_h.data(), max_len);
    // int list_size = fill_with_sentinel((int*)m_indices.data_ptr(), m_indices.size(0), offsets, experts, max_len);

    // 3) allocate offsets/experts on GPU (int32)
    auto opts_i32_cuda = torch::TensorOptions().device(a.device()).dtype(torch::kInt32);
    auto offsets_t = torch::empty({max_len}, opts_i32_cuda);
    auto experts_t = torch::empty({max_len}, opts_i32_cuda);

    // 4) copy host -> device (async on current stream)
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    cudaMemcpyAsync(offsets_t.data_ptr<int>(), offsets_h.data(),
                    max_len * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(experts_t.data_ptr<int>(), experts_h.data(),
                    max_len * sizeof(int), cudaMemcpyHostToDevice, stream);

    // 5) pass device pointers to kernel
    void* offsets = (void*)offsets_t.data_ptr<int>();
    void* experts = (void*)experts_t.data_ptr<int>();

    // Launch
    const SM100BF16GemmRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .offsets = offsets,
        .experts = experts,
        .list_size = list_size,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd
    };
    const auto& code = SM100BF16GemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_bf16_m_grouped_asym_gemm_contiguous", code);
    SM100BF16GemmRuntime::launch(runtime, args);
}

} // namespace deep_gemm
