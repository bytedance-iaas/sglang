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
#include "runtime_utils.hpp"

namespace asym_gemm {

class SM100BF16AsymGemmRuntime final: public LaunchRuntime<SM100BF16AsymGemmRuntime> {
public:
    struct Args {
        int m, n, k, num_groups;
        const std::string& compiled_dims;

        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void* offsets;
        void* experts;

        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <asym_gemm/impls/sm100_bf16_asym_gemm.cuh>

using namespace asym_gemm;

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
            args.offsets, args.experts, 
            args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_cd));
    }
};

// int fill_with_sentinel(
//     int* m_indices, int M,
//     int* offsets, int* experts, int capacity
// ) {
//     if (!offsets || !experts || capacity <= 0) return 0;

//     if (M <= 0 || !m_indices) {
//         return 0;
//     }

//     int write = 0;

//     auto maybe_emit = [&](int start_idx) {
//         int e = m_indices[start_idx];
//         if (e != -1) {
//             if (write < capacity) {
//                 offsets[write] = start_idx;
//                 experts[write] = e;
//             }
//             ++write;
//         }
//     };

//     maybe_emit(0);
//     for (int i = 1; i < M; ++i) {
//         if (m_indices[i] != m_indices[i - 1]) {
//             maybe_emit(i);
//         }
//     }

//     // Append sentinel: (M, -1)
//     if (write < capacity) {
//         offsets[write] = M;
//         experts[write] = -1;
//     }
//     ++write;

//     return std::min(write, capacity);
// }

static void sm100_m_grouped_bf16_asym_gemm_contiguous(const torch::Tensor& a,
                                                 const torch::Tensor& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& offsets_t,
                                                 const torch::Tensor& experts_t,
                                                 const int& list_size,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                 const std::string& compiled_dims) {
    const auto& aligned_k = align(k, 64);
    const int block_m = 128;
    // SM100 legality requires block_n <= 128 when k <= 256.
    // const int block_n = (k <= 256) ? 128 : 256;
    // const int block_k = 64;

    const int block_n = 128;
    const int block_k = 64;

    const bool use_manual_config = block_m > 0 or block_n > 0 or block_k > 0;
    if (use_manual_config)
        DG_HOST_ASSERT(block_m > 0 and block_n > 0 and block_k > 0);
    const auto& config = use_manual_config
        ? get_manual_config_asym<SM100ArchSpec>(
            GemmType::MGroupedContiguous, KernelType::KernelNoSF,
            // NOTES: `num_groups` is 1, since the contiguous layout is seen as a whole
            m, n, k, 1, major_a, major_b,
            torch::kBFloat16, d.scalar_type(), false,
            device_runtime->get_num_sms(),
            block_m, block_n, block_k)
        : get_best_config_asym<SM100ArchSpec>(
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

    int max_len = (int)b.size(0) + 1;

    // 1) read m_indices on CPU
    // auto m_indices_cpu = m_indices.to(torch::kCPU);
    // auto* mi = m_indices_cpu.data_ptr<int>();
    // int M = (int)m_indices_cpu.numel();

    // 2) build offsets/experts on CPU
    // std::vector<int> offsets_h(max_len);
    // std::vector<int> experts_h(max_len);
    // int list_size = fill_with_sentinel(mi, m_indices.size(0), offsets_h.data(), experts_h.data(), max_len);
    // (void)list_size;

    // // 3) allocate offsets/experts on GPU (int32)
    // auto opts_i32_cuda = torch::TensorOptions().device(a.device()).dtype(torch::kInt32);
    // auto offsets_t = torch::empty({max_len}, opts_i32_cuda);
    // auto experts_t = torch::empty({max_len}, opts_i32_cuda);

    // // 4) copy host -> device (async on current stream)
    // cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    // cudaMemcpyAsync(offsets_t.data_ptr<int>(), offsets_h.data(),
    //                 max_len * sizeof(int), cudaMemcpyHostToDevice, stream);
    // cudaMemcpyAsync(experts_t.data_ptr<int>(), experts_h.data(),
    //                 max_len * sizeof(int), cudaMemcpyHostToDevice, stream);

    // // 5) pass device pointers to kernel
    // void* offsets = (void*)offsets_t.data_ptr<int>();
    // void* experts = (void*)experts_t.data_ptr<int>();

    // std::cout << "list_size = " << list_size << " (max_len=" << max_len << ")\n";
    // for (int i = 0; i < list_size; ++i) {
    //     std::cout << "pair[" << i << "]: offset=" << offsets_h[i]
    //             << ", expert=" << experts_h[i] << "\n";
    // }

    // printf("ceil_div(n, config.block_n): %d, num_groups: %d \n", ceil_div(n, config.block_n), num_groups);

    // Launch
    const SM100BF16AsymGemmRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs({ceil_div(n, config.block_n), list_size - 1}, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .offsets = offsets_t.data_ptr<int>(),
        .experts = experts_t.data_ptr<int>(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd
    };
    const auto& code = SM100BF16AsymGemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_bf16_m_grouped_asym_gemm_contiguous", code);
    SM100BF16AsymGemmRuntime::launch(runtime, args);
}

// static void sm100_m_grouped_bf16_asym_gemm_contiguous_with_offsets(const torch::Tensor& a,
//                                                 const torch::Tensor& b,
//                                                 const torch::Tensor& d,
//                                                 const torch::Tensor& m_indices,
//                                                 const torch::Tensor& offsets_t,
//                                                 const torch::Tensor& experts_t,
//                                                 const int& list_size,
//                                                 const int& num_groups, const int& m, const int& n, const int& k,
//                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
//                                                 const std::string& compiled_dims) {
//     const auto& aligned_k = align(k, 64);
//     const auto& config = get_best_config_asym<SM100ArchSpec>(
//         GemmType::MGroupedContiguous, KernelType::KernelNoSF,
//         // NOTES: `num_groups` is 1, since the contiguous layout is seen as a whole
//         m, n, k, 1, major_a, major_b,
//         torch::kBFloat16, d.scalar_type(), false,
//         device_runtime->get_num_sms());

//     const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
//                                                SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
//                                                config.block_k,
//                                                static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
//                                                config.smem_config.swizzle_a_mode);
//     const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
//                                                SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
//                                                config.block_k,
//                                                static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
//                                                config.smem_config.swizzle_b_mode);
//     const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
//                                                  SM100ArchSpec::get_cd_store_block_m(config.block_m),
//                                                  SM100ArchSpec::get_cd_store_block_n(config.block_n),
//                                                  static_cast<int>(d.stride(-2)), 1,
//                                                  config.smem_config.swizzle_cd_mode);

//     DG_HOST_ASSERT(offsets_t.is_cuda() && experts_t.is_cuda());
//     DG_HOST_ASSERT(offsets_t.is_contiguous() && experts_t.is_contiguous());
//     DG_HOST_ASSERT(offsets_t.scalar_type() == torch::kInt && experts_t.scalar_type() == torch::kInt);
//     DG_HOST_ASSERT(offsets_t.numel() >= list_size && experts_t.numel() >= list_size);

//     void* offsets = (void*)offsets_t.data_ptr<int>();
//     void* experts = (void*)experts_t.data_ptr<int>();

//     const SM100BF16AsymGemmRuntime::Args& args = {
//         .m = m, .n = n, .k = aligned_k,
//         .compiled_dims = compiled_dims,
//         .gemm_config = config,
//         .launch_args = LaunchArgs({ceil_div(n, config.block_n), num_groups}, config.thread_config.num_threads,
//                                   config.smem_config.smem_size,
//                                   config.multicast_config.num_multicast),
//         .offsets = offsets,
//         .experts = experts,
//         .tensor_map_a = tensor_map_a,
//         .tensor_map_b = tensor_map_b,
//         .tensor_map_cd = tensor_map_cd
//     };
//     const auto& code = SM100BF16AsymGemmRuntime::generate(args);
//     const auto& runtime = compiler->build("sm100_bf16_m_grouped_asym_gemm_contiguous_with_offsets", code);
//     SM100BF16AsymGemmRuntime::launch(runtime, args);
// }

} // namespace asym_gemm
