#pragma once

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_bf16_asym_gemm.hpp"
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm100_fp8_asym_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp"
#endif 

#include "../jit_kernels/impls/smxx_cublaslt.hpp"

#include "layout.hpp"

namespace asym_gemm::gemm {

static bool early_return(const int& m, const int &n, const int& k,
                         const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    // Do nothing if the problem is empty
    if (m == 0 or n == 0)
        return true;

    // Checks
    const bool& is_cd_same = c.has_value() and c->data_ptr() == d.data_ptr();
    if (is_cd_same)
        DG_HOST_ASSERT(c->sizes() == d.sizes() and c->strides() == d.strides());
    if (c.has_value()) {
        check_major_type_cd(c.value());
        DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(c.value().scalar_type() == torch::kFloat);
    }

    // No accumulation
    if (k == 0) {
        if (not is_cd_same)
            c.has_value() ? d.copy_(c.value()) : d.zero_();
        return true;
    }

    // With accumulation, do copy before GEMM (assuming the GEMM kernel does not support different C/D)
    if (c.has_value() and not is_cd_same)
        d.copy_(c.value());
    return false;
}

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE

#endif

#if DG_TENSORMAP_COMPATIBLE
static void m_grouped_bf16_asym_gemm_nt_contiguous(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d,
                                              const torch::Tensor& offsets, const torch::Tensor& experts,
                                              const int& list_size,
                                              const std::string& compiled_dims) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);

    // Type and shape checks
    const auto& [m, k] = get_shape<2>(a);
    const auto& [num_groups, n, k_] = get_shape<3>(b);
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    DG_HOST_ASSERT(offsets.is_cuda() && experts.is_cuda());
    DG_HOST_ASSERT(offsets.is_contiguous() && experts.is_contiguous());
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt && experts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(offsets.numel() >= list_size && experts.numel() >= list_size);

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    const auto& arch_major = device_runtime->get_arch_major();
    sm100_m_grouped_bf16_asym_gemm_contiguous(a, b, d,
                                            offsets, experts, list_size,
                                            num_groups, m, n, k, major_a, major_b, compiled_dims);
}

static void m_grouped_bf16_gemm_nt_contiguous(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d, const torch::Tensor& m_indices,
                                              const std::string& compiled_dims) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(m_indices.is_contiguous());

    // Type and shape checks
    const auto& [m, k] = get_shape<2>(a);
    const auto& [num_groups, n, k_] = get_shape<3>(b);
    const auto& [m_, n_] = get_shape<2>(d);
    const auto& m__ = static_cast<int>(m_indices.numel());
    DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    sm100_m_grouped_bf16_gemm_contiguous(a, b, d, m_indices,
                                            num_groups, m, n, k, major_a, major_b, compiled_dims);
}
#endif

static void register_apis(pybind11::module_& m) {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    // FP8 GEMMs
#endif

#if DG_TENSORMAP_COMPATIBLE
    // BF16 GEMMs
    m.def("m_grouped_bf16_asym_gemm_nt_contiguous",
          static_cast<void(*)(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                              const torch::Tensor&, const torch::Tensor&, const int&, const std::string&)>(
              &m_grouped_bf16_asym_gemm_nt_contiguous),
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("offsets"), py::arg("experts"), py::arg("list_size"),
          py::arg("compiled_dims") = "nk");
    m.def("m_grouped_bf16_gemm_nt_contiguous", &m_grouped_bf16_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("m_indices"),
          py::arg("compiled_dims") = "nk");
#endif
}

} // namespace asym_gemm::gemm
