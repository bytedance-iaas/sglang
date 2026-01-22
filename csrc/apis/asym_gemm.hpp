#pragma once

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_fp8_asym_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_bf16_asym_gemm.hpp"
#endif 

#include "../jit_kernels/impls/smxx_cublaslt.hpp"

#include "layout.hpp"

namespace deep_gemm::gemm {

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
static void m_grouped_fp8_asym_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& m_indices,
                                             std::optional<std::tuple<int, int, int>> recipe,
                                             const std::string& compiled_dims,
                                             const bool& disable_ue8m0_cast) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    if (fp8_requires_k_major())
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(m_indices.is_contiguous());

    // Type and shape checks
    const auto& [m, k] = get_shape<2>(a.first);
    const auto& [num_groups, n, k_] = get_shape<3>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);
    const auto& m__ = static_cast<int>(m_indices.numel());
    DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Transform SFA and SFB into compute-required layout
    if (not recipe.has_value())
        recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    const auto& sfa = layout::transform_sf_into_required_layout(a.second, m, k, recipe.value(), std::nullopt,  true, disable_ue8m0_cast);
    const auto& sfb = layout::transform_sf_into_required_layout(b.second, n, k, recipe.value(),   num_groups, false, disable_ue8m0_cast);

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    sm100_m_grouped_fp8_asym_gemm_contiguous_1d1d(a.first, sfa, b.first, sfb, d, m_indices,
                                                 num_groups, m, n, k, major_a, major_b, compiled_dims);
}

static void m_grouped_bf16_asym_gemm_nt_contiguous(const torch::Tensor& a, const torch::Tensor& b,
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
    sm100_m_grouped_bf16_asym_gemm_contiguous(a, b, d, m_indices,
                                            num_groups, m, n, k, major_a, major_b, compiled_dims);
}

#endif

static void register_apis(pybind11::module_& m) {
    #if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
        // FP8 GEMMs
        m.def("m_grouped_fp8_asym_gemm_nt_contiguous", &m_grouped_fp8_asym_gemm_nt_contiguous,
            py::arg("a"), py::arg("b"), py::arg("d"), py::arg("m_indices"),
            py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk",
            py::arg("disable_ue8m0_cast") = false);
    #endif
}

} // namespace deep_gemm::gemm
