#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/cutlass.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <cstdint>

namespace py = pybind11;

// Your existing launcher
// void launch_MoECompute(
//     const void* A_ptr,
//     const void* B_ptr,
//     void* C_ptr,
//     int64_t M, int64_t N, int64_t K,
//     int64_t expert_size,
//     int64_t list_size,
//     const int32_t* expert_list_ptr,
//     const int32_t* index_list_ptr,
//     cudaStream_t stream);

void launch_MoECompute(cutlass::half_t* A, cutlass::half_t* B, cutlass::half_t* C, int M, int N, int K, int expert_size, int list_size, int* expert_list, int* index_list, cudaStream_t stream);

// -------- helpers to parse (Tensor) or (Tensor, Tensor) --------
static inline torch::Tensor parse_first_tensor(const py::handle& obj, const char* name) {
    if (py::isinstance<torch::Tensor>(obj)) return py::cast<torch::Tensor>(obj);
    if (py::isinstance<py::tuple>(obj)) {
        py::tuple t = py::cast<py::tuple>(obj);
        TORCH_CHECK(t.size() == 2, name, " must be Tensor or (Tensor, Tensor)");
        TORCH_CHECK(py::isinstance<torch::Tensor>(t[0]), name, "[0] must be Tensor");
        return py::cast<torch::Tensor>(t[0]);
    }
    TORCH_CHECK(false, name, " must be Tensor or (Tensor, Tensor)");
}

static inline py::handle unwrap_rhs_if_needed(const py::handle& rhs) {
    if (py::isinstance<torch::Tensor>(rhs) || py::isinstance<py::tuple>(rhs)) return rhs;
    // If you have a wrapper object, add rules here:
    if (py::hasattr(rhs, "weight")) return rhs.attr("weight");
    return rhs;
}

// -------- convert m_indices -> expert_list + index_list (CPU scan) --------
static inline void build_expert_and_offsets_cpu(
    const int32_t* m_idx,
    int64_t M,
    std::vector<int32_t>& expert_list,
    std::vector<int32_t>& index_list
) {
    expert_list.clear();
    index_list.clear();

    // Handle empty
    if (M <= 0) { index_list.push_back(0); return; }

    // Many pipelines pad with -1 at the end. If you do that, stop at first -1.
    int64_t M_eff = M;
    while (M_eff > 0 && m_idx[M_eff - 1] < 0) M_eff--;

    if (M_eff == 0) { index_list.push_back(0); return; }

    int32_t cur = m_idx[0];
    TORCH_CHECK(cur >= 0, "m_indices[0] must be >=0 (or you passed all padding)");

    expert_list.push_back(cur);
    index_list.push_back(0);

    for (int64_t i = 1; i < M_eff; ++i) {
        int32_t eid = m_idx[i];
        TORCH_CHECK(eid >= 0, "m_indices contains negative before end padding at i=", i);
        // contig assumption: expert id changes only at boundaries
        if (eid != cur) {
            // Optional: enforce monotonic nondecreasing if you expect expert-sorted packing
            // TORCH_CHECK(eid > cur, "m_indices is not sorted/contiguous-by-expert; need pack first");
            cur = eid;
            expert_list.push_back(cur);
            index_list.push_back((int32_t)i);
        }
    }
    index_list.push_back((int32_t)M_eff); // end offset
}

void grouped_gemm_nt_f8f8bf16_contig_cpp(
    torch::Tensor A,          // (hidden_states, hidden_states_scale)  -> we take hidden_states only for your kernel
    torch::Tensor A_s,          // (hidden_states, hidden_states_scale)  -> we take hidden_states only for your kernel
    torch::Tensor B,          // w13_weight_fp8 (we take weight tensor only for your kernel)
    torch::Tensor out,       // gateup_output
    torch::Tensor m_indices  // [M]
) {
    // Parse A from lhs, and B from rhs (weight only)
    // torch::Tensor A = parse_first_tensor(lhs, "lhs");
    // py::handle rhs_unwrapped = unwrap_rhs_if_needed(rhs);
    // TORCH_CHECK(py::isinstance<torch::Tensor>(rhs_unwrapped), "rhs must be a Tensor (or wrapper exposing .weight)");
    // torch::Tensor B = py::cast<torch::Tensor>(rhs_unwrapped);

    TORCH_CHECK(A.is_cuda() && B.is_cuda() && out.is_cuda() && m_indices.is_cuda(),
                "All tensors must be CUDA tensors");
    TORCH_CHECK(m_indices.scalar_type() == at::kInt, "m_indices must be int32");
    TORCH_CHECK(m_indices.is_contiguous(), "m_indices must be contiguous");
    TORCH_CHECK(A.dim() == 2, "A must be [M, K]");
    TORCH_CHECK(B.dim() >= 2, "B must have expert dim (e.g., [E, N, K] or similar)");
    TORCH_CHECK(out.dim() == 2, "out must be [M, N]");

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    // Adjust these if your B layout differs; many MoE weights are [E, N, K]
    int64_t expert_size = B.size(0);
    int64_t N = (B.dim() == 3) ? B.size(1) : B.size(1);  // customize as needed
    TORCH_CHECK(out.size(0) == M, "out.size(0) must equal A.size(0)");

    // ---- Build expert_list/index_list from m_indices (CPU scan) ----
    // Copy m_indices to CPU (test/demo path)
    torch::Tensor m_cpu = m_indices.to(at::kCPU, /*non_blocking=*/false);
    auto* m_ptr = m_cpu.data_ptr<int32_t>();

    std::vector<int32_t> expert_list_h;
    std::vector<int32_t> index_list_h;
    build_expert_and_offsets_cpu(m_ptr, M, expert_list_h, index_list_h);

    int64_t list_size = (int64_t)expert_list_h.size();
    TORCH_CHECK(index_list_h.size() == (size_t)(list_size + 1),
                "index_list must be list_size+1");

    // Copy expert_list/index_list to CUDA tensors
    auto opts = torch::TensorOptions().device(A.device()).dtype(torch::kInt32);
    torch::Tensor expert_list_tensor = torch::empty({list_size}, opts);
    torch::Tensor index_list_tensor  = torch::empty({list_size + 1}, opts);

    torch::Tensor expert_list_cpu = torch::from_blob(
        expert_list_h.data(), {list_size}, torch::TensorOptions().dtype(torch::kInt32)
    ).clone();
    torch::Tensor index_list_cpu = torch::from_blob(
        index_list_h.data(), {list_size + 1}, torch::TensorOptions().dtype(torch::kInt32)
    ).clone();

    expert_list_tensor.copy_(expert_list_cpu, /*non_blocking=*/false);
    index_list_tensor.copy_(index_list_cpu, /*non_blocking=*/false);

    const int32_t* expert_list_ptr = expert_list_tensor.data_ptr<int32_t>();
    const int32_t* index_list_ptr  = index_list_tensor.data_ptr<int32_t>();

    // ---- Call your existing launcher ----
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // NOTE: Your original code used cutlass::half_t pointers.
    // If you’re still using fp16/bf16, keep that. If you move to fp8, update types accordingly.
    const void* A_ptr = A.data_ptr();
    const void* B_ptr = B.data_ptr();
    void* C_ptr = out.data_ptr();

    launch_MoECompute(
        A_ptr, B_ptr, C_ptr,
        M, out.size(1), K,
        expert_size,
        list_size,
        expert_list_ptr,
        index_list_ptr,
        stream
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_gemm_nt_f8f8bf16_contig",
          &grouped_gemm_nt_f8f8bf16_contig_cpp,
          "Custom grouped_gemm_nt_f8f8bf16_contig((A, A_s), w, out, m_indices)");
}