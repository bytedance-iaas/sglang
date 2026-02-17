#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "apis/gemm.hpp"
// #include "apis/asym_gemm.hpp"
#include "apis/layout.hpp"
#include "apis/runtime.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";
    asym_gemm::gemm::register_apis(m);
    asym_gemm::layout::register_apis(m);
    asym_gemm::runtime::register_apis(m);
}
