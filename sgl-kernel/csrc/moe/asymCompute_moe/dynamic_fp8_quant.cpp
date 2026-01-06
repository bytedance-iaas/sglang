// dynamic_fp8_quant.cpp
#include <torch/extension.h>
#include <vector>

void dynamic_scaled_fp8_quant_cuda(
    const torch::Tensor& x,
    torch::Tensor& y,
    torch::Tensor& scale,
    double fp8_max);

static void dynamic_scaled_fp8_quant(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor scale,
    double fp8_max) {

  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
  TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");

  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
  TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");

  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "x must be fp16 or bf16, got ", x.scalar_type());

  TORCH_CHECK(y.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              "y must be torch.float8_e4m3fn, got ", y.scalar_type());

  TORCH_CHECK(scale.scalar_type() == at::kFloat,
              "scale must be torch.float32, got ", scale.scalar_type());

  TORCH_CHECK(scale.numel() == 1, "scale must have numel()==1, got ", scale.numel());
  TORCH_CHECK(x.numel() == y.numel(),
              "x and y must have same numel: ", x.numel(), " vs ", y.numel());

  dynamic_scaled_fp8_quant_cuda(x, y, scale, fp8_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dynamic_scaled_fp8_quant", &dynamic_scaled_fp8_quant,
        "Dynamic per-tensor FP8 quant OUT (CUDA): writes y_fp8 and scale[0]",
        py::arg("x"), py::arg("y"), py::arg("scale"), py::arg("fp8_max"));
}