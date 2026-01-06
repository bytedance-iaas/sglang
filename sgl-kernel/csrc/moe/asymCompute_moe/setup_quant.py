from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define the extension
setup(
    name="fp8_quant_ext",
    ext_modules=[
        CUDAExtension(
            name="fp8_quant_ext",
            sources=["dynamic_fp8_quant.cpp", "fp8_quant_kernel.cu"],  # Assuming .cpp; use .cu if renamed
            include_dirs=[
                "/sgl-workspace/sglang/sgl-kernel/csrc/moe/asymCompute_moe/flash-attention/csrc/flash_attn/src",
                "/sgl-workspace/sglang/sgl-kernel/csrc/moe/asymCompute_moe/flash-attention/csrc/cutlass/include",
                "/usr/local/lib/python3.12/pybind11/include",
                "/usr/lib/python3.12/site-packages/torch/include",
                "/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include",
                "/usr/include/python3.12",
            ],
            library_dirs=[
                "/usr/local/cuda/lib64",
            ],
            libraries=[
                "torch",
                "torch_cpu",
                "torch_cuda",
                "c10",
                "c10_cuda",
                "cudart",
            ],
            extra_compile_args={
                "cxx": ["-O2", "-std=c++17", "-fpermissive"],
                "nvcc": [
                    "-O2",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--use_fast_math",
                    "--extended-lambda",
                    "-arch=sm_100",  # Adjust to your GPU (sm_90 from your latest command)
                ],
            },
            extra_link_args=[
                "-Wl,-rpath,/usr/local/lib/python3.12/dist-packages/torch/lib",
                "-Wl,-rpath,/usr/local/cuda/lib64",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)