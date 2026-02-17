# rm asymCompKernelMain
# /usr/local/cuda-12.9/bin/nvcc -arch=sm_90 -O3 asymCompKernelMain.cu -o asymCompKernelMain -I/sgl-workspace/sglang/sgl-kernel/csrc/moe/asymCompute_moe/flash-attention/csrc/flash_attn \
#     -I/sgl-workspace/sglang/sgl-kernel/csrc/moe/asymCompute_moe/flash-attention/csrc/flash_attn/src \
#     -I/sgl-workspace/sglang/AsymGEMM/third-party/cutlass/include \
#     -I/usr/local/lib/python3.12/pybind11/include \
#     -I/usr/lib/python3.12/site-packages/torch/include \
#     -I/usr/lib/python3.12/site-packages/torch/include/torch/csrc/api/include \
#     -L/usr/lib/python3.12/site-packages/torch/lib \
#     -L/usr/local/lib/python3.12/dist-packages/torch/lib \
#     -L/usr/local/cuda-12.9/lib64 \
#     -I/usr/include/python3.12 \
#     -I/sgl-workspace/sglang/AsymGEMM/asym_gemm/include \
#     -I/usr/local/lib/python3.12/dist-packages/torch/include \
#     -I/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include \
#     -L/usr/local/cuda/lib64 \
#     -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart \
#     -std=c++17 --expt-relaxed-constexpr --use_fast_math > compile.log 2>&1
rm asymCompKernelMain_bf16
/usr/local/cuda-12.9/bin/nvcc -arch=sm_90 -O3 asymCompKernelMain_bf16.cu -o asymCompKernelMain_bf16 \
  -I/sgl-workspace/sglang/sgl-kernel/csrc/moe/asymCompute_moe/flash-attention/csrc/flash_attn \
  -I/sgl-workspace/sglang/sgl-kernel/csrc/moe/asymCompute_moe/flash-attention/csrc/flash_attn/src \
  -I/sgl-workspace/sglang/AsymGEMM/third-party/cutlass/include \
  -I/sgl-workspace/sglang/AsymGEMM/asym_gemm/include \
  -I/usr/local/lib/python3.12/pybind11/include \
  -I/usr/lib/python3.12/site-packages/torch/include \
  -I/usr/lib/python3.12/site-packages/torch/include/torch/csrc/api/include \
  -I/usr/local/lib/python3.12/dist-packages/torch/include \
  -I/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include \
  -I/usr/include/python3.12 \
  -L/usr/lib/python3.12/site-packages/torch/lib \
  -L/usr/local/lib/python3.12/dist-packages/torch/lib \
  -L/usr/local/cuda-12.9/lib64 \
  -L/usr/local/cuda/lib64 \
  -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart \
  -lnvrtc \
  -Xlinker -rpath -Xlinker /usr/local/cuda-12.9/lib64 \
  -Xlinker -rpath -Xlinker /usr/lib/python3.12/site-packages/torch/lib \
  -Xlinker -rpath -Xlinker /usr/local/lib/python3.12/dist-packages/torch/lib \
  -std=c++17 --expt-relaxed-constexpr --use_fast_math \
  > compile.log 2>&1
