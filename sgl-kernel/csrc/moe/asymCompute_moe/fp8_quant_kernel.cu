// dynamic_fp8_quant.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>   // provides __nv_fp8_e4m3 (CUDA 12+)

#include <stdint.h>
#include <cmath>

__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ uint8_t float_to_e4m3fn_u8(float v) {
  // Saturating cast to FP8 E4M3FN, store raw 1-byte representation.
  union {
    __nv_fp8_e4m3 fp8;
    uint8_t u8;
  } u;
  u.fp8 = __nv_fp8_e4m3(v);
  return u.u8;
}

// atomicMax for float using atomicCAS on int bits
__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
  int* addr_as_i = reinterpret_cast<int*>(addr);
  int old = *addr_as_i;
  int assumed;
  while (true) {
    assumed = old;
    float old_f = __int_as_float(assumed);
    float new_f = fmaxf(old_f, val);
    int new_i = __float_as_int(new_f);
    old = atomicCAS(addr_as_i, assumed, new_i);
    if (old == assumed) break;
  }
}

template <typename scalar_t>
__global__ void absmax_kernel(const scalar_t* __restrict__ x, int64_t n, float* __restrict__ out) {
  float local = 0.0f;
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;

  for (int64_t i = tid; i < n; i += stride) {
    float v = fabsf(to_float(x[i]));
    local = fmaxf(local, v);
  }

  // block reduce max
  __shared__ float smem[256];
  smem[threadIdx.x] = local;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + offset]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicMaxFloat(out, smem[0]);
  }
}

__global__ void compute_scale_kernel(float* scale, double fp8_max) {
  float amax = scale[0];
  float s = amax / (float)fp8_max;
  if (s < 1e-12f) s = 1e-12f;
  scale[0] = s;
}

template <typename scalar_t>
__global__ void quant_kernel(const scalar_t* __restrict__ x, uint8_t* __restrict__ y_u8, int64_t n, const float* __restrict__ scale) {
  float inv = 1.0f / scale[0];
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = to_float(x[i]) * inv;
    y_u8[i] = float_to_e4m3fn_u8(v);
  }
}

void dynamic_scaled_fp8_quant_cuda(
    const torch::Tensor& x,
    torch::Tensor& y,
    torch::Tensor& scale,
    double fp8_max) {

  const int64_t n = x.numel();
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  // Use scale[0] as temp amax accumulator: set to 0 first
  // scale is float32 scalar on CUDA
  cudaMemsetAsync(scale.data_ptr<float>(), 0, sizeof(float), stream);

  const int threads = 256;
  const int blocks_abs = (int)std::min<int64_t>((n + threads - 1) / threads, 4096);

  if (x.scalar_type() == at::kHalf) {
    absmax_kernel<<<blocks_abs, threads, 0, stream>>>(
        (const __half*)x.data_ptr<at::Half>(), n, scale.data_ptr<float>());
    compute_scale_kernel<<<1, 1, 0, stream>>>(scale.data_ptr<float>(), fp8_max);

    // y is float8 tensor, write raw bytes
    uint8_t* y_u8 = (uint8_t*)y.data_ptr();
    int blocks_q = (int)((n + threads - 1) / threads);
    quant_kernel<<<blocks_q, threads, 0, stream>>>(
        (const __half*)x.data_ptr<at::Half>(), y_u8, n, scale.data_ptr<float>());

  } else { // bf16
    absmax_kernel<<<blocks_abs, threads, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(), n, scale.data_ptr<float>());
    compute_scale_kernel<<<1, 1, 0, stream>>>(scale.data_ptr<float>(), fp8_max);

    uint8_t* y_u8 = (uint8_t*)y.data_ptr();
    int blocks_q = (int)((n + threads - 1) / threads);
    quant_kernel<<<blocks_q, threads, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(), y_u8, n, scale.data_ptr<float>());
  }

  // Optional: add CUDA error checks here if you want
}