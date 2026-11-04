#pragma once
#include <cuda_runtime.h>

namespace iaas_kernels {
// Continuous FP32 activation scale. SM90 WGMMA has no hardware block-scale operand (the SF
// is a plain FFMA in the epilogue), so the previous UE8M0 (power-of-two) scale bought nothing
// on SM90 and only cost precision; the SF pool is already fp32, so this is byte/layout neutral.
// clamp amax before the reciprocal: padded rows have amax==0, and 448/0=inf -> 0*inf=NaN.
__forceinline__ __device__ void
sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(const float2& amax, float2& sf, float2& sf_inv) {
  constexpr float kScale = 1.0f / 448.0f;
  const auto ax = fmaxf(amax.x, 1e-10f);
  const auto ay = fmaxf(amax.y, 1e-10f);
  sf.x = __fmul_rn(ax, kScale), sf_inv.x = 1.0f / sf.x;
  sf.y = __fmul_rn(ay, kScale), sf_inv.y = 1.0f / sf.y;
}

}  // namespace iaas_kernels
