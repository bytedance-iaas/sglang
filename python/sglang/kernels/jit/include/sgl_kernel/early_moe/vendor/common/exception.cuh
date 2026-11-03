#pragma once

#include <sgl_kernel/early_moe/vendor/common/compile.cuh>

#include <cuda/std/cstdint>

#ifdef __CLION_IDE__

CUTLASS_HOST_DEVICE void host_device_printf(const char* format, ...) {
  asm volatile("trap;");
}

#define printf host_device_printf
#endif

#ifndef IAAS_KERNELS_DEVICE_ASSERT
#define IAAS_KERNELS_DEVICE_ASSERT(cond)                                             \
  do {                                                                               \
    if (not(cond)) {                                                                 \
      printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
      asm("trap;");                                                                  \
    }                                                                                \
  } while (0)
#endif

#ifndef IAAS_KERNELS_TRAP_ONLY_DEVICE_ASSERT
#define IAAS_KERNELS_TRAP_ONLY_DEVICE_ASSERT(cond) \
  do {                                             \
    if (not(cond)) asm("trap;");                   \
  } while (0)
#endif

#ifndef IAAS_KERNELS_STATIC_ASSERT
#define IAAS_KERNELS_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef IAAS_KERNELS_UNIFIED_ASSERT
#ifdef IAAS_KERNELS_IN_CUDA_COMPILATION
#define IAAS_KERNELS_UNIFIED_ASSERT(cond) IAAS_KERNELS_DEVICE_ASSERT(cond)
#else
#define IAAS_KERNELS_UNIFIED_ASSERT(cond) IAAS_KERNELS_HOST_ASSERT(cond)
#endif
#endif
