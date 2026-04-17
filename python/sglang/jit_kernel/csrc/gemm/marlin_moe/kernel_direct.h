
#include <sgl_kernel/scalar_type.hpp>

#include "../marlin/marlin.cuh"
#include "../marlin/marlin_dtypes.cuh"
#include "marlin_tma_utils.h"

#define MARLIN_DIRECT_KERNEL_PARAMS                                                                       \
  const int4 *__restrict__ A, const int4 *__restrict__ B, int4 *__restrict__ C, int4 *__restrict__ C_tmp, \
      const int4 *__restrict__ b_bias_ptr, const int4 *__restrict__ scales_ptr,                            \
      const uint16_t *__restrict__ scale2_ptr, const int4 *__restrict__ zp_ptr, const int *__restrict__ g_idx, \
      const int32_t *__restrict__ masked_m_ptr, const int32_t *__restrict__ active_expert_ids_ptr,         \
      const int32_t *__restrict__ active_expert_count_ptr,                                                  \
      const __grid_constant__ device::marlin_moe::tma::LocalTensorMapDesc tensor_map_b_desc,               \
      bool has_tensor_map_b_desc, const __grid_constant__ device::marlin_moe::tma::LocalTensorMapDesc tensor_map_s_desc, \
      bool has_tensor_map_s_desc, int expected_m,                                                           \
      int a_expert_stride, int c_expert_stride, int num_groups, int prob_m, int prob_n, int prob_k,       \
      int *locks, bool has_bias, bool use_atomic_add, bool use_fp32_reduce, int max_shared_mem

namespace device::marlin_moe_direct {
template <
    typename scalar_t,
    const host::ScalarTypeId w_type_id,
    const host::ScalarTypeId s_type_id,
    const int threads,
    const int thread_m_blocks,
    const int thread_n_blocks,
    const int thread_k_blocks,
    const bool m_block_size_8,
    const int stages,
    const int group_blocks,
    const bool is_zp_float>
__global__ void MarlinDirect(MARLIN_DIRECT_KERNEL_PARAMS);

}  // namespace device::marlin_moe_direct
