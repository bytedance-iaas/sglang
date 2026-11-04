// Ported from the pinned IAAS SM90 BN256 non-swap path. Arithmetic and
// vector epilogues retain the upstream source; only token/SF addressing,
// dynamic clamp and complete-M64 publication are adapted. Stage/phase are
// owned by each persistent role and are NOT reset at the tile boundary.
const uint32_t valid_m = uint32_t(c.workspace->tasks[job.task].rows);
const uint32_t pool_block_idx = uint32_t(job.task);
const uint32_t m_idx = pool_block_idx * BLOCK_M;
const uint32_t n_idx = n_block_idx * BLOCK_N;
const uint32_t epilogue_wg_m_idx = epilogue_wg_idx / kWarpgroupSplitN;
const uint32_t epilogue_wg_n_idx = epilogue_wg_idx - epilogue_wg_m_idx * kWarpgroupSplitN;
const uint32_t wg_n_offset = epilogue_wg_n_idx * WG_BLOCK_N;
const uint32_t wg_l1_out_n_offset = epilogue_wg_n_idx * WG_L1_OUT_BLOCK_N;
const uint32_t row_base = epilogue_wg_m_idx * WG_BLOCK_M;
const uint32_t row_offset_r0 = row_base + r_0;
const uint32_t row_offset_r1 = row_base + r_1;
const uint32_t sf_n_block_idx = kSplitNSharesSF ? n_block_idx : (n_block_idx * kWarpgroupSplitN + epilogue_wg_n_idx);
const uint32_t smem_a_wg_offset = epilogue_wg_m_idx * WG_BLOCK_M * BLOCK_K;
const uint32_t smem_b_wg_offset = epilogue_wg_n_idx * WG_BLOCK_N * BLOCK_K;
// In the shared-tile case the WG stages into the joint L1-output tile
// at its own column offset (row stride L1_OUT_BLOCK_N); otherwise each
// WG owns a disjoint contiguous WG_BLOCK_M x WG_L1_OUT_BLOCK_N slice.
const uint32_t smem_cd_l1_wg_offset =
    kSplitNSharesSF ? wg_l1_out_n_offset : (epilogue_wg_idx * WG_BLOCK_M * WG_L1_OUT_BLOCK_N);
const uint32_t smem_cd_l2_wg_offset = epilogue_wg_idx * WG_BLOCK_M * WG_BLOCK_N;
const bool valid_r0 = row_offset_r0 < valid_m;
const bool valid_r1 = row_offset_r1 < valid_m;

using WGMMA = L1WGMMA;
constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;
float final_accum[kAccumPerThread] = {};
float accum[kAccumPerThread];
for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
  shared.full[stage_idx].wait(phase);

  // Read SF (must precede warpgroup_arrive)
  float scale_a_0_lo, scale_a_1_lo;
  float scale_a_0_hi, scale_a_1_hi;  // Only used in L2 (per-64 K)
  if (block_phase == BlockPhase::Linear1) {
    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
  } else {
    // L2: SFA layout is (K=2, M=BLOCK_M) MN-major; first half SF at offset 0, second at BLOCK_M
    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r0);
    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r1);
    scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r0);
    scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r1);
  }

  // ----- Block (128, 128) weight SF (loaded directly from global) -----
  // L1 weight SF shape: (E, 2*IH/128, H/128) MN-major. The N axis is
  // [gate(IH/128), up(IH/128)]; with the gate/up gran-8 interleave on
  // the FP8 weight, each logical 128-wide N tile covers 64 rows of gate
  // plus 64 rows of up taken from the same original 128-row block, so:
  //     gate_sf_n = sf_n_block_idx / 2
  //     up_sf_n   = (IH/128) + sf_n_block_idx / 2
  //
  // L2 weight SF shape: (E, H/128, IH/128) MN-major. One scalar per
  // logical 128x128 weight-SF tile, broadcast across the matching
  // WGMMA accumulators.
  //
  // Load the weight scale after the barrier from all WG threads.
  // This keeps scale loads close to their WGMMA use and lets the
  // read-only cache coalesce the same-address accesses.
  constexpr uint32_t kL1SFKBlocks = kHidden / 128;
  constexpr uint32_t kL2SFKBlocks = kIntermediateHidden / 128;
  constexpr uint32_t kL1SFGateBlks = kIntermediateHidden / 128;
  constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
  constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;
  float gate_sf = 0.0f, up_sf = 0.0f, l2_sf = 0.0f;
  if (block_phase == BlockPhase::Linear1) {
    const uint32_t gate_n = sf_n_block_idx / 2u;
    const uint32_t up_n = kL1SFGateBlks + gate_n;
    const float* base = l1_weights_sf + local_expert_idx * kL1SFPerExpert + k_block_idx;
    gate_sf = __ldg(base + gate_n * kL1SFKBlocks);
    up_sf = __ldg(base + up_n * kL1SFKBlocks);
  } else {
    l2_sf = __ldg(l2_weights_sf + local_expert_idx * kL2SFPerExpert + sf_n_block_idx * kL2SFKBlocks + k_block_idx);
  }

  if (block_phase == BlockPhase::Linear1) {
    // Descriptor offsets are in 16-byte units, as in FC2.
    cute::GmmaDescriptor fc1_desc_a, fc1_desc_b;
    fc1_desc_a = iaas_kernels::mma::sm90::make_smem_desc(smem_a[stage_idx] + smem_a_wg_offset, 1);
    fc1_desc_b = iaas_kernels::mma::sm90::make_smem_desc(smem_b[stage_idx] + smem_b_wg_offset, 1);
// Single per-128 K-block WGMMA group
#pragma unroll
    for (uint32_t i = 0; i < kAccumPerThread; ++i)
      ptx::warpgroup_fence_operand(accum[i]);
    ptx::warpgroup_arrive();
#pragma unroll
    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
      cute::GmmaDescriptor desc_a, desc_b;
      desc_a.desc_ = fc1_desc_a.desc_ + (k * WGMMA::K) / 16;
      desc_b.desc_ = fc1_desc_b.desc_ + (k * WGMMA::K) / 16;
      WGMMA::wgmma(desc_a, desc_b, accum, k);
    }
    ptx::warpgroup_commit_batch();
#pragma unroll
    for (uint32_t i = 0; i < kAccumPerThread; ++i)
      ptx::warpgroup_fence_operand(accum[i]);
    ptx::warpgroup_wait<0>();

    if (lane_idx == 0) shared.empty[stage_idx].arrive();

// L1: gate/up alternate at gran=8 along N; each `i` block of 8
// cols belongs entirely to one of {gate, up}, so .x and .y
// share the same scalar.
#pragma unroll
    for (uint32_t i = 0; i < kAccumPerThread / 4; ++i) {
      const float sb = (i & 1u) ? up_sf : gate_sf;
      final_accum[i * 4 + 0] += scale_a_0_lo * sb * accum[i * 4 + 0];
      final_accum[i * 4 + 1] += scale_a_0_lo * sb * accum[i * 4 + 1];
      final_accum[i * 4 + 2] += scale_a_1_lo * sb * accum[i * 4 + 2];
      final_accum[i * 4 + 3] += scale_a_1_lo * sb * accum[i * 4 + 3];
    }

  } else {
// Build one descriptor per operand. K offsets are 16-byte
// units in the low 14 bits; all addresses remain in smem.

// L2: split BLOCK_K=128 into two halves (per-64 SFA), each 2 WGMMAs.
// First half: K=0..63, SFA = scale_a_*_lo
#pragma unroll
    for (uint32_t i = 0; i < kAccumPerThread; ++i)
      ptx::warpgroup_fence_operand(accum[i]);
    ptx::warpgroup_arrive();
#pragma unroll
    for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++k) {
      cute::GmmaDescriptor desc_a, desc_b;
      desc_a = iaas_kernels::mma::sm90::make_smem_desc(smem_a[stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
      desc_b = iaas_kernels::mma::sm90::make_smem_desc(smem_b[stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
      WGMMA::wgmma(desc_a, desc_b, accum, k);
    }
    ptx::warpgroup_commit_batch();
#pragma unroll
    for (uint32_t i = 0; i < kAccumPerThread; ++i)
      ptx::warpgroup_fence_operand(accum[i]);
    ptx::warpgroup_wait<0>();

// L2 first half: single scalar `l2_sf` broadcast across N.
#pragma unroll
    for (uint32_t i = 0; i < kAccumPerThread / 4; ++i) {
      final_accum[i * 4 + 0] += scale_a_0_lo * l2_sf * accum[i * 4 + 0];
      final_accum[i * 4 + 1] += scale_a_0_lo * l2_sf * accum[i * 4 + 1];
      final_accum[i * 4 + 2] += scale_a_1_lo * l2_sf * accum[i * 4 + 2];
      final_accum[i * 4 + 3] += scale_a_1_lo * l2_sf * accum[i * 4 + 3];
    }

// Second half: K=64..127, SFA = scale_a_*_hi
#pragma unroll
    for (uint32_t i = 0; i < kAccumPerThread; ++i)
      ptx::warpgroup_fence_operand(accum[i]);
    ptx::warpgroup_arrive();
#pragma unroll
    for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++k) {
      const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
      cute::GmmaDescriptor desc_a, desc_b;
      desc_a = iaas_kernels::mma::sm90::make_smem_desc(smem_a[stage_idx] + smem_a_wg_offset + k_off, 1);
      desc_b = iaas_kernels::mma::sm90::make_smem_desc(smem_b[stage_idx] + smem_b_wg_offset + k_off, 1);
      WGMMA::wgmma(desc_a, desc_b, accum, k);
    }
    ptx::warpgroup_commit_batch();
#pragma unroll
    for (uint32_t i = 0; i < kAccumPerThread; ++i)
      ptx::warpgroup_fence_operand(accum[i]);
    ptx::warpgroup_wait<0>();

    if (lane_idx == 0) shared.empty[stage_idx].arrive();

// L2 second half: same broadcast scalar `l2_sf`.
#pragma unroll
    for (uint32_t i = 0; i < kAccumPerThread / 4; ++i) {
      final_accum[i * 4 + 0] += scale_a_0_hi * l2_sf * accum[i * 4 + 0];
      final_accum[i * 4 + 1] += scale_a_0_hi * l2_sf * accum[i * 4 + 1];
      final_accum[i * 4 + 2] += scale_a_1_hi * l2_sf * accum[i * 4 + 2];
      final_accum[i * 4 + 3] += scale_a_1_hi * l2_sf * accum[i * 4 + 3];
    }
  }
}
if (block_phase == BlockPhase::Linear1) {
  // ---------------- L1 EPILOGUE: activation + FP8 quantize + TMA store ----------------
  // Layout in `final_accum`:
  //   kAccumPerThread/4 chunks, each chunk = 4 floats per thread =
  //   (r0c0, r0c1, r1c0, r1c1).
  //   Gate and up chunks alternate; pair `p` uses chunks 2p and 2p+1.
  //
  // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
  // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

  constexpr uint32_t kNumPairs = kAccumPerThread / 8;
  float sf_r0, sf_inv_r0;
  float sf_r1, sf_inv_r1;

  float swiglu_r0[kNumPairs][2];
  float swiglu_r1[kNumPairs][2];
  float amax_r0 = 0.0f, amax_r1 = 0.0f;

  auto clamp_gate = [&](float& x) {
    if (c.clamp != cute::numeric_limits<float>::infinity()) x = cute::min(x, c.clamp);
  };
  auto clamp_up = [&](float& x) {
    if (c.clamp != cute::numeric_limits<float>::infinity()) x = cute::min(cute::max(x, -c.clamp), c.clamp);
  };
  auto silu = [&](float x) -> float {
    float e;
    if (c.clamp <= 80.0f) {
      // Gate has already been upper-clamped. Thus -x*log2(e)
      // is >= -115.416, never in __expf's < -126 repair path.
      // Keep the same rounded multiply and EX2 operation.
      const float exponent = __fmul_rn(x, -1.4426950408889634f);
      asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(exponent));
    } else {
      e = kFastMath ? __expf(-x) : expf(-x);
    }
    const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
    return x * sig;
  };

#pragma unroll
  for (uint32_t p = 0; p < kNumPairs; ++p) {
    const uint32_t gate = 2 * p, up = 2 * p + 1;

    float g_r0_c0 = final_accum[gate * 4 + 0];
    float g_r0_c1 = final_accum[gate * 4 + 1];
    float g_r1_c0 = final_accum[gate * 4 + 2];
    float g_r1_c1 = final_accum[gate * 4 + 3];
    float u_r0_c0 = final_accum[up * 4 + 0];
    float u_r0_c1 = final_accum[up * 4 + 1];
    float u_r1_c0 = final_accum[up * 4 + 2];
    float u_r1_c1 = final_accum[up * 4 + 3];
    clamp_gate(g_r0_c0);
    clamp_gate(g_r0_c1);
    clamp_gate(g_r1_c0);
    clamp_gate(g_r1_c1);
    clamp_up(u_r0_c0);
    clamp_up(u_r0_c1);
    clamp_up(u_r1_c0);
    clamp_up(u_r1_c1);

    swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
    swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
    amax_r0 = fmaxf(amax_r0, fmaxf(fabsf(swiglu_r0[p][0]), fabsf(swiglu_r0[p][1])));
    swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
    swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
    amax_r1 = fmaxf(amax_r1, fmaxf(fabsf(swiglu_r1[p][0]), fabsf(swiglu_r1[p][1])));
  }

  // Apply token weight: SwiGLU * topk_weight (single load per row)
  const float weight_r0 = valid_r0 ? c.workspace->tasks[pool_block_idx].token_weights[row_offset_r0] : 0.0f;
  const float weight_r1 = valid_r1 ? c.workspace->tasks[pool_block_idx].token_weights[row_offset_r1] : 0.0f;
#pragma unroll
  for (uint32_t p = 0; p < kNumPairs; ++p) {
    swiglu_r0[p][0] *= weight_r0;
    swiglu_r0[p][1] *= weight_r0;
    swiglu_r1[p][0] *= weight_r1;
    swiglu_r1[p][1] *= weight_r1;
  }

  amax_r0 *= fabsf(weight_r0);
  amax_r1 *= fabsf(weight_r1);

  // Reduce amax across the 4 col-lanes that share the same row. In the
  // SM90 WGMMA output layout, lanes with the same `lane_idx >> 2` and
  // different `lane_idx & 3` partition the WG-owned output columns for
  // the same r_0/r_1, so this is an INTRA-group reduction
  // (`warp_reduce<4, false>`). Using `<4, true>` would instead merge
  // amax across 8 different rows -- giving wrong per-row SF.
  amax_r0 = math::warp_reduce<4, false>(amax_r0, math::ReduceMax<float>());
  amax_r1 = math::warp_reduce<4, false>(amax_r1, math::ReduceMax<float>());

  // Phase 2: cross-WG amax. When two N-split warpgroups share one
  // per-64 SF group, each WG so far only saw its own
  // WG_L1_OUT_BLOCK_N columns; the true per-row amax spans both
  // halves. Reduce across both warpgroups through a small smem
  // scratch (carved from the upper, currently-unused half of the
  // CD staging region) so BOTH WGs quantize with the SAME SF.
  if constexpr (kSplitNSharesSF) {
    float* amax_scratch = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_cd_l1) + SMEM_CD_SIZE / 2);
#pragma unroll
    for (uint32_t i = epilogue_thread_idx; i < BLOCK_M; i += kNumEpilogueThreads)
      amax_scratch[i] = 0.0f;
    sync_epilogue(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
    if (col_idx == 0) {
      atomicMax(reinterpret_cast<unsigned int*>(&amax_scratch[r_0]), __float_as_uint(amax_r0));
      atomicMax(reinterpret_cast<unsigned int*>(&amax_scratch[r_1]), __float_as_uint(amax_r1));
    }
    sync_epilogue(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
    amax_r0 = amax_scratch[r_0];
    amax_r1 = amax_scratch[r_1];
    sync_epilogue(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
  }

  // Compute SF and inverse SF for each row
  float2 amax_pair = {amax_r0, amax_r1};
  float2 sf_pair, sf_inv_pair;
  sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);

  sf_r0 = sf_pair.x;
  sf_inv_r0 = sf_inv_pair.x;
  sf_r1 = sf_pair.y;
  sf_inv_r1 = sf_inv_pair.y;

  // Quantize and write to the shared-memory staging tile.
  auto* smem_cd_l1_wg = smem_cd_l1 + smem_cd_l1_wg_offset;
  auto l1_store_offset = [&](uint32_t row, uint32_t col) {
    const uint32_t offset = row * WG_SMEM_CD_L1_STRIDE_N + col;

    return offset;
  };
  IAAS_KERNELS_STATIC_ASSERT(kNumPairs % 2 == 0, "L1 staging stores two 8-byte chunks at once");
#pragma unroll
  for (uint32_t p_base = 0; p_base < kNumPairs; p_base += 2) {
    uint16_t r0_bits[2], r1_bits[2];
#pragma unroll
    for (uint32_t q = 0; q < 2; ++q) {
      const uint32_t p = p_base + q;
      const float v00 = swiglu_r0[p][0] * sf_inv_r0;
      const float v01 = swiglu_r0[p][1] * sf_inv_r0;
      const float v10 = swiglu_r1[p][0] * sf_inv_r1;
      const float v11 = swiglu_r1[p][1] * sf_inv_r1;

      const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
      const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));
      r0_bits[q] = valid_r0 ? r0_pair.__x : 0u;
      r1_bits[q] = valid_r1 ? r1_pair.__x : 0u;
    }

#pragma unroll
    for (uint32_t q = 0; q < 2; ++q) {
      const uint32_t p = p_base + q;
      const uint32_t col = p * 8 + col_idx * 2;
      auto* p0 = reinterpret_cast<uint16_t*>(smem_cd_l1_wg + l1_store_offset(r_0, col));
      auto* p1 = reinterpret_cast<uint16_t*>(smem_cd_l1_wg + l1_store_offset(r_1, col));
      if (valid_r0) *p0 = r0_bits[q];
      if (valid_r1) *p1 = r1_bits[q];
    }
  }

  // Write SF as float at `[token, n_block_idx]` in L2 acts SF buffer (per-64 layout).
  // Each row is contributed by lanes col_idx in {0..3}; only col_idx == 0 writes.
  // In the shared-SF split both warpgroups own the same per-64 group and rows, so
  // only the first N-split warpgroup publishes the SF slot to avoid a write race.
  if (col_idx == 0 and (not kSplitNSharesSF or epilogue_wg_n_idx == 0)) {
    auto sf_base_ptr = a.sf2;
    // SF buffer is (kNumPaddedSFPoolTokens x kIntermediateHidden/64), MN-major:
    //   addr[k_idx * num_padded_sf_pool_tokens + token_idx]
    const uint32_t token_r0 = pool_block_idx * SF_BLOCK_M + row_offset_r0;
    const uint32_t token_r1 = pool_block_idx * SF_BLOCK_M + row_offset_r1;
    const uint32_t k_sf_idx = sf_n_block_idx;  // one per-64 post-SwiGLU group
    if (valid_r0) sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r0] = sf_r0;
    if (valid_r1) sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r1] = sf_r1;
  }

  // Sync the warpgroup before TMA store. In the shared-tile split
  // both N-split warpgroups must finish writing their halves of the
  // joint L1-output tile, so sync across all epilogue threads.
  if constexpr (kSplitNSharesSF)
    sync_epilogue(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
  else
    sync_epilogue(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

  // Issue TMA store of the entire tile. Padding rows beyond
  // `valid_m` are written with stale/garbage FP8 to the L1-output
  // pool buffer, but they never reach valid output rows: the L2
  // GEMM tile loads them, but its NVLink-scatter epilogue is
  // gated by `m_idx_in_block >= valid_m`, and stale SF in the
  // padding rows can produce NaN accumulators that simply stay
  // in registers (only valid rows are converted to BF16 and
  // STSM'd into smem). Using TMA for partial tiles is a large
  // win for low-batch / decode where every tile is partial.
  if constexpr (kSplitNSharesSF) {
    // One combined store of the joint L1_OUT_BLOCK_N tile, issued
    // by the first N-split warpgroup once both halves are staged.
    if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
      const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
      cute::tma_store_fence();
      cute::SM90_TMA_STORE_2D::copy(&tensor_map_l1_output, smem_cd_l1, out_n_idx, m_idx + row_base);
      cute::tma_store_arrive();
    }
  } else {
    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
      const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_offset;
      cute::tma_store_fence();
      cute::SM90_TMA_STORE_2D::copy(
          &tensor_map_l1_output, smem_cd_l1 + smem_cd_l1_wg_offset, out_n_idx, m_idx + row_base);
      cute::tma_store_arrive();
    }
  }
  __syncwarp();
  ptx::tma_store_wait<0>();

  // Publish one arrival-mask bit after all warpgroups finish
  // their stores for this L1 N tile.
  sync_epilogue(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

  __syncwarp();
  // In the shared-tile split only the first warpgroup issues and
  // drains the combined TMA store; gate the other warpgroup so it
  // cannot overwrite the joint smem tile in the next block until
  // that store has drained.
  if constexpr (kSplitNSharesSF) sync_epilogue(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

} else {
  const uint32_t tid = warp_idx_in_wg * 32 + lane_idx;
  auto scratch = reinterpret_cast<uint8_t*>(smem_cd_l2 + smem_cd_l2_wg_offset);
  const uint32_t store_base =
      ((tid << 4) & 1920u) | ((tid << 3) & 16u) | ((tid & 1u) ? 8256u : 0u) | ((tid & 4u) ? 4128u : 0u);
#pragma unroll
  for (uint32_t g = 0; g < 4; ++g) {
#pragma unroll
    for (uint32_t h = 0; h < 2; ++h) {
      const uint32_t i = g * 4 + h * 16;
      const uint4 packed = make_uint4(
          math::cast_into_bf16_and_pack(final_accum[i], final_accum[i + 1]),
          math::cast_into_bf16_and_pack(final_accum[i + 2], final_accum[i + 3]),
          math::cast_into_bf16_and_pack(final_accum[i + 32], final_accum[i + 33]),
          math::cast_into_bf16_and_pack(final_accum[i + 34], final_accum[i + 35]));
      const uint32_t addr =
          static_cast<uint32_t>(__cvta_generic_to_shared(scratch + (store_base ^ (g * 32u)) + h * 2048u));
      asm volatile("st.volatile.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(addr),
                   "r"(packed.x),
                   "r"(packed.y),
                   "r"(packed.z),
                   "r"(packed.w)
                   : "memory");
    }
  }

  // Reader ownership spans all four warps in this partition.
  sync_epilogue(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

  const uint32_t load_base =
      ((((tid << 4) & 112u) | ((tid << 2) & 384u)) ^ ((tid & 16u) ? 4128u : 0u)) | ((tid << 8) & 2048u);
  const uint32_t row_start = (tid >> 4) & 7u;
  const uint32_t col_start = (tid << 2) & 60u;
  auto make_scatter_ptr = [&](uint32_t row) -> uint2* {
    const uint32_t token_row = row_base + row;
    if (token_row >= valid_m) return nullptr;
    uint64_t base;
    // Ordinary coherent load: dispatch writes this
    // table during the same kernel invocation.
    base = c.workspace->tasks[pool_block_idx].output_ptrs[token_row];
    return reinterpret_cast<uint2*>(base + (n_idx + wg_n_offset + col_start) * sizeof(nv_bfloat16));
  };
  auto store_scatter_ptr = [&](uint2* remote, uint2 low, uint2 high, bool valid) {
    if (valid) {
      const uint64_t addr = __cvta_generic_to_global(remote);
      asm volatile(
          "st.global.v2.b32 [%0], {%1, %2};\n"
          "st.global.v2.b32 [%0+128], {%3, %4};" ::"l"(addr),
          "r"(low.x),
          "r"(low.y),
          "r"(high.x),
          "r"(high.y)
          : "memory");
    }
  };
#pragma unroll
  for (uint32_t g = 0; g < 4; ++g) {
    // Bound the live set to two row addresses, rather than
    // retaining eight pointers throughout the epilogue.
    auto ptr0 = make_scatter_ptr(row_start + g * 16u);
    auto ptr1 = make_scatter_ptr(row_start + g * 16u + 8u);
    uint4 a, b;
    const uint32_t addr0 = static_cast<uint32_t>(__cvta_generic_to_shared(scratch + load_base + g * 512u));
    const uint32_t addr1 = static_cast<uint32_t>(__cvta_generic_to_shared(scratch + (load_base ^ 8256u) + g * 512u));
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w)
                 : "r"(addr0)
                 : "memory");
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
                 : "r"(addr1)
                 : "memory");
    store_scatter_ptr(ptr0, make_uint2(a.x, b.x), make_uint2(a.z, b.z), row_base + row_start + g * 16u < valid_m);
    store_scatter_ptr(ptr1, make_uint2(a.y, b.y), make_uint2(a.w, b.w), row_base + row_start + g * 16u + 8u < valid_m);
  }
}
