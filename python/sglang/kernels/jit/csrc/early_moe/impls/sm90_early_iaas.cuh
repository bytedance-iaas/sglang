#pragma once
#include <sgl_kernel/early_moe/vendor/common/activation_scale.cuh>
#include <sgl_kernel/early_moe/vendor/common/tma_copy.cuh>
#include <sgl_kernel/early_moe/vendor/common/utils.cuh>
#include <sgl_kernel/early_moe/vendor/ptx/ld_st.cuh>
#include <sgl_kernel/early_moe/vendor/ptx/tma.cuh>
#include <sgl_kernel/early_moe/vendor/ptx/utils.cuh>

#include <cutlass/arch/reg_reconfig.h>
#include <early_moe/impls/sm90_early_common.cuh>
#include <early_moe/scheduler/early_iaas.cuh>

namespace early_moe {
namespace math = iaas_kernels::math;
enum class BlockPhase { Linear1, Linear2 };
namespace tma = iaas_kernels::tma;
using iaas_kernels::sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv;
constexpr int IAAS_FIFO = 2;
struct IAASTile {
  uint32_t phase, task, n;
};
template <int STAGES>
struct IAASShared {
  alignas(1024) uint8_t cd[BM * 256 * 2];
  alignas(1024) FP8 a[STAGES][BM * BK];
  alignas(1024) FP8 b[STAGES][256 * BK];
  alignas(128) float sf[STAGES][2 * BM];
  Barrier full[STAGES], empty[STAGES];
  IAASTile jobs[IAAS_FIFO], selected;
  uint32_t ready[IAAS_FIFO], retired, prep_slot;
};
__device__ inline uint32_t block_acquire(uint32_t* p) {
  return cuda::atomic_ref<uint32_t, cuda::thread_scope_block>(*p).load(cuda::memory_order_acquire);
}
__device__ inline void block_release(uint32_t* p, uint32_t v) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_block>(*p).store(v, cuda::memory_order_release);
}
__device__ inline void producer_sync() {
  // Barrier 0 belongs to full-CTA/grid synchronization. A/B or math roles
  // can reach kernel teardown while dispatch still drains its last job.
  asm volatile("bar.sync 1, 64;" ::: "memory");
}
template <int STAGES>
__device__ inline IAASTile next_iaas_job(IAASShared<STAGES>& s, uint32_t ticket) {
  while (block_acquire(s.ready + ticket % IAAS_FIFO) != ticket + 1)
    __nanosleep(32);
  return s.jobs[ticket % IAAS_FIFO];
}
__device__ inline IAASTile select_iaas_job(const Call& c, int first_slot, bool bulk_prepared) {
  // Complete downstream work promptly; a claimed tile keeps its M64 live
  // until completion, so its descriptor cannot be recycled under the FIFO.
  // All lanes of dispatch warp 0 participate. Scan the task slots in parallel
  // instead of issuing up to 3 * groups dependent acquire loads on lane 0.
  const int lane = threadIdx.x % 32;
  const bool active = lane < c.groups;
  int slot = first_slot + lane;
  if (slot >= c.groups) slot -= c.groups;
  const int first_phase = bulk_prepared ? int(L1) : int(PREP);
  for (int phase = int(L2); phase >= first_phase; --phase) {
    uint64_t* counter = nullptr;
    bool available = false;
    if (active) {
      auto& t = c.workspace->tasks[slot];
      counter = phase == L2 ? &t.claim_l2 : phase == L1 ? &t.claim_l1 : &t.claim_prep;
      available = uint32_t(acquire(counter)) < iaas_phase_tiles(phase);
    }
    unsigned candidates = __ballot_sync(0xffffffffu, available);
    while (candidates) {
      const int leader = __ffs(candidates) - 1;
      int n = -1;
      // Retain the generation-tagged claim protocol. A concurrent CTA
      // may exhaust or recycle this slot after the availability snapshot.
      if (lane == leader) n = claim_iaas_tile(counter, iaas_phase_tiles(phase));
      n = __shfl_sync(0xffffffffu, n, leader);
      const int selected_slot = __shfl_sync(0xffffffffu, slot, leader);
      if (n >= 0) return {uint32_t(phase), uint32_t(selected_slot), uint32_t(n)};
      candidates &= candidates - 1;
    }
  }
  return {IDLE, 0, 0};
}
__device__ inline void iaas_prepare_bulk(const KernelArgs& a) {
  const auto& c = a.c;
  const auto& w = *c.workspace;
  const int lane = threadIdx.x % 32;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int warp_stride = gridDim.x * blockDim.x / 32;
  // Keep a row on one warp: expert lookup and queue indirection are reused
  // across all 32 scale groups, instead of repeated for every 128 channels.
  for (int row = warp; row < w.input_offsets[S]; row += warp_stride) {
    // Find the expert containing this padded pool row. Repeated offsets
    // from empty experts are valid; choose the last offset <= row.
    int low = 0, high = S;
    while (low + 1 < high) {
      const int middle = (low + high) / 2;
      if (w.input_offsets[middle] <= row)
        low = middle;
      else
        high = middle;
    }
    const int expert_row = row - w.input_offsets[low];
    const bool valid = expert_row < w.histogram[low];
    const int token = valid ? c.queue[w.offsets[low] + expert_row] / K : 0;
    const int lane_in_group = lane % 16;
    for (int group_base = 0; group_base < H / 128; group_base += 2) {
      const int group = group_base + lane / 16;
      uint4 packed_input = make_uint4(0, 0, 0, 0);
      if (valid) {
        packed_input = *reinterpret_cast<const uint4*>(c.hidden + int64_t(token) * H + group * 128 + lane_in_group * 8);
      }
      // Each half-warp handles one 128-channel scale group. A lane reads
      // eight BF16 values in one aligned 128-bit load, then emits FP8x8.
      const float v[8] = {
          __uint_as_float(packed_input.x << 16),
          __uint_as_float(packed_input.x & 0xffff0000u),
          __uint_as_float(packed_input.y << 16),
          __uint_as_float(packed_input.y & 0xffff0000u),
          __uint_as_float(packed_input.z << 16),
          __uint_as_float(packed_input.z & 0xffff0000u),
          __uint_as_float(packed_input.w << 16),
          __uint_as_float(packed_input.w & 0xffff0000u)};
      float maximum = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        maximum = fmaxf(maximum, fabsf(v[i]));
#pragma unroll
      for (int d = 8; d; d >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, d, 16));
      const float sf = fmaxf(maximum, 1e-10f) / 448.0f, inverse = 1.0f / sf;
      const __nv_fp8x2_e4m3 p0(make_float2(v[0] * inverse, v[1] * inverse));
      const __nv_fp8x2_e4m3 p1(make_float2(v[2] * inverse, v[3] * inverse));
      const __nv_fp8x2_e4m3 p2(make_float2(v[4] * inverse, v[5] * inverse));
      const __nv_fp8x2_e4m3 p3(make_float2(v[6] * inverse, v[7] * inverse));
      const uint64_t packed =
          uint64_t(p0.__x) | (uint64_t(p1.__x) << 16) | (uint64_t(p2.__x) << 32) | (uint64_t(p3.__x) << 48);
      *reinterpret_cast<uint64_t*>(a.a1 + int64_t(row) * H + group * 128 + lane_in_group * 8) = packed;
      if (!lane_in_group) a.sf1[int64_t(group) * a.input_rows + row] = sf;
    }
  }
  // The following grid barrier joins all writers before admission. Publish
  // generic stores to the async proxy used by the subsequent TMA loads too.
  __threadfence();
  async_global_fence();
}
template <int STAGES>
__device__ inline void
iaas_prepare_piece(const KernelArgs& a, IAASShared<STAGES>& shared, IAASTile job, uint32_t ticket) {
  const auto& c = a.c;
  auto& task = c.workspace->tasks[job.task];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  // Four rows per shared-queue job; all twelve warps cooperate. This keeps
  // BF16 gather/quantize out of a long serial dispatch-warp loop.
  for (int j = warp; j < 4 * (H / 128); j += 12) {
    const int row = job.n * 4 + j / (H / 128), group = j % (H / 128);
    const int token = row < task.rows ? task.assignments[row] / K : 0;
    float v[4], maximum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[i] = row < task.rows ? __bfloat162float(c.hidden[token * H + group * 128 + lane * 4 + i]) : 0;
      maximum = fmaxf(maximum, fabsf(v[i]));
    }
#pragma unroll
    for (int d = 16; d; d >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, d));
    const float sf = fmaxf(maximum, 1e-10f) / 448.0f, inverse = 1.0f / sf;
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      __nv_fp8_e4m3 q(v[i] * inverse);
      packed |= uint32_t(q.__x) << (8 * i);
    }
    *reinterpret_cast<uint32_t*>(a.a1 + (job.task * BM + row) * H + group * 128 + lane * 4) = packed;
    if (!lane) a.sf1[group * 128 * c.groups + job.task * 128 + row] = sf;
  }
  __threadfence();
  asm volatile("bar.sync 6, 384;" ::: "memory");
  if (!threadIdx.x) {
    auto count = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(task.prep_completed);
    if (count.fetch_add(1u, cuda::memory_order_acq_rel) == IAAS_PREP_TILES - 1) {
      task.prep_end = now();
      release(&task.state, uint32_t(L1));
      release(&task.claim_l1, uint64_t(task.generation) << 32);
    }
    block_release(&shared.retired, ticket + 1);
  }
  asm volatile("bar.sync 6, 384;" ::: "memory");
}
template <int STAGES>
__device__ inline void iaas_dispatch(const KernelArgs& a, IAASShared<STAGES>& s, int worker) {
  const auto& c = a.c;
  uint32_t ticket = 0;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  // Give workers different preferred slots, while preserving locality across
  // their queued tiles and falling back to every other eligible task slot.
  const int first_slot = worker % c.groups;
  while (true) {
    if (!warp) {
      int select = 0;
      if (!lane) {
        s.prep_slot = a.input_rows ? 0u : acquire(&c.workspace->mailboxes[worker].grant);
        s.selected = {IDLE, 0, 0};
        if (!s.prep_slot) {
          bool space = ticket < IAAS_FIFO || block_acquire(&s.retired) >= ticket - IAAS_FIFO + 1;
          if (space) {
            if (acquire(&c.workspace->command) == EXIT)
              s.selected.phase = EXIT;
            else
              select = 1;
          }
        }
      }
      // This branch is uniform across the complete first producer warp;
      // the second producer warp joins at the existing 64-thread barrier.
      select = __shfl_sync(0xffffffffu, select, 0);
      if (select) {
        const auto job = select_iaas_job(c, first_slot, a.input_rows != 0);
        if (!lane) {
          s.selected = job;
          if (job.phase == IDLE && !a.input_rows) release(&c.workspace->mailboxes[worker].request, 1u);
        }
      }
    }
    producer_sync();
    if (s.prep_slot) {
      const int slot = s.prep_slot - 1;
      auto& task = c.workspace->tasks[slot];
      if (!threadIdx.x) {
        release(&task.claim_prep, uint64_t(task.generation) << 32);
        release(&c.workspace->mailboxes[worker].grant, 0u);
      }
      producer_sync();
      continue;
    }
    if (s.selected.phase != IDLE) {
      if (!threadIdx.x) {
        s.jobs[ticket % IAAS_FIFO] = s.selected;
        block_release(s.ready + ticket % IAAS_FIFO, ticket + 1);
      }
      const auto job = s.selected;
      const bool exit = job.phase == EXIT;
      producer_sync();
      if (job.phase == PREP) iaas_prepare_piece(a, s, job, ticket);
      ++ticket;
      if (exit) return;
    } else {
      __nanosleep(64);
    }
  }
}

template <int STAGES>
__device__ inline void iaas_worker(const KernelArgs& a, IAASShared<STAGES>& shared, int worker) {
  const auto& c = a.c;
  constexpr uint32_t BLOCK_M = BM, BLOCK_N = 256, BLOCK_K = BK;
  constexpr uint32_t kHidden = H, kIntermediateHidden = I;
  constexpr uint32_t kNumEpilogueThreads = 256;
  constexpr uint32_t kWarpgroupSplitN = 2, WG_BLOCK_M = 64, WG_BLOCK_N = 128;
  constexpr uint32_t WG_L1_OUT_BLOCK_N = 64, L1_OUT_BLOCK_N = 128;
  constexpr uint32_t WG_SMEM_CD_L1_STRIDE_N = 64, SF_BLOCK_M = 128;
  constexpr uint32_t kEpilogueFullBarrierIdx = 2, kEpilogueWGBarrierStartIdx = 3;
  constexpr uint32_t SMEM_CD_SIZE = BM * 256 * 2;
  constexpr bool kSplitNSharesSF = false, kFastMath = true;
  const uint32_t kNumPaddedSFPoolTokens = c.groups * 128;
  using L1WGMMA = typename iaas_kernels::mma::sm90::FP8MMASelector<128>::type;
  const uint32_t thread_idx = threadIdx.x, warp_idx = thread_idx / 32, lane_idx = thread_idx % 32;
  if (!thread_idx) {
    shared.retired = 0;
    for (int i = 0; i < IAAS_FIFO; ++i)
      shared.ready[i] = 0;
  }
  if (warp_idx == 2 && !lane_idx) {
    for (int i = 0; i < STAGES; ++i) {
      shared.full[i].init(2);
      shared.empty[i].init(8);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  // This cursor is persistent across all tiles, exactly as in IAAS.
  uint32_t stage_idx = 0, phase = 0;
  auto advance_pipeline = [&](uint32_t& k) {
    ++k;
    stage_idx = stage_idx == STAGES - 1 ? 0 : stage_idx + 1;
    phase ^= stage_idx == 0;
  };
  const auto& full_barriers = shared.full;
  const auto& empty_barriers = shared.empty;
  auto* smem_a = shared.a;
  auto* smem_b = shared.b;
  auto* smem_sfa = shared.sf;
  const auto& tensor_map_l1_output = a.tout1;
  auto* smem_cd_l1 = reinterpret_cast<FP8*>(shared.cd);
  auto* smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(shared.cd);
  const float* l1_weights_sf = a.wsf1;
  const float* l2_weights_sf = a.wsf2;
  if (warp_idx < 4) {
    cutlass::arch::warpgroup_reg_dealloc<48>();
    if (warp_idx < 2) {
      iaas_dispatch(a, shared, worker);
    } else {
      uint32_t ticket = 0;
      while (true) {
        const auto job = next_iaas_job(shared, ticket++);
        if (job.phase == EXIT) break;
        if (job.phase == PREP) {
          iaas_prepare_piece(a, shared, job, ticket - 1);
          continue;
        }
        const bool linear1 = job.phase == L1;
        const uint32_t num_k_blocks = linear1 ? H / BK : I / BK;
        const int input_row = linear1 && a.input_rows ? c.workspace->tasks[job.task].input_row : job.task * BM;
        const int input_sf_row = linear1 && a.input_rows ? input_row : job.task * 128;
        if (warp_idx == 2) async_global_fence();
        for (uint32_t k = 0; k < num_k_blocks; advance_pipeline(k)) {
          shared.empty[stage_idx].wait(phase ^ 1);
          if (!lane_idx) {
            if (warp_idx == 2) {
              load_tma(linear1 ? &a.ta1 : &a.ta2, &shared.full[stage_idx], shared.a[stage_idx], k * BK, input_row);
              load_tma(
                  linear1 ? &a.tsf1 : &a.tsf2,
                  &shared.full[stage_idx],
                  shared.sf[stage_idx],
                  input_sf_row,
                  linear1 ? k : 2 * k);
              if (!linear1)
                load_tma(&a.tsf2, &shared.full[stage_idx], shared.sf[stage_idx] + BM, job.task * 128, 2 * k + 1);
              shared.full[stage_idx].arrive_and_expect_tx(BM * BK + (linear1 ? 1 : 2) * BM * sizeof(float));
            } else {
              load_tma(
                  linear1 ? &a.tw1 : &a.tw2,
                  &shared.full[stage_idx],
                  shared.b[stage_idx],
                  k * BK,
                  c.workspace->tasks[job.task].slot * (linear1 ? 2 * I : H) + job.n * BLOCK_N);
              shared.full[stage_idx].arrive_and_expect_tx(BLOCK_N * BK);
            }
          }
          __syncwarp();
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<168>();
    auto sync_epilogue = [](uint32_t count, uint32_t index) {
      asm volatile("bar.sync %0, %1;" ::"r"(index), "r"(count) : "memory");
    };
    const uint32_t epilogue_warp_idx = warp_idx - 4, epilogue_wg_idx = epilogue_warp_idx / 4;
    const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
    const uint32_t warp_idx_in_wg = epilogue_warp_idx % 4;
    const uint32_t row_idx = lane_idx / 4, col_idx = lane_idx % 4;
    const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx, r_1 = r_0 + 8;
    uint32_t ticket = 0;
    while (true) {
      const auto job = next_iaas_job(shared, ticket);
      if (job.phase == EXIT) break;
      if (job.phase == PREP) {
        iaas_prepare_piece(a, shared, job, ticket++);
        continue;
      }
      const auto block_phase = job.phase == L1 ? BlockPhase::Linear1 : BlockPhase::Linear2;
      const uint32_t local_expert_idx = c.workspace->tasks[job.task].slot;
      const uint32_t n_block_idx = job.n, num_k_blocks = job.phase == L1 ? H / BK : I / BK;
#include <early_moe/impls/early_iaas_math.inc>
      // Every writer joins before the last arrival makes this phase
      // visible to the other CTAs / controller. No per-phase gang wait.
      __threadfence();
      sync_epilogue(256, kEpilogueFullBarrierIdx);
      if (!epilogue_thread_idx) {
        auto& task = c.workspace->tasks[job.task];
        auto* count = job.phase == L1 ? &task.l1_completed : &task.l2_completed;
        auto ref = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(*count);
        if (ref.fetch_add(1u, cuda::memory_order_acq_rel) == iaas_phase_tiles(job.phase) - 1) {
          if (job.phase == L1) {
            task.l1_end = now();
            release(&task.state, uint32_t(L2));
            release(&task.claim_l2, uint64_t(task.generation) << 32);
          } else {
            task.l2_end = now();
            release(&task.state, IAAS_COMPLETE);
          }
        }
        block_release(&shared.retired, ticket + 1);
      }
      sync_epilogue(256, kEpilogueFullBarrierIdx);
      ++ticket;
    }
  }
}

template <int STAGES>
__global__ __launch_bounds__(384, 1) void fused_early_iaas_sm90(const __grid_constant__ KernelArgs a) {
  extern __shared__ __align__(1024) uint8_t smem[];
  auto& shared = *reinterpret_cast<IAASShared<STAGES>*>(smem);
  const auto& c = a.c;
  auto& w = *c.workspace;
  auto grid = cooperative_groups::this_grid();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  if (!tid) {
    w.command = IDLE;
    w.arrived = 0;
    w.status = OK;
    w.stopped = w.exhausted = w.generation = w.inflight = 0;
    w.ready_mask = w.poll_cursor = 0;
    c.stats[START] = now();
    peer_store(c.own_flags, c.epoch);
    c.stats[PUBLISH] = now();
  }
  for (int i = tid; i < STAT_COUNT; i += stride)
    if (i != START && i != PUBLISH) c.stats[i] = 0;
  for (int i = tid; i < E; i += stride) {
    c.early_count[i] = 0;
    c.remaining_count[i] = 0;
  }
  for (int i = tid; i < S; i += stride)
    w.histogram[i] = w.cursor[i] = w.taken[i] = 0;
  for (int i = tid; i < c.groups; i += stride) {
    w.tasks[i].state = IDLE;
    w.tasks[i].claim_prep = w.tasks[i].claim_l1 = w.tasks[i].claim_l2 = IAAS_CLOSED;
  }
  for (int i = tid; i < c.workers; i += stride)
    w.mailboxes[i] = {0, 0};
  grid.sync();
  for (int e = tid; e < E; e += stride) {
    int slot = c.local_map[e];
    if (slot < -1 || slot >= S || (slot >= 0 && c.reverse_map[slot] != e)) abort_call(c, BAD_MAPPING);
  }
  for (int p = tid; p < c.tokens * K; p += stride) {
    int e = c.ids[p];
    c.remaining[p] = e;
    c.done[p] = 0;
    if (e == -1) continue;
    if (e < 0 || e >= E) {
      abort_call(c, BAD_ID);
      continue;
    }
    const int lane = threadIdx.x % 32;
    const unsigned same = __match_any_sync(__activemask(), e);
    if (lane == __ffs(same) - 1) {
      atomicAdd(c.remaining_count + e, __popc(same));
      int slot = c.local_map[e];
      if (slot >= 0 && slot < S) atomicAdd(w.histogram + slot, __popc(same));
    }
  }
  grid.sync();
  if (!tid) {
    peer_store(c.own_flags + 2, c.epoch);
    for (int e = 0; e < E; ++e)
      c.stats[VALID] += c.remaining_count[e];
    w.offsets[0] = w.input_offsets[0] = 0;
    for (int s = 0; s < S; ++s) {
      w.offsets[s + 1] = w.offsets[s] + w.histogram[s];
      w.input_offsets[s + 1] = w.input_offsets[s] + ((w.histogram[s] + BM - 1) / BM) * BM;
    }
    c.stats[ELIGIBLE] = w.offsets[S];
  }
  if (blockIdx.x == 0) {
    __syncthreads();
    if (threadIdx.x < 32) initial_gate_scan(c);
    __syncthreads();
    if (!threadIdx.x && w.stopped) release(&w.command, EXIT);
  }
  grid.sync();
  if (acquire(&w.command) != EXIT) {
    for (int p = tid; p < c.tokens * K; p += stride) {
      int e = c.ids[p];
      if (e < 0 || e >= E) continue;
      int slot = c.local_map[e];
      if (slot >= 0 && slot < S) {
        const int lane = threadIdx.x % 32;
        const unsigned same = __match_any_sync(__activemask(), slot);
        const int leader = __ffs(same) - 1;
        int base = lane == leader ? atomicAdd(w.cursor + slot, __popc(same)) : 0;
        base = __shfl_sync(same, base, leader);
        c.queue[w.offsets[slot] + base + __popc(same & ((1u << lane) - 1u))] = p;
      }
    }
  }
  grid.sync();
  // No FFN block has been admitted yet. The command is stable across this
  // grid-wide branch; a pre-observed stop skips preparation on every thread.
  if (a.input_rows && acquire(&w.command) != EXIT) iaas_prepare_bulk(a);
  if (a.input_rows) grid.sync();
  if (blockIdx.x == 0) {
    if (threadIdx.x < 32) iaas_controller(a);
    __syncthreads();
    if (threadIdx.x < c.world) {
      auto* peer = reinterpret_cast<uint64_t*>(c.peers[threadIdx.x]);
      while (peer_load(peer + 2) < c.epoch) {
        if (now() - c.stats[START] > c.timeout_ns) {
          abort_call(c, TIMEOUT);
          break;
        }
        __nanosleep(128);
      }
      if (peer_load(peer + 1) >= c.epoch) abort_call(c, PEER_ABORT);
    }
    __syncthreads();
  } else {
    iaas_worker(a, shared, blockIdx.x - 1);
  }
  grid.sync();
  for (int t = tid; t < c.tokens; t += stride) {
    bool any = false;
    for (int k = 0; k < K; ++k)
      any |= bool(c.done[t * K + k]);
    unsigned active = __activemask(), completed = __ballot_sync(active, any);
    if (completed && int(threadIdx.x % 32) == __ffs(active) - 1)
      atomicAdd(reinterpret_cast<unsigned long long*>(c.stats + UNIQUE_TOKENS), uint64_t(__popc(completed)));
  }
  grid.sync();
  if (!tid) {
    c.stats[STATUS] = w.status;
    c.stats[MAPPING_VERSION] = c.mapping_version;
    if (w.status != OK) c.stats[STOP_REASON] = 2;
    c.stats[EPOCH] = c.epoch;
    c.stats[WORKERS] = c.workers;
    c.stats[GROUPS] = c.groups;
    c.stats[END] = now();
  }
}
}  // namespace early_moe
