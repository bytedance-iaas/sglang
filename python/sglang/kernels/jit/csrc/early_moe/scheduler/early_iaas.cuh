#pragma once
#include <early_moe/comm/gate_ready.cuh>

namespace early_moe {
constexpr uint32_t IAAS_COMPLETE = 4;
// Preparation visits four M rows per job. GEMM phases have independent N
// extents: DSV4.1 has 2*I=4608 but H=5120, so their tile counts differ.
constexpr uint32_t IAAS_PREP_TILES = BM / 4;
constexpr uint32_t IAAS_L1_TILES = 2 * I / 256;
constexpr uint32_t IAAS_L2_TILES = H / 256;
constexpr uint32_t IAAS_CLOSED = 0xffffffffu;
__device__ __forceinline__ uint32_t iaas_phase_tiles(uint32_t phase) {
  return phase == PREP ? IAAS_PREP_TILES : phase == L1 ? IAAS_L1_TILES : IAAS_L2_TILES;
}

// A generation-tagged counter prevents a delayed claimant from claiming a
// recycled slot. Counters remain closed until their inputs are fully visible.
__device__ inline int claim_iaas_tile(uint64_t* counter, uint32_t tiles) {
  auto ref = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(*counter);
  uint64_t value = ref.load(cuda::memory_order_acquire);
  while (uint32_t(value) < tiles) {
    if (ref.compare_exchange_weak(value, value + 1, cuda::memory_order_acq_rel, cuda::memory_order_acquire))
      return int(uint32_t(value));
  }
  return -1;
}

// An external barrier only needs a local stop read. Peer abort checks are
// amortized; the exit validation still joins all peers' validation epochs.
__device__ inline bool poll_iaas(const Call& c, uint32_t& polls) {
  if (!c.stop_signal) return poll(c);
  auto& w = *c.workspace;
  if (w.stopped) return true;
  if (!(++polls & 255u)) {
    const int r = w.poll_cursor++ % c.world;
    auto* peer = reinterpret_cast<uint64_t*>(c.peers[r]);
    if (peer_load(peer + 1) >= c.epoch) abort_call(c, PEER_ABORT);
  }
  bool stop = peer_load(c.stop_signal) >= c.epoch;
  if (w.status != OK || now() - c.stats[START] > c.timeout_ns) {
    if (w.status == OK) abort_call(c, TIMEOUT);
    stop = true;
  }
  if (stop) {
    w.stopped = 1;
    c.stats[OBSERVED] = now();
    c.stats[INFLIGHT_AT_STOP] = w.inflight;
  }
  return stop;
}

// One control warp handles M64 admission/commit, never individual GEMM tiles.
// Queue gather and result publication are parallel across the warp.
__device__ inline void iaas_controller(const KernelArgs& a) {
  const auto& c = a.c;
  auto& w = *c.workspace;
  const int lane = threadIdx.x;
  __shared__ int action, slot, worker, expert, count;
  uint32_t polls = 0;
  int next_worker = 0;
  uint64_t service_begin = 0;
  while (true) {
    if (!lane) poll_iaas(c, polls);
    const uint32_t task_state = lane < c.groups ? acquire(&w.tasks[lane].state) : EXIT;
    const unsigned completed = __ballot_sync(0xffffffffu, task_state == IAAS_COMPLETE);
    const unsigned idle = __ballot_sync(0xffffffffu, task_state == IDLE);
    // Each lane acquires one task's completion publication. Join those
    // observations before lane 0 consumes the selected task's metadata.
    __syncwarp();
    if (!lane) {
      action = completed ? 2 : 0;
      slot = __ffs(completed) - 1;
      if (!action && !w.stopped && !w.exhausted && w.inflight < c.groups) {
        slot = __ffs(idle) - 1;
        worker = -1;
        if (a.input_rows) worker = 0;  // Prepared tasks need no producer handshake.
        for (int i = 0; !a.input_rows && i < c.workers; ++i) {
          int j = (next_worker + i) % c.workers;
          if (acquire(&w.mailboxes[j].request) && !acquire(&w.mailboxes[j].grant)) {
            worker = j;
            next_worker = (j + 1) % c.workers;
            break;
          }
        }
        if (worker >= 0 && slot >= 0) {
          expert = -1;
          count = 0;
          for (int s = 0; s < S; ++s) {
            int n = w.histogram[s] - w.taken[s];
            if (n > count) {
              count = n;
              expert = s;
            }
          }
          if (expert < 0)
            w.exhausted = 1;
          else if (!poll_iaas(c, polls)) {
            action = 1;
            auto& t = w.tasks[slot];
            t.slot = expert;
            t.logical = c.reverse_map[expert];
            t.rows = min(count, BM);
            t.input_row = w.input_offsets[expert] + w.taken[expert];
            t.generation = ++w.generation;
            t.prep_completed = t.l1_completed = t.l2_completed = 0;
            const uint64_t closed = (uint64_t(t.generation) << 32) | IAAS_CLOSED;
            // Dispatch warps inspect closed counters continuously.
            // Use atomic stores even when recycling an idle slot.
            release(&t.claim_prep, closed);
            release(&t.claim_l1, closed);
            release(&t.claim_l2, closed);
          }
        }
      }
      if (!action && !w.inflight && w.stopped) action = 3;
    }
    __syncwarp();
    if (action == 3) break;
    if (action == 1) {
      auto& t = w.tasks[slot];
      for (int m = lane; m < t.rows; m += 32) {
        int p = c.queue[w.offsets[expert] + w.taken[expert] + m];
        t.assignments[m] = p;
        t.token_weights[m] = c.routing[p];
        t.output_ptrs[m] = reinterpret_cast<uint64_t>(c.values + int64_t(p) * H);
      }
      __threadfence();
      __syncwarp();
      if (!lane) {
        w.taken[expert] += t.rows;
        t.admitted = now();
        if (!service_begin) service_begin = t.admitted;
        ++w.inflight;
        ++c.stats[ADMITTED];
        c.stats[PEAK_INFLIGHT] = max(c.stats[PEAK_INFLIGHT], uint64_t(w.inflight));
        if (c.diagnostic) {
          if (t.generation <= c.audit_capacity)
            c.audit[t.generation - 1] = {
                uint64_t(t.generation), uint64_t(t.logical), uint64_t(t.rows), t.admitted, 0, 0, 0, 0};
          else
            ++c.stats[AUDIT_OVERFLOW];
        }
        if (a.input_rows) {
          t.prep_end = t.admitted;
          release(&t.state, uint32_t(L1));
          release(&t.claim_l1, uint64_t(t.generation) << 32);
        } else {
          release(&t.state, uint32_t(PREP));
          release(&w.mailboxes[worker].request, 0u);
          release(&w.mailboxes[worker].grant, uint32_t(slot + 1));
        }
      }
    } else if (action == 2) {
      auto& t = w.tasks[slot];
      for (int m = lane; m < t.rows; m += 32) {
        int p = t.assignments[m];
        c.done[p] = 1;
        c.remaining[p] = -1;
      }
      __threadfence();
      __syncwarp();
      if (!lane) {
        c.early_count[t.logical] += t.rows;
        c.remaining_count[t.logical] -= t.rows;
        c.stats[PREP_NS] += t.prep_end - t.admitted;
        c.stats[L1_NS] += t.l1_end - t.prep_end;
        c.stats[L2_NS] += t.l2_end - t.l1_end;
        c.stats[ROWS] += t.rows;
        ++c.stats[TASKS];
        c.stats[TAILS] += t.rows < BM;
        c.stats[PADDED] += BM - t.rows;
        c.stats[LAST_COMMIT] = now();
        if (c.diagnostic && t.generation <= c.audit_capacity) {
          auto& audit = c.audit[t.generation - 1];
          audit.prep_end = t.prep_end;
          audit.l1_end = t.l1_end;
          audit.l2_end = t.l2_end;
          audit.committed = c.stats[LAST_COMMIT];
        }
        --w.inflight;
        release(&t.state, uint32_t(IDLE));
      }
    }
    __syncwarp();
    if (!lane && !w.inflight && service_begin) {
      c.stats[SERVICE_NS] += now() - service_begin;
      service_begin = 0;
    }
    __syncwarp();
  }
  if (!lane) {
    c.stats[STOP_REASON] = w.status ? 2 : c.stop_signal ? 3 : w.exhausted ? 1 : 0;
    release(&w.command, EXIT);
  }
}
}  // namespace early_moe
