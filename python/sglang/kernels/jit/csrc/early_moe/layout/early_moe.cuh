#pragma once
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

namespace early_moe {
// E denotes physical routing IDs. Physical replica identity is preserved for any supported EP geometry.
// Explicit compile-time geometry, checked again through the exported C ABI.
constexpr int H = IAAS_EARLY_H, I = IAAS_EARLY_I, K = IAAS_EARLY_K;
constexpr int E = IAAS_EARLY_E, S = IAAS_EARLY_S;
static_assert(E > 0 && S > 0 && E % S == 0 && E / S <= 32, "physical expert IDs must evenly cover 1..32 shards");
static_assert(H % 256 == 0 && I % 128 == 0, "BN256 phase tiles require aligned dimensions");
constexpr int BM = 64, BN = 128, BK = 128, THREADS = 384, MAX_WORKERS = 64;
constexpr int MAX_GROUPS = 16;
constexpr uint32_t EXIT = 0xffffffffu;
enum Phase : uint32_t { IDLE = 0, PREP = 1, L1 = 2, L2 = 3 };
enum Status : int { OK = 0, BAD_MAPPING = 1, BAD_ID = 2, PEER_ABORT = 3, TIMEOUT = 4 };
// All timestamps use this GPU's globaltimer (ns). No cross-GPU subtraction.
enum Stat : int {
  START,
  PUBLISH,
  OBSERVED,
  LAST_COMMIT,
  END,
  VALID,
  ELIGIBLE,
  ROWS,
  TASKS,
  TAILS,
  PADDED,
  STOP_REASON,
  INFLIGHT_AT_STOP,
  ADMITTED,
  AUDIT_OVERFLOW,
  PREP_NS,
  L1_NS,
  L2_NS,
  MATH_NS,
  UNIQUE_TOKENS,
  STATUS,
  MAPPING_VERSION,
  EPOCH,
  WORKERS,
  SERVICE_NS,
  GROUPS,
  PEAK_INFLIGHT,
  STAT_COUNT = 32
};
struct alignas(128) Task {
  uint32_t command, arrived;
  int logical, slot, rows, generation, input_row;
  int assignments[BM];
  uint64_t phase_begin;
  uint64_t math_begin[MAX_WORKERS], math_end[MAX_WORKERS];
  uint32_t pair_ready[I / 64];
  float pair_amax[2 * I / 64][BM];
  uint64_t claim_prep, claim_l1, claim_l2;
  uint32_t state, prep_completed, l1_completed, l2_completed;
  uint64_t admitted, prep_end, l1_end, l2_end;
  uint64_t output_ptrs[BM];
  float token_weights[BM];
};
struct WorkerMailbox {
  uint32_t request, grant;
};
struct alignas(128) Workspace {
  uint32_t command, arrived;
  int status, stopped, exhausted;
  int inflight, generation;
  uint32_t ready_mask, poll_cursor;
  int histogram[S], offsets[S + 1], cursor[S], taken[S];
  int input_offsets[S + 1];  // M64-padded expert segments for optional bulk input preparation.
  Task tasks[MAX_GROUPS];
  WorkerMailbox mailboxes[MAX_WORKERS];
};
struct Audit {
  uint64_t generation, logical, rows, admitted, prep_end, l1_end, l2_end, committed;
};
// C ABI: keep the matching ctypes structure in early_moe.py in this order.
struct Call {
  const nv_bfloat16* hidden;
  const int* ids;
  const float* routing;
  const int* local_map;
  const int* reverse_map;
  int* remaining;
  uint8_t* done;
  nv_bfloat16* values;
  int* early_count;
  int* remaining_count;
  int* queue;
  Workspace* workspace;
  uint64_t* stats;
  Audit* audit;
  const uint64_t* peers;
  uint64_t* own_flags;
  uint64_t* stop_signal;  // Optional local epoch, published after the external barrier.
  uint64_t epoch, mapping_version, timeout_ns;
  int tokens, world, rank, workers, audit_capacity, diagnostic, groups;
  float clamp;
};
struct alignas(128) KernelArgs {
  Call c;
  int input_rows;  // Zero: per-slot prepare; nonzero: padded bulk a1/sf1 row capacity.
  uint8_t *a1, *a2;
  float *sf1, *sf2;
  const float *wsf1, *wsf2;
  CUtensorMap ta1, tsf1, tw1, tout1, ta2, tsf2, tw2;
};
}  // namespace early_moe
