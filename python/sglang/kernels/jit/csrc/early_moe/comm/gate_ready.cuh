#pragma once
#include <cuda/atomic>
#include <early_moe/layout/early_moe.cuh>

namespace early_moe {
__device__ __forceinline__ uint64_t now() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
template <class T>
__device__ __forceinline__ T acquire(T* p) {
  return cuda::atomic_ref<T, cuda::thread_scope_device>(*p).load(cuda::memory_order_acquire);
}
template <class T>
__device__ __forceinline__ void release(T* p, T v) {
  cuda::atomic_ref<T, cuda::thread_scope_device>(*p).store(v, cuda::memory_order_release);
}
__device__ __forceinline__ uint64_t peer_load(uint64_t* p) {
  return cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*p).load(cuda::memory_order_acquire);
}
__device__ __forceinline__ void peer_store(uint64_t* p, uint64_t v) {
  cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*p).store(v, cuda::memory_order_release);
}
__device__ __forceinline__ void abort_call(const Call& c, int error) {
  atomicCAS(&c.workspace->status, OK, error);
  peer_store(c.own_flags + 1, c.epoch);
}
// Only the control warp calls this at the initialization boundary. Independent
// lanes issue the eight peer reads concurrently; every rank is still checked.
__device__ __forceinline__ void initial_gate_scan(const Call& c) {
  auto& w = *c.workspace;
  const int lane = threadIdx.x % 32;
  bool ready = false;
  if (lane < c.world) {
    auto* p = reinterpret_cast<uint64_t*>(c.peers[lane]);
    if (peer_load(p + 1) >= c.epoch) abort_call(c, PEER_ABORT);
    ready = peer_load(p) >= c.epoch;
  }
  const unsigned mask = __ballot_sync(0xffffffffu, ready);
  __syncwarp();
  if (!lane) {
    w.ready_mask = mask;
    w.poll_cursor = 0;
    const bool stop = c.stop_signal ? peer_load(c.stop_signal) >= c.epoch : mask == (0xffffffffu >> (32 - c.world));
    if (stop || w.status != OK) {
      w.stopped = 1;
      c.stats[OBSERVED] = now();
      c.stats[INFLIGHT_AT_STOP] = 0;
    }
  }
}
// Only control thread calls poll. A sticky closure serializes admission and stop.
__device__ __forceinline__ bool poll(const Call& c, bool full_scan = false) {
  auto& w = *c.workspace;
  if (w.stopped) return true;
  // Poll one peer between local completion checks. A serial full scan costs
  // several NVLink round trips and used to quantize every phase to ~14 us.
  // Ready bits are sticky within a monotonically increasing invocation.
  for (int i = 0; i < (full_scan ? c.world : 1); ++i) {
    const int r = w.poll_cursor++ % c.world;
    auto p = reinterpret_cast<uint64_t*>(c.peers[r]);
    if (peer_load(p + 1) >= c.epoch) abort_call(c, PEER_ABORT);
    if (!(w.ready_mask & (1u << r)) && peer_load(p) >= c.epoch) w.ready_mask |= 1u << r;
  }
  bool ready = c.stop_signal ? peer_load(c.stop_signal) >= c.epoch : w.ready_mask == (0xffffffffu >> (32 - c.world));
  if (w.status != OK || now() - c.stats[START] > c.timeout_ns) {
    if (w.status == OK) abort_call(c, TIMEOUT);
    ready = true;
  }
  if (ready) {
    w.stopped = 1;
    c.stats[OBSERVED] = now();
    c.stats[INFLIGHT_AT_STOP] = w.inflight;
  }
  return ready;
}
__device__ __forceinline__ void arrive(Task* w) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(w->arrived).fetch_add(1u, cuda::memory_order_acq_rel);
}
}  // namespace early_moe
