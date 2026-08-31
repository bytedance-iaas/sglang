#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <cuda/atomic>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace sidp {

constexpr int32_t kFreeOwner = -1;
constexpr int32_t kErrorClaimTimeout = 1;
constexpr int32_t kErrorWaitTimeout = 2;
constexpr int32_t kErrorReleaseMismatch = 3;
constexpr int32_t kErrorInvalidSelection = 4;
constexpr int32_t kCopyBlockSize = 512;
constexpr int32_t kCopyUnroll = 8;

__device__ __forceinline__ void fail_stop(int32_t* error_state, int32_t code) {
  atomicCAS(error_state, 0, code);
  asm volatile("trap;");
}

__device__ __forceinline__ int32_t load_device_acquire(int32_t* value) {
  cuda::atomic_ref<int32_t, cuda::thread_scope_device> ref(*value);
  return ref.load(cuda::memory_order_acquire);
}

__device__ __forceinline__ void store_device_release(int32_t* value, int32_t desired) {
  cuda::atomic_ref<int32_t, cuda::thread_scope_device> ref(*value);
  ref.store(desired, cuda::memory_order_release);
}

__global__ void reset_forward_state_kernel(
    int32_t* fill_gen,
    int32_t* comp_gen,
    int32_t num_slots,
    int32_t resident_slots,
    int32_t* error_state) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_slots) {
    fill_gen[index] = index < resident_slots ? 1 : 0;
    comp_gen[index] = 0;
  }
  if (index == 0) error_state[0] = 0;
}

__global__ void reset_cycle_state_kernel(
    uint8_t* done,
    int32_t count,
    int32_t* selected,
    int64_t* claim_spins,
    int64_t* claim_collisions) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) done[index] = 0;
  if (index == 0) {
    selected[0] = -1;
    claim_spins[0] = 0;
    claim_collisions[0] = 0;
  }
}

__global__ void select_fixed_kernel(int32_t* selected, int32_t index) {
  if (blockIdx.x == 0 && threadIdx.x == 0) selected[0] = index;
}

__global__ void claim_owner_kernel(
    const uint64_t* owner_state_ptrs,
    int32_t owner_count,
    const int32_t* candidate_owners,
    const int32_t* candidate_slots,
    uint8_t* done,
    int32_t candidate_count,
    int32_t* comp_gen,
    int32_t required_comp_gen,
    int32_t* probe_cursor,
    int32_t* selected,
    int64_t* claim_spins,
    int64_t* claim_collisions,
    int32_t requester_rank,
    int32_t backoff_ns,
    uint64_t timeout_clocks,
    int32_t* error_state) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  const uint64_t started = clock64();
  int64_t spins = 0;
  int64_t collisions = 0;
  int32_t start = probe_cursor[0];
  if (start < 0) start = 0;
  start %= candidate_count;

  while (true) {
    for (int32_t offset = 0; offset < candidate_count; ++offset) {
      const int32_t index = (start + offset) % candidate_count;
      if (done[index]) continue;

      const int32_t slot = candidate_slots[index];
      if (load_device_acquire(&comp_gen[slot]) < required_comp_gen) continue;

      const int32_t owner = candidate_owners[index];
      if (owner < 0 || owner >= owner_count) {
        fail_stop(error_state, kErrorInvalidSelection);
      }
      auto* state = reinterpret_cast<int32_t*>(
          static_cast<uintptr_t>(owner_state_ptrs[owner]));
      cuda::atomic_ref<int32_t, cuda::thread_scope_system> owner_ref(*state);
      int32_t expected = kFreeOwner;
      if (owner_ref.compare_exchange_strong(
              expected,
              requester_rank,
              cuda::memory_order_acq_rel,
              cuda::memory_order_relaxed)) {
        done[index] = 1;
        selected[0] = index;
        probe_cursor[0] = (index + 1) % candidate_count;
        claim_spins[0] = spins;
        claim_collisions[0] = collisions;
        return;
      }
      ++collisions;
    }

    ++spins;
    if (timeout_clocks > 0 && clock64() - started >= timeout_clocks) {
      claim_spins[0] = spins;
      claim_collisions[0] = collisions;
      fail_stop(error_state, kErrorClaimTimeout);
    }
    __nanosleep(static_cast<unsigned>(max(backoff_ns, 0)));
  }
}

__global__ void copy_selected_kernel(
    const uint64_t* src_ptrs,
    const uint64_t* dst_ptrs,
    const int64_t* sizes,
    int32_t candidate_count,
    const int32_t* selected,
    int32_t* error_state) {
  const int32_t selected_index = selected[0];
  if (selected_index < 0 || selected_index >= candidate_count) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      fail_stop(error_state, kErrorInvalidSelection);
    }
    return;
  }

  const auto* src_bytes = reinterpret_cast<const uint8_t*>(
      static_cast<uintptr_t>(src_ptrs[selected_index]));
  auto* dst_bytes = reinterpret_cast<uint8_t*>(
      static_cast<uintptr_t>(dst_ptrs[selected_index]));
  const int64_t nbytes = sizes[selected_index];
  const int64_t n4 = nbytes / static_cast<int64_t>(sizeof(int4));
  const auto* src = reinterpret_cast<const int4*>(src_bytes);
  auto* dst = reinterpret_cast<int4*>(dst_bytes);

  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; i + static_cast<int64_t>(kCopyUnroll - 1) * stride < n4;
       i += static_cast<int64_t>(kCopyUnroll) * stride) {
    int4 values[kCopyUnroll];
#pragma unroll
    for (int32_t item = 0; item < kCopyUnroll; ++item) {
      values[item] = src[i + static_cast<int64_t>(item) * stride];
    }
#pragma unroll
    for (int32_t item = 0; item < kCopyUnroll; ++item) {
      dst[i + static_cast<int64_t>(item) * stride] = values[item];
    }
  }
  for (; i < n4; i += stride) dst[i] = src[i];

  const int64_t tail_start = n4 * static_cast<int64_t>(sizeof(int4));
  for (int64_t byte = tail_start +
       static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       byte < nbytes;
       byte += stride) {
    dst_bytes[byte] = src_bytes[byte];
  }
}

__global__ void release_owner_kernel(
    const uint64_t* owner_state_ptrs,
    int32_t owner_count,
    const int32_t* candidate_owners,
    int32_t candidate_count,
    const int32_t* selected,
    int32_t requester_rank,
    int32_t* error_state) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const int32_t selected_index = selected[0];
  if (selected_index < 0 || selected_index >= candidate_count) {
    fail_stop(error_state, kErrorInvalidSelection);
  }
  const int32_t owner = candidate_owners[selected_index];
  if (owner < 0 || owner >= owner_count) {
    fail_stop(error_state, kErrorInvalidSelection);
  }
  auto* state = reinterpret_cast<int32_t*>(
      static_cast<uintptr_t>(owner_state_ptrs[owner]));
  cuda::atomic_ref<int32_t, cuda::thread_scope_system> owner_ref(*state);
  int32_t expected = requester_rank;
  if (!owner_ref.compare_exchange_strong(
          expected,
          kFreeOwner,
          cuda::memory_order_release,
          cuda::memory_order_relaxed)) {
    fail_stop(error_state, kErrorReleaseMismatch);
  }
}

__global__ void publish_selected_fill_kernel(
    int32_t* fill_gen,
    const int32_t* candidate_slots,
    int32_t candidate_count,
    const int32_t* selected,
    int32_t target_gen,
    int32_t* error_state) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const int32_t selected_index = selected[0];
  if (selected_index < 0 || selected_index >= candidate_count) {
    fail_stop(error_state, kErrorInvalidSelection);
  }
  store_device_release(&fill_gen[candidate_slots[selected_index]], target_gen);
}

__global__ void wait_generation_kernel(
    int32_t* generations,
    int32_t slot,
    int32_t target,
    int32_t backoff_ns,
    uint64_t timeout_clocks,
    int32_t* error_state) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const uint64_t started = clock64();
  while (load_device_acquire(&generations[slot]) < target) {
    if (timeout_clocks > 0 && clock64() - started >= timeout_clocks) {
      fail_stop(error_state, kErrorWaitTimeout);
    }
    __nanosleep(static_cast<unsigned>(max(backoff_ns, 0)));
  }
}

__global__ void publish_generation_kernel(
    int32_t* generations, int32_t slot, int32_t target) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    store_device_release(&generations[slot], target);
  }
}

__global__ void record_trace_kernel(
    const int32_t* selected,
    const int64_t* claim_spins,
    const int64_t* claim_collisions,
    int32_t* selected_trace,
    int64_t* spins_trace,
    int64_t* collisions_trace,
    int32_t trace_index) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    selected_trace[trace_index] = selected[0];
    spins_trace[trace_index] = claim_spins[0];
    collisions_trace[trace_index] = claim_collisions[0];
  }
}

struct SidpSmCopyKernels {
  static void reset_forward_state(
      tvm::ffi::TensorView fill_gen,
      tvm::ffi::TensorView comp_gen,
      int64_t resident_slots,
      tvm::ffi::TensorView error_state) {
    using namespace host;
    auto N = SymbolicSize{"num_slots"};
    auto device = SymbolicDevice{};
    TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(fill_gen).verify(comp_gen);
    TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(error_state);
    RuntimeCheck(resident_slots >= 0 && resident_slots <= N.unwrap(), "invalid resident slot count");
    LaunchKernel(div_ceil(N.unwrap(), int64_t{256}), 256, device.unwrap())(
        reset_forward_state_kernel,
        static_cast<int32_t*>(fill_gen.data_ptr()),
        static_cast<int32_t*>(comp_gen.data_ptr()),
        static_cast<int32_t>(N.unwrap()),
        static_cast<int32_t>(resident_slots),
        static_cast<int32_t*>(error_state.data_ptr()));
  }

  static void reset_cycle_state(
      tvm::ffi::TensorView done,
      tvm::ffi::TensorView selected,
      tvm::ffi::TensorView claim_spins,
      tvm::ffi::TensorView claim_collisions) {
    using namespace host;
    auto C = SymbolicSize{"candidate_count"};
    auto device = SymbolicDevice{};
    TensorMatcher({C}).with_dtype<uint8_t>().with_device<kDLGPU>(device).verify(done);
    TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(selected);
    TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(claim_spins).verify(claim_collisions);
    RuntimeCheck(C.unwrap() > 0, "SiDP cycle must have at least one candidate");
    LaunchKernel(div_ceil(C.unwrap(), int64_t{256}), 256, device.unwrap())(
        reset_cycle_state_kernel,
        static_cast<uint8_t*>(done.data_ptr()),
        static_cast<int32_t>(C.unwrap()),
        static_cast<int32_t*>(selected.data_ptr()),
        static_cast<int64_t*>(claim_spins.data_ptr()),
        static_cast<int64_t*>(claim_collisions.data_ptr()));
  }

  static void select_fixed(tvm::ffi::TensorView selected, int64_t index) {
    using namespace host;
    auto device = SymbolicDevice{};
    TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(selected);
    LaunchKernel(1, 1, device.unwrap())(
        select_fixed_kernel,
        static_cast<int32_t*>(selected.data_ptr()),
        static_cast<int32_t>(index));
  }

  static void claim_owner(
      tvm::ffi::TensorView owner_state_ptrs,
      tvm::ffi::TensorView candidate_owners,
      tvm::ffi::TensorView candidate_slots,
      tvm::ffi::TensorView done,
      tvm::ffi::TensorView comp_gen,
      int64_t required_comp_gen,
      tvm::ffi::TensorView probe_cursor,
      tvm::ffi::TensorView selected,
      tvm::ffi::TensorView claim_spins,
      tvm::ffi::TensorView claim_collisions,
      int64_t requester_rank,
      int64_t backoff_ns,
      int64_t timeout_clocks,
      tvm::ffi::TensorView error_state) {
    using namespace host;
    auto D = SymbolicSize{"owner_count"};
    auto C = SymbolicSize{"candidate_count"};
    auto S = SymbolicSize{"slot_count"};
    auto device = SymbolicDevice{};
    TensorMatcher({D}).with_dtype<uint64_t>().with_device<kDLGPU>(device).verify(owner_state_ptrs);
    TensorMatcher({C}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(candidate_owners).verify(candidate_slots);
    TensorMatcher({C}).with_dtype<uint8_t>().with_device<kDLGPU>(device).verify(done);
    TensorMatcher({S}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(comp_gen);
    TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(probe_cursor).verify(selected).verify(error_state);
    TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(claim_spins).verify(claim_collisions);
    RuntimeCheck(D.unwrap() > 1 && C.unwrap() > 0 && S.unwrap() > 0, "invalid SiDP claim dimensions");
    LaunchKernel(1, 1, device.unwrap())(
        claim_owner_kernel,
        static_cast<const uint64_t*>(owner_state_ptrs.data_ptr()),
        static_cast<int32_t>(D.unwrap()),
        static_cast<const int32_t*>(candidate_owners.data_ptr()),
        static_cast<const int32_t*>(candidate_slots.data_ptr()),
        static_cast<uint8_t*>(done.data_ptr()),
        static_cast<int32_t>(C.unwrap()),
        static_cast<int32_t*>(comp_gen.data_ptr()),
        static_cast<int32_t>(required_comp_gen),
        static_cast<int32_t*>(probe_cursor.data_ptr()),
        static_cast<int32_t*>(selected.data_ptr()),
        static_cast<int64_t*>(claim_spins.data_ptr()),
        static_cast<int64_t*>(claim_collisions.data_ptr()),
        static_cast<int32_t>(requester_rank),
        static_cast<int32_t>(backoff_ns),
        static_cast<uint64_t>(timeout_clocks),
        static_cast<int32_t*>(error_state.data_ptr()));
  }

  static void copy_selected(
      tvm::ffi::TensorView src_ptrs,
      tvm::ffi::TensorView dst_ptrs,
      tvm::ffi::TensorView sizes,
      tvm::ffi::TensorView selected,
      int64_t grid_blocks,
      tvm::ffi::TensorView error_state) {
    using namespace host;
    auto C = SymbolicSize{"candidate_count"};
    auto device = SymbolicDevice{};
    TensorMatcher({C}).with_dtype<uint64_t>().with_device<kDLGPU>(device).verify(src_ptrs).verify(dst_ptrs);
    TensorMatcher({C}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(sizes);
    TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(selected).verify(error_state);
    RuntimeCheck(C.unwrap() > 0 && grid_blocks > 0, "invalid SiDP SM copy dimensions");
    LaunchKernel(static_cast<uint32_t>(grid_blocks), kCopyBlockSize, device.unwrap())(
        copy_selected_kernel,
        static_cast<const uint64_t*>(src_ptrs.data_ptr()),
        static_cast<const uint64_t*>(dst_ptrs.data_ptr()),
        static_cast<const int64_t*>(sizes.data_ptr()),
        static_cast<int32_t>(C.unwrap()),
        static_cast<const int32_t*>(selected.data_ptr()),
        static_cast<int32_t*>(error_state.data_ptr()));
  }

  static void release_owner(
      tvm::ffi::TensorView owner_state_ptrs,
      tvm::ffi::TensorView candidate_owners,
      tvm::ffi::TensorView selected,
      int64_t requester_rank,
      tvm::ffi::TensorView error_state) {
    using namespace host;
    auto D = SymbolicSize{"owner_count"};
    auto C = SymbolicSize{"candidate_count"};
    auto device = SymbolicDevice{};
    TensorMatcher({D}).with_dtype<uint64_t>().with_device<kDLGPU>(device).verify(owner_state_ptrs);
    TensorMatcher({C}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(candidate_owners);
    TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(selected).verify(error_state);
    LaunchKernel(1, 1, device.unwrap())(
        release_owner_kernel,
        static_cast<const uint64_t*>(owner_state_ptrs.data_ptr()),
        static_cast<int32_t>(D.unwrap()),
        static_cast<const int32_t*>(candidate_owners.data_ptr()),
        static_cast<int32_t>(C.unwrap()),
        static_cast<const int32_t*>(selected.data_ptr()),
        static_cast<int32_t>(requester_rank),
        static_cast<int32_t*>(error_state.data_ptr()));
  }

  static void publish_selected_fill(
      tvm::ffi::TensorView fill_gen,
      tvm::ffi::TensorView candidate_slots,
      tvm::ffi::TensorView selected,
      int64_t target_gen,
      tvm::ffi::TensorView error_state) {
    using namespace host;
    auto S = SymbolicSize{"slot_count"};
    auto C = SymbolicSize{"candidate_count"};
    auto device = SymbolicDevice{};
    TensorMatcher({S}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(fill_gen);
    TensorMatcher({C}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(candidate_slots);
    TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(selected).verify(error_state);
    LaunchKernel(1, 1, device.unwrap())(
        publish_selected_fill_kernel,
        static_cast<int32_t*>(fill_gen.data_ptr()),
        static_cast<const int32_t*>(candidate_slots.data_ptr()),
        static_cast<int32_t>(C.unwrap()),
        static_cast<const int32_t*>(selected.data_ptr()),
        static_cast<int32_t>(target_gen),
        static_cast<int32_t*>(error_state.data_ptr()));
  }

  static void wait_generation(
      tvm::ffi::TensorView generations,
      int64_t slot,
      int64_t target,
      int64_t backoff_ns,
      int64_t timeout_clocks,
      tvm::ffi::TensorView error_state) {
    using namespace host;
    auto S = SymbolicSize{"slot_count"};
    auto device = SymbolicDevice{};
    TensorMatcher({S}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(generations);
    TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(error_state);
    RuntimeCheck(slot >= 0 && slot < S.unwrap(), "invalid SiDP generation slot");
    LaunchKernel(1, 1, device.unwrap())(
        wait_generation_kernel,
        static_cast<int32_t*>(generations.data_ptr()),
        static_cast<int32_t>(slot),
        static_cast<int32_t>(target),
        static_cast<int32_t>(backoff_ns),
        static_cast<uint64_t>(timeout_clocks),
        static_cast<int32_t*>(error_state.data_ptr()));
  }

  static void publish_generation(
      tvm::ffi::TensorView generations, int64_t slot, int64_t target) {
    using namespace host;
    auto S = SymbolicSize{"slot_count"};
    auto device = SymbolicDevice{};
    TensorMatcher({S}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(generations);
    RuntimeCheck(slot >= 0 && slot < S.unwrap(), "invalid SiDP generation slot");
    LaunchKernel(1, 1, device.unwrap())(
        publish_generation_kernel,
        static_cast<int32_t*>(generations.data_ptr()),
        static_cast<int32_t>(slot),
        static_cast<int32_t>(target));
  }

  static void record_trace(
      tvm::ffi::TensorView selected,
      tvm::ffi::TensorView claim_spins,
      tvm::ffi::TensorView claim_collisions,
      tvm::ffi::TensorView selected_trace,
      tvm::ffi::TensorView spins_trace,
      tvm::ffi::TensorView collisions_trace,
      int64_t trace_index) {
    using namespace host;
    auto T = SymbolicSize{"trace_size"};
    auto device = SymbolicDevice{};
    TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(selected);
    TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(claim_spins).verify(claim_collisions);
    TensorMatcher({T}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(selected_trace);
    TensorMatcher({T}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(spins_trace).verify(collisions_trace);
    RuntimeCheck(trace_index >= 0 && trace_index < T.unwrap(), "invalid SiDP trace index");
    LaunchKernel(1, 1, device.unwrap())(
        record_trace_kernel,
        static_cast<const int32_t*>(selected.data_ptr()),
        static_cast<const int64_t*>(claim_spins.data_ptr()),
        static_cast<const int64_t*>(claim_collisions.data_ptr()),
        static_cast<int32_t*>(selected_trace.data_ptr()),
        static_cast<int64_t*>(spins_trace.data_ptr()),
        static_cast<int64_t*>(collisions_trace.data_ptr()),
        static_cast<int32_t>(trace_index));
  }

  static int64_t native_peer_atomic_supported(int64_t device, int64_t peer) {
    int value = 0;
    const cudaError_t error = cudaDeviceGetP2PAttribute(
        &value,
        cudaDevP2PAttrNativeAtomicSupported,
        static_cast<int>(device),
        static_cast<int>(peer));
    if (error != cudaSuccess) return 0;
    return static_cast<int64_t>(value);
  }
};

}  // namespace sidp
