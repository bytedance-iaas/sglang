#include <early_moe/runtime/sm90_early_moe.hpp>

#include <cstring>
#include <memory>

using namespace early_moe;
static thread_local std::string error;
#define TRY \
  try {     \
    error.clear();
#define CATCH                       \
  }                                 \
  catch (const std::exception& e) { \
    error = e.what();               \
    return -1;                      \
  }                                 \
  return 0;
extern "C" const char* early_error() {
  return error.c_str();
}
extern "C" size_t early_workspace_size() {
  return sizeof(Workspace);
}
extern "C" size_t early_call_size() {
  return sizeof(Call);
}
extern "C" void early_geometry(int* values) {
  const int geometry[] = {H, I, K, E, S, BM, STAT_COUNT};
  std::memcpy(values, geometry, sizeof(geometry));
}
extern "C" int early_create(
    void** out,
    void* a1,
    void* sf1,
    void* a2,
    void* sf2,
    void* w1,
    void* wsf1,
    void* w2,
    void* wsf2,
    int workers,
    int stages,
    int tile_n,
    int groups,
    int input_rows) {
  TRY auto r = std::make_unique<Runtime>(workers, stages, tile_n, groups);
  if (input_rows && (tile_n != 256 || input_rows < BM || input_rows % 128))
    throw std::runtime_error("bulk preparation requires BN256 and a 128-aligned input row capacity");
  r->args.input_rows = input_rows;
  r->args.a1 = static_cast<uint8_t*>(a1);
  r->args.sf1 = static_cast<float*>(sf1);
  r->args.a2 = static_cast<uint8_t*>(a2);
  r->args.sf2 = static_cast<float*>(sf2);
  r->args.wsf1 = static_cast<float*>(wsf1);
  r->args.wsf2 = static_cast<float*>(wsf2);
  r->args.ta1 = descriptor(a1, H, input_rows ? input_rows : BM * groups, BK, BM);
  r->args.tsf1 = descriptor(sf1, input_rows ? input_rows : 128 * groups, H / 128, BM, 1, true, false);
  r->args.tw1 = descriptor(w1, H, S * 2 * I, BK, tile_n);
  r->args.tout1 = descriptor(a2, I, BM * groups, tile_n == 256 ? 64 : tile_n / 2, BM, false, false);
  r->args.ta2 = descriptor(a2, I, BM * groups, BK, BM);
  r->args.tsf2 = descriptor(sf2, 128 * groups, I / 64, BM, 1, true, false);
  r->args.tw2 = descriptor(w2, I, S * H, BK, tile_n);
  *out = r.release();
  CATCH
}
extern "C" int early_run(void* ctx, const Call* call, void* stream) {
  TRY static_cast<Runtime*>(ctx)->launch(*call, static_cast<cudaStream_t>(stream));
  CATCH
}
extern "C" int early_busy(void* ctx) {
  try {
    return static_cast<Runtime*>(ctx)->busy() ? 1 : 0;
  } catch (const std::exception& e) {
    error = e.what();
    return -1;
  }
}
extern "C" int early_destroy(void* ctx) {
  TRY auto* r = static_cast<Runtime*>(ctx);
  if (r->busy()) throw std::runtime_error("cannot destroy in-flight context");
  delete r;
  CATCH
}
extern "C" int early_info(void* ctx, int* values) {
  TRY auto* r = static_cast<Runtime*>(ctx);
  cudaFuncAttributes a;
  check(cudaFuncGetAttributes(&a, r->kernel));
  values[0] = r->shared;
  values[1] = r->capacity;
  values[2] = a.numRegs;
  values[3] = a.localSizeBytes;
  values[4] = r->threads;
  values[5] = r->workers + 1;
  values[6] = r->tile_n;
  values[7] = r->groups;
  values[8] = r->tile_n == 256 ? 0 : r->workers / r->groups;
  CATCH
}
extern "C" int early_export(void* ptr, char* handle, uint64_t* offset) {
  TRY CUdeviceptr base;
  size_t bytes;
  auto e = cuMemGetAddressRange(&base, &bytes, reinterpret_cast<CUdeviceptr>(ptr));
  if (e != CUDA_SUCCESS) throw std::runtime_error("cannot export CUDA allocation base");
  cudaIpcMemHandle_t h;
  check(cudaIpcGetMemHandle(&h, reinterpret_cast<void*>(base)));
  std::memcpy(handle, &h, sizeof(h));
  *offset = reinterpret_cast<uint64_t>(ptr) - base;
  CATCH
}
extern "C" int early_open(const char* handle, uint64_t offset, int peer, void** base, void** ptr) {
  TRY int dev, access, atomics;
  check(cudaGetDevice(&dev));
  check(cudaDeviceCanAccessPeer(&access, dev, peer));
  check(cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported, dev, peer));
  if (!access || !atomics) throw std::runtime_error("peer access and native system atomics required");
  cudaIpcMemHandle_t h;
  std::memcpy(&h, handle, sizeof(h));
  check(cudaIpcOpenMemHandle(base, h, cudaIpcMemLazyEnablePeerAccess));
  *ptr = static_cast<char*>(*base) + offset;
  CATCH
}
extern "C" int early_close(void* base) {
  TRY check(cudaIpcCloseMemHandle(base));
  CATCH
}

// Benchmark only: independent IPC allocation, independent epoch, before target.
__global__ void early_benchmark_start_delay(
    const uint64_t* peers, uint64_t* own, int world, int rank, uint64_t epoch, uint64_t delay_ns) {
  if (threadIdx.x) return;
  peer_store(own, epoch);
  auto* leader = reinterpret_cast<uint64_t*>(peers[0]);
  if (!rank) {
    for (int r = 0; r < world; ++r)
      while (peer_load(reinterpret_cast<uint64_t*>(peers[r])) < epoch)
        __nanosleep(64);
    peer_store(leader + 1, epoch);
  } else {
    while (peer_load(leader + 1) < epoch)
      __nanosleep(64);
  }
  uint64_t start = now();
  while (now() - start < delay_ns)
    __nanosleep(64);
}
extern "C" int
early_delay(void* peers, void* own, int world, int rank, uint64_t epoch, uint64_t delay_ns, void* stream) {
  TRY early_benchmark_start_delay<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<uint64_t*>(peers), static_cast<uint64_t*>(own), world, rank, epoch, delay_ns);
  check(cudaGetLastError());
  CATCH
}

// External communication stream enqueues this marker after its actual barrier.
// It is outside the one-kernel compute operator, like the barrier itself.
__global__ void early_publish_stop(uint64_t* signal, uint64_t epoch) {
  if (!threadIdx.x) peer_store(signal, epoch);
}
extern "C" int early_signal_stop(void* signal, uint64_t epoch, void* stream) {
  TRY early_publish_stop<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<uint64_t*>(signal), epoch);
  check(cudaGetLastError());
  CATCH
}

// Benchmark only: concurrent stream models barrier completion after the target
// has entered. All input preparation precedes both streams via a CUDA event.
__global__ void early_benchmark_external_stop(
    const uint64_t* peers,
    uint64_t* own,
    int world,
    int rank,
    uint64_t epoch,
    uint64_t delay_ns,
    uint64_t* target_ready,
    uint64_t* signal) {
  if (threadIdx.x) return;
  peer_store(own, epoch);
  auto* leader = reinterpret_cast<uint64_t*>(peers[0]);
  if (!rank) {
    for (int r = 0; r < world; ++r)
      while (peer_load(reinterpret_cast<uint64_t*>(peers[r])) < epoch)
        __nanosleep(64);
    peer_store(leader + 1, epoch);
  } else {
    while (peer_load(leader + 1) < epoch)
      __nanosleep(64);
  }
  while (peer_load(target_ready) < epoch)
    __nanosleep(64);
  const uint64_t start = now();
  while (now() - start < delay_ns)
    __nanosleep(64);
  // Benchmark diagnostics only: same-GPU timestamp immediately before the
  // stop store, retained for the last invocation in each measured batch.
  peer_store(own + 2, now());
  peer_store(signal, epoch);
}
extern "C" int early_delay_stop(
    void* peers,
    void* own,
    int world,
    int rank,
    uint64_t epoch,
    uint64_t delay_ns,
    void* ready,
    void* signal,
    void* stream) {
  TRY early_benchmark_external_stop<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<uint64_t*>(peers),
      static_cast<uint64_t*>(own),
      world,
      rank,
      epoch,
      delay_ns,
      static_cast<uint64_t*>(ready),
      static_cast<uint64_t*>(signal));
  check(cudaGetLastError());
  CATCH
}
