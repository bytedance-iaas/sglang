"""Cross-member device barrier for SiDP Direction A coordinated_static (Phase 2).

A sense-reversing counting barrier implemented as a single ordinary CUDA kernel
so it can be captured into the SiDP comm-stream CUDA Graph as a normal node (no
conditional node, no host involvement). Validated mechanism: see
``cudagraph_preempt/AB方向技术验证说明.md`` (A.1/A.4) and its ``dev_barrier``
kernel, which reached ~99% of the lockstep bandwidth ceiling on 8xH20.

Layout (``bar`` is an int[2] living in one member's HBM, IPC-shared to all):
    bar[0] = arrive count, bar[1] = sense (generation).
``nptr`` is a device int giving the number of participating members; reading it
at kernel runtime (not baking it as an immediate) is what keeps the barrier
correct under CUDA Graph replay and lets the participant count change between
forwards (Direction A dynamic nptr, a later phase).

All atomics are ``_system`` scoped: the barrier synchronizes across separate
processes/GPUs over NVLink/NVSwitch, exactly like the SiDP owner atomics.
"""

from torch.utils.cpp_extension import load_inline

# The kernel mirrors the validated dev_barrier. It is launched <<<1,1>>>: a
# single thread per member arrives and spins, which is all a counting barrier
# needs. backoff_ns paces the spin via __nanosleep.
_CUDA_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void sidp_dev_barrier(int* bar, const int* nptr,
                                            int depth, int bpf, int backoff_ns) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int my_sense = atomicAdd_system(&bar[1], 0);         // read current generation
    // Generational nptr ring (Direction A dynamic nptr). bar[1] is the global
    // completed-barrier count; this barrier's *forward* index is my_sense / bpf
    // (bpf = barriers launched per forward, a constant). All bpf barriers in one
    // forward map to the same slot, and consecutive forwards map to different
    // slots. Reading the per-forward slot means a later forward's set_live_nptr
    // write (to a DIFFERENT slot) cannot clobber the participant count this
    // in-flight barrier still needs -- the overwrite race that deadlocked us.
    int n = nptr[(my_sense / bpf) % depth];              // participants (runtime)
    __threadfence_system();
    int cnt = atomicAdd_system(&bar[0], 1) + 1;          // announce arrival
    if (cnt == n) {                                      // last arriver:
        atomicExch_system(&bar[0], 0);                   //   reset arrival count
        __threadfence_system();
        atomicAdd_system(&bar[1], 1);                    //   flip generation -> release
    } else {
        while (atomicAdd_system(&bar[1], 0) == my_sense) // spin until released
            __nanosleep(backoff_ns);
    }
}
"""

_CPP_SRC = r"""
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <utility>
#include <stdexcept>

void sidp_launch_dev_barrier(uintptr_t bar_ptr, uintptr_t nptr_ptr,
                             int depth, int bpf, int backoff_ns,
                             uintptr_t stream_ptr);
// Allocate the shared bar[2] on the current device, zero it, and return
// (device_ptr, 64-byte cudaIpcMemHandle_t as hex). Passing/creating the handle
// by value is done in native C++ (ctypes byval-struct passing is unreliable).
std::pair<uintptr_t, std::string> sidp_alloc_shared_bar(int nbytes);
// Open a peer's bar from its hex handle; returns the local device pointer.
uintptr_t sidp_open_shared_bar(const std::string& hex);
"""

_LAUNCHER_SRC = r"""
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <stdexcept>

extern "C" __global__ void sidp_dev_barrier(int* bar, const int* nptr,
                                            int depth, int bpf, int backoff_ns);

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(what) + " failed: " +
                                 cudaGetErrorString(e));
    }
}

static std::string to_hex(const unsigned char* p, int n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    s.resize(n * 2);
    for (int i = 0; i < n; ++i) {
        s[2 * i] = d[(p[i] >> 4) & 0xf];
        s[2 * i + 1] = d[p[i] & 0xf];
    }
    return s;
}

static void from_hex(const std::string& s, unsigned char* p, int n) {
    auto v = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
    };
    for (int i = 0; i < n; ++i) p[i] = (v(s[2 * i]) << 4) | v(s[2 * i + 1]);
}

void sidp_launch_dev_barrier(uintptr_t bar_ptr, uintptr_t nptr_ptr,
                             int depth, int bpf, int backoff_ns,
                             uintptr_t stream_ptr) {
    int* bar = reinterpret_cast<int*>(bar_ptr);
    const int* nptr = reinterpret_cast<const int*>(nptr_ptr);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    sidp_dev_barrier<<<1, 1, 0, stream>>>(bar, nptr, depth, bpf, backoff_ns);
}

std::pair<uintptr_t, std::string> sidp_alloc_shared_bar(int nbytes) {
    void* p = nullptr;
    ck(cudaMalloc(&p, nbytes), "cudaMalloc(bar)");
    ck(cudaMemset(p, 0, nbytes), "cudaMemset(bar)");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize(bar)");
    cudaIpcMemHandle_t h;
    ck(cudaIpcGetMemHandle(&h, p), "cudaIpcGetMemHandle(bar)");
    std::string hex = to_hex(reinterpret_cast<unsigned char*>(&h), sizeof(h));
    return std::make_pair(reinterpret_cast<uintptr_t>(p), hex);
}

uintptr_t sidp_open_shared_bar(const std::string& hex) {
    cudaIpcMemHandle_t h;
    if ((int)hex.size() != (int)sizeof(h) * 2) {
        throw std::runtime_error("bar handle hex length mismatch");
    }
    from_hex(hex, reinterpret_cast<unsigned char*>(&h), sizeof(h));
    void* p = nullptr;
    ck(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess),
       "cudaIpcOpenMemHandle(bar)");
    return reinterpret_cast<uintptr_t>(p);
}
"""


class SidpDeviceBarrier:
    """JIT-compiled sense-reversing device barrier, launchable on any stream.

    The barrier state (``bar``) and participant count (``nptr``) are device
    pointers owned by :class:`SidpManager` (allocated on one member and shared to
    the rest via CUDA IPC). This component only compiles and launches the kernel.
    """

    def __init__(self, backoff_ns: int = 200):
        self.backoff_ns = backoff_ns
        self._module = load_inline(
            name="sidp_device_barrier",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC + _LAUNCHER_SRC,
            functions=[
                "sidp_launch_dev_barrier",
                "sidp_alloc_shared_bar",
                "sidp_open_shared_bar",
            ],
            with_cuda=True,
            verbose=False,
        )

    def barrier_state_size_bytes(self) -> int:
        """Bytes needed for the shared ``bar`` buffer (int[2])."""
        return 2 * 4

    def alloc_shared_bar(self, nbytes: int) -> tuple[int, str]:
        """Owner: allocate+zero bar on this device; return (device_ptr, hex_handle).

        Done in native C++ so the 64-byte cudaIpcMemHandle_t is created and later
        passed by value correctly (the validated probe path).
        """
        ptr, hex_handle = self._module.sidp_alloc_shared_bar(nbytes)
        return int(ptr), hex_handle

    def open_shared_bar(self, hex_handle: str) -> int:
        """Non-owner: open a peer's bar from its hex handle; return device ptr.

        Uses cudaIpcOpenMemHandle(LazyEnablePeerAccess): the returned pointer is
        directly dereferenceable by the barrier kernel's atomicAdd_system.
        """
        return int(self._module.sidp_open_shared_bar(hex_handle))

    def launch(
        self,
        bar_ptr: int,
        nptr_ptr: int,
        depth: int,
        bpf: int,
        stream_ptr: int,
    ) -> None:
        """Enqueue one barrier on ``stream_ptr`` (capturable into a CUDA Graph).

        ``bar_ptr``/``nptr_ptr`` are raw device addresses (``data_ptr()``).
        ``nptr`` is a ring of ``depth`` int32 participant counts; the kernel
        indexes it by ``(bar[1] / bpf) % depth`` so each forward reads its own
        slot (``bpf`` = barriers launched per forward, a captured constant).
        ``depth``/``bpf`` are immediates baked at capture; they never change
        after setup, so baking them is graph-safe (only the ring *contents* and
        ``bar`` are read at runtime).
        """
        self._module.sidp_launch_dev_barrier(
            bar_ptr, nptr_ptr, depth, bpf, self.backoff_ns, stream_ptr
        )

