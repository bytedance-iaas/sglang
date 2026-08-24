"""ctypes wrapper for cudaMemcpyAsync — the only copy path that can be captured into CUDA graph (D7)."""

import ctypes
import os
from typing import Optional


def _find_libcudart() -> str:
    """Find libcudart.so from /proc/self/maps, skipping stubs."""
    candidates = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libcudart" not in line or "/" not in line:
                    continue
                path = line[line.index("/") :].strip()
                if path.endswith(" (deleted)"):
                    path = path[: -len(" (deleted)")]
                if os.path.basename(path).startswith("libcudart"):
                    candidates.append(path)
    except FileNotFoundError:
        pass

    for path in candidates:
        if "stub" not in os.path.basename(path):
            return path
    if candidates:
        return candidates[0]

    for search in [
        os.environ.get("CUDA_HOME", ""),
        "/usr/local/cuda",
        "/usr/lib/x86_64-linux-gnu",
    ]:
        if not search:
            continue
        for root, _, files in os.walk(os.path.join(search, "lib")):
            for f in files:
                if f.startswith("libcudart.so"):
                    return os.path.join(root, f)
    raise RuntimeError("Cannot find libcudart.so")


class SidpCudaMemcpy:
    """Thin ctypes wrapper around cudaMemcpyAsync + cudaDeviceEnablePeerAccess."""

    _CUDA_MEMCPY_DEFAULT = 4  # cudaMemcpyDefault

    def __init__(self, so_path: Optional[str] = None):
        if so_path is None:
            so_path = _find_libcudart()
        self._lib = ctypes.CDLL(so_path)

        # cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
        #                             cudaMemcpyKind kind, cudaStream_t stream)
        fn = self._lib.cudaMemcpyAsync
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._memcpy_async = fn

        # cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
        fn2 = self._lib.cudaDeviceEnablePeerAccess
        fn2.restype = ctypes.c_int
        fn2.argtypes = [ctypes.c_int, ctypes.c_uint]
        self._enable_peer_access = fn2

    def async_copy(self, dst_ptr: int, src_ptr: int, nbytes: int, stream_ptr: int):
        rc = self._memcpy_async(
            dst_ptr, src_ptr, nbytes, self._CUDA_MEMCPY_DEFAULT, stream_ptr
        )
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync failed with error code {rc}")

    def enable_peer_access(self, peer_device: int):
        rc = self._enable_peer_access(peer_device, 0)
        # 704 = cudaErrorPeerAccessAlreadyEnabled
        if rc not in (0, 704):
            raise RuntimeError(
                f"cudaDeviceEnablePeerAccess({peer_device}) failed: {rc}"
            )
