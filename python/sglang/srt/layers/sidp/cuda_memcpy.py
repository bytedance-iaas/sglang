"""ctypes wrapper for cudaMemcpyAsync — the only copy path that can be captured into CUDA graph (D7)."""

import ctypes
import os
from typing import Optional


class _CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


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
    _CUDA_IPC_LAZY_ENABLE_PEER_ACCESS = 1

    def __init__(self, so_path: Optional[str] = None):
        if so_path is None:
            so_path = _find_libcudart()
        self._lib = ctypes.CDLL(so_path)
        self._driver = ctypes.CDLL("libcuda.so.1")

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

        # Raw requester-context mappings are needed by SM kernels. Use the
        # driver API so CUDA 13's tagged PyTorch sharing handle stays out of the
        # wire format; CUipcMemHandle itself remains a fixed 64-byte object.
        get_range = self._driver.cuMemGetAddressRange_v2
        get_range.restype = ctypes.c_int
        get_range.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_uint64,
        ]
        self._mem_get_address_range = get_range

        get_handle = self._driver.cuIpcGetMemHandle
        get_handle.restype = ctypes.c_int
        get_handle.argtypes = [
            ctypes.POINTER(_CudaIpcMemHandle),
            ctypes.c_uint64,
        ]
        self._ipc_get_mem_handle = get_handle

        open_handle = self._driver.cuIpcOpenMemHandle_v2
        open_handle.restype = ctypes.c_int
        open_handle.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            _CudaIpcMemHandle,
            ctypes.c_uint,
        ]
        self._ipc_open_mem_handle = open_handle

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

    def open_ipc_allocation(self, handle_bytes: bytes) -> int:
        """Open one raw IPC allocation in the current CUDA device context."""
        if len(handle_bytes) != ctypes.sizeof(_CudaIpcMemHandle):
            raise RuntimeError("invalid CUDA IPC memory handle size")
        handle = _CudaIpcMemHandle()
        ctypes.memmove(
            ctypes.addressof(handle), handle_bytes, ctypes.sizeof(_CudaIpcMemHandle)
        )
        allocation_base = ctypes.c_uint64()
        rc = self._ipc_open_mem_handle(
            ctypes.byref(allocation_base),
            handle,
            self._CUDA_IPC_LAZY_ENABLE_PEER_ACCESS,
        )
        if rc != 0 or not allocation_base.value:
            raise RuntimeError(f"cudaIpcOpenMemHandle failed with error code {rc}")
        return allocation_base.value

    def export_ipc_pointer(self, device_ptr: int, nbytes: int) -> dict:
        """Export a tensor pointer as a driver IPC handle plus byte offset."""
        allocation_base = ctypes.c_uint64()
        allocation_size = ctypes.c_size_t()
        rc = self._mem_get_address_range(
            ctypes.byref(allocation_base),
            ctypes.byref(allocation_size),
            ctypes.c_uint64(device_ptr),
        )
        if rc != 0 or not allocation_base.value:
            raise RuntimeError(f"cuMemGetAddressRange failed with error code {rc}")
        offset = device_ptr - allocation_base.value
        if offset < 0 or offset + nbytes > allocation_size.value:
            raise RuntimeError(
                "CUDA IPC component exceeds its backing allocation: "
                f"offset={offset}, nbytes={nbytes}, allocation={allocation_size.value}"
            )

        handle = _CudaIpcMemHandle()
        rc = self._ipc_get_mem_handle(
            ctypes.byref(handle), ctypes.c_uint64(allocation_base.value)
        )
        if rc != 0:
            raise RuntimeError(f"cuIpcGetMemHandle failed with error code {rc}")
        return {
            "handle": bytes(handle.reserved),
            "offset": offset,
            "nbytes": nbytes,
        }

    def open_ipc_pointer(self, descriptor: dict) -> int:
        """Open one raw IPC descriptor in the current CUDA device context."""
        allocation_base = self.open_ipc_allocation(descriptor["handle"])
        return allocation_base + int(descriptor["offset"])
