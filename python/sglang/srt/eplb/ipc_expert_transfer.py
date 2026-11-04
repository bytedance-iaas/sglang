"""Same-node CUDA IPC pull into staging. Keeps source tensors alive.
All ranks must finish reads before any active slot is overwritten.
"""

import ctypes as C

import torch
import torch.distributed as dist


class Handle(C.Structure):
    _fields_ = [("reserved", C.c_byte * 64)]


def check(code, where):
    if code:
        raise RuntimeError(f"{where}: CUDA error {code}")


_runtime = None
_open = {}


class Runtime:
    def __init__(self):
        self.rt = C.CDLL("libcudart.so")
        self.driver = C.CDLL("libcuda.so.1")
        specs = {
            "cudaIpcGetMemHandle": ([C.POINTER(Handle), C.c_void_p], C.c_int),
            "cudaIpcOpenMemHandle": (
                [C.POINTER(C.c_void_p), Handle, C.c_uint],
                C.c_int,
            ),
            "cudaIpcCloseMemHandle": ([C.c_void_p], C.c_int),
        }
        for name, (args, ret) in specs.items():
            fn = getattr(self.rt, name)
            fn.argtypes = args
            fn.restype = ret
        self.driver.cuMemGetAddressRange_v2.argtypes = [
            C.POINTER(C.c_uint64),
            C.POINTER(C.c_size_t),
            C.c_uint64,
        ]
        self.driver.cuMemGetAddressRange_v2.restype = C.c_int

    def export(self, t):
        base = C.c_uint64()
        size = C.c_size_t()
        check(
            self.driver.cuMemGetAddressRange_v2(
                C.byref(base), C.byref(size), t.data_ptr()
            ),
            "cuMemGetAddressRange",
        )
        handle = Handle()
        check(
            self.rt.cudaIpcGetMemHandle(C.byref(handle), C.c_void_p(base.value)),
            "cudaIpcGetMemHandle",
        )
        return dict(
            handle=bytes(handle).hex(),
            offset=t.data_ptr() - base.value,
            stride=t.stride(0) * t.element_size(),
            row_bytes=t[:1].nbytes,
            allocation_bytes=size.value,
        )

    def open(self, d):
        h = d["handle"]
        if h not in _open:
            p = C.c_void_p()
            check(
                self.rt.cudaIpcOpenMemHandle(
                    C.byref(p), Handle.from_buffer_copy(bytes.fromhex(h)), 1
                ),
                "cudaIpcOpenMemHandle",
            )
            _open[h] = p.value
        return _open[h] + d["offset"]


def runtime():
    global _runtime
    if _runtime is None:
        _runtime = Runtime()
    return _runtime


class ExpertIpcPeers:
    """Keep registered source allocations alive for GPU-driven peer reads."""

    def __init__(self, tensors, cpu_group, rank):
        self.tensors = tensors
        rt = runtime()
        torch.cuda.synchronize()
        local = [rt.export(t) for t in tensors]
        metadata = [None] * dist.get_world_size(cpu_group)
        dist.all_gather_object(metadata, local, group=cpu_group)
        self.ptrs = {
            peer: [rt.open(desc) for desc in descriptions]
            for peer, descriptions in enumerate(metadata)
            if peer != rank
        }
