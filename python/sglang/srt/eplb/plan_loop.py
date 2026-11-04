"""Exact parallel K1 search, device WHILE stops at native active=false or 16 rounds.

The body remains two parallel kernels; this is graph control fusion, not one
monolithic compute kernel. The select kernel also publishes WHILE continuation.
Inputs are live, persistent per-layer GPU mapping/quota buffers.
"""

import ctypes as C
from functools import lru_cache

import torch

from sglang.srt.eplb.balance_config import CUDA_FLAGS

from ._ffi import check, stream_ptr


@lru_cache(maxsize=None)
def library(shape):
    from ._native import build_library

    lib = build_library("plan_loop", tuple(CUDA_FLAGS + shape.cuda_flags))
    m = C.CDLL(str(lib))
    p = C.c_void_p
    i = C.c_int
    specs = {
        "plan_create": [C.POINTER(p), C.POINTER(p), i],
        "plan_begin": [p, i, p],
        "plan_end": [p],
        "plan_init": [p, p],
        "plan_body": [p, p],
        "plan_finish": [p, p],
        "plan_add_while": [p],
        "plan_instantiate": [p],
        "plan_launch": [p, p],
        "plan_destroy": [p],
    }
    for name, args in specs.items():
        f = getattr(m, name)
        f.argtypes = args
        f.restype = i
    return m


class PlanLoop:
    def __init__(self, p, m, q):
        self.lib = library(p.shape)
        self.handle = C.c_void_p()
        self.inputs = (m.data_ptr(), q.data_ptr())
        self.refs = (m, q, p)
        pointers = (C.c_void_p * 10)(
            *[
                t.data_ptr()
                for t in [
                    m,
                    p.baseline_info,
                    p.mapping,
                    p.quota,
                    p.info,
                    p.copies,
                    p.counter,
                    p.history,
                    p.mmwork,
                    p.mmkeys,
                ]
            ]
        )
        check(self.lib.plan_create(C.byref(self.handle), pointers, p.budget))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            p.mod.plan(m, q, p.work, p.keys, p.baseline_info, p.mapping, p.quota)
            stream.synchronize()
            check(self.lib.plan_begin(self.handle, 0, stream_ptr()))
            p.mod.plan(m, q, p.work, p.keys, p.baseline_info, p.mapping, p.quota)
            check(self.lib.plan_init(self.handle, stream_ptr()))
            check(self.lib.plan_end(stream_ptr()))
            check(self.lib.plan_add_while(self.handle))
            check(self.lib.plan_begin(self.handle, 1, stream_ptr()))
            check(self.lib.plan_body(self.handle, stream_ptr()))
            check(self.lib.plan_end(stream_ptr()))
            check(self.lib.plan_begin(self.handle, 2, stream_ptr()))
            check(self.lib.plan_finish(self.handle, stream_ptr()))
            check(self.lib.plan_end(stream_ptr()))
            check(self.lib.plan_instantiate(self.handle))
        torch.cuda.current_stream().wait_stream(stream)

    def launch(self, m, q):
        assert self.inputs == (
            m.data_ptr(),
            q.data_ptr(),
        ), "persistent planner input changed"
        check(self.lib.plan_launch(self.handle, stream_ptr()))

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.plan_destroy(self.handle)
