"""Independent CPU integer oracle; never used in model timing."""

import ctypes as C
from functools import lru_cache

from sglang.srt.eplb._native import build_library
from sglang.srt.eplb.balance_config import CUDA_FLAGS


@lru_cache(None)
def library(shape):
    lib = C.CDLL(
        str(
            build_library(
                "multi_swap_cpu", tuple(CUDA_FLAGS + shape.cuda_flags), cpu=True
            )
        )
    )
    lib.plan_multi.argtypes = [C.POINTER(C.c_int64), C.POINTER(C.c_int64), C.c_int] + [
        C.POINTER(C.c_int64)
    ] * 6
    lib.plan_multi.restype = C.c_int
    return lib


def plan(mapping, counts, budget, shape):
    assert len(mapping) == len(counts) == shape.physical
    arr = C.c_int64 * shape.physical
    m, q = arr(*mapping), arr(*counts)
    outm, outq, info = arr(), arr(), (C.c_int64 * 12)()
    copies = (C.c_int64 * (2 * shape.max_copies))()
    cnt = (C.c_int64 * (2 * shape.world))()
    sw = (C.c_int64 * 32)()
    n = library(shape).plan_multi(m, q, budget, outm, outq, info, copies, cnt, sw)
    assert n >= 0, n
    return dict(
        mapping=list(outm),
        quota=list(outq),
        info=list(info),
        copies=[
            dict(
                source=copies[2 * i],
                destination=copies[2 * i + 1],
                hot=m[copies[2 * i]],
            )
            for i in range(n)
        ],
        counts=list(cnt),
        swaps=[list(sw[2 * i : 2 * i + 2]) for i in range(info[1])],
    )
