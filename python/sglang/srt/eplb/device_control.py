"""GPU exact K1 planner and same-forward device control; experiment only."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.eplb.balance_config import BLOCK_SIZE, CUDA_FLAGS


@cache_once
def module(shape):
    return load_jit(
        f"block_device_control_{shape.key}_b{BLOCK_SIZE}",
        extra_cuda_cflags=CUDA_FLAGS + shape.cuda_flags,
        cuda_files=["lplb/device_control.cuh"],
        cuda_wrappers=[
            (n, "dc_" + n)
            for n in ["plan", "metadata", "copy", "counts", "route", "snapshot"]
        ],
    )


class DevicePlanner:
    def __init__(self, device, shape):
        self.shape = shape
        self.mod = module(shape)
        self.work = torch.empty(shape.device_work, dtype=torch.int32, device=device)
        self.keys = torch.empty((shape.candidates, 8), dtype=torch.int64, device=device)
        self.info = torch.empty(9, dtype=torch.int32, device=device)
        self.mapping = torch.empty(shape.physical, dtype=torch.int64, device=device)
        self.quota = torch.empty(shape.physical, dtype=torch.int32, device=device)

    def plan(self, m, q):
        self.mod.plan(m, q, self.work, self.keys, self.info, self.mapping, self.quota)
        return self.info, self.mapping, self.quota
