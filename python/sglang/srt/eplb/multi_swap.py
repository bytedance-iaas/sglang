"""Experimental per-layer, per-rank send/receive budget; GPU joint-swap planner."""

import os

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.eplb.balance_config import BLOCK_SIZE, CUDA_FLAGS
from sglang.srt.eplb.device_control import DevicePlanner


@cache_once
def module(shape):
    return load_jit(
        f"block_multi_swap_{shape.key}_b{BLOCK_SIZE}",
        extra_cuda_cflags=CUDA_FLAGS + shape.cuda_flags,
        cuda_files=["lplb/multi_swap.cuh"],
        cuda_wrappers=[
            (n, "mm_" + n) for n in ["improve", "parallel", "copy", "snapshot"]
        ],
    )


class MultiPlanner(DevicePlanner):
    def __init__(self, device, budget, shape):
        super().__init__(device, shape)
        assert 1 <= budget <= 4
        self.budget = budget
        self.mm = module(shape)
        self.baseline_info = self.info
        self.info = torch.empty(12, dtype=torch.int32, device=device)
        self.route_info = self.info[:9]
        self.parallel = os.getenv("WAN22_MULTI_PARALLEL", "1") == "1"
        self.graph_enabled = os.getenv("WAN22_MULTI_GRAPH", "1") == "1"
        self.graph = None
        self.capture_stream = torch.cuda.Stream(device=device)
        self.mmwork = torch.empty(shape.multi_work, dtype=torch.int32, device=device)
        self.mmkeys = torch.empty((64, 6), dtype=torch.int64, device=device)
        self.input_m = torch.empty(shape.physical, dtype=torch.int64, device=device)
        self.input_q = torch.empty(shape.physical, dtype=torch.int32, device=device)
        self.copies = torch.empty(
            (shape.max_copies, 2), dtype=torch.int32, device=device
        )
        self.counter = torch.empty(2 * shape.world, dtype=torch.int32, device=device)
        self.history = torch.empty((16, 2), dtype=torch.int32, device=device)

    def _run(self, m, q):
        self.mod.plan(
            m, q, self.work, self.keys, self.baseline_info, self.mapping, self.quota
        )
        if self.parallel:
            self.mm.parallel(
                m,
                q,
                self.baseline_info,
                self.mapping,
                self.quota,
                self.info,
                self.copies,
                self.counter,
                self.history,
                self.mmwork,
                self.mmkeys,
                self.budget,
            )
        else:
            self.mm.improve(
                m,
                q,
                self.baseline_info,
                self.mapping,
                self.quota,
                self.info,
                self.copies,
                self.counter,
                self.history,
                self.budget,
            )

    def plan(self, m, q):
        if os.getenv("WAN22_PLAN_LOOP", "0") == "1":
            from sglang.srt.eplb.plan_loop import PlanLoop

            # Keep graph inputs stable even when the caller uses fresh routing
            # buffers for a different token count.
            self.input_m.copy_(m)
            self.input_q.copy_(q)
            if not hasattr(self, "loop_graph"):
                self.loop_graph = PlanLoop(self, self.input_m, self.input_q)
            self.loop_graph.launch(self.input_m, self.input_q)
            return self.info, self.mapping, self.quota
        if not self.graph_enabled:
            self._run(m, q)
        else:
            self.input_m.copy_(m)
            self.input_q.copy_(q)
            if self.graph is None:
                self.capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.capture_stream):
                    self._run(self.input_m, self.input_q)
                self.capture_stream.synchronize()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=self.capture_stream):
                    self._run(self.input_m, self.input_q)
                torch.cuda.current_stream().wait_stream(self.capture_stream)
            self.graph.replay()
        return self.info, self.mapping, self.quota
