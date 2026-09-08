"""Production conditional-DMA builder: eager and torch capture, no demo kernels.

Single-GPU pseudo-owners isolate Graph wiring, main/extras, tail bytes, fixed
slots and WAR generations. Real peer arbitration is covered by SiDP DP runs.
"""

import pytest
import torch

from sglang.kernels.ops.sidp import (
    load_sidp_sm_copy_module,
    publish_generation,
    reset_forward_state,
    wait_generation,
)
from sglang.kernels.ops.sidp.dma_graph import load_sidp_dma_graph_module
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or tuple(int(v) for v in (torch.version.cuda or "0.0").split(".")[:2]) < (12, 8),
    reason="CUDA 12.8+ SWITCH support required",
)
def test_conditional_dma_eager_capture_fixed_slots_extras_and_reuse():
    module = load_sidp_dma_graph_module()
    load_sidp_sm_copy_module()
    device = torch.cuda.current_device()
    state = [torch.full((1,), -1, dtype=torch.int32, device=device) for _ in range(3)]
    owner_ptrs = torch.tensor(
        [s.data_ptr() for s in state], dtype=torch.uint64, device=device
    )
    owners = torch.tensor([1, 2], dtype=torch.int32, device=device)
    slots = torch.tensor([0, 1], dtype=torch.int32, device=device)
    done = torch.empty(2, dtype=torch.uint8, device=device)
    fill = torch.zeros(2, dtype=torch.int32, device=device)
    comp = torch.zeros_like(fill)
    cursor = torch.zeros(1, dtype=torch.int32, device=device)
    selected = torch.empty_like(cursor)
    error = torch.zeros_like(cursor)
    spins = torch.zeros(1, dtype=torch.int64, device=device)
    collisions = torch.zeros_like(spins)
    sources = [
        [torch.full((n,), n, dtype=torch.uint8, device=device) for n in sizes]
        for sizes in ((37, 67), (5, 13))
    ]
    destinations = [
        [torch.zeros_like(src) for src in component] for component in sources
    ]
    copies = torch.tensor(
        [
            [
                (src.data_ptr(), dst.data_ptr(), src.nbytes)
                for src, dst in zip(srcs, dsts)
            ]
            for srcs, dsts in zip(sources, destinations)
        ],
        dtype=torch.uint64,
        device="cpu",
    )

    def args(required, target):
        return (
            owner_ptrs,
            owners,
            slots,
            done,
            fill,
            comp,
            cursor,
            selected,
            spins,
            collisions,
            error,
            copies,
            required,
            target,
            0,
            500,
            10**12,
            selected,
            spins,
            collisions,
            copies,
            -1,
            copies,
            selected,
            selected,
            -1,
        )

    main = torch.cuda.Stream()
    comm = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(comm):
        eager = module.create(*args(0, 1))
        module.launch(eager, selected)
    comm.synchronize()
    for srcs, dsts in zip(sources, destinations):
        for src, dst in zip(srcs, dsts):
            assert torch.equal(src, dst)
    module.destroy(eager)

    graphs = []
    outputs = []
    # Different graphs share stable buffers but not conditional handles. No
    # owner reset during capture or replay, just local per-forward bookkeeping.
    for _ in range(2):
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        captured_outputs = []
        with torch.cuda.graph(graph, stream=main):
            reset_forward_state(fill, comp, 0, error)
            comm.wait_stream(main)
            # Reuse each destination three times in one forward.
            for generation in range(1, 4):
                with torch.cuda.stream(comm):
                    module.append_cycle_to_capture(*args(generation - 1, generation))
                for slot in range(2):
                    wait_generation(fill, slot, generation, 500, 10**12, error)
                    captured_outputs.extend(dsts[slot].clone() for dsts in destinations)
                    publish_generation(comp, slot, generation)
            main.wait_stream(comm)
        graph.instantiate()
        graphs.append(graph)
        outputs.append(captured_outputs)

    for index in (0, 1, 0, 1):
        for srcs in sources:
            for src in srcs:
                src.add_(1)
        graphs[index].replay()
        torch.cuda.synchronize()
        expected = [
            sources[c][slot] for _ in range(3) for slot in range(2) for c in range(2)
        ]
        for actual, wanted in zip(outputs[index], expected):
            assert torch.equal(actual, wanted)
        assert fill.tolist() == comp.tolist() == [3, 3]
        assert all(s.item() == -1 for s in state)
        assert error.item() == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
