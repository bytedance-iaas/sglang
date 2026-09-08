from contextlib import nullcontext
from types import SimpleNamespace

from sglang.srt.layers.sidp import fixed_dma_flag_backend as backend_module
from sglang.srt.layers.sidp.fixed_dma_flag_backend import SidpFixedDmaFlagBackend
from sglang.srt.layers.sidp.sidp_manager import SidpManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _RecordingCycleBackend(SidpFixedDmaFlagBackend):
    def __init__(self):
        self.calls = []

    def enqueue_cycle(self, cycle, **kwargs):
        self.calls.append((cycle, kwargs))


class _RecordingOtherFlagBackend:
    def __init__(self):
        self.calls = []

    def enqueue_cycle(self, cycle, **kwargs):
        self.calls.append((cycle, kwargs))


def test_manager_propagates_no_consume_wait_for_cycle_one():
    manager = object.__new__(SidpManager)
    manager._queued_cycles = set()
    manager._cycle_layers = {1: [1]}
    manager._graph_profiler = None
    manager._uses_flag_sync = True
    manager._cycle_backend = _RecordingCycleBackend()
    manager.dma_slices = 1

    manager._enqueue_cycle(1, wait_for_consume=False)

    assert manager._cycle_backend.calls == [
        (1, {"wait_for_consume": False})
    ]


def test_manager_does_not_extend_other_flag_backend_interfaces():
    manager = object.__new__(SidpManager)
    manager._queued_cycles = set()
    manager._cycle_layers = {1: [1]}
    manager._graph_profiler = None
    manager._uses_flag_sync = True
    manager._cycle_backend = _RecordingOtherFlagBackend()
    manager.dma_slices = 1

    manager._enqueue_cycle(1, wait_for_consume=False)

    assert manager._cycle_backend.calls == [(1, {})]


def test_flag_backend_skips_war_kernel_when_slot_is_already_safe(monkeypatch):
    waits = []
    publishes = []
    monkeypatch.setattr(
        backend_module.torch.cuda, "stream", lambda _stream: nullcontext()
    )
    monkeypatch.setattr(
        backend_module,
        "wait_generation",
        lambda *args: waits.append(args),
    )
    monkeypatch.setattr(
        backend_module,
        "publish_generation",
        lambda *args: publishes.append(args),
    )

    layer_plan = SimpleNamespace(
        slot=0,
        layer_id=1,
        total_nbytes=0,
        slices=((),),
    )
    manager = SimpleNamespace(
        comm_stream=SimpleNamespace(cuda_stream=0),
        dma_slices=1,
        _graph_profiler=None,
        memcpy=SimpleNamespace(async_copy=lambda *args: None),
        _cycle_cache_depth=2,
    )
    backend = object.__new__(SidpFixedDmaFlagBackend)
    backend.manager = manager
    backend._cycle_groups = {1: ((layer_plan,),)}
    backend.fill_gen = object()
    backend.comp_gen = object()
    backend.error_state = object()
    backend.timeout_clocks = 1

    backend.enqueue_cycle(
        1,
        wait_for_consume=False,
        target_fill_gen=1,
        required_comp_gen=0,
    )

    assert waits == []
    assert len(publishes) == 1


def test_flag_backend_keeps_war_kernel_for_reused_slot(monkeypatch):
    waits = []
    monkeypatch.setattr(
        backend_module.torch.cuda, "stream", lambda _stream: nullcontext()
    )
    monkeypatch.setattr(
        backend_module,
        "wait_generation",
        lambda *args: waits.append(args),
    )
    monkeypatch.setattr(backend_module, "publish_generation", lambda *args: None)

    layer_plan = SimpleNamespace(
        slot=0,
        layer_id=2,
        total_nbytes=0,
        slices=((),),
    )
    manager = SimpleNamespace(
        comm_stream=SimpleNamespace(cuda_stream=0),
        dma_slices=1,
        _graph_profiler=None,
        memcpy=SimpleNamespace(async_copy=lambda *args: None),
        _cycle_cache_depth=2,
    )
    backend = object.__new__(SidpFixedDmaFlagBackend)
    backend.manager = manager
    backend._cycle_groups = {2: ((layer_plan,),)}
    backend.fill_gen = object()
    backend.comp_gen = object()
    backend.error_state = object()
    backend.timeout_clocks = 1

    backend.enqueue_cycle(
        2,
        wait_for_consume=True,
        target_fill_gen=2,
        required_comp_gen=1,
    )

    assert len(waits) == 1
