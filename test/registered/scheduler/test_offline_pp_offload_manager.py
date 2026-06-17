from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang.srt.managers.offline_pp_offload_manager import (
    OfflinePPStateOffloadManager,
    StateCodec,
    WaveState,
)


class _FakeStream:
    def synchronize(self):
        pass


class _FakeDeviceModule:
    def stream(self, _stream):
        return nullcontext()


class _FakeKVCache:
    layer_num = 1

    def __init__(self):
        self.loads = []

    def get_cpu_copy(self, kv_indices, mamba_indices=None):
        return kv_indices.clone()

    def load_cpu_copy(self, cpu_state, kv_indices, mamba_indices=None):
        self.loads.append((cpu_state, kv_indices.clone(), mamba_indices))


class _FakeKVAllocator:
    def __init__(self, *, size=64, page_size=1, fail_alloc=False):
        self.size = size
        self.page_size = page_size
        self.fail_alloc = fail_alloc
        self.kv_cache = _FakeKVCache()
        self.free_indices = torch.arange(1, size + 1, dtype=torch.int64)
        self.alloc_calls = []
        self.freed = []

    def get_kvcache(self):
        return self.kv_cache

    def available_size(self):
        return int(self.free_indices.numel())

    def alloc(self, need_size):
        self.alloc_calls.append(need_size)
        if self.fail_alloc or need_size > self.free_indices.numel():
            return None
        out = self.free_indices[:need_size].clone()
        self.free_indices = self.free_indices[need_size:]
        return out

    def free(self, indices):
        self.freed.append(indices.clone())
        if indices.numel() > 0:
            self.free_indices = torch.cat([indices.to(torch.int64), self.free_indices])


class _FakeReqToTokenPool:
    device = "cpu"

    def __init__(self, *, fail_alloc=False):
        self.fail_alloc = fail_alloc
        self.req_to_token = torch.zeros((16, 128), dtype=torch.int64)
        self.free_slots = list(range(1, 16))
        self.writes = []
        self.freed = []

    def alloc(self, reqs):
        if self.fail_alloc or len(reqs) > len(self.free_slots):
            return None
        selected = self.free_slots[: len(reqs)]
        self.free_slots = self.free_slots[len(reqs) :]
        for req, idx in zip(reqs, selected):
            req.req_pool_idx = idx
        return selected

    def write(self, indices, values):
        self.writes.append((indices, values.clone()))
        self.req_to_token[indices] = values

    def free(self, req):
        self.freed.append(req.req_pool_idx)
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None


def _req(rid, prompt_len, *, max_new_tokens=4, req_pool_idx=None):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=list(range(prompt_len)),
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
        req_pool_idx=req_pool_idx,
        mamba_pool_idx=None,
        kv_committed_len=prompt_len,
    )


def _manager(*, kv_allocator=None, req_pool=None):
    kv_allocator = kv_allocator or _FakeKVAllocator()
    req_pool = req_pool or _FakeReqToTokenPool()
    mgr = OfflinePPStateOffloadManager.__new__(OfflinePPStateOffloadManager)
    mgr.req_to_token_pool = req_pool
    mgr.token_to_kv_pool_allocator = kv_allocator
    mgr.kv_cache = kv_allocator.get_kvcache()
    mgr.codec = StateCodec()
    mgr.stall_ticks = 2
    mgr.is_hybrid = False
    mgr.offload_stream = _FakeStream()
    mgr.prefetch_stream = _FakeStream()
    mgr.device_module = _FakeDeviceModule()
    mgr.layer_num = mgr.kv_cache.layer_num
    mgr._next_wave_id = 0
    mgr.waves = {}
    mgr.offloaded_queue = []
    mgr._tick = 0
    return mgr


def test_infer_layer_num_from_hybrid_linear_pool_shape():
    mgr = OfflinePPStateOffloadManager.__new__(OfflinePPStateOffloadManager)
    mgr.kv_cache = SimpleNamespace(
        full_layer_nums=12,
        mamba_pool=SimpleNamespace(num_mamba_layers=16),
    )

    assert mgr._infer_layer_num() == 28


def test_admit_wave_split_preserves_cpu_state():
    mgr = _manager(kv_allocator=_FakeKVAllocator(size=20))
    reqs = [_req("r0", 3), _req("r1", 3), _req("r2", 3)]
    wave = mgr.new_wave(reqs)
    wave.state = WaveState.OFFLOADED
    wave.offload_ready = True
    for req in reqs:
        wave.entries[req.rid].cpu_state = f"cpu-{req.rid}"

    assert mgr.admit_wave(wave) is True

    assert [req.rid for req in wave.reqs] == ["r0"]
    assert wave.entries["r0"].cpu_state == "cpu-r0"
    remainder = mgr.waves[mgr.offloaded_queue[-1]]
    assert [req.rid for req in remainder.reqs] == ["r1", "r2"]
    assert remainder.entries["r1"].cpu_state == "cpu-r1"
    assert remainder.entries["r2"].cpu_state == "cpu-r2"


def test_prefetch_req_alloc_failure_frees_kv_allocation():
    kv_allocator = _FakeKVAllocator(size=16)
    req_pool = _FakeReqToTokenPool(fail_alloc=True)
    mgr = _manager(kv_allocator=kv_allocator, req_pool=req_pool)
    req = _req("r0", 5)
    wave = mgr.new_wave([req])
    wave.entries[req.rid].cpu_state = torch.arange(5)

    consumed = mgr.prefetch_step(wave, free_slots=16)

    assert consumed == 0
    assert wave.entries[req.rid].device_kv_indices is None
    assert req.req_pool_idx is None
    assert kv_allocator.freed[-1].numel() == 5


def test_prefetch_kv_alloc_failure_does_not_allocate_req_slot():
    kv_allocator = _FakeKVAllocator(size=16, fail_alloc=True)
    req_pool = _FakeReqToTokenPool()
    mgr = _manager(kv_allocator=kv_allocator, req_pool=req_pool)
    req = _req("r0", 5)
    wave = mgr.new_wave([req])
    wave.entries[req.rid].cpu_state = torch.arange(5)

    consumed = mgr.prefetch_step(wave, free_slots=16)

    assert consumed == 0
    assert req.req_pool_idx is None
    assert req_pool.freed == []
    assert req_pool.writes == []


def test_prefetch_page_aligned_allocation_maps_committed_only():
    kv_allocator = _FakeKVAllocator(size=32, page_size=4)
    req_pool = _FakeReqToTokenPool()
    mgr = _manager(kv_allocator=kv_allocator, req_pool=req_pool)
    req = _req("r0", 5)
    wave = mgr.new_wave([req])
    wave.entries[req.rid].cpu_state = torch.arange(5)

    consumed = mgr.prefetch_step(wave, free_slots=32)

    assert consumed == 8
    assert kv_allocator.alloc_calls[-1] == 8
    assert wave.entries[req.rid].device_kv_indices.numel() == 8
    assert req_pool.writes[-1][1].numel() == 5
    assert kv_allocator.kv_cache.loads[-1][1].numel() == 5


def test_stall_rollback_frees_partial_prefetch_and_requeues_wave():
    kv_allocator = _FakeKVAllocator(size=16)
    req_pool = _FakeReqToTokenPool()
    mgr = _manager(kv_allocator=kv_allocator, req_pool=req_pool)
    req = _req("r0", 4)
    wave = mgr.new_wave([req])
    wave.entries[req.rid].cpu_state = torch.arange(4)
    wave.state = WaveState.PREFETCHING
    mgr.prefetch_step(wave, free_slots=16)

    mgr.rollback_wave(wave)

    assert wave.state == WaveState.OFFLOADED
    assert wave.retry_count == 1
    assert mgr.offloaded_queue[0] == wave.wave_id
    assert wave.entries[req.rid].device_kv_indices is None
    assert req.req_pool_idx is None
