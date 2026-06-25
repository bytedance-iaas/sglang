from __future__ import annotations

from types import SimpleNamespace

import torch

import sglang.srt.managers.scheduler as scheduler_mod
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _FakeReq:
    def __init__(self, rid="r0"):
        self.rid = rid
        self.req_pool_idx = 3
        self.origin_input_ids = [11, 12, 13]
        self.output_ids = [42]
        self.return_logprob = False
        self.return_hidden_states = False
        self.is_prefill_only = False
        self.grammar = None
        self.multimodal_inputs = None

    def finished(self):
        return False


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.forward_mode = ForwardMode.EXTEND
        self.filtered_to_empty = False

    def is_empty(self):
        return len(self.reqs) == 0

    def filter_batch(self, keep_indices=None, **kwargs):
        assert keep_indices == []
        self.filtered_to_empty = True
        self.reqs = []


class _EmptyBatch:
    offline_pp_wave_id = None

    def is_empty(self):
        return True


class _FakeOfflinePPManager:
    def __init__(self):
        self.offloaded = []
        self.entered_draining = []
        self.filling = True
        self.active_epoch_waves = False
        self.decode_ready_wave = None

    def offload_prefilled_wave(self, reqs, source_stream=None):
        self.offloaded.append(list(reqs))
        self.active_epoch_waves = True

    def is_filling(self):
        return self.filling

    def is_draining(self):
        return not self.filling

    def local_fill_stop_reason(self, waiting_queue_empty):
        return "waiting_empty" if waiting_queue_empty and self.active_epoch_waves else None

    def enter_draining(self, reason):
        self.entered_draining.append(reason)
        self.filling = False

    def has_active_epoch_waves(self):
        return self.active_epoch_waves

    def has_decoding_wave(self):
        return False

    def ensure_decode_ready_for_schedule(self, free_slots):
        pass

    def take_decode_ready_wave(self, mb_id=None):
        wave = self.decode_ready_wave
        self.decode_ready_wave = None
        return wave

    def retire_wave_by_id(self, wave_id):
        self.active_epoch_waves = False

    def wait_prefetch_for_decode(self, wave):
        pass


class _FakeFutureMap:
    def __init__(self):
        self.stashes = []

    def stash(self, indices, values):
        self.stashes.append((indices.clone(), values.clone()))


class _FakeDecodeBatch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.return_logprob = False
        self.prepare_for_decode_called = False
        self.out_cache_loc = None

    def prepare_for_decode(self):
        self.prepare_for_decode_called = True
        self.out_cache_loc = torch.arange(len(self.reqs), dtype=torch.int64)
        self.seq_lens = self.seq_lens + 1
        self.seq_lens_cpu = self.seq_lens_cpu + 1
        self.orig_seq_lens = self.orig_seq_lens + 1
        self.seq_lens_sum = None


def test_prefill_batch_offload_is_consumed_only_once():
    req = _FakeReq()
    batch = _FakeBatch([req])
    mgr = _FakeOfflinePPManager()
    sched = SimpleNamespace(
        last_batch=batch,
        running_batch=_EmptyBatch(),
        waiting_queue=[],
        chunked_req=None,
        offline_pp_offload_manager=mgr,
        get_new_batch_prefill=lambda: None,
        token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 1024),
    )

    assert Scheduler._get_next_batch_offline_pp(sched) is None
    assert Scheduler._get_next_batch_offline_pp(sched) is None

    assert mgr.offloaded == [[req]]
    assert mgr.entered_draining == ["waiting_empty"]
    assert batch.filtered_to_empty is True
    assert batch.is_empty()


def test_draining_epoch_does_not_insert_new_prefill():
    mgr = _FakeOfflinePPManager()
    mgr.filling = False
    calls = {"prefill": 0}

    def _get_new_batch_prefill():
        calls["prefill"] += 1
        return object()

    sched = SimpleNamespace(
        last_batch=None,
        running_batch=_EmptyBatch(),
        waiting_queue=[_FakeReq("waiting")],
        chunked_req=None,
        offline_pp_offload_manager=mgr,
        get_new_batch_prefill=_get_new_batch_prefill,
        token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 1024),
    )

    assert Scheduler._get_next_batch_offline_pp(sched) is None
    assert calls["prefill"] == 0


def test_pp_fill_stop_reason_sync_prioritizes_host_limit():
    pp_group = SimpleNamespace(
        world_size=2,
        all_gather_object=lambda _reason: ["", "host"],
    )
    sched = SimpleNamespace(pp_group=pp_group)

    assert Scheduler._offline_pp_sync_fill_stop_reason(sched, None) == "host"


def test_decode_ready_wave_batch_stashes_last_token_and_prepares_decode(monkeypatch):
    req = _FakeReq()
    wave = SimpleNamespace(
        wave_id=7,
        reqs=[req],
        entries={req.rid: SimpleNamespace(committed_len=3)},
    )
    future_map = _FakeFutureMap()
    sched = SimpleNamespace(
        device="cpu",
        req_to_token_pool=SimpleNamespace(device="cpu"),
        token_to_kv_pool_allocator=SimpleNamespace(),
        tree_cache=SimpleNamespace(),
        model_config=SimpleNamespace(vocab_size=100),
        enable_overlap=False,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        future_map=future_map,
        offline_pp_offload_manager=SimpleNamespace(
            wait_prefetch_for_decode=lambda wave: None
        ),
    )

    def _init_new(**kwargs):
        return _FakeDecodeBatch(**kwargs)

    monkeypatch.setattr(
        scheduler_mod.ScheduleBatch, "init_new", staticmethod(_init_new)
    )
    monkeypatch.setattr(
        scheduler_mod.SamplingBatchInfo,
        "from_schedule_batch",
        staticmethod(lambda batch, vocab_size: SimpleNamespace(vocab_size=vocab_size)),
    )

    batch = Scheduler._build_offline_pp_decode_batch(sched, wave)

    assert batch.prepare_for_decode_called is True
    assert batch.out_cache_loc is not None
    assert batch.offline_pp_wave_id == 7
    assert batch.multimodal_inputs == [None]
    assert batch.req_pool_indices_cpu.tolist() == [3]
    assert batch.seq_lens_cpu.tolist() == [4]
    assert future_map.stashes[0][1].tolist() == [42]


def test_is_fully_idle_false_when_offline_pp_waves_are_active():
    sched = SimpleNamespace(
        running_batch=_EmptyBatch(),
        chunked_req=None,
        dllm_manager=SimpleNamespace(any_staging_reqs=lambda: False),
        last_batch=None,
        cur_batch=None,
        enable_overlap=False,
        result_queue=[],
        _pp_microbatches_drained=lambda: True,
        waiting_queue=[],
        offline_pp_offload_manager=SimpleNamespace(has_active_waves=lambda: True),
        grammar_manager=SimpleNamespace(grammar_queue=[]),
        disaggregation_mode=DisaggregationMode.NULL,
        enable_hisparse=False,
        enable_hierarchical_cache=False,
    )

    assert Scheduler.is_fully_idle(sched) is False
