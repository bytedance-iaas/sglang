from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)


class _HostPool:
    def __init__(self, indices):
        self.indices = torch.tensor(indices, dtype=torch.int64)
        self.calls = []

    def alloc_paged_token_slots(
        self,
        req_to_host_pool,
        req_to_host_pool_allocated_len,
        req_pool_idx,
        start_pos,
        num_tokens,
    ):
        self.calls.append((req_pool_idx, start_pos, num_tokens))
        return self.indices[:num_tokens]


def _coordinator(indices):
    return SimpleNamespace(
        mem_pool_host=_HostPool(indices),
        req_to_host_pool=torch.full((4, 16), -1, dtype=torch.int64),
        req_to_host_pool_allocated_len=torch.zeros(4, dtype=torch.int64),
        host_token_len=lambda length: length,
    )


def test_hisparse_pd_draft_uses_host_pool():
    draft_device_pool = object()
    draft_host_pool = object()
    queue = SimpleNamespace(
        draft_token_to_kv_pool=draft_device_pool,
        scheduler=SimpleNamespace(
            enable_hisparse=True,
            draft_hisparse_coordinator=SimpleNamespace(
                mem_pool_host=draft_host_pool
            ),
        ),
    )

    pool, kind = DecodePreallocQueue._draft_pd_transfer_pool(queue)

    assert pool is draft_host_pool
    assert kind == "DRAM"


def test_hisparse_pd_draft_requires_coordinator():
    queue = SimpleNamespace(
        draft_token_to_kv_pool=object(),
        scheduler=SimpleNamespace(
            enable_hisparse=True, draft_hisparse_coordinator=None
        ),
    )

    with pytest.raises(RuntimeError, match="draft HiSparse coordinator"):
        DecodePreallocQueue._draft_pd_transfer_pool(queue)


def test_hisparse_pd_allocates_lockstep_target_and_draft_slots():
    target = _coordinator(range(8))
    draft = _coordinator(range(8))
    queue = SimpleNamespace(
        scheduler=SimpleNamespace(
            hisparse_coordinator=target,
            draft_hisparse_coordinator=draft,
        )
    )
    req = SimpleNamespace(req_pool_idx=2, rid="req-1")

    indices = DecodePreallocQueue._allocate_hisparse_host_slots(queue, req, 8)

    torch.testing.assert_close(indices, torch.arange(8, dtype=torch.int64))
    assert target.mem_pool_host.calls == [(2, 0, 8)]
    assert draft.mem_pool_host.calls == [(2, 0, 8)]


def test_hisparse_pd_rejects_diverged_target_and_draft_slots():
    queue = SimpleNamespace(
        scheduler=SimpleNamespace(
            hisparse_coordinator=_coordinator(range(8)),
            draft_hisparse_coordinator=_coordinator(range(1, 9)),
        )
    )
    req = SimpleNamespace(req_pool_idx=1, rid="req-diverged")

    with pytest.raises(RuntimeError, match="host slots diverged"):
        DecodePreallocQueue._allocate_hisparse_host_slots(queue, req, 8)


def test_finished_request_releases_target_and_draft_hisparse_state():
    calls = []
    target = SimpleNamespace(
        request_finished=lambda req: calls.append(("target", req))
    )
    draft = SimpleNamespace(
        request_finished=lambda req: calls.append(("draft", req))
    )
    processor = SimpleNamespace(
        hisparse_coordinator=target,
        draft_hisparse_coordinator=draft,
    )
    req = object()

    SchedulerBatchResultProcessor._finish_hisparse_request(processor, req)

    assert calls == [("target", req), ("draft", req)]
