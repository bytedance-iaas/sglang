"""Regression coverage for requester-context pointer ownership and bounds."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.sidp.sidp_manager import SidpManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def manager(monkeypatch):
    result = object.__new__(SidpManager)
    result._weight_ipc_device = 0
    result._weight_ipc_bases = {}
    result.opened, result.closed = [], []

    def open_allocation(handle):
        result.opened.append(handle)
        return 4096 * len(result.opened)

    result.memcpy = SimpleNamespace(
        open_ipc_allocation=open_allocation,
        close_ipc_allocation=result.closed.append,
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return result


def descriptor(offset=0):
    return dict(
        shape=(2, 4),
        dtype=torch.float32,
        nbytes=32,
        offset=offset,
        allocation_nbytes=128,
        handle=b"h" * 64,
    )


def test_same_owner_allocation_is_opened_and_closed_once(monkeypatch):
    m = manager(monkeypatch)
    parameter = torch.empty((2, 4))
    first = m._import_weight_ipc(3, descriptor(), parameter)
    second = m._import_weight_ipc(3, descriptor(64), parameter)
    assert first.data_ptr() == 4096
    assert second.data_ptr() == 4160
    assert len(m.opened) == 1
    assert first.nbytes == 32 and first.numel() == 8
    m.close_weight_ipc()
    m.close_weight_ipc()
    assert m.closed == [4096]  # close allocation base, not each component pointer


@pytest.mark.parametrize(
    "change",
    [
        dict(offset=-1),
        dict(offset=100),
        dict(nbytes=64),
        dict(shape=(4, 2)),
        dict(dtype=torch.float16),
        dict(handle=b"short"),
    ],
)
def test_bad_peer_layout_rejected_before_mapping(monkeypatch, change):
    m = manager(monkeypatch)
    with pytest.raises(ValueError, match="descriptor"):
        m._import_weight_ipc(3, dict(descriptor(), **change), torch.empty((2, 4)))
    assert not m.opened


def test_foreign_current_context_rejected(monkeypatch):
    m = manager(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    with pytest.raises(RuntimeError, match="requester context"):
        m._import_weight_ipc(3, descriptor(), torch.empty((2, 4)))
    assert not m.opened
