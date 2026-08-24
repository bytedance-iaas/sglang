import pytest

from sglang.srt.layers.sidp.sync_strategy import (
    ForceSyncStrategy,
    NoSyncStrategy,
    build_peak_sync_strategy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeStore:
    def __init__(self):
        self.values = {}
        self.waits = []

    def set(self, key, value):
        self.values[key] = value

    def wait(self, keys, timeout):
        self.waits.append((keys, timeout))
        missing = [key for key in keys if key not in self.values]
        if missing:
            raise RuntimeError(f"missing keys: {missing}")


class _FakeStream:
    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


def test_none_strategy_never_synchronizes():
    result = NoSyncStrategy().before_launch(raw_batch_size=64, graph_batch_size=64)
    assert result["synchronized"] is False
    assert result["sync_index"] is None


def test_force_sync_arms_at_bulk_threshold_and_honors_bound(monkeypatch):
    store = _FakeStore()
    stream = _FakeStream()
    monkeypatch.setattr("torch.cuda.current_stream", lambda: stream)
    strategy = ForceSyncStrategy(
        store=store,
        dp_rank=0,
        dp_size=1,
        min_raw_bs=64,
        max_replays=2,
        timeout_s=3.0,
    )

    before_bulk = strategy.before_launch(raw_batch_size=63, graph_batch_size=64)
    first = strategy.before_launch(raw_batch_size=64, graph_batch_size=64)
    second = strategy.before_launch(raw_batch_size=1, graph_batch_size=1)
    after_bound = strategy.before_launch(raw_batch_size=64, graph_batch_size=64)

    assert before_bulk["synchronized"] is False
    assert (first["synchronized"], first["sync_index"]) == (True, 0)
    assert (second["synchronized"], second["sync_index"]) == (True, 1)
    assert after_bound["synchronized"] is False
    assert stream.synchronize_calls == 2
    assert "sidp/peak_sync/ready/0" in store.values
    assert "sidp/peak_sync/step/0/0" in store.values
    assert "sidp/peak_sync/step/1/0" in store.values


def test_builder_disables_sync_without_peak_shifting():
    strategy = build_peak_sync_strategy(
        "force_sync",
        enabled=False,
        store=_FakeStore(),
        dp_rank=0,
        dp_size=1,
        min_raw_bs=64,
        max_replays=0,
        timeout_s=30.0,
    )
    assert isinstance(strategy, NoSyncStrategy)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
