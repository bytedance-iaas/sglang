"""PD admission budgets apply per DPA rank / persistent PP slot."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_scheduler(pp_size=2, pool_size=16, max_running=16):
    return SimpleNamespace(
        ps=SimpleNamespace(pp_size=pp_size),
        grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
        enable_priority_scheduling=False,
        waiting_queue=[],
        req_to_token_pool=SimpleNamespace(size=pool_size, available_size=lambda: 0),
        max_running_requests=max_running,
        tree_cache=object(),
        token_to_kv_pool_allocator=object(),
        model_config=object(),
        enable_overlap=False,
        spec_algorithm=object(),
        future_map=None,
    )


def request(index):
    return SimpleNamespace(
        rid=f"req-{index}",
        last_node=None,
        kv=SimpleNamespace(kv_committed_len=None),
        init_next_round_input=MagicMock(),
    )


def admit(scheduler, running, limit):
    def init_new(reqs, *_):
        batch = MagicMock()
        batch.reqs = list(reqs)
        return batch

    module = "sglang.srt.disaggregation.decode"
    with (
        patch(
            f"{module}.get_parallel",
            return_value=SimpleNamespace(pp_max_micro_batch_size=limit),
        ),
        patch(
            f"{module}.get_disagg",
            return_value=SimpleNamespace(
                disaggregation_decode_enable_radix_cache=False
            ),
        ),
        patch(f"{module}.set_time_batch"),
        patch(f"{module}.ScheduleBatch.init_new", side_effect=init_new),
    ):
        return SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch(
            scheduler, SimpleNamespace(batch_size=lambda: len(running))
        )


@pytest.mark.parametrize("limit", [1, 8, 16, 32, 64])
def test_rank_local_slots_admit_refill_and_drain_without_duplicate_rids(limit):
    scheduler = make_scheduler()
    scheduler.waiting_queue = [request(i) for i in range(41)]
    slots = [[], []]
    completed = []
    budget = min(limit, scheduler.max_running_requests)
    for _ in range(50):
        for running in slots:
            batch = admit(scheduler, running, limit)
            if batch is not None:
                running.extend(batch.reqs)
                batch.prepare_for_prebuilt.assert_called_once()
                batch.process_prebuilt.assert_called_once_with(None)
            assert len(running) <= budget
            assert admit(scheduler, running, limit) is None
            active = [r.rid for slot in slots for r in slot]
            queued = [r.rid for r in scheduler.waiting_queue]
            assert len(active + queued + completed) == len(
                set(active + queued + completed)
            )
            # Partial completion forces repeated admission into nonempty slots.
            if running:
                completed.append(running.pop(0).rid)
        if not scheduler.waiting_queue and not any(slots):
            break
    assert sorted(completed) == sorted(f"req-{i}" for i in range(41))


@pytest.mark.parametrize("pool_size,max_running,expected", [(3, 16, 3), (16, 5, 5)])
def test_slot_limit_preserves_pool_and_request_capacity(
    pool_size, max_running, expected
):
    scheduler = make_scheduler(pool_size=pool_size, max_running=max_running)
    scheduler.waiting_queue = [request(i) for i in range(20)]
    batch = admit(scheduler, [], 64)
    assert len(batch.reqs) == expected
    assert len(scheduler.waiting_queue) == 20 - expected


def test_pp1_ignores_microbatch_and_preallocated_rows_need_no_free_rows():
    scheduler = make_scheduler(pp_size=1)
    scheduler.waiting_queue = [request(i) for i in range(16)]
    batch = admit(scheduler, [], 1)
    assert len(batch.reqs) == 16
    assert not scheduler.waiting_queue


def test_lowered_limit_does_not_retract_or_mutate_queued_requests():
    scheduler = make_scheduler()
    scheduler.waiting_queue = [request(20)]
    assert admit(scheduler, [request(i) for i in range(8)], 1) is None
    assert [r.rid for r in scheduler.waiting_queue] == ["req-20"]
    scheduler.waiting_queue[0].init_next_round_input.assert_not_called()
