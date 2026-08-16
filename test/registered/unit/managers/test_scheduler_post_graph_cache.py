from unittest.mock import Mock, call, patch

import pytest

from sglang.srt.managers.scheduler import Scheduler


class _StopAfterCacheReclaim(Exception):
    pass


def test_model_worker_reclaims_unused_cache_after_all_graphs():
    scheduler = Scheduler.__new__(Scheduler)
    calls = Mock()
    scheduler.init_tp_model_worker = Mock(
        side_effect=lambda: calls("init_tp_model_worker")
    )
    scheduler.maybe_init_draft_worker = Mock(
        side_effect=lambda: calls("maybe_init_draft_worker")
    )
    scheduler.init_memory_pools = Mock(side_effect=lambda: calls("init_memory_pools"))
    scheduler.init_all_attention_backends = Mock(
        side_effect=lambda: calls("init_all_attention_backends")
    )
    scheduler.init_all_cuda_graphs = Mock(
        side_effect=lambda: calls("init_all_cuda_graphs")
    )

    def reclaim_and_stop():
        calls("empty_cache")
        raise _StopAfterCacheReclaim

    with (
        patch(
            "sglang.srt.managers.scheduler.current_platform.empty_cache",
            side_effect=reclaim_and_stop,
        ),
        pytest.raises(_StopAfterCacheReclaim),
    ):
        scheduler.init_model_worker()

    assert calls.call_args_list == [
        call("init_tp_model_worker"),
        call("maybe_init_draft_worker"),
        call("init_memory_pools"),
        call("init_all_attention_backends"),
        call("init_all_cuda_graphs"),
        call("empty_cache"),
    ]
