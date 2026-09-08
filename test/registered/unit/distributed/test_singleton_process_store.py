"""Tests for the port-free singleton distributed initialization path."""

from unittest.mock import patch, sentinel

import pytest

from sglang.srt.distributed import parallel_state
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("use_in_process_store", [False, True])
def test_singleton_process_group_store_selection(use_in_process_store):
    with (
        patch.object(parallel_state, "_WORLD", None),
        patch.object(
            parallel_state.torch.distributed,
            "is_initialized",
            return_value=False,
        ),
        patch.object(
            parallel_state.torch.distributed, "get_world_size", return_value=1
        ),
        patch.object(
            parallel_state.torch.distributed,
            "HashStore",
            return_value=sentinel.hash_store,
        ) as hash_store,
        patch.object(
            parallel_state.torch.distributed, "init_process_group"
        ) as init_process_group,
        patch.object(
            parallel_state, "get_torch_distributed_pg_options", return_value=None
        ),
        patch.object(
            parallel_state, "init_world_group", return_value=sentinel.world_group
        ),
    ):
        init_method = "tcp://127.0.0.1:46443"
        parallel_state.init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=init_method,
            local_rank=0,
            backend="gloo",
            use_in_process_store=use_in_process_store,
        )

        kwargs = init_process_group.call_args.kwargs
        if use_in_process_store:
            hash_store.assert_called_once_with()
            assert kwargs["store"] is sentinel.hash_store
            assert "init_method" not in kwargs
        else:
            hash_store.assert_not_called()
            assert kwargs["init_method"] == init_method
            assert "store" not in kwargs


def test_in_process_store_rejects_non_singleton_group():
    with (
        patch.object(parallel_state, "_WORLD", None),
        patch.object(
            parallel_state.torch.distributed,
            "is_initialized",
            return_value=False,
        ),
        patch.object(
            parallel_state, "get_torch_distributed_pg_options", return_value=None
        ),
        pytest.raises(ValueError, match="world_size=1 and rank=0"),
    ):
        parallel_state.init_distributed_environment(
            world_size=2,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:46443",
            local_rank=0,
            backend="gloo",
            use_in_process_store=True,
        )
