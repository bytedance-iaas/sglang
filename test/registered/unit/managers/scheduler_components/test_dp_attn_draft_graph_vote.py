"""Contracts for the observation-only DP draft-graph eager vote."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _sync_info(*, can_draft_cuda_graph=True):
    return dp_attn.MLPSyncBatchInfo(
        dp_size=2,
        tp_size=1,
        cp_size=1,
        num_tokens=4,
        num_tokens_for_logprob=4,
        can_run_decode_cuda_graph=True,
        can_run_prefill_cuda_graph=False,
        can_draft_cuda_graph=can_draft_cuda_graph,
        is_extend_in_batch=False,
        local_can_run_tbo=True,
        local_forward_mode=ForwardMode.DECODE.value,
    )


def test_draft_graph_vote_is_min_reduced_in_existing_metadata_gather():
    info = _sync_info()
    gathered = torch.stack(
        [
            info._get_local_tensor(device="cpu"),
            _sync_info(can_draft_cuda_graph=False)._get_local_tensor(device="cpu"),
        ]
    )

    def gather_into(output, _local, group):
        output.copy_(gathered.flatten())

    with (
        patch.object(
            torch.distributed,
            "all_gather_into_tensor",
            side_effect=gather_into,
        ),
        patch.object(
            dp_attn,
            "get_tp_group",
            return_value=SimpleNamespace(
                active_ranks_cpu=torch.ones(2, dtype=torch.int64)
            ),
        ),
    ):
        info.all_gather(device="cpu", group=object())

    assert info.can_draft_cuda_graph is False
    assert info.tp0_info.shape == (2, 8)


def test_inactive_rank_is_permissive_for_draft_graph_vote():
    fallback = _sync_info(can_draft_cuda_graph=False)._get_fallback_tensor(
        device="cpu"
    )

    assert fallback.numel() == 8
    assert fallback[-1].item() == 1
