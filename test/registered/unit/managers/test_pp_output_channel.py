from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.scheduler_pp_mixin import (
    SchedulerPPMixin,
    _pp_snapshot_graph_output_tensors,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.speculative.eagle_info import EaglePPVerifyInputRaw
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_graph_output_is_detached_recursively():
    tensor = torch.tensor([1, 2, 3])
    nested = torch.tensor([4, 5])
    source = {"token_ids": tensor, "nested": [nested], "metadata": None}

    snapshot = _pp_snapshot_graph_output_tensors(source, True)
    tensor.add_(10)
    nested.add_(10)

    assert snapshot["token_ids"].tolist() == [1, 2, 3]
    assert snapshot["nested"][0].tolist() == [4, 5]
    assert snapshot["token_ids"].data_ptr() != tensor.data_ptr()
    assert snapshot["nested"][0].data_ptr() != nested.data_ptr()


def test_eager_output_keeps_original_objects():
    source = {"token_ids": torch.tensor([1])}

    assert _pp_snapshot_graph_output_tensors(source, False) is source


def test_pp_eagle_result_schedules_full_cpu_copy():
    raw = EaglePPVerifyInputRaw.build_dummy_from_bonus_tokens(
        torch.tensor([7, 8]), num_draft=4
    )
    pp_outputs = PPProxyTensors(
        {"next_token_ids": torch.tensor([7, 8]), **raw.to_tensor_dict()}
    )
    copy_done = object()
    scheduler = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
        pp_group=SimpleNamespace(is_first_rank=False),
        device_module=SimpleNamespace(Event=MagicMock(return_value=copy_done)),
    )
    batch = SimpleNamespace(return_logprob=False, return_hidden_states=False)

    with (
        patch(
            "sglang.srt.managers.utils.GenerationBatchResult.copy_to_cpu"
        ) as copy_to_cpu,
        patch(
            "sglang.srt.managers.scheduler_pp_mixin.get_spec",
            return_value=SimpleNamespace(speculative_num_draft_tokens=4),
        ),
    ):
        result = SchedulerPPMixin._pp_prep_batch_result(
            scheduler, batch, SimpleNamespace(can_run_cuda_graph=True), pp_outputs
        )

    assert isinstance(batch.spec_info, EaglePPVerifyInputRaw)
    assert result.copy_done is copy_done
    copy_to_cpu.assert_called_once_with(
        return_logprob=False, return_hidden_states=False
    )
