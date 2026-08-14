from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.utils import MetadataBuffers
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_pp_mixin import (
    PPBatchMetadata,
    SchedulerPPMixin,
    _pp_can_skip_output_comm,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_utils import (
    get_draft_recurrent_hidden_state_spec_from_config,
)


class _SpecAlgorithm:
    @staticmethod
    def is_none():
        return False

    @staticmethod
    def is_eagle():
        return True

    @staticmethod
    def is_standalone():
        return False


class _ProxyOutputs:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, key):
        return self.tensors[key]


def test_mtp_middle_chunk_skips_unused_pp_output_ring():
    batch = SimpleNamespace(
        spec_algorithm=_SpecAlgorithm(),
        forward_mode=ForwardMode.EXTEND,
        reqs=[SimpleNamespace(rid="r0")],
        contains_last_prefill_chunk=False,
        return_logprob=False,
    )

    with patch.object(
        envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM, "get", return_value=True
    ):
        assert _pp_can_skip_output_comm(batch)


def test_pp_prefill_rebuilds_one_authoritative_draft_input():
    topk_p = torch.randn(2, 1)
    topk_index = torch.tensor([[3], [7]], dtype=torch.int64)
    hidden_states = torch.randn(2, 8)
    bonus_tokens = torch.tensor([11, 13], dtype=torch.int64)
    dsa_topk_indices = torch.tensor([[2, 4], [6, 8]], dtype=torch.int32)
    pp_outputs = _ProxyOutputs(
        {
            "next_token_ids": bonus_tokens,
            "spec_prefill_topk_p": topk_p,
            "spec_prefill_topk_index": topk_index,
            "spec_prefill_hidden_states": hidden_states,
            "spec_prefill_dsa_topk_indices": dsa_topk_indices,
        }
    )
    batch = SimpleNamespace(
        spec_algorithm=_SpecAlgorithm(),
        reqs=[SimpleNamespace(rid="r0"), SimpleNamespace(rid="r1")],
        req_pool_indices=torch.tensor([0, 1]),
        forward_mode=SimpleNamespace(is_extend=Mock(return_value=True)),
        return_logprob=False,
        input_ids=bonus_tokens.clone(),
        spec_info=None,
    )
    scheduler = SimpleNamespace(
        spec_algorithm=_SpecAlgorithm(),
        server_args=SimpleNamespace(speculative_num_draft_tokens=5),
        future_map=SimpleNamespace(stash=Mock()),
        device_module=SimpleNamespace(Event=Mock(return_value=Mock())),
        _pp_spec_store_bonus=Mock(),
    )

    with patch.object(
        GenerationBatchResult, "copy_to_cpu", autospec=True
    ) as copy_to_cpu:
        result = SchedulerPPMixin._pp_prep_batch_result(
            scheduler,
            batch,
            PPBatchMetadata(can_run_cuda_graph=True, fwd_batch=None),
            pp_outputs,
        )

    assert batch.spec_info is result.next_draft_input
    assert result.copy_done is None
    copy_to_cpu.assert_not_called()
    assert result.next_draft_input.topk_p is topk_p
    assert result.next_draft_input.topk_index is topk_index
    assert result.next_draft_input.hidden_states is hidden_states
    assert result.next_draft_input.bonus_tokens is bonus_tokens
    assert result.next_draft_input.dsa_topk_indices is dsa_topk_indices
    assert batch.input_ids is None
    scheduler.future_map.stash.assert_called_once()


def test_spec_only_aux_indices_follow_optional_sampling_mask_layout():
    for sampling_mask_tokens, expected in (
        (0, [6, 7, 8, 9]),
        (32, [9, 10, 11, 12]),
    ):
        buffers = MetadataBuffers(
            size=2,
            hidden_size=8,
            hidden_states_dtype=torch.float32,
            max_sampling_mask_tokens=sampling_mask_tokens,
            output_dsa_topk_indices_dim=4,
        )
        ptrs, _, _ = buffers.get_buf_infos()
        indices = buffers.get_spec_only_aux_indices()
        assert indices == expected
        assert ptrs[indices[0]] == buffers.output_topk_p.data_ptr()
        assert ptrs[indices[1]] == buffers.output_topk_index.data_ptr()
        assert ptrs[indices[2]] == buffers.output_hidden_states.data_ptr()
        assert ptrs[indices[3]] == buffers.output_dsa_topk_indices.data_ptr()

    buffers_without_dsa = MetadataBuffers(
        size=2,
        hidden_size=8,
        hidden_states_dtype=torch.float32,
        max_sampling_mask_tokens=0,
        output_dsa_topk_indices_dim=0,
    )
    assert buffers_without_dsa.get_spec_only_aux_indices() == [6, 7, 8]


def test_draft_hidden_state_wire_schema_does_not_require_a_local_runner():
    config = SimpleNamespace(spec_hidden_size=6144, dtype=torch.bfloat16)

    hidden_size, dtype = get_draft_recurrent_hidden_state_spec_from_config(
        config, _SpecAlgorithm()
    )

    assert hidden_size == 6144
    assert dtype is torch.bfloat16
