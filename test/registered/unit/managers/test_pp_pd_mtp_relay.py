from collections import defaultdict, deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.disaggregation.utils import MetadataBuffers
from sglang.srt.distributed.bootstrap import _prewarm_nccl
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_pp_mixin import (
    PPBatchMetadata,
    SchedulerPPMixin,
    _pp_can_skip_output_comm,
    _pp_pack_control_ring_message,
    _pp_unpack_control_ring_message,
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


def test_nccl_prewarm_initializes_distinct_tp_and_pp_groups():
    tp_group = object()
    pp_group_handle = object()
    pp_group = SimpleNamespace(
        device_group=pp_group_handle,
        ranks=[0, 8],
        rank_in_group=0,
    )
    pp_output_group_handle = object()
    pp_output_group = SimpleNamespace(
        device_group=pp_output_group_handle,
        ranks=[0, 8],
        rank_in_group=0,
    )
    warmup_tensor = object()
    recv_tensor = object()
    send_work = Mock()
    recv_work = Mock()

    with (
        patch(
            "sglang.srt.distributed.bootstrap.get_tp_group",
            return_value=SimpleNamespace(device_group=tp_group),
        ),
        patch(
            "sglang.srt.distributed.bootstrap.get_pp_group",
            return_value=pp_group,
        ),
        patch(
            "sglang.srt.distributed.bootstrap.get_pp_output_group",
            return_value=pp_output_group,
        ),
        patch(
            "sglang.srt.distributed.bootstrap.torch.zeros",
            return_value=warmup_tensor,
        ),
        patch(
            "sglang.srt.distributed.bootstrap.torch.empty_like",
            return_value=recv_tensor,
        ),
        patch(
            "sglang.srt.distributed.bootstrap.torch.cuda.current_device",
            return_value=0,
        ),
        patch("sglang.srt.distributed.bootstrap.dist.all_reduce") as all_reduce,
        patch(
            "sglang.srt.distributed.bootstrap.dist.isend", return_value=send_work
        ) as isend,
        patch(
            "sglang.srt.distributed.bootstrap.dist.irecv", return_value=recv_work
        ) as irecv,
        patch("sglang.srt.distributed.bootstrap.dist.barrier") as barrier,
        patch("sglang.srt.distributed.bootstrap.current_platform.synchronize"),
    ):
        _prewarm_nccl(tp_size=8, pp_size=2, moe_ep_size=1)

    assert all_reduce.call_args_list == [
        call(warmup_tensor, group=tp_group),
        call(warmup_tensor, group=pp_group_handle),
        call(warmup_tensor, group=pp_output_group_handle),
    ]
    assert isend.call_args_list == [
        call(warmup_tensor, dst=8, group=pp_group_handle),
        call(warmup_tensor, dst=8, group=pp_output_group_handle),
    ]
    assert irecv.call_args_list == [
        call(recv_tensor, src=8, group=pp_group_handle),
        call(recv_tensor, src=8, group=pp_output_group_handle),
    ]
    assert send_work.wait.call_count == 2
    assert recv_work.wait.call_count == 2
    assert barrier.call_args_list == [
        call(group=pp_group_handle),
        call(group=pp_group_handle),
        call(group=pp_output_group_handle),
        call(group=pp_output_group_handle),
    ]


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


def test_pp_control_ring_forwards_valid_empty_payload():
    events = []
    payload = [[], []]
    incoming = _pp_pack_control_ring_message("bootstrap", True, payload)
    control_group = object()
    scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=False),
        pp_disagg_control_group=control_group,
        _pp_recv_pyobj_from_prev_stage=Mock(
            side_effect=lambda group: events.append(("recv", group)) or incoming
        ),
        _pp_send_pyobj_to_next_stage=Mock(
            side_effect=lambda message, async_send, group: events.append(
                ("send", group)
            )
            or [object()]
        ),
        _pp_commit_comm_work=Mock(side_effect=lambda work: events.append("commit")),
    )
    process_payload = Mock(side_effect=lambda value: events.append("process") or value)

    result = SchedulerPPMixin._pp_run_control_ring_phase(
        scheduler,
        phase="bootstrap",
        origin_has_payload=False,
        origin_payload=None,
        process_payload=process_payload,
    )

    assert result == payload
    assert events == [
        ("recv", control_group),
        "process",
        ("send", control_group),
        "commit",
    ]
    forwarded = scheduler._pp_send_pyobj_to_next_stage.call_args.args[0]
    assert _pp_unpack_control_ring_message(forwarded, "bootstrap") == (
        True,
        payload,
    )


def test_pp_control_ring_last_stage_returns_typed_noop():
    events = []
    incoming = _pp_pack_control_ring_message("release", False, None)
    control_group = object()
    scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=True),
        pp_disagg_control_group=control_group,
        _pp_recv_pyobj_from_prev_stage=Mock(
            side_effect=lambda group: events.append(("recv", group)) or incoming
        ),
        _pp_send_pyobj_to_next_stage=Mock(
            side_effect=lambda message, async_send, group: events.append(
                ("send", group)
            )
            or [object()]
        ),
        _pp_commit_comm_work=Mock(side_effect=lambda work: events.append("commit")),
    )
    process_payload = Mock()

    result = SchedulerPPMixin._pp_run_control_ring_phase(
        scheduler,
        phase="release",
        origin_has_payload=False,
        origin_payload=([], []),
        process_payload=process_payload,
    )

    assert result is None
    assert events == [
        ("send", control_group),
        ("recv", control_group),
        "commit",
    ]
    process_payload.assert_not_called()
    originated = scheduler._pp_send_pyobj_to_next_stage.call_args.args[0]
    assert _pp_unpack_control_ring_message(originated, "release") == (False, None)


def test_pp_linear_payload_is_forwarded_before_following_control_phase():
    events = []
    previous_work = [object()]
    next_work = [object()]
    scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=False),
        _pp_commit_comm_work=Mock(
            side_effect=lambda work: events.append(("commit", work))
        ),
        _pp_send_pyobj_to_next_stage=Mock(
            side_effect=lambda payload, async_send: events.append(
                ("send", payload, async_send)
            )
            or next_work
        ),
    )

    result = SchedulerPPMixin._pp_forward_stage_payload(
        scheduler, previous_work, ["request"]
    )

    assert result is next_work
    assert events == [
        ("commit", previous_work),
        ("send", ["request"], True),
    ]


def test_pp_last_stage_consumes_linear_payload_without_forwarding():
    previous_work = [object()]
    scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=True),
        _pp_commit_comm_work=Mock(),
        _pp_send_pyobj_to_next_stage=Mock(),
    )

    result = SchedulerPPMixin._pp_forward_stage_payload(
        scheduler, previous_work, ["request"]
    )

    assert result == []
    scheduler._pp_commit_comm_work.assert_called_once_with(previous_work)
    scheduler._pp_send_pyobj_to_next_stage.assert_not_called()


def test_pp_proxy_exchange_is_committed_before_reusing_the_ring():
    events = []
    proxy_work = [object()]
    tensor_dict = {"hidden_states": object()}
    scheduler = SimpleNamespace(
        send_proxy_work=[],
        _pp_send_dict_to_next_stage=Mock(
            side_effect=lambda tensors, async_send, msg_type: events.append(
                ("send", tensors, async_send, msg_type)
            )
            or proxy_work
        ),
        _pp_commit_comm_work=Mock(
            side_effect=lambda work: events.append(("commit", work))
        ),
    )

    SchedulerPPMixin._pp_send_and_commit_proxy(scheduler, tensor_dict)

    assert scheduler.send_proxy_work is proxy_work
    assert events == [
        ("send", tensor_dict, True, "proxy"),
        ("commit", proxy_work),
    ]


def test_pp_proxy_and_output_use_independent_tensor_channels():
    proxy_group = SimpleNamespace(
        send_tensor_dict=Mock(return_value=[]),
        recv_tensor_dict=Mock(return_value={"__msg_type__": "proxy"}),
    )
    output_group = SimpleNamespace(
        send_tensor_dict=Mock(return_value=[]),
        recv_tensor_dict=Mock(return_value={"__msg_type__": "output"}),
    )
    all_gather_group = object()
    scheduler = SimpleNamespace(
        pp_group=proxy_group,
        pp_output_group=output_group,
        _pp_tensor_dict_inbox=defaultdict(deque),
        require_attn_tp_allgather=False,
        attn_tp_group=all_gather_group,
    )

    proxy_tensors = {"hidden_states": object()}
    output_tensors = {"next_token_ids": object()}
    SchedulerPPMixin._pp_send_dict_to_next_stage(
        scheduler, proxy_tensors, async_send=True, msg_type="proxy"
    )
    SchedulerPPMixin._pp_send_dict_to_next_stage(
        scheduler, output_tensors, async_send=True, msg_type="output"
    )
    SchedulerPPMixin._pp_recv_typed_dict(
        scheduler, expected_kind="proxy", all_gather_group=all_gather_group
    )
    SchedulerPPMixin._pp_recv_typed_dict(
        scheduler, expected_kind="output", all_gather_group=all_gather_group
    )

    proxy_group.send_tensor_dict.assert_called_once_with(
        tensor_dict=proxy_tensors,
        all_gather_group=None,
        async_send=True,
    )
    output_group.send_tensor_dict.assert_called_once_with(
        tensor_dict=output_tensors,
        all_gather_group=None,
        async_send=True,
    )
    proxy_group.recv_tensor_dict.assert_called_once_with(
        all_gather_group=all_gather_group
    )
    output_group.recv_tensor_dict.assert_called_once_with(
        all_gather_group=all_gather_group
    )


def test_pp_disagg_output_ring_relays_fresh_payload_before_control_phase():
    events = []
    received_tensors = {"next_token_ids": object()}
    send_work = [object()]
    recorded_event = Mock()
    target = SimpleNamespace(forward_mode=SimpleNamespace(is_prebuilt=lambda: False))
    scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=False),
        copy_stream_ctx=nullcontext(),
        copy_stream=SimpleNamespace(
            wait_stream=lambda stream: events.append(("wait_stream", stream))
        ),
        schedule_stream=object(),
        device_module=SimpleNamespace(
            Event=Mock(return_value=recorded_event),
            current_stream=Mock(return_value=object()),
        ),
        _pp_recv_dict_from_prev_stage=Mock(
            side_effect=lambda: events.append("recv") or received_tensors
        ),
        _pp_prep_batch_result=Mock(
            side_effect=lambda batch, metadata, outputs: events.append("prep")
            or object()
        ),
        _pp_send_dict_to_next_stage=Mock(
            side_effect=lambda tensors, async_send, msg_type: events.append(
                ("send", tensors, async_send, msg_type)
            )
            or send_work
        ),
        _pp_send_output_to_next_stage=Mock(),
        _pp_commit_comm_work=Mock(
            side_effect=lambda work: events.append(("commit", work))
        ),
    )

    with patch(
        "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
        return_value=False,
    ):
        outputs, _, event, work = (
            SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
                scheduler,
                next_first_rank_mb_id=0,
                next_mb_id=0,
                mbs=[target],
                mb_metadata=[object()],
                last_rank_comm_queue=deque(),
                pp_outputs=None,
                relay_output_immediately=True,
            )
        )

    assert outputs.tensors is received_tensors
    assert event is recorded_event
    assert work == []
    assert events[-2:] == [
        ("send", received_tensors, True, "output"),
        ("commit", send_work),
    ]
    scheduler._pp_send_output_to_next_stage.assert_not_called()


def test_pp_disagg_output_ring_last_stage_starts_relay_chain():
    events = []
    send_work = [object()]
    target = SimpleNamespace(forward_mode=SimpleNamespace(is_prebuilt=lambda: False))
    recorded_event = Mock()
    scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=True),
        copy_stream_ctx=nullcontext(),
        copy_stream=SimpleNamespace(wait_stream=Mock()),
        schedule_stream=object(),
        device_module=SimpleNamespace(
            Event=Mock(return_value=recorded_event),
            current_stream=Mock(return_value=object()),
        ),
        _pp_send_output_to_next_stage=Mock(
            side_effect=lambda *args: events.append("send") or send_work
        ),
        _pp_recv_dict_from_prev_stage=Mock(
            side_effect=lambda: events.append("recv")
            or {"next_token_ids": object()}
        ),
        _pp_prep_batch_result=Mock(return_value=object()),
        _pp_commit_comm_work=Mock(
            side_effect=lambda work: events.append(("commit", work))
        ),
    )

    with patch(
        "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
        return_value=False,
    ):
        _, _, _, work = SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
            scheduler,
            next_first_rank_mb_id=0,
            next_mb_id=0,
            mbs=[target],
            mb_metadata=[object()],
            last_rank_comm_queue=deque(),
            pp_outputs=None,
            relay_output_immediately=True,
        )

    assert work is send_work
    assert events == ["send", "recv"]
    scheduler._pp_commit_comm_work.assert_not_called()


def test_pp_disagg_output_origin_send_survives_empty_return_slot():
    send_work = [object()]
    origin_target = object()
    scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=True),
        _pp_send_output_to_next_stage=Mock(return_value=send_work),
        _pp_recv_dict_from_prev_stage=Mock(),
        _pp_commit_comm_work=Mock(),
    )

    _, _, _, work = SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
        scheduler,
        next_first_rank_mb_id=0,
        next_mb_id=1,
        mbs=[origin_target, None],
        mb_metadata=[object(), None],
        last_rank_comm_queue=deque(),
        pp_outputs=None,
        relay_output_immediately=True,
    )

    assert work is send_work
    scheduler._pp_send_output_to_next_stage.assert_called_once()
    scheduler._pp_recv_dict_from_prev_stage.assert_not_called()
    scheduler._pp_commit_comm_work.assert_not_called()


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
