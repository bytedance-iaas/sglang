"""#47406 on POC: retain PP proxies, compact CP rows and DSpark auxiliaries."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner import eager_runner
from sglang.srt.models import deepseek_v4, deepseek_v4_nextn
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@contextmanager
def _shard(*_args):
    yield None, None, None


def _case(*, bounded=True, capture=True, last=True, aux_list=False):
    batch = SimpleNamespace(
        input_ids=torch.arange(512),
        positions=torch.arange(512),
        attn_cp_metadata=object(),
        return_logprob=False,
        extend_seq_lens_cpu=[302, 210],
        extend_logprob_start_lens_cpu=[302, 210],
    )
    tail = LateLayerTail(
        token_indices=torch.tensor([75, 127]),
        positions=torch.arange(2),
        extend_seq_lens=torch.tensor([2, 2]),
        extend_seq_lens_cpu=[2, 2],
        swa_out_cache_loc=torch.arange(4),
        cp_metadata=object(),
        global_token_indices=torch.tensor([300, 301, 510, 511]),
    )
    local = (torch.ones(2, 3), torch.ones(2, 6))
    aux = [torch.ones(2, 5), torch.ones(2, 7)] if aux_list else torch.ones(2, 9)
    body = Mock(spec=[], return_value=(local, aux) if capture else local)
    body.late_layer_start = 3 if bounded else None
    model = deepseek_v4.DeepseekV4ForCausalLM.__new__(deepseek_v4.DeepseekV4ForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = body
    model.pp_group = SimpleNamespace(is_last_rank=last)
    model.capture_aux_hidden_states = capture
    model.prepare_cp_inputs = lambda *_a, **_k: (None, batch.positions, {})
    model.logits_processor = Mock(return_value=SimpleNamespace())
    model.lm_head = object()
    runner = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=model,
            attn_backend=SimpleNamespace(
                tail_forward_metadata=SimpleNamespace(late_layer_tail=tail)
            ),
        )
    )
    return runner, model, batch, tail, local, aux


@pytest.mark.parametrize("aux_list", [False, True])
def test_compact_global_rows_and_tensor_or_list_aux_reach_logits(aux_list):
    runner, model, batch, tail, local, aux = _case(aux_list=aux_list)
    original = batch.attn_cp_metadata
    seen = []

    def gather(value, fb, stream):
        seen.append(fb.attn_cp_metadata)
        if isinstance(value, tuple):
            return tuple(torch.arange(4 * v.shape[1]).view(4, -1) for v in value)
        return torch.arange(4 * value.shape[1]).view(4, -1)

    with (
        patch.object(eager_runner, "cp_shard_model_inputs", _shard),
        patch.object(eager_runner, "cp_gather_after_forward", side_effect=gather),
        patch.object(torch.cuda, "current_stream", return_value=None),
        patch(
            "sglang.srt.layers.logits_processor.LogitsMetadata.from_forward_batch",
            return_value=SimpleNamespace(),
        ),
    ):
        output = eager_runner.EagerRunner._execute_extend_cp(runner, batch, {})
    args, kwargs = model.logits_processor.call_args
    assert args[0].tolist() == [300, 301, 510, 511]
    assert args[1].shape == (4, 3)
    assert (
        args[3].extend_seq_lens_cpu == args[3].extend_logprob_start_lens_cpu == [2, 2]
    )
    assert (
        [x.shape[0] for x in args[4]] == [4, 4] if aux_list else args[4].shape[0] == 4
    )
    assert "hidden_states_before_norm" not in kwargs
    assert seen == [tail.cp_metadata] * (3 if aux_list else 2)
    assert output.hidden_states_token_indices is tail.global_token_indices
    assert batch.attn_cp_metadata is original


@pytest.mark.parametrize("capture", [False, True])
def test_non_bounded_cp_preserves_full_layout_and_pre_head(capture):
    runner, model, batch, tail, local, aux = _case(bounded=False, capture=capture)
    original = batch.attn_cp_metadata
    with (
        patch.object(eager_runner, "cp_shard_model_inputs", _shard),
        patch.object(
            eager_runner, "cp_gather_after_forward", side_effect=lambda x, *_: x
        ),
        patch.object(torch.cuda, "current_stream", return_value=None),
    ):
        output = eager_runner.EagerRunner._execute_extend_cp(runner, batch, {})
    args, kwargs = model.logits_processor.call_args
    assert args[0] is batch.input_ids
    assert args[3] is batch
    assert args[4] is (aux if capture else None)
    assert ("hidden_states_before_norm" in kwargs) is not capture
    if not capture:
        assert kwargs["hidden_states_before_norm"] is local[1]
    assert not hasattr(output, "hidden_states_token_indices")
    assert batch.attn_cp_metadata is original


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_collective_failure_propagates_and_restores_metadata(fail_at):
    runner, model, batch, *_ = _case(aux_list=True)
    original = batch.attn_cp_metadata
    calls = 0

    def gather(value, *_):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise RuntimeError("collective failed")
        return value

    with (
        patch.object(eager_runner, "cp_shard_model_inputs", _shard),
        patch.object(eager_runner, "cp_gather_after_forward", side_effect=gather),
        patch.object(torch.cuda, "current_stream", return_value=None),
        pytest.raises(RuntimeError, match="collective failed"),
    ):
        eager_runner.EagerRunner._execute_extend_cp(runner, batch, {})
    assert batch.attn_cp_metadata is original
    model.logits_processor.assert_not_called()


@pytest.mark.parametrize("capture", [False, True])
def test_middle_pp_stage_returns_packed_proxy_before_aux_unpack(capture):
    runner, model, batch, *_ = _case(last=False, capture=capture)
    proxy = PPProxyTensors(
        {
            "hidden_states": torch.ones(2, 3),
            "dspark_aux_hidden_states": torch.ones(2, 9),
        }
    )
    model.model.return_value = proxy
    with (
        patch.object(eager_runner, "cp_shard_model_inputs", _shard),
        patch.object(eager_runner, "cp_gather_after_forward") as gather,
        patch.object(torch.cuda, "current_stream") as stream,
    ):
        output = eager_runner.EagerRunner._execute_extend_cp(runner, batch, {})
    assert output is proxy
    gather.assert_not_called()
    stream.assert_not_called()
    model.logits_processor.assert_not_called()


def test_nextn_explicitly_disables_inherited_target_aux_capture():
    parallel = SimpleNamespace(
        tp_size=4, pp_group=SimpleNamespace(is_last_rank=True), enable_dp_lm_head=False
    )
    with (
        patch.object(deepseek_v4_nextn, "get_parallel", return_value=parallel),
        patch.object(deepseek_v4_nextn, "wo_a_fp8_gemm_enabled", return_value=False),
        patch.object(
            deepseek_v4_nextn.DeepseekV4ForCausalLMNextN,
            "determine_num_fused_shared_experts",
        ),
        patch.object(
            deepseek_v4_nextn, "DeepseekV4ModelNextN", return_value=torch.nn.Identity()
        ),
        patch.object(
            deepseek_v4_nextn, "ParallelLMHead", return_value=torch.nn.Identity()
        ),
        patch.object(deepseek_v4_nextn, "LogitsProcessor", return_value=Mock()),
    ):
        model = deepseek_v4_nextn.DeepseekV4ForCausalLMNextN(
            SimpleNamespace(vocab_size=16, hidden_size=3)
        )
    assert model.capture_aux_hidden_states is False


def test_non_dspark_forward_scatters_each_compact_tensor_only_once():
    compact = torch.tensor([[[10.0, 20.0, 30.0]], [[40.0, 50.0, 60.0]]])
    tail = LateLayerTail(
        token_indices=torch.tensor([2, 3]),
        positions=torch.arange(2),
        extend_seq_lens=torch.tensor([2]),
        extend_seq_lens_cpu=[2],
        swa_out_cache_loc=torch.arange(2),
    )
    model = SimpleNamespace(
        pp_group=SimpleNamespace(is_first_rank=False, is_last_rank=True, world_size=2),
        hc_mult=1,
        hidden_size=3,
        dspark_layers_to_capture=None,
        _install_pp_state=lambda *_: None,
        _can_run_tbo=lambda _: False,
        hc_pre_from_prev_sublayer=True,
        start_layer=0,
        layers=[SimpleNamespace(mlp=object())],
        norm=lambda x: x,
        _forward_layers_hc_pre_from_prev=Mock(
            return_value=(compact, torch.ones(2, 1), tail)
        ),
    )
    ids = torch.arange(4)
    batch = SimpleNamespace(attn_cp_metadata=None)
    proxy = PPProxyTensors({"hidden_states": torch.zeros(4, 3)})
    with (
        patch.object(
            deepseek_v4, "get_parallel", return_value=SimpleNamespace(attn_dp_size=1)
        ),
        patch.object(deepseek_v4, "get_attn_backend", return_value=object()),
        patch.object(deepseek_v4, "_is_npu", False),
        patch.object(
            deepseek_v4,
            "dsa_prefill_cp_fused_symm_mem",
            side_effect=lambda *_: nullcontext(),
        ),
        patch(
            "sglang.kernels.ops.layernorm.mhc.hc_combine", side_effect=lambda x, *_: x
        ),
        patch.object(
            deepseek_v4, "_scatter_tail_rows", wraps=deepseek_v4._scatter_tail_rows
        ) as scatter,
        envs.SGLANG_DSV41_MEGAMOE_REDUCE_SCATTER.override(False),
    ):
        hidden, pre = deepseek_v4.DeepseekV4Model.forward(
            model, ids, ids, batch, None, proxy
        )
    assert scatter.call_count == 2
    torch.testing.assert_close(hidden[2:], compact.flatten(1))
    torch.testing.assert_close(pre[2:], compact.flatten(1))
