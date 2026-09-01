import types

from sglang.srt.model_executor.model_runner_components.misc_utils import (
    get_pp_proxy_tensor_ownership,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _layer(input_mode, output_mode):
    return types.SimpleNamespace(
        layer_scatter_modes=types.SimpleNamespace(
            layer_input_mode=input_mode, layer_output_mode=output_mode
        )
    )


def _wrapped_partition(input_mode, output_mode):
    missing = types.SimpleNamespace()
    transformer = types.SimpleNamespace(
        layers=[missing, _layer(input_mode, output_mode), missing],
        start_layer=1,
        end_layer=2,
    )
    return types.SimpleNamespace(model=transformer)


def test_scattered_boundary_owns_hidden_and_residual():
    mode = types.SimpleNamespace(name="SCATTERED")
    model = _wrapped_partition(mode, mode)
    assert get_pp_proxy_tensor_ownership(model) == {"hidden_states", "residual"}


def test_full_boundary_keeps_all_proxy_keys_replicated():
    full = types.SimpleNamespace(name="TP_ATTN_FULL")
    model = _wrapped_partition(full, full)
    assert not get_pp_proxy_tensor_ownership(model)


def test_only_outgoing_boundary_controls_transport_ownership():
    scattered = types.SimpleNamespace(name="SCATTERED")
    full = types.SimpleNamespace(name="TP_ATTN_FULL")
    model = types.SimpleNamespace(
        language_model=types.SimpleNamespace(
            model=_wrapped_partition(scattered, full).model
        )
    )

    assert not get_pp_proxy_tensor_ownership(model)


def test_explicit_model_ownership_is_preserved():
    full = types.SimpleNamespace(name="TP_ATTN_FULL")
    model = _wrapped_partition(full, full)
    model.pp_proxy_tensors_all_gather_exclude = {"v_first", "topk_indices"}

    assert get_pp_proxy_tensor_ownership(model) == {"v_first", "topk_indices"}


def test_explicit_and_boundary_ownership_are_combined():
    scattered = types.SimpleNamespace(name="SCATTERED")
    model = _wrapped_partition(scattered, scattered)
    model.pp_proxy_tensors_all_gather_exclude = {"topk_indices"}

    assert get_pp_proxy_tensor_ownership(model) == {
        "hidden_states",
        "residual",
        "topk_indices",
    }
