import types

from sglang.srt.model_executor.model_runner_components.misc_utils import (
    get_pp_proxy_tensor_ownership,
    get_pp_proxy_token_scatter_factor,
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
    assert get_pp_proxy_token_scatter_factor(model, 4, incoming=True) == 4
    assert get_pp_proxy_token_scatter_factor(model, 4, incoming=False) == 4


def test_full_boundary_keeps_all_proxy_keys_replicated():
    full = types.SimpleNamespace(name="TP_ATTN_FULL")
    model = _wrapped_partition(full, full)
    assert not get_pp_proxy_tensor_ownership(model)
    assert get_pp_proxy_token_scatter_factor(model, 4, incoming=True) == 1
    assert get_pp_proxy_token_scatter_factor(model, 4, incoming=False) == 1


def test_incoming_and_outgoing_boundaries_are_independent():
    scattered = types.SimpleNamespace(name="SCATTERED")
    full = types.SimpleNamespace(name="TP_ATTN_FULL")
    model = types.SimpleNamespace(
        language_model=types.SimpleNamespace(
            model=_wrapped_partition(scattered, full).model
        )
    )

    assert not get_pp_proxy_tensor_ownership(model)
    assert get_pp_proxy_token_scatter_factor(model, 8, incoming=True) == 8
    assert get_pp_proxy_token_scatter_factor(model, 8, incoming=False) == 1
