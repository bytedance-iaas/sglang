"""CPU-only contracts for the conditional-DMA Graph boundary."""

import pytest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def make_args(**kwargs):
    values = dict(
        model_path="dummy",
        device="cuda",
        dp_size=8,
        tp_size=1,
        sidp_size=8,
        sidp_rdzv_port=29347,
        sidp_prefetch_policy="dynamic_owner",
        sidp_copy_backend="dma",
        cuda_graph_config={
            "decode": {"backend": "full"},
            "prefill": {"backend": "disabled"},
        },
    )
    values.update(kwargs)
    # dummy skips hardware/model post-init resolution. Exercise the actual
    # SiDP and Graph parsers without loading weights or creating CUDA resources.
    args = ServerArgs(**values)
    args._handle_sidp()
    args._parse_cuda_graph_config()
    args._validate_sidp_graph_config()
    return args


@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("claim_order", ["rotating", "compute_priority"])
def test_dynamic_dma_full_decode(k, claim_order):
    args = make_args(sidp_k=k, sidp_dynamic_claim_order=claim_order)
    assert args.cuda_graph_config.decode.backend == "full"
    assert args.cuda_graph_config.prefill.backend == "disabled"
    assert args.sidp_peak_sync_strategy == "none"
    assert args.sidp_dynamic_claim_order == claim_order
    assert args.sidp_slot_sync == "flag"


def test_dynamic_dma_eager_and_explicit_graph_override():
    args = make_args(disable_cuda_graph=True, cuda_graph_config=None)
    assert args.cuda_graph_config.decode.backend == "disabled"
    # Explicit JSON wins over the legacy bool, including with profiling.
    args = make_args(disable_cuda_graph=True, sidp_enable_graph_profiling=True)
    assert args.cuda_graph_config.decode.backend == "full"


@pytest.mark.parametrize(
    "phase,backend",
    [
        ("decode", "breakable"),
        ("decode", "tc_piecewise"),
        ("prefill", "full"),
        ("prefill", "breakable"),
        ("prefill", "tc_piecewise"),
    ],
)
def test_reject_unsupported_capture_backend(phase, backend):
    graph = {"decode": {"backend": "full"}, "prefill": {"backend": "disabled"}}
    graph[phase]["backend"] = backend
    with pytest.raises(ValueError, match="full decode"):
        make_args(cuda_graph_config=graph)


@pytest.mark.parametrize(
    "extra",
    [
        {"enable_torch_compile": True},
        {"sidp_peak_sync_strategy": "force_sync"},
        {"sidp_sm_use_event_sync": True},
        {"sidp_sm_copy_ctas": 10},
        {"sidp_dma_slices": 3},
        {"enable_memory_saver": True},
        {"enable_pdmux": True},
        {"sidp_k": 8},
    ],
)
def test_reject_unsupported_combinations(extra):
    with pytest.raises((ValueError, AssertionError)):
        make_args(**extra)


def test_profile_rejects_effective_eager():
    with pytest.raises(ValueError, match="profiling requires"):
        make_args(
            cuda_graph_config={
                "decode": {"backend": "disabled"},
                "prefill": {"backend": "disabled"},
            },
            sidp_enable_graph_profiling=True,
        )


@pytest.mark.parametrize(
    "policy,peak,expected",
    [
        ("auto", False, ("compute", "dma")),
        ("auto", True, ("static_peak", "dma")),
        ("dynamic_owner", False, ("dynamic_owner", "sm")),
    ],
)
def test_auto_mapping_unchanged(policy, peak, expected):
    args = make_args(
        sidp_prefetch_policy=policy,
        sidp_copy_backend="auto",
        sidp_enable_peak_shifting=peak,
    )
    assert (args.sidp_prefetch_policy, args.sidp_copy_backend) == expected
    assert args.sidp_slot_sync == (
        "event" if expected == ("static_peak", "dma") else "flag"
    )


def test_sidp_disabled_ignores_new_gate():
    args = ServerArgs(model_path="dummy", sidp_size=0)
    assert args.sidp_slot_sync == "auto"
    # Even unresolved Graph config is not inspected when SiDP is disabled.
    args._validate_sidp_graph_config()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
