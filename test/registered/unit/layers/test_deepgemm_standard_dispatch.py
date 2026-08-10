import pytest

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    _get_compact_all_tokens,
    _should_use_masked_standard_layout,
)


@pytest.mark.parametrize(
    ("num_assignments", "num_experts", "expected"),
    [
        (1, 256, 128),
        (8, 256, 1024),
        (256, 256, 32768),
        (1024, 256, 33536),
    ],
)
def test_compact_all_tokens_is_graph_static_and_tightly_bounded(
    num_assignments, num_experts, expected
):
    assert _get_compact_all_tokens(num_assignments, num_experts) == expected


def test_standard_dispatch_layout_uses_masked_only_for_small_ep_partition():
    common = dict(
        hidden_size=4096,
        intermediate_size_per_partition=2048,
        top_k=8,
        activation="silu",
        is_gated=True,
    )
    assert _should_use_masked_standard_layout(
        MoeRunnerConfig(num_experts=256, num_local_experts=32, **common)
    )
    assert not _should_use_masked_standard_layout(
        MoeRunnerConfig(num_experts=256, num_local_experts=64, **common)
    )
    assert not _should_use_masked_standard_layout(
        MoeRunnerConfig(num_experts=256, num_local_experts=256, **common)
    )
