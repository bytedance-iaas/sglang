"""Short decoder replay chunks retain the preceding chunk's SWA window (#41303)."""

import pytest
import torch

from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    late_layer_tail_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("extend", [0, 1, 32, 127, 128, 129, 512])
@pytest.mark.parametrize("prefix", [0, 64, 8192])
def test_single_short_tail(extend, prefix):
    rows, lengths, floor = late_layer_tail_layout(
        extend_lens_cpu=[extend],
        seq_lens_cpu=[prefix + extend],
        tail_len=128,
        device=torch.device("cpu"),
    )
    expected = list(range(max(0, extend - 128), extend))
    assert rows.tolist() == expected
    assert lengths == [len(expected)]
    assert floor.tolist() == [max(0, prefix + extend - 128)] * len(expected)


def test_mixed_short_tail_uses_absolute_window_floor():
    extends = [200, 0, 1, 32, 127, 128, 129]
    prefixes = [100, 8192, 8192, 0, 8192, 0, 4096]
    rows, lengths, floor = late_layer_tail_layout(
        extend_lens_cpu=extends,
        seq_lens_cpu=[p + e for p, e in zip(prefixes, extends)],
        tail_len=128,
        device=torch.device("cpu"),
    )
    expected_rows, expected_floor, start = [], [], 0
    for prefix, extend in zip(prefixes, extends):
        for pos in range(max(0, extend - 128), extend):
            expected_rows.append(start + pos)
            expected_floor.append(max(0, prefix + extend - 128))
        start += extend
    assert rows.tolist() == expected_rows
    assert lengths == [min(128, e) for e in extends]
    assert floor.tolist() == expected_floor
    assert floor.dtype == torch.int32
