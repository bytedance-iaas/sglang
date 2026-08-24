import pytest

from sglang.srt.layers.sidp.scheduler import owner_of, prefetch_order
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_compute_order_is_cycle_outer():
    assert prefetch_order(0, 8, 1, 24) == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
    ]


def test_peak_shifting_is_a_permutation_at_every_wave():
    dp_size, k, num_layers = 8, 1, 48
    orders = [
        prefetch_order(rank, dp_size, k, num_layers, peak_shifting=True)
        for rank in range(dp_size)
    ]
    remote_per_cycle = dp_size - k
    for cycle in range(num_layers // dp_size):
        offset = cycle * remote_per_cycle
        for wave in range(remote_per_cycle):
            owners = {
                owner_of(orders[rank][offset + wave], dp_size)
                for rank in range(dp_size)
            }
            assert owners == set(range(dp_size))


def test_peak_shifting_changes_only_order_not_coverage():
    for rank in range(8):
        compute_order = prefetch_order(rank, 8, 1, 48)
        shifted_order = prefetch_order(rank, 8, 1, 48, peak_shifting=True)
        assert set(shifted_order) == set(compute_order)
        assert len(shifted_order) == len(compute_order)

    assert prefetch_order(1, 8, 1, 8) == [0, 2, 3, 4, 5, 6, 7]
    assert prefetch_order(1, 8, 1, 8, peak_shifting=True) == [2, 3, 4, 5, 6, 7, 0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
