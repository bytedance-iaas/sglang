"""Owner assignment + peak-shifting schedule (D4)."""

from typing import List, Tuple


def owner_of(layer_idx: int, dp_size: int) -> int:
    return layer_idx % dp_size


def is_local_layer(layer_idx: int, dp_rank: int, dp_size: int, k: int) -> bool:
    for i in range(k):
        if layer_idx % dp_size == (dp_rank + i) % dp_size:
            return True
    return False


def shift_waves(rank: int, D: int, k: int) -> List[Tuple[int, int, int, int]]:
    """Generate the peak-shifting schedule for one rank.

    Returns list of (recv_pos, src_rank, send_pos, dst_rank).
    Each wave is a permutation — no incast.
    """
    plan = []
    for w in range(k, D):
        recv_pos = (rank + w) % D
        src = recv_pos
        send_pos = rank
        dst = (rank - w) % D
        plan.append((recv_pos, src, send_pos, dst))
    return plan


def remote_positions(
    dp_rank: int, dp_size: int, k: int, peak_shifting: bool = False
) -> List[int]:
    """Return remote positions within one cycle.

    The default follows model compute order and is the A/B baseline. With
    ``peak_shifting=True``, positions follow this rank's permutation waves.
    """
    if peak_shifting:
        return [recv_pos for recv_pos, _, _, _ in shift_waves(dp_rank, dp_size, k)]
    return [
        pos for pos in range(dp_size) if not is_local_layer(pos, dp_rank, dp_size, k)
    ]


def prefetch_order(
    dp_rank: int,
    dp_size: int,
    k: int,
    num_layers: int,
    peak_shifting: bool = False,
) -> List[int]:
    """Return non-local layers with cycle as the outer scheduling unit."""
    order = []
    positions = remote_positions(dp_rank, dp_size, k, peak_shifting)
    for base in range(0, num_layers, dp_size):
        for pos in positions:
            lid = base + pos
            if lid < num_layers:
                order.append(lid)
    return order
