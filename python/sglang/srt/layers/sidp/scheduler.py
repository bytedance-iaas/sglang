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


def prefetch_order(dp_rank: int, dp_size: int, k: int, num_layers: int) -> List[int]:
    """Return the ordered list of non-local layer indices this rank needs to prefetch,
    organized by shift_waves order for peak-shifting."""
    order = []
    waves = shift_waves(dp_rank, dp_size, k)
    for recv_pos, _, _, _ in waves:
        for base in range(0, num_layers, dp_size):
            lid = base + recv_pos
            if lid < num_layers:
                order.append(lid)
    return order
