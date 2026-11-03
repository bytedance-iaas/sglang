"""Rank token and row-block metrics shared by audits and reference tests."""

from sglang.srt.eplb.balance_config import BLOCK_SIZE


def metrics(q, block_size=BLOCK_SIZE, *, world):
    if len(q) % world:
        raise ValueError("Physical slots must divide evenly across ranks")
    slots = len(q) // world
    return (
        [sum(q[r * slots : (r + 1) * slots]) for r in range(world)],
        [
            sum(
                (v + block_size - 1) // block_size
                for v in q[r * slots : (r + 1) * slots]
            )
            for r in range(world)
        ],
    )
