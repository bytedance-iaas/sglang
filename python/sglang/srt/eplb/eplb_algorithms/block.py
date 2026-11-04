"""Bmax/Tmax-only EPLB placement with per-observed-step Pareto acceptance.

DeepSeek supplies variable replica counts. This stage places copies using integer
uniform quotas (same model for incumbent and candidate), then accepts only if
neither tail regresses on ANY training step. No Bsum score. This is a historical
prediction contract; unseen requests still require runtime LPLB protection.
"""

import numpy as np
import torch

from . import deepseek


def physical_loads(counts, mapping):
    out = np.zeros((len(counts), len(mapping)), dtype=np.int64)
    for expert in range(counts.shape[1]):
        slots = np.flatnonzero(mapping == expert)
        if not len(slots):
            raise ValueError("Missing logical expert")
        base, rem = np.divmod(counts[:, expert], len(slots))
        for j, slot in enumerate(slots):
            out[:, slot] = base + (j < rem)
    return out


def tails(loads, ranks, block_size):
    return np.stack(
        (
            ((loads + block_size - 1) // block_size)
            .reshape(len(loads), ranks, -1)
            .sum(2)
            .max(1),
            loads.reshape(len(loads), ranks, -1).sum(2).max(1),
        ),
        axis=1,
    )


def acceptable(reference, candidate):
    return bool(np.all(candidate <= reference) and np.any(candidate < reference))


def candidate_layout(counts, replica_counts, ranks, slots, block_size):
    # Place largest copies first, respecting rank capacity and avoiding colocated
    # replicas whenever there is an eligible rank. Peak ratios jointly score B/T.
    items = []
    for e, c in enumerate(replica_counts):
        base, rem = np.divmod(counts[:, e], c)
        for j in range(c):
            items.append((e, base + (j < rem)))
    items.sort(key=lambda z: (-int(z[1].max()), -int(z[1].sum()), z[0]))
    rank_t = np.zeros((len(counts), ranks), dtype=np.int64)
    rank_b = rank_t.copy()
    layout = [[] for _ in range(ranks)]
    tref = np.maximum(1, counts.sum(1) / ranks)
    bref = np.maximum(1, tref / block_size)
    for e, v in items:
        eligible = [
            r for r in range(ranks) if len(layout[r]) < slots and e not in layout[r]
        ]
        if not eligible:
            eligible = [r for r in range(ranks) if len(layout[r]) < slots]
        best = None
        for r in eligible:
            t = rank_t.copy()
            b = rank_b.copy()
            t[:, r] += v
            b[:, r] += (v + block_size - 1) // block_size
            ratios = np.stack((b.max(1) / bref, t.max(1) / tref), axis=1)
            score = (float(ratios.max()), float(ratios.sum()), r)
            if best is None or score < best[0]:
                best = (score, r)
        r = best[1]
        layout[r].append(e)
        rank_t[:, r] += v
        rank_b[:, r] += (v + block_size - 1) // block_size
    return np.asarray(sum(layout, []), dtype=np.int64)


def refine_layout(counts, mapping, ranks, block_size, attempts=2048):
    mapping = mapping.copy()
    loads = physical_loads(counts, mapping)
    slots = len(mapping) // ranks
    blocks = (loads + block_size - 1) // block_size
    rt = loads.reshape(len(loads), ranks, slots).sum(2)
    rb = blocks.reshape(len(loads), ranks, slots).sum(2)
    ref = np.stack((rb.max(1), rt.max(1)), axis=1)
    rng = np.random.default_rng(12345)
    for a, b in rng.integers(0, len(mapping), size=(attempts, 2)):
        ra, rr = a // slots, b // slots
        if ra == rr or mapping[a] == mapping[b]:
            continue
        # Preserve distinct-rank replicas when possible.
        if (
            mapping[a] in mapping[rr * slots : (rr + 1) * slots]
            or mapping[b] in mapping[ra * slots : (ra + 1) * slots]
        ):
            continue
        t = rt.copy()
        bl = rb.copy()
        delta = loads[:, b] - loads[:, a]
        db = blocks[:, b] - blocks[:, a]
        t[:, ra] += delta
        t[:, rr] -= delta
        bl[:, ra] += db
        bl[:, rr] -= db
        cand = np.stack((bl.max(1), t.max(1)), axis=1)
        if not acceptable(ref, cand):
            continue
        proposal = mapping.copy()
        proposal[a], proposal[b] = proposal[b], proposal[a]
        exact = physical_loads(counts, proposal)
        exact_tail = tails(exact, ranks, block_size)
        if not acceptable(ref, exact_tail):
            continue
        mapping = proposal
        loads = exact
        blocks = (loads + block_size - 1) // block_size
        rt = loads.reshape(len(loads), ranks, slots).sum(2)
        rb = blocks.reshape(len(loads), ranks, slots).sum(2)
        ref = exact_tail
    return mapping


def rebalance_experts(
    tokens_per_expert,
    num_physical_experts,
    num_local_physical_experts,
    num_groups,
    num_nodes,
    block_size,
    enable_hierarchical,
):
    if block_size not in (64, 128):
        raise ValueError("Block-EPLB requires block_size=64 or 128")
    counts = tokens_per_expert.to(dtype=torch.int64, device="cpu")
    if counts.ndim != 3 or counts.shape[0] == 0 or torch.any(counts < 0):
        raise ValueError("Expected nonnegative [steps, layers, experts] counts")
    ranks = num_physical_experts // num_local_physical_experts
    phy, log, rep = deepseek.rebalance_experts(
        weight=counts.sum(0),
        num_replicas=num_physical_experts,
        num_groups=num_groups,
        num_nodes=1,
        num_gpus=ranks,
        enable_hierarchical=False,
    )
    accepted = 0
    for layer in range(counts.shape[1]):
        c = counts[:, layer].numpy()
        old = phy[layer].numpy()
        candidate = candidate_layout(
            c, rep[layer].numpy(), ranks, num_local_physical_experts, block_size
        )
        reference = tails(physical_loads(c, old), ranks, block_size)
        predicted = tails(physical_loads(c, candidate), ranks, block_size)
        start = candidate if acceptable(reference, predicted) else old
        candidate = refine_layout(c, start, ranks, block_size)
        predicted = tails(physical_loads(c, candidate), ranks, block_size)
        if acceptable(reference, predicted):
            phy[layer] = torch.from_numpy(candidate)
            log[layer].fill_(-1)
            for e in range(c.shape[1]):
                ids = torch.where(phy[layer] == e)[0]
                log[layer, e, : len(ids)] = ids
            accepted += 1
    import logging

    logging.getLogger(__name__).info(
        "Block-EPLB accepted %d/%d layers; Bsum excluded",
        accepted,
        counts.shape[1],
    )
    return phy, log, rep
