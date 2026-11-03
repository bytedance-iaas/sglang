"""CPU mapping buffers for the opt-in migration path.

Stable physical-slot order and first-local preference match native metadata.
NumPy fills preallocated tensors, avoiding per-expert nested Python lists.
"""

import numpy as np
import torch


class CpuMetadataMirror:
    def __init__(self, physical, logical, rank, world):
        assert physical.device.type == logical.device.type == "cpu"
        self.num_physical = physical.numel()
        self.num_logical = logical.shape[0]
        assert self.num_physical % world == 0
        self.num_local = self.num_physical // world
        self.physical, self.logical, self.rank = physical, logical, rank
        self.valid = torch.empty(self.num_logical, dtype=torch.int64)
        self.p, self.l, self.v = physical.numpy(), logical.numpy(), self.valid.numpy()
        self.slots = np.arange(self.num_physical, dtype=np.int64)
        self.first_local = np.empty(self.num_logical, dtype=np.int64)
        self.local_slots = self.slots[
            rank * self.num_local : (rank + 1) * self.num_local
        ]

    def update(self, mapping):
        self.p[:] = mapping
        counts = np.bincount(self.p, minlength=self.num_logical)
        assert len(counts) == self.num_logical and counts.min() >= 1
        assert counts.max() <= self.l.shape[1]
        # Stable sort preserves ascending physical slot order within experts.
        order = np.argsort(self.p, kind="stable")
        starts = np.cumsum(counts) - counts
        ordinal = self.slots - np.repeat(starts, counts)
        self.l.fill(-1)
        self.l[self.p[order], ordinal] = order
        self.v[:] = counts
        self.first_local.fill(self.num_physical)
        np.minimum.at(self.first_local, self.p[self.local_slots], self.local_slots)
        local = np.flatnonzero(self.first_local < self.num_physical)
        self.l[local, :] = -1
        self.l[local, 0] = self.first_local[local]
        self.v[local] = 1
        return self.physical, self.logical, self.valid
