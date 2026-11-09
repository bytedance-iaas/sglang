from __future__ import annotations

import math

import torch


class CappedRequestCompensation:
    """Experimental priority credit, without changing the SPS or graph budget.

    A cap hit is evidence of censoring, not proof that an unverified token is
    correct. Only bounded ranking credit is granted; observed rejection decays
    it. Owners are kept only for the current batch to prevent slot/id reuse.
    """

    def __init__(self, *, pool_size, device, max_boost, credit_cap=8):
        if not math.isfinite(max_boost) or not 0 < max_boost <= 1:
            raise ValueError("compensation max_boost must be in (0, 1]")
        if credit_cap <= 0:
            raise ValueError("compensation credit_cap must be positive")
        self.max_boost = float(max_boost)
        self.credit_cap = int(credit_cap)
        self.credit = torch.zeros(pool_size, device=device, dtype=torch.float32)
        self._owners = {}

    def initialize_pool(self, pool_size):
        """Bind late-created request storage without resetting accumulated credit."""
        if pool_size <= 0:
            raise ValueError("compensation request pool must have positive size")
        if self.credit.numel() == 0:
            self.credit = torch.zeros(
                pool_size, device=self.credit.device, dtype=self.credit.dtype
            )
        elif self.credit.numel() != pool_size:
            raise ValueError("compensation request pool size changed after binding")

    def prepare(self, *, reqs, req_pool_indices, req_pool_indices_cpu):
        owners = dict(zip(req_pool_indices_cpu.tolist(), reqs))
        reset = [
            slot for slot, req in owners.items() if self._owners.get(slot) is not req
        ]
        self._owners = owners
        if reset:
            slots = torch.tensor(reset, device=self.credit.device, dtype=torch.int64)
            self.credit.index_fill_(0, slots, 0)
        credit = self.credit[req_pool_indices.to(torch.int64)]
        return credit, 1.0 + credit * (self.max_boost / self.credit_cap)

    def update(self, *, req_pool_indices, verify_lens, commit_lens, full_width):
        slots = req_pool_indices.to(torch.int64)
        old = self.credit[slots]
        cap_hit = (verify_lens < full_width) & (commit_lens >= verify_lens)
        updated = torch.where(cap_hit, (old + 1).clamp_max(self.credit_cap), old * 0.5)
        self.credit[slots] = updated
