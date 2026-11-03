"""Fused specialization constraints, independent of PyTorch.

Model dimensions are explicit; model profiles belong in tests.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    hidden: int
    intermediate: int
    topk: int
    physical_experts: int
    local_slots: int

    @property
    def dimensions(self):
        return (
            self.hidden,
            self.intermediate,
            self.topk,
            self.physical_experts,
            self.local_slots,
        )

    @property
    def abi(self):
        return (*self.dimensions, 64, 32)

    @property
    def key(self):
        return "_".join(map(str, self.dimensions))

    @property
    def cuda_flags(self):
        return tuple(
            f"-DIAAS_EARLY_{name}={value}"
            for name, value in zip("HIKES", self.dimensions)
        )

    @property
    def world(self):
        return self.physical_experts // self.local_slots

    def validate(self):
        if any(not isinstance(v, int) or v <= 0 for v in self.dimensions):
            raise ValueError("All early-MoE dimensions must be positive integers")
        if self.hidden % 256 or self.intermediate % 128:
            raise ValueError("SM90 tiles require H divisible by 256 and I by 128")
        if self.physical_experts % self.local_slots or not 1 <= self.world <= 32:
            raise ValueError("Physical slots must evenly cover 1..32 EP ranks")
        if self.topk > self.physical_experts:
            raise ValueError("top-k exceeds the physical expert domain")
