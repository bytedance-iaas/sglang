from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RecoveryLifecycle(str, Enum):
    LIVE = "live"
    STORE_PENDING = "store_pending"
    EVICTED = "evicted"
    CLOSED = "closed"
    RELEASED = "released"


class RecoveryDescriptorError(ValueError):
    pass


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionRecoveryDescriptor:
    """Fail-closed contract for one exact session-state recovery cut.

    ``cut`` is the number of consumed tokens; the next input position is
    ``cut``. Storage can be padded beyond it, but every effective component
    and the scheduler credit must describe the same cut. This interface is a
    contract prototype: it does not enable PD decode radix cache or migrate
    state by itself.
    """

    tenant_id: str
    session_id: str
    session_generation: int
    slot_id: int
    slot_generation: int
    owner_dp_rank: int

    cut: int
    dsa_kv_valid_cut: int
    dsa_index_valid_cut: int
    dsa_kv_allocated_len: int
    dsa_index_allocated_rows: int
    kda_state_position: int
    scheduler_credited_len: int
    recompute_start: int

    committed_draft_tokens: int
    committed_bonus_tokens: int
    uncommitted_tail_start: int
    uncommitted_tail_end: int

    page_size: int = 256
    kpool_size: int = 4
    store_complete: bool = False
    lifecycle: RecoveryLifecycle = RecoveryLifecycle.STORE_PENDING

    @property
    def identity(self) -> tuple[str, str, int, int, int]:
        return (
            self.tenant_id,
            self.session_id,
            self.session_generation,
            self.slot_id,
            self.slot_generation,
        )

    @property
    def minimum_index_rows(self) -> int:
        return _ceil_div(self.cut, self.kpool_size)

    def validate_for_restore(
        self,
        *,
        tenant_id: str,
        session_id: str,
        session_generation: int,
        slot_id: int,
        slot_generation: int,
        prompt_len: int,
    ) -> range:
        errors: list[str] = []

        expected_identity = (
            tenant_id,
            session_id,
            session_generation,
            slot_id,
            slot_generation,
        )
        if self.identity != expected_identity:
            errors.append(
                f"identity mismatch: descriptor={self.identity!r}, "
                f"request={expected_identity!r}"
            )

        if self.lifecycle is not RecoveryLifecycle.LIVE:
            errors.append(f"lifecycle is {self.lifecycle.value}, not live")
        if not self.store_complete:
            errors.append("async store has not completed")
        if self.session_generation <= 0 or self.slot_generation <= 0:
            errors.append("session and slot generations must be positive")
        if self.slot_id < 0 or self.owner_dp_rank < 0:
            errors.append("slot_id and owner_dp_rank must be nonnegative")
        if self.page_size <= 0 or self.kpool_size <= 0:
            errors.append("page_size and kpool_size must be positive")
        if prompt_len < 0 or self.cut < 0 or self.cut > prompt_len:
            errors.append(
                f"invalid recovery cut {self.cut} for prompt_len {prompt_len}"
            )

        effective_positions = {
            "dsa_kv_valid_cut": self.dsa_kv_valid_cut,
            "dsa_index_valid_cut": self.dsa_index_valid_cut,
            "kda_state_position": self.kda_state_position,
            "scheduler_credited_len": self.scheduler_credited_len,
            "recompute_start": self.recompute_start,
            "uncommitted_tail_start": self.uncommitted_tail_start,
        }
        for name, position in effective_positions.items():
            if position != self.cut:
                errors.append(f"{name}={position} does not match cut={self.cut}")

        if self.dsa_kv_allocated_len < self.cut:
            errors.append("DSA KV allocation ends before the valid cut")
        if self.page_size > 0 and self.dsa_kv_allocated_len % self.page_size:
            errors.append(
                f"DSA KV allocation {self.dsa_kv_allocated_len} is not "
                f"page_size={self.page_size} aligned"
            )
        if self.kpool_size > 0 and (
            self.dsa_index_allocated_rows < self.minimum_index_rows
        ):
            errors.append(
                f"DSA index has {self.dsa_index_allocated_rows} rows, needs at "
                f"least {self.minimum_index_rows} for cut={self.cut} and "
                f"kpool_size={self.kpool_size}"
            )

        if self.committed_draft_tokens < 0 or self.committed_bonus_tokens < 0:
            errors.append("committed speculative counts must be nonnegative")
        if self.committed_draft_tokens + self.committed_bonus_tokens > self.cut:
            errors.append("committed speculative counts exceed the recovery cut")
        if self.uncommitted_tail_end < self.uncommitted_tail_start:
            errors.append("uncommitted tail ends before it starts")

        if errors:
            raise RecoveryDescriptorError("; ".join(errors))
        return range(self.recompute_start, prompt_len)
