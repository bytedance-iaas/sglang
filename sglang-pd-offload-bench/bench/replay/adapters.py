from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from bench.replay.base import ReplayAdapter
from bench.types import RequestRecord


@dataclass
class _BaseReplay(ReplayAdapter):
    name: str
    config: dict[str, Any] = field(default_factory=dict)

    def prepare(self, config: dict[str, Any]) -> None:
        self.config = config

    def generate_requests(self) -> list[dict[str, Any]]:
        total = int(self.config.get("num_requests", 30))
        return [{"id": f"{self.name}-{i}"} for i in range(total)]

    def replay(self, backend_name: str) -> list[RequestRecord]:
        requests = self.generate_requests()
        seed = hash((self.name, backend_name, self.config.get("seed", 7))) & 0xFFFF
        rng = random.Random(seed)
        base = float(self.config.get("ttft_base_ms", 120))
        variance = float(self.config.get("ttft_jitter_ms", 40))
        success = float(self.config.get("success_prob", 0.97))
        rows: list[RequestRecord] = []
        for r in requests:
            ttft = max(5.0, rng.gauss(base, variance))
            ok = rng.random() <= success
            rows.append(RequestRecord(request_id=r["id"], ttft_ms=ttft, success=ok, status="ok" if ok else "timeout"))
        return rows

    def summarize(self) -> dict[str, Any]:
        return {"name": self.name, "config": self.config}

    def export_manifest(self) -> dict[str, Any]:
        return {"adapter": self.name, "num_requests": self.config.get("num_requests", 30)}


class SharedContextReplay(_BaseReplay):
    def __init__(self) -> None:
        super().__init__("shared_context_replay")


class MultiTurnReplay(_BaseReplay):
    def __init__(self) -> None:
        super().__init__("multiturn_replay")


class BurstTraceReplay(_BaseReplay):
    def __init__(self) -> None:
        super().__init__("burst_trace_replay")
