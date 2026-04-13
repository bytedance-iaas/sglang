from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bench.backends.base import BackendAdapter


@dataclass
class _BaseMock(BackendAdapter):
    name: str
    started: bool = False

    def start(self, config: dict[str, Any]) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def healthcheck(self) -> bool:
        return self.started

    def collect_logs(self) -> dict[str, Any]:
        return {"backend": self.name, "started": self.started}

    def collect_metrics(self) -> dict[str, Any]:
        return {"backend": self.name}

    def describe_runtime(self) -> dict[str, Any]:
        return {"name": self.name, "kind": "mock"}


class GPUOnlyBackend(_BaseMock):
    def __init__(self) -> None:
        super().__init__(name="gpu_only")


class EICOffloadBackend(_BaseMock):
    def __init__(self) -> None:
        super().__init__(name="eic_offload")


class HiCacheMooncakeBackend(_BaseMock):
    def __init__(self) -> None:
        super().__init__(name="hicache_mooncake")
