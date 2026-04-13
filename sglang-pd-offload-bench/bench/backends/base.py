from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BackendAdapter(ABC):
    name: str

    @abstractmethod
    def start(self, config: dict[str, Any]) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    def restart(self, config: dict[str, Any]) -> None:
        self.stop()
        self.start(config)

    @abstractmethod
    def healthcheck(self) -> bool: ...

    @abstractmethod
    def collect_logs(self) -> dict[str, Any]: ...

    @abstractmethod
    def collect_metrics(self) -> dict[str, Any]: ...

    @abstractmethod
    def describe_runtime(self) -> dict[str, Any]: ...

    def warmup(self) -> None:
        return None

    def populate_remote_cache(self) -> None:
        return None

    def flush_gpu_cache(self) -> None:
        return None

    def flush_local_cache(self) -> None:
        return None

    def flush_remote_cache(self) -> None:
        return None
