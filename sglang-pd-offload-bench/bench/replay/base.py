from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bench.types import RequestRecord


class ReplayAdapter(ABC):
    name: str

    @abstractmethod
    def prepare(self, config: dict[str, Any]) -> None: ...

    @abstractmethod
    def generate_requests(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def replay(self, backend_name: str) -> list[RequestRecord]: ...

    @abstractmethod
    def summarize(self) -> dict[str, Any]: ...

    @abstractmethod
    def export_manifest(self) -> dict[str, Any]: ...
