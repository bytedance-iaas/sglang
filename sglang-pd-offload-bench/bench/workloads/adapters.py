"""Synthetic/static workload adapters (phase-2 extension points)."""

from dataclasses import dataclass
from typing import Any


@dataclass
class WorkloadAdapter:
    name: str

    def prepare(self, config: dict[str, Any]) -> None:
        self.config = config


SUPPORTED_WORKLOAD_ADAPTERS = [
    "generated_shared_prefix",
    "long_doc_qa_adapter",
    "prefix_repetition_adapter",
    "sharegpt_adapter",
    "burstgpt_adapter",
]
