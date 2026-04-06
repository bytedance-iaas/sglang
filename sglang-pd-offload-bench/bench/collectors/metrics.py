from __future__ import annotations

from typing import Iterable

import pandas as pd

from bench.types import RequestRecord


def request_dataframe(records: Iterable[RequestRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in records])


def summarize_request_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"ttft_p50_ms": 0.0, "ttft_p95_ms": 0.0, "ttft_p99_ms": 0.0, "success_rate": 0.0, "throughput_rps": 0.0}
    return {
        "ttft_p50_ms": float(df["ttft_ms"].quantile(0.50)),
        "ttft_p95_ms": float(df["ttft_ms"].quantile(0.95)),
        "ttft_p99_ms": float(df["ttft_ms"].quantile(0.99)),
        "success_rate": float(df["success"].mean()),
        "throughput_rps": float(len(df) / max(df["ttft_ms"].sum() / 1000.0, 1e-6)),
    }
