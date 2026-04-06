from __future__ import annotations

import pandas as pd


def detect_breakpoints(df: pd.DataFrame, x_col: str = "reuse_distance", y_col: str = "ttft_p95_ms") -> pd.DataFrame:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame(columns=["metric", "breakpoint", "note"])
    ranked = df.sort_values(x_col)
    delta = ranked[y_col].diff().fillna(0)
    threshold = delta.abs().quantile(0.9)
    points = ranked.loc[delta.abs() >= threshold, [x_col, y_col]].copy()
    points["metric"] = y_col
    points["breakpoint"] = points[x_col]
    points["note"] = "high slope change"
    return points[["metric", "breakpoint", "note"]]
