from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bench.analysis.breakpoints import detect_breakpoints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    run_dir = Path(args.input)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    row = {
        "reuse_distance": summary.get("manifest", {}).get("num_requests", 0),
        "ttft_p95_ms": summary["request_metrics"]["ttft_p95_ms"],
    }
    bp = detect_breakpoints(pd.DataFrame([row]))
    bp.to_csv(run_dir / "breakpoints.csv", index=False)
    print(f"analysis_done={run_dir}")


if __name__ == "__main__":
    main()
