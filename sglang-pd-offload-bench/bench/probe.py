from __future__ import annotations

import argparse
import json

import pandas as pd

from bench.common import load_config, new_run_dir
from bench.runners.orchestrator import run_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    sweep = cfg.get("sweep", {"reuse_distance": [1, 4, 16]})
    base = cfg["base"]
    rows = []
    out = new_run_dir(base=cfg.get("results_dir", "results"))

    for rd in sweep.get("reuse_distance", [1, 4, 16]):
        local = dict(base)
        local["reuse_distance"] = rd
        summary = run_one(local, out / f"probe_rd_{rd}")
        rows.append({"reuse_distance": rd, "ttft_p95_ms": summary["request_metrics"]["ttft_p95_ms"], "validity": summary["validity"]})

    pd.DataFrame(rows).to_csv(out / "probe_summary.csv", index=False)
    (out / "probe_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"probe_dir={out}")


if __name__ == "__main__":
    main()
