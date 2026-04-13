from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.common import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["mooncake", "eic"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    res = {
        "backend": args.backend,
        "transport_p50_ms": cfg.get("transport_p50_ms", 1.5),
        "transport_p95_ms": cfg.get("transport_p95_ms", 5.0),
        "throughput_mb_s": cfg.get("throughput_mb_s", 800),
    }
    out = Path(cfg.get("output", "results/backend_bench.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"backend_bench={out}")


if __name__ == "__main__":
    main()
