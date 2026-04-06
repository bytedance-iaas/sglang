from __future__ import annotations

import argparse

from bench.common import load_config, new_run_dir
from bench.runners.orchestrator import run_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = new_run_dir(base=cfg.get("results_dir", "results"))
    summary = run_one(cfg, out)
    print(f"run_dir={out}")
    print(f"validity={summary['validity']}")


if __name__ == "__main__":
    main()
