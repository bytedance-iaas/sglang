from __future__ import annotations

import argparse
from pathlib import Path

from bench.runners.orchestrator import aggregate_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    args = parser.parse_args()
    dirs = [Path(x) for x in args.inputs]
    output = Path("results") / "aggregate.csv"
    aggregate_runs(dirs, output)
    print(f"aggregate_csv={output}")


if __name__ == "__main__":
    main()
