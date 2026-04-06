from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    run_dir = Path(args.input)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    md = [
        f"# Workload Report: {summary['replay']}",
        "",
        f"- Backend: `{summary['backend']}`",
        f"- Cache mode: `{summary['cache_mode']}`",
        f"- Validity: `{summary['validity']}`",
        "",
        "## Request Metrics",
        f"- TTFT p95 (ms): {summary['request_metrics']['ttft_p95_ms']:.2f}",
        f"- Success rate: {summary['request_metrics']['success_rate']:.3f}",
        "",
        "## Cache Tier Participation",
        f"- GPU hit ratio: {summary['cache_metrics']['gpu_hit_ratio']}",
        f"- Remote hit ratio: {summary['cache_metrics']['remote_hit_ratio']}",
        f"- Eviction count: {summary['cache_metrics']['eviction_count']}",
    ]
    (run_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"report_generated={run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
