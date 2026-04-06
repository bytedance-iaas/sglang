#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Offline-friendly: avoid network package installs in restricted environments.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python -m bench.run --config configs/pd_compare.yaml
latest_run=$(ls -dt results/run_* | head -n1)
python -m bench.report --input "$latest_run"

echo "Done. Latest run: $latest_run"
echo "Summary: $latest_run/summary.json"
echo "Report:  $latest_run/report.md"
