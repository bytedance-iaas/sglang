#!/usr/bin/env bash
set -euo pipefail
python -m bench.run --config configs/pd_compare.yaml
