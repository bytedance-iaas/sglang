# Repository Structure Design

## Top-level layout
- README.md
- pyproject.toml
- configs/
- backends/
- workloads/
- replay/
- runners/
- collectors/
- analysis/
- reports/
- scripts/
- docs/
- results/
- tests/
- bench/ (Python package implementation)

## Core interfaces
- `bench.backends.base.BackendAdapter`
- `bench.replay.base.ReplayAdapter`
- `bench.types.{RequestRecord, CacheMetrics, BackendMetrics, RunArtifacts}`

## MVP implementation plan
1. Load config and run one backend+replay pair.
2. Produce request-level CSV and summary JSON.
3. Compute validity labels from cache metrics.
4. Add probe/analyze/report/aggregate/backend_bench CLI.
5. Add breakpoint and SLO extension points.
