# Engineering legacy ledger

Last reviewed: 2026-10-23
Owner: governance-owner (see [manifest.yaml](manifest.yaml))

Known debt and inherited constraints an agent or engineer must know before
changing related code. Add a row when you find one; close it with the PR that
removes it. `make agent-check` fails when this file is older than 90 days.

| ID | Item | Impact | Owner role | Status |
|---|---|---|---|---|
| L-001 | `python/sglang/srt/model_executor/model_runner.py` is a frozen core file | New logic must live elsewhere; see `large-class-style` skill | runtime-owner | Open (by design) |
| L-002 | Many upstream workflows run only when `github.repository == 'sgl-project/sglang'` | GPU test stages do not run on this fork; agent-verify and lint are the gates here | ci-owner | Open |
| L-003 | `.github/CODEOWNERS` lists upstream maintainers | Fork reviews are not auto-routed except for governance paths | governance-owner | Open |
| L-004 | No Python runtime lockfile; per-platform `pyproject_*.toml` use ranges | Installs drift over time; GPU runtime is pinned by Docker images | release-manager | Open |
| L-005 | `lint.yml` push trigger covers `main` only | Direct pushes to other long-lived branches skip lint | ci-owner | Open |
| L-006 | Fork `main` last synced with upstream on 2026-09-14 (433c999dd0) | Upstream fixes after that date are absent | repo-owner | Open |
| L-007 | Most CPU/GPU tests need the full runtime install | `make test` needs Linux; `make agent-verify` is GPU-free but shallow | ci-owner | Open |
