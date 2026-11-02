# Agent onboarding status

Last reviewed: 2026-10-23
Reviewed at commit: 433c999dd0 (main)
Owner: governance-owner (see [manifest.yaml](manifest.yaml))

Snapshot of how far this repository meets the AI-native SDLC checklist, and what
is left. `make agent-check` fails when this file is older than 30 days.

## In place

- Entry points: README.md (human), AGENTS.md (agent), CLAUDE.md (imports AGENTS.md).
- Manifest with module responsibilities/boundaries, path-to-check mapping,
  high-risk paths with approver roles, owners: [manifest.yaml](manifest.yaml).
- Standard commands bound in the root Makefile: bootstrap, bootstrap-agent, lint,
  test, verify, agent-check, agent-verify, agent-record.
- Hash-locked agent toolchain (scripts/agent/requirements-agent.lock); Python pinned
  in .tool-versions; Rust pinned by rust/rust-toolchain.toml and Cargo.lock.
- CI gate .github/workflows/agent-verify.yml runs the same `make agent-verify`.
- PR change-contract template; .gitignore secret/env patterns; secret scan.
- Directory tree, legacy ledger, human review policy, ADRs, agent change log.

## To do

| Item | Owner role | Notes |
|---|---|---|
| Confirm real GitHub handles for each owner role | governance-owner | All roles currently point at the governance owner |
| Requirement / Delivery / Evidence / Policy objects and object-graph validation | governance-owner | Waiting for the platform's object schema |
| Python runtime lockfile | release-manager | Multi-platform CUDA/ROCm/NPU deps; images pin runtime today (L-004) |
| GPU-free stand-in for dependent services beyond the CPU suite | runtime-owner | docker/compose.yaml needs GPU + RDMA |
| Make agent-verify a required status check on `main` | repo-owner | Branch protection setting, not in git |
