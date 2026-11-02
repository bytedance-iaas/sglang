# 0003. `agent-verify` is GPU-free; GPU coverage stays in pr-test

- Status: Accepted
- Date: 2026-10-23
- Deciders: ci-owner, runtime-owner

## Context

Most SGLang tests need GPUs, model weights and the full CUDA dependency stack.
An agent on a laptop or a hosted runner cannot run them, yet every change needs a
check it can actually run and that CI runs identically.

## Decision

`make agent-verify` runs only what needs no GPU and no runtime install: governance
checks, pre-commit on changed files, and lint-script unit tests. Its toolchain is
hash-locked in `scripts/agent/requirements-agent.lock`. `make test` (CPU suite)
and the GPU stages in `pr-test.yml` remain the functional gates.

## Consequences

`agent-verify` green is necessary, not sufficient, for runtime changes; the PR
Change contract must state which functional checks ran. The Python runtime has no
lockfile; Docker images pin it (ledger L-004).
