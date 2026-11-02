# 0002. One agent entry point and Makefile-bound standard commands

- Status: Accepted
- Date: 2026-10-23
- Deciders: governance-owner, ci-owner

## Context

Agents need a short, reliable entry: which commands to run, what they may not do,
and what to read for a given path. Instructions spread over README, docs and
skills drift, and commands described in prose go stale.

## Decision

- `AGENTS.md` is the agent entry (CLAUDE.md imports it). It routes; it does not
  copy product specs. It stays under 200 lines.
- `governance/manifest.yaml` is the machine-readable map: owners, modules,
  path-to-check mapping, high-risk paths, standard commands.
- Standard commands are root `Makefile` targets. CI calls the same targets.
- `scripts/lint/check_agent_governance.py` validates the manifest, AGENTS.md links
  and commands, freshness of dated docs, `.gitignore` coverage and secret shapes.

## Consequences

Renaming a target or moving a linked file breaks `make agent-check` until AGENTS.md
and the manifest are updated, which is the intent.
