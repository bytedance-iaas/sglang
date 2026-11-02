# 0001. Record architecture decisions

- Status: Accepted
- Date: 2026-10-23
- Deciders: governance-owner

## Context

Decisions about this repository (fork strategy, tooling, gates) live in chat
threads and people's heads. Agents and new engineers cannot recover the reasons.

## Decision

Record significant decisions as ADRs in `governance/adr/NNNN-title.md`, using
this template: Status, Date, Deciders, Context, Decision, Consequences. An ADR is
never edited after acceptance except to change its Status to Superseded by NNNN.

## Consequences

`make agent-check` requires at least one ADR. Reviewers ask for an ADR when a PR
changes a gate, a toolchain, a module boundary or the branch strategy.
