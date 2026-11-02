# Agent key-change log

Records medium- and high-risk changes made with agent assistance, so reviewers
and later agents can trace what changed, when, by whom, and at what risk.
Git history and PRs remain the source of truth; this is the index.

## Convention

- Add an entry for any change with `RISK=medium` or `RISK=high`, any change to a
  manifest `high_risk_paths` entry, and any governance change.
- Use the entry point, do not hand-edit lines: `make agent-record MSG="<summary, PR link>" RISK=medium`
- Format (one line, newest last):
  `- YYYY-MM-DD | <author> | risk=<low|medium|high> | branch=<branch> | base=<short sha> | <summary>`

## Entries

- 2026-10-23 | chenqianfzh | risk=medium | branch=dev/ai-native-sdlc | base=433c999dd0 | Add AI-native SDLC governance: AGENTS.md, manifest, standard commands, agent-verify CI gate
