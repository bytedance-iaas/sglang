# Human review policy

Who must review what, and at which checkpoint. Role names resolve to GitHub
handles through `owners` in [manifest.yaml](manifest.yaml). Agents prepare
artifacts; humans in these roles approve them. Nothing here can be delegated to
an agent.

## Roles

| Role | Accountable for |
|---|---|
| `repo-owner` | Scope and priorities; final call on conflicts |
| `governance-owner` | AGENTS.md, governance/, CODEOWNERS, this policy |
| `ci-owner` | `.github/workflows/`, CI scripts, test infrastructure |
| `runtime-owner` | `python/sglang/srt`, `rust/`, `sgl-model-gateway`, frozen core files |
| `kernel-owner` | `python/sglang/kernels` and kernel build changes |
| `release-manager` | `python/pyproject*.toml`, `docker/`, releases and rollbacks |
| `security-owner` | Secrets handling, `.gitignore` coverage, dependency security |

## Checkpoints

| # | Checkpoint | When | Approver | Evidence |
|---|---|---|---|---|
| C1 | Intent / plan | Before implementing non-trivial or high-risk work | Owner role of the affected module | Plan in the PR description or issue |
| C2 | Pre-merge review | Every PR | CODEOWNERS for the paths; at least one human | PR approval |
| C3 | High-risk path | PR touches a manifest `high_risk_paths` entry | That entry's `approver` role | Approval from that role on the PR |
| C4 | Verification | Every PR | Reviewer at C2 | `agent-verify` check green; test evidence in the Change contract |
| C5 | Dependency / image change | `pyproject*.toml`, `docker/` | `release-manager` | Approval + image build log |
| C6 | Release / rollback | Tag, image publish, rollback | `release-manager` | Release notes, rollback plan |
| C7 | Governance change | AGENTS.md, governance/, CODEOWNERS, red lines | `governance-owner` | Approval on the PR |
