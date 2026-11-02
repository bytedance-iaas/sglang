# Standard commands for humans, agents and CI. Declared in governance/manifest.yaml
# and AGENTS.md; CI (.github/workflows/agent-verify.yml) calls the same targets.

PYTHON ?= python3
AGENT_BASE_REF ?= origin/main

.PHONY: help bootstrap bootstrap-agent lint test verify agent-check agent-verify agent-record

help:
	@echo "bootstrap        Full dev install (Linux + CUDA; python[dev] editable)"
	@echo "bootstrap-agent  Install the locked agent toolchain (pre-commit, PyYAML); works on any OS"
	@echo "lint             pre-commit on all files (same as lint.yml)"
	@echo "test             CPU unit suite base-a-test-cpu (needs bootstrap)"
	@echo "verify           Alias of agent-verify"
	@echo "agent-check      Fast: governance checks + pre-commit on files changed vs AGENT_BASE_REF"
	@echo "agent-verify     agent-check + lint-script unit tests (the CI gate)"
	@echo "agent-record     Append an entry to governance/AGENT_CHANGELOG.md (MSG=... RISK=low|medium|high)"

bootstrap: bootstrap-agent
	$(PYTHON) -m pip install uv
	uv pip install --system -e "python[dev]" --index-strategy unsafe-best-match --prerelease allow

bootstrap-agent:
	scripts/agent/bootstrap-agent.sh

lint:
	pre-commit run --all-files

test:
	cd test && $(PYTHON) run_suite.py --hw cpu --suite base-a-test-cpu

verify: agent-verify

agent-check:
	AGENT_BASE_REF=$(AGENT_BASE_REF) scripts/agent/agent-check.sh

agent-verify:
	AGENT_BASE_REF=$(AGENT_BASE_REF) scripts/agent/agent-verify.sh

agent-record:
	scripts/agent/record-change.sh "$(MSG)" "$(RISK)"
