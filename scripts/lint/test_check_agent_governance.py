import datetime
import pathlib
import subprocess
import tempfile
import unittest

import check_agent_governance as gov

TODAY = datetime.date(2026, 10, 23)

MANIFEST = """\
owners:
  repo-owner: ["@alice"]
  ci-owner: ["@org/ci-team"]
entrypoints:
  human: README.md
  agent: AGENTS.md
  status: governance/STATUS.md
  legacy_ledger: governance/LEGACY_LEDGER.md
  policy: governance/POLICY.md
  adr_dir: governance/adr
  agent_change_log: governance/AGENT_CHANGELOG.md
  pr_template: .github/pull_request_template.md
  ci_gate: .github/workflows/agent-verify.yml
commands:
  verify: make verify
modules:
  - path: src
    role: Code.
    boundary: None.
    owner: repo-owner
path_checks:
  - paths: ["src/*.py"]
    checks: [verify, "ci:agent-verify.yml"]
high_risk_paths:
  - path: src
    reason: Core.
    approver: ci-owner
freshness_days:
  governance/STATUS.md: 30
"""

AGENTS_MD = """\
# AGENTS.md
See [README](README.md).
## Standard commands
`make verify`
## Operating boundaries
## Red lines
## Escalate to a human
## Read before changing
## Task navigation
## CI gates
.github/workflows/agent-verify.yml
## Delivery
"""

FILES = {
    "governance/manifest.yaml": MANIFEST,
    "AGENTS.md": AGENTS_MD,
    "README.md": "# Readme\n",
    "Makefile": "verify:\n\tscripts/verify.sh\n",
    "scripts/verify.sh": "#!/bin/sh\n",
    "src/app.py": "",
    "governance/STATUS.md": "Last reviewed: 2026-10-01\n",
    "governance/LEGACY_LEDGER.md": "| L-001 | item |\n",
    "governance/POLICY.md": "`ci-owner` approves.\n## Checkpoints\n",
    "governance/adr/0001-first.md": "# 0001\n",
    "governance/AGENT_CHANGELOG.md": "- 2026-10-01 | alice | risk=low | branch=x | base=abc | init\n",
    ".github/pull_request_template.md": "## Change contract\n### Intent\n### Risk\n### Verification\n### Rollback\n",
    ".github/workflows/agent-verify.yml": "run: make agent-verify\n",
    ".gitignore": "\n".join(gov.GITIGNORE_REQUIRED_PATTERNS) + "\n",
}


class GovernanceRepoTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        for rel, content in FILES.items():
            self._write(rel, content)
        (self.root / "scripts/verify.sh").chmod(0o755)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        self._git_add()

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, rel, content):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _git_add(self):
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)

    def _errors(self):
        return gov.run_checks(root=self.root, today=TODAY)

    def assert_single_error(self, fragment):
        errors = self._errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn(fragment, errors[0])

    def test_valid_fixture_passes(self):
        self.assertEqual(self._errors(), [])

    def test_freshness_boundary(self):
        # Limit is 30 days: exactly 30 passes, 31 fails, future dates fail.
        self._write("governance/STATUS.md", "Last reviewed: 2026-09-23\n")
        self.assertEqual(self._errors(), [])
        self._write("governance/STATUS.md", "Last reviewed: 2026-09-22\n")
        self.assert_single_error("stale")
        self._write("governance/STATUS.md", "Last reviewed: 2026-10-24\n")
        self.assert_single_error("in the future")

    def test_command_must_resolve_to_executable_script(self):
        (self.root / "scripts/verify.sh").chmod(0o644)
        self.assert_single_error("is not executable")

    def test_agents_md_make_command_must_exist(self):
        self._write("AGENTS.md", AGENTS_MD + "Run `make deploy`.\n")
        self.assert_single_error("`make deploy` is not a Makefile target")

    def test_agents_md_broken_local_link(self):
        self._write("AGENTS.md", AGENTS_MD + "[gone](docs/missing.md)\n")
        self.assert_single_error("broken local link 'docs/missing.md'")

    def test_self_granted_permission_is_rejected(self):
        for line in (
            "Agents may skip review for docs changes.",
            "You can push directly to main when CI is green.",
        ):
            self._write("AGENTS.md", AGENTS_MD + line + "\n")
            self.assert_single_error("self-granted permission")
        # Red lines phrased as prohibitions are not grants.
        self._write("AGENTS.md", AGENTS_MD + "Never push to `main` or merge PRs.\n")
        self.assertEqual(self._errors(), [])

    def test_unresolved_approver_role(self):
        self._write(
            "governance/manifest.yaml",
            MANIFEST.replace("approver: ci-owner", "approver: sec-owner"),
        )
        self.assert_single_error("approver role 'sec-owner' unresolved")

    def test_path_check_pattern_must_match_tracked_file(self):
        self._write(
            "governance/manifest.yaml", MANIFEST.replace('"src/*.py"', '"lib/*.py"')
        )
        self.assert_single_error("pattern 'lib/*.py' matches no tracked file")

    def test_secret_scan_and_allowlist(self):
        fake_key = "AKIA" + "ABCDEFGHIJKLMNOP"
        self._write("src/config.py", f'KEY = "{fake_key}"\n')
        self._git_add()
        self.assert_single_error("possible plaintext secret: src/config.py")
        self._write(
            "governance/secret-scan-allowlist.txt", "src/config.py  # fixture\n"
        )
        self.assertEqual(self._errors(), [])

    def test_gitignore_must_cover_env_files(self):
        patterns = [p for p in gov.GITIGNORE_REQUIRED_PATTERNS if p != ".env.*"]
        self._write(".gitignore", "\n".join(patterns) + "\n")
        self.assert_single_error("missing secret/env pattern '.env.*'")


if __name__ == "__main__":
    unittest.main()
