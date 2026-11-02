#!/usr/bin/env python3
"""Validate the agent-governance artifacts declared in governance/manifest.yaml.

Each check returns a list of error strings; the script fails if any check
reports one. Run via `make agent-check` / `make agent-verify`.
"""

import argparse
import datetime
import fnmatch
import pathlib
import re
import subprocess
import sys

import yaml

MANIFEST_PATH = "governance/manifest.yaml"
AGENTS_MD_MAX_LINES = 200
HANDLE_RE = re.compile(
    r"^@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:/[A-Za-z0-9._-]+)?$"
)
LAST_REVIEWED_RE = re.compile(
    r"^Last reviewed:\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE
)
MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
MAKE_CMD_RE = re.compile(r"`make ([a-z][a-z0-9-]*)")
MAKE_TARGET_RE = re.compile(r"^([A-Za-z0-9_-]+):", re.MULTILINE)
RECIPE_SCRIPT_RE = re.compile(r"(scripts/[A-Za-z0-9_./-]+\.(?:sh|py))")
CHANGELOG_ENTRY_RE = re.compile(
    r"^- \d{4}-\d{2}-\d{2} \| [^|]+ \| risk=(?:low|medium|high) \| ", re.MULTILINE
)

AGENTS_MD_REQUIRED_HEADINGS = (
    "Standard commands",
    "Operating boundaries",
    "Red lines",
    "Escalate to a human",
    "Read before changing",
    "Task navigation",
    "CI gates",
    "Delivery",
)
PR_TEMPLATE_REQUIRED_HEADINGS = (
    "Change contract",
    "Intent",
    "Risk",
    "Verification",
    "Rollback",
)
GITIGNORE_REQUIRED_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    ".netrc",
    "credentials.json",
    "*.kubeconfig",
)
# Phrases by which an agent document would grant itself authority it does not have.
SELF_GRANT_RE = re.compile(
    r"(you|agents?|claude)\s+(may|can|are allowed to|are permitted to)\s+"
    r"(skip|bypass|disable|ignore|merge|approve|force[- ]push|push (directly )?to (main|master))",
    re.IGNORECASE,
)
# High-confidence secret shapes. Written so this file never matches itself.
SECRET_PATTERNS = (
    r"AKIA[0-9A-Z]{16}",
    r"AKLT[A-Za-z0-9]{30,}",
    r"gh[pousr]_[A-Za-z0-9]{36}",
    r"github_pat_[A-Za-z0-9_]{60,}",
    r"sk-ant-[A-Za-z0-9_-]{40,}",
    r"xox[baprs]-[A-Za-z0-9-]{20,}",
    r"-{5}BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-{5}",
)
SECRET_SCAN_ALLOWLIST = "governance/secret-scan-allowlist.txt"


def _read(root: pathlib.Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8")


def _tracked_files(root: pathlib.Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    return out.splitlines()


def _make_targets(root: pathlib.Path) -> dict[str, str]:
    """Map each Makefile target to its recipe text."""
    text = _read(root, "Makefile")
    targets = {}
    matches = list(MAKE_TARGET_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        targets[m.group(1)] = text[m.start() : end]
    return targets


def check_entrypoints(*, root: pathlib.Path, manifest: dict) -> list[str]:
    errors = []
    for name, rel in manifest["entrypoints"].items():
        if not (root / rel).exists():
            errors.append(f"entrypoint '{name}' missing: {rel}")
    adr_dir = root / manifest["entrypoints"]["adr_dir"]
    if adr_dir.is_dir() and not list(adr_dir.glob("[0-9][0-9][0-9][0-9]-*.md")):
        errors.append(f"no ADR (NNNN-title.md) in {adr_dir.relative_to(root)}")
    return errors


def check_owners(*, manifest: dict) -> list[str]:
    errors = []
    owners = manifest.get("owners") or {}
    if "repo-owner" not in owners:
        errors.append("owners: 'repo-owner' role is required")
    for role, handles in owners.items():
        if not handles:
            errors.append(f"owners: role '{role}' has no handle")
        for handle in handles or []:
            if not HANDLE_RE.match(handle):
                errors.append(f"owners: role '{role}' has malformed handle '{handle}'")
    for module in manifest.get("modules", []):
        if module.get("owner") not in owners:
            errors.append(
                f"module {module['path']}: owner role '{module.get('owner')}' unresolved"
            )
    for item in manifest.get("high_risk_paths", []):
        if item.get("approver") not in owners:
            errors.append(
                f"high-risk path {item['path']}: approver role '{item.get('approver')}' unresolved"
            )
    return errors


def check_commands(*, root: pathlib.Path, manifest: dict) -> list[str]:
    errors = []
    targets = _make_targets(root)
    for name, command in manifest["commands"].items():
        parts = command.split()
        if parts[:1] != ["make"] or len(parts) != 2:
            errors.append(f"command '{name}' must be 'make <target>', got '{command}'")
            continue
        recipe = targets.get(parts[1])
        if recipe is None:
            errors.append(f"command '{name}': Makefile has no target '{parts[1]}'")
            continue
        for script in RECIPE_SCRIPT_RE.findall(recipe):
            path = root / script
            if not path.is_file():
                errors.append(f"command '{name}': script {script} not found")
            elif script.endswith(".sh") and not path.stat().st_mode & 0o111:
                errors.append(f"command '{name}': script {script} is not executable")
    return errors


def check_modules(*, root: pathlib.Path, manifest: dict) -> list[str]:
    errors = []
    for module in manifest.get("modules", []):
        if not (root / module["path"]).exists():
            errors.append(f"module path missing: {module['path']}")
        for field in ("role", "boundary"):
            if not str(module.get(field, "")).strip():
                errors.append(f"module {module['path']}: '{field}' is empty")
    return errors


def check_path_checks(
    *, root: pathlib.Path, manifest: dict, tracked: list[str]
) -> list[str]:
    errors = []
    targets = _make_targets(root)
    for rule in manifest.get("path_checks", []):
        for pattern in rule["paths"]:
            if not any(fnmatch.fnmatch(f, pattern) for f in tracked):
                errors.append(
                    f"path_checks: pattern '{pattern}' matches no tracked file"
                )
        if not rule.get("checks"):
            errors.append(f"path_checks: {rule['paths']} has no checks")
        for check in rule.get("checks", []):
            if check.startswith("ci:"):
                workflow = check[3:].split()[0]
                if not (root / ".github/workflows" / workflow).is_file():
                    errors.append(f"path_checks: workflow '{workflow}' not found")
            elif check not in targets:
                errors.append(f"path_checks: check '{check}' is not a Makefile target")
    for item in manifest.get("high_risk_paths", []):
        if not (root / item["path"]).exists():
            errors.append(f"high-risk path missing: {item['path']}")
    return errors


def _local_link_errors(*, root: pathlib.Path, doc: str, text: str) -> list[str]:
    errors = []
    for target in MD_LINK_RE.findall(text):
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue
        rel = target.split("#", 1)[0]
        if not (root / pathlib.Path(doc).parent / rel).exists():
            errors.append(f"{doc}: broken local link '{target}'")
    return errors


def check_agents_md(*, root: pathlib.Path, manifest: dict) -> list[str]:
    doc = manifest["entrypoints"]["agent"]
    text = _read(root, doc)
    errors = _local_link_errors(root=root, doc=doc, text=text)
    if len(text.splitlines()) > AGENTS_MD_MAX_LINES:
        errors.append(
            f"{doc}: longer than {AGENTS_MD_MAX_LINES} lines; keep it a short entry"
        )
    for heading in AGENTS_MD_REQUIRED_HEADINGS:
        if not re.search(rf"^#+ .*{re.escape(heading)}", text, re.MULTILINE):
            errors.append(f"{doc}: missing section '{heading}'")
    targets = _make_targets(root)
    for target in MAKE_CMD_RE.findall(text):
        if target not in targets:
            errors.append(f"{doc}: `make {target}` is not a Makefile target")
    for name, command in manifest["commands"].items():
        if f"`{command}`" not in text:
            errors.append(
                f"{doc}: standard command '{name}' (`{command}`) not declared"
            )
    if manifest["entrypoints"]["ci_gate"] not in text:
        errors.append(
            f"{doc}: does not reference CI gate {manifest['entrypoints']['ci_gate']}"
        )
    for line_no, line in enumerate(text.splitlines(), 1):
        if SELF_GRANT_RE.search(line):
            errors.append(f"{doc}:{line_no}: self-granted permission: {line.strip()}")
    return errors


def check_freshness(
    *, root: pathlib.Path, manifest: dict, today: datetime.date
) -> list[str]:
    errors = []
    for rel, max_days in manifest.get("freshness_days", {}).items():
        path = root / rel
        if not path.is_file():
            errors.append(f"{rel}: missing")
            continue
        m = LAST_REVIEWED_RE.search(path.read_text(encoding="utf-8"))
        if not m:
            errors.append(f"{rel}: no 'Last reviewed: YYYY-MM-DD' line")
            continue
        reviewed = datetime.date.fromisoformat(m.group(1))
        if reviewed > today:
            errors.append(f"{rel}: 'Last reviewed' {reviewed} is in the future")
        elif (today - reviewed).days > max_days:
            errors.append(
                f"{rel}: stale, last reviewed {reviewed} (limit {max_days} days)"
            )
    return errors


def check_governance_docs(*, root: pathlib.Path, manifest: dict) -> list[str]:
    errors = []
    entry = manifest["entrypoints"]
    pr_template = _read(root, entry["pr_template"])
    for heading in PR_TEMPLATE_REQUIRED_HEADINGS:
        if not re.search(rf"^#+ .*{heading}", pr_template, re.MULTILINE):
            errors.append(f"{entry['pr_template']}: missing section '{heading}'")
    if "make agent-verify" not in _read(root, entry["ci_gate"]):
        errors.append(f"{entry['ci_gate']}: does not run `make agent-verify`")
    if not CHANGELOG_ENTRY_RE.search(_read(root, entry["agent_change_log"])):
        errors.append(f"{entry['agent_change_log']}: no entry in the documented format")
    if not re.search(r"^\| L-\d+ ", _read(root, entry["legacy_ledger"]), re.MULTILINE):
        errors.append(f"{entry['legacy_ledger']}: no ledger entry (| L-NNN | ...)")
    policy = _read(root, entry["policy"])
    for role in re.findall(r"`([a-z]+-(?:owner|manager))`", policy):
        if role not in manifest["owners"]:
            errors.append(
                f"{entry['policy']}: role '{role}' not defined in manifest owners"
            )
    if "## Checkpoints" not in policy:
        errors.append(f"{entry['policy']}: missing '## Checkpoints' section")
    return errors


def check_gitignore(*, root: pathlib.Path) -> list[str]:
    lines = {line.strip() for line in _read(root, ".gitignore").splitlines()}
    return [
        f".gitignore: missing secret/env pattern '{p}'"
        for p in GITIGNORE_REQUIRED_PATTERNS
        if p not in lines
    ]


def check_secrets(*, root: pathlib.Path) -> list[str]:
    allow_path = root / SECRET_SCAN_ALLOWLIST
    allow = []
    if allow_path.is_file():
        allow = [
            line.split("#", 1)[0].strip()
            for line in allow_path.read_text(encoding="utf-8").splitlines()
            if line.split("#", 1)[0].strip()
        ]
    cmd = ["git", "grep", "-I", "-n", "-E"]
    for pattern in SECRET_PATTERNS:
        cmd += ["-e", pattern]
    out = subprocess.run(cmd, cwd=root, capture_output=True, text=True).stdout
    errors = []
    for hit in out.splitlines():
        path = hit.split(":", 1)[0]
        if not any(fnmatch.fnmatch(path, glob) for glob in allow):
            errors.append(f"possible plaintext secret: {hit[:160]}")
    return errors


def run_checks(*, root: pathlib.Path, today: datetime.date) -> list[str]:
    manifest = yaml.safe_load(_read(root, MANIFEST_PATH))
    tracked = _tracked_files(root)
    errors = check_entrypoints(root=root, manifest=manifest)
    if errors:
        return errors
    errors += check_owners(manifest=manifest)
    errors += check_commands(root=root, manifest=manifest)
    errors += check_modules(root=root, manifest=manifest)
    errors += check_path_checks(root=root, manifest=manifest, tracked=tracked)
    errors += check_agents_md(root=root, manifest=manifest)
    errors += check_freshness(root=root, manifest=manifest, today=today)
    errors += check_governance_docs(root=root, manifest=manifest)
    errors += check_gitignore(root=root)
    errors += check_secrets(root=root)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument(
        "--today", type=datetime.date.fromisoformat, default=datetime.date.today()
    )
    args = parser.parse_args()
    errors = run_checks(root=args.root.resolve(), today=args.today)
    for error in errors:
        print(f"[agent-governance] {error}")
    if errors:
        print(
            f"[agent-governance] {len(errors)} problem(s); see governance/manifest.yaml"
        )
        return 1
    print("[agent-governance] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
