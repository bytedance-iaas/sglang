"""Persist Broker direct routing on an explicitly identified idle systemd runner."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

BROKER = "broker.actions.githubusercontent.com"
MARKER = "# SGLang Broker direct routing"


def listeners(service):
    found = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            if (process / "comm").read_text().strip() != "Runner.Listener":
                continue
            if service in (process / "cgroup").read_text():
                found.append(process)
        except OSError:
            continue
    return found


def worker_active():
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            if (process / "comm").read_text().strip() == "Runner.Worker":
                return True
        except OSError:
            continue
    return False


def environment(process):
    return dict(
        item.split("=", 1)
        for item in (process / "environ").read_bytes().decode().split("\0")
        if "=" in item
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--wait-idle", type=int, default=0)
    args = parser.parse_args()
    if socket.gethostname() != args.expected_host:
        raise SystemExit("Refusing a host identity mismatch")
    if not args.service.startswith("actions.runner.") or not args.service.endswith(".service"):
        raise SystemExit("Refusing an unexpected service")
    running = listeners(args.service)
    if len(running) != 1:
        raise SystemExit("Expected exactly one active listener for the target service")
    if args.wait_idle:
        deadline = time.monotonic() + args.wait_idle
        while worker_active() and time.monotonic() < deadline:
            time.sleep(1)
    if worker_active() and not (args.stage_only or args.verify_only):
        raise SystemExit("Runner.Worker is active; defer the restart until the host is idle")
    env = environment(running[0])
    exclusions = []
    for variable in ("NO_PROXY", "no_proxy"):
        for item in env.get(variable, "").split(","):
            item = item.strip()
            if item and item not in exclusions:
                exclusions.append(item)
    if BROKER not in exclusions:
        exclusions.append(BROKER)
    merged = ",".join(exclusions)
    if any(character in merged for character in "\r\n\0"):
        raise SystemExit("Invalid proxy exclusion list")
    escaped = merged.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    content = f'{MARKER}\n[Service]\nEnvironment="NO_PROXY={escaped}"\nEnvironment="no_proxy={escaped}"\n'
    # Sort after an existing override.conf, which can also set NO_PROXY.
    target = Path("/etc/systemd/system") / (args.service + ".d") / "zz-broker-direct.conf"
    if target.exists() and not target.read_text().startswith(MARKER):
        raise SystemExit("Refusing to replace an unrelated drop-in")
    result = {"host": args.expected_host, "service": args.service,
              "drop_in": str(target), "NO_PROXY": merged, "apply": args.apply}
    if args.verify_only:
        if not all(BROKER in env.get(key, "").split(",") for key in ("NO_PROXY", "no_proxy")):
            raise SystemExit("The active listener is missing Broker direct routing")
        if not target.exists() or not target.read_text().startswith(MARKER):
            raise SystemExit("The persistent Broker drop-in is missing")
        result.update(listener_pid=int(running[0].name), verified=True)
        print(json.dumps(result))
        return
    if not (args.apply or args.stage_only):
        print(json.dumps(result))
        return
    if os.geteuid() != 0:
        raise SystemExit("Root is required to update the service")
    if worker_active() and not args.stage_only:
        raise SystemExit("A worker became active; defer the restart")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".conf.new")
    temporary.write_text(content)
    temporary.chmod(0o644)
    temporary.replace(target)
    legacy = target.parent / "90-broker-direct.conf"
    if legacy.exists() and legacy.read_text().startswith(MARKER):
        legacy.unlink()
    if args.stage_only:
        result.update(staged=True)
        print(json.dumps(result))
        return
    subprocess.run(["systemctl", "daemon-reload"], check=True, timeout=15)
    subprocess.run(["systemctl", "restart", args.service], check=True, timeout=45)
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        current = listeners(args.service)
        if len(current) == 1:
            current_env = environment(current[0])
            if all(BROKER in current_env.get(key, "").split(",") for key in ("NO_PROXY", "no_proxy")):
                subprocess.run(["systemctl", "is-active", "--quiet", args.service], check=True)
                result.update(listener_pid=int(current[0].name), verified=True)
                print(json.dumps(result))
                return
        time.sleep(1)
    raise SystemExit("The restarted listener did not expose the expected exclusions")


if __name__ == "__main__":
    main()
