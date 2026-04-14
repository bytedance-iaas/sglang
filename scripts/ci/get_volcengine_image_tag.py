#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime


def get_sglang_version() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    content = (repo_root / "python/sglang/_version.py").read_text()
    match = re.search(r"__version__\s*=\s*version\s*=\s*'([^']+)'", content)
    if not match:
        raise SystemExit("failed to extract sglang version from python/sglang/_version.py")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Volcengine CR image tags for fork workflows.")
    parser.add_argument("--mode", choices=["manual", "nightly", "version"], required=True)
    parser.add_argument("--tag-value", default="", help="Required for version mode; inserted after .byted.")
    parser.add_argument("--cuda-suffix", choices=["", "cu130"], default="")
    args = parser.parse_args()

    version = get_sglang_version()
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d%H%M")

    if args.mode == "manual":
        tag = f"v{version}.iaas.dev.{timestamp}"
    elif args.mode == "nightly":
        tag = f"v{version}.iaas.nightly.{timestamp}"
    else:
        if not args.tag_value:
            raise SystemExit("--tag-value is required when --mode=version")
        tag = f"v{version}.byted.{args.tag_value}.{timestamp}"

    if args.cuda_suffix:
        tag = f"{tag}-{args.cuda_suffix}"

    print(tag)


if __name__ == "__main__":
    main()
