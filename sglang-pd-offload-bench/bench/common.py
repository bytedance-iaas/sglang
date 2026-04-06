from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def new_run_dir(base: str = "results") -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = Path(base) / f"run_{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return path
