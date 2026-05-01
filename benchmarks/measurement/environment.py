from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def collect_environment(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config_bytes = path.read_bytes()
    return {
        "timestamp_unix": time.time(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
        "git_branch": _git_value(["git", "branch", "--show-current"]),
        "git_dirty": bool(_git_value(["git", "status", "--porcelain"])),
        "config_path": str(path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True).strip()
    except Exception:
        return ""
