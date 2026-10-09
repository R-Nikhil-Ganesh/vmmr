#!/usr/bin/env python3
"""
Machine-specific paths for the Python scripts.

paths.sh is the single source of truth. This module resolves the same variables by sourcing paths.sh in bash
(so paths.local.sh and environment overrides apply identically to shell runners and Python scripts).

    import paths
    paths.PM_IMG_DIR            # str
    python paths.py --check     # print every resolved path and whether it exists
"""

import os
import subprocess
import sys
from pathlib import Path

MOBILENET_DIR = Path(__file__).resolve().parent
KEYS = ["REPO_ROOT", "ECON_ROOT", "PM_IMG_DIR", "PM_MANIFEST_CSV", "PM_LABEL_MAP", "EXT_DATASETS_DIR", "PYTHON_BIN", "AUTO_GIT_PUSH"]


def _resolve() -> dict:
    sh = MOBILENET_DIR / "paths.sh"
    try:
        out = subprocess.run(["bash", "-c", '. "$1" && env -0', "_", str(sh)],
                             capture_output=True, check=True, timeout=30).stdout.decode()
        env = dict(item.split("=", 1) for item in out.split("\0") if "=" in item)
    except Exception as e:  # no bash (e.g. Windows): fall back to the process environment only
        print(f"[paths] could not source {sh} ({e}); using environment variables only", file=sys.stderr)
        env = dict(os.environ)
    return {k: env.get(k, "") for k in KEYS}


_VALUES = _resolve()
REPO_ROOT = _VALUES["REPO_ROOT"]
ECON_ROOT = _VALUES["ECON_ROOT"]
PM_IMG_DIR = _VALUES["PM_IMG_DIR"]
PM_MANIFEST_CSV = _VALUES["PM_MANIFEST_CSV"]
PM_LABEL_MAP = _VALUES["PM_LABEL_MAP"]
EXT_DATASETS_DIR = _VALUES["EXT_DATASETS_DIR"]
PYTHON_BIN = _VALUES["PYTHON_BIN"]
AUTO_GIT_PUSH = _VALUES["AUTO_GIT_PUSH"]


if __name__ == "__main__":
    ok = True
    for k in KEYS:
        v = _VALUES[k]
        if k == "AUTO_GIT_PUSH":
            print(f"  {k:18s} {v}")
            continue
        exists = bool(v) and Path(v).exists()
        ok &= exists
        print(f"  {'OK     ' if exists else 'MISSING'} {k:18s} {v}")
    local = MOBILENET_DIR / "paths.local.sh"
    print(f"\n  overrides file: {local} ({'present' if local.exists() else 'absent, using defaults'})")
    sys.exit(0 if ok else 1)
