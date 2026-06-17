#!/usr/bin/env python3
"""Compatibility shim; prefer scripts/data/prepare_babylm_text.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    try:
        runpy.run_path(str(ROOT / "scripts" / "data" / "prepare_babylm_text.py"), run_name="__main__")
    except KeyboardInterrupt:
        sys.exit(130)
