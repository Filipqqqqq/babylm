#!/usr/bin/env python3
"""Compatibility shim; prefer scripts/material_world/audit_material_world_corpus.py."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.cli.audit_material_world_corpus import *  # noqa: F403
from babysteps.cli.audit_material_world_corpus import main


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
