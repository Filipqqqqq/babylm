#!/usr/bin/env python3
"""Deprecated compatibility shim; use scripts/data/build_token_shards.py."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.cli.build_token_shards import *  # noqa: F403
from babysteps.cli.build_token_shards import main


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
