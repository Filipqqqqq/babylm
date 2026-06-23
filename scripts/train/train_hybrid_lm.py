#!/usr/bin/env python3
"""Train GPT-BERT with a hybrid causal + masked-next-token objective."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.cli.train_hybrid_lm import main


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
