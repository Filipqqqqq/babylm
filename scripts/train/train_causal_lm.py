#!/usr/bin/env python3
"""Train a GPT-2 or LLaMA-style causal LM from token chunks or token shards."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.cli.train_causal_lm import main, release_cuda_memory


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        release_cuda_memory()
        sys.exit(130)
