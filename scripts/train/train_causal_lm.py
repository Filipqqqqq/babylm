#!/usr/bin/env python3
"""Train a GPT-2 or LLaMA-style causal LM from token chunks or token shards."""

from __future__ import annotations

import sys
from pathlib import Path


# This repository uses a ``src`` layout: the installable ``babysteps`` Python
# package lives in ``<repository>/src/babysteps`` instead of beside this script.
# Add ``src`` to Python's import path so this thin launcher also works directly
# from a checkout, without first running ``pip install -e .``.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# This dotted import resolves to ``src/babysteps/cli/train_causal_lm.py``, where
# the command-line interface and the actual training orchestration are defined.
from babysteps.cli.train_causal_lm import main, release_cuda_memory


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        release_cuda_memory()
        sys.exit(130)
