#!/usr/bin/env python3
"""Train a causal LM from token-chunk pickle files."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_gpt2_small import main, release_cuda_memory


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        release_cuda_memory()
        sys.exit(130)
