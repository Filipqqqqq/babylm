#!/usr/bin/env python3
"""Train a Word2Vec model from a configured BabyLM text stream."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.analysis.word2vec.training import load_word2vec_config, train_word2vec


DEFAULT_CONFIG = ROOT / "configs" / "analysis" / "word2vec" / "train_babylm2026_strict_small.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Word2Vec training YAML config.")
    parser.add_argument("--output-dir", help="Override output.dir from the config.")
    parser.add_argument("--max-docs", type=int, help="Optional smoke-test document cap.")
    parser.add_argument("--workers", type=int, help="Override word2vec.workers.")
    parser.add_argument("--epochs", type=int, help="Override word2vec.epochs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_word2vec_config(args.config)
    result = train_word2vec(
        config,
        root_dir=ROOT,
        output_dir=args.output_dir,
        max_docs=args.max_docs,
        workers=args.workers,
        epochs=args.epochs,
    )

    print(f"Output: {result.output_dir}")
    print(f"Model: {result.files['model']}")
    print(f"Vectors: {result.files['vectors']}")
    print(f"Vocab: {result.files['vocab']}")
    print(f"Manifest: {result.files['manifest']}")
    print(f"Vocab size: {result.manifest['vocab_size']:,}")
    print(f"Corpus words: {result.manifest['corpus_total_words']:,}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
