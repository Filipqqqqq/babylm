#!/usr/bin/env python3
"""Compare two Word2Vec EWoK context-sensitivity reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.analysis.word2vec.reports import (
    compare_domain_summaries,
    read_domain_summary,
)
from babysteps.paths import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="Left eval run directory or domain_summary.csv.")
    parser.add_argument("--right", required=True, help="Right eval run directory or domain_summary.csv.")
    parser.add_argument("--left-label", default="google_news", help="Label for the left run.")
    parser.add_argument("--right-label", default="babylm", help="Label for the right run.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for comparison CSV and plot.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Write CSV only.")
    return parser.parse_args()


def _summary_path(value: str) -> Path:
    path = resolve_path(value, base_dir=Path.cwd())
    if path.is_dir():
        path = path / "domain_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing domain summary CSV: {path}")
    return path


def main() -> None:
    args = parse_args()
    left_path = _summary_path(args.left)
    right_path = _summary_path(args.right)
    paths = compare_domain_summaries(
        read_domain_summary(left_path),
        read_domain_summary(right_path),
        left_label=args.left_label,
        right_label=args.right_label,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
    )
    print(f"Left: {left_path}")
    print(f"Right: {right_path}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
