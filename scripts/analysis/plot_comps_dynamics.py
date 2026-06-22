#!/usr/bin/env python3
"""Plot COMPS likelihood dynamics from a summary CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.analysis.comps_dynamics import plot_comps_summary  # noqa: E402
from babysteps.paths import resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to likelihood_summary.csv.")
    parser.add_argument(
        "--output-dir",
        help="Where plots are written. Defaults to the summary parent folder.",
    )
    parser.add_argument("--run-name", help="Optional run_name filter.")
    parser.add_argument(
        "--x-axis",
        choices=("step", "tokens_seen", "words_seen"),
        default="step",
        help="Summary column to use for the x-axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = resolve_path(args.summary, base_dir=ROOT)
    output_dir = resolve_path(args.output_dir, base_dir=ROOT) if args.output_dir else summary.parent
    plot_comps_summary(
        summary_csv=summary,
        output_dir=output_dir,
        run_name=args.run_name,
        x_axis=args.x_axis,
    )
    print(f"Wrote COMPS plots: {output_dir}")


if __name__ == "__main__":
    main()
