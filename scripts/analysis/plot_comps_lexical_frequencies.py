#!/usr/bin/env python3
"""Plot COMPS lexical target frequencies from target_coverage.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.analysis.lexical.comps_frequency import build_comps_frequency_report
from babysteps.paths import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-coverage",
        required=True,
        help="Path to target_coverage.csv from a COMPS lexical report.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Where plots are written. Defaults to a comps_frequency_plots "
            "directory next to target_coverage.csv."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=30,
        help="Number of targets in each top/middle/bottom slice.",
    )
    parser.add_argument(
        "--frequency-column",
        default="total_mentions",
        help="Column in target_coverage.csv to plot.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Plot image format.",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a log-scaled x axis when positive frequencies are present.",
    )
    parser.add_argument(
        "--title-prefix",
        default="COMPS lexical",
        help="Prefix used in plot titles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_coverage = resolve_path(args.target_coverage, base_dir=ROOT)
    output_dir = (
        resolve_path(args.output_dir, base_dir=ROOT)
        if args.output_dir
        else target_coverage.parent / "comps_frequency_plots"
    )
    report = build_comps_frequency_report(
        target_coverage_csv=target_coverage,
        output_dir=output_dir,
        frequency_column=args.frequency_column,
        n=args.n,
        image_format=args.format,
        log_x=bool(args.log_x),
        title_prefix=args.title_prefix,
    )

    print(f"Targets: {report['target_count']}")
    print(f"N: {report['n']}")
    print(f"Frequency column: {report['frequency_column']}")
    print(f"CSV: {report['csv']}")
    for path in report["plots"]:
        print(f"Plot: {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
