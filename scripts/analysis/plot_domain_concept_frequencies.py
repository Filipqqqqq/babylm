#!/usr/bin/env python3
"""Plot lexical target frequencies grouped by concept domain."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.analysis.lexical.domain_frequency import build_domain_frequency_report


DEFAULT_RUNS_DIR = Path(
    os.environ.get("BABYSTEPS_RUNS_DIR", "~/babylm-baby-steps-runs")
).expanduser()
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "analysis" / "lexical" / "domain_frequency"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-coverage",
        required=True,
        help="Path to target_coverage.csv from report_lexical_analysis.py.",
    )
    parser.add_argument(
        "--concepts",
        default=str(ROOT / "configs" / "analysis" / "targets" / "example_domain_concepts.yaml"),
        help="YAML file assigning concepts to domains.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for domain_concept_frequencies.csv and per-domain plots.",
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
        "--sort-by",
        default="frequency",
        choices=["frequency", "input"],
        help="Order concepts within each domain.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a log-scaled y axis when positive frequencies are present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_domain_frequency_report(
        target_coverage_csv=args.target_coverage,
        concepts_yaml=args.concepts,
        output_dir=args.output_dir,
        frequency_column=args.frequency_column,
        image_format=args.format,
        log_y=bool(args.log_y),
        sort_by=args.sort_by,
    )

    print(f"Concepts: {report['concept_count']}")
    print(f"Domains: {report['domain_count']}")
    print(f"CSV: {report['csv']}")
    for path in report["plots"]:
        print(f"Plot: {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
