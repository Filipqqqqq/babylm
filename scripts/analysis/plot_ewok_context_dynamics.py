#!/usr/bin/env python3
"""Plot EWoK context-sensitivity domain scores over training steps."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.analysis.ewok_context_dynamics import (  # noqa: E402
    build_panel_series,
    load_domain_rows,
    panel_item_counts,
    plot_domain_grid,
    plot_ewok_context_summary,
)


DEFAULT_RUNS_DIR = Path(os.environ.get("BABYSTEPS_RUNS_DIR", "~/babylm-baby-steps-runs")).expanduser()
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "analysis" / "ewok_context_dynamics"


def default_output_path(summary_csv: str | Path) -> Path:
    summary_csv = Path(summary_csv)
    parent_label = summary_csv.parent.name or "ewok_context"
    return DEFAULT_OUTPUT_DIR / f"{parent_label}_domain_dynamics.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to ewok_context_summary.csv.")
    parser.add_argument("--output", help="Output plot path.")
    parser.add_argument("--run-name", help="Filter to one run_name in the summary CSV.")
    parser.add_argument(
        "--score-reduction",
        choices=("sum", "mean", "both"),
        help="Filter to one score reduction. Omit or pass both to plot all rows.",
    )
    parser.add_argument(
        "--title",
        default="EWoK Context Sensitivity by Domain",
        help="Plot title.",
    )
    parser.add_argument(
        "--x-axis",
        choices=("step", "tokens_seen", "words_seen"),
        default="step",
        help="Summary column to use for the x-axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else default_output_path(args.summary)
    score_reduction = None if args.score_reduction in {None, "both"} else args.score_reduction
    rows = load_domain_rows(
        args.summary,
        run_name=args.run_name,
        score_reduction=score_reduction,
        x_axis=args.x_axis,
    )
    plot_domain_grid(
        panel_series=build_panel_series(rows),
        output_path=output_path,
        panel_counts=panel_item_counts(rows),
        title=args.title,
        x_axis=args.x_axis,
    )

    print(f"Loaded {len(rows):,} rows from {args.summary}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)


__all__ = [
    "build_panel_series",
    "load_domain_rows",
    "panel_item_counts",
    "plot_domain_grid",
    "plot_ewok_context_summary",
]
