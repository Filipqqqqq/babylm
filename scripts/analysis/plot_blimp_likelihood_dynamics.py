#!/usr/bin/env python3
"""Plot BLiMP-style likelihood dynamics from summary CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.analysis.blimp_likelihood_dynamics import (  # noqa: E402
    plot_blimp_likelihood_summary,
    plot_overall_accuracy,
)
from babysteps.paths import resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Path to likelihood_summary.csv. Can be passed more than once.",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Plot label for each --summary. Defaults to the summary parent folder name.",
    )
    parser.add_argument(
        "--output-dir",
        help="Where plots are written. Defaults to each summary parent for per-summary plots.",
    )
    parser.add_argument("--run-name", help="Optional run_name filter.")
    parser.add_argument(
        "--score-reduction",
        choices=("sum", "mean", "both"),
        help="Filter to one score reduction. Omit or pass both to plot all rows.",
    )
    parser.add_argument(
        "--group-scope",
        action="append",
        default=None,
        help="Metadata grouping to plot for each summary, e.g. field, linguistics_term, uid.",
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
    summaries = [resolve_path(path, base_dir=ROOT) for path in args.summary]
    labels = args.label or [summary.parent.name for summary in summaries]
    if len(labels) != len(summaries):
        raise SystemExit("--label must be passed once per --summary.")

    output_dir = resolve_path(args.output_dir, base_dir=ROOT) if args.output_dir else None
    group_scopes = tuple(args.group_scope or ("field", "linguistics_term", "uid"))
    score_reduction = None if args.score_reduction in {None, "both"} else args.score_reduction

    for summary, label in zip(summaries, labels, strict=True):
        target_dir = output_dir or summary.parent
        plot_blimp_likelihood_summary(
            summary_csv=summary,
            output_dir=target_dir,
            benchmark_label=label,
            run_name=args.run_name,
            group_scopes=group_scopes,
            score_reduction=score_reduction,
            x_axis=args.x_axis,
        )
        print(f"Wrote plots for {label}: {target_dir}")

    if len(summaries) > 1:
        combined_dir = output_dir or summaries[0].parent.parent
        output_path = combined_dir / "blimp_likelihood_overall_accuracy.png"
        plot_overall_accuracy(
            summary_csvs=dict(zip(labels, summaries, strict=True)),
            output_path=output_path,
            run_name=args.run_name,
            score_reduction=score_reduction,
            x_axis=args.x_axis,
        )
        print(f"Wrote combined overall plot: {output_path}")


if __name__ == "__main__":
    main()
