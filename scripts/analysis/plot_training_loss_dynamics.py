#!/usr/bin/env python3
"""Plot training loss curves from one or more metrics_history.json files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.analysis.loss_dynamics import DEFAULT_LOSS_KEYS
from babysteps.analysis.loss_dynamics import write_training_loss_plots


def resolve_metrics_path(path: str | Path) -> Path:
    """Accept either a run directory or a metrics_history.json path."""

    path = Path(path)
    if path.is_dir():
        return path / "metrics_history.json"
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        nargs="+",
        help="Run directories or metrics_history.json files to plot.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        dest="labels",
        help="Series label, repeat once per metrics path.",
    )
    parser.add_argument(
        "--output-dir",
        default="loss_dynamics",
        help="Directory where PNG plots will be written.",
    )
    parser.add_argument(
        "--x-axis",
        default="words_seen",
        choices=("step", "tokens_seen", "words_seen"),
        help="Training progress field to use for the x-axis.",
    )
    parser.add_argument(
        "--loss-key",
        action="append",
        default=None,
        dest="loss_keys",
        help=(
            "Loss metric key to plot. Repeat to plot multiple keys. "
            f"Defaults to: {', '.join(DEFAULT_LOSS_KEYS)}."
        ),
    )
    args = parser.parse_args()

    metrics_paths = [resolve_metrics_path(path) for path in args.metrics]
    missing = [path for path in metrics_paths if not path.exists()]
    if missing:
        parser.error("Missing metrics file(s): " + ", ".join(str(path) for path in missing))

    loss_keys = tuple(args.loss_keys) if args.loss_keys else DEFAULT_LOSS_KEYS
    try:
        written = write_training_loss_plots(
            metrics_paths=metrics_paths,
            output_dir=args.output_dir,
            labels=args.labels,
            loss_keys=loss_keys,
            x_axis=args.x_axis,
        )
    except ValueError as exc:
        parser.error(str(exc))

    for path in written:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
