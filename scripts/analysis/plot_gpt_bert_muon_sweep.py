#!/usr/bin/env python3
"""Summarize and plot GPT-BERT mixed Muon LR/seed sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RUN_PREFIX = "gpt_bert_baseline_mixed_babylm2026_strict_small_muon"
DEFAULT_RUNS_DIR = Path(
    os.environ.get("BABYSTEPS_RUNS_DIR", "~/babylm-baby-steps-runs")
).expanduser()
DEFAULT_TRAINING_ROOT = DEFAULT_RUNS_DIR / "training"
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "analysis" / "gpt_bert_mixed_muon_sweep"

RUN_NAME_RE = re.compile(
    r".*_muon_lr(?P<muon_lr>m?\d+p?\d*)_seed(?P<seed>\d+)(?:_.*)?$"
)

TRAIN_METRICS = (
    "loss",
    "ce_loss",
    "train_causal_loss",
    "train_mntp_loss",
)
VALIDATION_METRICS = (
    "val_causal_loss",
    "val_mntp_loss",
)
DEFAULT_PLOT_METRICS = (
    "train/loss",
    "train/train_causal_loss",
    "train/train_mntp_loss",
    "validation/val_causal_loss",
    "validation/val_mntp_loss",
    "mntp_likelihood/entity_tracking/sum/score",
    "ewok_context_sensitivity_full/sum/score",
)


@dataclass(frozen=True)
class RunInfo:
    """Metadata for one timestamped training run."""

    run_name: str
    run_dir: Path
    metrics_path: Path
    muon_lr: float | None
    seed: int | None
    config: dict[str, Any]


def slug_to_float(value: str) -> float:
    """Convert launcher slugs like ``0p005`` back to floats."""

    return float(value.replace("m", "-").replace("p", "."))


def parse_optional_float(value: Any) -> float | None:
    """Return a finite float, or None for missing/non-finite values."""

    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_optional_int(value: Any) -> int | None:
    """Return an int when possible."""

    parsed = parse_optional_float(value)
    if parsed is None:
        return None
    return int(parsed)


def read_json(path: Path) -> Any:
    """Load JSON from a file."""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_history(path: Path) -> list[dict[str, Any]]:
    """Load a metrics_history.json list."""

    history = read_json(path)
    if not isinstance(history, list):
        raise ValueError(f"Expected a metrics list in {path}.")
    return [record for record in history if isinstance(record, dict)]


def run_name_metadata(run_name: str) -> tuple[float | None, int | None]:
    """Parse muon LR and seed from a sweep run name."""

    match = RUN_NAME_RE.fullmatch(run_name)
    if match is None:
        return None, None
    return slug_to_float(match.group("muon_lr")), int(match.group("seed"))


def read_run_info(metrics_path: Path, *, run_prefix: str) -> RunInfo | None:
    """Read metadata for one metrics file, filtering by run prefix."""

    run_dir = metrics_path.parent
    config_path = run_dir / "config.json"
    config = read_json(config_path) if config_path.exists() else {}
    run_config = config.get("run", {}) if isinstance(config, dict) else {}
    training_config = config.get("training", {}) if isinstance(config, dict) else {}

    run_name = str(run_config.get("name") or run_dir.parent.name)
    if not run_name.startswith(run_prefix):
        return None

    parsed_lr, parsed_seed = run_name_metadata(run_name)
    muon_lr = parse_optional_float(training_config.get("muon_lr"))
    seed = parse_optional_int(run_config.get("seed"))

    return RunInfo(
        run_name=run_name,
        run_dir=run_dir,
        metrics_path=metrics_path,
        muon_lr=muon_lr if muon_lr is not None else parsed_lr,
        seed=seed if seed is not None else parsed_seed,
        config=config if isinstance(config, dict) else {},
    )


def discover_runs(training_root: Path, *, run_prefix: str) -> list[RunInfo]:
    """Find timestamped training runs that match the sweep prefix."""

    runs: list[RunInfo] = []
    for metrics_path in sorted(training_root.glob("*/*/metrics_history.json")):
        info = read_run_info(metrics_path, run_prefix=run_prefix)
        if info is not None:
            runs.append(info)
    return runs


def base_row(info: RunInfo, record: dict[str, Any]) -> dict[str, Any]:
    """Build the common row fields for one metrics record."""

    return {
        "run_name": info.run_name,
        "run_dir": str(info.run_dir),
        "muon_lr": "" if info.muon_lr is None else info.muon_lr,
        "seed": "" if info.seed is None else info.seed,
        "step": record.get("step", ""),
        "epoch": record.get("epoch", ""),
        "tokens_seen": record.get("tokens_seen", ""),
        "record_type": record.get("type", ""),
    }


def metric_rows_for_record(info: RunInfo, record: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize one metrics record into long-form metric rows."""

    record_type = str(record.get("type", ""))
    rows: list[dict[str, Any]] = []

    if record_type == "train":
        for metric in TRAIN_METRICS:
            value = parse_optional_float(record.get(metric))
            if value is not None:
                rows.append(
                    {
                        **base_row(info, record),
                        "metric_group": "train",
                        "metric": metric,
                        "benchmark": "",
                        "score_reduction": "",
                        "phase": "",
                        "value": value,
                    }
                )
        return rows

    if record_type == "validation":
        for metric in VALIDATION_METRICS:
            value = parse_optional_float(record.get(metric))
            if value is not None:
                rows.append(
                    {
                        **base_row(info, record),
                        "metric_group": "validation",
                        "metric": metric,
                        "benchmark": "",
                        "score_reduction": "",
                        "phase": "",
                        "value": value,
                    }
                )
        return rows

    if record_type == "mntp_likelihood":
        score = parse_optional_float(record.get("score"))
        if score is not None:
            rows.append(
                {
                    **base_row(info, record),
                    "metric_group": "mntp_likelihood",
                    "metric": "score",
                    "benchmark": record.get("benchmark", ""),
                    "score_reduction": record.get("score_reduction", ""),
                    "phase": "",
                    "value": score,
                }
            )
        return rows

    if record_type in {
        "ewok_context_sensitivity_full",
        "ewok_mntp_context_sensitivity",
    }:
        score = parse_optional_float(record.get("score"))
        if score is not None:
            rows.append(
                {
                    **base_row(info, record),
                    "metric_group": record_type,
                    "metric": "score",
                    "benchmark": "",
                    "score_reduction": record.get("score_reduction", ""),
                    "phase": "",
                    "value": score,
                }
            )

    return rows


def metric_id(row: dict[str, Any]) -> str:
    """Return a stable metric identifier for filtering and plots."""

    parts = [str(row["metric_group"])]
    benchmark = str(row.get("benchmark") or "")
    score_reduction = str(row.get("score_reduction") or "")
    if benchmark:
        parts.append(benchmark)
    if score_reduction:
        parts.append(score_reduction)
    parts.append(str(row["metric"]))
    return "/".join(parts)


def annotate_benchmark_phases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add midpoint/end labels to benchmark rows when possible."""

    benchmark_steps: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if row["metric_group"] not in {
            "mntp_likelihood",
            "ewok_context_sensitivity_full",
            "ewok_mntp_context_sensitivity",
        }:
            continue
        step = parse_optional_int(row.get("step"))
        if step is not None:
            benchmark_steps[str(row["run_dir"])].add(step)

    phase_by_run_step: dict[tuple[str, int], str] = {}
    for run_dir, steps in benchmark_steps.items():
        ordered = sorted(steps)
        if not ordered:
            continue
        if 0 in ordered:
            phase_by_run_step[(run_dir, 0)] = "start"
        positive_steps = [step for step in ordered if step > 0]
        if positive_steps:
            phase_by_run_step[(run_dir, positive_steps[-1])] = "end"
        if len(positive_steps) >= 2:
            phase_by_run_step[(run_dir, positive_steps[0])] = "midpoint"

    annotated: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        step = parse_optional_int(row.get("step"))
        if step is not None:
            copied["phase"] = phase_by_run_step.get((str(row["run_dir"]), step), "")
        annotated.append(copied)
    return annotated


def collect_metric_rows(runs: list[RunInfo]) -> list[dict[str, Any]]:
    """Collect long-form metrics from all discovered runs."""

    rows: list[dict[str, Any]] = []
    for info in runs:
        for record in read_history(info.metrics_path):
            rows.extend(metric_rows_for_record(info, record))
    return annotate_benchmark_phases(rows)


def column_slug(value: str) -> str:
    """Make a metric id safe as a CSV column name."""

    return (
        value.replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "p")
        .lower()
    )


def latest_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the row with the largest step."""

    if not rows:
        return None
    return max(rows, key=lambda row: parse_optional_float(row.get("step")) or -1.0)


def build_run_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one wide summary row per run."""

    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run[str(row["run_dir"])].append(row)

    summaries: list[dict[str, Any]] = []
    for run_dir, run_rows in sorted(by_run.items()):
        first = run_rows[0]
        summary: dict[str, Any] = {
            "run_name": first["run_name"],
            "run_dir": run_dir,
            "muon_lr": first["muon_lr"],
            "seed": first["seed"],
            "max_step": max(
                parse_optional_float(row.get("step")) or 0.0 for row in run_rows
            ),
        }

        by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in run_rows:
            by_metric[metric_id(row)].append(row)

        for name, metric_rows in sorted(by_metric.items()):
            row = latest_row(metric_rows)
            if row is not None:
                summary[f"latest_{column_slug(name)}"] = row["value"]

        for row in run_rows:
            phase = str(row.get("phase") or "")
            if phase not in {"start", "midpoint", "end"}:
                continue
            summary[f"{phase}_{column_slug(metric_id(row))}"] = row["value"]

        summaries.append(summary)
    return summaries


def build_lr_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate metrics by LR, metric, and phase."""

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        muon_lr = str(row.get("muon_lr", ""))
        if not muon_lr:
            continue
        phase = str(row.get("phase") or "latest")
        if row["metric_group"] in {"train", "validation"}:
            phase = "latest"
        grouped[(muon_lr, metric_id(row), phase, str(row["metric_group"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (muon_lr, name, phase, metric_group), metric_rows in sorted(grouped.items()):
        selected_rows = metric_rows
        if phase == "latest":
            latest_by_run: dict[str, dict[str, Any]] = {}
            for row in metric_rows:
                current = latest_by_run.get(str(row["run_dir"]))
                if current is None or (
                    parse_optional_float(row.get("step")) or -1.0
                ) > (parse_optional_float(current.get("step")) or -1.0):
                    latest_by_run[str(row["run_dir"])] = row
            selected_rows = list(latest_by_run.values())

        values = [float(row["value"]) for row in selected_rows]
        if not values:
            continue
        seeds = sorted(
            {
                int(seed)
                for row in selected_rows
                for seed in [parse_optional_int(row.get("seed"))]
                if seed is not None
            }
        )
        summaries.append(
            {
                "muon_lr": muon_lr,
                "metric_group": metric_group,
                "metric": name,
                "phase": phase,
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "seed_count": len(seeds),
                "seeds": " ".join(str(seed) for seed in seeds),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dict rows with a stable union of fieldnames."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean_by_step(rows: list[dict[str, Any]]) -> dict[float, float]:
    """Average row values at each step."""

    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        step = parse_optional_float(row.get("step"))
        value = parse_optional_float(row.get("value"))
        if step is not None and value is not None:
            grouped[step].append(value)
    return {step: statistics.mean(values) for step, values in grouped.items()}


def plot_metric(rows: list[dict[str, Any]], *, metric: str, output_path: Path) -> bool:
    """Plot one metric with seed traces and LR means."""

    metric_rows = [row for row in rows if metric_id(row) == metric]
    if not metric_rows:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_lr_seed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rows_by_lr: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        lr = str(row.get("muon_lr", "unknown"))
        seed = str(row.get("seed", "unknown"))
        rows_by_lr_seed[(lr, seed)].append(row)
        rows_by_lr[lr].append(row)

    fig, axis = plt.subplots(figsize=(9, 4.8))
    for (lr, seed), series_rows in sorted(rows_by_lr_seed.items()):
        series_rows = sorted(
            series_rows,
            key=lambda row: parse_optional_float(row.get("step")) or 0.0,
        )
        axis.plot(
            [float(row["step"]) for row in series_rows],
            [float(row["value"]) for row in series_rows],
            linewidth=1.0,
            alpha=0.35,
            label=f"lr={lr} seed={seed}",
        )

    for lr, lr_rows in sorted(rows_by_lr.items()):
        means = mean_by_step(lr_rows)
        axis.plot(
            sorted(means),
            [means[step] for step in sorted(means)],
            linewidth=2.6,
            label=f"lr={lr} mean",
        )

    axis.set_title(metric.replace("/", " / "))
    axis.set_xlabel("Optimizer update step")
    axis.set_ylabel("Loss" if "loss" in metric else "Score")
    axis.grid(True, axis="both", alpha=0.22)
    axis.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def write_plots(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    plot_metrics: tuple[str, ...],
) -> list[Path]:
    """Write configured metric plots."""

    written: list[Path] = []
    for metric in plot_metrics:
        output_path = output_dir / f"{column_slug(metric)}.png"
        if plot_metric(rows, metric=metric, output_path=output_path):
            written.append(output_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=DEFAULT_TRAINING_ROOT,
        help="Root containing timestamped training run directories.",
    )
    parser.add_argument(
        "--run-prefix",
        default=DEFAULT_RUN_PREFIX,
        help="Only include runs whose config run.name starts with this prefix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV summaries and plots.",
    )
    parser.add_argument(
        "--plot-metric",
        action="append",
        default=None,
        dest="plot_metrics",
        help="Metric id to plot. Repeat for multiple metrics.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSVs only.",
    )
    args = parser.parse_args()

    runs = discover_runs(args.training_root, run_prefix=args.run_prefix)
    if not runs:
        parser.error(f"No runs found under {args.training_root} with prefix {args.run_prefix!r}.")

    rows = collect_metric_rows(runs)
    if not rows:
        parser.error("Runs were found, but no supported metric rows were found.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows_path = output_dir / "metric_rows.csv"
    run_summary_path = output_dir / "run_summary.csv"
    lr_summary_path = output_dir / "lr_summary.csv"

    write_csv(metric_rows_path, rows)
    write_csv(run_summary_path, build_run_summary(rows))
    write_csv(lr_summary_path, build_lr_summary(rows))

    print(f"Wrote: {metric_rows_path}")
    print(f"Wrote: {run_summary_path}")
    print(f"Wrote: {lr_summary_path}")

    if not args.no_plots:
        plot_metrics = tuple(args.plot_metrics or DEFAULT_PLOT_METRICS)
        for path in write_plots(rows, output_dir=output_dir / "plots", plot_metrics=plot_metrics):
            print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
