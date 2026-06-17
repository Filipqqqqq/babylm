#!/usr/bin/env python3
"""Plot BabyLM fast-eval sweep results by optimizer, tokenizer, LR, and seed."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EVAL_ROOT = Path("/SSD2/babylm-baby-steps-runs/evaluation")
DEFAULT_OUTPUT_DIR = Path("/SSD2/babylm-baby-steps-runs/analysis/eval_sweep")

ADAMW_RUN_RE = re.compile(
    r"gpt2_(?P<layers>\d+)l(?P<hidden_size>\d+)_bpe(?P<bpe>\d+)_lr(?P<lr_slug>[^_]+)"
    r"(?:_bs(?P<batch_size>\d+))?_seed(?P<seed>\d+)_fast_eval$"
)
MUON_RUN_RE = re.compile(
    r"gpt2_(?P<layers>\d+)l(?P<hidden_size>\d+)_bpe(?P<bpe>\d+)_muon_lr(?P<lr_slug>[^_]+)"
    r"_wu(?P<warmup_slug>[^_]+)_mwd(?P<weight_decay_slug>[^_]+)"
    r"(?:_bs(?P<batch_size>\d+))?_seed(?P<seed>\d+)_fast_eval$"
)

ACCURACY_TASKS = (
    "blimp_fast",
    "supplement_fast",
    "ewok_fast",
    "entity_tracking_fast",
)
LANGUAGE_TASKS = (
    "blimp_fast",
    "supplement_fast",
    "ewok_fast",
)
READING_METRICS = (
    "eye_tracking_score",
    "self_paced_reading_score",
)

TASK_LABELS = {
    "blimp_fast": "BLiMP",
    "supplement_fast": "Supplement",
    "ewok_fast": "EWoK",
    "entity_tracking_fast": "Entity Tracking",
}

READING_LABELS = {
    "eye_tracking_score": "Eye Tracking",
    "self_paced_reading_score": "Self-Paced Reading",
}

MACRO_LABELS = {
    "accuracy_macro": "Accuracy Macro",
    "language_macro": "BLiMP + Supplement + EWoK",
}

MODEL_SIZE_COLORS = {
    "8l512": "#4c78a8",
    "12l384": "#59a14f",
    "12l512": "#f28e2b",
}


def lr_slug_to_float(slug: str) -> float:
    """Convert launcher LR slugs like ``0p000025`` back to floats."""

    return float(slug.replace("m", "-").replace("p", "."))


def format_lr(value: float) -> str:
    """Compact, stable LR labels for plots."""

    if value == 0:
        return "0"
    return f"{value:.1e}".replace(".0e", "e")


def parse_run_name(name: str) -> dict[str, Any] | None:
    match = ADAMW_RUN_RE.fullmatch(name)
    if match:
        model_size = f"{match.group('layers')}l{match.group('hidden_size')}"
        return {
            "run_name": name.removesuffix("_fast_eval"),
            "model_size": model_size,
            "layers": int(match.group("layers")),
            "hidden_size": int(match.group("hidden_size")),
            "optimizer": "adamw",
            "bpe": int(match.group("bpe")),
            "lr": lr_slug_to_float(match.group("lr_slug")),
            "batch_size": match.group("batch_size") or "",
            "seed": int(match.group("seed")),
            "warmup_ratio": "",
            "muon_weight_decay": "",
        }

    match = MUON_RUN_RE.fullmatch(name)
    if match:
        model_size = f"{match.group('layers')}l{match.group('hidden_size')}"
        return {
            "run_name": name.removesuffix("_fast_eval"),
            "model_size": model_size,
            "layers": int(match.group("layers")),
            "hidden_size": int(match.group("hidden_size")),
            "optimizer": "muon",
            "bpe": int(match.group("bpe")),
            # For Muon plots, lr means the Muon matrix update learning rate.
            "lr": lr_slug_to_float(match.group("lr_slug")),
            "batch_size": match.group("batch_size") or "",
            "seed": int(match.group("seed")),
            "warmup_ratio": lr_slug_to_float(match.group("warmup_slug")),
            "muon_weight_decay": lr_slug_to_float(match.group("weight_decay_slug")),
        }

    return None


def batch_size_matches(actual: str, requested: str) -> bool:
    """Return whether a parsed run batch size matches a CLI filter."""

    if requested == "all":
        return True
    if requested == "default":
        return actual == ""
    return actual == requested


def batch_size_title(requested: str) -> str:
    """Human-readable suffix for plot titles."""

    if requested == "all":
        return "All Batch Sizes"
    if requested == "default":
        return "Default Batch Size"
    return f"Batch Size {requested}"


def selection_matches(actual: Any, requested: tuple[str, ...]) -> bool:
    """Return whether a parsed metadata value passes a CLI selection."""

    return "all" in requested or str(actual) in requested


def selection_label(label: str, requested: tuple[str, ...]) -> str:
    """Human-readable title fragment for a CLI selection."""

    if "all" in requested:
        return ""
    return f"{label} {'/'.join(requested)}"


def load_eval_scores(
    eval_root: str | Path,
    *,
    optimizer: str,
    batch_size: str,
    model_sizes: tuple[str, ...],
    bpes: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Read all sweep summary CSV files into a long score table."""

    eval_root = Path(eval_root)
    records: list[dict[str, Any]] = []

    for summary_path in sorted(eval_root.glob("*_fast_eval/summary.csv")):
        run_info = parse_run_name(summary_path.parent.name)
        if run_info is None:
            continue
        if optimizer != "all" and run_info["optimizer"] != optimizer:
            continue
        if not batch_size_matches(run_info["batch_size"], batch_size):
            continue
        if not selection_matches(run_info["model_size"], model_sizes):
            continue
        if not selection_matches(run_info["bpe"], bpes):
            continue

        with summary_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    value = float(row["value"])
                except (KeyError, ValueError):
                    continue

                records.append(
                    {
                        **run_info,
                        "eval_dir": str(summary_path.parent),
                        "summary_csv": str(summary_path),
                        "checkpoint_path": row.get("checkpoint_path", ""),
                        "task": row.get("task", ""),
                        "metric_scope": row.get("metric_scope", ""),
                        "metric_name": row.get("metric_name", ""),
                        "value": value,
                    }
                )

    return records


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "model_size",
        "layers",
        "hidden_size",
        "optimizer",
        "bpe",
        "lr",
        "batch_size",
        "seed",
        "warmup_ratio",
        "muon_weight_decay",
        "task",
        "metric_scope",
        "metric_name",
        "value",
        "eval_dir",
        "summary_csv",
        "checkpoint_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def average_score_rows(records: list[dict[str, Any]], tasks: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if row["task"] in tasks
        and row["metric_scope"] == "average"
        and row["metric_name"] == "average"
    ]


def reading_score_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in records:
        if (
            row["task"] == "reading"
            and row["metric_scope"] == "report"
            and row["metric_name"] in READING_METRICS
        ):
            # The eval summary stores both reading scores under task="reading".
            # For plotting, treat each metric as its own panel.
            plot_row = dict(row)
            plot_row["task"] = row["metric_name"]
            rows.append(plot_row)
    return rows


def build_macro_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build per-run macro averages from benchmark average rows."""

    by_run: dict[tuple[str, str, int, int, int, float, str, int, str, Any, Any], dict[str, float]] = defaultdict(dict)
    for row in average_score_rows(records, ACCURACY_TASKS):
        key = (
            row["model_size"],
            row["optimizer"],
            row["layers"],
            row["hidden_size"],
            row["bpe"],
            row["lr"],
            row["batch_size"],
            row["seed"],
            row["run_name"],
            row["warmup_ratio"],
            row["muon_weight_decay"],
        )
        by_run[key][row["task"]] = row["value"]

    macro_rows: list[dict[str, Any]] = []
    for (
        model_size,
        optimizer,
        layers,
        hidden_size,
        bpe,
        lr,
        batch_size,
        seed,
        run_name,
        warmup_ratio,
        muon_weight_decay,
    ), task_scores in sorted(by_run.items()):
        for macro_name, tasks in (
            ("accuracy_macro", ACCURACY_TASKS),
            ("language_macro", LANGUAGE_TASKS),
        ):
            values = [task_scores[task] for task in tasks if task in task_scores]
            if not values:
                continue
            macro_rows.append(
                {
                    "run_name": run_name,
                    "model_size": model_size,
                    "layers": layers,
                    "hidden_size": hidden_size,
                    "optimizer": optimizer,
                    "bpe": bpe,
                    "lr": lr,
                    "batch_size": batch_size,
                    "seed": seed,
                    "warmup_ratio": warmup_ratio,
                    "muon_weight_decay": muon_weight_decay,
                    "task": macro_name,
                    "metric_scope": "macro",
                    "metric_name": "average",
                    "value": statistics.mean(values),
                    "eval_dir": "",
                    "summary_csv": "",
                    "checkpoint_path": "",
                }
            )

    return macro_rows


def make_plot_backend_ready() -> None:
    import matplotlib

    matplotlib.use("Agg")


def plot_score_grid(
    *,
    rows: list[dict[str, Any]],
    tasks: tuple[str, ...],
    labels: dict[str, str],
    output_path: str | Path,
    ylabel: str,
    xlabel: str,
    title: str,
) -> None:
    make_plot_backend_ready()
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    series = sorted(
        {
            (
                row["optimizer"],
                row["bpe"],
                row.get("batch_size", ""),
                row.get("model_size", ""),
            )
            for row in rows
        }
    )
    lrs = sorted({row["lr"] for row in rows})
    if not series or not lrs:
        raise ValueError("No rows available to plot.")

    colors = {
        ("adamw", 8192): "#1f77b4",
        ("adamw", 16384): "#d62728",
        ("muon", 8192): "#2ca02c",
        ("muon", 16384): "#9467bd",
    }
    fallback_colors = [
        "#4c78a8",
        "#f28e2b",
        "#59a14f",
        "#b07aa1",
        "#e15759",
        "#76b7b2",
    ]
    offsets = {
        series_key: (idx - (len(series) - 1) / 2) * 0.12
        for idx, series_key in enumerate(series)
    }
    x_positions = list(range(len(lrs)))

    ncols = 2
    nrows = (len(tasks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * nrows), squeeze=False)
    axes_flat = list(axes.ravel())

    for axis, task in zip(axes_flat, tasks, strict=False):
        task_rows = [row for row in rows if row["task"] == task]
        for series_idx, series_key in enumerate(series):
            optimizer, bpe, batch_size, model_size = series_key
            series_rows = [
                row
                for row in task_rows
                if row["optimizer"] == optimizer
                and row["bpe"] == bpe
                and row.get("batch_size", "") == batch_size
                and row.get("model_size", "") == model_size
            ]
            means = []
            lower = []
            upper = []
            scatter_x = []
            scatter_y = []

            for x, lr in zip(x_positions, lrs, strict=True):
                values = [row["value"] for row in series_rows if row["lr"] == lr]
                if not values:
                    means.append(float("nan"))
                    lower.append(0.0)
                    upper.append(0.0)
                    continue

                mean = statistics.mean(values)
                means.append(mean)
                lower.append(mean - min(values))
                upper.append(max(values) - mean)
                scatter_x.extend([x + offsets[series_key]] * len(values))
                scatter_y.extend(values)

            color = MODEL_SIZE_COLORS.get(
                model_size,
                colors.get((optimizer, bpe), fallback_colors[series_idx % len(fallback_colors)]),
            )
            shifted_x = [x + offsets[series_key] for x in x_positions]
            batch_label = f" bs{batch_size}" if batch_size else ""
            model_label = f"{model_size} " if len({row.get("model_size", "") for row in rows}) > 1 else ""
            legend_label = (
                f"{model_label}BPE {bpe}{batch_label}"
                if len({row["optimizer"] for row in rows}) == 1
                else f"{optimizer} {model_label}BPE {bpe}{batch_label}"
            )
            axis.errorbar(
                shifted_x,
                means,
                yerr=[lower, upper],
                marker="o",
                linewidth=2,
                capsize=4,
                color=color,
                label=legend_label,
            )
            axis.scatter(
                scatter_x,
                scatter_y,
                color=color,
                alpha=0.45,
                s=32,
                edgecolors="none",
            )

        axis.set_title(labels.get(task, task))
        axis.set_xticks(x_positions)
        axis.set_xticklabels([format_lr(lr) for lr in lrs])
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)

    for axis in axes_flat[len(tasks) :]:
        axis.axis("off")

    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle(title, y=0.99, fontsize=14)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=max(1, min(len(series), 4)),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT), help="Root containing *_fast_eval dirs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Where plots/CSVs are written.")
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "muon", "all"),
        default="adamw",
        help="Which run-name family to plot. Defaults to AdamW to preserve the original sweep plots.",
    )
    parser.add_argument(
        "--batch-size",
        default="all",
        help=(
            "Filter by run batch size. Use 'default' for runs without a _bs suffix, "
            "'all' for everything, or a number like '32'."
        ),
    )
    parser.add_argument(
        "--model-size",
        nargs="+",
        default=["all"],
        help="Filter by model size, e.g. --model-size 12l384 12l512. Use 'all' for no filter.",
    )
    parser.add_argument(
        "--bpe",
        nargs="+",
        default=["all"],
        help="Filter by tokenizer vocab size, e.g. --bpe 16384. Use 'all' for no filter.",
    )
    parser.add_argument(
        "--filename-prefix",
        default="",
        help="Optional prefix for output plot filenames, e.g. muon_arch_12layer_bpe16384_bs32.",
    )
    parser.add_argument(
        "--title-prefix",
        help="Optional plot title prefix. If omitted, one is built from the filters.",
    )
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    output_dir = Path(args.output_dir)

    records = load_eval_scores(
        eval_root,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        model_sizes=tuple(args.model_size),
        bpes=tuple(args.bpe),
    )
    if not records:
        raise SystemExit(
            f"No {args.optimizer} eval summary rows found under {eval_root} "
            f"for batch_size={args.batch_size!r}, model_size={args.model_size}, bpe={args.bpe}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    macro_rows = build_macro_rows(records)
    optimizer_title = {
        "adamw": "AdamW",
        "muon": "Muon",
        "all": "AdamW and Muon",
    }[args.optimizer]
    if args.title_prefix:
        title_prefix = args.title_prefix
    else:
        title_parts = [
            optimizer_title,
            selection_label("Model", tuple(args.model_size)),
            selection_label("BPE", tuple(args.bpe)),
            batch_size_title(args.batch_size),
        ]
        title_prefix = " ".join(part for part in title_parts if part)
    xlabel = "Muon learning rate" if args.optimizer == "muon" else "Learning rate"
    filename_prefix = f"{args.filename_prefix}_" if args.filename_prefix else ""
    benchmark_plot = output_dir / f"{filename_prefix}benchmark_scores_by_lr.png"
    macro_plot = output_dir / f"{filename_prefix}macro_scores_by_lr.png"
    reading_plot = output_dir / f"{filename_prefix}reading_scores_by_lr.png"

    write_csv(records, output_dir / "eval_scores.csv")
    write_csv(macro_rows, output_dir / "macro_scores.csv")

    benchmark_rows = average_score_rows(records, ACCURACY_TASKS)
    plot_score_grid(
        rows=benchmark_rows,
        tasks=ACCURACY_TASKS,
        labels=TASK_LABELS,
        output_path=benchmark_plot,
        ylabel="Accuracy (%)",
        xlabel=xlabel,
        title=f"{title_prefix} BabyLM Fast Eval Scores by Learning Rate",
    )

    plot_score_grid(
        rows=macro_rows,
        tasks=("accuracy_macro", "language_macro"),
        labels=MACRO_LABELS,
        output_path=macro_plot,
        ylabel="Mean score (%)",
        xlabel=xlabel,
        title=f"{title_prefix} Macro Scores by Learning Rate",
    )

    reading_rows = reading_score_rows(records)
    if reading_rows:
        plot_score_grid(
            rows=reading_rows,
            tasks=READING_METRICS,
            labels=READING_LABELS,
            output_path=reading_plot,
            ylabel="Reading score",
            xlabel=xlabel,
            title=f"{title_prefix} Reading Scores by Learning Rate",
        )

    print(
        f"Loaded {len(records):,} {args.optimizer} score rows from {eval_root} "
        f"for batch_size={args.batch_size}, model_size={args.model_size}, bpe={args.bpe}"
    )
    print(f"Wrote: {output_dir / 'eval_scores.csv'}")
    print(f"Wrote: {output_dir / 'macro_scores.csv'}")
    print(f"Wrote: {benchmark_plot}")
    print(f"Wrote: {macro_plot}")
    if reading_rows:
        print(f"Wrote: {reading_plot}")


if __name__ == "__main__":
    main()
