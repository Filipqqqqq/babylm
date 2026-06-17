#!/usr/bin/env python3
"""Compare best BabyLM fast-eval scores across optimizer/batch-size conditions."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EVAL_ROOT = Path("/SSD2/babylm-baby-steps-runs/evaluation")
DEFAULT_OUTPUT_DIR = Path("/SSD2/babylm-baby-steps-runs/analysis/best_optimizer_comparison")

ADAMW_RUN_RE = re.compile(
    r"gpt2_8l512_bpe(?P<bpe>\d+)_lr(?P<lr_slug>[^_]+)"
    r"(?:_bs(?P<batch_size>\d+))?_seed(?P<seed>\d+)_fast_eval$"
)
MUON_RUN_RE = re.compile(
    r"gpt2_8l512_bpe(?P<bpe>\d+)_muon_lr(?P<lr_slug>[^_]+)"
    r"_wu(?P<warmup_slug>[^_]+)_mwd(?P<weight_decay_slug>[^_]+)"
    r"(?:_bs(?P<batch_size>\d+))?_seed(?P<seed>\d+)_fast_eval$"
)

CONDITIONS = (
    ("AdamW bs16", "adamw", ""),
    ("Muon bs16", "muon", ""),
    ("Muon bs32", "muon", "32"),
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
MACRO_LABELS = {
    "accuracy_macro": "Accuracy Macro",
    "language_macro": "BLiMP + Supplement + EWoK",
}
READING_LABELS = {
    "eye_tracking_score": "Eye Tracking",
    "self_paced_reading_score": "Self-Paced Reading",
}
CONDITION_COLORS = {
    "AdamW bs16": "#4c78a8",
    "Muon bs16": "#59a14f",
    "Muon bs32": "#b07aa1",
}


def slug_to_float(slug: str) -> float:
    """Convert launcher slugs like ``0p00005`` back to floats."""

    return float(slug.replace("m", "-").replace("p", "."))


def format_lr(value: float) -> str:
    """Compact LR labels for plot annotations."""

    if value == 0:
        return "0"
    return f"{value:.1e}".replace(".0e", "e")


def parse_run_name(name: str) -> dict[str, Any] | None:
    match = ADAMW_RUN_RE.fullmatch(name)
    if match:
        return {
            "run_name": name.removesuffix("_fast_eval"),
            "optimizer": "adamw",
            "bpe": int(match.group("bpe")),
            "lr": slug_to_float(match.group("lr_slug")),
            "batch_size": match.group("batch_size") or "",
            "seed": int(match.group("seed")),
        }

    match = MUON_RUN_RE.fullmatch(name)
    if match:
        return {
            "run_name": name.removesuffix("_fast_eval"),
            "optimizer": "muon",
            "bpe": int(match.group("bpe")),
            "lr": slug_to_float(match.group("lr_slug")),
            "batch_size": match.group("batch_size") or "",
            "seed": int(match.group("seed")),
        }

    return None


def condition_for(run_info: dict[str, Any]) -> str | None:
    """Map parsed run metadata onto the comparison conditions."""

    for label, optimizer, batch_size in CONDITIONS:
        if run_info["optimizer"] == optimizer and run_info["batch_size"] == batch_size:
            return label
    return None


def load_eval_rows(eval_root: str | Path) -> list[dict[str, Any]]:
    """Load per-run average and reading scores from summary CSV files."""

    eval_root = Path(eval_root)
    rows: list[dict[str, Any]] = []

    for summary_path in sorted(eval_root.glob("*_fast_eval/summary.csv")):
        run_info = parse_run_name(summary_path.parent.name)
        if run_info is None:
            continue
        condition = condition_for(run_info)
        if condition is None:
            continue

        task_values: dict[str, float] = {}
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    value = float(row["value"])
                except (KeyError, ValueError):
                    continue

                task = row.get("task", "")
                metric_scope = row.get("metric_scope", "")
                metric_name = row.get("metric_name", "")

                if (
                    task in ACCURACY_TASKS
                    and metric_scope == "average"
                    and metric_name == "average"
                ):
                    task_values[task] = value
                    rows.append(
                        {
                            **run_info,
                            "condition": condition,
                            "metric_group": "benchmarks",
                            "metric": task,
                            "value": value,
                            "summary_csv": str(summary_path),
                        }
                    )

                if (
                    task == "reading"
                    and metric_scope == "report"
                    and metric_name in READING_METRICS
                ):
                    rows.append(
                        {
                            **run_info,
                            "condition": condition,
                            "metric_group": "reading",
                            "metric": metric_name,
                            "value": value,
                            "summary_csv": str(summary_path),
                        }
                    )

        if task_values:
            accuracy_values = [task_values[task] for task in ACCURACY_TASKS if task in task_values]
            language_values = [task_values[task] for task in LANGUAGE_TASKS if task in task_values]
            for metric, values in (
                ("accuracy_macro", accuracy_values),
                ("language_macro", language_values),
            ):
                if not values:
                    continue
                rows.append(
                    {
                        **run_info,
                        "condition": condition,
                        "metric_group": "macros",
                        "metric": metric,
                        "value": statistics.mean(values),
                        "summary_csv": str(summary_path),
                    }
                )

    return rows


def select_best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick the LR with the best mean score for each BPE/condition/metric."""

    grouped: dict[tuple[int, str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["bpe"],
            row["condition"],
            row["metric_group"],
            row["metric"],
            row["lr"],
        )
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for (bpe, condition, metric_group, metric, lr), lr_rows in grouped.items():
        values = [row["value"] for row in lr_rows]
        summaries.append(
            {
                "bpe": bpe,
                "condition": condition,
                "optimizer": lr_rows[0]["optimizer"],
                "batch_size": lr_rows[0]["batch_size"] or "16",
                "metric_group": metric_group,
                "metric": metric,
                "selected_lr": lr,
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "seed_count": len({row["seed"] for row in lr_rows}),
                "seeds": " ".join(str(seed) for seed in sorted({row["seed"] for row in lr_rows})),
            }
        )

    best_by_metric: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for summary in summaries:
        key = (
            summary["bpe"],
            summary["condition"],
            summary["metric_group"],
            summary["metric"],
        )
        current = best_by_metric.get(key)
        if current is None or summary["mean"] > current["mean"]:
            best_by_metric[key] = summary

    return sorted(
        best_by_metric.values(),
        key=lambda row: (
            row["bpe"],
            row["metric_group"],
            row["metric"],
            [label for label, _, _ in CONDITIONS].index(row["condition"]),
        ),
    )


def write_best_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bpe",
        "condition",
        "optimizer",
        "batch_size",
        "metric_group",
        "metric",
        "selected_lr",
        "mean",
        "min",
        "max",
        "seed_count",
        "seeds",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def make_plot_backend_ready() -> None:
    import matplotlib

    matplotlib.use("Agg")


def plot_grouped_bars(
    *,
    rows: list[dict[str, Any]],
    bpe: int,
    metric_group: str,
    metrics: tuple[str, ...],
    labels: dict[str, str],
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    make_plot_backend_ready()
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    condition_labels = [label for label, _, _ in CONDITIONS]
    rows_by_key = {
        (row["metric"], row["condition"]): row
        for row in rows
        if row["bpe"] == bpe and row["metric_group"] == metric_group
    }

    x_positions = list(range(len(metrics)))
    bar_width = 0.23
    offsets = {
        condition: (idx - (len(condition_labels) - 1) / 2) * bar_width
        for idx, condition in enumerate(condition_labels)
    }

    fig_width = max(8.5, 2.2 * len(metrics) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 5.9))

    for condition in condition_labels:
        means = []
        lower = []
        upper = []
        annotations = []
        for metric in metrics:
            row = rows_by_key.get((metric, condition))
            if row is None:
                means.append(float("nan"))
                lower.append(0.0)
                upper.append(0.0)
                annotations.append("")
                continue

            mean = row["mean"]
            means.append(mean)
            lower.append(mean - row["min"])
            upper.append(row["max"] - mean)
            annotations.append(f"lr {format_lr(row['selected_lr'])}\nn={row['seed_count']}")

        shifted_x = [x + offsets[condition] for x in x_positions]
        bars = ax.bar(
            shifted_x,
            means,
            width=bar_width,
            color=CONDITION_COLORS[condition],
            label=condition,
            yerr=[lower, upper],
            capsize=4,
            alpha=0.92,
        )

        for bar, annotation in zip(bars, annotations, strict=True):
            if not annotation:
                continue
            height = bar.get_height()
            ax.annotate(
                annotation,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    visible_rows = [
        row
        for row in rows
        if row["bpe"] == bpe and row["metric_group"] == metric_group and row["metric"] in metrics
    ]
    if visible_rows:
        min_value = min(row["min"] for row in visible_rows)
        max_value = max(row["max"] for row in visible_rows)
        value_range = max(max_value - min_value, 1.0)
        ax.set_ylim(top=max_value + value_range * 0.28)

    fig.suptitle(title, fontsize=14, y=0.985)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([labels.get(metric, metric) for metric in metrics])
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT), help="Root containing *_fast_eval dirs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Where plots/CSVs are written.")
    parser.add_argument("--bpes", nargs="+", type=int, default=[8192, 16384], help="BPE vocab sizes to plot.")
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    output_dir = Path(args.output_dir)

    rows = load_eval_rows(eval_root)
    if not rows:
        raise SystemExit(f"No matching eval rows found under: {eval_root}")

    best_rows = select_best_rows(rows)
    write_best_csv(best_rows, output_dir / "best_scores.csv")

    for bpe in args.bpes:
        plot_grouped_bars(
            rows=best_rows,
            bpe=bpe,
            metric_group="macros",
            metrics=("accuracy_macro", "language_macro"),
            labels=MACRO_LABELS,
            output_path=output_dir / f"best_macro_scores_bpe{bpe}.png",
            title=f"BPE {bpe}: Best Macro Scores by Optimizer and Batch Size",
            ylabel="Mean score (%)",
        )
        plot_grouped_bars(
            rows=best_rows,
            bpe=bpe,
            metric_group="benchmarks",
            metrics=ACCURACY_TASKS,
            labels=TASK_LABELS,
            output_path=output_dir / f"best_benchmark_scores_bpe{bpe}.png",
            title=f"BPE {bpe}: Best Benchmark Scores by Optimizer and Batch Size",
            ylabel="Accuracy (%)",
        )
        plot_grouped_bars(
            rows=best_rows,
            bpe=bpe,
            metric_group="reading",
            metrics=READING_METRICS,
            labels=READING_LABELS,
            output_path=output_dir / f"best_reading_scores_bpe{bpe}.png",
            title=f"BPE {bpe}: Best Reading Scores by Optimizer and Batch Size",
            ylabel="Reading score",
        )

    print(f"Loaded {len(rows):,} comparison score rows from {eval_root}")
    print(f"Wrote: {output_dir / 'best_scores.csv'}")
    for bpe in args.bpes:
        print(f"Wrote: {output_dir / f'best_macro_scores_bpe{bpe}.png'}")
        print(f"Wrote: {output_dir / f'best_benchmark_scores_bpe{bpe}.png'}")
        print(f"Wrote: {output_dir / f'best_reading_scores_bpe{bpe}.png'}")


if __name__ == "__main__":
    main()
