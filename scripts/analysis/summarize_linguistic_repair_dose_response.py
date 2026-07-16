#!/usr/bin/env python3
"""Aggregate and plot the controlled linguistic-repair dose-response runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


CONDITION_ORDER = ("repair_0pct", "repair_1pct", "repair_1p5pct")
CONDITION_LABEL = {
    "repair_0pct": "0%",
    "repair_1pct": "1%",
    "repair_1p5pct": "1.5%",
}
TASK_ORDER = ("boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli")


def parse_args() -> argparse.Namespace:
    default_root = Path(
        os.environ.get(
            "BABYSTEPS_LINGUISTIC_REPAIR_EXPERIMENT_ROOT",
            str(
                Path.home()
                / "babylm-baby-steps-runs"
                / "experiments"
                / "gpt_bert_linguistic_repair_dose_response_v1"
            ),
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=default_root)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Aggregate completed arms while the queues are still running.",
    )
    parser.add_argument(
        "--historical-baseline",
        type=float,
        help="Optional historical macro score, expressed on the 0-1 scale.",
    )
    return parser.parse_args()


def parse_summary_path(path: Path, finetuning_root: Path) -> dict[str, Any]:
    relative = path.relative_to(finetuning_root)
    parts = relative.parts
    if len(parts) < 7:
        raise ValueError(f"Unexpected fine-tuning summary path: {path}")
    condition, pretrain_dir, lr_dir, finetune_dir = parts[:4]
    if not pretrain_dir.startswith("pretrain_seed_"):
        raise ValueError(f"Unexpected pretraining-seed directory: {pretrain_dir}")
    if not lr_dir.startswith("lr_"):
        raise ValueError(f"Unexpected learning-rate directory: {lr_dir}")
    if not finetune_dir.startswith("finetune_seed_"):
        raise ValueError(f"Unexpected fine-tuning-seed directory: {finetune_dir}")
    return {
        "condition": condition,
        "pretrain_seed": int(pretrain_dir.removeprefix("pretrain_seed_")),
        "learning_rate": lr_dir.removeprefix("lr_"),
        "finetune_seed": int(finetune_dir.removeprefix("finetune_seed_")),
        "summary_path": str(path),
    }


def collect_results(experiment_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    finetuning_root = experiment_root / "finetuning"
    macro_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for path in sorted(finetuning_root.glob("**/superglue_summary.json")):
        identity = parse_summary_path(path, finetuning_root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        macro = payload.get("macro_primary")
        if macro is None:
            continue
        macro_rows.append({**identity, "macro_primary": float(macro)})
        for task_result in payload.get("task_results", []):
            task_rows.append(
                {
                    **identity,
                    "task": str(task_result["task"]),
                    "primary_metric": str(task_result["primary_metric"]),
                    "primary_score": float(task_result["primary_score"]),
                }
            )
    return macro_rows, task_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def paired_differences(
    macro_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flattened = [
        {
            **row,
            "task": "macro",
            "primary_metric": "macro_primary",
            "primary_score": row["macro_primary"],
        }
        for row in macro_rows
    ] + task_rows
    lookup = {
        (
            row["condition"],
            row["pretrain_seed"],
            row["finetune_seed"],
            row["learning_rate"],
            row["task"],
        ): row
        for row in flattened
    }
    output: list[dict[str, Any]] = []
    for row in flattened:
        if row["condition"] == "repair_0pct":
            continue
        control = lookup.get(
            (
                "repair_0pct",
                row["pretrain_seed"],
                row["finetune_seed"],
                row["learning_rate"],
                row["task"],
            )
        )
        if control is None:
            continue
        output.append(
            {
                "condition": row["condition"],
                "pretrain_seed": row["pretrain_seed"],
                "finetune_seed": row["finetune_seed"],
                "learning_rate": row["learning_rate"],
                "task": row["task"],
                "primary_metric": row["primary_metric"],
                "control_score": control["primary_score"],
                "repair_score": row["primary_score"],
                "difference": row["primary_score"] - control["primary_score"],
            }
        )
    return output


def lr_sort_key(value: str) -> float:
    return float(value)


def plot_dose_response(
    macro_rows: list[dict[str, Any]],
    *,
    output_path: Path,
    historical_baseline: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    learning_rates = sorted({row["learning_rate"] for row in macro_rows}, key=lr_sort_key)
    figure, axes = plt.subplots(
        1,
        len(learning_rates),
        figsize=(5.2 * len(learning_rates), 4.5),
        sharey=True,
        squeeze=False,
    )
    x_values = list(range(len(CONDITION_ORDER)))
    colors = {42: "#4477AA", 43: "#EE6677", 44: "#228833"}

    for axis, learning_rate in zip(axes[0], learning_rates):
        lr_rows = [row for row in macro_rows if row["learning_rate"] == learning_rate]
        seeds = sorted({row["pretrain_seed"] for row in lr_rows})
        lookup = {
            (row["pretrain_seed"], row["condition"]): row["macro_primary"] * 100
            for row in lr_rows
        }
        for seed in seeds:
            points = [lookup.get((seed, condition), math.nan) for condition in CONDITION_ORDER]
            axis.plot(
                x_values,
                points,
                marker="o",
                linewidth=1.2,
                alpha=0.75,
                color=colors.get(seed),
                label=f"seed {seed}",
            )

        means: list[float] = []
        deviations: list[float] = []
        for condition in CONDITION_ORDER:
            values = [
                row["macro_primary"] * 100
                for row in lr_rows
                if row["condition"] == condition
            ]
            means.append(mean(values) if values else math.nan)
            deviations.append(stdev(values) if len(values) > 1 else 0.0)
        axis.errorbar(
            x_values,
            means,
            yerr=deviations,
            color="black",
            marker="o",
            linewidth=2.6,
            capsize=4,
            label="mean ± SD",
            zorder=10,
        )
        if historical_baseline is not None:
            axis.axhline(
                historical_baseline * 100,
                color="0.45",
                linestyle="--",
                linewidth=1.2,
                label="historical baseline",
            )
        axis.set_title(f"Fine-tuning LR = {learning_rate}")
        axis.set_xticks(x_values, [CONDITION_LABEL[item] for item in CONDITION_ORDER])
        axis.set_xlabel("Repair-token fraction")
        axis.grid(axis="y", alpha=0.25)

    axes[0][0].set_ylabel("SuperGLUE macro score (%)")
    handles, labels = axes[0][-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=max(2, len(labels)))
    figure.suptitle("Paired GPT-BERT linguistic-repair dose response")
    figure.tight_layout(rect=(0, 0.12, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_task_delta_heatmap(paired_rows: list[dict[str, Any]], *, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    task_rows = [row for row in paired_rows if row["task"] != "macro"]
    learning_rates = sorted({row["learning_rate"] for row in task_rows}, key=lr_sort_key)
    columns = [
        (condition, learning_rate)
        for learning_rate in learning_rates
        for condition in CONDITION_ORDER[1:]
    ]
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in task_rows:
        grouped[(row["task"], row["condition"], row["learning_rate"])].append(
            row["difference"] * 100
        )
    matrix = np.full((len(TASK_ORDER), len(columns)), np.nan, dtype=float)
    for row_index, task in enumerate(TASK_ORDER):
        for column_index, (condition, learning_rate) in enumerate(columns):
            values = grouped[(task, condition, learning_rate)]
            if values:
                matrix[row_index, column_index] = mean(values)

    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(1.0, float(finite.max())) if finite.size else 1.0
    figure, axis = plt.subplots(figsize=(max(7, 1.5 * len(columns)), 5.2))
    image = axis.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_yticks(range(len(TASK_ORDER)), [task.upper() for task in TASK_ORDER])
    axis.set_xticks(
        range(len(columns)),
        [f"{CONDITION_LABEL[c]}\nLR {lr}" for c, lr in columns],
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            label = "—" if np.isnan(value) else f"{value:+.1f}"
            axis.text(column_index, row_index, label, ha="center", va="center", fontsize=9)
    axis.set_title("Mean paired task change relative to 0% control (percentage points)")
    figure.colorbar(image, ax=axis, label="Repair − control (points)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    macro_rows, task_rows = collect_results(experiment_root)
    if not macro_rows:
        raise SystemExit(f"No completed SuperGLUE summaries under {experiment_root / 'finetuning'}")

    expected_arms = len(CONDITION_ORDER) * 3 * 2
    if len(macro_rows) < expected_arms and not args.allow_incomplete:
        raise SystemExit(
            f"Found {len(macro_rows)}/{expected_arms} expected arms. "
            "Pass --allow-incomplete to summarize a running experiment."
        )

    summaries_dir = experiment_root / "summaries"
    figures_dir = experiment_root / "figures"
    paired_rows = paired_differences(macro_rows, task_rows)
    write_csv(
        summaries_dir / "macro_scores.csv",
        macro_rows,
        [
            "condition",
            "pretrain_seed",
            "finetune_seed",
            "learning_rate",
            "macro_primary",
            "summary_path",
        ],
    )
    write_csv(
        summaries_dir / "task_scores.csv",
        task_rows,
        [
            "condition",
            "pretrain_seed",
            "finetune_seed",
            "learning_rate",
            "task",
            "primary_metric",
            "primary_score",
            "summary_path",
        ],
    )
    write_csv(
        summaries_dir / "paired_differences.csv",
        paired_rows,
        [
            "condition",
            "pretrain_seed",
            "finetune_seed",
            "learning_rate",
            "task",
            "primary_metric",
            "control_score",
            "repair_score",
            "difference",
        ],
    )
    plot_dose_response(
        macro_rows,
        output_path=figures_dir / "superglue_dose_response.png",
        historical_baseline=args.historical_baseline,
    )
    plot_task_delta_heatmap(
        paired_rows,
        output_path=figures_dir / "superglue_task_delta_heatmap.png",
    )
    print(f"Completed arms: {len(macro_rows)}/{expected_arms}")
    print(f"Macro scores: {summaries_dir / 'macro_scores.csv'}")
    print(f"Paired differences: {summaries_dir / 'paired_differences.csv'}")
    print(f"Dose-response figure: {figures_dir / 'superglue_dose_response.png'}")


if __name__ == "__main__":
    main()
