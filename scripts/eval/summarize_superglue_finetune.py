#!/usr/bin/env python3
"""Summarize BabyLM SuperGLUE fine-tuning results for one checkpoint.

The official BabyLM fine-tuning runner writes one ``results.txt`` per task.
This helper collects those task metrics and writes a compact aggregate summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_TASKS = ("boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli")
PRIMARY_METRIC = {
    "boolq": "accuracy",
    "multirc": "accuracy",
    "rte": "accuracy",
    "wsc": "accuracy",
    "mrpc": "f1",
    "qqp": "f1",
    "mnli": "accuracy",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build macro summaries from BabyLM SuperGLUE finetune outputs."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="The results_dir passed to evaluation_pipeline.finetune.run.",
    )
    parser.add_argument(
        "--model-name",
        default="final",
        help="Model subdirectory written by the BabyLM runner. Local checkpoint dirs usually use 'final'.",
    )
    parser.add_argument(
        "--revision-name",
        default="main",
        help="Revision subdirectory written by the BabyLM runner.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Tasks to include in the aggregate.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Where to write the JSON summary. Defaults to <results-dir>/superglue_summary.json.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Where to write per-task CSV rows. Defaults to <results-dir>/superglue_summary.csv.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Summarize completed tasks instead of failing when a task result is missing.",
    )
    return parser.parse_args()


def _read_results(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    args = _parse_args()
    output_json = args.output_json or args.results_dir / "superglue_summary.json"
    output_csv = args.output_csv or args.results_dir / "superglue_summary.csv"

    task_rows = []
    missing = []
    primary_scores = []
    accuracy_scores = []

    for task in args.tasks:
        result_path = (
            args.results_dir
            / args.model_name
            / args.revision_name
            / "finetune"
            / task
            / "results.txt"
        )
        if not result_path.exists():
            missing.append(str(result_path))
            continue

        metrics = _read_results(result_path)
        primary_metric = PRIMARY_METRIC.get(task, "accuracy")
        primary_score = metrics.get(primary_metric)
        if primary_score is None:
            missing.append(f"{result_path} missing metric {primary_metric!r}")
            continue

        primary_scores.append(primary_score)
        accuracy = metrics.get("accuracy")
        if accuracy is not None:
            accuracy_scores.append(accuracy)

        task_rows.append(
            {
                "task": task,
                "primary_metric": primary_metric,
                "primary_score": primary_score,
                "accuracy": accuracy,
                "f1": metrics.get("f1"),
                "mcc": metrics.get("mcc"),
                "results_path": str(result_path),
            }
        )

    if missing and not args.allow_missing:
        raise SystemExit("Missing SuperGLUE result(s):\n" + "\n".join(missing))

    summary = {
        "results_dir": str(args.results_dir),
        "model_name": args.model_name,
        "revision_name": args.revision_name,
        "tasks_requested": list(args.tasks),
        "tasks_completed": [row["task"] for row in task_rows],
        "missing": missing,
        "macro_primary": _mean(primary_scores),
        "macro_accuracy": _mean(accuracy_scores),
        "primary_metric_by_task": PRIMARY_METRIC,
        "task_results": task_rows,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "primary_metric",
                "primary_score",
                "accuracy",
                "f1",
                "mcc",
                "results_path",
            ],
        )
        writer.writeheader()
        writer.writerows(task_rows)

    macro_primary = summary["macro_primary"]
    macro_accuracy = summary["macro_accuracy"]
    print(f"tasks_completed={len(task_rows)}/{len(args.tasks)}")
    if macro_primary is not None:
        print(f"macro_primary={macro_primary:.6f}")
    if macro_accuracy is not None:
        print(f"macro_accuracy={macro_accuracy:.6f}")
    print(f"summary_json={output_json}")
    print(f"summary_csv={output_csv}")


if __name__ == "__main__":
    main()
