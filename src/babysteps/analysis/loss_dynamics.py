"""Plot training loss curves from metrics_history.json files."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


LOSS_LABELS = {
    "loss": "Training Loss",
    "ce_loss": "Training CE Loss",
    "train_causal_loss": "GPT-BERT Causal Training Loss",
    "train_causal_ce_loss": "GPT-BERT Causal Training CE Loss",
    "train_mntp_loss": "GPT-BERT MNTP Training Loss",
    "train_mntp_ce_loss": "GPT-BERT MNTP Training CE Loss",
}
DEFAULT_LOSS_KEYS = (
    "loss",
    "ce_loss",
    "train_causal_loss",
    "train_mntp_loss",
)
X_AXIS_LABELS = {
    "step": "Optimizer update step",
    "tokens_seen": "Training tokens seen",
    "words_seen": "Training words seen",
}


def display_label(value: str) -> str:
    """Return a compact display label for config-style names."""

    return value.replace("_", " ").replace("-", " ").title()


def metric_label(metric_name: str) -> str:
    """Return a human-readable label for a loss metric key."""

    return LOSS_LABELS.get(metric_name, display_label(metric_name))


def default_run_label(metrics_path: str | Path) -> str:
    """Infer a readable run label from a metrics file path."""

    path = Path(metrics_path)
    if path.name == "metrics_history.json" and path.parent.name:
        return path.parent.name
    return path.stem


def load_metrics_history(metrics_path: str | Path) -> list[dict[str, Any]]:
    """Load a metrics_history.json list."""

    with Path(metrics_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {metrics_path}.")
    return data


def parse_optional_float(value: Any) -> float | None:
    """Return a finite float value or None."""

    if value is None or value == "":
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def load_training_loss_rows(
    metrics_paths: list[str | Path],
    *,
    labels: list[str] | None = None,
    loss_keys: tuple[str, ...] = DEFAULT_LOSS_KEYS,
    x_axis: str = "words_seen",
) -> list[dict[str, Any]]:
    """Load train loss rows from one or more metrics_history.json files."""

    if labels is not None and len(labels) != len(metrics_paths):
        raise ValueError("labels must have the same length as metrics_paths.")
    if x_axis not in X_AXIS_LABELS:
        raise ValueError(f"x_axis must be one of {sorted(X_AXIS_LABELS)}, got {x_axis!r}.")

    rows: list[dict[str, Any]] = []
    for path_index, metrics_path in enumerate(metrics_paths):
        series = labels[path_index] if labels is not None else default_run_label(metrics_path)
        for record in load_metrics_history(metrics_path):
            if not isinstance(record, dict) or record.get("type") != "train":
                continue
            x_value = parse_optional_float(record.get(x_axis))
            if x_value is None:
                continue
            step = parse_optional_float(record.get("step"))
            for loss_key in loss_keys:
                loss_value = parse_optional_float(record.get(loss_key))
                if loss_value is None:
                    continue
                rows.append(
                    {
                        "series": series,
                        "metric": loss_key,
                        "x": x_value,
                        "step": step,
                        "loss": loss_value,
                        "metrics_path": str(metrics_path),
                    }
                )
    return rows


def group_rows_by_metric(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group loaded rows by metric key."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["metric"])].append(row)
    return dict(grouped)


def plot_training_loss_rows(
    rows: list[dict[str, Any]],
    *,
    output_path: str | Path,
    metric_name: str,
    x_axis: str = "words_seen",
    title: str | None = None,
) -> None:
    """Write one training-loss dynamics plot."""

    if not rows:
        raise ValueError(f"No rows found for metric {metric_name!r}.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_series[str(row["series"])].append(row)

    fig, axis = plt.subplots(figsize=(9, 4.8))
    for series, series_rows in sorted(by_series.items()):
        series_rows = sorted(series_rows, key=lambda row: (float(row["x"]), float(row["step"] or 0.0)))
        axis.plot(
            [float(row["x"]) for row in series_rows],
            [float(row["loss"]) for row in series_rows],
            marker="o" if len(series_rows) <= 60 else None,
            markersize=3.5,
            linewidth=1.8,
            label=series,
        )

    axis.set_title(title or metric_label(metric_name))
    axis.set_xlabel(X_AXIS_LABELS.get(x_axis, display_label(x_axis)))
    axis.set_ylabel("Loss")
    axis.grid(True, axis="both", alpha=0.22)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_training_loss_plots(
    *,
    metrics_paths: list[str | Path],
    output_dir: str | Path,
    labels: list[str] | None = None,
    loss_keys: tuple[str, ...] = DEFAULT_LOSS_KEYS,
    x_axis: str = "words_seen",
) -> list[Path]:
    """Write one plot per available loss metric."""

    rows = load_training_loss_rows(
        metrics_paths,
        labels=labels,
        loss_keys=loss_keys,
        x_axis=x_axis,
    )
    grouped = group_rows_by_metric(rows)
    output_dir = Path(output_dir)
    written: list[Path] = []
    for metric_name in loss_keys:
        metric_rows = grouped.get(metric_name, [])
        if not metric_rows:
            continue
        output_path = output_dir / f"{metric_name}_by_{x_axis}.png"
        plot_training_loss_rows(
            metric_rows,
            output_path=output_path,
            metric_name=metric_name,
            x_axis=x_axis,
        )
        written.append(output_path)

    if not written:
        raise ValueError(
            "No training loss rows found to plot. Check that metrics include "
            f"{x_axis!r} and at least one of {loss_keys}."
        )
    return written


__all__ = [
    "DEFAULT_LOSS_KEYS",
    "LOSS_LABELS",
    "X_AXIS_LABELS",
    "default_run_label",
    "group_rows_by_metric",
    "load_metrics_history",
    "load_training_loss_rows",
    "plot_training_loss_rows",
    "write_training_loss_plots",
]
