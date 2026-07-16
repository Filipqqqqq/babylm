"""Plot COMPS continuation-likelihood scores over training steps."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SUBTASK_ORDER = (
    "base",
    "wugs",
    "wugs_dist_before",
    "wugs_dist_in_between",
)
NEGATIVE_SAMPLE_TYPES = (
    "taxonomic",
    "overlap",
    "co-occurrence",
    "random",
)
RANDOM_CHANCE_SCORE = 0.5
SCORE_YTICKS = tuple(index / 10 for index in range(0, 11))
X_AXIS_LABELS = {
    "step": "Training step",
    "tokens_seen": "Training tokens seen",
    "words_seen": "Training words seen",
}


def parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_x_value(value: str | None, *, x_axis: str) -> float | int | None:
    if x_axis == "step":
        return parse_optional_int(value)
    return parse_optional_float(value)


def x_axis_label(x_axis: str) -> str:
    return X_AXIS_LABELS.get(x_axis, display_label(x_axis))


def display_label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def format_title(label: str, item_count: int | None = None) -> str:
    if item_count is None:
        return label
    return f"{label} (n={item_count:,})"


def load_accuracy_rows(
    summary_csv: str | Path,
    *,
    group_scope: str,
    run_name: str | None = None,
    series_label: str | None = None,
    score_reduction: str | None = None,
    x_axis: str = "step",
) -> list[dict[str, Any]]:
    """Load step-indexed accuracy rows from a COMPS summary CSV."""

    rows: list[dict[str, Any]] = []
    with Path(summary_csv).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if run_name and row.get("run_name") != run_name:
                continue
            if row.get("group_scope") != group_scope:
                continue
            row_score_reduction = row.get("score_reduction", "")
            if score_reduction and row_score_reduction != score_reduction:
                continue
            if group_scope == "overall" and row.get("group_name") != "all":
                continue

            step = parse_x_value(row.get(x_axis), x_axis=x_axis)
            score = parse_optional_float(row.get("score"))
            if step is None or score is None:
                continue

            panel = "average" if group_scope == "overall" else str(row.get("group_name", ""))
            if not panel:
                continue
            series = (
                series_label
                or row.get("run_name")
                or row.get("checkpoint")
                or Path(summary_csv).parent.name
            )
            if row_score_reduction and score_reduction is None:
                series = f"{series} ({row_score_reduction})"
            rows.append(
                {
                    "panel": panel,
                    "series": series,
                    "step": step,
                    "score": score,
                    "items": parse_optional_int(row.get("items")),
                    "score_reduction": row_score_reduction,
                    "summary_csv": str(summary_csv),
                }
            )
    return rows


def build_panel_series(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Group rows into panel/series lists, averaging duplicate step entries."""

    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["panel"], row["series"], row["step"])].append(float(row["score"]))

    panel_series: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (panel, series, step), values in sorted(grouped.items()):
        panel_series[panel][series].append((step, statistics.mean(values)))
    return {panel: dict(series_map) for panel, series_map in panel_series.items()}


def panel_item_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Return one representative item count per panel."""

    counts: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row.get("items") is not None:
            counts[str(row["panel"])].append(int(row["items"]))
    return {
        panel: max(values)
        for panel, values in counts.items()
        if values
    }


def unique_legend_entries(axes: list[Any]) -> tuple[list[Any], list[str]]:
    """Collect de-duplicated legend entries from a list of axes."""

    by_label: dict[str, Any] = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            by_label.setdefault(label, handle)
    return list(by_label.values()), list(by_label)


def add_chance_line(axis: Any) -> None:
    axis.axhline(
        RANDOM_CHANCE_SCORE,
        color="#6f6f6f",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label="Random chance",
        zorder=0,
    )


def style_score_axis(axis: Any, *, ylabel: str | None = None) -> None:
    axis.set_ylim(-0.02, 1.02)
    axis.set_yticks(SCORE_YTICKS)
    axis.grid(True, axis="both", alpha=0.22)
    if ylabel:
        axis.set_ylabel(ylabel)


def plot_overall_accuracy(
    *,
    summary_csv: str | Path,
    output_path: str | Path,
    run_name: str | None = None,
    score_reduction: str | None = None,
    title: str = "COMPS Macro Accuracy",
    x_axis: str = "step",
) -> None:
    """Write the official-style COMPS macro accuracy dynamics plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_accuracy_rows(
        summary_csv,
        group_scope="overall",
        run_name=run_name,
        score_reduction=score_reduction,
        x_axis=x_axis,
    )
    panel_series = build_panel_series(rows)
    counts = panel_item_counts(rows)
    points_by_series = panel_series.get("average", {})
    if not points_by_series:
        raise ValueError("No step-indexed COMPS overall rows found to plot.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(9, 4.8))
    add_chance_line(axis)
    for series, points in sorted(points_by_series.items()):
        steps = [step for step, _ in points]
        scores = [score for _, score in points]
        axis.plot(
            steps,
            scores,
            marker="o" if len(points) <= 40 else None,
            markersize=4,
            linewidth=2.0,
            label=series,
        )
    axis.set_title(format_title(title, counts.get("average")))
    axis.set_xlabel(x_axis_label(x_axis))
    style_score_axis(axis, ylabel="Accuracy")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_group_grid(
    *,
    summary_csv: str | Path,
    output_path: str | Path,
    group_scope: str,
    panel_order: tuple[str, ...],
    title: str,
    run_name: str | None = None,
    score_reduction: str | None = None,
    x_axis: str = "step",
) -> None:
    """Write a 2x2 COMPS diagnostic grid for one metadata grouping."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_accuracy_rows(
        summary_csv,
        group_scope=group_scope,
        run_name=run_name,
        score_reduction=score_reduction,
        x_axis=x_axis,
    )
    panel_series = build_panel_series(rows)
    counts = panel_item_counts(rows)
    panels = [panel for panel in panel_order if panel in panel_series]
    panels.extend(panel for panel in sorted(panel_series) if panel not in panels)
    if not panels:
        raise ValueError(f"No step-indexed COMPS {group_scope} rows found to plot.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True, sharey=True)
    axes_flat = list(axes.ravel())

    all_series = sorted(
        {series for series_map in panel_series.values() for series in series_map}
    )
    colors = ["#4c78a8", "#f28e2b", "#59a14f", "#b07aa1", "#e15759", "#76b7b2"]
    color_for = {
        series: colors[index % len(colors)]
        for index, series in enumerate(all_series)
    }

    for axis, panel in zip(axes_flat, panels[:4], strict=False):
        add_chance_line(axis)
        for series, points in sorted(panel_series[panel].items()):
            steps = [step for step, _ in points]
            scores = [score for _, score in points]
            axis.plot(
                steps,
                scores,
                marker="o" if len(points) <= 40 else None,
                markersize=3.5,
                linewidth=1.8,
                color=color_for[series],
                label=series,
            )
        axis.set_title(format_title(display_label(panel), counts.get(panel)), fontsize=10)
        style_score_axis(axis)

    for axis in axes_flat[len(panels[:4]) :]:
        axis.axis("off")

    for axis in axes[-1]:
        axis.set_xlabel(x_axis_label(x_axis))
    for axis in axes[:, 0]:
        axis.set_ylabel("Accuracy")

    handles, labels = unique_legend_entries(axes_flat[: len(panels[:4])])
    fig.suptitle(title, fontsize=14, y=0.995)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=min(len(labels), 4),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_comps_summary(
    *,
    summary_csv: str | Path,
    output_dir: str | Path,
    run_name: str | None = None,
    score_reduction: str | None = None,
    x_axis: str = "step",
) -> None:
    """Write the standard COMPS dynamics plots."""

    output_dir = Path(output_dir)
    plot_overall_accuracy(
        summary_csv=summary_csv,
        output_path=output_dir / "overall_accuracy.png",
        run_name=run_name,
        score_reduction=score_reduction,
        x_axis=x_axis,
    )
    plot_group_grid(
        summary_csv=summary_csv,
        output_path=output_dir / "subtask_dynamics.png",
        group_scope="subtask",
        panel_order=SUBTASK_ORDER,
        title="COMPS by Subtask",
        run_name=run_name,
        score_reduction=score_reduction,
        x_axis=x_axis,
    )
    plot_group_grid(
        summary_csv=summary_csv,
        output_path=output_dir / "negative_sample_type_dynamics.png",
        group_scope="negative_sample_type",
        panel_order=NEGATIVE_SAMPLE_TYPES,
        title="COMPS by Negative Sample Type",
        run_name=run_name,
        score_reduction=score_reduction,
        x_axis=x_axis,
    )


__all__ = [
    "NEGATIVE_SAMPLE_TYPES",
    "RANDOM_CHANCE_SCORE",
    "SCORE_YTICKS",
    "SUBTASK_ORDER",
    "build_panel_series",
    "load_accuracy_rows",
    "panel_item_counts",
    "plot_comps_summary",
    "plot_group_grid",
    "plot_overall_accuracy",
]
