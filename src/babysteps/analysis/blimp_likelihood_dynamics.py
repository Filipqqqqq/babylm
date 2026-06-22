"""Plot BLiMP-style sentence-likelihood scores over training steps."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUP_LABELS = {
    "overall": "Average",
    "field": "Field",
    "linguistics_term": "Linguistics Term",
    "uid": "UID",
    "contrast": "Contrast",
    "source_file": "Source File",
}

FILENAME_BY_GROUP = {
    "field": "field_dynamics.png",
    "linguistics_term": "linguistics_term_dynamics.png",
    "uid": "uid_dynamics.png",
    "contrast": "contrast_dynamics.png",
    "source_file": "source_file_dynamics.png",
}

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
    x_axis: str = "step",
) -> list[dict[str, Any]]:
    """Load step-indexed accuracy rows from a BLiMP likelihood summary CSV."""

    rows: list[dict[str, Any]] = []
    with Path(summary_csv).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if run_name and row.get("run_name") != run_name:
                continue
            if row.get("group_scope") != group_scope:
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

            rows.append(
                {
                    "panel": panel,
                    "series": series_label
                    or row.get("run_name")
                    or row.get("checkpoint")
                    or Path(summary_csv).parent.name,
                    "step": step,
                    "score": score,
                    "items": parse_optional_int(row.get("items")),
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


def plot_overall_accuracy(
    *,
    summary_csvs: dict[str, str | Path],
    output_path: str | Path,
    run_name: str | None = None,
    title: str = "BLiMP Likelihood Overall Accuracy",
    x_axis: str = "step",
) -> None:
    """Write a single overall-accuracy dynamics plot for one or more summaries."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.axhline(
        RANDOM_CHANCE_SCORE,
        color="#6f6f6f",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label="Random chance",
        zorder=0,
    )
    colors = ["#4c78a8", "#f28e2b", "#59a14f", "#b07aa1", "#e15759", "#76b7b2"]
    plotted = 0
    all_counts: dict[str, int] = {}
    for index, (label, summary_csv) in enumerate(summary_csvs.items()):
        rows = load_accuracy_rows(
            summary_csv,
            group_scope="overall",
            run_name=run_name,
            series_label=label,
            x_axis=x_axis,
        )
        counts = panel_item_counts(rows)
        all_counts[label] = counts.get("average", 0)
        panel_series = build_panel_series(rows)
        for series, points in sorted(panel_series.get("average", {}).items()):
            steps = [step for step, _ in points]
            scores = [score for _, score in points]
            series_count = counts.get("average")
            axis.plot(
                steps,
                scores,
                marker="o" if len(points) <= 40 else None,
                markersize=4,
                linewidth=2.0,
                color=colors[index % len(colors)],
                label=format_title(series, series_count),
            )
            plotted += 1

    if plotted == 0:
        raise ValueError("No step-indexed BLiMP overall rows found to plot.")

    single_count = next(iter(all_counts.values())) if len(all_counts) == 1 else None
    axis.set_title(format_title(title, single_count or None))
    axis.set_xlabel(x_axis_label(x_axis))
    axis.set_ylabel("Accuracy")
    axis.set_ylim(-0.02, 1.02)
    axis.set_yticks(SCORE_YTICKS)
    axis.grid(True, axis="both", alpha=0.22)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_group_grid(
    *,
    rows: list[dict[str, Any]],
    output_path: str | Path,
    title: str,
    ylabel: str = "Accuracy",
    x_axis: str = "step",
) -> None:
    """Write a small-multiple accuracy plot for one metadata grouping."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel_series = build_panel_series(rows)
    counts = panel_item_counts(rows)
    panels = sorted(panel_series)
    if not panels:
        raise ValueError("No BLiMP group rows found to plot.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ncols = min(4, max(1, len(panels)))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.0 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_flat = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    colors = [
        "#4c78a8",
        "#f28e2b",
        "#59a14f",
        "#b07aa1",
        "#e15759",
        "#76b7b2",
        "#edc948",
        "#9c755f",
    ]
    all_series = sorted(
        {series for series_map in panel_series.values() for series in series_map}
    )
    color_for = {
        series: colors[index % len(colors)]
        for index, series in enumerate(all_series)
    }

    for axis, panel in zip(axes_flat, panels, strict=False):
        axis.axhline(
            RANDOM_CHANCE_SCORE,
            color="#6f6f6f",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label="Random chance",
            zorder=0,
        )
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
        axis.set_ylim(-0.02, 1.02)
        axis.set_yticks(SCORE_YTICKS)
        axis.grid(True, axis="both", alpha=0.22)

    for axis in axes_flat[len(panels) :]:
        axis.axis("off")

    for axis in axes_flat[-ncols:]:
        axis.set_xlabel(x_axis_label(x_axis))
    for row_start in range(0, len(axes_flat), ncols):
        axes_flat[row_start].set_ylabel(ylabel)

    handles, labels = unique_legend_entries(axes_flat[: len(panels)])

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


def plot_blimp_likelihood_summary(
    *,
    summary_csv: str | Path,
    output_dir: str | Path,
    benchmark_label: str,
    run_name: str | None = None,
    group_scopes: tuple[str, ...] = ("field", "linguistics_term"),
    x_axis: str = "step",
) -> None:
    """Write the standard plots for one BLiMP-like benchmark summary."""

    output_dir = Path(output_dir)
    plot_overall_accuracy(
        summary_csvs={benchmark_label: summary_csv},
        output_path=output_dir / "overall_accuracy.png",
        run_name=run_name,
        title=f"{benchmark_label} Overall Accuracy",
        x_axis=x_axis,
    )
    for group_scope in group_scopes:
        rows = load_accuracy_rows(
            summary_csv,
            group_scope=group_scope,
            run_name=run_name,
            x_axis=x_axis,
        )
        if not rows:
            continue
        plot_group_grid(
            rows=rows,
            output_path=output_dir / FILENAME_BY_GROUP.get(
                group_scope,
                f"{group_scope}_dynamics.png",
            ),
            title=f"{benchmark_label} by {GROUP_LABELS.get(group_scope, display_label(group_scope))}",
            x_axis=x_axis,
        )


__all__ = [
    "build_panel_series",
    "load_accuracy_rows",
    "panel_item_counts",
    "plot_blimp_likelihood_summary",
    "plot_group_grid",
    "plot_overall_accuracy",
]
