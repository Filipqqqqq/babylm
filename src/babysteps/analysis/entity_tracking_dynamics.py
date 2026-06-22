"""Plot Entity Tracking likelihood scores over training steps."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ENTITY_FAMILIES = ("regular", "ambiref", "move_contents")
NUMOPS = tuple(str(index) for index in range(6))
RANDOM_CHANCE_SCORE = 0.2
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
    """Load step-indexed accuracy rows from an Entity Tracking summary CSV."""

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
    title: str = "Entity Tracking Overall Accuracy",
    x_axis: str = "step",
) -> None:
    """Write the overall Entity Tracking accuracy plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_accuracy_rows(
        summary_csv,
        group_scope="overall",
        run_name=run_name,
        x_axis=x_axis,
    )
    panel_series = build_panel_series(rows)
    counts = panel_item_counts(rows)
    points_by_series = panel_series.get("average", {})
    if not points_by_series:
        raise ValueError("No step-indexed Entity Tracking overall rows found to plot.")

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


def plot_family_dynamics(
    *,
    summary_csv: str | Path,
    output_path: str | Path,
    run_name: str | None = None,
    title: str = "Entity Tracking by Family",
    x_axis: str = "step",
) -> None:
    """Write one plot with a curve for each Entity Tracking family."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_accuracy_rows(
        summary_csv,
        group_scope="family",
        run_name=run_name,
        x_axis=x_axis,
    )
    panel_series = build_panel_series(rows)
    counts = panel_item_counts(rows)
    if not panel_series:
        raise ValueError("No step-indexed Entity Tracking family rows found to plot.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(9, 4.8))
    add_chance_line(axis)
    for family in ENTITY_FAMILIES:
        series_map = panel_series.get(family)
        if not series_map:
            continue
        for _, points in sorted(series_map.items()):
            steps = [step for step, _ in points]
            scores = [score for _, score in points]
            axis.plot(
                steps,
                scores,
                marker="o" if len(points) <= 40 else None,
                markersize=4,
                linewidth=2.0,
                label=format_title(display_label(family), counts.get(family)),
            )
    axis.set_title(title)
    axis.set_xlabel(x_axis_label(x_axis))
    style_score_axis(axis, ylabel="Accuracy")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_family_numops_grid(
    *,
    summary_csv: str | Path,
    output_path: str | Path,
    run_name: str | None = None,
    title: str = "Entity Tracking by Family and Operation Count",
    x_axis: str = "step",
) -> None:
    """Write a 3x6 family-by-numops small-multiple plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_accuracy_rows(
        summary_csv,
        group_scope="uid",
        run_name=run_name,
        x_axis=x_axis,
    )
    panel_series = build_panel_series(rows)
    counts = panel_item_counts(rows)
    if not panel_series:
        raise ValueError("No step-indexed Entity Tracking uid rows found to plot.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    families = [family for family in ENTITY_FAMILIES if any(panel.startswith(family) for panel in panel_series)]
    if not families:
        families = sorted({panel.rsplit("_", 2)[0] for panel in panel_series})

    fig, axes = plt.subplots(
        len(families),
        len(NUMOPS),
        figsize=(18, 3.2 * len(families)),
        sharex=True,
        sharey=True,
    )
    if len(families) == 1:
        axes_flat = [axes[index] for index in range(len(NUMOPS))]
        axes_grid = [axes_flat]
    else:
        axes_grid = axes
        axes_flat = list(axes.ravel())

    for row_index, family in enumerate(families):
        for col_index, numops in enumerate(NUMOPS):
            axis = axes_grid[row_index][col_index]
            panel = f"{family}_{numops}_ops"
            add_chance_line(axis)
            series_map = panel_series.get(panel, {})
            for series, points in sorted(series_map.items()):
                steps = [step for step, _ in points]
                scores = [score for _, score in points]
                axis.plot(
                    steps,
                    scores,
                    marker="o" if len(points) <= 40 else None,
                    markersize=3.5,
                    linewidth=1.8,
                    label=series,
                )
            axis.set_title(
                format_title(f"{display_label(family)} {numops} Ops", counts.get(panel)),
                fontsize=9,
            )
            style_score_axis(axis)
            if not series_map:
                axis.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    color="#777777",
                )

    for axis in axes_grid[-1]:
        axis.set_xlabel(x_axis_label(x_axis))
    for row_axes in axes_grid:
        row_axes[0].set_ylabel("Accuracy")

    handles, labels = unique_legend_entries(axes_flat)
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


def plot_entity_tracking_summary(
    *,
    summary_csv: str | Path,
    output_dir: str | Path,
    run_name: str | None = None,
    x_axis: str = "step",
) -> None:
    """Write the standard Entity Tracking dynamics plots."""

    output_dir = Path(output_dir)
    plot_overall_accuracy(
        summary_csv=summary_csv,
        output_path=output_dir / "overall_accuracy.png",
        run_name=run_name,
        x_axis=x_axis,
    )
    plot_family_dynamics(
        summary_csv=summary_csv,
        output_path=output_dir / "family_dynamics.png",
        run_name=run_name,
        x_axis=x_axis,
    )
    plot_family_numops_grid(
        summary_csv=summary_csv,
        output_path=output_dir / "family_numops_dynamics.png",
        run_name=run_name,
        x_axis=x_axis,
    )


__all__ = [
    "build_panel_series",
    "load_accuracy_rows",
    "panel_item_counts",
    "plot_entity_tracking_summary",
    "plot_family_dynamics",
    "plot_family_numops_grid",
    "plot_overall_accuracy",
]
