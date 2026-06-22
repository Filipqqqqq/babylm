"""Plot EWoK context-sensitivity scores over training steps."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


EWOK_DOMAINS = (
    "agent-properties",
    "material-dynamics",
    "material-properties",
    "physical-dynamics",
    "physical-interactions",
    "physical-relations",
    "quantitative-properties",
    "social-interactions",
    "social-properties",
    "social-relations",
    "spatial-relations",
)

PANEL_ORDER = (*EWOK_DOMAINS, "average")

PANEL_LABELS = {
    "agent-properties": "Agent Properties",
    "material-dynamics": "Material Dynamics",
    "material-properties": "Material Properties",
    "physical-dynamics": "Physical Dynamics",
    "physical-interactions": "Physical Interactions",
    "physical-relations": "Physical Relations",
    "quantitative-properties": "Quantitative Properties",
    "social-interactions": "Social Interactions",
    "social-properties": "Social Properties",
    "social-relations": "Social Relations",
    "spatial-relations": "Spatial Relations",
    "average": "Average",
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


def display_label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def x_axis_label(x_axis: str) -> str:
    return X_AXIS_LABELS.get(x_axis, display_label(x_axis))


def load_domain_rows(
    summary_csv: str | Path,
    *,
    run_name: str | None = None,
    score_reduction: str | None = None,
    x_axis: str = "step",
) -> list[dict[str, Any]]:
    """Load domain and average rows from ``ewok_context_summary.csv``."""

    rows: list[dict[str, Any]] = []
    with Path(summary_csv).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if run_name and row.get("run_name") != run_name:
                continue
            if score_reduction and row.get("score_reduction") != score_reduction:
                continue

            step = parse_x_value(row.get(x_axis), x_axis=x_axis)
            score = parse_optional_float(row.get("score"))
            if step is None or score is None:
                continue

            group_scope = row.get("group_scope", "")
            group_name = row.get("group_name", "")
            if group_scope == "domain" and group_name in EWOK_DOMAINS:
                panel = group_name
            elif group_scope == "overall" and group_name == "all":
                panel = "average"
            else:
                continue

            rows.append(
                {
                    "panel": panel,
                    "series": row.get("run_name") or row.get("checkpoint") or "score",
                    "step": step,
                    "score": score,
                    "items": parse_optional_int(row.get("items")),
                    "score_reduction": row.get("score_reduction", ""),
                    "summary_csv": str(summary_csv),
                }
            )

    if not rows:
        raise ValueError(f"No step-indexed EWoK context summary rows found in {summary_csv}.")
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


def format_panel_title(label: str, item_count: int | None = None) -> str:
    if item_count is None:
        return label
    return f"{label} (n={item_count:,})"


def unique_legend_entries(axes: list[Any]) -> tuple[list[Any], list[str]]:
    """Collect de-duplicated legend entries from a list of axes."""

    by_label: dict[str, Any] = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            by_label.setdefault(label, handle)
    return list(by_label.values()), list(by_label)


def plot_domain_grid(
    *,
    panel_series: dict[str, dict[str, list[tuple[int, float]]]],
    output_path: str | Path,
    panel_counts: dict[str, int] | None = None,
    title: str = "EWoK Context Sensitivity by Domain",
    x_axis: str = "step",
) -> None:
    """Write the 4x3 EWoK context-sensitivity dynamics plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True, sharey=True)
    axes_flat = list(axes.ravel())
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

    for axis, panel in zip(axes_flat, PANEL_ORDER, strict=True):
        series_map = panel_series.get(panel, {})
        axis.axhline(
            RANDOM_CHANCE_SCORE,
            color="#6f6f6f",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label="Random chance",
            zorder=0,
        )
        for series, points in sorted(series_map.items()):
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
        axis.set_title(
            format_panel_title(PANEL_LABELS[panel], (panel_counts or {}).get(panel)),
            fontsize=10,
        )
        axis.set_ylim(-0.02, 1.02)
        axis.set_yticks(SCORE_YTICKS)
        axis.grid(True, axis="both", alpha=0.22)
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

    for axis in axes[-1, :]:
        axis.set_xlabel(x_axis_label(x_axis))
    for axis in axes[:, 0]:
        axis.set_ylabel("Context sensitivity score")

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
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_ewok_context_summary(
    *,
    summary_csv: str | Path,
    output_path: str | Path,
    run_name: str | None = None,
    score_reduction: str | None = "sum",
    title: str = "EWoK Context Sensitivity by Domain",
    x_axis: str = "step",
) -> None:
    """Load an EWoK context summary CSV and write the 4x3 dynamics plot."""

    rows = load_domain_rows(
        summary_csv,
        run_name=run_name,
        score_reduction=score_reduction,
        x_axis=x_axis,
    )
    plot_domain_grid(
        panel_series=build_panel_series(rows),
        output_path=output_path,
        panel_counts=panel_item_counts(rows),
        title=title,
        x_axis=x_axis,
    )
