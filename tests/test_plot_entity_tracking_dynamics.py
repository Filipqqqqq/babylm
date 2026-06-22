from __future__ import annotations

import csv
from pathlib import Path

from babysteps.analysis.entity_tracking_dynamics import (
    build_panel_series,
    load_accuracy_rows,
    panel_item_counts,
    plot_entity_tracking_summary,
)
from babysteps.evaluation.entity_tracking_likelihood import SUMMARY_FIELDNAMES


def write_summary(path: Path) -> None:
    rows = [
        {
            "run_name": "toy",
            "epoch": "1",
            "step": "10",
            "group_scope": "overall",
            "group_name": "all",
            "items": "12",
            "correct": "3",
            "score": "0.25",
            "mean_margin": "0.1",
            "ties": "0",
        },
        {
            "run_name": "toy",
            "epoch": "1",
            "step": "20",
            "group_scope": "overall",
            "group_name": "all",
            "items": "12",
            "correct": "6",
            "score": "0.5",
            "mean_margin": "0.2",
            "ties": "1",
        },
        {
            "run_name": "toy",
            "epoch": "1",
            "step": "20",
            "group_scope": "family",
            "group_name": "regular",
            "items": "6",
            "correct": "3",
            "score": "0.5",
            "mean_margin": "0.2",
            "ties": "0",
        },
        {
            "run_name": "toy",
            "epoch": "1",
            "step": "20",
            "group_scope": "uid",
            "group_name": "regular_0_ops",
            "items": "3",
            "correct": "2",
            "score": "0.6666666667",
            "mean_margin": "0.3",
            "ties": "0",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_load_accuracy_rows_and_item_counts(tmp_path: Path) -> None:
    summary = tmp_path / "likelihood_summary.csv"
    write_summary(summary)

    rows = load_accuracy_rows(summary, group_scope="overall", run_name="toy")

    assert [row["panel"] for row in rows] == ["average", "average"]
    assert panel_item_counts(rows) == {"average": 12}


def test_build_panel_series_averages_duplicate_steps() -> None:
    rows = [
        {"panel": "average", "series": "toy", "step": 10, "score": 0.2},
        {"panel": "average", "series": "toy", "step": 10, "score": 0.4},
    ]

    panel_series = build_panel_series(rows)

    assert panel_series["average"]["toy"][0][0] == 10
    assert panel_series["average"]["toy"][0][1] == 0.30000000000000004


def test_plot_entity_tracking_summary_writes_standard_plots(tmp_path: Path) -> None:
    summary = tmp_path / "likelihood_summary.csv"
    output_dir = tmp_path / "plots"
    write_summary(summary)

    plot_entity_tracking_summary(
        summary_csv=summary,
        output_dir=output_dir,
        run_name="toy",
    )

    for filename in (
        "overall_accuracy.png",
        "family_dynamics.png",
        "family_numops_dynamics.png",
    ):
        path = output_dir / filename
        assert path.exists()
        assert path.stat().st_size > 0
