from __future__ import annotations

import csv
from pathlib import Path

from babysteps.analysis.blimp_likelihood_dynamics import (
    build_panel_series,
    load_accuracy_rows,
    plot_group_grid,
    plot_overall_accuracy,
)
from babysteps.evaluation.blimp_likelihood import SUMMARY_FIELDNAMES


def write_summary(path: Path) -> None:
    rows = [
        {
            "run_name": "toy",
            "epoch": "1",
            "step": "10",
            "group_scope": "overall",
            "group_name": "all",
            "items": "100",
            "correct": "50",
            "score": "0.5",
            "mean_margin": "0.1",
            "ties": "0",
        },
        {
            "run_name": "toy",
            "epoch": "1",
            "step": "20",
            "group_scope": "overall",
            "group_name": "all",
            "items": "100",
            "correct": "75",
            "score": "0.75",
            "mean_margin": "0.2",
            "ties": "1",
        },
        {
            "run_name": "toy",
            "epoch": "1",
            "step": "20",
            "group_scope": "field",
            "group_name": "syntax",
            "items": "40",
            "correct": "30",
            "score": "0.75",
            "mean_margin": "0.3",
            "ties": "0",
        },
        {
            "run_name": "other",
            "epoch": "1",
            "step": "20",
            "group_scope": "overall",
            "group_name": "all",
            "items": "100",
            "correct": "20",
            "score": "0.2",
            "mean_margin": "0.0",
            "ties": "0",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_load_accuracy_rows_filters_overall_and_run_name(tmp_path: Path) -> None:
    summary = tmp_path / "likelihood_summary.csv"
    write_summary(summary)

    rows = load_accuracy_rows(summary, group_scope="overall", run_name="toy")

    assert [row["panel"] for row in rows] == ["average", "average"]
    assert [row["step"] for row in rows] == [10, 20]
    assert [row["score"] for row in rows] == [0.5, 0.75]


def test_load_accuracy_rows_keeps_requested_group_scope(tmp_path: Path) -> None:
    summary = tmp_path / "likelihood_summary.csv"
    write_summary(summary)

    rows = load_accuracy_rows(summary, group_scope="field", run_name="toy")

    assert len(rows) == 1
    assert rows[0]["panel"] == "syntax"
    assert rows[0]["score"] == 0.75


def test_build_panel_series_averages_duplicate_steps() -> None:
    rows = [
        {"panel": "average", "series": "toy", "step": 10, "score": 0.25},
        {"panel": "average", "series": "toy", "step": 10, "score": 0.75},
    ]

    panel_series = build_panel_series(rows)

    assert panel_series["average"]["toy"] == [(10, 0.5)]


def test_plot_overall_accuracy_writes_png_with_chance_baseline(tmp_path: Path) -> None:
    summary = tmp_path / "likelihood_summary.csv"
    output_path = tmp_path / "overall_accuracy.png"
    write_summary(summary)

    plot_overall_accuracy(
        summary_csvs={"BLiMP": summary},
        output_path=output_path,
        run_name="toy",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_group_grid_writes_png_with_chance_baseline(tmp_path: Path) -> None:
    output_path = tmp_path / "field_dynamics.png"

    plot_group_grid(
        rows=[
            {"panel": "syntax", "series": "toy", "step": 10, "score": 0.4},
            {"panel": "syntax", "series": "toy", "step": 20, "score": 0.7},
        ],
        output_path=output_path,
        title="BLiMP by Field",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
