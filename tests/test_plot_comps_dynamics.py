from __future__ import annotations

import csv
from pathlib import Path

from babysteps.analysis.comps_dynamics import (
    build_panel_series,
    load_accuracy_rows,
    panel_item_counts,
    plot_comps_summary,
)
from babysteps.evaluation.comps_likelihood import SUMMARY_FIELDNAMES


def write_summary(path: Path) -> None:
    rows = [
        {
            "run_name": "toy",
            "epoch": "1",
            "step": "10",
            "tokens_seen": "1000",
            "words_seen": "600",
            "group_scope": "overall",
            "group_name": "all",
            "items": "20",
            "correct": "10",
            "score": "0.5",
            "mean_margin": "0.0",
            "ties": "0",
            "score_kind": "subtask_macro",
            "micro_score": "0.5",
        },
        {
            "run_name": "toy",
            "epoch": "1",
            "step": "20",
            "tokens_seen": "2000",
            "words_seen": "1200",
            "group_scope": "overall",
            "group_name": "all",
            "items": "20",
            "correct": "12",
            "score": "0.6",
            "mean_margin": "0.1",
            "ties": "1",
            "score_kind": "subtask_macro",
            "micro_score": "0.6",
        },
    ]
    for subtask in ("base", "wugs", "wugs_dist_before", "wugs_dist_in_between"):
        rows.append(
            {
                "run_name": "toy",
                "epoch": "1",
                "step": "20",
                "tokens_seen": "2000",
                "words_seen": "1200",
                "group_scope": "subtask",
                "group_name": subtask,
                "items": "5",
                "correct": "3",
                "score": "0.6",
                "mean_margin": "0.1",
                "ties": "0",
                "score_kind": "micro",
                "micro_score": "0.6",
            }
        )
    for negative_sample_type in ("taxonomic", "overlap", "co-occurrence", "random"):
        rows.append(
            {
                "run_name": "toy",
                "epoch": "1",
                "step": "20",
                "tokens_seen": "2000",
                "words_seen": "1200",
                "group_scope": "negative_sample_type",
                "group_name": negative_sample_type,
                "items": "5",
                "correct": "3",
                "score": "0.6",
                "mean_margin": "0.1",
                "ties": "0",
                "score_kind": "micro",
                "micro_score": "0.6",
            }
        )

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
    assert panel_item_counts(rows) == {"average": 20}


def test_load_accuracy_rows_can_use_words_seen_x_axis(tmp_path: Path) -> None:
    summary = tmp_path / "likelihood_summary.csv"
    write_summary(summary)

    rows = load_accuracy_rows(
        summary,
        group_scope="overall",
        run_name="toy",
        x_axis="words_seen",
    )

    assert [row["step"] for row in rows] == [600.0, 1200.0]


def test_build_panel_series_averages_duplicate_steps() -> None:
    rows = [
        {"panel": "average", "series": "toy", "step": 10, "score": 0.2},
        {"panel": "average", "series": "toy", "step": 10, "score": 0.4},
    ]

    panel_series = build_panel_series(rows)

    assert panel_series["average"]["toy"][0][0] == 10
    assert panel_series["average"]["toy"][0][1] == 0.30000000000000004


def test_plot_comps_summary_writes_standard_plots(tmp_path: Path) -> None:
    summary = tmp_path / "likelihood_summary.csv"
    output_dir = tmp_path / "plots"
    write_summary(summary)

    plot_comps_summary(
        summary_csv=summary,
        output_dir=output_dir,
        run_name="toy",
    )

    for filename in (
        "overall_accuracy.png",
        "subtask_dynamics.png",
        "negative_sample_type_dynamics.png",
    ):
        path = output_dir / filename
        assert path.exists()
        assert path.stat().st_size > 0
