from __future__ import annotations

import csv
from pathlib import Path

from babysteps.analysis.ewok_context_dynamics import (
    build_panel_series,
    load_domain_rows,
    plot_domain_grid,
)


def write_summary(path: Path) -> None:
    fieldnames = [
        "run_name",
        "epoch",
        "step",
        "checkpoint",
        "checkpoint_path",
        "group_scope",
        "group_name",
        "score_reduction",
        "items",
        "comparisons",
        "correct_comparisons",
        "score",
        "mean_k_margin",
        "target_token_count_mismatches",
        "sum_mean_sign_disagreements",
    ]
    rows = [
        {
            "run_name": "toy",
            "epoch": "1",
            "step": "10",
            "group_scope": "domain",
            "group_name": "agent-properties",
            "score_reduction": "sum",
            "items": "2",
            "score": "0.25",
        },
        {
            "run_name": "toy",
            "epoch": "1",
            "step": "20",
            "group_scope": "domain",
            "group_name": "agent-properties",
            "score_reduction": "sum",
            "items": "2",
            "score": "0.75",
        },
        {
            "run_name": "toy",
            "epoch": "1",
            "step": "20",
            "group_scope": "overall",
            "group_name": "all",
            "score_reduction": "sum",
            "items": "22",
            "score": "0.5",
        },
        {
            "run_name": "toy",
            "epoch": "1",
            "step": "20",
            "group_scope": "context_type",
            "group_name": "direct",
            "score_reduction": "sum",
            "items": "3",
            "score": "1.0",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_load_domain_rows_keeps_domains_and_average_only(tmp_path: Path) -> None:
    summary = tmp_path / "ewok_context_summary.csv"
    write_summary(summary)

    rows = load_domain_rows(summary)

    assert [row["panel"] for row in rows] == [
        "agent-properties",
        "agent-properties",
        "average",
    ]
    assert rows[0]["step"] == 10
    assert rows[0]["score"] == 0.25


def test_build_panel_series_averages_duplicate_steps() -> None:
    rows = [
        {"panel": "average", "series": "toy", "step": 10, "score": 0.25},
        {"panel": "average", "series": "toy", "step": 10, "score": 0.75},
    ]

    panel_series = build_panel_series(rows)

    assert panel_series["average"]["toy"] == [(10, 0.5)]


def test_plot_domain_grid_writes_png_with_chance_baseline(tmp_path: Path) -> None:
    output_path = tmp_path / "ewok_context_domain_dynamics.png"

    plot_domain_grid(
        panel_series={
            "agent-properties": {"toy": [(10, 0.25), (20, 0.75)]},
            "average": {"toy": [(10, 0.5), (20, 0.6)]},
        },
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
