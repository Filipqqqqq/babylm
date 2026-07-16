from __future__ import annotations

import csv
from pathlib import Path

from babysteps.analysis.lexical.comps_frequency import (
    build_comps_frequency_report,
    load_frequency_targets,
    select_frequency_slice,
)


def write_target_coverage(path: Path) -> None:
    fieldnames = [
        "target_id",
        "label",
        "surfaces",
        "lemmas",
        "total_mentions",
        "surface_match_mentions",
        "lemma_match_mentions",
        "document_count",
        "exposure_bucket",
        "mean_subwords_per_occurrence",
        "mean_subwords_per_observed_type",
        "top_surfaces",
        "top_lemmas",
        "top_tokenizations",
        "top_sources",
    ]
    rows = [
        ("target_a", "accordion", 100, 90, "51-100"),
        ("target_b", "banana", 80, 75, "51-100"),
        ("target_c", "cabbage", 60, 55, "51-100"),
        ("target_d", "drum", 40, 35, "11-50"),
        ("target_e", "engine", 20, 18, "11-50"),
        ("target_f", "flute", 10, 9, "6-10"),
        ("target_g", "guitar", 5, 5, "1-5"),
        ("target_h", "harp", 0, 0, "0"),
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for target_id, label, mentions, docs, bucket in rows:
            writer.writerow(
                {
                    "target_id": target_id,
                    "label": label,
                    "surfaces": label,
                    "lemmas": label,
                    "total_mentions": mentions,
                    "surface_match_mentions": mentions,
                    "lemma_match_mentions": mentions,
                    "document_count": docs,
                    "exposure_bucket": bucket,
                    "mean_subwords_per_occurrence": "1.0000",
                    "mean_subwords_per_observed_type": "1.0000",
                    "top_surfaces": "",
                    "top_lemmas": "",
                    "top_tokenizations": "",
                    "top_sources": "",
                }
            )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_select_frequency_slices(tmp_path: Path) -> None:
    coverage = tmp_path / "target_coverage.csv"
    write_target_coverage(coverage)
    targets = load_frequency_targets(coverage)

    assert [target.label for target in select_frequency_slice(targets, slice_name="top", n=3)] == [
        "accordion",
        "banana",
        "cabbage",
    ]
    assert [
        target.label for target in select_frequency_slice(targets, slice_name="middle", n=3)
    ] == ["cabbage", "drum", "engine"]
    assert [
        target.label for target in select_frequency_slice(targets, slice_name="bottom", n=3)
    ] == ["flute", "guitar", "harp"]


def test_build_comps_frequency_report_writes_csv_and_plots(tmp_path: Path) -> None:
    coverage = tmp_path / "target_coverage.csv"
    output_dir = tmp_path / "plots"
    write_target_coverage(coverage)

    report = build_comps_frequency_report(
        target_coverage_csv=coverage,
        output_dir=output_dir,
        n=3,
        title_prefix="COMPS concepts",
    )

    assert report["target_count"] == 8
    rows = read_csv(output_dir / "frequency_top_middle_bottom.csv")
    assert len(rows) == 9
    assert {row["slice"] for row in rows} == {"top", "middle", "bottom"}

    expected_filenames = {
        "frequency_top_middle_bottom.png",
        "frequency_top.png",
        "frequency_middle.png",
        "frequency_bottom.png",
        "exposure_buckets.png",
    }
    assert {path.name for path in report["plots"]} == expected_filenames
    for path in report["plots"]:
        assert path.exists()
        assert path.stat().st_size > 0
