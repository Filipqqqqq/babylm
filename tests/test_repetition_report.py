from __future__ import annotations

import json
from pathlib import Path

from babysteps.cli.repetition_report import main as repetition_main
from babysteps.concept_curriculum.repetition import build_repetition_report


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def sample_rows() -> list[dict]:
    return [
        {
            "id": "doc_balloon",
            "text": (
                "The balloon crossed the roofline after the wind lifted it. "
                "The balloon started below the roofline."
            ),
            "metadata": {"family": "landmark_direction", "style_id": "everyday_observation"},
        },
        {
            "id": "doc_kite",
            "text": (
                "The kite crossed the roofline after the wind lifted it. "
                "The kite started below the roofline."
            ),
            "metadata": {"family": "landmark_direction", "style_id": "observation_log"},
        },
        {
            "id": "doc_crate",
            "text": "A crate fell past the ledge when the support broke.",
            "metadata": {"family": "support_stability", "style_id": "observation_log"},
        },
    ]


def test_repetition_report_summarizes_repeated_ngrams_across_documents():
    report = build_repetition_report(
        sample_rows(),
        min_n=3,
        max_n=4,
        top_k=10,
        group_by=["family"],
    )

    four_grams = {row["ngram"]: row for row in report["top_ngrams"]["4"]}

    assert report["document_count"] == 3
    assert report["token_count"] > 0
    assert report["summary"]["4"]["repeated_unique_ngrams"] > 0
    assert "crossed the roofline after" in four_grams
    assert four_grams["crossed the roofline after"]["count"] == 2
    assert four_grams["crossed the roofline after"]["document_count"] == 2
    assert four_grams["crossed the roofline after"]["document_ids"] == [
        "doc_balloon",
        "doc_kite",
    ]
    assert report["groups"]["family"]["landmark_direction"]["document_count"] == 2


def test_repetition_report_cli_writes_json_report(tmp_path):
    input_path = tmp_path / "documents_audited.jsonl"
    output_path = tmp_path / "repetition_report.json"
    write_jsonl(sample_rows(), input_path)

    repetition_main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--min-n",
            "3",
            "--max-n",
            "4",
            "--top-k",
            "10",
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    four_grams = {row["ngram"]: row for row in report["top_ngrams"]["4"]}

    assert report["document_count"] == 3
    assert report["settings"]["text_field"] == "text"
    assert "crossed the roofline after" in four_grams
