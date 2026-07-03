from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from babysteps.linguistic_repair import build_qa_turn_taking_pronoun_frames
from babysteps.linguistic_repair.turn_taking_pronoun_frames import (
    AUX_FAMILIES,
    BARE_POLARITY_ANSWER,
    CONTRACTED_NEGATIVE,
    DIRECT_DIALOGUE,
    FULL_NEGATIVE,
    MAPPING_FRAMES,
    NO_NEGATIVE,
    REPORT_TEMPLATES,
    WITH_POLARITY_MARKER,
)


ROOT = Path(__file__).resolve().parents[1]


def _base_config(tmp_path: Path, *, document_count: int = 500) -> dict:
    return {
        "name": "toy_qa_turn_taking_pronoun_frames",
        "seed": 42,
        "document_count": document_count,
        "direct_dialogue_ratio": 0.5,
        "yes_ratio": 0.5,
        "answer_marker_ratio": 0.5,
        "negative_contraction_ratio": 0.5,
        "weak_mapping_ratio": 0.7,
        "output_dir": str(tmp_path / "qa_turn_taking_pronoun_frames"),
        "mapping_families": list(MAPPING_FRAMES),
        "aux_families": list(AUX_FAMILIES),
    }


def test_turn_taking_frames_are_balanced_for_inspection_batch(tmp_path: Path) -> None:
    result = build_qa_turn_taking_pronoun_frames(_base_config(tmp_path), base_dir=ROOT)

    assert len(result.documents) == 500
    assert result.audit_report["counts"] == {"pass": 500, "warn": 0, "fail": 0}
    mapping_counts = Counter(row["mapping"] for row in result.documents)
    assert mapping_counts == {
        "i_to_you": 117,
        "we_to_you": 117,
        "you_to_we": 116,
        "you_to_i": 50,
        "she_to_she": 50,
        "they_to_they": 50,
    }
    style_counts = Counter(row["style"] for row in result.documents)
    assert style_counts == {"direct_dialogue": 250, "named_report": 250}
    report_template_counts = Counter(row["report_template"] for row in result.documents)
    assert report_template_counts[DIRECT_DIALOGUE] == 250
    for template in REPORT_TEMPLATES:
        assert report_template_counts[template] in {83, 84}
    polarity_counts = Counter(row["polarity"] for row in result.documents)
    assert polarity_counts == {"no": 250, "yes": 250}
    answer_surface_counts = Counter(row["answer_surface"] for row in result.documents)
    assert answer_surface_counts == {
        BARE_POLARITY_ANSWER: 250,
        WITH_POLARITY_MARKER: 250,
    }
    negative_form_counts = Counter(row["negative_form"] for row in result.documents)
    assert negative_form_counts == {
        CONTRACTED_NEGATIVE: 125,
        FULL_NEGATIVE: 125,
        NO_NEGATIVE: 250,
    }
    weak_counts = Counter(row["weak_target"] for row in result.documents)
    assert weak_counts == {True: 350, False: 150}
    assert len({row["text"] for row in result.documents}) == 500

    aux_counts = Counter(row["aux_family"] for row in result.documents)
    assert set(aux_counts) == set(AUX_FAMILIES)
    assert max(aux_counts.values()) - min(aux_counts.values()) <= 1

    for row in result.documents:
        text = row["text"]
        assert text.count("?") == 1
        if row["answer_surface"] == WITH_POLARITY_MARKER:
            assert row["answer"].startswith(("Yes, ", "No, "))
        else:
            assert not row["answer"].startswith(("Yes, ", "No, "))
        assert row["question"] in text
        assert row["answer"] in text
        assert "bad:" not in text.lower()
        assert "wrong:" not in text.lower()
        assert "\n" not in text


def test_turn_taking_frames_include_targeted_pronoun_switches(tmp_path: Path) -> None:
    result = build_qa_turn_taking_pronoun_frames(_base_config(tmp_path), base_dir=ROOT)
    rows_by_mapping = {
        mapping: [row for row in result.documents if row["mapping"] == mapping]
        for mapping in MAPPING_FRAMES
    }

    assert any(row["question"].startswith("Did I ") for row in rows_by_mapping["i_to_you"])
    assert any(row["answer"].startswith("Yes, you ") for row in rows_by_mapping["i_to_you"])
    assert any(row["answer"].startswith("No, you ") for row in rows_by_mapping["i_to_you"])
    assert any(row["question"].startswith("Have we ") for row in rows_by_mapping["we_to_you"])
    assert any(row["answer"].startswith("No, you ") for row in rows_by_mapping["we_to_you"])
    assert any(row["question"].startswith("Can you ") for row in rows_by_mapping["you_to_we"])
    assert any(row["answer"].startswith("Yes, we ") for row in rows_by_mapping["you_to_we"])
    assert any(row["question"].startswith("Should you ") for row in rows_by_mapping["you_to_i"])
    assert any(row["answer"].startswith("No, I ") for row in rows_by_mapping["you_to_i"])


def test_turn_taking_frames_cover_auxiliary_families(tmp_path: Path) -> None:
    result = build_qa_turn_taking_pronoun_frames(_base_config(tmp_path), base_dir=ROOT)
    text = "\n".join(row["text"] for row in result.documents)

    assert "No, you didn't." in text
    assert "Yes, you have." in text
    assert "you haven't." in text
    assert "you have not." in text
    assert "Yes, we can." in text
    assert "I shouldn't." in text
    assert "I should not." in text
    assert "Yes, she will." in text
    assert "they weren't." in text
    assert "they were not." in text


def test_turn_taking_frames_include_bare_answer_variants(tmp_path: Path) -> None:
    result = build_qa_turn_taking_pronoun_frames(_base_config(tmp_path), base_dir=ROOT)
    bare_rows = [row for row in result.documents if row["answer_surface"] == BARE_POLARITY_ANSWER]
    text = "\n".join(row["text"] for row in bare_rows)

    assert len(bare_rows) == 250
    assert any(row["answer"] == "You did." for row in bare_rows)
    assert any(row["answer"] == "You haven't." for row in bare_rows)
    assert any(row["answer"] == "You have not." for row in bare_rows)
    assert any(row["answer"] == "We can." for row in bare_rows)
    assert any(row["answer"] == "I shouldn't." for row in bare_rows)
    assert any(" not." in row["answer"] for row in bare_rows)
    assert "Yes, " not in text
    assert "No, " not in text


def test_turn_taking_frames_include_full_negative_variants(tmp_path: Path) -> None:
    result = build_qa_turn_taking_pronoun_frames(_base_config(tmp_path), base_dir=ROOT)
    text = "\n".join(row["text"] for row in result.documents)

    assert "didn't." in text
    assert "did not." in text
    assert "haven't." in text
    assert "have not." in text
    assert "can't." in text
    assert "can not." in text
    assert "shouldn't." in text
    assert "should not." in text
    assert "won't." in text
    assert "will not." in text
    assert "wasn't." in text
    assert "was not." in text
    assert "weren't." in text
    assert "were not." in text


def test_turn_taking_frames_vary_names_and_groups(tmp_path: Path) -> None:
    result = build_qa_turn_taking_pronoun_frames(_base_config(tmp_path), base_dir=ROOT)
    named_rows = [row for row in result.documents if row["style"] == "named_report"]

    assert len({row["asker"] for row in named_rows}) >= 15
    assert len({row["responder"] for row in named_rows}) >= 15
    assert len({row["referent"] for row in named_rows if row["referent"]}) >= 10


def test_turn_taking_frames_diversify_named_report_verbs(tmp_path: Path) -> None:
    result = build_qa_turn_taking_pronoun_frames(_base_config(tmp_path), base_dir=ROOT)
    text = "\n".join(row["text"] for row in result.documents)

    assert " said, " in text
    assert " answered, " in text
    assert " replied." in text
    assert text.count(" said, ") < 100


def test_turn_taking_frames_avoid_exact_eval_like_fragments(tmp_path: Path) -> None:
    result = build_qa_turn_taking_pronoun_frames(_base_config(tmp_path), base_dir=ROOT)
    text = "\n".join(row["text"] for row in result.documents).lower()

    assert "should you quit" not in text
    assert "should she quit" not in text
    assert "did you quit" not in text
    assert "did she quit" not in text
    assert "meet him" not in text
    assert "arrive" not in text
    assert "go somewhere after the movie" not in text
    assert "try to finish it on time or not" not in text
    assert "figure it out" not in text


def test_turn_taking_frame_artifacts_are_written(tmp_path: Path) -> None:
    result = build_qa_turn_taking_pronoun_frames(_base_config(tmp_path, document_count=24), base_dir=ROOT)

    assert result.outputs["documents"].exists()
    assert result.outputs["train"].exists()
    assert result.outputs["audit_report"].exists()
    assert result.outputs["manifest"].exists()
    train_lines = result.outputs["train"].read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 24
    manifest = json.loads(result.outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["counts"]["documents"] == 24
    assert manifest["counts"]["train_documents"] == 24


def test_turn_taking_frame_cli_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(
        "\n".join(
            [
                "name: cli_qa_turn_taking_pronoun_frames",
                "seed: 42",
                "document_count: 24",
                "direct_dialogue_ratio: 0.5",
                "yes_ratio: 0.5",
                "answer_marker_ratio: 0.5",
                "negative_contraction_ratio: 0.5",
                "weak_mapping_ratio: 0.7",
                f"output_dir: {output_dir}",
                "mapping_families:",
                "  - i_to_you",
                "  - we_to_you",
                "  - you_to_we",
                "  - you_to_i",
                "  - she_to_she",
                "  - they_to_they",
                "aux_families:",
                "  - did",
                "  - have",
                "  - can",
                "  - should",
                "  - will",
                "  - be_past",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "synthetic_data_generation"
                / "linguistic_repair"
                / "generate_qa_turn_taking_pronoun_frames.py"
            ),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "QA turn-taking pronoun frames: cli_qa_turn_taking_pronoun_frames" in completed.stdout
    assert (output_dir / "train.txt").exists()
    assert (output_dir / "documents.jsonl").exists()
