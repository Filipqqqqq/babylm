from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from babysteps.linguistic_repair import build_qa_what_object_frames
from babysteps.linguistic_repair.what_object_frames import PERSON_ANSWER_WORDS, WHAT_FRAME_FAMILIES


ROOT = Path(__file__).resolve().parents[1]


def _base_config(tmp_path: Path, *, document_count: int = 500) -> dict:
    return {
        "name": "toy_qa_what_object_frames",
        "seed": 42,
        "document_count": document_count,
        "direct_dialogue_ratio": 0.5,
        "second_person_ratio": 0.5,
        "first_person_echo_answer_ratio": 0.5,
        "output_dir": str(tmp_path / "qa_what_object_frames"),
        "frame_families": list(WHAT_FRAME_FAMILIES),
    }


def test_what_object_frames_are_balanced_for_inspection_batch(tmp_path: Path) -> None:
    result = build_qa_what_object_frames(_base_config(tmp_path), base_dir=ROOT)

    assert len(result.documents) == 500
    assert result.audit_report["counts"] == {"pass": 500, "warn": 0, "fail": 0}
    family_counts = Counter(row["frame_family"] for row in result.documents)
    assert family_counts == {
        "borrow_return_choose": 100,
        "craft_fold_decorate": 100,
        "open_lock_unlock": 100,
        "pack_carry_bring": 100,
        "repair_wash_clean": 100,
    }
    style_counts = Counter(row["style"] for row in result.documents)
    assert style_counts == {"direct_dialogue": 250, "scene_qa": 250}
    speaker_counts = Counter(row["speaker_mode"] for row in result.documents)
    assert speaker_counts == {"second_person": 250, "third_person": 250}
    answer_style_counts = Counter(row["answer_style"] for row in result.documents)
    assert answer_style_counts == {"first_person_echo_answer": 125, "fragment_answer": 375}

    styles_by_family = {
        family: Counter(row["style"] for row in result.documents if row["frame_family"] == family)
        for family in WHAT_FRAME_FAMILIES
    }
    for counts in styles_by_family.values():
        assert counts == {"direct_dialogue": 50, "scene_qa": 50}
    speakers_by_family = {
        family: Counter(row["speaker_mode"] for row in result.documents if row["frame_family"] == family)
        for family in WHAT_FRAME_FAMILIES
    }
    for counts in speakers_by_family.values():
        assert counts == {"second_person": 50, "third_person": 50}
    second_person_answer_styles = Counter(
        row["answer_style"]
        for row in result.documents
        if row["speaker_mode"] == "second_person"
    )
    assert second_person_answer_styles == {
        "first_person_echo_answer": 125,
        "fragment_answer": 125,
    }

    for row in result.documents:
        text = row["text"]
        answer = row["answer"].lower()
        assert text.count("?") == 1
        assert row["question"].startswith("What ")
        assert row["question_type"] == "what"
        assert "Who " not in text
        assert "Where " not in text
        assert "When " not in text
        assert "Why " not in text
        assert "How many" not in text
        assert "bad:" not in text.lower()
        assert "\n" not in text
        assert not any(
            re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", answer)
            for word in PERSON_ANSWER_WORDS
        )


def test_what_object_frames_avoid_exact_blimp_like_prompts(tmp_path: Path) -> None:
    result = build_qa_what_object_frames(_base_config(tmp_path), base_dir=ROOT)
    text = "\n".join(row["text"] for row in result.documents).lower()

    assert "what did you get" not in text
    assert "what did you sell" not in text
    assert "what did you make" not in text
    assert "what did you close" not in text
    assert "what did you break" not in text
    assert "what did you open?" not in text
    assert "what did you lock?" not in text


def test_what_object_frames_include_second_person_variants(tmp_path: Path) -> None:
    result = build_qa_what_object_frames(_base_config(tmp_path), base_dir=ROOT)
    second_person = [row for row in result.documents if row["speaker_mode"] == "second_person"]

    assert any(row["question"].startswith("What did you ") for row in second_person)
    assert any(row["answer"].startswith("I ") for row in second_person)
    assert any(not row["answer"].startswith("I ") for row in second_person)


def test_what_object_frames_include_fold_cloth_and_shirt(tmp_path: Path) -> None:
    result = build_qa_what_object_frames(_base_config(tmp_path), base_dir=ROOT)
    fold_answers = {
        row["answer_object"]
        for row in result.documents
        if row["frame_id"] == "fold_object"
    }

    assert "a cloth" in fold_answers
    assert "a shirt" in fold_answers


def test_what_object_frames_are_unique_for_inspection_batch(tmp_path: Path) -> None:
    result = build_qa_what_object_frames(_base_config(tmp_path), base_dir=ROOT)
    texts = [row["text"] for row in result.documents]

    assert len(texts) == len(set(texts))


def test_what_object_frame_artifacts_are_written(tmp_path: Path) -> None:
    result = build_qa_what_object_frames(_base_config(tmp_path, document_count=20), base_dir=ROOT)

    assert result.outputs["documents"].exists()
    assert result.outputs["train"].exists()
    assert result.outputs["audit_report"].exists()
    assert result.outputs["manifest"].exists()
    train_lines = result.outputs["train"].read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 20
    manifest = json.loads(result.outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["counts"]["documents"] == 20
    assert manifest["counts"]["train_documents"] == 20


def test_what_object_frame_cli_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(
        "\n".join(
            [
                "name: cli_qa_what_object_frames",
                "seed: 42",
                "document_count: 20",
                "direct_dialogue_ratio: 0.5",
                "second_person_ratio: 0.5",
                "first_person_echo_answer_ratio: 0.5",
                f"output_dir: {output_dir}",
                "frame_families:",
                "  - borrow_return_choose",
                "  - open_lock_unlock",
                "  - repair_wash_clean",
                "  - pack_carry_bring",
                "  - craft_fold_decorate",
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
                / "generate_qa_what_object_frames.py"
            ),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "QA what-object frames: cli_qa_what_object_frames" in completed.stdout
    assert (output_dir / "train.txt").exists()
    assert (output_dir / "documents.jsonl").exists()
