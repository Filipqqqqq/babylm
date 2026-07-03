from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from babysteps.linguistic_repair import build_qa_who_person_frames
from babysteps.linguistic_repair.who_person_frames import WHO_FRAME_FAMILIES


ROOT = Path(__file__).resolve().parents[1]


def _base_config(tmp_path: Path, *, document_count: int = 500) -> dict:
    return {
        "name": "toy_qa_who_person_frames",
        "seed": 42,
        "document_count": document_count,
        "direct_dialogue_ratio": 0.5,
        "output_dir": str(tmp_path / "qa_who_person_frames"),
        "frame_families": list(WHO_FRAME_FAMILIES),
    }


def test_who_person_frames_are_balanced_for_inspection_batch(tmp_path: Path) -> None:
    result = build_qa_who_person_frames(_base_config(tmp_path), base_dir=ROOT)

    assert len(result.documents) == 500
    assert result.audit_report["counts"] == {"pass": 500, "warn": 0, "fail": 0}
    family_counts = Counter(row["frame_family"] for row in result.documents)
    assert family_counts == {
        "assist_care_help": 125,
        "contact_visit_question": 125,
        "creation_task_doer": 125,
        "role_action_subject": 125,
    }
    style_counts = Counter(row["style"] for row in result.documents)
    assert style_counts == {"direct_dialogue": 250, "scene_qa": 250}

    styles_by_family = {
        family: Counter(row["style"] for row in result.documents if row["frame_family"] == family)
        for family in WHO_FRAME_FAMILIES
    }
    for counts in styles_by_family.values():
        assert set(counts) == {"direct_dialogue", "scene_qa"}
        assert max(counts.values()) - min(counts.values()) <= 1

    for row in result.documents:
        text = row["text"]
        assert text.count("?") == 1
        assert row["question"].startswith("Who ")
        assert row["question_type"] == "who"
        assert not row["answer"].endswith(" did.")
        assert "Where " not in text
        assert "When " not in text
        assert "Why " not in text
        assert "How many" not in text
        assert "bad:" not in text.lower()
        assert "\n" not in text


def test_who_person_frames_avoid_exact_blimp_like_prompts(tmp_path: Path) -> None:
    result = build_qa_who_person_frames(_base_config(tmp_path), base_dir=ROOT)
    text = "\n".join(row["text"] for row in result.documents).lower()

    assert "who did you see" not in text
    assert "who did you meet" not in text
    assert "who helped you" not in text
    assert "who is sleeping" not in text
    assert "who is drawing" not in text
    assert "who painted that picture" not in text


def test_who_person_frames_include_common_use_events(tmp_path: Path) -> None:
    result = build_qa_who_person_frames(_base_config(tmp_path), base_dir=ROOT)
    text = "\n".join(row["text"] for row in result.documents).lower()

    assert "who was using the restroom" in text
    assert "who was using the charger" in text


def test_who_person_frame_artifacts_are_written(tmp_path: Path) -> None:
    result = build_qa_who_person_frames(_base_config(tmp_path, document_count=16), base_dir=ROOT)

    assert result.outputs["documents"].exists()
    assert result.outputs["train"].exists()
    assert result.outputs["audit_report"].exists()
    assert result.outputs["manifest"].exists()
    train_lines = result.outputs["train"].read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 16
    manifest = json.loads(result.outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["counts"]["documents"] == 16
    assert manifest["counts"]["train_documents"] == 16


def test_who_person_frame_cli_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(
        "\n".join(
            [
                "name: cli_qa_who_person_frames",
                "seed: 42",
                "document_count: 16",
                "direct_dialogue_ratio: 0.5",
                f"output_dir: {output_dir}",
                "frame_families:",
                "  - contact_visit_question",
                "  - assist_care_help",
                "  - role_action_subject",
                "  - creation_task_doer",
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
                / "generate_qa_who_person_frames.py"
            ),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "QA who-person frames: cli_qa_who_person_frames" in completed.stdout
    assert (output_dir / "train.txt").exists()
    assert (output_dir / "documents.jsonl").exists()
