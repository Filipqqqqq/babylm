from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from babysteps.linguistic_repair import build_qa_who_common_name_frames
from babysteps.linguistic_repair.who_common_name_frames import (
    BANNED_BENCHMARK_PHRASES,
    BANNED_NAMES,
    WHO_COMMON_NAME_FRAMES,
)


ROOT = Path(__file__).resolve().parents[1]


def _base_config(tmp_path: Path, *, document_count: int = 500) -> dict:
    return {
        "name": "toy_qa_who_common_name_frames",
        "seed": 42,
        "document_count": document_count,
        "direct_dialogue_ratio": 0.5,
        "output_dir": str(tmp_path / "qa_who_common_name_frames"),
        "names": [
            "Ross",
            "Mark",
            "John",
            "Adam",
            "Paul",
            "Marky",
            "Ben",
            "Nina",
            "George",
            "Mary",
            "Edna",
            "Phil",
            "Eve",
            "Bessie",
            "Peter",
            "Jack",
            "Richard",
            "Dorothy",
        ],
        "excluded_names": ["Sarah", "David", "Davis"],
        "answer_style_weights": {
            "bare_name": 0.50,
            "bare_role": 0.25,
            "full_sentence_name": 0.20,
            "kinship_title": 0.05,
        },
        "frame_families": list(WHO_COMMON_NAME_FRAMES),
    }


def test_who_common_name_frames_are_balanced_and_name_clean(tmp_path: Path) -> None:
    result = build_qa_who_common_name_frames(_base_config(tmp_path), base_dir=ROOT)

    assert len(result.documents) == 500
    assert result.audit_report["counts"] == {"pass": 500, "warn": 0, "fail": 0}
    family_counts = Counter(row["frame_family"] for row in result.documents)
    assert family_counts == {family: 50 for family in WHO_COMMON_NAME_FRAMES}
    style_counts = Counter(row["style"] for row in result.documents)
    assert style_counts == {"direct_dialogue": 250, "scene_qa": 250}
    answer_style_counts = Counter(row["answer_style"] for row in result.documents)
    assert answer_style_counts == {
        "bare_name": 250,
        "bare_role": 125,
        "full_sentence_name": 100,
        "kinship_title": 25,
    }
    assert len({row["frame_id"] for row in result.documents}) == 30

    text = "\n".join(row["text"] for row in result.documents)
    lower = text.lower()
    for name in BANNED_NAMES:
        assert name not in text
    for phrase in BANNED_BENCHMARK_PHRASES:
        assert phrase not in lower

    for row in result.documents:
        assert row["question_type"] == "who"
        assert row["question"].startswith("Who ")
        assert row["text"].count("?") == 1
        assert "\n" not in row["text"]
        assert "bad:" not in row["text"].lower()


def test_who_common_name_frame_artifacts_are_written(tmp_path: Path) -> None:
    result = build_qa_who_common_name_frames(
        _base_config(tmp_path, document_count=20),
        base_dir=ROOT,
    )

    assert result.outputs["documents"].exists()
    assert result.outputs["train"].exists()
    assert result.outputs["audit_report"].exists()
    assert result.outputs["manifest"].exists()
    train_lines = result.outputs["train"].read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 20
    manifest = json.loads(result.outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["counts"]["documents"] == 20
    assert manifest["counts"]["train_documents"] == 20


def test_who_common_name_frame_cli_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(
        "\n".join(
            [
                "name: cli_qa_who_common_name_frames",
                "seed: 42",
                "document_count: 20",
                "direct_dialogue_ratio: 0.5",
                f"output_dir: {output_dir}",
                "excluded_names:",
                "  - Sarah",
                "  - David",
                "  - Davis",
                "frame_families:",
                "  - simple_agent_action",
                "  - household_task",
                "  - school_classroom",
                "  - helping_care",
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
                / "generate_qa_who_common_name_frames.py"
            ),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "QA who common-name frames: cli_qa_who_common_name_frames" in completed.stdout
    assert (output_dir / "train.txt").exists()
    assert (output_dir / "documents.jsonl").exists()
