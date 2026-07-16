from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from babysteps.linguistic_repair import build_qa_wherewhat_hardconfound_frames
from babysteps.linguistic_repair.where_what_hardconfound_frames import (
    BARE_FRAGMENT,
    BARE_FULL_SENTENCE,
    DESTINATION_PATH_COMMON,
    DESTINATION_PATH_NAMED,
    DIALOGUE_FRAGMENT,
    OBJECT_ANCHORED_LOCAL,
    WHAT_FIRST,
    WHERE_FIRST,
)


ROOT = Path(__file__).resolve().parents[1]


def _contains_casefold(text: str, phrase: str) -> bool:
    return phrase.casefold() in text.casefold()


def _base_config(tmp_path: Path, *, document_count: int = 500) -> dict:
    return {
        "name": "toy_qa_wherewhat_hardconfound_frames",
        "seed": 42,
        "document_count": document_count,
        "output_dir": str(tmp_path / "qa_wherewhat_hardconfound_frames"),
        "scenario_group_weights": {
            DESTINATION_PATH_NAMED: 0.15,
            DESTINATION_PATH_COMMON: 0.35,
            OBJECT_ANCHORED_LOCAL: 0.50,
        },
        "style_weights": {
            BARE_FRAGMENT: 0.50,
            DIALOGUE_FRAGMENT: 0.25,
            BARE_FULL_SENTENCE: 0.25,
            "dialogue_full_sentence": 0.0,
        },
        "second_person_ratio": 0.80,
        "where_first_ratio": 0.50,
        "speaker_pairs": [
            ["A", "B"],
            ["David", "Sarah"],
            ["Maya", "Omar"],
            ["Lia", "Ben"],
            ["Nora", "Theo"],
            ["Iris", "Leo"],
        ],
    }


def test_wherewhat_hardconfound_frames_preserve_roles_and_ratios(tmp_path: Path) -> None:
    result = build_qa_wherewhat_hardconfound_frames(
        _base_config(tmp_path),
        base_dir=ROOT,
    )

    assert len(result.documents) == 500
    assert result.audit_report["counts"] == {"pass": 500, "warn": 0, "fail": 0}
    assert Counter(row["scenario_group"] for row in result.documents) == {
        DESTINATION_PATH_NAMED: 75,
        DESTINATION_PATH_COMMON: 175,
        OBJECT_ANCHORED_LOCAL: 250,
    }
    assert Counter(row["style"] for row in result.documents) == {
        BARE_FRAGMENT: 250,
        DIALOGUE_FRAGMENT: 125,
        BARE_FULL_SENTENCE: 125,
    }
    assert Counter(row["order"] for row in result.documents) == {
        WHERE_FIRST: 250,
        WHAT_FIRST: 250,
    }
    assert Counter(row["person"] for row in result.documents) == {
        "second_person": 400,
        "third_person": 100,
    }
    assert result.manifest["counts"]["named_location_counts"] == {
        "generic": 425,
        "named": 75,
    }
    speaker_counts = result.manifest["counts"]["speaker_pair_counts"]
    nonempty_speakers = {
        speaker_pair: count
        for speaker_pair, count in speaker_counts.items()
        if speaker_pair != "none" and count > 0
    }
    assert len(nonempty_speakers) >= 5

    text = "\n".join(row["text"] for row in result.documents).lower()
    assert "to california" not in text
    assert "to europe" not in text
    assert "across the country" not in text
    assert "to the next town over" not in text
    assert "to the other side of town" not in text
    assert "under the papers" not in text
    assert "on the table" not in text
    assert "behind the sofa" not in text
    assert "next to the bush" not in text
    assert "where did you send the package?" not in text

    for row in result.documents:
        assert row["question_type"] == "where_what_pair"
        assert row["audit_status"] == "pass"
        assert row["text"].count("?") == 2
        assert "\n" not in row["text"]
        assert "bad:" not in row["text"].lower()
        assert row["where_question"].startswith("Where did ")
        assert row["what_question"].startswith("What did ")
        assert row["where_question"] in row["text"]
        assert row["where_answer"] in row["text"]
        assert row["what_question"] in row["text"]
        assert row["what_answer"] in row["text"]
        assert _contains_casefold(row["where_question"], row["theme"])
        assert _contains_casefold(row["what_answer"], row["theme"])
        assert _contains_casefold(row["what_question"], row["location"])
        assert _contains_casefold(row["where_answer"], row["location"])
        if row["style"] == DIALOGUE_FRAGMENT:
            assert row["speaker_pair"] is not None


def test_wherewhat_hardconfound_artifacts_are_written(tmp_path: Path) -> None:
    result = build_qa_wherewhat_hardconfound_frames(
        _base_config(tmp_path, document_count=23),
        base_dir=ROOT,
    )

    assert result.outputs["documents"].exists()
    assert result.outputs["train"].exists()
    assert result.outputs["audit_report"].exists()
    assert result.outputs["manifest"].exists()
    train_lines = result.outputs["train"].read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 23
    manifest = json.loads(result.outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["counts"]["documents"] == 23
    assert manifest["counts"]["train_documents"] == 23
    assert manifest["qwen_rewrite_ready"] is True
    assert "theme" in manifest["role_fields"]
    assert "location" in manifest["role_fields"]


def test_wherewhat_hardconfound_cli_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(
        "\n".join(
            [
                "name: cli_qa_wherewhat_hardconfound_frames",
                "seed: 42",
                "document_count: 32",
                f"output_dir: {output_dir}",
                "scenario_group_weights:",
                "  destination_path_named: 0.15",
                "  destination_path_common: 0.35",
                "  object_anchored_local: 0.50",
                "style_weights:",
                "  bare_fragment: 0.50",
                "  dialogue_fragment: 0.25",
                "  bare_full_sentence: 0.25",
                "  dialogue_full_sentence: 0.00",
                "second_person_ratio: 0.80",
                "where_first_ratio: 0.50",
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
                / "generate_qa_wherewhat_hardconfound_frames.py"
            ),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        "QA where/what hard-confound frames: cli_qa_wherewhat_hardconfound_frames"
        in completed.stdout
    )
    assert (output_dir / "train.txt").exists()
    assert (output_dir / "documents.jsonl").exists()
