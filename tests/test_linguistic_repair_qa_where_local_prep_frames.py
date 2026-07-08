from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from babysteps.linguistic_repair import build_qa_where_local_prep_frames
from babysteps.linguistic_repair.where_local_prep_frames import (
    BANNED_BENCHMARK_WORDS,
    LOCATIONS_BY_PREPOSITION,
    VERB_FRAMES,
    WHERE_DIRECT_ACTION,
    WHERE_END_UP,
    WHERE_INDIRECT,
    WHERE_NOW,
)


ROOT = Path(__file__).resolve().parents[1]


def _base_config(tmp_path: Path, *, document_count: int = 500) -> dict:
    return {
        "name": "toy_qa_where_local_prep_frames",
        "seed": 42,
        "document_count": document_count,
        "output_dir": str(tmp_path / "qa_where_local_prep_frames"),
        "style_weights": {
            "direct_dialogue": 0.4,
            "scene_qa": 0.35,
            "paired_what_where": 0.25,
        },
        "second_person_ratio": 0.60,
        "pronoun_question_ratio": 0.45,
        "full_answer_ratio": 0.45,
        "where_question_style_weights": {
            WHERE_DIRECT_ACTION: 0.70,
            WHERE_NOW: 0.15,
            WHERE_END_UP: 0.10,
            WHERE_INDIRECT: 0.05,
        },
        "prepositions": list(LOCATIONS_BY_PREPOSITION),
        "verb_families": list(VERB_FRAMES),
    }


def test_where_local_prep_frames_are_balanced_and_blimp_noun_clean(tmp_path: Path) -> None:
    result = build_qa_where_local_prep_frames(_base_config(tmp_path), base_dir=ROOT)

    assert len(result.documents) == 500
    assert result.audit_report["counts"] == {"pass": 500, "warn": 0, "fail": 0}
    preposition_counts = Counter(row["answer_preposition"] for row in result.documents)
    assert set(preposition_counts) == set(LOCATIONS_BY_PREPOSITION)
    assert max(preposition_counts.values()) - min(preposition_counts.values()) <= 1
    family_counts = Counter(row["verb_family"] for row in result.documents)
    assert set(family_counts) == set(VERB_FRAMES)
    assert max(family_counts.values()) - min(family_counts.values()) <= 1
    where_question_style_counts = Counter(row["where_question_style"] for row in result.documents)
    assert where_question_style_counts == {
        WHERE_DIRECT_ACTION: 350,
        WHERE_NOW: 75,
        WHERE_END_UP: 50,
        WHERE_INDIRECT: 25,
    }

    text = "\n".join(row["text"] for row in result.documents).lower()
    words = {word.strip(".,?!").lower() for word in text.split()}
    assert not (words & BANNED_BENCHMARK_WORDS)
    assert "behind " in text
    assert "next to " in text
    assert "under " in text

    for row in result.documents:
        assert row["question_type"] == "where"
        if row["where_question_style"] == WHERE_DIRECT_ACTION:
            assert row["question"].startswith("Where did ")
            if row["question_object_reference"] == "it":
                assert row["style"] != "direct_dialogue"
        elif row["where_question_style"] == WHERE_NOW:
            assert row["question"].startswith("Where is ")
            assert row["question"].endswith(" now?")
            assert row["question_object_reference"] != "it"
        elif row["where_question_style"] == WHERE_END_UP:
            assert row["question"].startswith("Where did ")
            assert row["question"].endswith(" end up?")
            assert row["question_object_reference"] != "it"
        elif row["where_question_style"] == WHERE_INDIRECT:
            assert row["question"].startswith("Can you tell me where ")
            assert row["question"].endswith(" is?")
            assert row["question_object_reference"] != "it"
        else:
            raise AssertionError(f"Unknown where question style: {row['where_question_style']}")
        assert "\n" not in row["text"]
        assert "bad:" not in row["text"].lower()
        assert row["answer_preposition"] in row["location"].lower()
        if row["style"] == "paired_what_where":
            assert row["text"].count("?") == 2
            assert "What did " in row["text"]
        else:
            assert row["text"].count("?") == 1


def test_where_local_prep_frame_artifacts_are_written(tmp_path: Path) -> None:
    result = build_qa_where_local_prep_frames(
        _base_config(tmp_path, document_count=21),
        base_dir=ROOT,
    )

    assert result.outputs["documents"].exists()
    assert result.outputs["train"].exists()
    assert result.outputs["audit_report"].exists()
    assert result.outputs["manifest"].exists()
    train_lines = result.outputs["train"].read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 21
    manifest = json.loads(result.outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["counts"]["documents"] == 21
    assert manifest["counts"]["train_documents"] == 21


def test_where_local_prep_frame_cli_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(
        "\n".join(
            [
                "name: cli_qa_where_local_prep_frames",
                "seed: 42",
                "document_count: 21",
                f"output_dir: {output_dir}",
                "style_weights:",
                "  direct_dialogue: 0.4",
                "  scene_qa: 0.35",
                "  paired_what_where: 0.25",
                "where_question_style_weights:",
                "  direct_action: 0.7",
                "  where_now: 0.15",
                "  end_up: 0.1",
                "  indirect_where: 0.05",
                "prepositions:",
                "  - behind",
                "  - next to",
                "  - under",
                "verb_families:",
                "  - put_place_set",
                "  - leave_drop_hide_store",
                "  - find_discover_spot",
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
                / "generate_qa_where_local_prep_frames.py"
            ),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "QA where local-prep frames: cli_qa_where_local_prep_frames" in completed.stdout
    assert (output_dir / "train.txt").exists()
    assert (output_dir / "documents.jsonl").exists()
