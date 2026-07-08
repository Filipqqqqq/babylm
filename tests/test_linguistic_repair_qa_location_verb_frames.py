from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from babysteps.linguistic_repair import build_qa_location_verb_frames
from babysteps.linguistic_repair.location_verb_frames import VERB_FAMILIES


ROOT = Path(__file__).resolve().parents[1]


def _base_config(tmp_path: Path, *, document_count: int = 500) -> dict:
    return {
        "name": "toy_qa_location_frames",
        "seed": 42,
        "document_count": document_count,
        "direct_dialogue_ratio": 0.7,
        "output_dir": str(tmp_path / "qa_location_frames"),
        "verb_families": list(VERB_FAMILIES),
    }


def test_location_verb_frames_are_balanced_for_inspection_batch(tmp_path: Path) -> None:
    result = build_qa_location_verb_frames(_base_config(tmp_path), base_dir=ROOT)

    assert len(result.documents) == 500
    assert result.audit_report["counts"] == {"pass": 500, "warn": 0, "fail": 0}
    family_counts = Counter(row["verb_family"] for row in result.documents)
    assert set(family_counts) == set(VERB_FAMILIES)
    assert max(family_counts.values()) - min(family_counts.values()) <= 1
    style_counts = Counter(row["style"] for row in result.documents)
    assert style_counts == {"direct_dialogue": 350, "scene_qa": 150}
    assert result.manifest["counts"]["second_person_counts"] == {"third_person": 500}
    assert result.manifest["counts"]["answer_style_counts"] == {"location_fragment": 500}

    for row in result.documents:
        text = row["text"]
        assert text.count("?") == 1
        assert row["question"].startswith("Where did ")
        assert not row["second_person"]
        assert row["question_type"] == "where"
        assert "When " not in text
        assert "Why " not in text
        assert "How many" not in text
        assert "Where did you " not in text
        assert "bad:" not in text.lower()
        assert "\n" not in text


def test_location_verb_frame_artifacts_are_written(tmp_path: Path) -> None:
    result = build_qa_location_verb_frames(_base_config(tmp_path, document_count=16), base_dir=ROOT)

    assert result.outputs["documents"].exists()
    assert result.outputs["train"].exists()
    assert result.outputs["audit_report"].exists()
    assert result.outputs["manifest"].exists()
    train_lines = result.outputs["train"].read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 16
    manifest = json.loads(result.outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["counts"]["documents"] == 16
    assert manifest["counts"]["train_documents"] == 16


def test_send_family_splits_location_compatibility(tmp_path: Path) -> None:
    config = _base_config(tmp_path, document_count=64)
    config["verb_families"] = ["send_mail_ship_deliver"]

    result = build_qa_location_verb_frames(config, base_dir=ROOT)
    by_verb = {
        verb: [row for row in result.documents if row["verb"] == verb]
        for verb in {"send", "mail", "ship", "deliver"}
    }

    assert {verb: len(rows) for verb, rows in by_verb.items()} == {
        "send": 16,
        "mail": 16,
        "ship": 16,
        "deliver": 16,
    }
    assert {
        row["source_location_type"]
        for verb in ("mail", "ship")
        for row in by_verb[verb]
    } == {"destination_path"}
    assert {
        row["source_location_type"]
        for row in by_verb["deliver"]
    }.isdisjoint({"destination_path"})
    assert {row["answer_preposition"] for row in by_verb["deliver"]} == {"to"}


def test_allowed_verbs_filter_frames_inside_family(tmp_path: Path) -> None:
    config = _base_config(tmp_path, document_count=24)
    config["verb_families"] = ["drive_park"]
    config["allowed_verbs"] = ["drive"]

    result = build_qa_location_verb_frames(config, base_dir=ROOT)

    assert result.audit_report["counts"] == {"pass": 24, "warn": 0, "fail": 0}
    assert result.manifest["allowed_verbs"] == ["drive"]
    assert {row["verb"] for row in result.documents} == {"drive"}
    assert all("Where did " in row["question"] for row in result.documents)
    assert all(" park" not in row["question"].lower() for row in result.documents)


def test_named_location_experiment_is_config_gated(tmp_path: Path) -> None:
    config = _base_config(tmp_path, document_count=24)
    config["verb_families"] = ["travel_ride_fly"]
    config["named_location_ratio"] = 1.0
    config["excluded_named_locations"] = ["Canada", "Europe", "California"]

    result = build_qa_location_verb_frames(config, base_dir=ROOT)

    assert result.audit_report["counts"] == {"pass": 24, "warn": 0, "fail": 0}
    assert result.manifest["counts"]["named_location_counts"] == {"named": 24}
    assert all(row["named_location"] for row in result.documents)
    assert {row["answer_preposition"] for row in result.documents} == {"to"}
    assert {
        row["source_location_type"]
        for row in result.documents
    } <= {"named_city_destination", "named_country_destination", "named_state_destination"}

    text = "\n".join(row["text"] for row in result.documents).lower()
    assert "canada" not in text
    assert "europe" not in text
    assert "california" not in text


def test_location_string_exclusions_filter_answers_and_sources(tmp_path: Path) -> None:
    config = _base_config(tmp_path, document_count=64)
    config["verb_families"] = ["travel_ride_fly"]
    config["named_location_ratio"] = 1.0
    config["excluded_location_strings"] = ["to France", "France"]

    result = build_qa_location_verb_frames(config, base_dir=ROOT)

    assert result.audit_report["counts"] == {"pass": 64, "warn": 0, "fail": 0}
    assert result.manifest["excluded_location_strings"] == ["france", "to france"]
    text = "\n".join(row["text"] for row in result.documents).lower()
    assert "france" not in text
    assert all(row["answer_location"].lower() != "to france" for row in result.documents)
    assert all(row["source_location"].lower() != "france" for row in result.documents)


def test_second_person_where_questions_are_config_gated(tmp_path: Path) -> None:
    config = _base_config(tmp_path, document_count=32)
    config["second_person_ratio"] = 1.0
    config["second_person_full_answer_ratio"] = 1.0

    result = build_qa_location_verb_frames(config, base_dir=ROOT)

    assert result.audit_report["counts"] == {"pass": 32, "warn": 0, "fail": 0}
    assert result.manifest["counts"]["second_person_counts"] == {"second_person": 32}
    assert result.manifest["counts"]["answer_style_counts"] == {
        "first_person_location_sentence": 32
    }
    assert all(row["second_person"] for row in result.documents)
    assert all(row["question"].startswith("Where did you ") for row in result.documents)
    assert all(" B: I " in row["text"] for row in result.documents)
    assert any("Where did you leave " in row["text"] for row in result.documents)
    assert any("Where did you put it?" in row["text"] for row in result.documents)
    assert any("Where did you send the " in row["text"] for row in result.documents)


def test_leave_second_person_can_mix_it_and_explicit_objects(tmp_path: Path) -> None:
    config = _base_config(tmp_path, document_count=24)
    config["verb_families"] = ["leave_drop_hide_store"]
    config["second_person_ratio"] = 1.0
    config["second_person_full_answer_ratio"] = 0.0
    config["second_person_pronoun_weights"] = {
        "leave": 0.5,
        "drop": 1.0,
        "hide": 1.0,
        "store": 1.0,
    }

    result = build_qa_location_verb_frames(config, base_dir=ROOT)
    leave_questions = [
        row["question"] for row in result.documents if row["verb"] == "leave"
    ]

    assert any(question == "Where did you leave it?" for question in leave_questions)
    assert any(question.startswith("Where did you leave the ") for question in leave_questions)
    assert result.manifest["counts"]["question_object_reference_counts"]["it"] >= 1


def test_location_verb_frame_cli_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(
        "\n".join(
            [
                "name: cli_qa_location_frames",
                "seed: 42",
                "document_count: 16",
                "direct_dialogue_ratio: 0.7",
                f"output_dir: {output_dir}",
                "verb_families:",
                "  - send_mail_ship_deliver",
                "  - go_walk_head_return",
                "  - travel_ride_fly",
                "  - drive_park",
                "  - find_discover_spot_notice",
                "  - leave_drop_hide_store",
                "  - carry_bring_take",
                "  - put_place_set",
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
                / "generate_qa_location_verb_frames.py"
            ),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "QA location verb frames: cli_qa_location_frames" in completed.stdout
    assert (output_dir / "train.txt").exists()
    assert (output_dir / "documents.jsonl").exists()
