from __future__ import annotations

from pathlib import Path

from babysteps.evaluation.babylm_fast_eval import (
    discover_checkpoints,
    flatten_summary_rows,
    parse_best_temperature_report,
    parse_reading_report,
)


def test_parse_best_temperature_report(tmp_path: Path) -> None:
    report = tmp_path / "best_temperature_report.txt"
    report.write_text(
        "\n".join(
            [
                "TEMPERATURE: 1.00",
                "",
                "### UID ACCURACY",
                "foo: 75.50",
                "bar: 25.00",
                "",
                "### AVERAGE ACCURACY",
                "50.25",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_best_temperature_report(report)

    assert parsed["temperature"] == 1.0
    assert parsed["average"] == 50.25
    assert parsed["sections"]["uid"] == {"foo": 75.50, "bar": 25.00}


def test_parse_reading_report(tmp_path: Path) -> None:
    report = tmp_path / "report.txt"
    report.write_text(
        "EYE TRACKING SCORE: 0.32\nSELF-PACED READING SCORE: 0.01\n",
        encoding="utf-8",
    )
    correlations = tmp_path / "correlations.txt"
    correlations.write_text("RTfirstfix\t0.4901\nN400\t-0.0717\n", encoding="utf-8")

    parsed = parse_reading_report(report, correlations)

    assert parsed["scores"]["eye_tracking_score"] == 0.32
    assert parsed["scores"]["self_paced_reading_score"] == 0.01
    assert parsed["correlations"] == {"RTfirstfix": 0.4901, "N400": -0.0717}


def test_discover_checkpoints_sorts_common_names(tmp_path: Path) -> None:
    for name in ["final", "checkpoint-epoch-0002", "checkpoint-epoch-0001"]:
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_text("", encoding="utf-8")

    checkpoints = discover_checkpoints(tmp_path)

    assert [checkpoint.name for checkpoint in checkpoints] == [
        "checkpoint-epoch-0001",
        "checkpoint-epoch-0002",
        "final",
    ]


def test_flatten_summary_rows_includes_average_and_subtasks() -> None:
    summary = {
        "checkpoints": [
            {
                "checkpoint": "checkpoint-epoch-0001",
                "checkpoint_path": "/tmp/checkpoint-epoch-0001",
                "tasks": {
                    "blimp_fast": {
                        "average": 81.13,
                        "sections": {
                            "uid": {
                                "determiner_noun_agreement_1": 99.50,
                            }
                        },
                    },
                    "reading": {
                        "scores": {"eye_tracking_score": 0.32},
                        "correlations": {"N400": -0.0717},
                    },
                },
            }
        ]
    }

    rows = flatten_summary_rows(summary)

    assert {
        "checkpoint": "checkpoint-epoch-0001",
        "checkpoint_path": "/tmp/checkpoint-epoch-0001",
        "task": "blimp_fast",
        "metric_scope": "average",
        "metric_name": "average",
        "value": 81.13,
    } in rows
    assert {
        "checkpoint": "checkpoint-epoch-0001",
        "checkpoint_path": "/tmp/checkpoint-epoch-0001",
        "task": "blimp_fast",
        "metric_scope": "uid",
        "metric_name": "determiner_noun_agreement_1",
        "value": 99.50,
    } in rows
