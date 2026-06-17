from __future__ import annotations

import json

from babysteps.material_world.ewok_coverage import (
    audit_ewok_coverage,
    build_coverage_audit,
    count_term_mentions,
    load_ewok_targets,
    load_generated_corpus,
)


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_ewok_targets_extracts_concepts_and_anchors(tmp_path) -> None:
    ewok_dir = tmp_path / "ewok"
    ewok_dir.mkdir()
    _write_jsonl(
        ewok_dir / "material-dynamics.jsonl",
        [
            {
                "Domain": "material-dynamics",
                "ConceptA": "stir",
                "ConceptB": "wrinkle",
                "ContextType": "indirect",
                "ContextDiff": "material",
                "TargetDiff": "concept swap",
                "Context1": "Ali sees the nitrogen.",
                "Context2": "Ali sees the twill.",
                "Target1": "Ali stirs it.",
                "Target2": "Ali wrinkles it.",
            },
            {
                "Domain": "material-dynamics",
                "ConceptA": "pour",
                "ConceptB": "squeeze",
                "ContextType": "direct",
                "ContextDiff": "material",
                "TargetDiff": "concept swap",
                "Context1": "Chao sees something that is liquid.",
                "Context2": "Chao sees something that is rigid.",
                "Target1": "Chao pours it.",
                "Target2": "Chao squeezes it.",
            },
        ],
    )

    targets = load_ewok_targets(ewok_dir, domains=["material-dynamics"])
    domain = targets["material-dynamics"]

    assert domain["concept_counts"] == {
        "pour": 1,
        "squeeze": 1,
        "stir": 1,
        "wrinkle": 1,
    }
    assert domain["anchor_counts"] == {
        "liquid": 1,
        "nitrogen": 1,
        "rigid": 1,
        "twill": 1,
    }


def test_build_coverage_audit_reports_missing_lexical_anchors(tmp_path) -> None:
    ewok_dir = tmp_path / "ewok"
    ewok_dir.mkdir()
    _write_jsonl(
        ewok_dir / "material-dynamics.jsonl",
        [
            {
                "Domain": "material-dynamics",
                "ConceptA": "stir",
                "ConceptB": "wrinkle",
                "ContextType": "indirect",
                "ContextDiff": "material",
                "TargetDiff": "concept swap",
                "Context1": "Ali sees the nitrogen.",
                "Context2": "Ali sees the twill.",
                "Target1": "Ali stirs it.",
                "Target2": "Ali wrinkles it.",
            }
        ],
    )
    generated_path = tmp_path / "examples.jsonl"
    _write_jsonl(
        generated_path,
        [
            {
                "id": "x1",
                "text": "Twill is a woven fabric. A twill jacket can wrinkle when folded.",
                "metadata": {
                    "material": "twill",
                    "dynamics": ["wrinkle"],
                },
            }
        ],
    )

    targets = load_ewok_targets(ewok_dir, domains=["material-dynamics"])
    generated = load_generated_corpus(generated_path)
    report = build_coverage_audit(ewok_targets=targets, generated=generated)
    summary = report["summary"]["domains"]["material-dynamics"]

    assert summary["concepts_covered"] == "1/2"
    assert "wrinkle" not in summary["missing_concepts"]
    assert "stir" in summary["missing_concepts"]
    assert summary["lexical_anchors_covered"] == "1/2"
    assert "twill" not in summary["missing_lexical_anchors"]
    assert "nitrogen" in summary["missing_lexical_anchors"]


def test_audit_ewok_coverage_writes_json_report(tmp_path) -> None:
    ewok_dir = tmp_path / "ewok"
    ewok_dir.mkdir()
    _write_jsonl(
        ewok_dir / "material-properties.jsonl",
        [
            {
                "Domain": "material-properties",
                "ConceptA": "soft",
                "ConceptB": "hard",
                "ContextType": "indirect",
                "ContextDiff": "material",
                "TargetDiff": "concept swap",
                "Context1": "The bin is made of jelly.",
                "Context2": "The bin is made of glass.",
                "Target1": "The bin is soft.",
                "Target2": "The bin is hard.",
            }
        ],
    )
    generated_path = tmp_path / "train.txt"
    generated_path.write_text(
        "Jelly is soft and wobbly. Glass is hard and rigid.\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    audit_ewok_coverage(
        ewok_dir=ewok_dir,
        generated_path=generated_path,
        report_json=report_path,
        domains=["material-properties"],
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report["summary"]["domains"]["material-properties"]
    assert summary["concepts_covered"] == "2/2"
    assert summary["lexical_anchors_covered"] == "2/2"


def test_count_term_mentions_handles_simple_variants() -> None:
    text = "Gases spread out. Play dough is soft. The play-doh toy bends."

    assert count_term_mentions(text, "gas") == 1
    assert count_term_mentions(text, "play-doh") == 2
