from __future__ import annotations

import json

from babysteps.material_world.audit import audit_rows


def _row(example_id, text, style, material, material_class, obj, dynamics, negative=False):
    return {
        "id": example_id,
        "text": text,
        "metadata": {
            "style_id": style,
            "material": material,
            "material_class": material_class,
            "object": obj,
            "dynamics": dynamics,
            "contains_negative_reporting": negative,
        },
    }


def _small_config():
    return {
        "version": "test",
        "openings": {
            "min_corpus_examples_for_rate_flags": 25,
            "first_5_warning_rate": 0.2,
            "first_8_warning_rate": 0.2,
            "hard_rate": 0.8,
            "min_group_count": 2,
            "watched_prefixes": ["when the"],
        },
        "connectives": {
            "phrases": ["because", "unlike", "instead of", "rather than", "while"],
            "because_example_rate_warning": 0.9,
            "contrast_phrases": ["unlike", "instead of", "rather than"],
            "contrast_combined_example_rate_warning": 0.9,
            "per_example_marker_warning_count": 3,
        },
        "styles": {
            "min_corpus_examples_for_balance_warnings": 25,
            "aliases": {},
            "targets": {"everyday_narrative": 0.5, "comparison_paragraph": 0.5},
            "target_tolerance_points": 0.6,
            "min_rate_warning": 0.0,
            "max_rate_warning": 1.0,
        },
        "negative_reporting": {
            "phrases": ["not", "unlike", "instead of"],
            "target_min_rate": 0.2,
            "target_max_rate": 0.3,
            "warning_min_rate": 0.0,
            "warning_max_rate": 1.0,
        },
        "lengths": {
            "buckets": {
                "short_64": [35, 60],
                "medium_128": [70, 120],
                "long_256": [140, 220],
                "xlong_512": [280, 420],
            },
            "target_rates": {},
            "single_bucket_warning_rate": 1.0,
            "too_short_words": 5,
            "too_long_words": 450,
            "style_length_variance_min_examples": 10,
            "style_length_std_warning": 8.0,
        },
        "dedupe": {
            "jaccard_warning_threshold": 0.8,
            "jaccard_hard_drop_threshold": 0.9,
            "min_words_for_jaccard": 4,
            "max_examples_for_all_pairs": 100,
            "length_ratio_min": 0.5,
            "repeated_first_12_warning_count": 2,
            "repeated_first_12_hard_count": 5,
            "eight_gram_warning_count": 3,
        },
        "collapse": {
            "min_examples_per_material": 3,
            "dominant_dynamic_rate_warning": 0.55,
        },
        "template_phrases": {
            "phrases": ["made of", "this shows", "the material"],
            "phrase_rate_warning": 1.0,
            "top_ngram_n": 4,
            "top_ngram_limit": 5,
        },
    }


def test_audit_flags_exact_duplicate_and_keeps_first_copy():
    text = "When the glass cup falls on tile, it can crack because glass is brittle."
    rows = [
        _row("a", text, "everyday_narrative", "glass", "brittle_hard_solids", "cup", ["cracks"]),
        _row("b", text, "everyday_narrative", "glass", "brittle_hard_solids", "cup", ["cracks"]),
    ]

    result = audit_rows(rows, config=_small_config())

    by_id = {row["id"]: row for row in result.audited_rows}
    assert by_id["a"]["audit_status"] in {"keep", "warn"}
    assert by_id["b"]["audit_status"] == "drop"
    assert "b" in result.drop_ids
    assert result.report["dedupe"]["exact_duplicate_count"] == 1


def test_audit_reports_material_dynamic_collapse_without_overfiltering():
    rows = [
        _row(
            f"g{i}",
            f"Glass cup example {i} cracks after a hard bump, but the scene stays ordinary.",
            "comparison_paragraph",
            "glass",
            "brittle_hard_solids",
            "cup",
            ["cracks"],
            negative=True,
        )
        for i in range(4)
    ]
    rows.append(
        _row(
            "r1",
            "Rubber cup example bends after a small bump and returns toward its shape.",
            "everyday_narrative",
            "rubber",
            "elastic_materials",
            "cup",
            ["bends"],
        )
    )

    result = audit_rows(rows, config=_small_config())

    warnings = result.report["material_dynamic_summary"]["warnings"]
    assert warnings
    assert warnings[0]["material"] == "glass"
    assert warnings[0]["dynamic"] == "cracks"
    assert result.report["examples_dropped"] == 0


def test_audit_report_is_json_serializable():
    rows = [
        _row(
            "x1",
            "A paper cup may soften after water sits inside it for a while.",
            "everyday_narrative",
            "paper",
            "paper_cardboard",
            "cup",
            ["softens"],
        )
    ]

    result = audit_rows(rows, config=_small_config())

    json.dumps(result.report)
