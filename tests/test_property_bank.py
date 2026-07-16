from __future__ import annotations

import json
import random
from pathlib import Path

from babysteps.concept_curriculum.property_bank import (
    DETERMINISTIC,
    FactCandidate,
    PropertyBankConfig,
    SourceParagraph,
    TemplateCandidate,
    build_property_bank,
    filter_fact_candidates,
    filter_templates,
    read_jsonl,
    select_initial_paragraphs,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _concept_row(concept_id: str, concept: str, styles: list[str]) -> dict:
    return {
        "concept_id": concept_id,
        "concept": concept,
        "concept_types": ["animal"],
        "paragraph_count": len(styles),
        "paragraphs": [
            {
                "id": f"{concept_id}__{style}",
                "style_id": style,
                "index": index,
                "text": f"{concept} paragraph for {style}.",
            }
            for index, style in enumerate(styles)
        ],
    }


def _paragraph_lookup(concept_id: str, styles: list[str]) -> dict[str, dict]:
    return {
        f"{concept_id}__{style}": {
            "id": f"{concept_id}__{style}",
            "style_id": style,
            "source_text": f"Source sentence for {style}.",
        }
        for style in styles
    }


def test_select_initial_paragraphs_uses_priority_styles_and_seeded_random() -> None:
    styles = [
        "natural_contrasts",
        "parts_and_properties",
        "contexts_and_interactions",
        "operation_and_components",
        "core_profile",
    ]
    row = _concept_row("iguana", "iguana", styles)
    selected = select_initial_paragraphs(
        concept_row=row,
        paragraph_lookup=_paragraph_lookup("iguana", styles),
        rng=random.Random(42),
    )

    assert [paragraph.style_id for paragraph in selected[:2]] == [
        "parts_and_properties",
        "operation_and_components",
    ]
    assert selected[2].style_id in {"core_profile", "contexts_and_interactions"}
    assert "natural_contrasts" not in [paragraph.style_id for paragraph in selected]
    assert {paragraph.selection_round for paragraph in selected} == {"initial"}


def test_select_initial_paragraphs_falls_back_to_natural_contrasts() -> None:
    styles = ["parts_and_properties", "operation_and_components", "natural_contrasts"]
    row = _concept_row("lamp", "lamp", styles)
    selected = select_initial_paragraphs(
        concept_row=row,
        paragraph_lookup=_paragraph_lookup("lamp", styles),
        rng=random.Random(42),
    )

    assert [paragraph.style_id for paragraph in selected] == [
        "parts_and_properties",
        "operation_and_components",
        "natural_contrasts",
    ]


def test_fact_filter_rejects_leaks_generic_unsupported_and_low_specificity() -> None:
    paragraph = SourceParagraph(
        concept_id="dragon",
        concept="dragon",
        concept_types=("animal",),
        paragraph_id="dragon__parts",
        style_id="parts_and_properties",
        text="A dragon has wings and scales. A dragon breathes fire.",
        text_source="source_text",
        selection_round="initial",
        selection_reason="priority_parts_and_properties",
        source_index=0,
    )
    facts = [
        FactCandidate(
            id="good",
            concept_id="dragon",
            concept="dragon",
            extraction_round="initial",
            fact="has wings and scales",
            evidence="A dragon has wings and scales.",
            source_paragraph_id="dragon__parts",
            fact_type="appearance",
            handle_safe=True,
            specificity="high",
            raw_index=0,
        ),
        FactCandidate(
            id="leak",
            concept_id="dragon",
            concept="dragon",
            extraction_round="initial",
            fact="a dragon has wings",
            evidence="A dragon has wings and scales.",
            source_paragraph_id="dragon__parts",
            fact_type="appearance",
            handle_safe=True,
            specificity="high",
            raw_index=1,
        ),
        FactCandidate(
            id="plural_leak",
            concept_id="dragon",
            concept="dragon",
            extraction_round="initial",
            fact="dragons have wings",
            evidence="A dragon has wings and scales.",
            source_paragraph_id="dragon__parts",
            fact_type="appearance",
            handle_safe=True,
            specificity="high",
            raw_index=2,
        ),
        FactCandidate(
            id="unsupported",
            concept_id="dragon",
            concept="dragon",
            extraction_round="initial",
            fact="has feathers",
            evidence="A dragon has feathers.",
            source_paragraph_id="dragon__parts",
            fact_type="appearance",
            handle_safe=True,
            specificity="high",
            raw_index=3,
        ),
        FactCandidate(
            id="generic",
            concept_id="dragon",
            concept="dragon",
            extraction_round="initial",
            fact="is useful",
            evidence="A dragon breathes fire.",
            source_paragraph_id="dragon__parts",
            fact_type="other",
            handle_safe=True,
            specificity="high",
            raw_index=4,
        ),
        FactCandidate(
            id="low",
            concept_id="dragon",
            concept="dragon",
            extraction_round="initial",
            fact="breathes fire",
            evidence="A dragon breathes fire.",
            source_paragraph_id="dragon__parts",
            fact_type="behavior",
            handle_safe=True,
            specificity="low",
            raw_index=5,
        ),
    ]

    accepted, rejections = filter_fact_candidates(facts, [paragraph])

    assert [fact.id for fact in accepted] == ["good"]
    assert rejections["concept_leak"] == 2
    assert rejections["unsupported_evidence"] == 1
    assert rejections["generic_fact"] == 1
    assert rejections["low_specificity"] == 1


def test_template_filter_requires_handle_and_rejects_leaks_and_wugs_like_text() -> None:
    templates = [
        TemplateCandidate(
            id="good",
            concept_id="dragon",
            concept="dragon",
            fact_id="fact",
            fact="has wings",
            template="{handle} opened its wings.",
            template_type="direct",
            raw_index=0,
        ),
        TemplateCandidate(
            id="missing",
            concept_id="dragon",
            concept="dragon",
            fact_id="fact",
            fact="has wings",
            template="It opened its wings.",
            template_type="direct",
            raw_index=1,
        ),
        TemplateCandidate(
            id="leak",
            concept_id="dragon",
            concept="dragon",
            fact_id="fact",
            fact="has wings",
            template="{handle} was a dragon with wings.",
            template_type="direct",
            raw_index=2,
        ),
        TemplateCandidate(
            id="therefore",
            concept_id="dragon",
            concept="dragon",
            fact_id="fact",
            fact="has wings",
            template="Therefore, {handle} had wings.",
            template_type="direct",
            raw_index=3,
        ),
        TemplateCandidate(
            id="binding",
            concept_id="dragon",
            concept="dragon",
            fact_id="fact",
            fact="has wings",
            template="A {handle} is a thing with wings.",
            template_type="direct",
            raw_index=4,
        ),
    ]

    accepted, rejections = filter_templates(templates)

    assert [template.id for template in accepted] == ["good"]
    assert rejections["missing_handle"] == 1
    assert rejections["concept_leak"] == 1
    assert rejections["wugs_like"] == 2


def test_build_property_bank_deterministic_smoke_writes_expected_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "comps"
    by_concept_path = input_dir / "documents_by_concept_less_repetitive_reasoning_low_preserve_details.jsonl"
    paragraphs_path = input_dir / "documents_less_repetitive_reasoning_low_preserve_details.jsonl"
    concepts = [
        ("sheep", "sheep", "animal"),
        ("toaster", "toaster", "device"),
    ]
    concept_rows = []
    paragraph_rows = []
    styles = ["parts_and_properties", "operation_and_components", "core_profile"]
    for concept_id, concept, concept_type in concepts:
        concept_rows.append(
            {
                "concept_id": concept_id,
                "concept": concept,
                "concept_types": [concept_type],
                "paragraph_count": 3,
                "paragraphs": [
                    {
                        "id": f"{concept_id}__{style}",
                        "style_id": style,
                        "index": index,
                        "text": f"{concept} fallback text.",
                    }
                    for index, style in enumerate(styles)
                ],
            }
        )
        for style in styles:
            paragraph_rows.append(
                {
                    "id": f"{concept_id}__{style}",
                    "style_id": style,
                    "source_text": (
                        f"A {concept} has a visible part for {style}. "
                        f"A {concept} makes a typical sound for {style}. "
                        f"A {concept} uses a familiar action for {style}. "
                        f"A {concept} holds a concrete detail for {style}."
                    ),
                }
            )
    _write_jsonl(by_concept_path, concept_rows)
    _write_jsonl(paragraphs_path, paragraph_rows)

    result = build_property_bank(
        config=PropertyBankConfig(
            input_dir=input_dir,
            output_dir=input_dir / "property_bank" / "v1_seed42",
            renderer=DETERMINISTIC,
            seed=42,
        )
    )

    for path in result.outputs.values():
        assert path.exists()
    manifest = json.loads(result.outputs["manifest"].read_text(encoding="utf-8"))
    audit = json.loads(result.outputs["audit_report"].read_text(encoding="utf-8"))
    facts = read_jsonl(result.outputs["extracted_facts_filtered"])
    templates = read_jsonl(result.outputs["surface_templates_filtered"])

    assert manifest["counts"]["concepts"] == 2
    assert manifest["counts"]["source_paragraphs"] == 6
    assert manifest["counts"]["extraction_prompts"] == 2
    assert manifest["counts"]["surface_template_prompts"] == 2
    assert manifest["coverage"]["concepts_with_min_facts_fraction"] == 1.0
    assert audit["deficient_concepts"] == []
    assert len(facts) == 24
    assert len(templates) == 72
    assert all("{handle}" in row["template"] for row in templates)
    assert not any(row["concept"] in row["template"].lower() for row in templates)
