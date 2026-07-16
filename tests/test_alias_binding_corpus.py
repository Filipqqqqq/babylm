from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from babysteps.concept_curriculum.alias_binding_corpus import (
    AliasBindingConfig,
    CARRIER_TYPES,
    COLOR_NONCE_NAMES,
    NONCE_NAMES,
    ORDINARY_NAMES,
    PROPERTY_ORDER_MODES,
    build_alias_binding_corpus,
    concept_with_article,
    load_sampling_index,
    parse_int_tuple,
    read_jsonl,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _toy_bank(path: Path) -> None:
    facts = []
    templates = []
    concepts = [
        ("cactus", "cactus"),
        ("violin", "violin"),
        ("turtle", "turtle"),
    ]
    fact_types = ("appearance", "function")
    template_types = ("direct", "retrieval")
    for concept_id, concept in concepts:
        for fact_index, fact_type in enumerate(fact_types):
            fact_id = f"{concept_id}__fact{fact_index}"
            fact_text = (
                f"had a concrete {fact_type} detail"
                if fact_type == "appearance"
                else f"used a concrete {fact_type} action"
            )
            facts.append(
                {
                    "id": fact_id,
                    "concept_id": concept_id,
                    "concept": concept,
                    "fact": fact_text,
                    "fact_type": fact_type,
                    "evidence": f"A {concept} {fact_text}.",
                }
            )
            for template_type in template_types:
                template_id = f"{fact_id}__{template_type}"
                if template_type == "direct":
                    template = f"{{handle}} {fact_text}."
                else:
                    template = f"The one that {fact_text} was {{handle}}."
                templates.append(
                    {
                        "id": template_id,
                        "concept_id": concept_id,
                        "concept": concept,
                        "fact_id": fact_id,
                        "fact": fact_text,
                        "template": template,
                        "template_type": template_type,
                    }
                )
    _write_jsonl(path / "extracted_facts_filtered.jsonl", facts)
    _write_jsonl(path / "surface_templates_filtered.jsonl", templates)


def test_load_sampling_index_joins_fact_types(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    _toy_bank(bank)

    index = load_sampling_index(
        facts_path=bank / "extracted_facts_filtered.jsonl",
        templates_path=bank / "surface_templates_filtered.jsonl",
    )

    assert index.concept_ids == ("cactus", "turtle", "violin")
    assert index.fact_types == ("appearance", "function")
    assert index.template_types == ("direct", "retrieval")
    assert {record.fact_type for record in index.by_concept["cactus"]} == {
        "appearance",
        "function",
    }


def test_build_alias_binding_corpus_balances_core_dimensions(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    _toy_bank(bank)
    result = build_alias_binding_corpus(
        AliasBindingConfig(
            property_bank_dir=bank,
            output_dir=tmp_path / "alias",
            document_count=60,
            seed=42,
        )
    )

    concept_counts: Counter[str] = Counter()
    template_type_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    carrier_counts: Counter[str] = Counter()
    property_order_counts: Counter[str] = Counter()
    for document in result.documents:
        sequence_counts[document.sequence_family] += 1
        carrier_counts[document.carrier_type] += 1
        property_order_counts[document.property_order_mode] += 1
        for binding in document.bindings:
            concept_counts[binding.concept_id] += 1
            template_type_counts[binding.template_type] += 1

    assert len(result.documents) == 60
    assert set(concept_counts.values()) == {40}
    assert template_type_counts == {"direct": 60, "retrieval": 60}
    assert set(sequence_counts.values()) == {10}
    assert set(carrier_counts.values()) == {20}
    assert set(property_order_counts.values()) == {12}
    assert all("{handle}" not in document.text for document in result.documents)


def test_build_alias_binding_corpus_writes_mixable_outputs(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    _toy_bank(bank)
    output = tmp_path / "alias"

    result = build_alias_binding_corpus(
        AliasBindingConfig(
            property_bank_dir=bank,
            output_dir=output,
            document_count=5,
            seed=7,
        )
    )

    for path in result.outputs.values():
        assert path.exists()
    docs = read_jsonl(output / "documents.jsonl")
    train_lines = (output / "train.txt").read_text(encoding="utf-8").splitlines()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "audit_report.json").read_text(encoding="utf-8"))
    sampling_index = json.loads((output / "sampling_index.json").read_text(encoding="utf-8"))

    assert len(docs) == 5
    assert len(train_lines) == 5
    assert [row["text"] for row in docs] == train_lines
    assert manifest["counts"]["documents"] == 5
    assert manifest["counts"]["bindings"] == 10
    assert set(manifest["sampling"]["property_order_modes"]) == set(PROPERTY_ORDER_MODES)
    assert audit["counts"]["documents"] == 5
    assert "property_order_modes" in audit["balance"]
    assert sampling_index["concept_count"] == 3
    assert all(row["word_count"] > 0 for row in docs)
    assert all(row["bindings"] for row in docs)


def test_parse_int_tuple_requires_two_or_more_bindings() -> None:
    assert parse_int_tuple("2,3") == (2, 3)
    assert concept_with_article("accordion") == "an accordion"
    assert concept_with_article("turtle") == "a turtle"
    try:
        parse_int_tuple("1")
    except Exception as exc:  # argparse raises ArgumentTypeError.
        assert ">= 2" in str(exc)
    else:  # pragma: no cover - defensive failure path.
        raise AssertionError("expected parse_int_tuple to reject one binding")


def test_carrier_pools_are_balanced_at_eight_handles() -> None:
    pools = {
        "ordinary_name": ORDINARY_NAMES,
        "nonce_name": NONCE_NAMES,
        "color_nonce": COLOR_NONCE_NAMES,
    }

    assert tuple(pools) == CARRIER_TYPES
    assert {len(pool) for pool in pools.values()} == {8}


def test_reverse_property_order_metadata(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    _toy_bank(bank)

    result = build_alias_binding_corpus(
        AliasBindingConfig(
            property_bank_dir=bank,
            output_dir=tmp_path / "alias",
            document_count=1,
            seed=42,
            property_order_modes=("reverse_order",),
        )
    )

    document = result.documents[0]
    binding_handles = tuple(binding.handle for binding in document.bindings)
    assert document.property_order_mode == "reverse_order"
    assert document.property_order_handles == tuple(reversed(binding_handles))


def test_ordinary_name_bindings_use_role_play_wording(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    _toy_bank(bank)

    result = build_alias_binding_corpus(
        AliasBindingConfig(
            property_bank_dir=bank,
            output_dir=tmp_path / "alias",
            document_count=1,
            seed=42,
            carrier_types=("ordinary_name",),
            sequence_families=("role_binding",),
        )
    )

    text = result.documents[0].text
    assert "played as" in text or "acted as" in text
