from __future__ import annotations

import json
from pathlib import Path

from babysteps.container_state import (
    EVAL_SPLITS,
    HELDOUT_STYLES,
    TRAIN_STYLES,
    build_multidomain_corpus,
    load_domain_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs/container_state/multidomain_catalog_v3.yaml"


def test_catalog_has_disjoint_domain_and_surface_splits() -> None:
    training, heldout = load_domain_catalog(CATALOG)

    assert len(training) == 12
    assert len(heldout) == 3
    assert not ({domain.id for domain in training} & {domain.id for domain in heldout})
    for domain in (*training, *heldout):
        assert not (set(domain.train_objects) & set(domain.validation_objects))
        assert not (set(domain.train_containers) & set(domain.validation_containers))


def test_multidomain_generation_is_audited_and_deterministic(tmp_path: Path) -> None:
    def build(output: Path):
        return build_multidomain_corpus(
            {
                "name": "multidomain_test",
                "seed": 19,
                "output_dir": str(output),
                "catalog": str(CATALOG),
                "train_pairs": 48,
                "eval_pairs_per_split": 12,
                "surface_classifier_max_auc": 1.0,
            },
            base_dir=ROOT,
        )

    first = build(tmp_path / "first")
    second = build(tmp_path / "second")

    assert [pair["text1"] for pair in first.train_pairs] == [
        pair["text1"] for pair in second.train_pairs
    ]
    assert set(first.eval_pairs) == set(EVAL_SPLITS)
    assert len(first.train_pairs) == 48
    assert all(len(pairs) == 12 for pairs in first.eval_pairs.values())
    assert all(pair["Target1"] != pair["Target2"] for pair in first.train_pairs)
    assert all(pair["target_items1"] and pair["target_items2"] for pair in first.train_pairs)
    assert sum(pair["equal_cardinality"] for pair in first.train_pairs) == 24
    assert first.audit_report["leakage"]["normalized_cross_split_collisions"] == 0
    assert first.audit_report["leakage"]["train_validation_surface_overlap"] == []

    iid_rows = [
        json.loads(line)
        for line in first.outputs["iid_entity_tracking"].read_text(encoding="utf-8").splitlines()
    ]
    assert len(iid_rows) == 24
    assert all(len(row["options"]) == 5 for row in iid_rows)
    assert all(row["target_index"] == 0 for row in iid_rows)
    assert all(len(set(row["option_item_counts"])) == 1 for row in iid_rows)
    assert {pair["renderer"] for pair in first.eval_pairs["template_ood"]} <= set(
        HELDOUT_STYLES
    )
    assert {pair["renderer"] for pair in first.eval_pairs["iid"]} <= set(TRAIN_STYLES)


def test_all_five_choice_distractors_match_target_cardinality(tmp_path: Path) -> None:
    result = build_multidomain_corpus(
        {
            "name": "multidomain_options_test",
            "seed": 23,
            "output_dir": str(tmp_path / "output"),
            "catalog": str(CATALOG),
            "train_pairs": 24,
            "eval_pairs_per_split": 6,
            "surface_classifier_max_auc": 1.0,
        },
        base_dir=ROOT,
    )
    for split in EVAL_SPLITS:
        rows = [
            json.loads(line)
            for line in result.outputs[f"{split}_entity_tracking"]
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        for row in rows:
            assert len(set(row["option_item_counts"])) == 1
