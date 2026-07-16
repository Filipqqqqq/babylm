from __future__ import annotations

import json
from pathlib import Path

from babysteps.container_state import build_counterfactual_corpus


ROOT = Path(__file__).resolve().parents[1]


def test_counterfactual_corpus_has_nonempty_paired_targets(tmp_path: Path) -> None:
    result = build_counterfactual_corpus(
        {
            "name": "counterfactual_test",
            "seed": 7,
            "output_dir": str(tmp_path / "output"),
            "lexicons": "configs/container_state/lexicons_v1.yaml",
            "pool_min_words": 2_000,
            "eval_pairs": 12,
            "world": {
                "container_count_weights": {3: 1.0},
                "min_items_per_container": 2,
                "max_items_per_container": 3,
            },
            "operations": {"common_count_weights": {0: 1, 1: 1, 2: 1}},
        },
        base_dir=ROOT,
    )

    assert result.audit_report["counts"]["empty_targets"] == 0
    assert result.audit_report["counts"]["identical_targets"] == 0
    assert result.audit_report["counts"]["train_eval_world_overlap"] == 0
    assert set(result.audit_report["counterfactual_kind_counts"]) == {
        "add",
        "move",
        "remove",
    }
    assert all(pair["Target1"] != pair["Target2"] for pair in result.train_pairs)
    assert all(pair["target_items1"] and pair["target_items2"] for pair in result.train_pairs)

    rows = [
        json.loads(line)
        for line in result.outputs["train_counterfactual_rank"]
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == len(result.train_pairs)
    assert {row["rank_objective"] for row in rows} == {"counterfactual_rank"}
