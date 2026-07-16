from __future__ import annotations

import json
import math

import numpy as np

from babysteps.analysis.word2vec.corpus import TokenizationConfig, tokenize_text
from babysteps.analysis.word2vec.ewok_eval import (
    EwokItem,
    _build_item_record,
    _half_credit_accuracy_from_margin,
    _paper_context_sensitivity_combined,
    _pooled_vector,
    load_ewok_items,
)
from babysteps.analysis.word2vec.reports import domain_summary_rows, item_score_rows


class ToyVectors:
    def __init__(self, values: dict[str, list[float]]) -> None:
        self.values = {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def __getitem__(self, key: str):
        return self.values[key]


def test_tokenize_text_is_lowercase_word_oriented() -> None:
    tokens = tokenize_text("The balloon isn't sturdy, but it broke.", TokenizationConfig())
    assert tokens == ["the", "balloon", "isn't", "sturdy", "but", "it", "broke"]


def test_pooled_vector_drops_oov_tokens() -> None:
    vectors = ToyVectors({"hot": [1, 0], "warm": [1, 1]})
    pooled = _pooled_vector("hot cold warm", vectors, tokenization=TokenizationConfig())

    assert pooled.tokens == ["hot", "cold", "warm"]
    assert pooled.in_vocab_tokens == ["hot", "warm"]
    assert pooled.oov_tokens == ["cold"]
    assert pooled.in_vocab_count == 2
    assert pooled.total_tokens == 3
    assert np.allclose(pooled.vector, np.asarray([1.0, 0.5]))


def test_half_credit_margin_logic() -> None:
    assert _half_credit_accuracy_from_margin(0.2) == 1.0
    assert _half_credit_accuracy_from_margin(0.0) == 0.5
    assert _half_credit_accuracy_from_margin(-0.2) == 0.0
    assert math.isnan(_half_credit_accuracy_from_margin(math.nan))


def test_paper_context_sensitivity_counts_both_correct_and_both_reversed() -> None:
    assert _paper_context_sensitivity_combined(1.0, 1.0) == 1.0
    assert _paper_context_sensitivity_combined(0.0, 0.0) == 1.0
    assert _paper_context_sensitivity_combined(1.0, 0.0) == 0.0


def test_build_item_record_scores_paper_context_sensitivity() -> None:
    vectors = ToyVectors(
        {
            "alpha": [1, 0],
            "beta": [0, 1],
            "apple": [1, 0],
            "berry": [0, 1],
        }
    )
    item = EwokItem(
        item_id="toy",
        domain="toy-domain",
        context_1="alpha",
        target_1="apple",
        context_2="beta",
        target_2="berry",
    )

    record = _build_item_record(item, vectors, tokenization=TokenizationConfig())

    assert record.status == "counted"
    assert record.k1 > 0
    assert record.k2 > 0
    assert record.acc1 == 1.0
    assert record.acc2 == 1.0
    assert record.context_sensitivity == 1.0


def test_both_reversed_pattern_is_context_sensitive() -> None:
    vectors = ToyVectors(
        {
            "alpha": [1, 0],
            "beta": [0, 1],
            "apple": [0, 1],
            "berry": [1, 0],
        }
    )
    item = EwokItem(
        item_id="toy",
        domain="toy-domain",
        context_1="alpha",
        target_1="apple",
        context_2="beta",
        target_2="berry",
    )

    record = _build_item_record(item, vectors, tokenization=TokenizationConfig())

    assert record.k1 < 0
    assert record.k2 < 0
    assert record.acc1 == 0.0
    assert record.acc2 == 0.0
    assert record.context_sensitivity == 1.0


def test_load_ewok_items_from_jsonl(tmp_path) -> None:
    path = tmp_path / "material-properties.jsonl"
    path.write_text(
        json.dumps(
            {
                "Domain": "material-properties",
                "ConceptA": "fragile",
                "ConceptB": "sturdy",
                "ContextType": "direct",
                "ContextDiff": "antonym",
                "TargetDiff": "concept swap",
                "Context1": "It broke.",
                "Context2": "It remained intact.",
                "Target1": "It is fragile.",
                "Target2": "It is sturdy.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    items = load_ewok_items(path)

    assert len(items) == 1
    assert items[0].item_id == "material-properties:1"
    assert items[0].domain == "material-properties"
    assert items[0].concept_a == "fragile"
    assert items[0].target_diff == "concept swap"


def test_report_rows_include_context_sensitivity_and_domain_summary() -> None:
    vectors = ToyVectors(
        {
            "alpha": [1, 0],
            "beta": [0, 1],
            "apple": [1, 0],
            "berry": [0, 1],
        }
    )
    item = EwokItem(
        item_id="toy",
        domain="toy-domain",
        context_1="alpha",
        target_1="apple",
        context_2="beta",
        target_2="berry",
    )
    record = _build_item_record(item, vectors, tokenization=TokenizationConfig())

    item_rows = item_score_rows([record])
    domain_rows = domain_summary_rows([record])

    assert item_rows[0]["context_sensitivity"] == "1.000000"
    assert item_rows[0]["S11"] == "1.000000"
    assert domain_rows[0]["domain"] == "toy-domain"
    assert domain_rows[0]["items_counted"] == 1
    assert domain_rows[0]["context_sensitivity_mean"] == "1.000000"
