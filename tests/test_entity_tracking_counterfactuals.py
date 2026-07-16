from __future__ import annotations

from types import SimpleNamespace

import torch

from babysteps.evaluation.entity_tracking_counterfactuals import (
    PROBE_FAMILIES,
    build_scored_item_records,
    generate_counterfactual_items,
    lexical_recency_best_indices,
    read_counterfactual_jsonl,
    summarize_scored_records,
    validate_pairs,
    write_counterfactual_jsonl,
)
from babysteps.evaluation.mlm_likelihood import (
    MlmTextSpanRequest,
    encode_mlm_text_span,
    score_mlm_text_spans,
)


class FakeMlmTokenizer:
    pad_token_id = 0
    mask_token_id = 99
    cls_token_id = 101
    sep_token_id = 102
    eos_token_id = 102
    bos_token_id = None

    def __init__(self) -> None:
        self.vocab = {"context": 1, "good": 2, "bad": 3}

    def _core(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids = []
        offsets = []
        search_from = 0
        for token in text.split():
            start = text.index(token, search_from)
            end = start + len(token)
            ids.append(self.vocab[token])
            offsets.append((start, end))
            search_from = end
        return ids, offsets

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids, _ = self._core(text)
        return [self.cls_token_id, *ids, self.sep_token_id] if add_special_tokens else ids

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        ids, offsets = self._core(text)
        output: dict[str, list[int] | list[tuple[int, int]]] = {
            "input_ids": [self.cls_token_id, *ids, self.sep_token_id],
            "attention_mask": [1] * (len(ids) + 2),
        }
        if return_offsets_mapping:
            output["offset_mapping"] = [(0, 0), *offsets, (0, 0)]
        return output


class MaskPositionPrefersGoodModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        del attention_mask
        logits = torch.zeros((*input_ids.shape, 128), device=input_ids.device)
        for row in range(input_ids.shape[0]):
            for index in range(input_ids.shape[1]):
                if int(input_ids[row, index]) == FakeMlmTokenizer.mask_token_id:
                    logits[row, index, 2] = 5.0
        return SimpleNamespace(logits=logits)


def test_generate_counterfactual_items_is_deterministic_and_balanced() -> None:
    first = generate_counterfactual_items(pairs_per_family=6, seed=42)
    second = generate_counterfactual_items(pairs_per_family=6, seed=42)

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len(first) == 4 * 6 * 2
    validate_pairs(first, expected_pairs_per_family=6)
    assert {item.family for item in first} == set(PROBE_FAMILIES)


def test_pair_interventions_hold_expected_features_constant() -> None:
    items = generate_counterfactual_items(pairs_per_family=3, seed=7)
    pairs = {}
    for item in items:
        pairs.setdefault(item.pair_id, []).append(item)

    for pair in pairs.values():
        first, second = pair
        assert first.initial_state == second.initial_state
        assert first.options == second.options
        assert first.query_box == second.query_box
        assert "nothing." not in first.options
        assert len({option.count("the ") for option in first.options}) == 1
        if first.family == "irrelevant_reordering":
            assert first.target_index == second.target_index
            near = first if first.condition == "near" else second
            far = second if first.condition == "near" else first
            assert near.last_touch_distance == 0
            assert far.last_touch_distance > 0
            assert near.target_index in lexical_recency_best_indices(near)
            assert far.target_index not in lexical_recency_best_indices(far)
        else:
            assert first.target_index != second.target_index
            assert first.last_touch == second.last_touch


def test_counterfactual_jsonl_round_trip(tmp_path) -> None:
    items = generate_counterfactual_items(pairs_per_family=2, seed=13)
    path = write_counterfactual_jsonl(tmp_path / "pairs.jsonl", items)

    restored = read_counterfactual_jsonl(path)

    assert [item.to_dict() for item in restored] == [item.to_dict() for item in items]


def test_summary_distinguishes_changed_and_stable_targets() -> None:
    items = generate_counterfactual_items(pairs_per_family=1, seed=21)
    scores = []
    counts = []
    for item in items:
        item_scores = [0.0] * len(item.options)
        item_scores[item.target_index] = 10.0
        scores.append(item_scores)
        counts.append([1] * len(item.options))
    records = build_scored_item_records(
        items=items,
        option_scores=scores,
        scored_token_counts=counts,
    )

    summary = summarize_scored_records(records, bootstrap_seed=5)

    for family in PROBE_FAMILIES:
        values = summary["families"][family]
        assert values["accuracy"]["estimate"] == 1.0
        assert values["both_correct"]["estimate"] == 1.0
        if family == "irrelevant_reordering":
            assert values["prediction_invariant"]["estimate"] == 1.0
            assert values["near_minus_far_margin_mean"] == 0.0
        else:
            assert values["prediction_changed"]["estimate"] == 1.0
            assert values["directional_target_shift_positive"]["estimate"] == 1.0


def test_mlm_span_scoring_masks_and_scores_the_same_target_position() -> None:
    tokenizer = FakeMlmTokenizer()
    encoded = encode_mlm_text_span(
        tokenizer,
        MlmTextSpanRequest(text="context good", target_start_char=len("context ")),
    )

    assert encoded.input_ids == (101, 1, 2, 102)
    assert encoded.target_indices == (2,)
    assert encoded.target_token_ids == (2,)

    scores = score_mlm_text_spans(
        model=MaskPositionPrefersGoodModel(),
        tokenizer=tokenizer,
        requests=[
            MlmTextSpanRequest("context good", len("context ")),
            MlmTextSpanRequest("context bad", len("context ")),
        ],
        batch_size=2,
    )

    assert scores[0].sum_logprob > scores[1].sum_logprob
    assert scores[0].scored_token_count == 1
