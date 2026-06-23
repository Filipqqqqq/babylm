from __future__ import annotations

from types import SimpleNamespace

import torch

from babysteps.evaluation.blimp_likelihood import BlimpItem
from babysteps.evaluation.comps_likelihood import CompsItem
from babysteps.evaluation.entity_tracking_likelihood import EntityTrackingItem
from babysteps.evaluation.mntp_likelihood import (
    MntpTextSpanRequest,
    encode_mntp_text_span,
    evaluate_blimp_mntp_likelihood,
    evaluate_comps_mntp_likelihood,
    evaluate_entity_tracking_mntp_likelihood,
    score_mntp_text_spans,
)


class FakeMntpTokenizer:
    pad_token_id = 0
    mask_token_id = 99
    cls_token_id = 101
    sep_token_id = 102
    eos_token_id = 102
    bos_token_id = None
    cls_token = "[CLS]"
    mask_token = "[MASK]"
    eos_token = "[SEP]"
    pad_token = "[PAD]"

    def __init__(self) -> None:
        self.vocab = {
            "context": 1,
            "other": 2,
            "good": 3,
            "bad": 4,
            "Box": 5,
            "contains": 6,
            "bad1": 7,
            "bad2": 8,
            "bad3": 9,
            "bad4": 10,
        }

    def _core(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        token_ids = []
        offsets = []
        search_from = 0
        for token in text.split():
            start = text.index(token, search_from)
            end = start + len(token)
            token_ids.append(self.vocab[token])
            offsets.append((start, end))
            search_from = end
        return token_ids, offsets

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        token_ids, _ = self._core(text)
        if add_special_tokens:
            return [self.cls_token_id, *token_ids, self.sep_token_id]
        return token_ids

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        token_ids, offsets = self._core(text)
        input_ids = [self.cls_token_id, *token_ids, self.sep_token_id]
        output: dict[str, list[int] | list[tuple[int, int]]] = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }
        if return_offsets_mapping:
            output["offset_mapping"] = [(0, 0), *offsets, (0, 0)]
        return output


class PreviousMaskPrefersGoodModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        logits = torch.zeros((*input_ids.shape, 128), dtype=torch.float32, device=input_ids.device)
        for row in range(input_ids.shape[0]):
            for index in range(input_ids.shape[1] - 1):
                if int(input_ids[row, index + 1].item()) == FakeMntpTokenizer.mask_token_id:
                    logits[row, index, 3] = 5.0
        return SimpleNamespace(logits=logits)


class ContextSensitiveMntpModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        logits = torch.zeros((*input_ids.shape, 128), dtype=torch.float32, device=input_ids.device)
        context_id = 1
        for row in range(input_ids.shape[0]):
            for index in range(input_ids.shape[1] - 1):
                if (
                    int(input_ids[row, index].item()) == context_id
                    and int(input_ids[row, index + 1].item()) == FakeMntpTokenizer.mask_token_id
                ):
                    logits[row, index, 3] = 5.0
        return SimpleNamespace(logits=logits)


def test_encode_mntp_text_span_uses_special_tokens_and_target_span() -> None:
    tokenizer = FakeMntpTokenizer()

    encoded = encode_mntp_text_span(
        tokenizer,
        MntpTextSpanRequest(text="context good", target_start_char=len("context ")),
    )

    assert encoded.input_ids == (101, 1, 3, 102)
    assert encoded.target_indices == (2,)
    assert encoded.target_token_ids == (3,)


def test_score_mntp_text_spans_masks_target_and_scores_previous_position() -> None:
    scores = score_mntp_text_spans(
        model=PreviousMaskPrefersGoodModel(),
        tokenizer=FakeMntpTokenizer(),
        requests=[
            MntpTextSpanRequest(text="context good", target_start_char=len("context ")),
            MntpTextSpanRequest(text="context bad", target_start_char=len("context ")),
        ],
        batch_size=2,
    )

    assert scores[0].sum_logprob > scores[1].sum_logprob
    assert scores[0].scored_token_count == 1


def test_evaluate_blimp_mntp_likelihood_uses_strict_good_greater_than_bad() -> None:
    item = BlimpItem(
        item_id="toy:1",
        source_file="toy.jsonl",
        line_number=1,
        sentence_good="context good",
        sentence_bad="context bad",
        field="syntax",
        linguistics_term="toy",
        uid="toy_uid",
    )

    result = evaluate_blimp_mntp_likelihood(
        model=PreviousMaskPrefersGoodModel(),
        tokenizer=FakeMntpTokenizer(),
        items=[item],
        batch_size=4,
    )

    record = result.item_records[0]
    assert record["correct"] is True
    assert record["score_good"] > record["score_bad"]


def test_evaluate_entity_tracking_mntp_likelihood_supports_mean_reduction() -> None:
    item = EntityTrackingItem(
        item_id="regular:1",
        source_file="regular.jsonl",
        line_number=1,
        family="regular",
        numops=0,
        uid="regular_0_ops",
        input_prefix="Box contains ",
        options=("good", "bad1", "bad2", "bad3", "bad4"),
    )

    result = evaluate_entity_tracking_mntp_likelihood(
        model=PreviousMaskPrefersGoodModel(),
        tokenizer=FakeMntpTokenizer(),
        items=[item],
        batch_size=5,
        score_reduction="mean",
    )

    record = result.item_records[0]
    assert record["score_reduction"] == "mean"
    assert record["correct"] is True
    assert record["score_correct"] > record["best_distractor_score"]


def test_evaluate_comps_mntp_likelihood_scores_conditioned_property_phrase() -> None:
    item = CompsItem(
        item_id="base:1",
        source_file="comps_base.jsonl",
        line_number=1,
        subtask="base",
        prefix_acceptable="context",
        prefix_unacceptable="other",
        property_phrase="good",
        negative_sample_type="taxonomic",
        distraction_type="none",
    )

    result = evaluate_comps_mntp_likelihood(
        model=ContextSensitiveMntpModel(),
        tokenizer=FakeMntpTokenizer(),
        items=[item],
        batch_size=2,
    )

    record = result.item_records[0]
    assert record["correct"] is True
    assert record["score_acceptable"] > record["score_unacceptable"]
