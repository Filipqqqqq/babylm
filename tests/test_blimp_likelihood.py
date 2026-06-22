from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from babysteps.evaluation.blimp_likelihood import (
    BlimpItem,
    SentenceScore,
    aggregate_item_records,
    append_item_records_jsonl,
    append_summary_csv,
    build_item_records,
    encode_sentence,
    evaluate_blimp_likelihood,
    read_blimp_items,
)


class FakeTokenizer:
    bos_token_id = 99
    eos_token_id = 98
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocab = {
            "good": 1,
            "word": 2,
            "bad": 3,
            "other": 4,
            "Who": 5,
            "left?": 6,
            "left": 7,
            "wrong?": 8,
        }
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        self.calls.append((text, add_special_tokens))
        token_ids = [self.vocab[token] for token in text.split()]
        if add_special_tokens:
            return [self.bos_token_id, *token_ids]
        return token_ids


class BigramPreferenceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        logits = torch.zeros((*input_ids.shape, 16), dtype=torch.float32, device=input_ids.device)
        for row_index in range(input_ids.shape[0]):
            for token_index in range(input_ids.shape[1] - 1):
                previous_token = int(input_ids[row_index, token_index].item())
                next_token = int(input_ids[row_index, token_index + 1].item())
                logits[row_index, token_index, next_token] = 5.0 if previous_token == 1 else -5.0
        return SimpleNamespace(logits=logits)


def make_item() -> BlimpItem:
    return BlimpItem(
        item_id="toy:1",
        source_file="toy.jsonl",
        line_number=1,
        sentence_good="good word",
        sentence_bad="bad other",
        field="syntax",
        linguistics_term="toy_term",
        uid="toy_uid",
    )


def test_read_blimp_and_supplement_items(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "adjunct_island.jsonl").write_text(
        json.dumps(
            {
                "sentence_good": "Who left?",
                "sentence_bad": "Who left wrong?",
                "field": "syntax",
                "linguistics_term": "island_effects",
                "UID": "adjunct_island",
                "pair_id": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "hypernym.jsonl").write_text(
        json.dumps(
            {
                "row": 3,
                "contrast": "base/hypernym",
                "sentence_good": "good word",
                "sentence_bad": "bad other",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    items = read_blimp_items(data_dir)

    assert [item.item_id for item in items] == ["adjunct_island:7", "hypernym:3"]
    assert items[0].field == "syntax"
    assert items[0].linguistics_term == "island_effects"
    assert items[1].field == "supplement"
    assert items[1].linguistics_term == "supplement"
    assert items[1].uid == "hypernym"
    assert items[1].contrast == "base/hypernym"


def test_encode_sentence_does_not_add_bos_or_special_tokens() -> None:
    tokenizer = FakeTokenizer()

    token_ids = encode_sentence(tokenizer, "good word")

    assert token_ids == (1, 2)
    assert tokenizer.calls == [("good word", False)]


def test_build_item_records_uses_strict_good_greater_than_bad() -> None:
    item = make_item()
    records = build_item_records(
        items=[item, item],
        good_scores=[
            SentenceScore(sum_logprob=-1.0, scored_token_count=1, input_token_count=2),
            SentenceScore(sum_logprob=-2.0, scored_token_count=1, input_token_count=2),
        ],
        bad_scores=[
            SentenceScore(sum_logprob=-2.0, scored_token_count=1, input_token_count=2),
            SentenceScore(sum_logprob=-2.0, scored_token_count=1, input_token_count=2),
        ],
    )

    assert records[0]["score_good"] == -1.0
    assert records[0]["score_bad"] == -2.0
    assert records[0]["correct"] is True
    assert records[1]["margin"] == 0.0
    assert records[1]["correct"] is False
    assert records[1]["score"] == 0.0


def test_aggregate_counts_ties_as_wrong() -> None:
    item = make_item()
    records = build_item_records(
        items=[item, item],
        good_scores=[
            SentenceScore(sum_logprob=-1.0, scored_token_count=1, input_token_count=2),
            SentenceScore(sum_logprob=-2.0, scored_token_count=1, input_token_count=2),
        ],
        bad_scores=[
            SentenceScore(sum_logprob=-3.0, scored_token_count=1, input_token_count=2),
            SentenceScore(sum_logprob=-2.0, scored_token_count=1, input_token_count=2),
        ],
    )

    overall = next(row for row in aggregate_item_records(records) if row["group_scope"] == "overall")

    assert overall["items"] == 2
    assert overall["correct"] == 1
    assert overall["score"] == 0.5
    assert overall["ties"] == 1


def test_aggregate_uses_uid_macro_and_preserves_micro_score() -> None:
    records = [
        {
            "correct": True,
            "margin": 1.0,
            "source_file": "a.jsonl",
            "field": "syntax",
            "linguistics_term": "toy",
            "uid": "small_uid",
            "contrast": "",
        },
        *[
            {
                "correct": False,
                "margin": -1.0,
                "source_file": "b.jsonl",
                "field": "syntax",
                "linguistics_term": "toy",
                "uid": "large_uid",
                "contrast": "",
            }
            for _ in range(3)
        ],
    ]

    rows = aggregate_item_records(records)
    overall = next(row for row in rows if row["group_scope"] == "overall")
    overall_micro = next(row for row in rows if row["group_scope"] == "overall_micro")

    assert overall["score"] == 0.5
    assert overall_micro["score"] == 0.25


def test_evaluate_blimp_likelihood_sums_next_token_logprobs() -> None:
    result = evaluate_blimp_likelihood(
        model=BigramPreferenceModel(),
        tokenizer=FakeTokenizer(),
        items=[make_item()],
        batch_size=1,
    )

    record = result.item_records[0]
    assert record["correct"] is True
    assert record["good_scored_token_count"] == 1
    assert record["bad_scored_token_count"] == 1
    assert record["score_good"] > record["score_bad"]


def test_append_outputs_accept_training_metadata(tmp_path: Path) -> None:
    item = make_item()
    records = build_item_records(
        items=[item],
        good_scores=[SentenceScore(sum_logprob=-1.0, scored_token_count=1, input_token_count=2)],
        bad_scores=[SentenceScore(sum_logprob=-2.0, scored_token_count=1, input_token_count=2)],
    )
    summary_rows = aggregate_item_records(records)
    metadata = {"run_name": "toy", "epoch": 1, "step": 50}

    items_path = tmp_path / "likelihood_items.jsonl"
    summary_path = tmp_path / "likelihood_summary.csv"
    append_item_records_jsonl(records, items_path, metadata=metadata)
    append_summary_csv(summary_rows, summary_path, metadata=metadata)

    item_record = json.loads(items_path.read_text(encoding="utf-8").strip())
    assert item_record["run_name"] == "toy"
    assert item_record["step"] == 50

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    overall = next(row for row in rows if row["group_scope"] == "overall")
    assert overall["run_name"] == "toy"
    assert overall["epoch"] == "1"
    assert overall["score"] == "1.0"
