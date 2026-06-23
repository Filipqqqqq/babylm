from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from babysteps.evaluation.entity_tracking_likelihood import (
    ContinuationScore,
    EntityTrackingItem,
    SUMMARY_FIELDNAMES,
    aggregate_item_records,
    append_item_records_jsonl,
    append_summary_csv,
    build_item_records,
    evaluate_entity_tracking_likelihood,
    read_entity_tracking_items,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    bos_token_id = None

    def __init__(self) -> None:
        self.vocab = {
            "Box": 1,
            "contains": 2,
            "good.": 3,
            "bad1.": 4,
            "bad2.": 5,
            "bad3.": 6,
            "bad4.": 7,
        }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [self.vocab[token] for token in text.split()]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        assert add_special_tokens is False
        token_ids = []
        offsets = []
        search_from = 0
        for token in text.split():
            start = text.index(token, search_from)
            end = start + len(token)
            token_ids.append(self.vocab[token])
            offsets.append((start, end))
            search_from = end
        output: dict[str, list[int] | list[tuple[int, int]]] = {
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
        }
        if return_offsets_mapping:
            output["offset_mapping"] = offsets
        return output


class GoodOptionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        logits = torch.zeros((*input_ids.shape, 128), dtype=torch.float32, device=input_ids.device)
        logits[:, :, 3] = 5.0
        return SimpleNamespace(logits=logits)


def make_item() -> EntityTrackingItem:
    return EntityTrackingItem(
        item_id="regular:1",
        source_file="regular.jsonl",
        line_number=1,
        family="regular",
        numops=2,
        uid="regular_2_ops",
        input_prefix="Box contains ",
        options=("good.", "bad1.", "bad2.", "bad3.", "bad4."),
    )


def test_read_entity_tracking_items_builds_uid_from_family_and_numops(tmp_path: Path) -> None:
    data_dir = tmp_path / "entity_tracking"
    data_dir.mkdir()
    (data_dir / "ambiref.jsonl").write_text(
        json.dumps(
            {
                "sample_id": 123,
                "example_id": 7,
                "numops": 3,
                "input_prefix": "Box contains ",
                "options": ["good.", "bad1.", "bad2.", "bad3.", "bad4."],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    items = read_entity_tracking_items(data_dir)

    assert len(items) == 1
    assert items[0].item_id == "ambiref:7"
    assert items[0].family == "ambiref"
    assert items[0].numops == 3
    assert items[0].uid == "ambiref_3_ops"


def test_build_item_records_uses_strict_option_zero_greater_than_all() -> None:
    item = make_item()
    records = build_item_records(
        items=[item, item],
        score_map={
            0: {
                0: ContinuationScore(-1.0, 1, 3),
                1: ContinuationScore(-2.0, 1, 3),
                2: ContinuationScore(-3.0, 1, 3),
                3: ContinuationScore(-4.0, 1, 3),
                4: ContinuationScore(-5.0, 1, 3),
            },
            1: {
                0: ContinuationScore(-1.0, 1, 3),
                1: ContinuationScore(-1.0, 1, 3),
                2: ContinuationScore(-3.0, 1, 3),
                3: ContinuationScore(-4.0, 1, 3),
                4: ContinuationScore(-5.0, 1, 3),
            },
        },
    )

    assert records[0]["correct"] is True
    assert records[0]["margin"] == 1.0
    assert records[1]["correct"] is False
    assert records[1]["margin"] == 0.0
    assert records[1]["tie_for_best"] is True


def test_build_item_records_can_use_mean_token_log_likelihood() -> None:
    records = build_item_records(
        items=[make_item()],
        score_map={
            0: {
                0: ContinuationScore(-2.0, 2, 4),
                1: ContinuationScore(-1.5, 1, 3),
                2: ContinuationScore(-3.0, 1, 3),
                3: ContinuationScore(-4.0, 1, 3),
                4: ContinuationScore(-5.0, 1, 3),
            },
        },
        score_reduction="mean",
    )

    assert records[0]["score_reduction"] == "mean"
    assert records[0]["score_correct"] == -1.0
    assert records[0]["best_distractor_score"] == -1.5
    assert records[0]["correct"] is True


def test_evaluate_entity_tracking_likelihood_scores_continuations() -> None:
    result = evaluate_entity_tracking_likelihood(
        model=GoodOptionModel(),
        tokenizer=FakeTokenizer(),
        items=[make_item()],
        batch_size=5,
    )

    record = result.item_records[0]
    assert record["correct"] is True
    assert record["score_correct"] > record["best_distractor_score"]

    overall = next(row for row in result.summary_rows if row["group_scope"] == "overall")
    assert overall["score"] == 1.0
    assert overall["items"] == 1


def test_aggregate_uses_family_uid_macro_and_preserves_micro_score() -> None:
    def make_records(family: str, numops: int, correctness: list[bool]) -> list[dict[str, object]]:
        return [
            {
                "family": family,
                "numops": numops,
                "uid": f"{family}_{numops}_ops",
                "correct": correct,
                "margin": 1.0 if correct else -1.0,
                "tie_for_best": False,
            }
            for correct in correctness
        ]

    records = [
        *make_records("regular", 0, [True]),
        *make_records("regular", 1, [False, False, False]),
        *make_records("ambiref", 0, [True, True]),
        *make_records("ambiref", 1, [False, False]),
        *make_records("move_contents", 0, [True, False, False, False]),
    ]

    rows = aggregate_item_records(records)
    overall = next(row for row in rows if row["group_scope"] == "overall")
    overall_micro = next(row for row in rows if row["group_scope"] == "overall_micro")
    regular = next(
        row
        for row in rows
        if row["group_scope"] == "family" and row["group_name"] == "regular"
    )

    assert regular["score"] == 0.5
    assert abs(overall["score"] - ((0.5 + 0.5 + 0.25) / 3)) < 1e-12
    assert overall_micro["score"] == 4 / 12


def test_append_outputs_accept_training_metadata(tmp_path: Path) -> None:
    item = make_item()
    records = build_item_records(
        items=[item],
        score_map={
            0: {
                index: ContinuationScore(-float(index + 1), 1, 3)
                for index in range(5)
            }
        },
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
    assert list(rows[0]) == SUMMARY_FIELDNAMES
