from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from babysteps.evaluation.comps_likelihood import (
    CompsItem,
    ContinuationScore,
    SUMMARY_FIELDNAMES,
    aggregate_item_records,
    append_item_records_jsonl,
    append_summary_csv,
    build_item_records,
    evaluate_comps_likelihood,
    read_comps_items,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    bos_token_id = None

    def __init__(self) -> None:
        self.vocab = {
            "a": 1,
            "good": 2,
            "bad": 3,
            "runs.": 4,
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


class GoodPrefixModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        logits = torch.zeros((*input_ids.shape, 128), dtype=torch.float32, device=input_ids.device)
        good_context = input_ids == 2
        logits[good_context, 4] = 5.0
        return SimpleNamespace(logits=logits)


def make_item() -> CompsItem:
    return CompsItem(
        item_id="base:1",
        source_file="comps_base.jsonl",
        line_number=1,
        subtask="base",
        prefix_acceptable="a good",
        prefix_unacceptable="a bad",
        property_phrase="runs.",
        negative_sample_type="taxonomic",
        distraction_type="none",
        similarity=0.5,
    )


def test_read_comps_items_maps_filenames_to_subtasks(tmp_path: Path) -> None:
    data_dir = tmp_path / "comps"
    data_dir.mkdir()
    (data_dir / "comps_wugs_dist-before.jsonl").write_text(
        json.dumps(
            {
                "id": 7,
                "prefix_acceptable": "a good",
                "prefix_unacceptable": "a bad",
                "property_phrase": "runs.",
                "negative_sample_type": "overlap",
                "distraction_type": "before",
                "similarity": 0.25,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    items = read_comps_items(data_dir)

    assert len(items) == 1
    assert items[0].item_id == "wugs_dist_before:7"
    assert items[0].subtask == "wugs_dist_before"
    assert items[0].negative_sample_type == "overlap"


def test_build_item_records_uses_strict_acceptable_greater_than_unacceptable() -> None:
    item = make_item()
    records = build_item_records(
        items=[item, item],
        scores=[
            ContinuationScore(-1.0, 1, 3),
            ContinuationScore(-2.0, 1, 3),
            ContinuationScore(-1.0, 1, 3),
            ContinuationScore(-1.0, 1, 3),
        ],
    )

    assert records[0]["correct"] is True
    assert records[0]["margin"] == 1.0
    assert records[1]["correct"] is False
    assert records[1]["margin"] == 0.0
    assert records[1]["tie"] is True


def test_aggregate_item_records_uses_subtask_macro_for_overall_score() -> None:
    records = []
    for index in range(4):
        records.append(
            {
                "subtask": "base",
                "negative_sample_type": "taxonomic",
                "distraction_type": "none",
                "correct": index < 3,
                "margin": 1.0 if index < 3 else -1.0,
                "tie": False,
            }
        )
    records.append(
        {
            "subtask": "wugs",
            "negative_sample_type": "taxonomic",
            "distraction_type": "undistracted",
            "correct": False,
            "margin": -1.0,
            "tie": False,
        }
    )

    rows = aggregate_item_records(records)

    overall = next(row for row in rows if row["group_scope"] == "overall")
    assert overall["score_kind"] == "subtask_macro"
    assert overall["score"] == 0.375
    assert overall["micro_score"] == 0.6


def test_evaluate_comps_likelihood_scores_property_phrase_conditionally() -> None:
    result = evaluate_comps_likelihood(
        model=GoodPrefixModel(),
        tokenizer=FakeTokenizer(),
        items=[make_item()],
        batch_size=2,
    )

    record = result.item_records[0]
    assert record["correct"] is True
    assert record["score_acceptable"] > record["score_unacceptable"]

    overall = next(row for row in result.summary_rows if row["group_scope"] == "overall")
    assert overall["score"] == 1.0
    assert overall["items"] == 1


def test_append_outputs_accept_training_metadata(tmp_path: Path) -> None:
    records = build_item_records(
        items=[make_item()],
        scores=[
            ContinuationScore(-1.0, 1, 3),
            ContinuationScore(-2.0, 1, 3),
        ],
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
