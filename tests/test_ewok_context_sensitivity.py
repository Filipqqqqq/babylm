from __future__ import annotations

import csv
import json
from pathlib import Path

from babysteps.evaluation.ewok_context_sensitivity import (
    EwokItem,
    ScoringRequest,
    TargetScore,
    aggregate_item_records,
    append_item_records_jsonl,
    append_summary_csv,
    build_item_records,
    encode_request,
    read_ewok_items,
    resolve_bos_like_token_id,
)


class FakeTokenizer:
    bos_token_id = None
    eos_token_id = 99
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [abs(hash(token)) % 1000 for token in text.split()]


def make_item() -> EwokItem:
    return EwokItem(
        item_id="material-dynamics:1",
        source_file="material-dynamics.jsonl",
        line_number=1,
        domain="material-dynamics",
        concept_a="tap",
        concept_b="stir",
        context_type="direct",
        context_diff="material",
        target_diff="concept swap",
        context1="Mohammed sees the fork.",
        context2="Mohammed sees the nitrogen.",
        target1="Mohammed taps it.",
        target2="Mohammed stirs it.",
    )


def test_read_ewok_items_from_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "ewok_fast"
    data_dir.mkdir()
    (data_dir / "material-dynamics.jsonl").write_text(
        json.dumps(
            {
                "Domain": "material-dynamics",
                "ConceptA": "tap",
                "ConceptB": "stir",
                "ContextType": "direct",
                "ContextDiff": "material",
                "TargetDiff": "concept swap",
                "Context1": "Mohammed sees the fork.",
                "Context2": "Mohammed sees the nitrogen.",
                "Target1": "Mohammed taps it.",
                "Target2": "Mohammed stirs it.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    items = read_ewok_items(data_dir)

    assert len(items) == 1
    assert items[0].item_id == "material-dynamics:1"
    assert items[0].domain == "material-dynamics"
    assert items[0].target2 == "Mohammed stirs it."


def test_resolve_bos_like_token_id_falls_back_to_eos() -> None:
    assert resolve_bos_like_token_id(FakeTokenizer()) == 99


def test_encode_request_prepends_bos_and_slices_target_tokens() -> None:
    tokenizer = FakeTokenizer()
    request = ScoringRequest(
        item_index=0,
        name="s11",
        context="Mohammed sees the fork.",
        target="Mohammed taps it.",
    )

    encoded = encode_request(tokenizer, request, bos_like_token_id=99)

    assert encoded.input_ids[0] == 99
    assert encoded.target_start == len(tokenizer.encode(request.context, add_special_tokens=False))
    assert encoded.target_end == len(
        tokenizer.encode(f"{request.context} {request.target}", add_special_tokens=False)
    )
    assert encoded.target_end > encoded.target_start


def test_build_item_records_uses_strict_context_sensitivity_only() -> None:
    item = make_item()
    score_map = {
        0: {
            "s11": TargetScore(-1.0, -0.5, 2, 8),
            "s21": TargetScore(-2.0, -1.0, 2, 8),
            "s22": TargetScore(-3.0, -1.0, 3, 8),
            "s12": TargetScore(-2.2, -1.1, 2, 8),
        }
    }

    records = build_item_records(items=[item], score_map=score_map, score_reduction="sum")

    assert len(records) == 1
    assert records[0]["k1"] == 1.0
    assert records[0]["k2"] == -0.7999999999999998
    assert records[0]["correct_k1"] is True
    assert records[0]["correct_k2"] is False
    assert records[0]["score"] == 0.5
    assert records[0]["target_token_count_mismatches"] == 1
    assert records[0]["sum_mean_sign_disagreements"] == 1


def test_aggregate_uses_domain_macro_and_preserves_micro_score() -> None:
    def make_record(domain: str, correct_comparisons: int) -> dict[str, object]:
        return {
            "domain": domain,
            "context_type": "direct",
            "context_diff": "material",
            "target_diff": "concept swap",
            "score_reduction": "sum",
            "comparisons": 2,
            "correct_comparisons": correct_comparisons,
            "k_margin": 1.0 if correct_comparisons else -1.0,
            "target_token_count_mismatches": 0,
            "sum_mean_sign_disagreements": 0,
        }

    records = [
        make_record("small-domain", 2),
        make_record("large-domain", 0),
        make_record("large-domain", 0),
    ]

    rows = aggregate_item_records(records)
    overall = next(row for row in rows if row["group_scope"] == "overall")
    overall_micro = next(row for row in rows if row["group_scope"] == "overall_micro")

    assert overall["score"] == 0.5
    assert overall_micro["score"] == 2 / 6


def test_aggregate_and_append_outputs_accept_training_metadata(tmp_path: Path) -> None:
    item = make_item()
    score_map = {
        0: {
            "s11": TargetScore(-1.0, -0.5, 2, 8),
            "s21": TargetScore(-2.0, -1.0, 2, 8),
            "s22": TargetScore(-1.0, -0.5, 2, 8),
            "s12": TargetScore(-3.0, -1.5, 2, 8),
        }
    }
    records = build_item_records(items=[item], score_map=score_map, score_reduction="sum")
    summary_rows = aggregate_item_records(records)
    metadata = {"run_name": "toy", "epoch": 1, "step": 50}

    items_path = tmp_path / "ewok_context_items.jsonl"
    summary_path = tmp_path / "ewok_context_summary.csv"
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
