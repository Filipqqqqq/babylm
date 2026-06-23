"""Simple BLiMP-style sentence likelihood diagnostics."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch


ScoreReduction = Literal["sum", "mean"]
VALID_SCORE_REDUCTIONS = {"sum", "mean"}


@dataclass(frozen=True)
class BlimpItem:
    """One BLiMP or BLiMP-supplement minimal-pair item."""

    item_id: str
    source_file: str
    line_number: int
    sentence_good: str
    sentence_bad: str
    field: str = ""
    linguistics_term: str = ""
    uid: str = ""
    contrast: str = ""


@dataclass(frozen=True)
class SentenceScore:
    """Sum of next-token log probabilities for one sentence."""

    sum_logprob: float
    scored_token_count: int
    input_token_count: int


@dataclass(frozen=True)
class BlimpLikelihoodResult:
    """Per-item records and aggregate rows for one evaluated model/checkpoint."""

    item_records: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]


def read_blimp_items(path: str | Path) -> list[BlimpItem]:
    """Read BLiMP-style JSONL files from a directory or single file."""

    path = Path(path)
    paths = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No BLiMP JSONL files found at: {path}")

    items: list[BlimpItem] = []
    for jsonl_path in paths:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row_id = row.get("pair_id", row.get("row", line_number))
                uid = str(row.get("UID", jsonl_path.stem))
                field = str(row.get("field", "supplement"))
                if field == "syntax_semantics":
                    field = "syntax/semantics"
                linguistics_term = str(row.get("linguistics_term", "supplement"))
                items.append(
                    BlimpItem(
                        item_id=f"{jsonl_path.stem}:{row_id}",
                        source_file=jsonl_path.name,
                        line_number=line_number,
                        sentence_good=str(row["sentence_good"]),
                        sentence_bad=str(row["sentence_bad"]),
                        field=field,
                        linguistics_term=linguistics_term,
                        uid=uid,
                        contrast=str(row.get("contrast", "")),
                    )
                )
    return items


def encode_sentence(tokenizer: Any, sentence: str) -> tuple[int, ...]:
    """Tokenize one sentence without special tokens or BOS."""

    token_ids = tuple(tokenizer.encode(sentence, add_special_tokens=False))
    if len(token_ids) < 2:
        raise ValueError(f"Sentence has fewer than two tokens after tokenization: {sentence!r}")
    return token_ids


def resolve_padding_token_id(tokenizer: Any) -> int:
    """Resolve a token id used only for batch padding."""

    for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return 0


def model_device(model: torch.nn.Module) -> torch.device:
    """Return the model parameter device, or CPU for parameter-free test doubles."""

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _batched(items: list[tuple[int, ...]], batch_size: int) -> list[list[tuple[int, ...]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def score_sentence_logprobs(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    sentences: list[str],
    batch_size: int = 64,
    device: torch.device | str | None = None,
) -> list[SentenceScore]:
    """Return ``sum log P(sentence)`` for each sentence.

    This intentionally does not prepend BOS. The first token has no left context,
    so the sum starts at the second token.
    """

    if not sentences:
        return []

    was_training = model.training
    model.eval()
    model_device_ = torch.device(device) if device is not None else model_device(model)
    padding_token_id = resolve_padding_token_id(tokenizer)
    encoded_sentences = [encode_sentence(tokenizer, sentence) for sentence in sentences]
    scores: list[SentenceScore] = []

    try:
        with torch.inference_mode():
            for batch in _batched(encoded_sentences, batch_size):
                max_length = max(len(token_ids) for token_ids in batch)
                input_ids = torch.full(
                    (len(batch), max_length),
                    padding_token_id,
                    dtype=torch.long,
                    device=model_device_,
                )
                attention_mask = torch.zeros_like(input_ids)
                for row_index, token_ids in enumerate(batch):
                    row_ids = torch.tensor(token_ids, dtype=torch.long, device=model_device_)
                    input_ids[row_index, : len(token_ids)] = row_ids
                    attention_mask[row_index, : len(token_ids)] = 1

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
                next_tokens = input_ids[:, 1:]
                token_log_probs = torch.gather(
                    log_probs,
                    dim=-1,
                    index=next_tokens.unsqueeze(-1),
                ).squeeze(-1)

                for row_index, token_ids in enumerate(batch):
                    scored_token_count = len(token_ids) - 1
                    sentence_score = float(
                        token_log_probs[row_index, :scored_token_count].sum().item()
                    )
                    scores.append(
                        SentenceScore(
                            sum_logprob=sentence_score,
                            scored_token_count=scored_token_count,
                            input_token_count=len(token_ids),
                        )
                    )
    finally:
        if was_training:
            model.train()

    return scores


def validate_score_reduction(score_reduction: str) -> ScoreReduction:
    """Validate the token-score reduction used to compare candidates."""

    if score_reduction not in VALID_SCORE_REDUCTIONS:
        raise ValueError(
            "score_reduction must be 'sum' or 'mean', "
            f"got {score_reduction!r}."
        )
    return score_reduction  # type: ignore[return-value]


def reduce_sentence_score(score: SentenceScore, score_reduction: ScoreReduction) -> float:
    """Return the candidate score used for ranking."""

    if score_reduction == "mean":
        return score.sum_logprob / score.scored_token_count
    return score.sum_logprob


def build_item_records(
    *,
    items: list[BlimpItem],
    good_scores: list[SentenceScore],
    bad_scores: list[SentenceScore],
    score_reduction: ScoreReduction = "sum",
) -> list[dict[str, Any]]:
    """Build records using strict ``score_good > score_bad`` correctness."""

    score_reduction = validate_score_reduction(score_reduction)
    if len(good_scores) != len(items) or len(bad_scores) != len(items):
        raise ValueError("items, good_scores, and bad_scores must have the same length.")

    records: list[dict[str, Any]] = []
    for item_index, (item, good, bad) in enumerate(zip(items, good_scores, bad_scores)):
        score_good = reduce_sentence_score(good, score_reduction)
        score_bad = reduce_sentence_score(bad, score_reduction)
        margin = score_good - score_bad
        correct = score_good > score_bad
        records.append(
            {
                "item_index": item_index,
                "item_id": item.item_id,
                "source_file": item.source_file,
                "line_number": item.line_number,
                "field": item.field,
                "linguistics_term": item.linguistics_term,
                "uid": item.uid,
                "contrast": item.contrast,
                "score_reduction": score_reduction,
                "sentence_good": item.sentence_good,
                "sentence_bad": item.sentence_bad,
                "score_good": score_good,
                "score_bad": score_bad,
                "margin": margin,
                "correct": correct,
                "score": float(correct),
                "good_scored_token_count": good.scored_token_count,
                "bad_scored_token_count": bad.scored_token_count,
                "good_input_token_count": good.input_token_count,
                "bad_input_token_count": bad.input_token_count,
                "token_count_delta": good.input_token_count - bad.input_token_count,
            }
        )
    return records


def _row_for_group(
    scope: str,
    name: str,
    records: list[dict[str, Any]],
    *,
    override_score: float | None = None,
) -> dict[str, Any]:
    correct = sum(int(record["correct"]) for record in records)
    margins = [float(record["margin"]) for record in records]
    micro_score = correct / len(records) if records else 0.0
    return {
        "group_scope": scope,
        "group_name": name,
        "score_reduction": records[0].get("score_reduction", "") if records else "",
        "items": len(records),
        "correct": correct,
        "score": micro_score if override_score is None else override_score,
        "mean_margin": statistics.mean(margins) if margins else 0.0,
        "ties": sum(int(float(record["margin"]) == 0.0) for record in records),
    }


def aggregate_item_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate BLiMP records with the official-style UID macro average."""

    if not records:
        return []

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for scope, field in (
            ("source_file", "source_file"),
            ("field", "field"),
            ("linguistics_term", "linguistics_term"),
            ("uid", "uid"),
            ("contrast", "contrast"),
        ):
            value = str(record.get(field, ""))
            if value:
                groups[(scope, value)].append(record)

    rows: list[dict[str, Any]] = []
    uid_rows = {
        name: _row_for_group("uid", name, group_records)
        for (scope, name), group_records in sorted(groups.items())
        if scope == "uid"
    }
    uid_macro = (
        statistics.mean(row["score"] for row in uid_rows.values())
        if uid_rows
        else _row_for_group("overall_micro", "all", records)["score"]
    )
    rows.append(_row_for_group("overall", "all", records, override_score=uid_macro))
    rows.append(_row_for_group("overall_micro", "all", records))

    for (scope, name), group_records in sorted(groups.items()):
        if scope == "uid":
            rows.append(uid_rows[name])
        else:
            rows.append(_row_for_group(scope, name, group_records))
    return rows


def evaluate_blimp_likelihood(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[BlimpItem],
    batch_size: int = 64,
    device: torch.device | str | None = None,
    score_reduction: ScoreReduction = "sum",
) -> BlimpLikelihoodResult:
    """Evaluate sentence likelihood pairs with sum or mean token logprobs."""

    score_reduction = validate_score_reduction(score_reduction)

    good_scores = score_sentence_logprobs(
        model=model,
        tokenizer=tokenizer,
        sentences=[item.sentence_good for item in items],
        batch_size=batch_size,
        device=device,
    )
    bad_scores = score_sentence_logprobs(
        model=model,
        tokenizer=tokenizer,
        sentences=[item.sentence_bad for item in items],
        batch_size=batch_size,
        device=device,
    )
    item_records = build_item_records(
        items=items,
        good_scores=good_scores,
        bad_scores=bad_scores,
        score_reduction=score_reduction,
    )
    return BlimpLikelihoodResult(
        item_records=item_records,
        summary_rows=aggregate_item_records(item_records),
    )


RUN_METADATA_FIELDNAMES = [
    "run_name",
    "epoch",
    "step",
    "tokens_seen",
    "words_seen",
    "checkpoint",
    "checkpoint_path",
]

ITEM_RECORD_FIELDNAMES = [
    *RUN_METADATA_FIELDNAMES,
    "item_index",
    "item_id",
    "source_file",
    "line_number",
    "field",
    "linguistics_term",
    "uid",
    "contrast",
    "score_reduction",
    "sentence_good",
    "sentence_bad",
    "score_good",
    "score_bad",
    "margin",
    "correct",
    "score",
    "good_scored_token_count",
    "bad_scored_token_count",
    "good_input_token_count",
    "bad_input_token_count",
    "token_count_delta",
]

SUMMARY_FIELDNAMES = [
    *RUN_METADATA_FIELDNAMES,
    "group_scope",
    "group_name",
    "score_reduction",
    "items",
    "correct",
    "score",
    "mean_margin",
    "ties",
]


def add_record_metadata(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return copies of rows tagged with run/checkpoint metadata."""

    if not metadata:
        return [dict(row) for row in rows]
    return [{**metadata, **row} for row in rows]


def append_item_records_jsonl(
    records: list[dict[str, Any]],
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append per-item records as JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in add_record_metadata(records, metadata):
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")


def write_item_records_jsonl(
    records: list[dict[str, Any]],
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write per-item records as JSONL."""

    path = Path(path)
    if path.exists():
        path.unlink()
    append_item_records_jsonl(records, path, metadata=metadata)


def append_summary_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append aggregate rows as CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in add_record_metadata(rows, metadata):
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDNAMES})


def write_summary_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write aggregate rows as CSV."""

    path = Path(path)
    if path.exists():
        path.unlink()
    append_summary_csv(rows, path, metadata=metadata)


__all__ = [
    "ITEM_RECORD_FIELDNAMES",
    "RUN_METADATA_FIELDNAMES",
    "SUMMARY_FIELDNAMES",
    "BlimpItem",
    "BlimpLikelihoodResult",
    "SentenceScore",
    "ScoreReduction",
    "add_record_metadata",
    "aggregate_item_records",
    "append_item_records_jsonl",
    "append_summary_csv",
    "build_item_records",
    "encode_sentence",
    "evaluate_blimp_likelihood",
    "read_blimp_items",
    "reduce_sentence_score",
    "score_sentence_logprobs",
    "validate_score_reduction",
    "write_item_records_jsonl",
    "write_summary_csv",
]
