"""Simple Entity Tracking continuation-likelihood diagnostics."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch


ENTITY_TRACKING_FAMILIES = ("regular", "ambiref", "move_contents")
ScoreReduction = Literal["sum", "mean"]
VALID_SCORE_REDUCTIONS = {"sum", "mean"}


@dataclass(frozen=True)
class EntityTrackingItem:
    """One Entity Tracking item with five candidate continuations."""

    item_id: str
    source_file: str
    line_number: int
    family: str
    numops: int
    uid: str
    input_prefix: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class ContinuationRequest:
    """One internal option-likelihood request."""

    item_index: int
    option_index: int
    input_prefix: str
    option: str


@dataclass(frozen=True)
class EncodedContinuation:
    """Tokenized continuation with target span in next-token coordinates."""

    request: ContinuationRequest
    input_ids: tuple[int, ...]
    score_start: int
    score_end: int


@dataclass(frozen=True)
class ContinuationScore:
    """Sum of next-token log probabilities for one continuation."""

    sum_logprob: float
    scored_token_count: int
    input_token_count: int


@dataclass(frozen=True)
class EntityTrackingResult:
    """Per-item records and aggregate rows for one evaluated model/checkpoint."""

    item_records: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]


def read_entity_tracking_items(path: str | Path) -> list[EntityTrackingItem]:
    """Read Entity Tracking JSONL files from a directory or single file."""

    path = Path(path)
    paths = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No Entity Tracking JSONL files found at: {path}")

    items: list[EntityTrackingItem] = []
    for jsonl_path in paths:
        family = jsonl_path.stem
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                numops = int(row["numops"])
                uid = f"{family}_{numops}_ops"
                options = tuple(str(option) for option in row["options"])
                if len(options) < 2:
                    raise ValueError(f"Entity Tracking item has fewer than two options: {uid}")
                items.append(
                    EntityTrackingItem(
                        item_id=f"{family}:{row.get('example_id', line_number)}",
                        source_file=jsonl_path.name,
                        line_number=line_number,
                        family=family,
                        numops=numops,
                        uid=uid,
                        input_prefix=str(row["input_prefix"]),
                        options=options,
                    )
                )
    return items


def make_continuation_requests(items: list[EntityTrackingItem]) -> list[ContinuationRequest]:
    """Build one likelihood request per candidate option."""

    requests: list[ContinuationRequest] = []
    for item_index, item in enumerate(items):
        for option_index, option in enumerate(item.options):
            requests.append(
                ContinuationRequest(
                    item_index=item_index,
                    option_index=option_index,
                    input_prefix=item.input_prefix,
                    option=option,
                )
            )
    return requests


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


def _target_span_with_offsets(tokenizer: Any, text: str, start_char: int) -> tuple[list[int], int, int]:
    output = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = list(output["input_ids"])
    offsets = output["offset_mapping"]
    target_indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start_char and end > start
    ]
    if not target_indices:
        raise ValueError("Continuation has no tokens after prefix slicing.")
    return token_ids, min(target_indices), max(target_indices) + 1


def _target_span_with_prefix_length(
    tokenizer: Any,
    prefix: str,
    text: str,
) -> tuple[list[int], int, int]:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= len(prefix_ids):
        raise ValueError("Continuation has no tokens after prefix slicing.")
    return token_ids, len(prefix_ids), len(token_ids)


def encode_continuation(tokenizer: Any, request: ContinuationRequest) -> EncodedContinuation:
    """Tokenize ``input_prefix + option`` and score only option tokens."""

    text = f"{request.input_prefix}{request.option}"
    try:
        token_ids, target_start, target_end = _target_span_with_offsets(
            tokenizer,
            text,
            len(request.input_prefix),
        )
    except (NotImplementedError, TypeError, KeyError, ValueError):
        token_ids, target_start, target_end = _target_span_with_prefix_length(
            tokenizer,
            request.input_prefix,
            text,
        )

    if target_start == 0:
        raise ValueError("Entity Tracking continuations require a non-empty prefix.")
    return EncodedContinuation(
        request=request,
        input_ids=tuple(token_ids),
        score_start=target_start - 1,
        score_end=target_end - 1,
    )


def _batched(items: list[EncodedContinuation], batch_size: int) -> list[list[EncodedContinuation]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def score_continuations(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    requests: list[ContinuationRequest],
    batch_size: int = 64,
    device: torch.device | str | None = None,
) -> dict[int, dict[int, ContinuationScore]]:
    """Score option-token log probabilities for all continuation requests."""

    if not requests:
        return {}

    was_training = model.training
    model.eval()
    model_device_ = torch.device(device) if device is not None else model_device(model)
    padding_token_id = resolve_padding_token_id(tokenizer)
    encoded = [encode_continuation(tokenizer, request) for request in requests]
    scores: dict[int, dict[int, ContinuationScore]] = defaultdict(dict)

    try:
        with torch.inference_mode():
            for batch in _batched(encoded, batch_size):
                max_length = max(len(item.input_ids) for item in batch)
                input_ids = torch.full(
                    (len(batch), max_length),
                    padding_token_id,
                    dtype=torch.long,
                    device=model_device_,
                )
                attention_mask = torch.zeros_like(input_ids)
                for row_index, item in enumerate(batch):
                    row_ids = torch.tensor(item.input_ids, dtype=torch.long, device=model_device_)
                    input_ids[row_index, : len(item.input_ids)] = row_ids
                    attention_mask[row_index, : len(item.input_ids)] = 1

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
                next_tokens = input_ids[:, 1:]
                token_log_probs = torch.gather(
                    log_probs,
                    dim=-1,
                    index=next_tokens.unsqueeze(-1),
                ).squeeze(-1)

                for row_index, item in enumerate(batch):
                    continuation_log_probs = token_log_probs[
                        row_index,
                        item.score_start : item.score_end,
                    ]
                    total = float(continuation_log_probs.sum().item())
                    count = int(item.score_end - item.score_start)
                    scores[item.request.item_index][item.request.option_index] = (
                        ContinuationScore(
                            sum_logprob=total,
                            scored_token_count=count,
                            input_token_count=len(item.input_ids),
                        )
                    )
    finally:
        if was_training:
            model.train()

    return {item_index: dict(item_scores) for item_index, item_scores in scores.items()}


def _require_scores(
    score_map: dict[int, dict[int, ContinuationScore]],
    item_index: int,
    option_count: int,
) -> dict[int, ContinuationScore]:
    item_scores = score_map.get(item_index, {})
    missing = [index for index in range(option_count) if index not in item_scores]
    if missing:
        raise ValueError(f"Missing option scores for item {item_index}: {missing}")
    return item_scores


def validate_score_reduction(score_reduction: str) -> ScoreReduction:
    """Validate the token-score reduction used to compare candidates."""

    if score_reduction not in VALID_SCORE_REDUCTIONS:
        raise ValueError(
            "score_reduction must be 'sum' or 'mean', "
            f"got {score_reduction!r}."
        )
    return score_reduction  # type: ignore[return-value]


def reduce_continuation_score(
    score: ContinuationScore,
    score_reduction: ScoreReduction,
) -> float:
    """Return the candidate score used for ranking."""

    if score_reduction == "mean":
        return score.sum_logprob / score.scored_token_count
    return score.sum_logprob


def build_item_records(
    *,
    items: list[EntityTrackingItem],
    score_map: dict[int, dict[int, ContinuationScore]],
    score_reduction: ScoreReduction = "sum",
) -> list[dict[str, Any]]:
    """Build records using strict option-0-greater-than-all correctness."""

    score_reduction = validate_score_reduction(score_reduction)
    records: list[dict[str, Any]] = []
    for item_index, item in enumerate(items):
        item_scores = _require_scores(score_map, item_index, len(item.options))
        option_scores = [
            reduce_continuation_score(item_scores[index], score_reduction)
            for index in range(len(item.options))
        ]
        correct_score = option_scores[0]
        best_distractor_score = max(option_scores[1:])
        margin = correct_score - best_distractor_score
        best_score = max(option_scores)
        best_indices = [
            index
            for index, score in enumerate(option_scores)
            if score == best_score
        ]
        correct = correct_score > best_distractor_score

        records.append(
            {
                "item_index": item_index,
                "item_id": item.item_id,
                "source_file": item.source_file,
                "line_number": item.line_number,
                "family": item.family,
                "numops": item.numops,
                "uid": item.uid,
                "score_reduction": score_reduction,
                "input_prefix": item.input_prefix,
                "options": list(item.options),
                "score_correct": correct_score,
                "best_distractor_score": best_distractor_score,
                "margin": margin,
                "option_scores": option_scores,
                "predicted_index": best_indices[0] if len(best_indices) == 1 else "",
                "best_indices": best_indices,
                "correct": correct,
                "score": float(correct),
                "tie_for_best": len(best_indices) > 1,
                "scored_token_counts": [
                    item_scores[index].scored_token_count
                    for index in range(len(item.options))
                ],
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
        "ties": sum(int(record["tie_for_best"]) for record in records),
    }


def _ordered_families(groups: dict[tuple[str, str], list[dict[str, Any]]]) -> list[str]:
    present = {name for scope, name in groups if scope == "family"}
    ordered = [family for family in ENTITY_TRACKING_FAMILIES if family in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def aggregate_item_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate Entity Tracking records with the official-style macro average.

    The official BabyLM harness first computes each ``family_numops`` UID score,
    averages UIDs inside each family, then averages the three family scores.
    ``overall_micro`` and ``family_micro`` rows preserve the item-weighted values.
    """

    if not records:
        return []

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[("family", str(record["family"]))].append(record)
        groups[("numops", str(record["numops"]))].append(record)
        groups[("uid", str(record["uid"]))].append(record)

    rows: list[dict[str, Any]] = []
    uid_rows = {
        name: _row_for_group("uid", name, group_records)
        for (scope, name), group_records in sorted(groups.items())
        if scope == "uid"
    }

    family_rows: list[dict[str, Any]] = []
    for family in _ordered_families(groups):
        family_uid_scores = [
            row["score"]
            for uid, row in uid_rows.items()
            if uid.startswith(f"{family}_")
        ]
        if not family_uid_scores:
            continue
        family_score = statistics.mean(family_uid_scores)
        family_rows.append(
            _row_for_group(
                "family",
                family,
                groups[("family", family)],
                override_score=family_score,
            )
        )

    overall_score = (
        statistics.mean(row["score"] for row in family_rows)
        if family_rows
        else _row_for_group("overall_micro", "all", records)["score"]
    )
    rows.append(_row_for_group("overall", "all", records, override_score=overall_score))
    rows.append(_row_for_group("overall_micro", "all", records))
    rows.extend(family_rows)

    for family in _ordered_families(groups):
        rows.append(_row_for_group("family_micro", family, groups[("family", family)]))

    for (scope, name), group_records in sorted(groups.items()):
        if scope in {"family", "uid"}:
            continue
        rows.append(_row_for_group(scope, name, group_records))

    rows.extend(uid_rows[name] for name in sorted(uid_rows))
    return rows


def evaluate_entity_tracking_likelihood(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[EntityTrackingItem],
    batch_size: int = 64,
    device: torch.device | str | None = None,
    score_reduction: ScoreReduction = "sum",
) -> EntityTrackingResult:
    """Evaluate option likelihoods with sum or mean token logprobs."""

    score_reduction = validate_score_reduction(score_reduction)

    score_map = score_continuations(
        model=model,
        tokenizer=tokenizer,
        requests=make_continuation_requests(items),
        batch_size=batch_size,
        device=device,
    )
    item_records = build_item_records(
        items=items,
        score_map=score_map,
        score_reduction=score_reduction,
    )
    return EntityTrackingResult(
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
    "family",
    "numops",
    "uid",
    "score_reduction",
    "input_prefix",
    "options",
    "score_correct",
    "best_distractor_score",
    "margin",
    "option_scores",
    "predicted_index",
    "best_indices",
    "correct",
    "score",
    "tie_for_best",
    "scored_token_counts",
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


__all__ = [
    "ITEM_RECORD_FIELDNAMES",
    "RUN_METADATA_FIELDNAMES",
    "SUMMARY_FIELDNAMES",
    "ContinuationRequest",
    "ContinuationScore",
    "EntityTrackingItem",
    "EntityTrackingResult",
    "ScoreReduction",
    "aggregate_item_records",
    "append_item_records_jsonl",
    "append_summary_csv",
    "build_item_records",
    "encode_continuation",
    "evaluate_entity_tracking_likelihood",
    "make_continuation_requests",
    "read_entity_tracking_items",
    "reduce_continuation_score",
    "score_continuations",
    "validate_score_reduction",
]
