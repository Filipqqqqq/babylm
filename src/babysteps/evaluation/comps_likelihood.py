"""Simple COMPS continuation-likelihood diagnostics."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SUBTASK_BY_FILE = {
    "comps_base": "base",
    "comps_wugs": "wugs",
    "comps_wugs_dist-before": "wugs_dist_before",
    "comps_wugs_dist-in-between": "wugs_dist_in_between",
}

SUBTASK_ORDER = (
    "base",
    "wugs",
    "wugs_dist_before",
    "wugs_dist_in_between",
)


@dataclass(frozen=True)
class CompsItem:
    """One COMPS acceptable/unacceptable continuation pair."""

    item_id: str
    source_file: str
    line_number: int
    subtask: str
    prefix_acceptable: str
    prefix_unacceptable: str
    property_phrase: str
    negative_sample_type: str
    distraction_type: str
    similarity: float | None = None


@dataclass(frozen=True)
class ContinuationRequest:
    """One prefix-conditioned continuation likelihood request."""

    prefix: str
    continuation: str


@dataclass(frozen=True)
class EncodedContinuation:
    """Tokenized continuation with target span in next-token coordinates."""

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
class CompsLikelihoodResult:
    """Per-item records and aggregate rows for one evaluated model/checkpoint."""

    item_records: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]


def read_comps_items(path: str | Path) -> list[CompsItem]:
    """Read COMPS JSONL files from a directory or single file."""

    path = Path(path)
    paths = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No COMPS JSONL files found at: {path}")

    items: list[CompsItem] = []
    for jsonl_path in paths:
        subtask = SUBTASK_BY_FILE.get(jsonl_path.stem, jsonl_path.stem)
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row_id = row.get("id", line_number)
                items.append(
                    CompsItem(
                        item_id=f"{subtask}:{row_id}",
                        source_file=jsonl_path.name,
                        line_number=line_number,
                        subtask=subtask,
                        prefix_acceptable=str(row["prefix_acceptable"]),
                        prefix_unacceptable=str(row["prefix_unacceptable"]),
                        property_phrase=str(row["property_phrase"]),
                        negative_sample_type=str(row.get("negative_sample_type", "")),
                        distraction_type=str(row.get("distraction_type", "none")),
                        similarity=(
                            None
                            if row.get("similarity") is None
                            else float(row["similarity"])
                        ),
                    )
                )
    return items


def make_continuation_requests(items: list[CompsItem]) -> list[ContinuationRequest]:
    """Build acceptable/unacceptable continuation requests for all items."""

    requests: list[ContinuationRequest] = []
    for item in items:
        requests.append(
            ContinuationRequest(
                prefix=item.prefix_acceptable,
                continuation=item.property_phrase,
            )
        )
        requests.append(
            ContinuationRequest(
                prefix=item.prefix_unacceptable,
                continuation=item.property_phrase,
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


def _target_span_with_offsets(
    tokenizer: Any,
    text: str,
    start_char: int,
) -> tuple[list[int], int, int]:
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
        raise ValueError("COMPS continuation has no tokens after prefix slicing.")
    return token_ids, min(target_indices), max(target_indices) + 1


def _target_span_with_prefix_length(
    tokenizer: Any,
    prefix: str,
    text: str,
) -> tuple[list[int], int, int]:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= len(prefix_ids):
        raise ValueError("COMPS continuation has no tokens after prefix slicing.")
    return token_ids, len(prefix_ids), len(token_ids)


def encode_continuation(tokenizer: Any, request: ContinuationRequest) -> EncodedContinuation:
    """Tokenize ``prefix + ' ' + continuation`` and score continuation tokens."""

    text = f"{request.prefix} {request.continuation}"
    start_char = len(request.prefix) + 1
    try:
        token_ids, target_start, target_end = _target_span_with_offsets(
            tokenizer,
            text,
            start_char,
        )
    except (NotImplementedError, TypeError, KeyError, ValueError):
        token_ids, target_start, target_end = _target_span_with_prefix_length(
            tokenizer,
            request.prefix,
            text,
        )

    if target_start == 0:
        raise ValueError("COMPS continuations require a non-empty prefix.")
    return EncodedContinuation(
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
) -> list[ContinuationScore]:
    """Score continuation-token log probabilities for all requests."""

    if not requests:
        return []

    was_training = model.training
    model.eval()
    model_device_ = torch.device(device) if device is not None else model_device(model)
    padding_token_id = resolve_padding_token_id(tokenizer)
    encoded = [encode_continuation(tokenizer, request) for request in requests]
    scores: list[ContinuationScore] = []

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
                    scores.append(
                        ContinuationScore(
                            sum_logprob=float(continuation_log_probs.sum().item()),
                            scored_token_count=int(item.score_end - item.score_start),
                            input_token_count=len(item.input_ids),
                        )
                    )
    finally:
        if was_training:
            model.train()

    return scores


def build_item_records(
    *,
    items: list[CompsItem],
    scores: list[ContinuationScore],
) -> list[dict[str, Any]]:
    """Build records using strict ``score_acceptable > score_unacceptable``."""

    if len(scores) != 2 * len(items):
        raise ValueError("COMPS scoring expects exactly two continuation scores per item.")

    records: list[dict[str, Any]] = []
    for item_index, item in enumerate(items):
        acceptable = scores[2 * item_index]
        unacceptable = scores[2 * item_index + 1]
        score_acceptable = acceptable.sum_logprob
        score_unacceptable = unacceptable.sum_logprob
        margin = score_acceptable - score_unacceptable
        correct = score_acceptable > score_unacceptable
        records.append(
            {
                "item_index": item_index,
                "item_id": item.item_id,
                "source_file": item.source_file,
                "line_number": item.line_number,
                "subtask": item.subtask,
                "negative_sample_type": item.negative_sample_type,
                "distraction_type": item.distraction_type,
                "similarity": item.similarity if item.similarity is not None else "",
                "prefix_acceptable": item.prefix_acceptable,
                "prefix_unacceptable": item.prefix_unacceptable,
                "property_phrase": item.property_phrase,
                "sentence_acceptable": f"{item.prefix_acceptable} {item.property_phrase}",
                "sentence_unacceptable": f"{item.prefix_unacceptable} {item.property_phrase}",
                "score_acceptable": score_acceptable,
                "score_unacceptable": score_unacceptable,
                "margin": margin,
                "correct": correct,
                "score": float(correct),
                "tie": margin == 0.0,
                "acceptable_scored_token_count": acceptable.scored_token_count,
                "unacceptable_scored_token_count": unacceptable.scored_token_count,
                "acceptable_input_token_count": acceptable.input_token_count,
                "unacceptable_input_token_count": unacceptable.input_token_count,
            }
        )
    return records


def _row_for_group(
    scope: str,
    name: str,
    records: list[dict[str, Any]],
    *,
    score_kind: str = "micro",
    override_score: float | None = None,
) -> dict[str, Any]:
    correct = sum(int(record["correct"]) for record in records)
    margins = [float(record["margin"]) for record in records]
    micro_score = correct / len(records) if records else 0.0
    return {
        "group_scope": scope,
        "group_name": name,
        "items": len(records),
        "correct": correct,
        "score": micro_score if override_score is None else override_score,
        "mean_margin": statistics.mean(margins) if margins else 0.0,
        "ties": sum(int(record["tie"]) for record in records),
        "score_kind": score_kind,
        "micro_score": micro_score,
    }


def aggregate_item_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate COMPS records overall, by subtask, and by negative sample type."""

    if not records:
        return []

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for scope, field in (
            ("source_file", "source_file"),
            ("subtask", "subtask"),
            ("negative_sample_type", "negative_sample_type"),
            ("distraction_type", "distraction_type"),
        ):
            value = str(record.get(field, ""))
            if value:
                groups[(scope, value)].append(record)

        subtask = str(record.get("subtask", ""))
        negative_sample_type = str(record.get("negative_sample_type", ""))
        if subtask and negative_sample_type:
            groups[
                ("subtask_negative_sample_type", f"{subtask}/{negative_sample_type}")
            ].append(record)

    rows: list[dict[str, Any]] = []
    subtask_rows = {
        name: _row_for_group("subtask", name, groups[("subtask", name)])
        for name in SUBTASK_ORDER
        if groups.get(("subtask", name))
    }
    macro_score = statistics.mean(
        row["score"] for row in subtask_rows.values()
    ) if subtask_rows else _row_for_group("overall_micro", "all", records)["score"]
    rows.append(
        _row_for_group(
            "overall",
            "all",
            records,
            score_kind="subtask_macro",
            override_score=macro_score,
        )
    )
    rows.append(_row_for_group("overall_micro", "all", records))

    for subtask in SUBTASK_ORDER:
        row = subtask_rows.get(subtask)
        if row:
            rows.append(row)

    remaining = sorted(
        (key, group_records)
        for key, group_records in groups.items()
        if key[0] != "subtask"
    )
    for (scope, name), group_records in remaining:
        rows.append(_row_for_group(scope, name, group_records))
    return rows


def evaluate_comps_likelihood(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[CompsItem],
    batch_size: int = 64,
    device: torch.device | str | None = None,
) -> CompsLikelihoodResult:
    """Evaluate ``sum log P(property_phrase | prefix_*)`` for COMPS pairs."""

    scores = score_continuations(
        model=model,
        tokenizer=tokenizer,
        requests=make_continuation_requests(items),
        batch_size=batch_size,
        device=device,
    )
    item_records = build_item_records(items=items, scores=scores)
    return CompsLikelihoodResult(
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
    "subtask",
    "negative_sample_type",
    "distraction_type",
    "similarity",
    "prefix_acceptable",
    "prefix_unacceptable",
    "property_phrase",
    "sentence_acceptable",
    "sentence_unacceptable",
    "score_acceptable",
    "score_unacceptable",
    "margin",
    "correct",
    "score",
    "tie",
    "acceptable_scored_token_count",
    "unacceptable_scored_token_count",
    "acceptable_input_token_count",
    "unacceptable_input_token_count",
]

SUMMARY_FIELDNAMES = [
    *RUN_METADATA_FIELDNAMES,
    "group_scope",
    "group_name",
    "items",
    "correct",
    "score",
    "mean_margin",
    "ties",
    "score_kind",
    "micro_score",
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
    "SUBTASK_BY_FILE",
    "SUBTASK_ORDER",
    "CompsItem",
    "CompsLikelihoodResult",
    "ContinuationRequest",
    "ContinuationScore",
    "aggregate_item_records",
    "append_item_records_jsonl",
    "append_summary_csv",
    "build_item_records",
    "encode_continuation",
    "evaluate_comps_likelihood",
    "make_continuation_requests",
    "read_comps_items",
    "score_continuations",
]
