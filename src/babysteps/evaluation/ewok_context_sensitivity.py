"""Moonshot-style EWoK LM context-sensitivity diagnostics."""

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
ScoreName = Literal["s11", "s21", "s22", "s12"]


@dataclass(frozen=True)
class EwokItem:
    """One EWoK minimal-pair item."""

    item_id: str
    source_file: str
    line_number: int
    domain: str
    concept_a: str
    concept_b: str
    context_type: str
    context_diff: str
    target_diff: str
    context1: str
    context2: str
    target1: str
    target2: str


@dataclass(frozen=True)
class ScoringRequest:
    """One internal conditional target-likelihood request."""

    item_index: int
    name: ScoreName
    context: str
    target: str


@dataclass(frozen=True)
class EncodedRequest:
    """Tokenized request with target span in next-token log-prob coordinates."""

    request: ScoringRequest
    input_ids: tuple[int, ...]
    target_start: int
    target_end: int


@dataclass(frozen=True)
class TargetScore:
    """Reduced target-token log probabilities for one request."""

    sum_logprob: float
    mean_logprob: float
    target_token_count: int
    input_token_count: int


@dataclass(frozen=True)
class EwokContextResult:
    """Item records and aggregate rows for one evaluated model/checkpoint."""

    item_records: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]


def read_ewok_items(path: str | Path) -> list[EwokItem]:
    """Read EWoK fast-eval JSONL files from a directory or single file."""

    path = Path(path)
    if path.is_file():
        paths = [path]
    else:
        paths = sorted(path.glob("*.jsonl"))

    if not paths:
        raise FileNotFoundError(f"No EWoK JSONL files found at: {path}")

    items: list[EwokItem] = []
    for jsonl_path in paths:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                items.append(
                    EwokItem(
                        item_id=f"{jsonl_path.stem}:{line_number}",
                        source_file=jsonl_path.name,
                        line_number=line_number,
                        domain=str(row["Domain"]),
                        concept_a=str(row.get("ConceptA", "")),
                        concept_b=str(row.get("ConceptB", "")),
                        context_type=str(row.get("ContextType", "")),
                        context_diff=str(row.get("ContextDiff", "")),
                        target_diff=str(row.get("TargetDiff", "")),
                        context1=str(row["Context1"]),
                        context2=str(row["Context2"]),
                        target1=str(row["Target1"]),
                        target2=str(row["Target2"]),
                    )
                )

    return items


def resolve_bos_like_token_id(tokenizer: Any) -> int:
    """Resolve the BOS-like token used to start causal scoring."""

    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Tokenizer has no bos_token_id, eos_token_id, or pad_token_id.")


def resolve_padding_token_id(tokenizer: Any, *, fallback_token_id: int) -> int:
    """Resolve a token id used only for batch padding."""

    for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return fallback_token_id


def make_scoring_requests(items: list[EwokItem]) -> list[ScoringRequest]:
    """Build the four internal context-sensitivity likelihood requests per item."""

    requests: list[ScoringRequest] = []
    for item_index, item in enumerate(items):
        requests.extend(
            [
                ScoringRequest(item_index, "s11", item.context1, item.target1),
                ScoringRequest(item_index, "s21", item.context2, item.target1),
                ScoringRequest(item_index, "s22", item.context2, item.target2),
                ScoringRequest(item_index, "s12", item.context1, item.target2),
            ]
        )
    return requests


def encode_request(
    tokenizer: Any,
    request: ScoringRequest,
    *,
    bos_like_token_id: int,
) -> EncodedRequest:
    """Encode ``context + " " + target`` and slice off context tokens."""

    context_ids = tokenizer.encode(request.context, add_special_tokens=False)
    full_text = f"{request.context} {request.target}"
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    target_start = len(context_ids)
    target_end = len(full_ids)
    if target_end <= target_start:
        raise ValueError(
            "Target span has no tokens after context slicing for "
            f"item_index={request.item_index}, score={request.name!r}."
        )

    return EncodedRequest(
        request=request,
        input_ids=tuple([bos_like_token_id, *full_ids]),
        target_start=target_start,
        target_end=target_end,
    )


def _batched(items: list[EncodedRequest], batch_size: int) -> list[list[EncodedRequest]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def score_target_likelihoods(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    requests: list[ScoringRequest],
    batch_size: int = 64,
    device: torch.device | str | None = None,
) -> dict[int, dict[ScoreName, TargetScore]]:
    """Score target-token log probabilities for all internal requests."""

    if not requests:
        return {}

    was_training = model.training
    model.eval()
    model_device = torch.device(device) if device is not None else next(model.parameters()).device
    bos_like_token_id = resolve_bos_like_token_id(tokenizer)
    padding_token_id = resolve_padding_token_id(tokenizer, fallback_token_id=bos_like_token_id)
    encoded = [
        encode_request(tokenizer, request, bos_like_token_id=bos_like_token_id)
        for request in requests
    ]
    scores: dict[int, dict[ScoreName, TargetScore]] = defaultdict(dict)

    try:
        with torch.inference_mode():
            for batch in _batched(encoded, batch_size):
                max_length = max(len(item.input_ids) for item in batch)
                input_ids = torch.full(
                    (len(batch), max_length),
                    padding_token_id,
                    dtype=torch.long,
                    device=model_device,
                )
                attention_mask = torch.zeros_like(input_ids)
                for row_index, item in enumerate(batch):
                    row_ids = torch.tensor(item.input_ids, dtype=torch.long, device=model_device)
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
                    target_log_probs = token_log_probs[
                        row_index,
                        item.target_start : item.target_end,
                    ]
                    target_sum = float(target_log_probs.sum().item())
                    target_count = int(item.target_end - item.target_start)
                    scores[item.request.item_index][item.request.name] = TargetScore(
                        sum_logprob=target_sum,
                        mean_logprob=target_sum / target_count,
                        target_token_count=target_count,
                        input_token_count=len(item.input_ids),
                    )
    finally:
        if was_training:
            model.train()

    return {item_index: dict(item_scores) for item_index, item_scores in scores.items()}


def _select_score(score: TargetScore, reduction: ScoreReduction) -> float:
    if reduction == "sum":
        return score.sum_logprob
    if reduction == "mean":
        return score.mean_logprob
    raise ValueError(f"Unsupported score reduction: {reduction!r}")


def _is_strictly_correct(value: float) -> bool:
    return value > 0.0


def _require_scores(
    score_map: dict[int, dict[ScoreName, TargetScore]],
    item_index: int,
) -> dict[ScoreName, TargetScore]:
    required: tuple[ScoreName, ...] = ("s11", "s21", "s22", "s12")
    item_scores = score_map.get(item_index, {})
    missing = [name for name in required if name not in item_scores]
    if missing:
        raise ValueError(f"Missing scores for item {item_index}: {', '.join(missing)}")
    return item_scores


def build_item_records(
    *,
    items: list[EwokItem],
    score_map: dict[int, dict[ScoreName, TargetScore]],
    score_reduction: ScoreReduction = "sum",
) -> list[dict[str, Any]]:
    """Build moonshot-style LM context-sensitivity records."""

    records: list[dict[str, Any]] = []
    for item_index, item in enumerate(items):
        item_scores = _require_scores(score_map, item_index)
        s11 = item_scores["s11"]
        s21 = item_scores["s21"]
        s22 = item_scores["s22"]
        s12 = item_scores["s12"]

        k1 = _select_score(s11, score_reduction) - _select_score(s21, score_reduction)
        k2 = _select_score(s22, score_reduction) - _select_score(s12, score_reduction)
        correct_k1 = _is_strictly_correct(k1)
        correct_k2 = _is_strictly_correct(k2)

        k1_sum = s11.sum_logprob - s21.sum_logprob
        k2_sum = s22.sum_logprob - s12.sum_logprob
        k1_mean = s11.mean_logprob - s21.mean_logprob
        k2_mean = s22.mean_logprob - s12.mean_logprob
        sum_mean_sign_disagreements = int(_is_strictly_correct(k1_sum) != _is_strictly_correct(k1_mean))
        sum_mean_sign_disagreements += int(_is_strictly_correct(k2_sum) != _is_strictly_correct(k2_mean))

        target_token_count_mismatches = int(s11.target_token_count != s21.target_token_count)
        target_token_count_mismatches += int(s22.target_token_count != s12.target_token_count)

        correct_comparisons = int(correct_k1) + int(correct_k2)
        records.append(
            {
                "item_index": item_index,
                "item_id": item.item_id,
                "source_file": item.source_file,
                "line_number": item.line_number,
                "domain": item.domain,
                "concept_a": item.concept_a,
                "concept_b": item.concept_b,
                "context_type": item.context_type,
                "context_diff": item.context_diff,
                "target_diff": item.target_diff,
                "score_reduction": score_reduction,
                "k1": k1,
                "k2": k2,
                "k_margin": 0.5 * (k1 + k2),
                "correct_k1": correct_k1,
                "correct_k2": correct_k2,
                "correct_comparisons": correct_comparisons,
                "comparisons": 2,
                "score": 0.5 * correct_comparisons,
                "t1_token_count_context1": s11.target_token_count,
                "t1_token_count_context2": s21.target_token_count,
                "t2_token_count_context2": s22.target_token_count,
                "t2_token_count_context1": s12.target_token_count,
                "target_token_count_mismatches": target_token_count_mismatches,
                "sum_mean_sign_disagreements": sum_mean_sign_disagreements,
            }
        )

    return records


def _row_for_group(
    scope: str,
    name: str,
    group_records: list[dict[str, Any]],
    *,
    override_score: float | None = None,
) -> dict[str, Any]:
    comparisons = sum(int(record["comparisons"]) for record in group_records)
    correct_comparisons = sum(int(record["correct_comparisons"]) for record in group_records)
    micro_score = correct_comparisons / comparisons if comparisons else 0.0
    return {
        "group_scope": scope,
        "group_name": name,
        "score_reduction": group_records[0]["score_reduction"] if group_records else "",
        "items": len(group_records),
        "comparisons": comparisons,
        "correct_comparisons": correct_comparisons,
        "score": micro_score if override_score is None else override_score,
        "mean_k_margin": statistics.mean(
            float(record["k_margin"]) for record in group_records
        ) if group_records else 0.0,
        "target_token_count_mismatches": sum(
            int(record["target_token_count_mismatches"]) for record in group_records
        ),
        "sum_mean_sign_disagreements": sum(
            int(record["sum_mean_sign_disagreements"]) for record in group_records
        ),
    }


def aggregate_item_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate item records with a domain-macro overall score."""

    if not records:
        return []

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for scope, field in (
            ("domain", "domain"),
            ("context_type", "context_type"),
            ("context_diff", "context_diff"),
            ("target_diff", "target_diff"),
        ):
            groups[(scope, str(record.get(field, "")))].append(record)

    rows: list[dict[str, Any]] = []
    domain_rows = {
        name: _row_for_group("domain", name, group_records)
        for (scope, name), group_records in sorted(groups.items())
        if scope == "domain"
    }
    domain_macro = (
        statistics.mean(row["score"] for row in domain_rows.values())
        if domain_rows
        else _row_for_group("overall_micro", "all", records)["score"]
    )
    rows.append(_row_for_group("overall", "all", records, override_score=domain_macro))
    rows.append(_row_for_group("overall_micro", "all", records))

    for (scope, name), group_records in sorted(groups.items()):
        if scope == "domain":
            rows.append(domain_rows[name])
        else:
            rows.append(_row_for_group(scope, name, group_records))
    return rows


def evaluate_ewok_context_sensitivity(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[EwokItem],
    batch_size: int = 64,
    score_reduction: ScoreReduction = "sum",
    device: torch.device | str | None = None,
) -> EwokContextResult:
    """Evaluate moonshot-style LM context sensitivity over EWoK items."""

    score_map = score_target_likelihoods(
        model=model,
        tokenizer=tokenizer,
        requests=make_scoring_requests(items),
        batch_size=batch_size,
        device=device,
    )
    item_records = build_item_records(
        items=items,
        score_map=score_map,
        score_reduction=score_reduction,
    )
    return EwokContextResult(
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
    "domain",
    "concept_a",
    "concept_b",
    "context_type",
    "context_diff",
    "target_diff",
    "score_reduction",
    "k1",
    "k2",
    "k_margin",
    "correct_k1",
    "correct_k2",
    "correct_comparisons",
    "comparisons",
    "score",
    "t1_token_count_context1",
    "t1_token_count_context2",
    "t2_token_count_context2",
    "t2_token_count_context1",
    "target_token_count_mismatches",
    "sum_mean_sign_disagreements",
]

SUMMARY_FIELDNAMES = [
    *RUN_METADATA_FIELDNAMES,
    "group_scope",
    "group_name",
    "score_reduction",
    "items",
    "comparisons",
    "correct_comparisons",
    "score",
    "mean_k_margin",
    "target_token_count_mismatches",
    "sum_mean_sign_disagreements",
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
    "EwokContextResult",
    "EwokItem",
    "ScoringRequest",
    "TargetScore",
    "add_record_metadata",
    "aggregate_item_records",
    "append_item_records_jsonl",
    "append_summary_csv",
    "build_item_records",
    "encode_request",
    "evaluate_ewok_context_sensitivity",
    "make_scoring_requests",
    "read_ewok_items",
    "resolve_bos_like_token_id",
    "score_target_likelihoods",
    "write_item_records_jsonl",
    "write_summary_csv",
]
