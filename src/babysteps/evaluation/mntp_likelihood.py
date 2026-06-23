"""Masked next-token prediction likelihood diagnostics."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from babysteps.evaluation import blimp_likelihood
from babysteps.evaluation import comps_likelihood
from babysteps.evaluation import entity_tracking_likelihood


ScoreReduction = Literal["sum", "mean"]
VALID_SCORE_REDUCTIONS = {"sum", "mean"}


@dataclass(frozen=True)
class MntpTextSpanRequest:
    """One text span whose tokens should be scored with MNTP masking."""

    text: str
    target_start_char: int


@dataclass(frozen=True)
class EncodedMntpTextSpan:
    """Tokenized text with target-token positions for MNTP scoring."""

    input_ids: tuple[int, ...]
    target_indices: tuple[int, ...]
    target_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class MntpScore:
    """Sum of MNTP token log probabilities for one candidate."""

    sum_logprob: float
    scored_token_count: int
    input_token_count: int


@dataclass(frozen=True)
class MntpBenchmarkResult:
    """Per-item records and aggregate rows for one evaluated MNTP benchmark."""

    item_records: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]


def validate_score_reduction(score_reduction: str) -> ScoreReduction:
    """Validate the token-score reduction used to compare candidates."""

    if score_reduction not in VALID_SCORE_REDUCTIONS:
        raise ValueError(
            "score_reduction must be 'sum' or 'mean', "
            f"got {score_reduction!r}."
        )
    return score_reduction  # type: ignore[return-value]


def resolve_mask_token_id(tokenizer: Any) -> int:
    """Return the mask-token id required by the MNTP backend."""

    value = getattr(tokenizer, "mask_token_id", None)
    if value is None:
        raise ValueError("MNTP likelihood evaluation requires tokenizer.mask_token_id.")
    return int(value)


def resolve_padding_token_id(tokenizer: Any) -> int:
    """Resolve a token id used only for batch padding."""

    for attr in ("pad_token_id", "cls_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return 0


def has_prepended_special_token(tokenizer: Any) -> bool:
    """Match the official MNTP convention for whether token 0 is scoreable."""

    return (
        getattr(tokenizer, "cls_token_id", None) is not None
        or getattr(tokenizer, "bos_token_id", None) is not None
    )


def model_device(model: torch.nn.Module) -> torch.device:
    """Return the model parameter device, or CPU for parameter-free test doubles."""

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _flatten_single_tokenizer_output(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    output = tokenizer(text, return_offsets_mapping=True)
    token_ids = list(_flatten_single_tokenizer_output(output["input_ids"]))
    offsets = list(_flatten_single_tokenizer_output(output["offset_mapping"]))
    return token_ids, offsets


def _find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence:
        return 0
    for start in range(0, len(sequence) - len(subsequence) + 1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    raise ValueError("Could not locate non-special tokens inside special-token sequence.")


def _target_indices_without_offsets(
    tokenizer: Any,
    text: str,
    target_start_char: int,
) -> tuple[list[int], list[int]]:
    core_ids = list(tokenizer.encode(text, add_special_tokens=False))
    prefix_ids = list(tokenizer.encode(text[:target_start_char], add_special_tokens=False))
    token_ids = list(tokenizer.encode(text, add_special_tokens=True))
    first_core_index = _find_subsequence(token_ids, core_ids)
    target_indices = [
        first_core_index + index
        for index in range(len(prefix_ids), len(core_ids))
    ]
    return token_ids, target_indices


def encode_mntp_text_span(
    tokenizer: Any,
    request: MntpTextSpanRequest,
) -> EncodedMntpTextSpan:
    """Encode one candidate using the official MNTP target-position convention."""

    if request.target_start_char < 0 or request.target_start_char > len(request.text):
        raise ValueError(
            "target_start_char must be within the text, "
            f"got {request.target_start_char} for {request.text!r}."
        )

    try:
        token_ids, offsets = _tokenize_with_offsets(tokenizer, request.text)
        target_indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > request.target_start_char and end > start
        ]
    except (NotImplementedError, TypeError, KeyError, ValueError):
        token_ids, target_indices = _target_indices_without_offsets(
            tokenizer,
            request.text,
            request.target_start_char,
        )

    if not has_prepended_special_token(tokenizer):
        target_indices = [index for index in target_indices if index != 0]
    if not target_indices:
        raise ValueError(f"MNTP request has no scoreable target tokens: {request.text!r}")
    if min(target_indices) <= 0:
        raise ValueError("MNTP target tokens require a previous position for scoring.")

    return EncodedMntpTextSpan(
        input_ids=tuple(token_ids),
        target_indices=tuple(target_indices),
        target_token_ids=tuple(token_ids[index] for index in target_indices),
    )


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _logits_from_outputs(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs[0]


def score_mntp_text_spans(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    requests: list[MntpTextSpanRequest],
    batch_size: int = 64,
    device: torch.device | str | None = None,
) -> list[MntpScore]:
    """Score text spans by masking each target token and predicting it at ``i - 1``."""

    if not requests:
        return []

    was_training = model.training
    model.eval()
    model_device_ = torch.device(device) if device is not None else model_device(model)
    padding_token_id = resolve_padding_token_id(tokenizer)
    mask_token_id = resolve_mask_token_id(tokenizer)
    encoded = [encode_mntp_text_span(tokenizer, request) for request in requests]

    flat_examples: list[tuple[int, tuple[int, ...], int, int]] = []
    for request_index, item in enumerate(encoded):
        for target_index, target_token_id in zip(
            item.target_indices,
            item.target_token_ids,
            strict=True,
        ):
            masked_ids = list(item.input_ids)
            masked_ids[target_index] = mask_token_id
            flat_examples.append(
                (request_index, tuple(masked_ids), target_index - 1, target_token_id)
            )

    sums = [0.0 for _ in encoded]
    counts = [0 for _ in encoded]

    try:
        with torch.inference_mode():
            for batch in _batched(flat_examples, batch_size):
                max_length = max(len(input_ids) for _, input_ids, _, _ in batch)
                input_ids_tensor = torch.full(
                    (len(batch), max_length),
                    padding_token_id,
                    dtype=torch.long,
                    device=model_device_,
                )
                attention_mask = torch.zeros_like(input_ids_tensor)
                logit_indices = torch.empty(len(batch), dtype=torch.long, device=model_device_)
                target_ids = torch.empty(len(batch), dtype=torch.long, device=model_device_)

                for row_index, (_, input_ids, logit_index, target_id) in enumerate(batch):
                    row_ids = torch.tensor(input_ids, dtype=torch.long, device=model_device_)
                    input_ids_tensor[row_index, : len(input_ids)] = row_ids
                    attention_mask[row_index, : len(input_ids)] = 1
                    logit_indices[row_index] = logit_index
                    target_ids[row_index] = target_id

                outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask)
                logits = _logits_from_outputs(outputs).float()
                row_indices = torch.arange(len(batch), device=model_device_)
                selected_logits = logits[row_indices, logit_indices]
                log_probs = torch.log_softmax(selected_logits, dim=-1)
                token_log_probs = torch.gather(
                    log_probs,
                    dim=-1,
                    index=target_ids.unsqueeze(-1),
                ).squeeze(-1)

                for row_index, (request_index, _, _, _) in enumerate(batch):
                    sums[request_index] += float(token_log_probs[row_index].item())
                    counts[request_index] += 1
    finally:
        if was_training:
            model.train()

    return [
        MntpScore(
            sum_logprob=sums[index],
            scored_token_count=counts[index],
            input_token_count=len(item.input_ids),
        )
        for index, item in enumerate(encoded)
    ]


def score_mntp_sentences(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    sentences: list[str],
    batch_size: int = 64,
    device: torch.device | str | None = None,
) -> list[MntpScore]:
    """Score whole sentences with the MNTP backend."""

    return score_mntp_text_spans(
        model=model,
        tokenizer=tokenizer,
        requests=[
            MntpTextSpanRequest(text=sentence, target_start_char=0)
            for sentence in sentences
        ],
        batch_size=batch_size,
        device=device,
    )


def score_mntp_comps_continuations(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    requests: list[comps_likelihood.ContinuationRequest],
    batch_size: int = 64,
    device: torch.device | str | None = None,
) -> list[MntpScore]:
    """Score COMPS property-phrase continuations with the MNTP backend."""

    return score_mntp_text_spans(
        model=model,
        tokenizer=tokenizer,
        requests=[
            MntpTextSpanRequest(
                text=f"{request.prefix} {request.continuation}",
                target_start_char=len(request.prefix) + 1,
            )
            for request in requests
        ],
        batch_size=batch_size,
        device=device,
    )


def score_mntp_entity_tracking_continuations(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    requests: list[entity_tracking_likelihood.ContinuationRequest],
    batch_size: int = 64,
    device: torch.device | str | None = None,
) -> dict[int, dict[int, MntpScore]]:
    """Score Entity Tracking options with the MNTP backend."""

    scores = score_mntp_text_spans(
        model=model,
        tokenizer=tokenizer,
        requests=[
            MntpTextSpanRequest(
                text=f"{request.input_prefix}{request.option}",
                target_start_char=len(request.input_prefix),
            )
            for request in requests
        ],
        batch_size=batch_size,
        device=device,
    )
    score_map: dict[int, dict[int, MntpScore]] = {}
    for request, score in zip(requests, scores, strict=True):
        score_map.setdefault(request.item_index, {})[request.option_index] = score
    return score_map


def evaluate_blimp_mntp_likelihood(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[blimp_likelihood.BlimpItem],
    batch_size: int = 64,
    device: torch.device | str | None = None,
    score_reduction: ScoreReduction = "sum",
) -> MntpBenchmarkResult:
    """Evaluate BLIMP-style sentence pairs with MNTP scores."""

    score_reduction = validate_score_reduction(score_reduction)
    good_scores = score_mntp_sentences(
        model=model,
        tokenizer=tokenizer,
        sentences=[item.sentence_good for item in items],
        batch_size=batch_size,
        device=device,
    )
    bad_scores = score_mntp_sentences(
        model=model,
        tokenizer=tokenizer,
        sentences=[item.sentence_bad for item in items],
        batch_size=batch_size,
        device=device,
    )
    records = blimp_likelihood.build_item_records(
        items=items,
        good_scores=good_scores,  # type: ignore[arg-type]
        bad_scores=bad_scores,  # type: ignore[arg-type]
        score_reduction=score_reduction,
    )
    return MntpBenchmarkResult(
        item_records=records,
        summary_rows=blimp_likelihood.aggregate_item_records(records),
    )


def evaluate_comps_mntp_likelihood(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[comps_likelihood.CompsItem],
    batch_size: int = 64,
    device: torch.device | str | None = None,
    score_reduction: ScoreReduction = "sum",
) -> MntpBenchmarkResult:
    """Evaluate COMPS continuation pairs with MNTP scores."""

    score_reduction = validate_score_reduction(score_reduction)
    scores = score_mntp_comps_continuations(
        model=model,
        tokenizer=tokenizer,
        requests=comps_likelihood.make_continuation_requests(items),
        batch_size=batch_size,
        device=device,
    )
    records = comps_likelihood.build_item_records(
        items=items,
        scores=scores,  # type: ignore[arg-type]
        score_reduction=score_reduction,
    )
    return MntpBenchmarkResult(
        item_records=records,
        summary_rows=comps_likelihood.aggregate_item_records(records),
    )


def evaluate_entity_tracking_mntp_likelihood(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[entity_tracking_likelihood.EntityTrackingItem],
    batch_size: int = 64,
    device: torch.device | str | None = None,
    score_reduction: ScoreReduction = "sum",
) -> MntpBenchmarkResult:
    """Evaluate Entity Tracking option likelihoods with MNTP scores."""

    score_reduction = validate_score_reduction(score_reduction)
    score_map = score_mntp_entity_tracking_continuations(
        model=model,
        tokenizer=tokenizer,
        requests=entity_tracking_likelihood.make_continuation_requests(items),
        batch_size=batch_size,
        device=device,
    )
    records = entity_tracking_likelihood.build_item_records(
        items=items,
        score_map=score_map,  # type: ignore[arg-type]
        score_reduction=score_reduction,
    )
    return MntpBenchmarkResult(
        item_records=records,
        summary_rows=entity_tracking_likelihood.aggregate_item_records(records),
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

SUMMARY_FIELDNAMES = [
    *RUN_METADATA_FIELDNAMES,
    "benchmark",
    "group_scope",
    "group_name",
    "score_reduction",
    "items",
    "correct",
    "score",
    "mean_margin",
    "ties",
    "score_kind",
    "micro_score",
]


def add_summary_metadata(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return copies of summary rows tagged with benchmark and run metadata."""

    metadata = metadata or {}
    return [{**metadata, "benchmark": benchmark, **row} for row in rows]


def append_summary_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    benchmark: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append aggregate MNTP summary rows as CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in add_summary_metadata(rows, benchmark=benchmark, metadata=metadata):
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDNAMES})


def append_item_records_jsonl(
    records: list[dict[str, Any]],
    path: str | Path,
    *,
    benchmark: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append per-item MNTP records as JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps({**metadata, "benchmark": benchmark, **record}, sort_keys=True))
            f.write("\n")


__all__ = [
    "RUN_METADATA_FIELDNAMES",
    "SUMMARY_FIELDNAMES",
    "EncodedMntpTextSpan",
    "MntpBenchmarkResult",
    "MntpScore",
    "MntpTextSpanRequest",
    "ScoreReduction",
    "add_summary_metadata",
    "append_item_records_jsonl",
    "append_summary_csv",
    "encode_mntp_text_span",
    "evaluate_blimp_mntp_likelihood",
    "evaluate_comps_mntp_likelihood",
    "evaluate_entity_tracking_mntp_likelihood",
    "resolve_mask_token_id",
    "score_mntp_comps_continuations",
    "score_mntp_entity_tracking_continuations",
    "score_mntp_sentences",
    "score_mntp_text_spans",
    "validate_score_reduction",
]
