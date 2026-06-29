"""Differentiable branch scoring for causal language models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch


BranchScoreReduction = Literal["sum", "mean"]


@dataclass(frozen=True)
class Branch:
    """One conditional scoring request: score ``target`` after ``context``."""

    context: str
    target: str


@dataclass(frozen=True)
class EncodedBranch:
    """Tokenized branch with a target span in next-token coordinates."""

    input_ids: tuple[int, ...]
    target_start: int
    target_end: int


@dataclass(frozen=True)
class BranchScoreOutput:
    """Reduced branch scores and target token counts."""

    scores: torch.Tensor
    target_token_counts: torch.Tensor


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def resolve_bos_like_token_id(tokenizer: Any) -> int:
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            return int(token_id)
    raise ValueError("Tokenizer has no bos_token_id, eos_token_id, or pad_token_id.")


def resolve_padding_token_id(tokenizer: Any, *, fallback_token_id: int) -> int:
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            return int(token_id)
    return int(fallback_token_id)


def encode_branch(
    tokenizer: Any,
    branch: Branch,
    *,
    bos_like_token_id: int,
) -> EncodedBranch:
    """Encode ``context + " " + target`` and locate target tokens."""

    context_ids = _encode(tokenizer, branch.context)
    full_ids = _encode(tokenizer, f"{branch.context} {branch.target}")
    target_start = len(context_ids)
    target_end = len(full_ids)
    if target_end <= target_start:
        raise ValueError(
            "Target span has no tokens after context slicing for "
            f"context={branch.context!r}, target={branch.target!r}."
        )
    return EncodedBranch(
        input_ids=tuple([bos_like_token_id, *full_ids]),
        target_start=target_start,
        target_end=target_end,
    )


def score_causal_branches(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    branches: Sequence[Branch],
    score_reduction: BranchScoreReduction = "mean",
    device: torch.device | str | None = None,
) -> BranchScoreOutput:
    """Return differentiable target-token log-probability scores."""

    if score_reduction not in {"sum", "mean"}:
        raise ValueError("score_reduction must be 'sum' or 'mean'.")

    model_device = torch.device(device) if device is not None else next(model.parameters()).device
    if not branches:
        empty = torch.empty(0, dtype=torch.float32, device=model_device)
        return BranchScoreOutput(scores=empty, target_token_counts=empty)

    bos_like_token_id = resolve_bos_like_token_id(tokenizer)
    padding_token_id = resolve_padding_token_id(
        tokenizer,
        fallback_token_id=bos_like_token_id,
    )
    encoded = [
        encode_branch(tokenizer, branch, bos_like_token_id=bos_like_token_id)
        for branch in branches
    ]

    max_length = max(len(item.input_ids) for item in encoded)
    input_ids = torch.full(
        (len(encoded), max_length),
        padding_token_id,
        dtype=torch.long,
        device=model_device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, item in enumerate(encoded):
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

    scores: list[torch.Tensor] = []
    counts: list[int] = []
    for row_index, item in enumerate(encoded):
        target_log_probs = token_log_probs[row_index, item.target_start : item.target_end]
        target_sum = target_log_probs.sum()
        target_count = int(item.target_end - item.target_start)
        counts.append(target_count)
        if score_reduction == "sum":
            scores.append(target_sum)
        else:
            scores.append(target_sum / max(1, target_count))

    return BranchScoreOutput(
        scores=torch.stack(scores),
        target_token_counts=torch.tensor(counts, dtype=torch.float32, device=model_device),
    )


__all__ = [
    "Branch",
    "BranchScoreOutput",
    "BranchScoreReduction",
    "EncodedBranch",
    "encode_branch",
    "resolve_bos_like_token_id",
    "resolve_padding_token_id",
    "score_causal_branches",
]
