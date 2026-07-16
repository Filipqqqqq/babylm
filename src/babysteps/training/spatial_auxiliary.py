"""Auxiliary spatial-rank batches and metrics."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import torch

from babysteps.training.branch_scoring import (
    Branch,
    BranchScoreReduction,
    score_causal_branches,
)
from babysteps.training.spatial_contrastive import (
    completion_rank_loss,
    context_rank_loss,
    counterfactual_rank_loss,
)


SpatialAuxiliaryObjective = Literal[
    "completion_rank",
    "context_rank",
    "counterfactual_rank",
]


@dataclass(frozen=True)
class CompletionRankExample:
    """One context with a correct and hard-negative completion."""

    item_id: str
    context: str
    completion: str
    hard_negative: str
    template_family: str = ""


@dataclass(frozen=True)
class ContextRankExample:
    """One EWoK-style context-rank pair."""

    item_id: str
    context1: str
    context2: str
    target1: str
    target2: str
    template_family: str = ""


@dataclass(frozen=True)
class CounterfactualRankExample:
    """A minimal context pair with different correct completions."""

    item_id: str
    context1: str
    context2: str
    target1: str
    target2: str
    template_family: str = ""


SpatialAuxiliaryExample = (
    CompletionRankExample | ContextRankExample | CounterfactualRankExample
)


class SpatialAuxiliarySampler:
    """Infinite shuffled sampler over spatial auxiliary examples."""

    def __init__(
        self,
        examples: Sequence[SpatialAuxiliaryExample],
        *,
        batch_size: int,
        seed: int = 0,
    ) -> None:
        if not examples:
            raise ValueError("spatial auxiliary examples must not be empty.")
        if batch_size <= 0:
            raise ValueError("spatial auxiliary batch_size must be positive.")
        self.examples = list(examples)
        self.batch_size = int(batch_size)
        self._rng = random.Random(seed)
        self._order: list[int] = []
        self._index = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._order = list(range(len(self.examples)))
        self._rng.shuffle(self._order)
        self._index = 0

    def next_batch(self) -> list[SpatialAuxiliaryExample]:
        batch: list[SpatialAuxiliaryExample] = []
        while len(batch) < self.batch_size:
            if self._index >= len(self._order):
                self._reshuffle()
            take_index = self._order[self._index]
            self._index += 1
            batch.append(self.examples[take_index])
        return batch


def _jsonl_paths(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.glob("*.jsonl"))


def _row_objective(row: dict[str, Any]) -> SpatialAuxiliaryObjective:
    raw_objective = row.get("rank_objective")
    if raw_objective:
        objective = str(raw_objective)
    elif "Context1" in row and "Target1" in row:
        objective = "context_rank"
    else:
        objective = "completion_rank"
    if objective not in {
        "completion_rank",
        "context_rank",
        "counterfactual_rank",
    }:
        raise ValueError(f"Unsupported spatial auxiliary rank_objective={objective!r}.")
    return objective  # type: ignore[return-value]


def parse_spatial_auxiliary_row(
    row: dict[str, Any],
    *,
    objective: SpatialAuxiliaryObjective | None = None,
    fallback_item_id: str = "",
) -> SpatialAuxiliaryExample:
    row_objective = _row_objective(row)
    if objective is not None and row_objective != objective:
        raise ValueError(
            f"Expected spatial auxiliary objective={objective!r}, "
            f"got row objective={row_objective!r}."
        )

    item_id = str(row.get("item_id", fallback_item_id))
    template_family = str(row.get("template_family", row.get("ContextType", "")))
    if row_objective == "completion_rank":
        return CompletionRankExample(
            item_id=item_id,
            context=str(row["context"]),
            completion=str(row["completion"]),
            hard_negative=str(row["hard_negative"]),
            template_family=template_family,
        )

    example_class = (
        CounterfactualRankExample
        if row_objective == "counterfactual_rank"
        else ContextRankExample
    )
    return example_class(
        item_id=item_id,
        context1=str(row["Context1"]),
        context2=str(row["Context2"]),
        target1=str(row["Target1"]),
        target2=str(row["Target2"]),
        template_family=template_family,
    )


def read_spatial_auxiliary_examples(
    path: str | Path,
    *,
    objective: SpatialAuxiliaryObjective | None = None,
    max_items: int | None = None,
) -> list[SpatialAuxiliaryExample]:
    paths = _jsonl_paths(path)
    if not paths:
        raise FileNotFoundError(f"No spatial auxiliary JSONL files found at: {path}")

    examples: list[SpatialAuxiliaryExample] = []
    for jsonl_path in paths:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                examples.append(
                    parse_spatial_auxiliary_row(
                        row,
                        objective=objective,
                        fallback_item_id=f"{jsonl_path.stem}:{line_number}",
                    )
                )
                if max_items is not None and len(examples) >= max_items:
                    return examples
    return examples


def split_spatial_auxiliary_examples(
    examples: Sequence[SpatialAuxiliaryExample],
    *,
    eval_items: int = 0,
    eval_fraction: float = 0.0,
    seed: int = 0,
    shuffle: bool = True,
) -> tuple[list[SpatialAuxiliaryExample], list[SpatialAuxiliaryExample]]:
    if eval_items < 0:
        raise ValueError("spatial auxiliary eval_items must be non-negative.")
    if eval_fraction < 0.0 or eval_fraction >= 1.0:
        raise ValueError("spatial auxiliary eval_fraction must be in [0.0, 1.0).")

    ordered = list(examples)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ordered)

    eval_count = int(eval_items)
    if eval_count == 0 and eval_fraction > 0.0:
        eval_count = int(round(len(ordered) * eval_fraction))
    if eval_count >= len(ordered) and ordered:
        raise ValueError("spatial auxiliary split would leave no training examples.")

    if eval_count <= 0:
        return ordered, []
    return ordered[:-eval_count], ordered[-eval_count:]


def objective_for_examples(
    examples: Sequence[SpatialAuxiliaryExample],
) -> SpatialAuxiliaryObjective:
    if not examples:
        raise ValueError("examples must not be empty.")
    first = examples[0]
    if isinstance(first, CompletionRankExample):
        return "completion_rank"
    if isinstance(first, CounterfactualRankExample):
        return "counterfactual_rank"
    return "context_rank"


def build_spatial_branches(
    examples: Sequence[SpatialAuxiliaryExample],
) -> list[Branch]:
    objective = objective_for_examples(examples)
    branches: list[Branch] = []
    if objective == "completion_rank":
        for example in examples:
            if not isinstance(example, CompletionRankExample):
                raise ValueError("Mixed spatial auxiliary objectives are not supported.")
            branches.extend(
                [
                    Branch(example.context, example.completion),
                    Branch(example.context, example.hard_negative),
                ]
            )
    else:
        for example in examples:
            if not isinstance(example, (ContextRankExample, CounterfactualRankExample)):
                raise ValueError("Mixed spatial auxiliary objectives are not supported.")
            branches.extend(
                [
                    Branch(example.context1, example.target1),
                    Branch(example.context2, example.target1),
                    Branch(example.context2, example.target2),
                    Branch(example.context1, example.target2),
                ]
            )
    return branches


def compute_spatial_auxiliary_loss(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    examples: Sequence[SpatialAuxiliaryExample],
    temperature: float = 1.0,
    score_reduction: BranchScoreReduction = "mean",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute the differentiable spatial rank loss for one example batch."""

    objective = objective_for_examples(examples)
    branches = build_spatial_branches(examples)
    scores = score_causal_branches(
        model=model,
        tokenizer=tokenizer,
        branches=branches,
        score_reduction=score_reduction,
        device=device,
    ).scores

    if objective == "completion_rank":
        pair_scores = scores.reshape(len(examples), 2)
        return completion_rank_loss(
            completion_scores=pair_scores[:, 0],
            hard_negative_scores=pair_scores[:, 1],
            temperature=temperature,
        )

    matrix_scores = scores.reshape(len(examples), 4)
    loss_function = (
        counterfactual_rank_loss
        if objective == "counterfactual_rank"
        else context_rank_loss
    )
    return loss_function(
        s11=matrix_scores[:, 0],
        s21=matrix_scores[:, 1],
        s22=matrix_scores[:, 2],
        s12=matrix_scores[:, 3],
        temperature=temperature,
    )


def _batched(
    examples: Sequence[SpatialAuxiliaryExample],
    batch_size: int,
) -> Iterable[list[SpatialAuxiliaryExample]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for index in range(0, len(examples), batch_size):
        yield list(examples[index : index + batch_size])


def evaluate_spatial_auxiliary(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    examples: Sequence[SpatialAuxiliaryExample],
    batch_size: int = 64,
    temperature: float = 1.0,
    score_reduction: BranchScoreReduction = "mean",
    device: torch.device | str | None = None,
) -> dict[str, float | int | str]:
    """Evaluate held-out spatial rank accuracy and margins."""

    if not examples:
        raise ValueError("spatial auxiliary eval examples must not be empty.")

    objective = objective_for_examples(examples)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_items = 0
    correct = 0
    comparisons = 0
    margin_sum = 0.0
    k1_correct = 0
    k2_correct = 0
    k1_margin_sum = 0.0
    k2_margin_sum = 0.0

    try:
        with torch.no_grad():
            for batch in _batched(examples, batch_size):
                branches = build_spatial_branches(batch)
                scores = score_causal_branches(
                    model=model,
                    tokenizer=tokenizer,
                    branches=branches,
                    score_reduction=score_reduction,
                    device=device,
                ).scores
                total_items += len(batch)

                if objective == "completion_rank":
                    pair_scores = scores.reshape(len(batch), 2)
                    batch_loss = completion_rank_loss(
                        completion_scores=pair_scores[:, 0],
                        hard_negative_scores=pair_scores[:, 1],
                        temperature=temperature,
                    )
                    total_loss += float(batch_loss.item()) * len(batch)
                    margins = pair_scores[:, 0] - pair_scores[:, 1]
                    correct += int((margins > 0).sum().item())
                    comparisons += len(batch)
                    margin_sum += float(margins.sum().item())
                else:
                    matrix_scores = scores.reshape(len(batch), 4)
                    loss_function = (
                        counterfactual_rank_loss
                        if objective == "counterfactual_rank"
                        else context_rank_loss
                    )
                    batch_loss = loss_function(
                        s11=matrix_scores[:, 0],
                        s21=matrix_scores[:, 1],
                        s22=matrix_scores[:, 2],
                        s12=matrix_scores[:, 3],
                        temperature=temperature,
                    )
                    total_loss += float(batch_loss.item()) * len(batch)
                    context_k1 = matrix_scores[:, 0] - matrix_scores[:, 1]
                    context_k2 = matrix_scores[:, 2] - matrix_scores[:, 3]
                    if objective == "counterfactual_rank":
                        completion_k1 = matrix_scores[:, 0] - matrix_scores[:, 3]
                        completion_k2 = matrix_scores[:, 2] - matrix_scores[:, 1]
                        margins = torch.stack(
                            [context_k1, context_k2, completion_k1, completion_k2]
                        )
                        correct += int((margins > 0).sum().item())
                        comparisons += 4 * len(batch)
                        margin_sum += float(margins.mean(dim=0).sum().item())
                    else:
                        correct += int((context_k1 > 0).sum().item())
                        correct += int((context_k2 > 0).sum().item())
                        comparisons += 2 * len(batch)
                        margin_sum += float(
                            (0.5 * (context_k1 + context_k2)).sum().item()
                        )
                    k1 = context_k1
                    k2 = context_k2
                    batch_k1_correct = int((k1 > 0).sum().item())
                    batch_k2_correct = int((k2 > 0).sum().item())
                    k1_correct += batch_k1_correct
                    k2_correct += batch_k2_correct
                    k1_margin_sum += float(k1.sum().item())
                    k2_margin_sum += float(k2.sum().item())
    finally:
        if was_training:
            model.train()

    result: dict[str, float | int | str] = {
        "objective": objective,
        "items": total_items,
        "comparisons": comparisons,
        "correct": correct,
        "score": correct / max(1, comparisons),
        "loss": total_loss / max(1, total_items),
        "mean_margin": margin_sum / max(1, total_items),
    }
    if objective in {"context_rank", "counterfactual_rank"}:
        result.update(
            {
                "k1_correct": k1_correct,
                "k2_correct": k2_correct,
                "k1_score": k1_correct / max(1, total_items),
                "k2_score": k2_correct / max(1, total_items),
                "mean_k1_margin": k1_margin_sum / max(1, total_items),
                "mean_k2_margin": k2_margin_sum / max(1, total_items),
            }
        )
    return result


__all__ = [
    "CompletionRankExample",
    "ContextRankExample",
    "CounterfactualRankExample",
    "SpatialAuxiliaryExample",
    "SpatialAuxiliaryObjective",
    "SpatialAuxiliarySampler",
    "build_spatial_branches",
    "compute_spatial_auxiliary_loss",
    "evaluate_spatial_auxiliary",
    "objective_for_examples",
    "parse_spatial_auxiliary_row",
    "read_spatial_auxiliary_examples",
    "split_spatial_auxiliary_examples",
]
