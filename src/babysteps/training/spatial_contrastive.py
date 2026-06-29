"""Score-level contrastive losses for synthetic spatial examples."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


RankReduction = Literal["mean", "sum", "none"]


def _validate_temperature(temperature: float) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    return float(temperature)


def _reduce_loss(loss: torch.Tensor, reduction: RankReduction) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be 'mean', 'sum', or 'none'.")


def completion_rank_loss(
    *,
    completion_scores: torch.Tensor,
    hard_negative_scores: torch.Tensor,
    temperature: float = 1.0,
    reduction: RankReduction = "mean",
) -> torch.Tensor:
    """Rank the correct completion above the hard-negative completion.

    Scores should be target log-likelihoods or another scalar where larger is
    better. The loss is low when ``completion_scores > hard_negative_scores``.
    """

    temperature = _validate_temperature(temperature)
    loss = F.softplus((hard_negative_scores - completion_scores) / temperature)
    return _reduce_loss(loss, reduction)


def context_rank_loss(
    *,
    s11: torch.Tensor,
    s21: torch.Tensor,
    s22: torch.Tensor,
    s12: torch.Tensor,
    temperature: float = 1.0,
    reduction: RankReduction = "mean",
) -> torch.Tensor:
    """EWoK-style context-rank loss over a 2x2 score matrix.

    ``s11`` scores Target1 given Context1, ``s21`` scores Target1 given the
    mismatched Context2, ``s22`` scores Target2 given Context2, and ``s12``
    scores Target2 given the mismatched Context1.
    """

    temperature = _validate_temperature(temperature)
    target1_loss = F.softplus((s21 - s11) / temperature)
    target2_loss = F.softplus((s12 - s22) / temperature)
    loss = 0.5 * (target1_loss + target2_loss)
    return _reduce_loss(loss, reduction)


__all__ = [
    "RankReduction",
    "completion_rank_loss",
    "context_rank_loss",
]
