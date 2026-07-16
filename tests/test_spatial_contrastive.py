from __future__ import annotations

import torch

from babysteps.training.spatial_contrastive import (
    completion_rank_loss,
    context_rank_loss,
    counterfactual_rank_loss,
)


def test_completion_rank_loss_prefers_correct_completion_scores() -> None:
    good_loss = completion_rank_loss(
        completion_scores=torch.tensor([3.0, 2.0]),
        hard_negative_scores=torch.tensor([0.0, -1.0]),
    )
    bad_loss = completion_rank_loss(
        completion_scores=torch.tensor([0.0, -1.0]),
        hard_negative_scores=torch.tensor([3.0, 2.0]),
    )

    assert good_loss < bad_loss


def test_context_rank_loss_prefers_matched_contexts() -> None:
    good_loss = context_rank_loss(
        s11=torch.tensor([3.0]),
        s21=torch.tensor([0.0]),
        s22=torch.tensor([2.5]),
        s12=torch.tensor([-0.5]),
    )
    bad_loss = context_rank_loss(
        s11=torch.tensor([0.0]),
        s21=torch.tensor([3.0]),
        s22=torch.tensor([-0.5]),
        s12=torch.tensor([2.5]),
    )

    assert good_loss < bad_loss


def test_rank_losses_support_none_reduction() -> None:
    loss = context_rank_loss(
        s11=torch.tensor([3.0, 2.0]),
        s21=torch.tensor([0.0, 0.5]),
        s22=torch.tensor([3.0, 2.0]),
        s12=torch.tensor([0.0, 0.5]),
        reduction="none",
    )

    assert loss.shape == (2,)


def test_counterfactual_rank_loss_requires_both_row_and_column_matches() -> None:
    good_loss = counterfactual_rank_loss(
        s11=torch.tensor([4.0]),
        s21=torch.tensor([0.0]),
        s22=torch.tensor([3.0]),
        s12=torch.tensor([-1.0]),
    )
    bad_loss = counterfactual_rank_loss(
        s11=torch.tensor([0.0]),
        s21=torch.tensor([4.0]),
        s22=torch.tensor([-1.0]),
        s12=torch.tensor([3.0]),
    )

    assert good_loss < bad_loss
