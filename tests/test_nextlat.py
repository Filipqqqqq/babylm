from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from babysteps.training.nextlat import (
    LatentDynamicsMLP,
    NextLatConfig,
    build_transition_mask,
    compute_nextlat_loss,
    compute_nextlat_rollout_losses,
    nextlat_kl_weight,
    nextlat_config_from_dict,
)


class _ZeroDynamics(nn.Module):
    def forward(
        self,
        current_hidden: torch.Tensor,
        next_token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(current_hidden)


def test_latent_dynamics_mlp_preserves_shape_device_and_dtype() -> None:
    dynamics = LatentDynamicsMLP(
        hidden_size=4,
        token_embedding_size=3,
        mlp_hidden_dim=8,
        mlp_layers=2,
    ).double()
    current_hidden = torch.randn(2, 5, 4, dtype=torch.float64)
    next_token_embeddings = torch.randn(2, 5, 3, dtype=torch.float64)

    output = dynamics(current_hidden, next_token_embeddings)

    assert output.shape == current_hidden.shape
    assert output.device == current_hidden.device
    assert output.dtype == current_hidden.dtype


def test_latent_dynamics_mlp_predicts_residual_delta_from_normalized_input() -> None:
    dynamics = LatentDynamicsMLP(
        hidden_size=4,
        token_embedding_size=3,
        mlp_hidden_dim=8,
        mlp_layers=2,
    )
    with torch.no_grad():
        for parameter in dynamics.net.parameters():
            parameter.zero_()
    current_hidden = torch.randn(2, 5, 4)
    next_token_embeddings = torch.randn(2, 5, 3)

    output = dynamics(current_hidden, next_token_embeddings)

    torch.testing.assert_close(output, current_hidden)


def test_transition_mask_excludes_eos_bos_and_padding() -> None:
    input_ids = torch.tensor([[1, 7, 2, 1, 8, 0]])
    attention_mask = torch.tensor([[True, True, True, True, True, False]])
    loss_mask = torch.tensor([[True, True, True, True, False, False]])

    mask = build_transition_mask(
        input_ids=input_ids,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        bos_token_id=1,
        eos_token_id=2,
        mask_eos_bos_boundaries=True,
    )

    expected = torch.tensor([[True, True, False, True, False]])
    torch.testing.assert_close(mask, expected)


def test_nextlat_smooth_l1_is_averaged_over_valid_transitions() -> None:
    hidden_states = torch.tensor(
        [
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 4.0],
            ]
        ]
    )
    input_ids = torch.tensor([[1, 3, 4]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    embeddings = nn.Embedding(8, 2)

    result = compute_nextlat_loss(
        dynamics=_ZeroDynamics(),  # type: ignore[arg-type]
        input_embeddings=embeddings,
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        config=NextLatConfig(enabled=True),
        bos_token_id=1,
        eos_token_id=2,
    )

    assert result.transition_count.item() == 2
    torch.testing.assert_close(result.next_h_loss, torch.tensor(1.25))
    assert result.persistence_transition_count.item() == 2
    torch.testing.assert_close(result.persistence_loss, torch.tensor(1.625))


def test_nextlat_horizon_rollout_counts_cumulative_valid_transitions() -> None:
    hidden_states = torch.randn(1, 4, 3)
    input_ids = torch.tensor([[1, 3, 4, 5]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    embeddings = nn.Embedding(8, 3)

    result = compute_nextlat_loss(
        dynamics=_ZeroDynamics(),  # type: ignore[arg-type]
        input_embeddings=embeddings,
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        config=NextLatConfig(enabled=True, horizon=2),
        bos_token_id=1,
        eos_token_id=2,
    )

    assert result.transition_count.item() == 5
    assert result.persistence_transition_count.item() == 3


def test_nextlat_rollout_diagnostics_report_exact_selected_horizons() -> None:
    hidden_states = torch.randn(1, 5, 3)
    input_ids = torch.tensor([[1, 3, 4, 5, 6]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    embeddings = nn.Embedding(8, 3)

    result = compute_nextlat_rollout_losses(
        dynamics=_ZeroDynamics(),  # type: ignore[arg-type]
        input_embeddings=embeddings,
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        horizons=(1, 2, 4),
        bos_token_id=1,
        eos_token_id=2,
    )

    assert tuple(result.loss_by_horizon) == (1, 2, 4)
    assert result.transition_count_by_horizon[1].item() == 4
    assert result.transition_count_by_horizon[2].item() == 3
    assert result.transition_count_by_horizon[4].item() == 1
    assert tuple(result.persistence_loss_by_horizon) == (1, 2, 4)


def test_nextlat_detaches_target_hidden_states_but_trains_dynamics() -> None:
    torch.manual_seed(123)
    dynamics = LatentDynamicsMLP(
        hidden_size=4,
        token_embedding_size=4,
        mlp_hidden_dim=8,
        mlp_layers=2,
    )
    hidden_states = torch.randn(1, 2, 4, requires_grad=True)
    input_ids = torch.tensor([[1, 3]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    embeddings = nn.Embedding(8, 4)

    result = compute_nextlat_loss(
        dynamics=dynamics,
        input_embeddings=embeddings,
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        config=NextLatConfig(enabled=True, detach_targets=True),
        bos_token_id=1,
        eos_token_id=2,
    )
    result.next_h_loss.backward()

    assert not result.persistence_loss.requires_grad
    assert hidden_states.grad is not None
    assert hidden_states.grad[:, 0].abs().sum() > 0
    torch.testing.assert_close(
        hidden_states.grad[:, 1],
        torch.zeros_like(hidden_states.grad[:, 1]),
    )
    assert any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in dynamics.parameters()
    )


def test_nextlat_topk_kl_matches_renormalized_teacher_topk() -> None:
    hidden_states = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        ]
    )
    input_ids = torch.tensor([[1, 3]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    input_embeddings = nn.Embedding(8, 3)
    output_embeddings = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        output_embeddings.weight.copy_(torch.eye(3))

    result = compute_nextlat_loss(
        dynamics=_ZeroDynamics(),  # type: ignore[arg-type]
        input_embeddings=input_embeddings,
        output_embeddings=output_embeddings,
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        config=NextLatConfig(enabled=True, lambda_kl=0.1, kl_top_k=2),
        bos_token_id=1,
        eos_token_id=2,
        kl_weight=0.1,
    )

    teacher_top_logits = torch.tensor([[2.0, 1.0]])
    teacher_log_probs = F.log_softmax(teacher_top_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(torch.zeros_like(teacher_top_logits), dim=-1)
    expected = (teacher_probs * (teacher_log_probs - student_log_probs)).sum()

    assert result.kl_transition_count.item() == 1
    torch.testing.assert_close(result.kl_loss, expected)


def test_nextlat_topk_kl_supports_tied_embedding_output_head() -> None:
    hidden_states = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        ]
    )
    input_ids = torch.tensor([[1, 3]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    input_embeddings = nn.Embedding(8, 3)
    output_embeddings = nn.Embedding(3, 3)
    with torch.no_grad():
        output_embeddings.weight.copy_(torch.eye(3))

    result = compute_nextlat_loss(
        dynamics=_ZeroDynamics(),  # type: ignore[arg-type]
        input_embeddings=input_embeddings,
        output_embeddings=output_embeddings,
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        config=NextLatConfig(enabled=True, lambda_kl=0.1, kl_top_k=2),
        bos_token_id=1,
        eos_token_id=2,
        kl_weight=0.1,
    )

    teacher_top_logits = torch.tensor([[2.0, 1.0]])
    teacher_log_probs = F.log_softmax(teacher_top_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(torch.zeros_like(teacher_top_logits), dim=-1)
    expected = (teacher_probs * (teacher_log_probs - student_log_probs)).sum()

    assert result.kl_transition_count.item() == 1
    torch.testing.assert_close(result.kl_loss, expected)


def test_nextlat_topk_kl_supports_tied_output_weight_parameter() -> None:
    hidden_states = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        ]
    )
    input_ids = torch.tensor([[1, 3]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    input_embeddings = nn.Embedding(8, 3)
    output_weight = nn.Parameter(torch.eye(3))

    result = compute_nextlat_loss(
        dynamics=_ZeroDynamics(),  # type: ignore[arg-type]
        input_embeddings=input_embeddings,
        output_embeddings=output_weight,
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        config=NextLatConfig(enabled=True, lambda_kl=0.1, kl_top_k=2),
        bos_token_id=1,
        eos_token_id=2,
        kl_weight=0.1,
    )

    teacher_top_logits = torch.tensor([[2.0, 1.0]])
    teacher_log_probs = F.log_softmax(teacher_top_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(torch.zeros_like(teacher_top_logits), dim=-1)
    expected = (teacher_probs * (teacher_log_probs - student_log_probs)).sum()

    assert result.kl_transition_count.item() == 1
    torch.testing.assert_close(result.kl_loss, expected)


def test_nextlat_kl_warmup_schedule() -> None:
    config = nextlat_config_from_dict(
        {
            "enabled": True,
            "lambda_kl": 0.1,
            "kl_warmup_start_words": 100_000_000,
            "kl_warmup_end_words": 200_000_000,
        }
    )

    assert nextlat_kl_weight(config, words_seen=None) == 0.0
    assert nextlat_kl_weight(config, words_seen=100_000_000) == 0.0
    assert nextlat_kl_weight(config, words_seen=150_000_000) == pytest.approx(0.05)
    assert nextlat_kl_weight(config, words_seen=200_000_000) == pytest.approx(0.1)


def test_nextlat_config_parses_rollout_diagnostics() -> None:
    config = nextlat_config_from_dict(
        {
            "enabled": True,
            "rollout_eval_horizons": [1, 2, 4, 8],
            "rollout_eval_words": [200_000_000, 300_000_000],
            "rollout_eval_max_batches": 2,
            "rollout_eval_max_seq_len": 256,
        }
    )

    assert config.rollout_eval_horizons == (1, 2, 4, 8)
    assert config.rollout_eval_words == (200_000_000.0, 300_000_000.0)
    assert config.rollout_eval_max_batches == 2
    assert config.rollout_eval_max_seq_len == 256
