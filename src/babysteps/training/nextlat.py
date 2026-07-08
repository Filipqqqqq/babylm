"""Experimental NextLat auxiliary loss prototype.

This module implements a conservative training-only Next-Latent Prediction
prototype for BabySteps ablations. It does not change the saved Hugging Face
model architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class NextLatConfig:
    """Configuration for the experimental NextLat auxiliary objective."""

    enabled: bool = False
    horizon: int = 1
    lambda_next_h: float = 1.0
    lambda_kl: float = 0.0
    kl_mode: str = "topk"
    kl_top_k: int = 256
    kl_transition_chunk_size: int = 1024
    kl_warmup_start_words: float | None = None
    kl_warmup_end_words: float | None = None
    detach_kl_lm_head: bool = True
    mlp_hidden_dim: int = 1024
    mlp_layers: int = 3
    detach_targets: bool = True
    mask_eos_bos_boundaries: bool = True


@dataclass(frozen=True)
class NextLatLossResult:
    """Loss components returned by the NextLat prototype."""

    next_h_loss: torch.Tensor
    transition_count: torch.Tensor
    persistence_loss: torch.Tensor
    persistence_transition_count: torch.Tensor
    kl_loss: torch.Tensor
    kl_transition_count: torch.Tensor
    kl_weight: torch.Tensor


class LatentDynamicsMLP(nn.Module):
    """Predict the next hidden state from a hidden state and next-token embedding."""

    def __init__(
        self,
        *,
        hidden_size: int,
        token_embedding_size: int | None = None,
        mlp_hidden_dim: int = 1024,
        mlp_layers: int = 3,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size)
        token_embedding_size = (
            hidden_size if token_embedding_size is None else int(token_embedding_size)
        )
        mlp_hidden_dim = int(mlp_hidden_dim)
        mlp_layers = int(mlp_layers)
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if token_embedding_size <= 0:
            raise ValueError("token_embedding_size must be positive.")
        if mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be positive.")
        if mlp_layers <= 0:
            raise ValueError("mlp_layers must be positive.")

        input_size = hidden_size + token_embedding_size
        layers: list[nn.Module] = []
        if mlp_layers == 1:
            layers.append(nn.Linear(input_size, hidden_size))
        else:
            layers.extend([nn.Linear(input_size, mlp_hidden_dim), nn.GELU()])
            for _ in range(mlp_layers - 2):
                layers.extend([nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.GELU()])
            layers.append(nn.Linear(mlp_hidden_dim, hidden_size))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        current_hidden: torch.Tensor,
        next_token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if current_hidden.shape[:-1] != next_token_embeddings.shape[:-1]:
            raise ValueError(
                "current_hidden and next_token_embeddings must share batch/time shape."
            )
        if next_token_embeddings.dtype != current_hidden.dtype:
            next_token_embeddings = next_token_embeddings.to(dtype=current_hidden.dtype)
        return self.net(torch.cat([current_hidden, next_token_embeddings], dim=-1))


def nextlat_config_from_dict(config: dict[str, Any] | None) -> NextLatConfig:
    """Parse and validate a NextLat config block."""

    raw = dict(config or {})
    lambda_kl = float(raw.get("lambda_kl", 0.0))
    kl_warmup_start_words = raw.get("kl_warmup_start_words")
    kl_warmup_end_words = raw.get("kl_warmup_end_words")
    parsed = NextLatConfig(
        enabled=bool(raw.get("enabled", False)),
        horizon=int(raw.get("horizon", 1)),
        lambda_next_h=float(raw.get("lambda_next_h", 1.0)),
        lambda_kl=lambda_kl,
        kl_mode=str(raw.get("kl_mode", "topk")).lower(),
        kl_top_k=int(raw.get("kl_top_k", 256)),
        kl_transition_chunk_size=int(raw.get("kl_transition_chunk_size", 1024)),
        kl_warmup_start_words=(
            None
            if kl_warmup_start_words is None
            else float(kl_warmup_start_words)
        ),
        kl_warmup_end_words=(
            None if kl_warmup_end_words is None else float(kl_warmup_end_words)
        ),
        detach_kl_lm_head=bool(raw.get("detach_kl_lm_head", True)),
        mlp_hidden_dim=int(raw.get("mlp_hidden_dim", 1024)),
        mlp_layers=int(raw.get("mlp_layers", 3)),
        detach_targets=bool(raw.get("detach_targets", True)),
        mask_eos_bos_boundaries=bool(raw.get("mask_eos_bos_boundaries", True)),
    )
    if parsed.horizon <= 0:
        raise ValueError("nextlat.horizon must be positive.")
    if parsed.lambda_next_h < 0.0:
        raise ValueError("nextlat.lambda_next_h must be non-negative.")
    if parsed.lambda_kl < 0.0:
        raise ValueError("nextlat.lambda_kl must be non-negative.")
    if parsed.kl_mode not in {"topk"}:
        raise ValueError("nextlat.kl_mode must be 'topk'.")
    if parsed.kl_top_k <= 0:
        raise ValueError("nextlat.kl_top_k must be positive.")
    if parsed.kl_transition_chunk_size <= 0:
        raise ValueError("nextlat.kl_transition_chunk_size must be positive.")
    if (parsed.kl_warmup_start_words is None) != (
        parsed.kl_warmup_end_words is None
    ):
        raise ValueError(
            "Set both nextlat.kl_warmup_start_words and "
            "nextlat.kl_warmup_end_words, or neither."
        )
    if parsed.kl_warmup_start_words is not None:
        if parsed.kl_warmup_start_words < 0.0:
            raise ValueError("nextlat.kl_warmup_start_words must be non-negative.")
        if parsed.kl_warmup_end_words <= parsed.kl_warmup_start_words:
            raise ValueError(
                "nextlat.kl_warmup_end_words must be greater than "
                "nextlat.kl_warmup_start_words."
            )
    if parsed.mlp_hidden_dim <= 0:
        raise ValueError("nextlat.mlp_hidden_dim must be positive.")
    if parsed.mlp_layers <= 0:
        raise ValueError("nextlat.mlp_layers must be positive.")
    return parsed


def nextlat_kl_weight(
    config: NextLatConfig,
    *,
    words_seen: float | None,
) -> float:
    """Return the scheduled KL weight for the current training point."""

    if not config.enabled or config.lambda_kl <= 0.0:
        return 0.0
    if config.kl_warmup_start_words is None:
        return config.lambda_kl
    if words_seen is None:
        return 0.0
    if words_seen <= config.kl_warmup_start_words:
        return 0.0
    if words_seen >= config.kl_warmup_end_words:
        return config.lambda_kl
    warmup_span = config.kl_warmup_end_words - config.kl_warmup_start_words
    progress = (words_seen - config.kl_warmup_start_words) / warmup_span
    return config.lambda_kl * max(0.0, min(1.0, progress))


def build_transition_mask(
    *,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    bos_token_id: int | None,
    eos_token_id: int | None,
    mask_eos_bos_boundaries: bool = True,
) -> torch.Tensor:
    """Return valid one-step hidden-state transition positions."""

    if input_ids.size(1) < 2:
        return torch.zeros(
            input_ids.size(0),
            0,
            dtype=torch.bool,
            device=input_ids.device,
        )
    transition_mask = (
        attention_mask[:, :-1].bool()
        & attention_mask[:, 1:].bool()
        & loss_mask[:, :-1].bool()
    )
    if (
        mask_eos_bos_boundaries
        and bos_token_id is not None
        and eos_token_id is not None
    ):
        eos_to_bos = (
            (input_ids[:, :-1] == int(eos_token_id))
            & (input_ids[:, 1:] == int(bos_token_id))
        )
        transition_mask = transition_mask & ~eos_to_bos
    return transition_mask


def _masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_dim_loss = F.smooth_l1_loss(pred, target, reduction="none")
    per_transition_loss = per_dim_loss.mean(dim=-1).float()
    weight = mask.to(dtype=torch.float32)
    transition_count = weight.sum()
    loss_sum = (per_transition_loss * weight).sum()
    return loss_sum, transition_count


def _output_logits(
    output_embeddings: nn.Module,
    hidden_states: torch.Tensor,
    *,
    detach_lm_head: bool,
) -> torch.Tensor:
    if detach_lm_head and isinstance(output_embeddings, nn.Linear):
        weight = output_embeddings.weight.detach()
        bias = (
            None
            if output_embeddings.bias is None
            else output_embeddings.bias.detach()
        )
        return F.linear(hidden_states, weight, bias)
    return output_embeddings(hidden_states)


def _masked_topk_kl(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    output_embeddings: nn.Module,
    top_k: int,
    transition_chunk_size: int,
    detach_lm_head: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate KL over teacher top-k vocabulary entries.

    Teacher top-k entries are selected in transition chunks to avoid materializing
    a full ``valid_transitions x vocab`` tensor. Teacher and student
    distributions are renormalized over the selected top-k set.
    """

    valid_mask = mask.reshape(-1).bool()
    transition_count = valid_mask.to(dtype=torch.float32).sum()
    if transition_count.item() == 0:
        zero = pred.new_zeros((), dtype=torch.float32)
        return zero, transition_count

    hidden_size = pred.size(-1)
    pred_flat = pred.reshape(-1, hidden_size)[valid_mask]
    target_flat = target.detach().reshape(-1, hidden_size)[valid_mask]
    loss_sum = pred.new_zeros((), dtype=torch.float32)

    for start in range(0, pred_flat.size(0), transition_chunk_size):
        stop = min(start + transition_chunk_size, pred_flat.size(0))
        pred_chunk = pred_flat[start:stop]
        target_chunk = target_flat[start:stop]

        with torch.no_grad():
            teacher_logits = _output_logits(
                output_embeddings,
                target_chunk,
                detach_lm_head=detach_lm_head,
            ).float()
            k = min(int(top_k), teacher_logits.size(-1))
            teacher_top_logits, top_indices = teacher_logits.topk(k, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_top_logits, dim=-1)
            teacher_probs = teacher_log_probs.exp()

        student_logits = _output_logits(
            output_embeddings,
            pred_chunk,
            detach_lm_head=detach_lm_head,
        ).float()
        student_top_logits = student_logits.gather(dim=-1, index=top_indices)
        student_log_probs = F.log_softmax(student_top_logits, dim=-1)
        kl_per_transition = teacher_probs * (teacher_log_probs - student_log_probs)
        loss_sum = loss_sum + kl_per_transition.sum(dim=-1).sum()

    return loss_sum, transition_count


def compute_nextlat_loss(
    *,
    dynamics: LatentDynamicsMLP,
    input_embeddings: nn.Module,
    output_embeddings: nn.Module | None = None,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    config: NextLatConfig,
    bos_token_id: int | None,
    eos_token_id: int | None,
    kl_weight: float = 0.0,
) -> NextLatLossResult:
    """Compute the NextLat prototype loss."""

    if not config.enabled:
        zero = hidden_states.new_zeros(())
        return NextLatLossResult(
            next_h_loss=zero,
            transition_count=zero.detach(),
            persistence_loss=zero.detach(),
            persistence_transition_count=zero.detach(),
            kl_loss=zero,
            kl_transition_count=zero.detach(),
            kl_weight=zero.detach(),
        )
    kl_weight_tensor = hidden_states.new_tensor(float(max(0.0, kl_weight)))
    compute_kl = config.lambda_kl > 0.0 and kl_weight_tensor.item() > 0.0
    if compute_kl and output_embeddings is None:
        raise ValueError("NextLat KL requires output_embeddings.")

    base_mask = build_transition_mask(
        input_ids=input_ids,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        mask_eos_bos_boundaries=config.mask_eos_bos_boundaries,
    )
    if base_mask.numel() == 0:
        zero = hidden_states.new_zeros(())
        return NextLatLossResult(
            next_h_loss=zero,
            transition_count=zero.detach(),
            persistence_loss=zero.detach(),
            persistence_transition_count=zero.detach(),
            kl_loss=zero,
            kl_transition_count=zero.detach(),
            kl_weight=kl_weight_tensor.detach(),
        )

    # Diagnostic only: compare the learned dynamics against a trivial
    # one-step persistence baseline that predicts h_t as h_{t+1}.
    persistence_loss_sum, persistence_transition_count = _masked_smooth_l1(
        hidden_states[:, :-1].detach(),
        hidden_states[:, 1:].detach(),
        base_mask,
    )

    current_states = hidden_states
    cumulative_mask = base_mask
    loss_sum = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
    transition_count = torch.zeros(
        (),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    kl_loss_sum = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
    kl_transition_count = torch.zeros(
        (),
        dtype=torch.float32,
        device=hidden_states.device,
    )

    for step in range(config.horizon):
        if current_states.size(1) < 2:
            break
        if step > 0:
            cumulative_mask = cumulative_mask[:, :-1] & base_mask[:, step:]
        if cumulative_mask.numel() == 0:
            break

        current_states = current_states[:, :-1]
        next_tokens = input_ids[:, step + 1 :]
        target_states = hidden_states[:, step + 1 :]
        if config.detach_targets:
            target_states = target_states.detach()
        next_token_embeddings = input_embeddings(next_tokens)
        pred_states = dynamics(current_states, next_token_embeddings)
        step_loss_sum, step_transition_count = _masked_smooth_l1(
            pred_states,
            target_states,
            cumulative_mask,
        )
        loss_sum = loss_sum + step_loss_sum
        transition_count = transition_count + step_transition_count
        if compute_kl:
            step_kl_loss_sum, step_kl_transition_count = _masked_topk_kl(
                pred=pred_states,
                target=target_states,
                mask=cumulative_mask,
                output_embeddings=output_embeddings,
                top_k=config.kl_top_k,
                transition_chunk_size=config.kl_transition_chunk_size,
                detach_lm_head=config.detach_kl_lm_head,
            )
            kl_loss_sum = kl_loss_sum + step_kl_loss_sum
            kl_transition_count = kl_transition_count + step_kl_transition_count
        current_states = pred_states

    next_h_loss = loss_sum / transition_count.clamp_min(1.0)
    persistence_loss = persistence_loss_sum / persistence_transition_count.clamp_min(1.0)
    kl_loss = kl_loss_sum / kl_transition_count.clamp_min(1.0)
    return NextLatLossResult(
        next_h_loss=next_h_loss,
        transition_count=transition_count.detach(),
        persistence_loss=persistence_loss.detach(),
        persistence_transition_count=persistence_transition_count.detach(),
        kl_loss=kl_loss,
        kl_transition_count=kl_transition_count.detach(),
        kl_weight=kl_weight_tensor.detach(),
    )


__all__ = [
    "LatentDynamicsMLP",
    "NextLatConfig",
    "NextLatLossResult",
    "build_transition_mask",
    "compute_nextlat_loss",
    "nextlat_kl_weight",
    "nextlat_config_from_dict",
]
