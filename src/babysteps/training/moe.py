"""Router losses and load statistics for sparse Mixture-of-Experts models.

The BabySteps causal loop computes its own cross-entropy from logits, so the
load-balancing loss that Hugging Face MoE models only add when *they* compute
the loss would otherwise be silently dropped. These helpers add it back,
together with a router z-loss, and report how evenly tokens are routed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MoeSettings:
    num_experts: int
    top_k: int
    aux_loss_coef: float
    z_loss_coef: float


@dataclass(frozen=True)
class MoeRouterLosses:
    aux_loss: torch.Tensor
    """Switch-style load-balancing loss; equals ``top_k`` when perfectly balanced."""
    z_loss: torch.Tensor
    """Mean squared log-sum-exp of the router logits."""
    weighted_loss: torch.Tensor
    """``aux_loss_coef * aux_loss + z_loss_coef * z_loss``, added to the LM loss."""
    max_load_ratio: torch.Tensor
    """Busiest expert's share of routed tokens divided by the even share (1.0 = balanced)."""


def moe_settings_from_model(model: torch.nn.Module) -> MoeSettings | None:
    """Return MoE settings for a sparse model, or ``None`` for dense models."""

    config = getattr(model, "config", None)
    num_experts = getattr(config, "num_experts", None)
    top_k = getattr(config, "num_experts_per_tok", None)
    if config is None or not num_experts or not top_k:
        return None
    if not getattr(config, "output_router_logits", False):
        raise ValueError(
            "MoE models must be built with output_router_logits=True so the "
            "training loop can apply the router losses."
        )
    return MoeSettings(
        num_experts=int(num_experts),
        top_k=int(top_k),
        aux_loss_coef=float(getattr(config, "router_aux_loss_coef", 0.0)),
        z_loss_coef=float(getattr(config, "babysteps_router_z_loss_coef", 0.0)),
    )


def _flatten_router_logits(router_logits: Any) -> torch.Tensor:
    if router_logits is None:
        raise RuntimeError(
            "The MoE model returned no router logits; expected output_router_logits=True."
        )
    if isinstance(router_logits, torch.Tensor):
        layers = [router_logits]
    else:
        layers = [layer for layer in router_logits if layer is not None]
    if not layers:
        raise RuntimeError("The MoE model returned an empty router_logits tuple.")
    return torch.cat([layer.reshape(-1, layer.size(-1)).float() for layer in layers], dim=0)


def _token_keep_mask(
    attention_mask: torch.Tensor | None,
    *,
    num_layers: int,
    num_rows: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    keep = attention_mask.reshape(-1).to(torch.bool)
    if keep.numel() * num_layers != num_rows:
        return None
    if bool(keep.all()):
        return None
    return keep.repeat(num_layers)


def compute_moe_router_losses(
    outputs: Any,
    settings: MoeSettings,
    *,
    attention_mask: torch.Tensor | None = None,
) -> MoeRouterLosses:
    """Compute load-balancing loss, z-loss, and expert load for one forward pass."""

    router_logits = getattr(outputs, "router_logits", None)
    num_layers = 1 if isinstance(router_logits, torch.Tensor) else len(router_logits or ())
    logits = _flatten_router_logits(router_logits)
    keep = _token_keep_mask(attention_mask, num_layers=max(1, num_layers), num_rows=logits.size(0))
    if keep is not None:
        logits = logits[keep]

    probs = F.softmax(logits, dim=-1)
    _, selected = torch.topk(probs, settings.top_k, dim=-1)
    # (tokens, top_k, experts) -> fraction of tokens sent to each expert per slot.
    expert_mask = F.one_hot(selected, num_classes=settings.num_experts).float()
    tokens_per_expert = expert_mask.mean(dim=0)
    router_prob_per_expert = probs.mean(dim=0)
    aux_loss = settings.num_experts * torch.sum(
        tokens_per_expert * router_prob_per_expert.unsqueeze(0)
    )
    z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
    weighted = settings.aux_loss_coef * aux_loss + settings.z_loss_coef * z_loss

    with torch.no_grad():
        load = expert_mask.sum(dim=(0, 1))
        share = load / load.sum().clamp_min(1.0)
        max_load_ratio = share.max() * settings.num_experts

    return MoeRouterLosses(
        aux_loss=aux_loss,
        z_loss=z_loss,
        weighted_loss=weighted,
        max_load_ratio=max_load_ratio,
    )
