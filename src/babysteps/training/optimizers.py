"""Optimizer builders for causal language-model training."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import torch
from torch.optim import AdamW

from babysteps.optim import MuonWithAuxAdamPE, build_muon_pe_param_groups


@dataclass(frozen=True)
class OptimizerBuildInfo:
    """Small metadata bundle for logging optimizer construction."""

    name: str
    primary_tensors: int
    auxiliary_tensors: int
    qkv_split_tensors: int = 0
    use_fused: bool = False


class Lamb(torch.optim.Optimizer):
    """Layer-wise adaptive moments optimizer used by the GPT-BERT baseline."""

    def __init__(
        self,
        params,
        lr: float = 7e-3,
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps, weight_decay, lr = group["eps"], group["weight_decay"], group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                m, v = state["m"], state["v"]
                state["step"] += 1
                step = state["step"]
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                m_hat = m / (1.0 - beta1**step)
                v_hat = v / (1.0 - beta2**step)
                update = m_hat / (v_hat.sqrt() + eps)
                if weight_decay != 0:
                    update = update.add(param, alpha=weight_decay)

                weight_norm = param.norm()
                update_norm = update.norm()
                if weight_norm > 0 and update_norm > 0:
                    trust_ratio = weight_norm / update_norm
                else:
                    trust_ratio = torch.ones((), device=param.device)
                param.add_(update, alpha=-lr * float(trust_ratio))
        return loss


def _as_betas(value: Any) -> tuple[float, float]:
    beta1, beta2 = value
    return float(beta1), float(beta2)


def _optimizer_settings(training_config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    optimizer_spec = training_config.get("optimizer", "adamw")
    if isinstance(optimizer_spec, dict):
        settings = {**training_config, **optimizer_spec}
        name = str(settings.get("name", "adamw")).strip().lower()
    else:
        settings = training_config
        name = str(optimizer_spec).strip().lower()
    return name, settings


def _trainable_named_parameters(
    model: torch.nn.Module,
    extra_modules: tuple[torch.nn.Module, ...] = (),
) -> dict[str, torch.nn.Parameter]:
    params = {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    for module_idx, module in enumerate(extra_modules):
        for name, param in module.named_parameters():
            if param.requires_grad:
                params[f"extra_{module_idx}.{name}"] = param
    return params


def _trainable_extra_parameters(
    extra_modules: tuple[torch.nn.Module, ...] = (),
) -> list[torch.nn.Parameter]:
    return [
        param
        for module in extra_modules
        for param in module.parameters()
        if param.requires_grad
    ]


def build_adamw_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    device: torch.device | None = None,
    use_fused_adamw: bool = False,
    extra_modules: tuple[torch.nn.Module, ...] = (),
) -> tuple[AdamW, OptimizerBuildInfo]:
    """Build the AdamW baseline with decay/no-decay parameter groups."""

    param_dict = _trainable_named_parameters(model, extra_modules)
    decay_params = [param for param in param_dict.values() if param.dim() >= 2]
    nodecay_params = [param for param in param_dict.values() if param.dim() < 2]

    param_groups: list[dict[str, Any]] = []
    if decay_params:
        param_groups.append(
            {
                "params": decay_params,
                "lr": float(learning_rate),
                "base_lr": float(learning_rate),
                "weight_decay": float(weight_decay),
            }
        )
    if nodecay_params:
        param_groups.append(
            {
                "params": nodecay_params,
                "lr": float(learning_rate),
                "base_lr": float(learning_rate),
                "weight_decay": 0.0,
            }
        )

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = (
        bool(use_fused_adamw)
        and fused_available
        and device is not None
        and device.type == "cuda"
    )
    optimizer_kwargs = {"fused": use_fused} if fused_available else {}

    optimizer = AdamW(
        param_groups,
        lr=float(learning_rate),
        betas=betas,
        eps=float(eps),
        **optimizer_kwargs,
    )
    info = OptimizerBuildInfo(
        name="adamw",
        primary_tensors=len(decay_params),
        auxiliary_tensors=len(nodecay_params),
        use_fused=use_fused,
    )
    return optimizer, info


def build_lamb_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    extra_modules: tuple[torch.nn.Module, ...] = (),
) -> tuple[Lamb, OptimizerBuildInfo]:
    """Build LAMB with decayed matrix/tensor params and no-decay vectors."""

    param_dict = _trainable_named_parameters(model, extra_modules)
    decay_params = [param for param in param_dict.values() if param.dim() >= 2]
    nodecay_params = [param for param in param_dict.values() if param.dim() < 2]

    param_groups: list[dict[str, Any]] = []
    if decay_params:
        param_groups.append(
            {
                "params": decay_params,
                "lr": float(learning_rate),
                "base_lr": float(learning_rate),
                "weight_decay": float(weight_decay),
            }
        )
    if nodecay_params:
        param_groups.append(
            {
                "params": nodecay_params,
                "lr": float(learning_rate),
                "base_lr": float(learning_rate),
                "weight_decay": 0.0,
            }
        )

    optimizer = Lamb(
        param_groups,
        lr=float(learning_rate),
        betas=betas,
        eps=float(eps),
    )
    info = OptimizerBuildInfo(
        name="lamb",
        primary_tensors=len(decay_params),
        auxiliary_tensors=len(nodecay_params),
        use_fused=False,
    )
    return optimizer, info


def build_muon_pe_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    muon_lr: float,
    muon_weight_decay: float,
    muon_momentum: float,
    muon_ns_steps: int,
    muon_nesterov: bool,
    muon_split_qkv: bool,
    muon_batch_updates: bool,
    extra_modules: tuple[torch.nn.Module, ...] = (),
) -> tuple[MuonWithAuxAdamPE, OptimizerBuildInfo]:
    """Build Muon for hidden matrices plus auxiliary AdamW groups."""

    groups = build_muon_pe_param_groups(
        model,
        aux_lr=float(learning_rate),
        aux_weight_decay=float(weight_decay),
        aux_betas=betas,
        aux_eps=float(eps),
        muon_lr=float(muon_lr),
        muon_weight_decay=float(muon_weight_decay),
        muon_momentum=float(muon_momentum),
        muon_ns_steps=int(muon_ns_steps),
        muon_nesterov=bool(muon_nesterov),
        split_qkv=bool(muon_split_qkv),
    )
    extra_params = _trainable_extra_parameters(extra_modules)
    extra_decay_params = [param for param in extra_params if param.dim() >= 2]
    extra_nodecay_params = [param for param in extra_params if param.dim() < 2]
    if extra_decay_params:
        groups.param_groups.append(
            {
                "params": extra_decay_params,
                "use_muon": False,
                "lr": float(learning_rate),
                "base_lr": float(learning_rate),
                "betas": betas,
                "eps": float(eps),
                "weight_decay": float(weight_decay),
            }
        )
    if extra_nodecay_params:
        groups.param_groups.append(
            {
                "params": extra_nodecay_params,
                "use_muon": False,
                "lr": float(learning_rate),
                "base_lr": float(learning_rate),
                "betas": betas,
                "eps": float(eps),
                "weight_decay": 0.0,
            }
        )
    optimizer = MuonWithAuxAdamPE(
        groups.param_groups,
        qkv_split_dims=groups.qkv_split_dims,
        batch_muon_updates=bool(muon_batch_updates),
    )
    info = OptimizerBuildInfo(
        name="muon_pe",
        primary_tensors=groups.muon_tensors,
        auxiliary_tensors=groups.aux_tensors + len(extra_params),
        qkv_split_tensors=groups.qkv_split_tensors,
        use_fused=False,
    )
    return optimizer, info


def build_optimizer(
    model: torch.nn.Module,
    training_config: dict[str, Any],
    *,
    device: torch.device | None = None,
    extra_modules: tuple[torch.nn.Module, ...] = (),
) -> tuple[torch.optim.Optimizer, OptimizerBuildInfo]:
    """Build the optimizer selected by a training config."""

    optimizer_name, settings = _optimizer_settings(training_config)
    learning_rate = float(settings.get("learning_rate", 5e-5))
    weight_decay = float(settings.get("weight_decay", 0.0))
    betas = _as_betas(settings.get("betas", [0.9, 0.999]))
    eps = float(settings.get("eps", 1e-8))

    if optimizer_name == "adamw":
        return build_adamw_optimizer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            device=device,
            use_fused_adamw=bool(settings.get("use_fused_adamw", False)),
            extra_modules=extra_modules,
        )

    if optimizer_name == "lamb":
        return build_lamb_optimizer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            extra_modules=extra_modules,
        )

    if optimizer_name == "muon_pe":
        muon_lr = float(settings.get("muon_lr", 0.02))
        return build_muon_pe_optimizer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            muon_lr=muon_lr,
            muon_weight_decay=float(settings.get("muon_weight_decay", weight_decay)),
            muon_momentum=float(settings.get("muon_momentum", 0.95)),
            muon_ns_steps=int(settings.get("muon_ns_steps", 5)),
            muon_nesterov=bool(settings.get("muon_nesterov", True)),
            muon_split_qkv=bool(settings.get("muon_split_qkv", True)),
            muon_batch_updates=bool(settings.get("muon_batch_updates", True)),
            extra_modules=extra_modules,
        )

    raise ValueError(
        f"Unsupported optimizer={optimizer_name!r}. Expected 'adamw', 'lamb', or 'muon_pe'."
    )
