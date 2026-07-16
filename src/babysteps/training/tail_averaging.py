"""Tail weight averaging for experimental training recipes."""

from __future__ import annotations

import math

import torch


class TailAverager:
    """Average model parameters over a configurable tail of optimizer steps."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        total_steps: int,
        fraction: float,
        every_steps: int = 10,
    ) -> None:
        if int(total_steps) <= 0:
            raise ValueError("total_steps must be positive.")
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("fraction must be in (0, 1].")
        if int(every_steps) <= 0:
            raise ValueError("every_steps must be positive.")

        self.total_steps = int(total_steps)
        self.fraction = float(fraction)
        self.every_steps = int(every_steps)
        self.start_step = max(
            1,
            math.floor((1.0 - self.fraction) * self.total_steps) + 1,
        )
        self.num_snapshots = 0
        self._averages: dict[int, torch.Tensor] = {}
        self._parameter_names = self._parameter_names_by_id(model)

    @staticmethod
    def _parameter_names_by_id(
        model: torch.nn.Module,
    ) -> dict[int, tuple[str, ...]]:
        names: dict[int, list[str]] = {}
        for name, param in model.named_parameters(remove_duplicate=False):
            names.setdefault(id(param), []).append(name)
        return {param_id: tuple(param_names) for param_id, param_names in names.items()}

    def should_update(self, step: int) -> bool:
        """Return whether a one-indexed optimizer step should be sampled."""

        step = int(step)
        if step < self.start_step or step > self.total_steps:
            return False
        return step % self.every_steps == 0 or step == self.total_steps

    @torch.no_grad()
    def update(self, model: torch.nn.Module, *, step: int) -> bool:
        """Sample model parameters when ``step`` is in the averaging schedule."""

        if not self.should_update(step):
            return False

        self.num_snapshots += 1
        seen: set[int] = set()
        for param in model.parameters():
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            value = param.detach().float().cpu()
            average = self._averages.get(param_id)
            if average is None:
                self._averages[param_id] = value.clone()
            else:
                average.add_((value - average) / self.num_snapshots)
        return True

    def averaged_state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Return a state dict with averaged parameters and current buffers."""

        if self.num_snapshots == 0:
            raise RuntimeError("TailAverager has not collected any snapshots.")

        state_dict = model.state_dict()
        for param_id, average in self._averages.items():
            for name in self._parameter_names.get(param_id, ()):
                current = state_dict[name]
                state_dict[name] = average.to(dtype=current.dtype)
        return state_dict


__all__ = ["TailAverager"]
