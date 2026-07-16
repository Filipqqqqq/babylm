from __future__ import annotations

import torch
from torch import nn

from babysteps.training.tail_averaging import TailAverager


class _TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)
        self.tied_projection = self.projection


def test_tail_averager_samples_requested_tail_and_preserves_aliases() -> None:
    model = _TiedModel()
    averager = TailAverager(
        model,
        total_steps=10,
        fraction=0.2,
        every_steps=1,
    )

    assert averager.start_step == 9
    with torch.no_grad():
        model.projection.weight.fill_(2.0)
    assert averager.update(model, step=8) is False
    assert averager.update(model, step=9) is True

    with torch.no_grad():
        model.projection.weight.fill_(4.0)
    assert averager.update(model, step=10) is True

    state_dict = averager.averaged_state_dict(model)
    expected = torch.full_like(model.projection.weight, 3.0)
    torch.testing.assert_close(state_dict["projection.weight"], expected)
    torch.testing.assert_close(state_dict["tied_projection.weight"], expected)
    assert averager.num_snapshots == 2


def test_tail_averager_always_samples_final_step() -> None:
    model = nn.Linear(2, 2)
    averager = TailAverager(
        model,
        total_steps=11,
        fraction=0.2,
        every_steps=10,
    )

    assert averager.should_update(10) is True
    assert averager.should_update(11) is True
