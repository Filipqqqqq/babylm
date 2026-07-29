"""Objective-batching policies for hybrid causal/MNTP training."""

from __future__ import annotations

from dataclasses import dataclass


OBJECTIVE_GRANULARITIES = ("update", "microbatch")
OBJECTIVE_CYCLE_STARTS = ("causal", "masked")


@dataclass(frozen=True)
class ObjectiveMicrobatch:
    """One contiguous sequence slice assigned to a training objective."""

    start: int
    stop: int
    objective: str

    @property
    def size(self) -> int:
        return self.stop - self.start


def normalize_objective_granularity(value: str | None) -> str:
    normalized = str(value or "update").strip().lower().replace("-", "_")
    aliases = {
        "per_update": "update",
        "whole_update": "update",
        "per_microbatch": "microbatch",
        "sid": "microbatch",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in OBJECTIVE_GRANULARITIES:
        raise ValueError(
            "hybrid.objective_granularity must be one of "
            f"{OBJECTIVE_GRANULARITIES}, got {value!r}."
        )
    return normalized


def normalize_objective_cycle_start(value: str | None) -> str:
    normalized = str(value or "causal").strip().lower()
    if normalized not in OBJECTIVE_CYCLE_STARTS:
        raise ValueError(
            "hybrid.objective_cycle_start must be 'causal' or 'masked', "
            f"got {value!r}."
        )
    return normalized


def objective_for_step(
    step: int,
    *,
    causal_steps: int,
    masked_steps: int,
    cycle_start: str = "causal",
) -> str:
    """Return the objective at one position in a deterministic cycle."""

    causal_steps = int(causal_steps)
    masked_steps = int(masked_steps)
    period = causal_steps + masked_steps
    if causal_steps < 0 or masked_steps < 0 or period <= 0:
        raise ValueError("causal_steps and masked_steps must be non-negative with a positive sum.")
    cycle_start = normalize_objective_cycle_start(cycle_start)
    phase = int(step) % period
    if cycle_start == "causal":
        return "causal" if phase < causal_steps else "masked"
    return "masked" if phase < masked_steps else "causal"


def plan_objective_microbatches(
    *,
    batch_size: int,
    cursor: int,
    causal_steps: int,
    masked_steps: int,
    granularity: str = "update",
    micro_batch_size: int | None = None,
    cycle_start: str = "causal",
) -> tuple[ObjectiveMicrobatch, ...]:
    """Plan objective-homogeneous slices for one loader batch."""

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    granularity = normalize_objective_granularity(granularity)
    if granularity == "update":
        return (
            ObjectiveMicrobatch(
                0,
                batch_size,
                objective_for_step(
                    cursor,
                    causal_steps=causal_steps,
                    masked_steps=masked_steps,
                    cycle_start=cycle_start,
                ),
            ),
        )

    if micro_batch_size is None or int(micro_batch_size) <= 0:
        raise ValueError(
            "hybrid.objective_micro_batch_size must be positive for microbatch granularity."
        )
    micro_batch_size = int(micro_batch_size)
    if batch_size % micro_batch_size:
        raise ValueError(
            f"Sequence batch size {batch_size} is not divisible by "
            f"objective_micro_batch_size={micro_batch_size}."
        )

    planned = []
    for index, start in enumerate(range(0, batch_size, micro_batch_size)):
        planned.append(
            ObjectiveMicrobatch(
                start,
                start + micro_batch_size,
                objective_for_step(
                    int(cursor) + index,
                    causal_steps=causal_steps,
                    masked_steps=masked_steps,
                    cycle_start=cycle_start,
                ),
            )
        )
    return tuple(planned)


__all__ = [
    "OBJECTIVE_CYCLE_STARTS",
    "OBJECTIVE_GRANULARITIES",
    "ObjectiveMicrobatch",
    "normalize_objective_cycle_start",
    "normalize_objective_granularity",
    "objective_for_step",
    "plan_objective_microbatches",
]
