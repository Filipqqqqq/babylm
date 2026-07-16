"""Diagnostic candidate construction for container-state validation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from babysteps.container_state.world import Operation, World


@dataclass(frozen=True)
class DistractorState:
    """One incorrect target-container state and the shortcut it represents."""

    kind: str
    items: tuple[str, ...]


def _without(items: Sequence[str], item: str) -> tuple[str, ...]:
    return tuple(candidate for candidate in items if candidate != item)


def _append_unique(items: Sequence[str], item: str) -> tuple[str, ...]:
    values = list(items)
    if item not in values:
        values.append(item)
    return tuple(values)


def build_distractor_states(
    *,
    initial_world: World,
    final_world: World,
    operations: Sequence[Operation],
    target_container: str,
    object_pool: Sequence[str],
    rng: random.Random,
    count: int = 4,
) -> tuple[DistractorState, ...]:
    """Build unique, interpretable wrong states for one validation item."""

    if count <= 0:
        raise ValueError("count must be positive.")

    correct = final_world.items_in(target_container)
    correct_key = frozenset(correct)
    candidates: list[DistractorState] = []
    seen = {correct_key}

    def add(kind: str, items: Sequence[str]) -> None:
        values = tuple(dict.fromkeys(str(item) for item in items))
        key = frozenset(values)
        if key in seen:
            return
        seen.add(key)
        candidates.append(DistractorState(kind=kind, items=values))

    initial = initial_world.items_in(target_container)
    add("initial_stale_state", initial)

    for operation in reversed(operations):
        if operation.source == target_container:
            add("ignored_outbound_change", _append_unique(correct, operation.item))
        if operation.destination == target_container:
            add("ignored_inbound_change", _without(correct, operation.item))

    other_containers = [
        container
        for container in final_world.containers
        if container != target_container
    ]
    rng.shuffle(other_containers)
    for container in other_containers:
        add("other_container_state", final_world.items_in(container))

    replacement_pool = list(
        dict.fromkeys(
            [
                *final_world.objects,
                *initial_world.objects,
                *(str(item) for item in object_pool),
            ]
        )
    )
    rng.shuffle(replacement_pool)

    if correct:
        for replace_index in range(len(correct)):
            for replacement in replacement_pool:
                if replacement in correct:
                    continue
                changed = list(correct)
                changed[replace_index] = replacement
                add("same_size_object_swap", changed)
                if len(candidates) >= count + 4:
                    break
            if len(candidates) >= count + 4:
                break

        for item in correct:
            add("missing_final_item", _without(correct, item))
    else:
        for replacement in replacement_pool:
            add("false_nonempty_state", (replacement,))
            if len(candidates) >= count + 4:
                break

    for replacement in replacement_pool:
        if replacement not in correct:
            add("extra_item", _append_unique(correct, replacement))
        if len(candidates) >= count + 4:
            break

    if len(candidates) < count:
        raise RuntimeError(
            f"Could only construct {len(candidates)} unique distractors; "
            f"needed {count}."
        )

    # Avoid a fixed relationship between the first distractor and its mechanism.
    rng.shuffle(candidates)
    return tuple(candidates[:count])


__all__ = ["DistractorState", "build_distractor_states"]
