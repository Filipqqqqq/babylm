"""Tiny symbolic spatial world for synthetic relation examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Position = tuple[int, int, int]
WorldRelation = Literal["east", "west", "north", "south", "above", "below", "same"]
HorizontalRelation = Literal["east", "west", "north", "south"]
RelativeRelation = Literal["front", "behind", "left", "right", "above", "below", "same"]
Facing = Literal["north", "east", "south", "west"]
Turn = Literal["left", "right", "around"]

DIRS: dict[str, Position] = {
    "east": (1, 0, 0),
    "west": (-1, 0, 0),
    "north": (0, 1, 0),
    "south": (0, -1, 0),
    "above": (0, 0, 1),
    "below": (0, 0, -1),
}

HORIZONTAL: tuple[Facing, ...] = ("north", "east", "south", "west")

OPPOSITE: dict[str, str] = {
    "east": "west",
    "west": "east",
    "north": "south",
    "south": "north",
    "above": "below",
    "below": "above",
    "front": "behind",
    "behind": "front",
    "left": "right",
    "right": "left",
    "close": "far",
    "far": "close",
}


@dataclass(frozen=True)
class State:
    """Two-position spatial scene with optional facing for each entity."""

    a_pos: Position
    b_pos: Position
    a_facing: Facing = "north"
    b_facing: Facing = "north"


@dataclass(frozen=True)
class Relations:
    """Absolute and observer-relative relation for one figure-ground pair."""

    world: WorldRelation
    relative: RelativeRelation


def add(u: Position, v: Position) -> Position:
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2])


def sub(u: Position, v: Position) -> Position:
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def neg(u: Position) -> Position:
    return (-u[0], -u[1], -u[2])


def _validate_facing(facing: str) -> Facing:
    if facing not in HORIZONTAL:
        raise ValueError(f"Unsupported facing {facing!r}; expected one of {HORIZONTAL}.")
    return facing  # type: ignore[return-value]


def turn_facing(facing: Facing, turn: Turn) -> Facing:
    """Return the new facing after a left/right/around turn."""

    facing = _validate_facing(facing)
    index = HORIZONTAL.index(facing)
    if turn == "left":
        return HORIZONTAL[(index - 1) % len(HORIZONTAL)]
    if turn == "right":
        return HORIZONTAL[(index + 1) % len(HORIZONTAL)]
    if turn == "around":
        return HORIZONTAL[(index + 2) % len(HORIZONTAL)]
    raise ValueError("turn must be 'left', 'right', or 'around'.")


def world_relation(delta: Position) -> WorldRelation:
    """Return the axis-aligned absolute relation described by ``delta``."""

    x, y, z = delta
    if delta == (0, 0, 0):
        return "same"
    if y == 0 and z == 0:
        return "east" if x > 0 else "west"
    if x == 0 and z == 0:
        return "north" if y > 0 else "south"
    if x == 0 and y == 0:
        return "above" if z > 0 else "below"
    raise ValueError(f"Only axis-aligned deltas are supported, got {delta!r}.")


def relative_relation(delta: Position, facing: Facing) -> RelativeRelation:
    """Return the relation from an observer with the given facing."""

    facing = _validate_facing(facing)
    relation = world_relation(delta)
    if relation in {"above", "below", "same"}:
        return relation

    unit = DIRS[relation]
    front = DIRS[facing]
    behind = neg(front)
    left = DIRS[turn_facing(facing, "left")]
    right = DIRS[turn_facing(facing, "right")]

    if unit == front:
        return "front"
    if unit == behind:
        return "behind"
    if unit == left:
        return "left"
    if unit == right:
        return "right"
    raise ValueError(f"Could not map {relation!r} relative to facing {facing!r}.")


def relative_to_world_relation(relation: str, facing: Facing) -> HorizontalRelation:
    """Map a horizontal egocentric relation to an absolute direction."""

    facing = _validate_facing(facing)
    if relation == "front":
        return facing
    if relation == "behind":
        return OPPOSITE[facing]  # type: ignore[return-value]
    if relation == "left":
        return turn_facing(facing, "left")
    if relation == "right":
        return turn_facing(facing, "right")
    raise ValueError("relation must be front, behind, left, or right.")


def compute_relations(state: State, observer: Literal["a", "b"]) -> Relations:
    """Compute the other entity's relation from observer ``a`` or ``b``."""

    if observer == "a":
        delta = sub(state.b_pos, state.a_pos)
        facing = state.a_facing
    elif observer == "b":
        delta = sub(state.a_pos, state.b_pos)
        facing = state.b_facing
    else:
        raise ValueError("observer must be 'a' or 'b'.")

    return Relations(
        world=world_relation(delta),
        relative=relative_relation(delta, facing),
    )


def position_for_world_relation(relation: str, *, distance: int = 1) -> Position:
    """Return an axis-aligned position offset for an absolute relation."""

    if relation not in DIRS:
        raise ValueError(f"Unsupported world relation {relation!r}.")
    if distance <= 0:
        raise ValueError("distance must be positive.")
    base = DIRS[relation]
    return (base[0] * distance, base[1] * distance, base[2] * distance)


__all__ = [
    "DIRS",
    "HORIZONTAL",
    "OPPOSITE",
    "Facing",
    "HorizontalRelation",
    "Position",
    "RelativeRelation",
    "Relations",
    "State",
    "Turn",
    "WorldRelation",
    "add",
    "compute_relations",
    "neg",
    "position_for_world_relation",
    "relative_relation",
    "relative_to_world_relation",
    "sub",
    "turn_facing",
    "world_relation",
]
