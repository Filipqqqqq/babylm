from __future__ import annotations

import pytest

from babysteps.spatial_synth.world import (
    OPPOSITE,
    State,
    compute_relations,
    relative_relation,
    relative_to_world_relation,
    sub,
    turn_facing,
    world_relation,
)


def test_world_relation_maps_axis_aligned_offsets() -> None:
    assert world_relation((1, 0, 0)) == "east"
    assert world_relation((-3, 0, 0)) == "west"
    assert world_relation((0, 2, 0)) == "north"
    assert world_relation((0, -1, 0)) == "south"
    assert world_relation((0, 0, 4)) == "above"
    assert world_relation((0, 0, -2)) == "below"


def test_world_relation_rejects_diagonals() -> None:
    with pytest.raises(ValueError, match="axis-aligned"):
        world_relation((1, 1, 0))


def test_relative_relation_uses_observer_facing() -> None:
    assert relative_relation((1, 0, 0), "north") == "right"
    assert relative_relation((1, 0, 0), "south") == "left"
    assert relative_relation((0, 1, 0), "east") == "left"
    assert relative_relation((0, -1, 0), "west") == "left"


def test_turn_facing_and_relative_to_world_relation() -> None:
    assert turn_facing("north", "left") == "west"
    assert turn_facing("north", "right") == "east"
    assert turn_facing("north", "around") == "south"
    assert relative_to_world_relation("front", "east") == "east"
    assert relative_to_world_relation("left", "east") == "north"
    assert relative_to_world_relation("right", "south") == "west"
    assert relative_to_world_relation("behind", "west") == "east"


def test_compute_relations_tracks_observer_roles() -> None:
    state = State(
        a_pos=(0, 0, 0),
        b_pos=(1, 0, 0),
        a_facing="north",
        b_facing="south",
    )

    from_a = compute_relations(state, "a")
    from_b = compute_relations(state, "b")

    assert from_a.world == "east"
    assert from_a.relative == "right"
    assert from_b.world == "west"
    assert from_b.relative == "right"
    assert from_b.world == OPPOSITE[from_a.world]


def test_reciprocal_world_relations_are_opposites() -> None:
    a = (0, 0, 0)
    b = (0, 0, 1)

    assert world_relation(sub(b, a)) == "above"
    assert world_relation(sub(a, b)) == "below"
