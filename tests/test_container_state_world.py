from __future__ import annotations

import pytest

from babysteps.container_state.world import (
    Operation,
    World,
    apply_operation,
    apply_operations,
)


def make_world() -> World:
    return World.from_mapping(
        {
            "Cedar drawer": ["marble", "spoon"],
            "Maple basket": ["ribbon"],
            "Willow locker": ["button"],
        }
    )


def test_add_remove_and_move_update_immutable_world() -> None:
    initial = make_world()
    operations = (
        Operation(kind="add", item="pencil", destination="Willow locker"),
        Operation(kind="move", item="spoon", source="Cedar drawer", destination="Maple basket"),
        Operation(kind="remove", item="button", source="Willow locker"),
    )

    final = apply_operations(initial, operations)

    assert initial.items_in("Cedar drawer") == ("marble", "spoon")
    assert final.items_in("Cedar drawer") == ("marble",)
    assert final.items_in("Maple basket") == ("ribbon", "spoon")
    assert final.items_in("Willow locker") == ("pencil",)
    assert final.locate("button") is None


def test_invalid_operations_are_rejected() -> None:
    world = make_world()

    with pytest.raises(ValueError, match="already in"):
        apply_operation(
            world,
            Operation(kind="add", item="marble", destination="Willow locker"),
        )
    with pytest.raises(ValueError, match="item is absent"):
        apply_operation(
            world,
            Operation(kind="remove", item="ribbon", source="Cedar drawer"),
        )
    with pytest.raises(ValueError, match="must differ"):
        Operation(
            kind="move",
            item="marble",
            source="Cedar drawer",
            destination="Cedar drawer",
        )


def test_world_rejects_duplicate_object_locations() -> None:
    with pytest.raises(ValueError, match="appears in both"):
        World.from_mapping(
            {
                "Cedar drawer": ["marble"],
                "Maple basket": ["marble"],
            }
        )
