"""Small symbolic worlds for container-state curriculum examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence


OperationKind = Literal["add", "remove", "move"]


def _clean_name(value: str, *, field: str) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError(f"{field} must not be empty.")
    return name


@dataclass(frozen=True)
class World:
    """Ordered mapping from uniquely named containers to unique objects."""

    contents: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        if not self.contents:
            raise ValueError("A world must contain at least one container.")

        container_names: set[str] = set()
        object_locations: dict[str, str] = {}
        for raw_container, raw_items in self.contents:
            container = _clean_name(raw_container, field="container name")
            if container in container_names:
                raise ValueError(f"Duplicate container name: {container!r}.")
            container_names.add(container)

            seen_here: set[str] = set()
            for raw_item in raw_items:
                item = _clean_name(raw_item, field="item name")
                if item in seen_here:
                    raise ValueError(
                        f"Duplicate item {item!r} inside container {container!r}."
                    )
                seen_here.add(item)
                if item in object_locations:
                    raise ValueError(
                        f"Item {item!r} appears in both {object_locations[item]!r} "
                        f"and {container!r}."
                    )
                object_locations[item] = container

    @classmethod
    def from_mapping(cls, value: Mapping[str, Sequence[str]]) -> "World":
        """Construct a world while preserving mapping and item order."""

        return cls(
            tuple(
                (str(container), tuple(str(item) for item in items))
                for container, items in value.items()
            )
        )

    @property
    def containers(self) -> tuple[str, ...]:
        return tuple(container for container, _ in self.contents)

    @property
    def objects(self) -> tuple[str, ...]:
        return tuple(item for _, items in self.contents for item in items)

    def as_dict(self) -> dict[str, list[str]]:
        return {container: list(items) for container, items in self.contents}

    def items_in(self, container: str) -> tuple[str, ...]:
        for candidate, items in self.contents:
            if candidate == container:
                return items
        raise KeyError(f"Unknown container: {container!r}.")

    def locate(self, item: str) -> str | None:
        for container, items in self.contents:
            if item in items:
                return container
        return None

    def signature(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Return a hashable semantic signature for split-leakage checks."""

        return tuple(
            (container, tuple(sorted(items)))
            for container, items in sorted(self.contents)
        )


@dataclass(frozen=True)
class Operation:
    """One valid state-changing operation."""

    kind: OperationKind
    item: str
    source: str | None = None
    destination: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"add", "remove", "move"}:
            raise ValueError(f"Unsupported operation kind: {self.kind!r}.")
        _clean_name(self.item, field="operation item")
        if self.source is not None:
            _clean_name(self.source, field="operation source")
        if self.destination is not None:
            _clean_name(self.destination, field="operation destination")

        if self.kind == "add":
            if self.source is not None or self.destination is None:
                raise ValueError("add requires only a destination.")
        elif self.kind == "remove":
            if self.source is None or self.destination is not None:
                raise ValueError("remove requires only a source.")
        elif self.kind == "move":
            if self.source is None or self.destination is None:
                raise ValueError("move requires a source and destination.")
            if self.source == self.destination:
                raise ValueError("move source and destination must differ.")

    @property
    def touched_containers(self) -> tuple[str, ...]:
        touched: list[str] = []
        for container in (self.source, self.destination):
            if container is not None and container not in touched:
                touched.append(container)
        return tuple(touched)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "kind": self.kind,
            "item": self.item,
            "source": self.source,
            "destination": self.destination,
        }


def apply_operation(world: World, operation: Operation) -> World:
    """Apply one checked operation and return a new immutable world."""

    state = world.as_dict()
    if operation.kind == "add":
        assert operation.destination is not None
        if operation.destination not in state:
            raise ValueError(f"Unknown destination: {operation.destination!r}.")
        location = world.locate(operation.item)
        if location is not None:
            raise ValueError(
                f"Cannot add {operation.item!r}; it is already in {location!r}."
            )
        state[operation.destination].append(operation.item)

    elif operation.kind == "remove":
        assert operation.source is not None
        if operation.source not in state:
            raise ValueError(f"Unknown source: {operation.source!r}.")
        if operation.item not in state[operation.source]:
            raise ValueError(
                f"Cannot remove {operation.item!r} from {operation.source!r}; "
                "the item is absent."
            )
        state[operation.source].remove(operation.item)

    elif operation.kind == "move":
        assert operation.source is not None
        assert operation.destination is not None
        if operation.source not in state:
            raise ValueError(f"Unknown source: {operation.source!r}.")
        if operation.destination not in state:
            raise ValueError(f"Unknown destination: {operation.destination!r}.")
        if operation.item not in state[operation.source]:
            raise ValueError(
                f"Cannot move {operation.item!r} from {operation.source!r}; "
                "the item is absent."
            )
        state[operation.source].remove(operation.item)
        state[operation.destination].append(operation.item)

    return World.from_mapping(state)


def apply_operations(world: World, operations: Iterable[Operation]) -> World:
    """Apply operations in discourse order."""

    current = world
    for operation in operations:
        current = apply_operation(current, operation)
    return current


__all__ = [
    "Operation",
    "OperationKind",
    "World",
    "apply_operation",
    "apply_operations",
]
