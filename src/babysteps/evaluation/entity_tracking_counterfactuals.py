"""Controlled counterfactual probes for Entity Tracking strategies."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


ProbeFamily = Literal[
    "earlier_removal",
    "earlier_inbound",
    "irrelevant_reordering",
    "whole_box_transfer",
]

PROBE_FAMILIES: tuple[ProbeFamily, ...] = (
    "earlier_removal",
    "earlier_inbound",
    "irrelevant_reordering",
    "whole_box_transfer",
)

# This is the 100-word object inventory used by the regular official Entity
# Tracking split. Keeping the inventory fixed avoids accidental vocabulary drift.
OBJECT_VOCAB = (
    "apple", "bag", "ball", "beer", "bell", "bill", "block", "boat",
    "bomb", "bone", "book", "boot", "bottle", "bowl", "brain", "branch",
    "bread", "brick", "bus", "cake", "camera", "car", "card", "cash",
    "cheese", "chemical", "cigarette", "clock", "coat", "coffee", "computer",
    "cream", "creature", "cross", "crown", "cup", "dish", "disk", "document",
    "dress", "drink", "drug", "egg", "engine", "fan", "fig", "file", "fish",
    "flower", "game", "gift", "glass", "guitar", "hat", "ice", "jacket",
    "key", "knife", "leaf", "letter", "machine", "magazine", "map", "meat",
    "medicine", "milk", "mirror", "newspaper", "note", "painting", "paper",
    "phone", "picture", "pipe", "plane", "plant", "plate", "pot", "radio",
    "ring", "rock", "rose", "seed", "sheet", "shell", "shirt", "shoe",
    "stone", "string", "suit", "tape", "tea", "television", "ticket", "tie",
    "tissue", "train", "watch", "wheel", "wire",
)
OBJECT_ORDER = {word: index for index, word in enumerate(OBJECT_VOCAB)}
WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
CONTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "box",
    "contains",
    "from",
    "into",
    "move",
    "nothing",
    "of",
    "put",
    "remove",
    "the",
    "to",
}


@dataclass(frozen=True)
class Operation:
    """One structured operation rendered with official benchmark templates."""

    kind: Literal["put", "remove", "move", "move_contents"]
    objects: tuple[str, ...] = ()
    source: int | None = None
    destination: int | None = None

    def touched_boxes(self) -> tuple[int, ...]:
        boxes = []
        if self.source is not None:
            boxes.append(self.source)
        if self.destination is not None and self.destination not in boxes:
            boxes.append(self.destination)
        return tuple(boxes)

    def render(self) -> str:
        if self.kind == "put":
            if self.destination is None or not self.objects:
                raise ValueError("put requires objects and a destination.")
            return f"Put {_render_object_phrase(self.objects)} into Box {self.destination}."
        if self.kind == "remove":
            if self.source is None or not self.objects:
                raise ValueError("remove requires objects and a source.")
            return f"Remove {_render_object_phrase(self.objects)} from Box {self.source}."
        if self.kind == "move":
            if self.source is None or self.destination is None or not self.objects:
                raise ValueError("move requires objects, a source, and a destination.")
            return (
                f"Move {_render_object_phrase(self.objects)} from Box {self.source} "
                f"to Box {self.destination}."
            )
        if self.kind == "move_contents":
            if self.source is None or self.destination is None:
                raise ValueError("move_contents requires a source and destination.")
            return f"Move the contents of Box {self.source} to Box {self.destination}."
        raise ValueError(f"Unsupported operation kind: {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "objects": list(self.objects),
            "source": self.source,
            "destination": self.destination,
            "text": self.render(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Operation:
        operation = cls(
            kind=value["kind"],
            objects=tuple(str(item) for item in value.get("objects", [])),
            source=value.get("source"),
            destination=value.get("destination"),
        )
        if value.get("text") not in (None, operation.render()):
            raise ValueError("Serialized operation text does not match its structure.")
        return operation


@dataclass(frozen=True)
class CounterfactualItem:
    """One member of a controlled pair."""

    pair_id: str
    family: ProbeFamily
    condition: str
    query_box: int
    initial_state: tuple[tuple[int, tuple[str, ...]], ...]
    operations: tuple[Operation, ...]
    input_prefix: str
    options: tuple[str, ...]
    target_index: int
    target_contents: tuple[str, ...]
    last_touch: str
    last_touch_distance: int
    expected_answer_relation: Literal["change", "stable"]
    generation_seed: int
    generation_index: int

    @property
    def target(self) -> str:
        return self.options[self.target_index]

    def initial_state_dict(self) -> dict[int, list[str]]:
        return {box: list(objects) for box, objects in self.initial_state}

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "family": self.family,
            "condition": self.condition,
            "query_box": self.query_box,
            "initial_state": {
                str(box): list(objects) for box, objects in self.initial_state
            },
            "operations": [operation.to_dict() for operation in self.operations],
            "input_prefix": self.input_prefix,
            "options": list(self.options),
            "target_index": self.target_index,
            "target": self.target,
            "target_contents": list(self.target_contents),
            "last_touch": self.last_touch,
            "last_touch_distance": self.last_touch_distance,
            "expected_answer_relation": self.expected_answer_relation,
            "generation_seed": self.generation_seed,
            "generation_index": self.generation_index,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CounterfactualItem:
        item = cls(
            pair_id=str(value["pair_id"]),
            family=value["family"],
            condition=str(value["condition"]),
            query_box=int(value["query_box"]),
            initial_state=tuple(
                (int(box), tuple(str(obj) for obj in objects))
                for box, objects in sorted(
                    value["initial_state"].items(), key=lambda pair: int(pair[0])
                )
            ),
            operations=tuple(
                Operation.from_dict(operation) for operation in value["operations"]
            ),
            input_prefix=str(value["input_prefix"]),
            options=tuple(str(option) for option in value["options"]),
            target_index=int(value["target_index"]),
            target_contents=tuple(str(obj) for obj in value["target_contents"]),
            last_touch=str(value["last_touch"]),
            last_touch_distance=int(value["last_touch_distance"]),
            expected_answer_relation=value["expected_answer_relation"],
            generation_seed=int(value["generation_seed"]),
            generation_index=int(value["generation_index"]),
        )
        if value.get("target") not in (None, item.target):
            raise ValueError("Serialized target does not match target_index.")
        validate_item(item)
        return item


def _canonical_objects(objects: Iterable[str]) -> tuple[str, ...]:
    unique = set(objects)
    unknown = unique.difference(OBJECT_ORDER)
    if unknown:
        raise ValueError(f"Unknown objects: {sorted(unknown)}")
    return tuple(sorted(unique, key=OBJECT_ORDER.__getitem__))


def _render_object_phrase(objects: Iterable[str]) -> str:
    values = _canonical_objects(objects)
    if not values:
        raise ValueError("An operation cannot have an empty object phrase.")
    if len(values) == 1:
        return f"the {values[0]}"
    return "the " + " and the ".join(values)


def render_contents(objects: Iterable[str]) -> str:
    values = _canonical_objects(objects)
    if not values:
        return "nothing."
    return _render_object_phrase(values) + "."


def _content_words(text: str) -> tuple[str, ...]:
    return tuple(
        word.lower()
        for word in WORD_RE.findall(text)
        if word.lower() not in CONTENT_STOPWORDS
    )


def lexical_recency_best_indices(item: CounterfactualItem) -> tuple[int, ...]:
    """Return candidates favored by the same mean-last-position heuristic."""

    query_suffix = f"Box {item.query_box} contains "
    if not item.input_prefix.endswith(query_suffix):
        raise ValueError("Probe input does not end with its structured query suffix.")
    context = item.input_prefix[: -len(query_suffix)]
    last_positions: dict[str, int] = {}
    for index, word in enumerate(_content_words(context)):
        last_positions[word] = index

    scores = []
    for option in item.options:
        words = tuple(dict.fromkeys(_content_words(option)))
        scores.append(
            statistics.fmean(last_positions.get(word, -1) for word in words)
        )
    maximum = max(scores)
    return tuple(index for index, score in enumerate(scores) if score == maximum)


def render_initial_state(state: dict[int, list[str]]) -> str:
    if sorted(state) != list(range(7)):
        raise ValueError("Initial states must contain exactly Boxes 0 through 6.")
    clauses = [
        f"Box {box} contains {render_contents(state[box]).removesuffix('.')}"
        for box in range(7)
    ]
    return ", ".join(clauses) + "."


def apply_operation(state: dict[int, list[str]], operation: Operation) -> None:
    """Apply one operation in place, rejecting internally inconsistent probes."""

    if operation.kind == "put":
        assert operation.destination is not None
        existing = {obj for contents in state.values() for obj in contents}
        duplicates = set(operation.objects).intersection(existing)
        if duplicates:
            raise ValueError(f"Cannot put already-present objects: {sorted(duplicates)}")
        state[operation.destination].extend(operation.objects)
        return

    if operation.kind in {"remove", "move"}:
        assert operation.source is not None
        missing = set(operation.objects).difference(state[operation.source])
        if missing:
            raise ValueError(
                f"Box {operation.source} is missing objects: {sorted(missing)}"
            )
        for obj in operation.objects:
            state[operation.source].remove(obj)
        if operation.kind == "move":
            assert operation.destination is not None
            state[operation.destination].extend(operation.objects)
        return

    if operation.kind == "move_contents":
        assert operation.source is not None and operation.destination is not None
        if operation.source == operation.destination:
            raise ValueError("Cannot move a box's contents into itself.")
        state[operation.destination].extend(state[operation.source])
        state[operation.source] = []
        return

    raise ValueError(f"Unsupported operation kind: {operation.kind!r}")


def simulate(
    initial_state: dict[int, list[str]],
    operations: Iterable[Operation],
) -> dict[int, list[str]]:
    state = {box: list(objects) for box, objects in initial_state.items()}
    for operation in operations:
        apply_operation(state, operation)
    return state


def render_input_prefix(
    initial_state: dict[int, list[str]],
    operations: Iterable[Operation],
    query_box: int,
) -> str:
    operation_text = " ".join(operation.render() for operation in operations)
    context = render_initial_state(initial_state)
    if operation_text:
        context += " " + operation_text
    return f"{context} Box {query_box} contains "


def find_last_touch(
    operations: tuple[Operation, ...], query_box: int
) -> tuple[str, int]:
    indices = [
        index
        for index, operation in enumerate(operations)
        if query_box in operation.touched_boxes()
    ]
    if not indices:
        raise ValueError("Every probe must contain an operation touching the query box.")
    index = indices[-1]
    return operations[index].render(), len(operations) - index - 1


def _state_tuple(state: dict[int, list[str]]) -> tuple[tuple[int, tuple[str, ...]], ...]:
    return tuple(
        (box, _canonical_objects(state[box]))
        for box in sorted(state)
    )


def _complete_initial_state(
    specified: dict[int, list[str]],
    object_pool: Iterable[str],
) -> dict[int, list[str]]:
    iterator = iter(object_pool)
    state = {box: list(objects) for box, objects in specified.items()}
    for box in range(7):
        if box not in state:
            state[box] = [next(iterator)]
    return state


def _put_distractors(
    *,
    object_pool: Iterable[str],
    safe_boxes: tuple[int, ...],
    count: int,
    rng: random.Random,
) -> tuple[Operation, ...]:
    iterator = iter(object_pool)
    return tuple(
        Operation(
            kind="put",
            objects=(next(iterator),),
            destination=rng.choice(safe_boxes),
        )
        for _ in range(count)
    )


def _shuffled_options(
    candidate_contents: list[Iterable[str]],
    rng: random.Random,
) -> tuple[str, ...]:
    rendered = [render_contents(contents) for contents in candidate_contents]
    if len(rendered) != 5 or len(set(rendered)) != 5:
        raise ValueError(f"Probe options must contain five unique choices: {rendered}")
    rng.shuffle(rendered)
    return tuple(rendered)


def _build_item(
    *,
    pair_id: str,
    family: ProbeFamily,
    condition: str,
    query_box: int,
    initial_state: dict[int, list[str]],
    operations: tuple[Operation, ...],
    options: tuple[str, ...],
    expected_answer_relation: Literal["change", "stable"],
    generation_seed: int,
    generation_index: int,
) -> CounterfactualItem:
    final_state = simulate(initial_state, operations)
    target_contents = _canonical_objects(final_state[query_box])
    target = render_contents(target_contents)
    if target not in options:
        raise ValueError(f"Correct answer {target!r} is absent from options {options!r}.")
    last_touch, last_touch_distance = find_last_touch(operations, query_box)
    item = CounterfactualItem(
        pair_id=pair_id,
        family=family,
        condition=condition,
        query_box=query_box,
        initial_state=_state_tuple(initial_state),
        operations=operations,
        input_prefix=render_input_prefix(initial_state, operations, query_box),
        options=options,
        target_index=options.index(target),
        target_contents=target_contents,
        last_touch=last_touch,
        last_touch_distance=last_touch_distance,
        expected_answer_relation=expected_answer_relation,
        generation_seed=generation_seed,
        generation_index=generation_index,
    )
    validate_item(item)
    return item


def _pair_earlier_removal(
    *, rng: random.Random, seed: int, index: int
) -> tuple[CounterfactualItem, CounterfactualItem]:
    words = iter(rng.sample(OBJECT_VOCAB, 32))
    anchor, removed_a, removed_b, incoming = (next(words) for _ in range(4))
    query_box = rng.randrange(7)
    initial = _complete_initial_state(
        {query_box: [anchor, removed_a, removed_b]}, words
    )
    background = next(
        obj
        for box, contents in initial.items()
        if box != query_box
        for obj in contents
    )
    safe_boxes = tuple(box for box in range(7) if box != query_box)
    intervening_count = index % 3
    post_touch_count = (index // 3) % 4
    distractors = _put_distractors(
        object_pool=words,
        safe_boxes=safe_boxes,
        count=intervening_count + post_touch_count,
        rng=rng,
    )
    intervening = distractors[:intervening_count]
    post_touch = distractors[intervening_count:]
    last_touch = Operation(kind="put", objects=(incoming,), destination=query_box)
    operations_a = (
        Operation(kind="remove", objects=(removed_a,), source=query_box),
        *intervening,
        last_touch,
        *post_touch,
    )
    operations_b = (
        Operation(kind="remove", objects=(removed_b,), source=query_box),
        *intervening,
        last_touch,
        *post_touch,
    )
    options = _shuffled_options(
        [
            (anchor, removed_b, incoming),
            (anchor, removed_a, incoming),
            (anchor, removed_a, removed_b),
            (removed_a, removed_b, incoming),
            (anchor, incoming, background),
        ],
        rng,
    )
    pair_id = f"earlier_removal_{index:04d}"
    common = dict(
        pair_id=pair_id,
        family="earlier_removal",
        query_box=query_box,
        initial_state=initial,
        options=options,
        expected_answer_relation="change",
        generation_seed=seed,
        generation_index=index,
    )
    return (
        _build_item(condition="remove_a", operations=operations_a, **common),
        _build_item(condition="remove_b", operations=operations_b, **common),
    )


def _pair_earlier_inbound(
    *, rng: random.Random, seed: int, index: int
) -> tuple[CounterfactualItem, CounterfactualItem]:
    words = iter(rng.sample(OBJECT_VOCAB, 32))
    anchor, removable, incoming_a, incoming_b = (next(words) for _ in range(4))
    query_box, source_a, source_b = rng.sample(range(7), 3)
    initial = _complete_initial_state(
        {
            query_box: [anchor, removable],
            source_a: [incoming_a],
            source_b: [incoming_b],
        },
        words,
    )
    background = next(
        obj
        for box, contents in initial.items()
        if box not in {query_box, source_a, source_b}
        for obj in contents
    )
    safe_boxes = tuple(
        box for box in range(7) if box not in {query_box, source_a, source_b}
    )
    intervening_count = index % 3
    post_touch_count = (index // 3) % 4
    distractors = _put_distractors(
        object_pool=words,
        safe_boxes=safe_boxes,
        count=intervening_count + post_touch_count,
        rng=rng,
    )
    intervening = distractors[:intervening_count]
    post_touch = distractors[intervening_count:]
    last_touch = Operation(kind="remove", objects=(removable,), source=query_box)
    operations_a = (
        Operation(
            kind="move",
            objects=(incoming_a,),
            source=source_a,
            destination=query_box,
        ),
        *intervening,
        last_touch,
        *post_touch,
    )
    operations_b = (
        Operation(
            kind="move",
            objects=(incoming_b,),
            source=source_b,
            destination=query_box,
        ),
        *intervening,
        last_touch,
        *post_touch,
    )
    options = _shuffled_options(
        [
            (anchor, incoming_a),
            (anchor, incoming_b),
            (anchor, removable),
            (incoming_a, incoming_b),
            (anchor, background),
        ],
        rng,
    )
    pair_id = f"earlier_inbound_{index:04d}"
    common = dict(
        pair_id=pair_id,
        family="earlier_inbound",
        query_box=query_box,
        initial_state=initial,
        options=options,
        expected_answer_relation="change",
        generation_seed=seed,
        generation_index=index,
    )
    return (
        _build_item(condition="inbound_a", operations=operations_a, **common),
        _build_item(condition="inbound_b", operations=operations_b, **common),
    )


def _pair_irrelevant_reordering(
    *, rng: random.Random, seed: int, index: int
) -> tuple[CounterfactualItem, CounterfactualItem]:
    words = iter(rng.sample(OBJECT_VOCAB, 32))
    anchor, incoming = next(words), next(words)
    query_box = rng.randrange(7)
    initial = _complete_initial_state({query_box: [anchor]}, words)
    background_words = [
        obj
        for box, contents in initial.items()
        if box != query_box
        for obj in contents
    ]
    safe_boxes = tuple(box for box in range(7) if box != query_box)
    irrelevant_count = 1 + index % 5
    irrelevant = _put_distractors(
        object_pool=words,
        safe_boxes=safe_boxes,
        count=irrelevant_count,
        rng=rng,
    )
    relevant = Operation(kind="put", objects=(incoming,), destination=query_box)
    operations_near = (*irrelevant, relevant)
    operations_far = (relevant, *irrelevant)
    recent_distractor = irrelevant[-1].objects[0]
    early_distractor = irrelevant[0].objects[0]
    second_distractor = (
        early_distractor
        if early_distractor != recent_distractor
        else background_words[0]
    )
    background_distractors = [
        word
        for word in background_words
        if word not in {second_distractor, recent_distractor}
    ][:2]
    options = _shuffled_options(
        [
            (anchor, incoming),
            (anchor, recent_distractor),
            (anchor, second_distractor),
            (anchor, background_distractors[0]),
            (anchor, background_distractors[1]),
        ],
        rng,
    )
    pair_id = f"irrelevant_reordering_{index:04d}"
    common = dict(
        pair_id=pair_id,
        family="irrelevant_reordering",
        query_box=query_box,
        initial_state=initial,
        options=options,
        expected_answer_relation="stable",
        generation_seed=seed,
        generation_index=index,
    )
    return (
        _build_item(condition="near", operations=operations_near, **common),
        _build_item(condition="far", operations=operations_far, **common),
    )


def _pair_whole_box_transfer(
    *, rng: random.Random, seed: int, index: int
) -> tuple[CounterfactualItem, CounterfactualItem]:
    words = iter(rng.sample(OBJECT_VOCAB, 32))
    anchor, source_base, incoming_a, incoming_b = (next(words) for _ in range(4))
    query_box, source_box = rng.sample(range(7), 2)
    initial = _complete_initial_state(
        {query_box: [anchor], source_box: [source_base]}, words
    )
    background = next(
        obj
        for box, contents in initial.items()
        if box not in {query_box, source_box}
        for obj in contents
    )
    safe_boxes = tuple(
        box for box in range(7) if box not in {query_box, source_box}
    )
    intervening_count = index % 3
    post_touch_count = (index // 3) % 4
    distractors = _put_distractors(
        object_pool=words,
        safe_boxes=safe_boxes,
        count=intervening_count + post_touch_count,
        rng=rng,
    )
    intervening = distractors[:intervening_count]
    post_touch = distractors[intervening_count:]
    last_touch = Operation(
        kind="move_contents", source=source_box, destination=query_box
    )
    operations_a = (
        Operation(kind="put", objects=(incoming_a,), destination=source_box),
        *intervening,
        last_touch,
        *post_touch,
    )
    operations_b = (
        Operation(kind="put", objects=(incoming_b,), destination=source_box),
        *intervening,
        last_touch,
        *post_touch,
    )
    options = _shuffled_options(
        [
            (anchor, source_base, incoming_a),
            (anchor, source_base, incoming_b),
            (anchor, source_base, background),
            (anchor, incoming_a, incoming_b),
            (source_base, incoming_a, incoming_b),
        ],
        rng,
    )
    pair_id = f"whole_box_transfer_{index:04d}"
    common = dict(
        pair_id=pair_id,
        family="whole_box_transfer",
        query_box=query_box,
        initial_state=initial,
        options=options,
        expected_answer_relation="change",
        generation_seed=seed,
        generation_index=index,
    )
    return (
        _build_item(condition="source_add_a", operations=operations_a, **common),
        _build_item(condition="source_add_b", operations=operations_b, **common),
    )


PAIR_BUILDERS = {
    "earlier_removal": _pair_earlier_removal,
    "earlier_inbound": _pair_earlier_inbound,
    "irrelevant_reordering": _pair_irrelevant_reordering,
    "whole_box_transfer": _pair_whole_box_transfer,
}


def generate_counterfactual_items(
    *,
    pairs_per_family: int = 300,
    seed: int = 42,
) -> list[CounterfactualItem]:
    """Generate two members for every controlled pair and validate the result."""

    if pairs_per_family <= 0:
        raise ValueError("pairs_per_family must be positive.")
    rng = random.Random(seed)
    items: list[CounterfactualItem] = []
    for family in PROBE_FAMILIES:
        builder = PAIR_BUILDERS[family]
        for index in range(pairs_per_family):
            items.extend(builder(rng=rng, seed=seed, index=index))
    validate_pairs(items, expected_pairs_per_family=pairs_per_family)
    return items


def validate_item(item: CounterfactualItem) -> None:
    if item.family not in PROBE_FAMILIES:
        raise ValueError(f"Unknown probe family: {item.family!r}")
    if len(item.options) != 5 or len(set(item.options)) != 5:
        raise ValueError("Each item must have exactly five unique options.")
    if not 0 <= item.target_index < len(item.options):
        raise ValueError("target_index is outside the option list.")
    initial = item.initial_state_dict()
    final = simulate(initial, item.operations)
    observed_contents = _canonical_objects(final[item.query_box])
    if observed_contents != item.target_contents:
        raise ValueError("Stored target contents disagree with symbolic simulation.")
    if render_contents(observed_contents) != item.target:
        raise ValueError("Stored target answer disagrees with symbolic simulation.")
    expected_prefix = render_input_prefix(initial, item.operations, item.query_box)
    if expected_prefix != item.input_prefix:
        raise ValueError("Stored input_prefix disagrees with structured state/operations.")
    last_touch, distance = find_last_touch(item.operations, item.query_box)
    if (last_touch, distance) != (item.last_touch, item.last_touch_distance):
        raise ValueError("Stored last-touch metadata is incorrect.")


def validate_pairs(
    items: Iterable[CounterfactualItem],
    *,
    expected_pairs_per_family: int | None = None,
) -> None:
    groups: dict[str, list[CounterfactualItem]] = defaultdict(list)
    for item in items:
        validate_item(item)
        groups[item.pair_id].append(item)
    if not groups:
        raise ValueError("Probe set is empty.")

    family_counts: Counter[str] = Counter()
    for pair_id, pair in groups.items():
        if len(pair) != 2:
            raise ValueError(f"Pair {pair_id!r} has {len(pair)} members instead of two.")
        first, second = pair
        if first.family != second.family:
            raise ValueError(f"Pair {pair_id!r} crosses probe families.")
        if first.condition == second.condition:
            raise ValueError(f"Pair {pair_id!r} repeats the same condition.")
        if first.initial_state != second.initial_state:
            raise ValueError(f"Pair {pair_id!r} changes the initial state.")
        if first.query_box != second.query_box or first.options != second.options:
            raise ValueError(f"Pair {pair_id!r} changes query box or answer choices.")
        family_counts[first.family] += 1

        if first.family == "irrelevant_reordering":
            if first.target_index != second.target_index:
                raise ValueError(f"Stable pair {pair_id!r} changes the correct answer.")
            if Counter(op.render() for op in first.operations) != Counter(
                op.render() for op in second.operations
            ):
                raise ValueError(f"Reordering pair {pair_id!r} changes operation content.")
            if {first.condition, second.condition} != {"near", "far"}:
                raise ValueError(f"Reordering pair {pair_id!r} lacks near/far members.")
            near = first if first.condition == "near" else second
            far = second if first.condition == "near" else first
            if near.last_touch_distance != 0 or far.last_touch_distance <= 0:
                raise ValueError(f"Reordering pair {pair_id!r} has invalid distances.")
        else:
            if first.target_index == second.target_index:
                raise ValueError(f"Change pair {pair_id!r} keeps the same correct answer.")
            if first.last_touch != second.last_touch:
                raise ValueError(f"Change pair {pair_id!r} changes its final query-box touch.")

    if expected_pairs_per_family is not None:
        expected = Counter({family: expected_pairs_per_family for family in PROBE_FAMILIES})
        if family_counts != expected:
            raise ValueError(
                f"Unexpected family counts: observed={family_counts}, expected={expected}"
            )


def write_counterfactual_jsonl(
    path: str | Path,
    items: Iterable[CounterfactualItem],
) -> Path:
    path = Path(path)
    materialized = list(items)
    validate_pairs(materialized)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in materialized:
            handle.write(json.dumps(item.to_dict(), sort_keys=True))
            handle.write("\n")
    return path


def read_counterfactual_jsonl(path: str | Path) -> list[CounterfactualItem]:
    path = Path(path)
    items = [
        CounterfactualItem.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_pairs(items)
    return items


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    path: str | Path,
    items: Iterable[CounterfactualItem],
    seed: int,
    pairs_per_family: int,
) -> dict[str, Any]:
    materialized = list(items)
    return {
        "schema_version": 1,
        "generator": "generate_entity_tracking_last_touch_pairs.py",
        "seed": seed,
        "pairs_per_family": pairs_per_family,
        "families": list(PROBE_FAMILIES),
        "pair_count": len(materialized) // 2,
        "item_count": len(materialized),
        "jsonl": Path(path).name,
        "sha256": file_sha256(path),
        "family_pair_counts": dict(
            sorted(Counter(item.family for item in materialized[::2]).items())
        ),
    }


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - half_width, center + half_width]


def _proportion_summary(values: Iterable[bool]) -> dict[str, Any]:
    materialized = list(values)
    successes = sum(materialized)
    total = len(materialized)
    return {
        "successes": successes,
        "total": total,
        "estimate": successes / total if total else None,
        "ci95": _wilson_interval(successes, total) if total else None,
    }


def _bootstrap_mean_interval(
    values: list[float],
    *,
    seed: int,
    samples: int = 2_000,
) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    )
    lower = means[int(0.025 * (samples - 1))]
    upper = means[int(0.975 * (samples - 1))]
    return [lower, upper]


def build_scored_item_records(
    *,
    items: list[CounterfactualItem],
    option_scores: list[list[float]],
    scored_token_counts: list[list[int]],
) -> list[dict[str, Any]]:
    """Attach candidate scores and deterministic predictions to probe items."""

    if len(items) != len(option_scores) or len(items) != len(scored_token_counts):
        raise ValueError("Item and score-list lengths do not match.")
    records: list[dict[str, Any]] = []
    for item, scores, token_counts in zip(
        items, option_scores, scored_token_counts, strict=True
    ):
        if len(scores) != len(item.options) or len(token_counts) != len(item.options):
            raise ValueError(f"Option score count mismatch for {item.pair_id}.")
        best_score = max(scores)
        best_indices = [index for index, score in enumerate(scores) if score == best_score]
        predicted_index = best_indices[0] if len(best_indices) == 1 else None
        lexical_best_indices = lexical_recency_best_indices(item)
        initial_answer = render_contents(item.initial_state_dict()[item.query_box])
        initial_option_index = (
            item.options.index(initial_answer) if initial_answer in item.options else None
        )
        target_score = scores[item.target_index]
        best_distractor_score = max(
            score for index, score in enumerate(scores) if index != item.target_index
        )
        records.append(
            {
                **item.to_dict(),
                "option_scores": scores,
                "scored_token_counts": token_counts,
                "best_indices": best_indices,
                "predicted_index": predicted_index,
                "prediction": (
                    item.options[predicted_index]
                    if predicted_index is not None
                    else None
                ),
                "correct": target_score > best_distractor_score,
                "tie_for_best": len(best_indices) > 1,
                "lexical_recency_best_indices": list(lexical_best_indices),
                "lexical_recency_correct_credit": (
                    1.0 / len(lexical_best_indices)
                    if item.target_index in lexical_best_indices
                    else 0.0
                ),
                "model_lexical_recency_agreement_credit": (
                    len(set(best_indices).intersection(lexical_best_indices))
                    / (len(best_indices) * len(lexical_best_indices))
                ),
                "initial_state_answer": initial_answer,
                "initial_state_option_index": initial_option_index,
                "predicted_initial_state_option": (
                    predicted_index == initial_option_index
                    if initial_option_index is not None
                    else None
                ),
                "target_score": target_score,
                "best_distractor_score": best_distractor_score,
                "margin": target_score - best_distractor_score,
            }
        )
    return records


def _summarize_pair_records(
    records: list[dict[str, Any]],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pairs[record["pair_id"]].append(record)
    if any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("Every scored probe pair must contain exactly two records.")

    prediction_changed: list[bool] = []
    both_correct: list[bool] = []
    directional_shifts: list[float] = []
    recency_margin_effects: list[float] = []
    pair_rows: list[dict[str, Any]] = []

    for pair_id, pair in sorted(pairs.items()):
        first, second = pair
        prediction_changed.append(first["best_indices"] != second["best_indices"])
        both_correct.append(bool(first["correct"] and second["correct"]))
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "prediction_changed": prediction_changed[-1],
            "both_correct": both_correct[-1],
        }

        if first["family"] == "irrelevant_reordering":
            near = first if first["condition"] == "near" else second
            far = second if first["condition"] == "near" else first
            effect = float(near["margin"] - far["margin"])
            recency_margin_effects.append(effect)
            row.update(
                {
                    "near_correct": near["correct"],
                    "far_correct": far["correct"],
                    "near_minus_far_margin": effect,
                    "far_last_touch_distance": far["last_touch_distance"],
                }
            )
        else:
            target_first = int(first["target_index"])
            target_second = int(second["target_index"])
            scores_first = first["option_scores"]
            scores_second = second["option_scores"]
            shift = float(
                (scores_second[target_second] - scores_second[target_first])
                - (scores_first[target_second] - scores_first[target_first])
            )
            directional_shifts.append(shift)
            row.update(
                {
                    "directional_target_shift": shift,
                    "directional_target_shift_positive": shift > 0.0,
                    "last_touch_distance": first["last_touch_distance"],
                }
            )
        pair_rows.append(row)

    accuracy = _proportion_summary(bool(record["correct"]) for record in records)
    changed = _proportion_summary(prediction_changed)
    both = _proportion_summary(both_correct)
    summary: dict[str, Any] = {
        "pairs": len(pairs),
        "items": len(records),
        "accuracy": accuracy,
        "prediction_changed": changed,
        "prediction_invariant": _proportion_summary(not value for value in prediction_changed),
        "both_correct": both,
        "ties": _proportion_summary(bool(record["tie_for_best"]) for record in records),
        "lexical_recency_expected_accuracy": statistics.fmean(
            float(record["lexical_recency_correct_credit"]) for record in records
        ),
        "model_lexical_recency_agreement": statistics.fmean(
            float(record["model_lexical_recency_agreement_credit"])
            for record in records
        ),
    }

    initial_eligible = [
        record for record in records if record["initial_state_option_index"] is not None
    ]
    if initial_eligible:
        summary["initial_state_option_selected"] = _proportion_summary(
            bool(record["predicted_initial_state_option"])
            for record in initial_eligible
        )

    if recency_margin_effects:
        near_records = [record for record in records if record["condition"] == "near"]
        far_records = [record for record in records if record["condition"] == "far"]
        summary.update(
            {
                "near_accuracy": _proportion_summary(
                    bool(record["correct"]) for record in near_records
                ),
                "far_accuracy": _proportion_summary(
                    bool(record["correct"]) for record in far_records
                ),
                "near_minus_far_margin_mean": statistics.fmean(recency_margin_effects),
                "near_minus_far_margin_ci95": _bootstrap_mean_interval(
                    recency_margin_effects, seed=bootstrap_seed
                ),
                "near_margin_better": _proportion_summary(
                    value > 0.0 for value in recency_margin_effects
                ),
                "near_lexical_recency_expected_accuracy": statistics.fmean(
                    float(record["lexical_recency_correct_credit"])
                    for record in near_records
                ),
                "far_lexical_recency_expected_accuracy": statistics.fmean(
                    float(record["lexical_recency_correct_credit"])
                    for record in far_records
                ),
            }
        )
    if directional_shifts:
        summary.update(
            {
                "directional_target_shift_mean": statistics.fmean(directional_shifts),
                "directional_target_shift_ci95": _bootstrap_mean_interval(
                    directional_shifts, seed=bootstrap_seed
                ),
                "directional_target_shift_positive": _proportion_summary(
                    value > 0.0 for value in directional_shifts
                ),
            }
        )
    summary["pair_records"] = pair_rows
    return summary


def summarize_scored_records(
    records: list[dict[str, Any]],
    *,
    bootstrap_seed: int = 1729,
) -> dict[str, Any]:
    """Summarize paired accuracy, invariance, and likelihood-margin effects."""

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[record["family"]].append(record)
    missing = set(PROBE_FAMILIES).difference(by_family)
    if missing:
        raise ValueError(f"Scored records are missing probe families: {sorted(missing)}")

    family_summaries: dict[str, Any] = {}
    for family_index, family in enumerate(PROBE_FAMILIES):
        family_records = by_family[family]
        summary = _summarize_pair_records(
            family_records,
            bootstrap_seed=bootstrap_seed + family_index,
        )

        distance_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        pair_records = summary.pop("pair_records")
        distance_key = (
            "far_last_touch_distance"
            if family == "irrelevant_reordering"
            else "last_touch_distance"
        )
        for pair_record in pair_records:
            distance_groups[str(pair_record[distance_key])].append(pair_record)
        summary["by_last_touch_distance"] = {
            distance: {
                "pairs": len(group),
                "prediction_changed": _proportion_summary(
                    bool(record["prediction_changed"]) for record in group
                ),
                "both_correct": _proportion_summary(
                    bool(record["both_correct"]) for record in group
                ),
            }
            for distance, group in sorted(
                distance_groups.items(), key=lambda pair: int(pair[0])
            )
        }
        family_summaries[family] = summary

    return {
        "schema_version": 1,
        "pair_count": len(records) // 2,
        "item_count": len(records),
        "bootstrap_seed": bootstrap_seed,
        "families": family_summaries,
    }


__all__ = [
    "CounterfactualItem",
    "OBJECT_VOCAB",
    "Operation",
    "PROBE_FAMILIES",
    "apply_operation",
    "build_manifest",
    "build_scored_item_records",
    "file_sha256",
    "find_last_touch",
    "generate_counterfactual_items",
    "read_counterfactual_jsonl",
    "lexical_recency_best_indices",
    "render_contents",
    "render_input_prefix",
    "simulate",
    "summarize_scored_records",
    "validate_item",
    "validate_pairs",
    "write_counterfactual_jsonl",
]
