"""Matched query-box-touch ladders for Entity Tracking diagnostics."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from babysteps.evaluation.entity_tracking_counterfactuals import (
    OBJECT_VOCAB,
    Operation,
    file_sha256,
    find_last_touch,
    render_contents,
    render_input_prefix,
    simulate,
)


LadderCondition = Literal["q1", "q3", "q5", "t3", "t5", "c3", "c5"]
LADDER_CONDITIONS: tuple[LadderCondition, ...] = ("q1", "q3", "q5", "c3", "c5")
EXTENDED_LADDER_CONDITIONS: tuple[LadderCondition, ...] = (
    "q1",
    "q3",
    "q5",
    "t3",
    "t5",
    "c3",
    "c5",
)
EXPECTED_COUNTS = {
    "q1": (1, 1),
    "q3": (3, 3),
    "q5": (5, 5),
    "t3": (3, 3),
    "t5": (5, 5),
    "c3": (3, 1),
    "c5": (5, 1),
}
OBJECT_INDEX = {word: index for index, word in enumerate(OBJECT_VOCAB)}


@dataclass(frozen=True)
class TouchLadderItem:
    """One condition for a world shared across the query-touch ladder."""

    base_id: str
    condition: LadderCondition
    query_box: int
    initial_state: tuple[tuple[int, tuple[str, ...]], ...]
    operations: tuple[Operation, ...]
    input_prefix: str
    options: tuple[str, ...]
    target_index: int
    target_contents: tuple[str, ...]
    total_operation_count: int
    query_touch_count: int
    last_touch: str
    last_touch_distance: int
    generation_seed: int
    generation_index: int

    @property
    def target(self) -> str:
        return self.options[self.target_index]

    def initial_state_dict(self) -> dict[int, list[str]]:
        return {box: list(objects) for box, objects in self.initial_state}

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_id": self.base_id,
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
            "total_operation_count": self.total_operation_count,
            "query_touch_count": self.query_touch_count,
            "last_touch": self.last_touch,
            "last_touch_distance": self.last_touch_distance,
            "generation_seed": self.generation_seed,
            "generation_index": self.generation_index,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TouchLadderItem:
        item = cls(
            base_id=str(value["base_id"]),
            condition=value["condition"],
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
            total_operation_count=int(value["total_operation_count"]),
            query_touch_count=int(value["query_touch_count"]),
            last_touch=str(value["last_touch"]),
            last_touch_distance=int(value["last_touch_distance"]),
            generation_seed=int(value["generation_seed"]),
            generation_index=int(value["generation_index"]),
        )
        if value.get("target") not in (None, item.target):
            raise ValueError("Serialized target disagrees with target_index.")
        validate_item(item)
        return item


def _state_tuple(state: dict[int, list[str]]) -> tuple[tuple[int, tuple[str, ...]], ...]:
    return tuple((box, tuple(state[box])) for box in sorted(state))


def _canonical_contents(objects: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(objects), key=OBJECT_INDEX.__getitem__))


def _query_touch_count(operations: Iterable[Operation], query_box: int) -> int:
    return sum(query_box in operation.touched_boxes() for operation in operations)


def _render_options(
    candidates: list[tuple[str, ...]], rng: random.Random
) -> tuple[str, ...]:
    options = [render_contents(candidate) for candidate in candidates]
    if len(options) != 5 or len(set(options)) != 5:
        raise ValueError(f"Expected five unique ladder options, got {options!r}.")
    if len({option.count("the ") for option in options}) != 1:
        raise ValueError("All ladder options must have equal object cardinality.")
    rng.shuffle(options)
    return tuple(options)


def _build_item(
    *,
    base_id: str,
    condition: LadderCondition,
    query_box: int,
    initial_state: dict[int, list[str]],
    operations: tuple[Operation, ...],
    options: tuple[str, ...],
    seed: int,
    index: int,
) -> TouchLadderItem:
    final_state = simulate(initial_state, operations)
    target_contents = _canonical_contents(final_state[query_box])
    target = render_contents(target_contents)
    if target not in options:
        raise ValueError(f"Target {target!r} is absent from ladder options.")
    last_touch, distance = find_last_touch(operations, query_box)
    item = TouchLadderItem(
        base_id=base_id,
        condition=condition,
        query_box=query_box,
        initial_state=_state_tuple(initial_state),
        operations=operations,
        input_prefix=render_input_prefix(initial_state, operations, query_box),
        options=options,
        target_index=options.index(target),
        target_contents=target_contents,
        total_operation_count=len(operations),
        query_touch_count=_query_touch_count(operations, query_box),
        last_touch=last_touch,
        last_touch_distance=distance,
        generation_seed=seed,
        generation_index=index,
    )
    validate_item(item)
    return item


def _generate_world(
    *, rng: random.Random, seed: int, index: int, include_transient: bool
) -> tuple[TouchLadderItem, ...]:
    words = iter(rng.sample(OBJECT_VOCAB, 24))
    anchor, restored_a, restored_b, incoming = (next(words) for _ in range(4))
    control_a, control_b = next(words), next(words)
    boxes = rng.sample(range(7), 7)
    (
        query_box,
        query_destination_a,
        query_destination_b,
        control_source_a,
        control_destination_a,
        control_source_b,
        control_destination_b,
    ) = boxes

    initial_state = {
        query_box: [anchor, restored_a, restored_b],
        query_destination_a: [next(words)],
        query_destination_b: [next(words)],
        control_source_a: [control_a],
        control_destination_a: [next(words)],
        control_source_b: [control_b],
        control_destination_b: [next(words)],
    }
    final_touch = Operation(kind="put", objects=(incoming,), destination=query_box)
    query_pair_a = (
        Operation(
            kind="move",
            objects=(restored_a,),
            source=query_box,
            destination=query_destination_a,
        ),
        Operation(
            kind="move",
            objects=(restored_a,),
            source=query_destination_a,
            destination=query_box,
        ),
    )
    query_pair_b = (
        Operation(
            kind="move",
            objects=(restored_b,),
            source=query_box,
            destination=query_destination_b,
        ),
        Operation(
            kind="move",
            objects=(restored_b,),
            source=query_destination_b,
            destination=query_box,
        ),
    )
    control_pair_a = (
        Operation(
            kind="move",
            objects=(control_a,),
            source=control_source_a,
            destination=control_destination_a,
        ),
        Operation(
            kind="move",
            objects=(control_a,),
            source=control_destination_a,
            destination=control_source_a,
        ),
    )
    control_pair_b = (
        Operation(
            kind="move",
            objects=(control_b,),
            source=control_source_b,
            destination=control_destination_b,
        ),
        Operation(
            kind="move",
            objects=(control_b,),
            source=control_destination_b,
            destination=control_source_b,
        ),
    )
    transient_pair_a = (
        Operation(
            kind="move",
            objects=(control_a,),
            source=control_source_a,
            destination=query_box,
        ),
        Operation(
            kind="move",
            objects=(control_a,),
            source=query_box,
            destination=control_source_a,
        ),
    )
    transient_pair_b = (
        Operation(
            kind="move",
            objects=(control_b,),
            source=control_source_b,
            destination=query_box,
        ),
        Operation(
            kind="move",
            objects=(control_b,),
            source=query_box,
            destination=control_source_b,
        ),
    )

    options = _render_options(
        [
            (anchor, restored_a, restored_b, incoming),
            (anchor, restored_a, restored_b, control_a),
            (anchor, restored_a, incoming, control_a),
            (anchor, restored_b, incoming, control_b),
            (restored_a, restored_b, incoming, control_b),
        ],
        rng,
    )
    operations = {
        "q1": (final_touch,),
        "q3": (*query_pair_a, final_touch),
        "q5": (*query_pair_a, *query_pair_b, final_touch),
        "t3": (*transient_pair_a, final_touch),
        "t5": (*transient_pair_a, *transient_pair_b, final_touch),
        "c3": (*control_pair_a, final_touch),
        "c5": (*control_pair_a, *control_pair_b, final_touch),
    }
    base_id = f"query_touch_ladder_{index:04d}"
    conditions = EXTENDED_LADDER_CONDITIONS if include_transient else LADDER_CONDITIONS
    return tuple(
        _build_item(
            base_id=base_id,
            condition=condition,
            query_box=query_box,
            initial_state=initial_state,
            operations=operations[condition],
            options=options,
            seed=seed,
            index=index,
        )
        for condition in conditions
    )


def generate_touch_ladder_items(
    *, worlds: int = 300, seed: int = 42, include_transient: bool = False
) -> list[TouchLadderItem]:
    if worlds <= 0:
        raise ValueError("worlds must be positive.")
    rng = random.Random(seed)
    items = [
        item
        for index in range(worlds)
        for item in _generate_world(
            rng=rng,
            seed=seed,
            index=index,
            include_transient=include_transient,
        )
    ]
    validate_ladders(items, expected_worlds=worlds)
    return items


def validate_item(item: TouchLadderItem) -> None:
    if item.condition not in EXTENDED_LADDER_CONDITIONS:
        raise ValueError(f"Unknown ladder condition: {item.condition!r}")
    if len(item.options) != 5 or len(set(item.options)) != 5:
        raise ValueError("Each ladder item must have five unique options.")
    if "nothing." in item.options:
        raise ValueError("The query-touch ladder excludes nothing options.")
    if len({option.count("the ") for option in item.options}) != 1:
        raise ValueError("Ladder option cardinalities differ.")
    initial = item.initial_state_dict()
    final = simulate(initial, item.operations)
    if _canonical_contents(final[item.query_box]) != item.target_contents:
        raise ValueError("Stored ladder target disagrees with simulation.")
    if render_contents(item.target_contents) != item.target:
        raise ValueError("Stored target string disagrees with simulation.")
    if render_input_prefix(initial, item.operations, item.query_box) != item.input_prefix:
        raise ValueError("Stored input prefix disagrees with structured operations.")
    total, touches = EXPECTED_COUNTS[item.condition]
    if (item.total_operation_count, item.query_touch_count) != (total, touches):
        raise ValueError("Stored operation counts disagree with ladder condition.")
    if item.last_touch_distance != 0:
        raise ValueError("The final operation must be the last query-box touch.")


def validate_ladders(
    items: Iterable[TouchLadderItem], *, expected_worlds: int | None = None
) -> None:
    groups: dict[str, list[TouchLadderItem]] = defaultdict(list)
    for item in items:
        validate_item(item)
        groups[item.base_id].append(item)
    if not groups:
        raise ValueError("Touch ladder is empty.")
    if expected_worlds is not None and len(groups) != expected_worlds:
        raise ValueError(f"Expected {expected_worlds} worlds, found {len(groups)}.")
    for base_id, group in groups.items():
        observed_conditions = frozenset(item.condition for item in group)
        if observed_conditions not in {
            frozenset(LADDER_CONDITIONS),
            frozenset(EXTENDED_LADDER_CONDITIONS),
        }:
            raise ValueError(f"World {base_id!r} lacks a complete condition ladder.")
        first = group[0]
        for item in group[1:]:
            if (
                item.initial_state != first.initial_state
                or item.query_box != first.query_box
                or item.options != first.options
                or item.target_index != first.target_index
                or item.target_contents != first.target_contents
                or item.last_touch != first.last_touch
            ):
                raise ValueError(f"World {base_id!r} changes a controlled field.")


def write_touch_ladder_jsonl(
    path: str | Path, items: Iterable[TouchLadderItem]
) -> Path:
    path = Path(path)
    materialized = list(items)
    validate_ladders(materialized)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in materialized:
            handle.write(json.dumps(item.to_dict(), sort_keys=True))
            handle.write("\n")
    return path


def read_touch_ladder_jsonl(path: str | Path) -> list[TouchLadderItem]:
    items = [
        TouchLadderItem.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_ladders(items)
    return items


def build_manifest(
    *, path: str | Path, items: list[TouchLadderItem], seed: int, worlds: int
) -> dict[str, Any]:
    conditions = tuple(
        condition
        for condition in EXTENDED_LADDER_CONDITIONS
        if any(item.condition == condition for item in items)
    )
    return {
        "schema_version": 1,
        "generator": "generate_entity_tracking_query_touch_ladder.py",
        "seed": seed,
        "world_count": worlds,
        "item_count": len(items),
        "conditions": list(conditions),
        "condition_counts": dict(sorted(Counter(item.condition for item in items).items())),
        "jsonl": Path(path).name,
        "sha256": file_sha256(path),
    }


def build_scored_records(
    *,
    items: list[TouchLadderItem],
    option_scores: list[list[float]],
    scored_token_counts: list[list[int]],
) -> list[dict[str, Any]]:
    if not (len(items) == len(option_scores) == len(scored_token_counts)):
        raise ValueError("Ladder items and score lists have different lengths.")
    records = []
    for item, scores, token_counts in zip(
        items, option_scores, scored_token_counts, strict=True
    ):
        if len(scores) != 5 or len(token_counts) != 5:
            raise ValueError("Every ladder item requires five option scores.")
        best_score = max(scores)
        best_indices = [index for index, score in enumerate(scores) if score == best_score]
        target_score = scores[item.target_index]
        best_distractor = max(
            score for index, score in enumerate(scores) if index != item.target_index
        )
        records.append(
            {
                **item.to_dict(),
                "option_scores": scores,
                "scored_token_counts": token_counts,
                "best_indices": best_indices,
                "predicted_index": best_indices[0] if len(best_indices) == 1 else None,
                "correct": target_score > best_distractor,
                "target_score": target_score,
                "best_distractor_score": best_distractor,
                "margin": target_score - best_distractor,
                "tie_for_best": len(best_indices) > 1,
            }
        )
    return records


def _wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - half, center + half]


def _proportion(values: Iterable[bool]) -> dict[str, Any]:
    materialized = list(values)
    successes = sum(materialized)
    return {
        "successes": successes,
        "total": len(materialized),
        "estimate": successes / len(materialized),
        "ci95": _wilson(successes, len(materialized)),
    }


def _bootstrap_mean(
    values: list[float], *, seed: int, samples: int = 2_000
) -> list[float]:
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    )
    return [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]]


def summarize_scored_records(
    records: list[dict[str, Any]], *, bootstrap_seed: int = 2718
) -> dict[str, Any]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        groups[record["base_id"]][record["condition"]] = record
    condition_sets = {frozenset(group) for group in groups.values()}
    if len(condition_sets) != 1:
        raise ValueError("Scored records do not contain complete ladders.")
    observed_conditions = next(iter(condition_sets))
    if observed_conditions not in {
        frozenset(LADDER_CONDITIONS),
        frozenset(EXTENDED_LADDER_CONDITIONS),
    }:
        raise ValueError("Scored records contain an unsupported ladder design.")
    conditions = tuple(
        condition
        for condition in EXTENDED_LADDER_CONDITIONS
        if condition in observed_conditions
    )

    condition_summary = {}
    for condition in conditions:
        condition_records = [group[condition] for group in groups.values()]
        margins = [float(record["margin"]) for record in condition_records]
        condition_summary[condition] = {
            "items": len(condition_records),
            "accuracy": _proportion(bool(record["correct"]) for record in condition_records),
            "mean_margin": statistics.fmean(margins),
            "mean_margin_ci95": _bootstrap_mean(
                margins, seed=bootstrap_seed + conditions.index(condition)
            ),
            "mean_max_option_tokens": statistics.fmean(
                max(record["scored_token_counts"]) for record in condition_records
            ),
        }

    contrasts = [
        ("q3_minus_c3", "q3", "c3"),
        ("q5_minus_c5", "q5", "c5"),
        ("q3_minus_q1", "q3", "q1"),
        ("q5_minus_q1", "q5", "q1"),
        ("c3_minus_q1", "c3", "q1"),
        ("c5_minus_q1", "c5", "q1"),
    ]
    if "t3" in observed_conditions:
        contrasts.extend(
            [
                ("t3_minus_c3", "t3", "c3"),
                ("t5_minus_c5", "t5", "c5"),
                ("q3_minus_t3", "q3", "t3"),
                ("q5_minus_t5", "q5", "t5"),
            ]
        )
    contrast_summary = {}
    for contrast_index, (name, treatment, control) in enumerate(contrasts):
        margin_differences = [
            float(group[treatment]["margin"] - group[control]["margin"])
            for group in groups.values()
        ]
        accuracy_differences = [
            float(group[treatment]["correct"] - group[control]["correct"])
            for group in groups.values()
        ]
        prediction_changes = [
            group[treatment]["best_indices"] != group[control]["best_indices"]
            for group in groups.values()
        ]
        contrast_summary[name] = {
            "treatment": treatment,
            "control": control,
            "pairs": len(groups),
            "mean_margin_difference": statistics.fmean(margin_differences),
            "mean_margin_difference_ci95": _bootstrap_mean(
                margin_differences, seed=bootstrap_seed + 100 + contrast_index
            ),
            "margin_higher": _proportion(value > 0 for value in margin_differences),
            "accuracy_difference": statistics.fmean(accuracy_differences),
            "accuracy_difference_ci95": _bootstrap_mean(
                accuracy_differences, seed=bootstrap_seed + 200 + contrast_index
            ),
            "prediction_changed": _proportion(prediction_changes),
        }

    return {
        "schema_version": 1,
        "world_count": len(groups),
        "item_count": len(records),
        "bootstrap_seed": bootstrap_seed,
        "condition_order": list(conditions),
        "conditions": condition_summary,
        "contrasts": contrast_summary,
    }


__all__ = [
    "EXPECTED_COUNTS",
    "EXTENDED_LADDER_CONDITIONS",
    "LADDER_CONDITIONS",
    "TouchLadderItem",
    "build_manifest",
    "build_scored_records",
    "generate_touch_ladder_items",
    "read_touch_ladder_jsonl",
    "summarize_scored_records",
    "validate_item",
    "validate_ladders",
    "write_touch_ladder_jsonl",
]
