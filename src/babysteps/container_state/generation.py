"""Generate naturalized, auditable container-state curriculum documents."""

from __future__ import annotations

import json
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from babysteps.container_state.validation import build_distractor_states
from babysteps.container_state.world import (
    Operation,
    OperationKind,
    World,
    apply_operation,
    apply_operations,
)
from babysteps.paths import expand_env_vars, resolve_path


WORD_RE = re.compile(r"\S+")
LEXICAL_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
TRAIN_RENDERERS = (
    "household_narrative",
    "inventory_report",
    "cleanup_summary",
    "conversational_recap",
)
HELDOUT_RENDERERS = (
    "storage_log",
    "observational_report",
)
SUPPORTED_RENDERERS = frozenset((*TRAIN_RENDERERS, *HELDOUT_RENDERERS))
OPERATION_KINDS: tuple[OperationKind, ...] = ("add", "remove", "move")
AGENTS = ("Mina", "Nora", "Theo", "Rafi", "Lila", "Omar", "Tessa", "Hugo")


@dataclass(frozen=True)
class LexiconSplit:
    """Disjoint surface vocabulary for one generation split."""

    container_types: tuple[str, ...]
    labels: tuple[str, ...]
    objects: tuple[str, ...]


@dataclass(frozen=True)
class LexiconBundle:
    train: LexiconSplit
    validation: LexiconSplit
    test: LexiconSplit
    benchmark_objects: frozenset[str]
    forbidden_patterns: tuple[str, ...]


@dataclass(frozen=True)
class RenderedScenario:
    """A complete training document plus its scorable final continuation."""

    text: str
    input_prefix: str
    completion: str
    completion_items: tuple[str, ...]
    completion_mode: str


@dataclass(frozen=True)
class ContainerStateCorpusResult:
    name: str
    output_dir: Path
    documents: tuple[dict[str, Any], ...]
    validation_items: dict[str, tuple[dict[str, Any], ...]]
    audit_report: dict[str, Any]
    outputs: dict[str, Path]
    manifest: dict[str, Any]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping with repository-style environment expansion."""

    with Path(path).open("r", encoding="utf-8") as f:
        value = expand_env_vars(yaml.safe_load(f))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping in {path}.")
    return value


def _as_nonempty_strings(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list.")
    values = tuple(str(item).strip() for item in value)
    if any(not item for item in values):
        raise ValueError(f"{field} must not contain empty values.")
    if len(set(item.lower() for item in values)) != len(values):
        raise ValueError(f"{field} must not contain duplicates.")
    return values


def _lexicon_split(mapping: Mapping[str, Any], *, field: str) -> LexiconSplit:
    return LexiconSplit(
        container_types=_as_nonempty_strings(
            mapping.get("container_types"), field=f"{field}.container_types"
        ),
        labels=_as_nonempty_strings(mapping.get("labels"), field=f"{field}.labels"),
        objects=_as_nonempty_strings(mapping.get("objects"), field=f"{field}.objects"),
    )


def _load_lexicons(path: Path) -> LexiconBundle:
    mapping = load_yaml_config(path)
    for split in ("train", "validation", "test"):
        if not isinstance(mapping.get(split), dict):
            raise ValueError(f"Lexicon config requires a {split!r} mapping.")
    exclusions = mapping.get("exclusions") or {}
    if not isinstance(exclusions, dict):
        raise TypeError("lexicon exclusions must be a mapping.")
    bundle = LexiconBundle(
        train=_lexicon_split(mapping["train"], field="train"),
        validation=_lexicon_split(mapping["validation"], field="validation"),
        test=_lexicon_split(mapping["test"], field="test"),
        benchmark_objects=frozenset(
            item.lower()
            for item in _as_nonempty_strings(
                exclusions.get("benchmark_objects"),
                field="exclusions.benchmark_objects",
            )
        ),
        forbidden_patterns=_as_nonempty_strings(
            exclusions.get("forbidden_patterns"),
            field="exclusions.forbidden_patterns",
        ),
    )
    _validate_lexicons(bundle)
    return bundle


def _normalized(values: Sequence[str]) -> set[str]:
    return {value.lower() for value in values}


def _validate_lexicons(bundle: LexiconBundle) -> None:
    splits = {
        "train": bundle.train,
        "validation": bundle.validation,
        "test": bundle.test,
    }
    for field in ("container_types", "labels", "objects"):
        values = {
            name: _normalized(getattr(split, field))
            for name, split in splits.items()
        }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = values[left] & values[right]
            if overlap:
                raise ValueError(
                    f"Lexicon {field} overlap between {left} and {right}: "
                    f"{sorted(overlap)}"
                )

    all_container_types = set().union(
        *(_normalized(split.container_types) for split in splits.values())
    )
    all_objects = set().union(*(_normalized(split.objects) for split in splits.values()))
    overlap = all_container_types & all_objects
    if overlap:
        raise ValueError(f"Container types cannot also be objects: {sorted(overlap)}")

    benchmark_overlap = all_objects & bundle.benchmark_objects
    if benchmark_overlap:
        raise ValueError(
            "Synthetic object vocabulary overlaps the benchmark inventory: "
            f"{sorted(benchmark_overlap)}"
        )

    for pattern in bundle.forbidden_patterns:
        re.compile(pattern, flags=re.IGNORECASE)


def _weighted_choice(mapping: Mapping[Any, Any], rng: random.Random) -> str:
    if not mapping:
        raise ValueError("A weighted choice mapping must not be empty.")
    values: list[str] = []
    weights: list[float] = []
    for value, raw_weight in mapping.items():
        weight = float(raw_weight)
        if weight < 0:
            raise ValueError("Choice weights must be non-negative.")
        if weight > 0:
            values.append(str(value))
            weights.append(weight)
    if not values:
        raise ValueError("At least one choice weight must be positive.")
    return rng.choices(values, weights=weights, k=1)[0]


def _container_name(label: str, container_type: str) -> str:
    return f"{label} {container_type}"


def _sample_world(
    *,
    lexicon: LexiconSplit,
    container_count: int,
    min_items: int,
    max_items: int,
    same_type_ratio: float,
    rng: random.Random,
) -> tuple[World, str]:
    if container_count <= 1:
        raise ValueError("container_count must be at least 2.")
    if container_count > len(lexicon.labels):
        raise ValueError("Not enough labels for the requested container count.")
    if min_items <= 0 or max_items < min_items:
        raise ValueError("Initial item bounds must satisfy 1 <= min <= max.")

    labels = rng.sample(list(lexicon.labels), container_count)
    same_type = rng.random() < same_type_ratio
    if same_type:
        container_type = rng.choice(lexicon.container_types)
        container_types = [container_type] * container_count
        world_style = "same_type_labeled"
    else:
        if container_count > len(lexicon.container_types):
            raise ValueError("Not enough container types for a mixed-type world.")
        container_types = rng.sample(list(lexicon.container_types), container_count)
        world_style = "mixed_types"

    item_counts = [rng.randint(min_items, max_items) for _ in range(container_count)]
    required_items = sum(item_counts)
    if required_items > len(lexicon.objects):
        raise ValueError("Not enough unique objects for the sampled world.")
    sampled_items = rng.sample(list(lexicon.objects), required_items)

    state: dict[str, list[str]] = {}
    cursor = 0
    for label, container_type, item_count in zip(labels, container_types, item_counts):
        name = _container_name(label, container_type)
        state[name] = sampled_items[cursor : cursor + item_count]
        cursor += item_count
    return World.from_mapping(state), world_style


def _sample_operation(
    world: World,
    *,
    kind: OperationKind,
    object_pool: Sequence[str],
    rng: random.Random,
) -> Operation:
    if kind == "add":
        available = [item for item in object_pool if world.locate(item) is None]
        if not available:
            raise ValueError("No unused object is available for an add operation.")
        return Operation(
            kind="add",
            item=rng.choice(available),
            destination=rng.choice(world.containers),
        )

    nonempty = [container for container in world.containers if world.items_in(container)]
    if not nonempty:
        raise ValueError(f"No nonempty source is available for {kind}.")
    source = rng.choice(nonempty)
    item = rng.choice(world.items_in(source))
    if kind == "remove":
        return Operation(kind="remove", item=item, source=source)
    if kind == "move":
        destinations = [container for container in world.containers if container != source]
        return Operation(
            kind="move",
            item=item,
            source=source,
            destination=rng.choice(destinations),
        )
    raise ValueError(f"Unsupported operation kind: {kind!r}.")


def _sample_operations(
    world: World,
    *,
    operation_count: int,
    operation_kind_weights: Mapping[Any, Any],
    object_pool: Sequence[str],
    rng: random.Random,
    forced_first_kind: OperationKind | None = None,
) -> tuple[tuple[Operation, ...], World]:
    current = world
    operations: list[Operation] = []
    for index in range(operation_count):
        if index == 0 and forced_first_kind is not None:
            kind = forced_first_kind
        else:
            raw_kind = _weighted_choice(operation_kind_weights, rng)
            if raw_kind not in OPERATION_KINDS:
                raise ValueError(f"Unsupported configured operation kind: {raw_kind!r}.")
            kind = raw_kind  # type: ignore[assignment]
        operation = _sample_operation(
            current,
            kind=kind,
            object_pool=object_pool,
            rng=rng,
        )
        operations.append(operation)
        current = apply_operation(current, operation)
    return tuple(operations), current


def _choose_target(
    world: World,
    operations: Sequence[Operation],
    *,
    rng: random.Random,
) -> tuple[str, str]:
    if not operations:
        return rng.choice(world.containers), "zero_operation"

    last = operations[-1]
    untouched = [
        container
        for container in world.containers
        if container not in last.touched_containers
    ]
    role_targets: list[tuple[str, str]] = []
    if last.source is not None:
        role_targets.append((last.source, "source"))
    if last.destination is not None:
        role_targets.append((last.destination, "destination"))
    if untouched:
        role_targets.append((rng.choice(untouched), "untouched"))
    return rng.choice(role_targets)


def _article(item: str) -> str:
    return "an" if item[:1].lower() in "aeiou" else "a"


def _ordered_items(items: Sequence[str], rng: random.Random) -> tuple[str, ...]:
    values = list(items)
    rng.shuffle(values)
    return tuple(values)


def _join_phrases(values: Sequence[str]) -> str:
    if not values:
        raise ValueError("Cannot join an empty phrase list.")
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _item_phrase(
    items: Sequence[str],
    *,
    rng: random.Random,
    articles: bool,
) -> tuple[str, tuple[str, ...]]:
    ordered = _ordered_items(items, rng)
    if not ordered:
        return "", ordered
    phrases = [f"{_article(item)} {item}" if articles else item for item in ordered]
    return _join_phrases(phrases), ordered


def _completion_for(
    items: Sequence[str],
    *,
    mode: str,
    rng: random.Random,
) -> tuple[str, tuple[str, ...]]:
    if not items:
        empty = {
            "held": "no items.",
            "listed": "no items.",
            "were": "none.",
            "log": "empty.",
        }[mode]
        return empty, ()
    phrase, ordered = _item_phrase(items, rng=rng, articles=mode != "log")
    return f"{phrase}.", ordered


def _initial_description(
    world: World,
    *,
    renderer: str,
    rng: random.Random,
) -> str:
    descriptions: list[str] = []
    for container, items in world.contents:
        if renderer == "storage_log":
            phrase, _ = _item_phrase(items, rng=rng, articles=False)
            descriptions.append(f"{container}: {phrase}")
        else:
            phrase, _ = _item_phrase(items, rng=rng, articles=True)
            if renderer == "inventory_report":
                descriptions.append(f"the {container} listed {phrase}")
            elif renderer == "cleanup_summary":
                descriptions.append(f"the {container} started with {phrase}")
            elif renderer == "conversational_recap":
                descriptions.append(f"the {container} had {phrase}")
            elif renderer == "observational_report":
                descriptions.append(f"the {container} held {phrase}")
            else:
                descriptions.append(f"The {container} held {phrase}.")

    if renderer == "household_narrative":
        return " ".join(descriptions)
    if renderer == "inventory_report":
        return f"The opening inventory said that {_join_phrases(descriptions)}."
    if renderer == "cleanup_summary":
        return f"Before cleanup, {_join_phrases(descriptions)}."
    if renderer == "conversational_recap":
        return f"{AGENTS[0]} read the storage list aloud: {_join_phrases(descriptions)}."
    if renderer == "storage_log":
        return f"Opening storage log: {'; '.join(descriptions)}."
    if renderer == "observational_report":
        return f"At the first inspection, {_join_phrases(descriptions)}."
    raise ValueError(f"Unknown renderer: {renderer!r}.")


def _operation_description(
    operation: Operation,
    *,
    renderer: str,
    agent: str,
    rng: random.Random,
) -> str:
    item = f"the {operation.item}"
    if operation.kind == "add":
        assert operation.destination is not None
        frames = (
            f"{agent} later placed {item} in the {operation.destination}.",
            f"Afterward, {agent} stored {item} in the {operation.destination}.",
            f"Next, {item} was added to the {operation.destination}.",
        )
    elif operation.kind == "remove":
        assert operation.source is not None
        frames = (
            f"{agent} later took {item} out of the {operation.source}.",
            f"Afterward, {agent} lifted {item} from the {operation.source}.",
            f"Next, {item} was taken out of the {operation.source}.",
        )
    else:
        assert operation.source is not None
        assert operation.destination is not None
        frames = (
            f"{agent} later transferred {item} from the {operation.source} "
            f"to the {operation.destination}.",
            f"Afterward, {agent} carried {item} from the {operation.source} "
            f"to the {operation.destination}.",
            f"Next, {item} was shifted from the {operation.source} "
            f"to the {operation.destination}.",
        )

    if renderer == "inventory_report":
        return f"An update followed: {rng.choice(frames)}"
    if renderer == "storage_log":
        return f"Transfer entry: {rng.choice(frames)}"
    return rng.choice(frames)


def _render_scenario(
    *,
    initial_world: World,
    final_world: World,
    operations: Sequence[Operation],
    target_container: str,
    renderer: str,
    rng: random.Random,
) -> RenderedScenario:
    if renderer not in SUPPORTED_RENDERERS:
        raise ValueError(f"Unknown container-state renderer: {renderer!r}.")

    initial = _initial_description(initial_world, renderer=renderer, rng=rng)
    agent = rng.choice(AGENTS)
    operation_text = " ".join(
        _operation_description(
            operation,
            renderer=renderer,
            agent=agent,
            rng=rng,
        )
        for operation in operations
    )
    middle = f" {operation_text}" if operation_text else ""

    if renderer == "household_narrative":
        prefix = (
            f"Several labeled storage places were organized during the afternoon. "
            f"{initial}{middle} When the organizing was finished, "
            f"the {target_container} held "
        )
        mode = "held"
    elif renderer == "inventory_report":
        prefix = f"{initial}{middle} The revised entry for the {target_container} listed "
        mode = "listed"
    elif renderer == "cleanup_summary":
        prefix = f"{initial}{middle} At the final check, the items in the {target_container} were "
        mode = "were"
    elif renderer == "conversational_recap":
        prefix = (
            f"Two friends checked a group of labeled storage places. {initial}{middle} "
            f"{agent}'s final report said that the {target_container} held "
        )
        mode = "held"
    elif renderer == "storage_log":
        prefix = f"{initial}{middle} Closing log - {target_container}: "
        mode = "log"
    else:
        prefix = (
            f"{initial}{middle} On the last inspection, the contents of "
            f"the {target_container} were "
        )
        mode = "were"

    prefix = re.sub(r"\s+", " ", prefix).strip() + " "
    completion, completion_items = _completion_for(
        final_world.items_in(target_container),
        mode=mode,
        rng=rng,
    )
    return RenderedScenario(
        text=f"{prefix}{completion}",
        input_prefix=prefix,
        completion=completion,
        completion_items=completion_items,
        completion_mode=mode,
    )


def _target_role(
    target_container: str,
    operations: Sequence[Operation],
) -> str:
    if not operations:
        return "zero_operation"
    last = operations[-1]
    if target_container == last.source:
        return "source"
    if target_container == last.destination:
        return "destination"
    if any(target_container in operation.touched_containers for operation in operations[:-1]):
        return "earlier_touched"
    return "untouched"


def _scenario_row(
    *,
    document_id: str,
    split: str,
    lexicon_name: str,
    lexicon: LexiconSplit,
    renderer: str,
    container_count: int,
    min_items: int,
    max_items: int,
    same_type_ratio: float,
    operation_count: int,
    operation_kind_weights: Mapping[Any, Any],
    rng: random.Random,
    forced_first_kind: OperationKind | None = None,
) -> dict[str, Any]:
    initial_world, world_style = _sample_world(
        lexicon=lexicon,
        container_count=container_count,
        min_items=min_items,
        max_items=max_items,
        same_type_ratio=same_type_ratio,
        rng=rng,
    )
    operations, final_world = _sample_operations(
        initial_world,
        operation_count=operation_count,
        operation_kind_weights=operation_kind_weights,
        object_pool=lexicon.objects,
        rng=rng,
        forced_first_kind=forced_first_kind,
    )
    target_container, _ = _choose_target(initial_world, operations, rng=rng)
    rendered = _render_scenario(
        initial_world=initial_world,
        final_world=final_world,
        operations=operations,
        target_container=target_container,
        renderer=renderer,
        rng=rng,
    )
    return {
        "document_id": document_id,
        "split": split,
        "lexicon_split": lexicon_name,
        "renderer": renderer,
        "world_style": world_style,
        "container_count": container_count,
        "operation_count": operation_count,
        "operation_kinds": [operation.kind for operation in operations],
        "target_container": target_container,
        "target_role": _target_role(target_container, operations),
        "initial_state": initial_world.as_dict(),
        "operations": [operation.to_dict() for operation in operations],
        "final_state": final_world.as_dict(),
        "input_prefix": rendered.input_prefix,
        "recap_completion": rendered.completion,
        "recap_items": list(rendered.completion_items),
        "completion_mode": rendered.completion_mode,
        "text": rendered.text,
        "word_count": len(WORD_RE.findall(rendered.text)),
    }


def _operations_from_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Operation, ...]:
    return tuple(
        Operation(
            kind=str(row["kind"]),  # type: ignore[arg-type]
            item=str(row["item"]),
            source=str(row["source"]) if row.get("source") is not None else None,
            destination=(
                str(row["destination"])
                if row.get("destination") is not None
                else None
            ),
        )
        for row in rows
    )


def _audit_row(row: Mapping[str, Any], lexicons: LexiconBundle) -> tuple[str, list[str]]:
    issues: list[str] = []
    try:
        initial = World.from_mapping(row["initial_state"])
        operations = _operations_from_rows(row["operations"])
        final = World.from_mapping(row["final_state"])
        replayed = apply_operations(initial, operations)
        if replayed != final:
            issues.append("operation replay does not match final_state")
        target = str(row["target_container"])
        if set(row["recap_items"]) != set(final.items_in(target)):
            issues.append("recap_items do not match the final target state")
    except (KeyError, TypeError, ValueError) as error:
        issues.append(f"invalid symbolic record: {error}")

    text = str(row.get("text", ""))
    if not text.strip():
        issues.append("text is empty")
    if "\n" in text or "\r" in text:
        issues.append("text is not a single line")
    if text != f"{row.get('input_prefix', '')}{row.get('recap_completion', '')}":
        issues.append("text is not input_prefix plus recap_completion")
    if int(row.get("word_count", -1)) != len(WORD_RE.findall(text)):
        issues.append("word_count is incorrect")

    lower_words = {word.lower() for word in LEXICAL_WORD_RE.findall(text)}
    overlap = lower_words & lexicons.benchmark_objects
    if overlap:
        issues.append(f"benchmark object terms appear in text: {sorted(overlap)}")
    for pattern in lexicons.forbidden_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            issues.append(f"forbidden surface pattern appears: {pattern}")

    return ("pass" if not issues else "fail"), issues


def _with_audit(row: dict[str, Any], lexicons: LexiconBundle) -> dict[str, Any]:
    status, issues = _audit_row(row, lexicons)
    return {**row, "audit_status": status, "audit_issues": issues}


def _world_signature(row: Mapping[str, Any]) -> str:
    # Split on the initial symbolic world before choosing operations, targets,
    # or renderers. Reusing a world with a different question would still leak
    # its bindings across the train/validation boundary.
    return json.dumps(row["initial_state"], sort_keys=True)


def _generate_train_documents(
    *,
    name: str,
    seed: int,
    pool_min_words: int,
    lexicons: LexiconBundle,
    world_config: Mapping[str, Any],
    operation_config: Mapping[str, Any],
    renderers: Sequence[str],
    used_signatures: set[str],
) -> tuple[dict[str, Any], ...]:
    count_weights = world_config.get("container_count_weights") or {3: 0.5, 4: 0.3, 5: 0.2}
    operation_count_weights = operation_config.get("count_weights") or {0: 0.5, 1: 0.5}
    kind_weights = operation_config.get("kind_weights") or {kind: 1.0 for kind in OPERATION_KINDS}
    min_items = int(world_config.get("min_items_per_container", 1))
    max_items = int(world_config.get("max_items_per_container", 3))
    same_type_ratio = float(world_config.get("same_type_ratio", 0.5))

    documents: list[dict[str, Any]] = []
    total_words = 0
    index = 0
    while total_words < pool_min_words:
        row_rng = random.Random(seed * 1_000_003 + index)
        container_count = int(_weighted_choice(count_weights, row_rng))
        operation_count = int(_weighted_choice(operation_count_weights, row_rng))
        forced_kind: OperationKind | None = None
        if operation_count == 1:
            forced_kind = OPERATION_KINDS[(index // 2) % len(OPERATION_KINDS)]
        renderer = renderers[index % len(renderers)]

        for attempt in range(100):
            attempt_rng = random.Random(seed * 1_000_003 + index * 101 + attempt)
            candidate = _scenario_row(
                document_id=f"{name}-train-{index:06d}",
                split="train",
                lexicon_name="train",
                lexicon=lexicons.train,
                renderer=renderer,
                container_count=container_count,
                min_items=min_items,
                max_items=max_items,
                same_type_ratio=same_type_ratio,
                operation_count=operation_count,
                operation_kind_weights=kind_weights,
                rng=attempt_rng,
                forced_first_kind=forced_kind,
            )
            signature = _world_signature(candidate)
            if signature not in used_signatures:
                used_signatures.add(signature)
                break
        else:
            raise RuntimeError("Could not sample a unique training scenario.")

        row = _with_audit(candidate, lexicons)
        if row["audit_status"] != "pass":
            raise RuntimeError(
                f"Generated training row failed audit: {row['audit_issues']}"
            )
        documents.append(row)
        total_words += int(row["word_count"])
        index += 1
    return tuple(documents)


def _validation_slice_specs(
    *,
    include_two_operations: bool,
) -> tuple[tuple[str, str, str, int | None], ...]:
    specs: list[tuple[str, str, str, int | None]] = [
        ("iid", "train", "train", None),
        ("lexical_ood", "validation", "train", None),
        ("template_ood", "train", "heldout", None),
    ]
    if include_two_operations:
        specs.append(("two_operations", "train", "train", 2))
    return tuple(specs)


def _validation_lexicon(bundle: LexiconBundle, name: str) -> LexiconSplit:
    return {
        "train": bundle.train,
        "validation": bundle.validation,
        "test": bundle.test,
    }[name]


def _generate_validation_items(
    *,
    name: str,
    seed: int,
    items_per_slice: int,
    include_two_operations: bool,
    lexicons: LexiconBundle,
    world_config: Mapping[str, Any],
    operation_config: Mapping[str, Any],
    train_renderers: Sequence[str],
    heldout_renderers: Sequence[str],
    used_signatures: set[str],
) -> dict[str, tuple[dict[str, Any], ...]]:
    count_weights = world_config.get("container_count_weights") or {3: 0.5, 4: 0.3, 5: 0.2}
    kind_weights = operation_config.get("kind_weights") or {kind: 1.0 for kind in OPERATION_KINDS}
    min_items = int(world_config.get("min_items_per_container", 1))
    max_items = int(world_config.get("max_items_per_container", 3))
    same_type_ratio = float(world_config.get("same_type_ratio", 0.5))

    output: dict[str, tuple[dict[str, Any], ...]] = {}
    for slice_index, (slice_name, lexicon_name, renderer_group, fixed_ops) in enumerate(
        _validation_slice_specs(include_two_operations=include_two_operations)
    ):
        lexicon = _validation_lexicon(lexicons, lexicon_name)
        renderer_pool = train_renderers if renderer_group == "train" else heldout_renderers
        rows: list[dict[str, Any]] = []
        for index in range(items_per_slice):
            base_seed = seed + (slice_index + 1) * 10_000_019 + index * 211
            row_rng = random.Random(base_seed)
            container_count = int(_weighted_choice(count_weights, row_rng))
            operation_count = fixed_ops if fixed_ops is not None else index % 2
            forced_kind: OperationKind | None = None
            if operation_count == 1:
                forced_kind = OPERATION_KINDS[(index // 2) % len(OPERATION_KINDS)]
            renderer = renderer_pool[index % len(renderer_pool)]

            for attempt in range(100):
                attempt_rng = random.Random(base_seed + attempt)
                candidate = _scenario_row(
                    document_id=f"{name}-{slice_name}-{index:06d}",
                    split=f"validation/{slice_name}",
                    lexicon_name=lexicon_name,
                    lexicon=lexicon,
                    renderer=renderer,
                    container_count=container_count,
                    min_items=min_items,
                    max_items=max_items,
                    same_type_ratio=same_type_ratio,
                    operation_count=operation_count,
                    operation_kind_weights=kind_weights,
                    rng=attempt_rng,
                    forced_first_kind=forced_kind,
                )
                signature = _world_signature(candidate)
                if signature not in used_signatures:
                    used_signatures.add(signature)
                    break
            else:
                raise RuntimeError(f"Could not sample a unique {slice_name} scenario.")

            audited = _with_audit(candidate, lexicons)
            if audited["audit_status"] != "pass":
                raise RuntimeError(
                    f"Generated validation row failed audit: {audited['audit_issues']}"
                )

            initial = World.from_mapping(audited["initial_state"])
            final = World.from_mapping(audited["final_state"])
            operations = _operations_from_rows(audited["operations"])
            option_rng = random.Random(base_seed + 100_000_007)
            distractors = build_distractor_states(
                initial_world=initial,
                final_world=final,
                operations=operations,
                target_container=str(audited["target_container"]),
                object_pool=lexicon.objects,
                rng=option_rng,
                count=4,
            )
            options = [str(audited["recap_completion"])]
            for distractor in distractors:
                completion, _ = _completion_for(
                    distractor.items,
                    mode=str(audited["completion_mode"]),
                    rng=option_rng,
                )
                options.append(completion)
            if len(set(options)) != 5:
                raise RuntimeError("Validation continuations are not unique.")

            rows.append(
                {
                    "example_id": index,
                    "sample_id": f"{slice_name}-{index:06d}",
                    "numops": operation_count,
                    "input_prefix": audited["input_prefix"],
                    "options": options,
                    "distractor_kinds": [distractor.kind for distractor in distractors],
                    "renderer": renderer,
                    "lexicon_split": lexicon_name,
                    "world_style": audited["world_style"],
                    "container_count": audited["container_count"],
                    "target_container": audited["target_container"],
                    "target_role": audited["target_role"],
                    "initial_state": audited["initial_state"],
                    "operations": audited["operations"],
                    "final_state": audited["final_state"],
                }
            )
        output[slice_name] = tuple(rows)
    return output


def _counter(values: Sequence[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _build_audit_report(
    *,
    name: str,
    documents: Sequence[Mapping[str, Any]],
    validation_items: Mapping[str, Sequence[Mapping[str, Any]]],
    lexicons: LexiconBundle,
) -> dict[str, Any]:
    statuses = _counter([row["audit_status"] for row in documents])
    failures = [
        {"document_id": row["document_id"], "issues": row["audit_issues"]}
        for row in documents
        if row["audit_status"] != "pass"
    ]
    return {
        "name": name,
        "counts": {
            "documents": len(documents),
            "words": sum(int(row["word_count"]) for row in documents),
            "pass": statuses.get("pass", 0),
            "fail": statuses.get("fail", 0),
            "validation_items": sum(len(rows) for rows in validation_items.values()),
        },
        "operation_count_counts": _counter([row["operation_count"] for row in documents]),
        "operation_kind_counts": _counter(
            [kind for row in documents for kind in row["operation_kinds"]]
        ),
        "container_count_counts": _counter([row["container_count"] for row in documents]),
        "renderer_counts": _counter([row["renderer"] for row in documents]),
        "world_style_counts": _counter([row["world_style"] for row in documents]),
        "target_role_counts": _counter([row["target_role"] for row in documents]),
        "validation_slice_counts": {
            slice_name: len(rows) for slice_name, rows in validation_items.items()
        },
        "lexical_checks": {
            "benchmark_object_overlap": [],
            "train_validation_object_overlap": sorted(
                _normalized(lexicons.train.objects)
                & _normalized(lexicons.validation.objects)
            ),
            "train_test_object_overlap": sorted(
                _normalized(lexicons.train.objects) & _normalized(lexicons.test.objects)
            ),
            "container_object_overlap": sorted(
                set().union(
                    _normalized(lexicons.train.container_types),
                    _normalized(lexicons.validation.container_types),
                    _normalized(lexicons.test.container_types),
                )
                & set().union(
                    _normalized(lexicons.train.objects),
                    _normalized(lexicons.validation.objects),
                    _normalized(lexicons.test.objects),
                )
            ),
        },
        "failures": failures,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _write_train(path: Path, documents: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for document in documents:
            if document["audit_status"] == "pass":
                f.write(str(document["text"]).strip())
                f.write("\n")


def build_container_state_corpus(
    mapping: Mapping[str, Any],
    *,
    base_dir: Path,
    config_path: Path | None = None,
) -> ContainerStateCorpusResult:
    """Build the training pool, held-out diagnostics, and provenance sidecars."""

    if not isinstance(mapping, Mapping):
        raise TypeError("Container-state config must be a mapping.")
    name = str(mapping["name"])
    seed = int(mapping.get("seed", 42))
    pool_min_words = int(mapping.get("pool_min_words", 130_000))
    if pool_min_words <= 0:
        raise ValueError("pool_min_words must be positive.")

    output_dir = resolve_path(mapping["output_dir"], base_dir=base_dir)
    lexicon_path = resolve_path(mapping["lexicons"], base_dir=base_dir)
    lexicons = _load_lexicons(lexicon_path)
    world_config = mapping.get("world") or {}
    operation_config = mapping.get("operations") or {}
    rendering_config = mapping.get("rendering") or {}
    validation_config = mapping.get("validation") or {}
    for field, value in (
        ("world", world_config),
        ("operations", operation_config),
        ("rendering", rendering_config),
        ("validation", validation_config),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{field} config must be a mapping.")

    train_renderers = tuple(
        str(value)
        for value in rendering_config.get("training_templates", TRAIN_RENDERERS)
    )
    heldout_renderers = tuple(
        str(value)
        for value in rendering_config.get("heldout_templates", HELDOUT_RENDERERS)
    )
    if not train_renderers or not heldout_renderers:
        raise ValueError("Training and held-out renderer lists must both be non-empty.")
    unknown_renderers = sorted(
        (set(train_renderers) | set(heldout_renderers)) - SUPPORTED_RENDERERS
    )
    if unknown_renderers:
        raise ValueError(f"Unknown renderers: {unknown_renderers}")
    overlap = set(train_renderers) & set(heldout_renderers)
    if overlap:
        raise ValueError(f"Training and held-out renderers overlap: {sorted(overlap)}")

    items_per_slice = int(validation_config.get("items_per_slice", 1_000))
    if items_per_slice <= 0:
        raise ValueError("validation.items_per_slice must be positive.")
    include_two_operations = bool(
        validation_config.get("include_two_operation_slice", True)
    )

    used_signatures: set[str] = set()
    documents = _generate_train_documents(
        name=name,
        seed=seed,
        pool_min_words=pool_min_words,
        lexicons=lexicons,
        world_config=world_config,
        operation_config=operation_config,
        renderers=train_renderers,
        used_signatures=used_signatures,
    )
    validation_items = _generate_validation_items(
        name=name,
        seed=seed,
        items_per_slice=items_per_slice,
        include_two_operations=include_two_operations,
        lexicons=lexicons,
        world_config=world_config,
        operation_config=operation_config,
        train_renderers=train_renderers,
        heldout_renderers=heldout_renderers,
        used_signatures=used_signatures,
    )
    audit_report = _build_audit_report(
        name=name,
        documents=documents,
        validation_items=validation_items,
        lexicons=lexicons,
    )
    if audit_report["counts"]["fail"]:
        raise RuntimeError("Container-state generation produced audit failures.")

    validation_dir = output_dir / "validation" / "entity_tracking"
    validation_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in validation_dir.glob("*.jsonl"):
        stale_path.unlink()

    outputs = {
        "train": output_dir / "train.txt",
        "documents": output_dir / "documents.jsonl",
        "audit_report": output_dir / "audit_report.json",
        "manifest": output_dir / "manifest.json",
        "config": output_dir / "container_state_config.yaml",
        "lexicons": output_dir / "lexicons.yaml",
        "validation_dir": validation_dir,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_train(outputs["train"], documents)
    _write_jsonl(outputs["documents"], documents)
    _write_json(outputs["audit_report"], audit_report)
    for slice_name, rows in validation_items.items():
        _write_jsonl(validation_dir / f"{slice_name}.jsonl", rows)
    if config_path is not None:
        shutil.copyfile(config_path, outputs["config"])
    shutil.copyfile(lexicon_path, outputs["lexicons"])

    manifest = {
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "deterministic_container_state_v1",
        "seed": seed,
        "pool_min_words": pool_min_words,
        "world": dict(world_config),
        "operations": dict(operation_config),
        "rendering": {
            "training_templates": list(train_renderers),
            "heldout_templates": list(heldout_renderers),
        },
        "validation": {
            "items_per_slice": items_per_slice,
            "include_two_operation_slice": include_two_operations,
            "slices": list(validation_items),
        },
        "counts": audit_report["counts"],
        "paths": {key: str(path) for key, path in outputs.items()},
    }
    _write_json(outputs["manifest"], manifest)
    return ContainerStateCorpusResult(
        name=name,
        output_dir=output_dir,
        documents=documents,
        validation_items=validation_items,
        audit_report=audit_report,
        outputs=outputs,
        manifest=manifest,
    )


__all__ = [
    "ContainerStateCorpusResult",
    "HELDOUT_RENDERERS",
    "TRAIN_RENDERERS",
    "build_container_state_corpus",
    "load_yaml_config",
]
