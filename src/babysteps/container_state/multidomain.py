"""Multi-domain paired counterfactual data for entity-tracking experiments."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from babysteps.container_state.generation import _sample_operations, _write_json, _write_jsonl
from babysteps.container_state.world import Operation, World, apply_operation
from babysteps.paths import resolve_path


TRAIN_STYLES = (
    "narrative",
    "observer_dialogue",
    "field_notes",
    "status_report",
    "procedural_recap",
    "question_answer_recount",
)
HELDOUT_STYLES = ("timestamped_log", "passive_inspection_report")
EVAL_SPLITS = (
    "iid",
    "lexical_ood",
    "template_ood",
    "domain_ood",
    "compositional_ood",
)
PAIR_KINDS = ("identity_move", "add", "identity_move", "move", "identity_move", "remove")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


@dataclass(frozen=True)
class DomainSpec:
    id: str
    setting: str
    agents: tuple[str, ...]
    train_containers: tuple[str, ...]
    validation_containers: tuple[str, ...]
    train_objects: tuple[str, ...]
    validation_objects: tuple[str, ...]
    verbs: Mapping[str, str]

    def containers(self, vocabulary: str) -> tuple[str, ...]:
        return self.train_containers if vocabulary == "train" else self.validation_containers

    def objects(self, vocabulary: str) -> tuple[str, ...]:
        return self.train_objects if vocabulary == "train" else self.validation_objects


@dataclass(frozen=True)
class MultidomainCorpusResult:
    name: str
    output_dir: Path
    train_pairs: tuple[dict[str, Any], ...]
    eval_pairs: Mapping[str, tuple[dict[str, Any], ...]]
    audit_report: dict[str, Any]
    outputs: Mapping[str, Path]


def _strings(value: Any, field: str, minimum: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list.")
    result = tuple(str(item).strip() for item in value)
    if len(result) < minimum or any(not item for item in result):
        raise ValueError(f"{field} must contain at least {minimum} nonempty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} contains duplicate values.")
    return result


def _parse_domain(value: Mapping[str, Any]) -> DomainSpec:
    containers = value.get("containers") or {}
    objects = value.get("objects") or {}
    verbs = value.get("verbs") or {}
    for kind in ("add", "remove", "move"):
        if not str(verbs.get(kind, "")).strip():
            raise ValueError(f"Domain {value.get('id')!r} is missing verb {kind!r}.")
    return DomainSpec(
        id=str(value["id"]).strip(),
        setting=str(value["setting"]).strip(),
        agents=_strings(value.get("agents"), "agents", 2),
        train_containers=_strings(containers.get("train"), "containers.train", 4),
        validation_containers=_strings(
            containers.get("validation"), "containers.validation", 4
        ),
        train_objects=_strings(objects.get("train"), "objects.train", 10),
        validation_objects=_strings(
            objects.get("validation"), "objects.validation", 10
        ),
        verbs={kind: str(verbs[kind]).strip() for kind in ("add", "remove", "move")},
    )


def load_domain_catalog(path: str | Path) -> tuple[tuple[DomainSpec, ...], tuple[DomainSpec, ...]]:
    with Path(path).open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError("The multidomain catalog must be a YAML mapping.")
    training = tuple(_parse_domain(item) for item in value.get("training_domains", []))
    heldout = tuple(_parse_domain(item) for item in value.get("heldout_domains", []))
    if len(training) < 10 or len(heldout) < 3:
        raise ValueError("The catalog requires at least 10 training and 3 held-out domains.")
    ids = [domain.id for domain in (*training, *heldout)]
    if len(ids) != len(set(ids)):
        raise ValueError("Domain ids must be unique across training and held-out domains.")
    return training, heldout


def _article(item: str) -> str:
    return "an" if item[:1].lower() in "aeiou" else "a"


def render_items(items: Iterable[str]) -> str:
    values = tuple(items)
    if not values:
        raise ValueError("Multidomain targets must be nonempty.")
    phrases = [f"{_article(item)} {item}" for item in values]
    if len(phrases) == 1:
        return phrases[0] + "."
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}."
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}."


def _state_clause(world: World) -> str:
    return "; ".join(
        f"the {container} contained {render_items(items).removesuffix('.')}"
        for container, items in world.contents
    )


def _operation_sentence(operation: Operation, domain: DomainSpec, agent: str) -> str:
    item = f"{_article(operation.item)} {operation.item}"
    if operation.kind == "add":
        assert operation.destination is not None
        return f"{agent} {domain.verbs['add']} {item} to the {operation.destination}."
    if operation.kind == "remove":
        assert operation.source is not None
        return f"{agent} {domain.verbs['remove']} {item} from the {operation.source}."
    assert operation.source is not None and operation.destination is not None
    return (
        f"{agent} {domain.verbs['move']} {item} from the {operation.source} "
        f"to the {operation.destination}."
    )


def _render_context(
    *,
    initial_world: World,
    operations: Sequence[Operation],
    target: str,
    domain: DomainSpec,
    style: str,
    seed: int,
) -> str:
    rng = random.Random(seed)
    agents = [domain.agents[index % len(domain.agents)] for index in range(len(operations))]
    op_sentences = [
        _operation_sentence(operation, domain, agents[index])
        for index, operation in enumerate(operations)
    ]
    initial = _state_clause(initial_world)
    updates = " ".join(op_sentences) if op_sentences else "No items were moved during the check."

    if style == "narrative":
        return (
            f"At {domain.setting}, {rng.choice(domain.agents)} reviewed several locations. "
            f"Initially, {initial}. {updates} At the end, the {target} contained "
        )
    if style == "observer_dialogue":
        return (
            f"\"Let's verify everything at {domain.setting},\" said {rng.choice(domain.agents)}. "
            f"\"At first, {initial}.\" {updates} \"What is in the {target} now?\" Answer: "
        )
    if style == "field_notes":
        return (
            f"Field notes from {domain.setting}. Starting inventory: {initial}. "
            f"Observed updates: {updates} Final check — {target}: "
        )
    if style == "status_report":
        return (
            f"Status report for {domain.setting}. Initial assignments: {initial}. "
            f"Recorded changes: {updates} Current contents of the {target}: "
        )
    if style == "procedural_recap":
        return (
            f"During the inventory at {domain.setting}, the starting arrangement was as follows: "
            f"{initial}. The team then completed these steps: {updates} To finish, the {target} held "
        )
    if style == "question_answer_recount":
        return (
            f"Q: What was the starting inventory at {domain.setting}? A: {initial}. "
            f"Q: What changed? A: {updates} Q: What does the {target} contain now? A: "
        )
    if style == "timestamped_log":
        timestamped = " ".join(
            f"[{8 + index:02d}:{5 * (index + 1):02d}] {sentence}"
            for index, sentence in enumerate(op_sentences)
        )
        if not timestamped:
            timestamped = "[08:05] No transfer was recorded."
        return (
            f"[08:00] Opening log, {domain.setting}: {initial}. {timestamped} "
            f"[09:00] {target} inventory: "
        )
    if style == "passive_inspection_report":
        passive = []
        for operation in operations:
            item = f"{_article(operation.item)} {operation.item}"
            if operation.kind == "add":
                passive.append(f"{item} was added to the {operation.destination}.")
            elif operation.kind == "remove":
                passive.append(f"{item} was removed from the {operation.source}.")
            else:
                passive.append(
                    f"{item} was moved from the {operation.source} to the {operation.destination}."
                )
        update_text = " ".join(passive) or "No change was observed."
        return (
            f"During an inspection of {domain.setting}, the initial arrangement was recorded: "
            f"{initial}. {update_text} On final inspection, the {target} contained "
        )
    raise ValueError(f"Unsupported multidomain style: {style!r}.")


def _sample_world(
    domain: DomainSpec, vocabulary: str, *, container_count: int, rng: random.Random
) -> World:
    containers = rng.sample(list(domain.containers(vocabulary)), container_count)
    objects = rng.sample(list(domain.objects(vocabulary)), 2 * container_count)
    return World.from_mapping(
        {
            container: objects[2 * index : 2 * index + 2]
            for index, container in enumerate(containers)
        }
    )


def _counterfactual_operations(
    world: World, *, target: str, kind: str, object_pool: Sequence[str], rng: random.Random
) -> tuple[Operation, Operation]:
    others = [container for container in world.containers if container != target]
    if kind == "identity_move":
        sources = [container for container in others if world.items_in(container)]
        if len(sources) < 2:
            raise ValueError("Identity-changing pairs require two populated sources.")
        source1, source2 = rng.sample(sources, 2)
        return (
            Operation("move", rng.choice(world.items_in(source1)), source1, target),
            Operation("move", rng.choice(world.items_in(source2)), source2, target),
        )
    if kind == "add":
        available = [item for item in object_pool if world.locate(item) is None]
        if not available:
            raise ValueError("No unused item is available for an add pair.")
        item = rng.choice(available)
        return (
            Operation("add", item, destination=target),
            Operation("add", item, destination=rng.choice(others)),
        )
    if kind == "move":
        sources = [container for container in others if world.items_in(container)]
        if not sources:
            raise ValueError("No populated source is available for a move pair.")
        source = rng.choice(sources)
        destinations = [container for container in others if container != source]
        if not destinations:
            raise ValueError("No alternate destination is available for a move pair.")
        item = rng.choice(world.items_in(source))
        return (
            Operation("move", item, source, target),
            Operation("move", item, source, rng.choice(destinations)),
        )
    if kind == "remove":
        if len(world.items_in(target)) < 2:
            raise ValueError("Removal must leave the target nonempty.")
        sources = [container for container in others if world.items_in(container)]
        if not sources:
            raise ValueError("No alternate source is available for a removal pair.")
        source2 = rng.choice(sources)
        return (
            Operation("remove", rng.choice(world.items_in(target)), source=target),
            Operation("remove", rng.choice(world.items_in(source2)), source=source2),
        )
    raise ValueError(f"Unknown counterfactual kind: {kind!r}.")


def _build_pair(
    *,
    pair_id: str,
    split: str,
    seed: int,
    domain: DomainSpec,
    vocabulary: str,
    style: str,
    counterfactual_kind: str,
    common_operation_count: int,
    container_count: int,
    swap_sides: bool,
) -> dict[str, Any]:
    object_pool = domain.objects(vocabulary)
    for attempt in range(1_000):
        rng = random.Random(seed + attempt)
        initial_world = _sample_world(
            domain, vocabulary, container_count=container_count, rng=rng
        )
        try:
            common_operations, common_world = _sample_operations(
                initial_world,
                operation_count=common_operation_count,
                operation_kind_weights={"add": 1.0, "remove": 1.0, "move": 1.0},
                object_pool=object_pool,
                rng=rng,
            )
            candidates = [container for container in common_world.containers if common_world.items_in(container)]
            if counterfactual_kind == "remove":
                candidates = [container for container in candidates if len(common_world.items_in(container)) >= 2]
            if not candidates:
                continue
            target = rng.choice(candidates)
            operation1, operation2 = _counterfactual_operations(
                common_world,
                target=target,
                kind=counterfactual_kind,
                object_pool=object_pool,
                rng=rng,
            )
            if swap_sides:
                operation1, operation2 = operation2, operation1
            final1 = apply_operation(common_world, operation1)
            final2 = apply_operation(common_world, operation2)
        except ValueError:
            continue
        items1 = final1.items_in(target)
        items2 = final2.items_in(target)
        if not items1 or not items2 or frozenset(items1) == frozenset(items2):
            continue
        operations1 = (*common_operations, operation1)
        operations2 = (*common_operations, operation2)
        render_seed = seed + 97_003
        context1 = _render_context(
            initial_world=initial_world,
            operations=operations1,
            target=target,
            domain=domain,
            style=style,
            seed=render_seed,
        )
        context2 = _render_context(
            initial_world=initial_world,
            operations=operations2,
            target=target,
            domain=domain,
            style=style,
            seed=render_seed,
        )
        target1 = render_items(items1)
        target2 = render_items(items2)
        if context1 == context2 or target1 == target2:
            continue
        return {
            "pair_id": pair_id,
            "split": split,
            "domain": domain.id,
            "setting": domain.setting,
            "vocabulary": vocabulary,
            "renderer": style,
            "container_count": container_count,
            "common_operation_count": common_operation_count,
            "operation_count": len(operations1),
            "counterfactual_kind": counterfactual_kind,
            "swapped_sides": swap_sides,
            "equal_cardinality": len(items1) == len(items2),
            "target_container": target,
            "object_pool": list(object_pool),
            "initial_state": initial_world.as_dict(),
            "common_operations": [operation.to_dict() for operation in common_operations],
            "counterfactual_operation1": operation1.to_dict(),
            "counterfactual_operation2": operation2.to_dict(),
            "final_state1": final1.as_dict(),
            "final_state2": final2.as_dict(),
            "Context1": context1,
            "Context2": context2,
            "Target1": target1,
            "Target2": target2,
            "target_items1": list(items1),
            "target_items2": list(items2),
            "text1": context1 + target1,
            "text2": context2 + target2,
            "word_count": len(WORD_RE.findall(context1 + target1))
            + len(WORD_RE.findall(context2 + target2)),
            "audit_status": "pass",
        }
    raise RuntimeError(f"Could not construct pair {pair_id} after 1,000 attempts.")


def _balanced(values: Sequence[Any], count: int, rng: random.Random) -> list[Any]:
    result = [values[index % len(values)] for index in range(count)]
    rng.shuffle(result)
    return result


def _generate_pairs(
    *,
    name: str,
    split: str,
    count: int,
    seed: int,
    domains: Sequence[DomainSpec],
    vocabulary: str,
    styles: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    domain_schedule = _balanced(domains, count, random.Random(seed + 11))
    style_schedule = _balanced(styles, count, random.Random(seed + 23))
    kind_schedule = _balanced(PAIR_KINDS, count, random.Random(seed + 37))
    common_schedule = _balanced((0, 1, 2), count, random.Random(seed + 41))
    container_schedule = _balanced((3, 4), count, random.Random(seed + 53))
    side_schedule = _balanced((False, True), count, random.Random(seed + 59))
    return tuple(
        _build_pair(
            pair_id=f"{name}-{split}-{index:06d}",
            split=split,
            seed=seed * 1_000_003 + index * 10_007,
            domain=domain_schedule[index],
            vocabulary=vocabulary,
            style=style_schedule[index],
            counterfactual_kind=kind_schedule[index],
            common_operation_count=common_schedule[index],
            container_count=container_schedule[index],
            swap_sides=side_schedule[index],
        )
        for index in range(count)
    )


def _same_cardinality_options(pair: Mapping[str, Any], side: int) -> list[str]:
    correct = tuple(str(item) for item in pair[f"target_items{side}"])
    final_state = pair[f"final_state{side}"]
    pool = list(dict.fromkeys(str(item) for item in pair["object_pool"]))
    pool.extend(
        item
        for items in final_state.values()
        for item in items
        if item not in pool
    )
    pool.extend(
        item
        for item in (*pair["target_items1"], *pair["target_items2"])
        if item not in pool
    )
    candidates: list[tuple[str, ...]] = []
    seen = {frozenset(correct)}
    for replacement in pool:
        if replacement in correct:
            continue
        for index in range(len(correct)):
            changed = list(correct)
            changed[index] = replacement
            key = frozenset(changed)
            if len(key) != len(correct) or key in seen:
                continue
            seen.add(key)
            candidates.append(tuple(changed))
            if len(candidates) == 4:
                return [render_items(correct), *(render_items(items) for items in candidates)]
    raise RuntimeError(f"Could not build four matched distractors for {pair['pair_id']} side {side}.")


def _entity_tracking_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        for side in (1, 2):
            rows.append(
                {
                    "example_id": f"{pair['pair_id']}-side{side}",
                    "sample_id": f"{pair['pair_id']}-side{side}",
                    "pair_id": pair["pair_id"],
                    "pair_side": side,
                    "domain": pair["domain"],
                    "renderer": pair["renderer"],
                    "lexicon_split": pair["vocabulary"],
                    "numops": pair["operation_count"],
                    "input_prefix": pair[f"Context{side}"],
                    "options": _same_cardinality_options(pair, side),
                    "option_item_counts": [len(pair[f"target_items{side}"])] * 5,
                    "target_index": 0,
                    "target_container": pair["target_container"],
                    "distractor_kinds": ["same_size_object_swap"] * 4,
                }
            )
    return rows


def _completion_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        for side, negative in ((1, 2), (2, 1)):
            rows.append(
                {
                    "rank_objective": "completion_rank",
                    "item_id": f"{pair['pair_id']}-side{side}",
                    "template_family": pair["renderer"],
                    "domain": pair["domain"],
                    "context": pair[f"Context{side}"],
                    "completion": pair[f"Target{side}"],
                    "hard_negative": pair[f"Target{negative}"],
                }
            )
    return rows


def _counterfactual_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank_objective": "counterfactual_rank",
            "item_id": pair["pair_id"],
            "template_family": pair["renderer"],
            "domain": pair["domain"],
            "Context1": pair["Context1"],
            "Context2": pair["Context2"],
            "Target1": pair["Target1"],
            "Target2": pair["Target2"],
        }
        for pair in pairs
    ]


def _tokens(text: str) -> frozenset[str]:
    return frozenset(token.lower() for token in WORD_RE.findall(text))


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return 0.5
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def surface_branch_auc(pairs: Sequence[Mapping[str, Any]], *, seed: int) -> float:
    """Predict branch index from bag-of-words only; high AUC indicates side leakage."""

    # Keep both branches of a pair on the same side of the classifier split.
    # Otherwise near-duplicate paired contexts leak lexical identities between
    # classifier train/eval and measure memorization rather than branch bias.
    shuffled_pairs = list(pairs)
    random.Random(seed).shuffle(shuffled_pairs)
    split_at = max(1, int(0.8 * len(shuffled_pairs)))
    train_pairs = shuffled_pairs[:split_at]
    evaluation_pairs = shuffled_pairs[split_at:]
    train = [
        (pair[f"Context{side}"], side - 1)
        for pair in train_pairs
        for side in (1, 2)
    ]
    evaluation = [
        (pair[f"Context{side}"], side - 1)
        for pair in evaluation_pairs
        for side in (1, 2)
    ]
    class_counts = Counter(label for _, label in train)
    token_counts = {0: Counter(), 1: Counter()}
    total_tokens = Counter()
    vocabulary: set[str] = set()
    for text, label in train:
        values = _tokens(text)
        token_counts[label].update(values)
        total_tokens[label] += len(values)
        vocabulary.update(values)
    denominator = {
        label: total_tokens[label] + len(vocabulary) for label in (0, 1)
    }
    log_prior = {
        label: math.log((class_counts[label] + 1) / (len(train) + 2))
        for label in (0, 1)
    }
    labels: list[int] = []
    scores: list[float] = []
    for text, label in evaluation:
        values = _tokens(text)
        class_scores = []
        for candidate in (0, 1):
            score = log_prior[candidate]
            for token in values:
                score += math.log(
                    (token_counts[candidate][token] + 1) / denominator[candidate]
                )
            class_scores.append(score)
        labels.append(label)
        scores.append(class_scores[1] - class_scores[0])
    raw = _auc(labels, scores)
    return max(raw, 1.0 - raw)


def _normalized(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def _count_values(pairs: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(pair[field]) for pair in pairs).items()))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_multidomain_corpus(
    mapping: Mapping[str, Any], *, base_dir: Path, config_path: Path | None = None
) -> MultidomainCorpusResult:
    name = str(mapping["name"])
    seed = int(mapping.get("seed", 42))
    output_dir = resolve_path(mapping["output_dir"], base_dir=base_dir)
    catalog_path = resolve_path(mapping["catalog"], base_dir=base_dir)
    training_domains, heldout_domains = load_domain_catalog(catalog_path)
    train_count = int(mapping.get("train_pairs", 4_800))
    eval_count = int(mapping.get("eval_pairs_per_split", 1_200))
    threshold = float(mapping.get("surface_classifier_max_auc", 0.55))
    enforce_gate = bool(mapping.get("enforce_surface_gate", True))
    if train_count <= 0 or eval_count <= 0:
        raise ValueError("train_pairs and eval_pairs_per_split must be positive.")

    train = _generate_pairs(
        name=name,
        split="train",
        count=train_count,
        seed=seed,
        domains=training_domains,
        vocabulary="train",
        styles=TRAIN_STYLES,
    )
    split_design = {
        "iid": (training_domains, "train", TRAIN_STYLES),
        "lexical_ood": (training_domains, "validation", TRAIN_STYLES),
        "template_ood": (training_domains, "train", HELDOUT_STYLES),
        "domain_ood": (heldout_domains, "train", TRAIN_STYLES),
        "compositional_ood": (heldout_domains, "validation", HELDOUT_STYLES),
    }
    evaluation = {
        split: _generate_pairs(
            name=name,
            split=split,
            count=eval_count,
            seed=seed + 1_000_003 * (index + 1),
            domains=domains,
            vocabulary=vocabulary,
            styles=styles,
        )
        for index, (split, (domains, vocabulary, styles)) in enumerate(split_design.items())
    }

    all_splits = {"train": train, **evaluation}
    seen: dict[str, str] = {}
    collisions: list[dict[str, str]] = []
    within_duplicates: dict[str, int] = {}
    for split, pairs in all_splits.items():
        local: set[str] = set()
        duplicates = 0
        for pair in pairs:
            for side in (1, 2):
                normalized = _normalized(str(pair[f"text{side}"]))
                if normalized in local:
                    duplicates += 1
                local.add(normalized)
                previous = seen.get(normalized)
                if previous is not None and previous != split:
                    collisions.append({"first_split": previous, "second_split": split})
                else:
                    seen[normalized] = split
        within_duplicates[split] = duplicates
    if collisions or any(within_duplicates.values()):
        raise RuntimeError("Normalized text duplicates were detected within/across splits.")

    branch_auc = surface_branch_auc(train, seed=seed + 71)
    gate_passed = branch_auc <= threshold
    if enforce_gate and not gate_passed:
        raise RuntimeError(
            f"Surface branch classifier AUC {branch_auc:.4f} exceeds gate {threshold:.4f}."
        )

    pairs_dir = output_dir / "pairs"
    entity_dir = output_dir / "entity_tracking"
    auxiliary_dir = output_dir / "auxiliary"
    outputs: dict[str, Path] = {
        "train": output_dir / "train.txt",
        "train_pairs": pairs_dir / "train.jsonl",
        "train_completion_rank": auxiliary_dir / "train_completion_rank.jsonl",
        "train_counterfactual_rank": auxiliary_dir / "train_counterfactual_rank.jsonl",
        "audit_report": output_dir / "audit_report.json",
        "manifest": output_dir / "manifest.json",
        "config": output_dir / "multidomain_config.yaml",
        "catalog": output_dir / "multidomain_catalog.yaml",
    }
    for split in EVAL_SPLITS:
        outputs[f"{split}_pairs"] = pairs_dir / f"{split}.jsonl"
        outputs[f"{split}_entity_tracking"] = entity_dir / f"{split}.jsonl"
        outputs[f"{split}_completion_rank"] = auxiliary_dir / f"{split}_completion_rank.jsonl"
        outputs[f"{split}_counterfactual_rank"] = auxiliary_dir / f"{split}_counterfactual_rank.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)
    with outputs["train"].open("w", encoding="utf-8") as f:
        for pair in train:
            f.write(str(pair["text1"]).strip() + "\n")
            f.write(str(pair["text2"]).strip() + "\n")
    _write_jsonl(outputs["train_pairs"], train)
    _write_jsonl(outputs["train_completion_rank"], _completion_rows(train))
    _write_jsonl(outputs["train_counterfactual_rank"], _counterfactual_rows(train))
    for split, pairs in evaluation.items():
        _write_jsonl(outputs[f"{split}_pairs"], pairs)
        _write_jsonl(outputs[f"{split}_entity_tracking"], _entity_tracking_rows(pairs))
        _write_jsonl(outputs[f"{split}_completion_rank"], _completion_rows(pairs))
        _write_jsonl(outputs[f"{split}_counterfactual_rank"], _counterfactual_rows(pairs))

    train_item_surfaces = {
        item.lower()
        for domain in training_domains
        for item in (*domain.train_containers, *domain.train_objects)
    }
    validation_item_surfaces = {
        item.lower()
        for domain in training_domains
        for item in (*domain.validation_containers, *domain.validation_objects)
    }
    audit = {
        "name": name,
        "counts": {
            "train_pairs": len(train),
            "train_documents": 2 * len(train),
            "train_words": sum(int(pair["word_count"]) for pair in train),
            "eval_pairs_per_split": {split: len(pairs) for split, pairs in evaluation.items()},
            "eval_items_per_split": {split: 2 * len(pairs) for split, pairs in evaluation.items()},
            "empty_targets": 0,
            "identical_targets": 0,
        },
        "training_domain_counts": _count_values(train, "domain"),
        "training_renderer_counts": _count_values(train, "renderer"),
        "training_counterfactual_kind_counts": _count_values(train, "counterfactual_kind"),
        "training_equal_cardinality_counts": _count_values(train, "equal_cardinality"),
        "training_common_operation_counts": _count_values(train, "common_operation_count"),
        "split_design": {
            split: {
                "domains": "training" if domains is training_domains else "heldout",
                "vocabulary": vocabulary,
                "styles": list(styles),
            }
            for split, (domains, vocabulary, styles) in split_design.items()
        },
        "leakage": {
            "normalized_cross_split_collisions": len(collisions),
            "normalized_within_split_duplicates": within_duplicates,
            "train_validation_surface_overlap": sorted(
                train_item_surfaces & validation_item_surfaces
            ),
            "training_heldout_domain_overlap": sorted(
                {domain.id for domain in training_domains}
                & {domain.id for domain in heldout_domains}
            ),
        },
        "surface_classifier": {
            "task": "predict counterfactual branch index from context bag-of-words",
            "roc_auc": branch_auc,
            "max_allowed_auc": threshold,
            "passed": gate_passed,
        },
    }
    if audit["leakage"]["train_validation_surface_overlap"]:
        raise RuntimeError("Train/validation catalog surfaces overlap.")
    _write_json(outputs["audit_report"], audit)
    manifest = {
        "name": name,
        "version": 3,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "design": "multi-domain paired counterfactual entity tracking",
        "outputs": {key: str(path) for key, path in outputs.items()},
        "audit": audit,
    }
    _write_json(outputs["manifest"], manifest)
    if config_path is None:
        with outputs["config"].open("w", encoding="utf-8") as f:
            yaml.safe_dump(dict(mapping), f, sort_keys=False)
    else:
        shutil.copy2(config_path, outputs["config"])
    shutil.copy2(catalog_path, outputs["catalog"])
    manifest["sha256"] = {
        key: _hash_file(path)
        for key, path in outputs.items()
        if key not in {"manifest"} and path.exists()
    }
    _write_json(outputs["manifest"], manifest)
    return MultidomainCorpusResult(name, output_dir, train, evaluation, audit, outputs)


__all__ = [
    "EVAL_SPLITS",
    "HELDOUT_STYLES",
    "TRAIN_STYLES",
    "DomainSpec",
    "MultidomainCorpusResult",
    "build_multidomain_corpus",
    "load_domain_catalog",
    "render_items",
    "surface_branch_auc",
]
