"""Generate paired counterfactual container-state training examples."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from babysteps.container_state.generation import (
    HELDOUT_RENDERERS,
    OPERATION_KINDS,
    TRAIN_RENDERERS,
    WORD_RE,
    LexiconBundle,
    LexiconSplit,
    _load_lexicons,
    _render_scenario,
    _sample_operations,
    _sample_world,
    _weighted_choice,
    _write_json,
    _write_jsonl,
)
from babysteps.container_state.world import Operation, World, apply_operation
from babysteps.paths import resolve_path


@dataclass(frozen=True)
class CounterfactualCorpusResult:
    """Generated ordinary-LM text, rank pairs, and audit metadata."""

    name: str
    output_dir: Path
    train_pairs: tuple[dict[str, Any], ...]
    eval_pairs: tuple[dict[str, Any], ...]
    audit_report: dict[str, Any]
    outputs: dict[str, Path]


def _operation_signature(operation: Operation) -> tuple[str, str, str, str]:
    return (
        operation.kind,
        operation.item,
        operation.source or "",
        operation.destination or "",
    )


def _make_counterfactual_operations(
    world: World,
    *,
    target: str,
    kind: str,
    object_pool: Sequence[str],
    rng: random.Random,
) -> tuple[Operation, Operation]:
    other_containers = [name for name in world.containers if name != target]
    if kind == "add":
        available = [item for item in object_pool if world.locate(item) is None]
        if not available:
            raise ValueError("No unused object is available for an add pair.")
        item = rng.choice(available)
        return (
            Operation(kind="add", item=item, destination=target),
            Operation(
                kind="add",
                item=item,
                destination=rng.choice(other_containers),
            ),
        )

    if kind == "move":
        sources = [name for name in other_containers if world.items_in(name)]
        rng.shuffle(sources)
        for source in sources:
            alternate_destinations = [
                name
                for name in other_containers
                if name != source
            ]
            if alternate_destinations:
                item = rng.choice(world.items_in(source))
                return (
                    Operation(
                        kind="move",
                        item=item,
                        source=source,
                        destination=target,
                    ),
                    Operation(
                        kind="move",
                        item=item,
                        source=source,
                        destination=rng.choice(alternate_destinations),
                    ),
                )
        raise ValueError("No source/alternate destination is available for a move pair.")

    if kind == "remove":
        if len(world.items_in(target)) < 2:
            raise ValueError("The target must retain a nonempty answer after removal.")
        alternate_sources = [
            name for name in other_containers if world.items_in(name)
        ]
        if not alternate_sources:
            raise ValueError("No alternate removal source is available.")
        alternate_source = rng.choice(alternate_sources)
        return (
            Operation(
                kind="remove",
                item=rng.choice(world.items_in(target)),
                source=target,
            ),
            Operation(
                kind="remove",
                item=rng.choice(world.items_in(alternate_source)),
                source=alternate_source,
            ),
        )

    raise ValueError(f"Unsupported counterfactual operation kind: {kind!r}.")


def _build_pair(
    *,
    pair_id: str,
    seed: int,
    lexicon_name: str,
    lexicon: LexiconSplit,
    renderer: str,
    container_count: int,
    min_items: int,
    max_items: int,
    same_type_ratio: float,
    common_operation_count: int,
    common_kind_weights: Mapping[Any, Any],
    counterfactual_kind: str,
) -> dict[str, Any]:
    for attempt in range(1_000):
        rng = random.Random(seed + attempt)
        initial_world, world_style = _sample_world(
            lexicon=lexicon,
            container_count=container_count,
            min_items=min_items,
            max_items=max_items,
            same_type_ratio=same_type_ratio,
            rng=rng,
        )
        target = rng.choice(initial_world.containers)
        try:
            common_operations, common_world = _sample_operations(
                initial_world,
                operation_count=common_operation_count,
                operation_kind_weights=common_kind_weights,
                object_pool=lexicon.objects,
                rng=rng,
            )
            if not common_world.items_in(target):
                continue
            operation1, operation2 = _make_counterfactual_operations(
                common_world,
                target=target,
                kind=counterfactual_kind,
                object_pool=lexicon.objects,
                rng=rng,
            )
            final1 = apply_operation(common_world, operation1)
            final2 = apply_operation(common_world, operation2)
        except ValueError:
            continue

        items1 = set(final1.items_in(target))
        items2 = set(final2.items_in(target))
        if not items1 or not items2 or items1 == items2:
            continue

        operations1 = (*common_operations, operation1)
        operations2 = (*common_operations, operation2)
        render_seed = seed + 10_000_019
        rendered1 = _render_scenario(
            initial_world=initial_world,
            final_world=final1,
            operations=operations1,
            target_container=target,
            renderer=renderer,
            rng=random.Random(render_seed),
        )
        rendered2 = _render_scenario(
            initial_world=initial_world,
            final_world=final2,
            operations=operations2,
            target_container=target,
            renderer=renderer,
            rng=random.Random(render_seed),
        )
        if (
            not rendered1.completion_items
            or not rendered2.completion_items
            or rendered1.completion == rendered2.completion
            or rendered1.input_prefix == rendered2.input_prefix
        ):
            continue

        return {
            "pair_id": pair_id,
            "lexicon_split": lexicon_name,
            "renderer": renderer,
            "world_style": world_style,
            "container_count": container_count,
            "operation_count": len(operations1),
            "common_operation_count": common_operation_count,
            "counterfactual_kind": counterfactual_kind,
            "target_container": target,
            "initial_state": initial_world.as_dict(),
            "common_operations": [operation.to_dict() for operation in common_operations],
            "counterfactual_operation1": operation1.to_dict(),
            "counterfactual_operation2": operation2.to_dict(),
            "final_state1": final1.as_dict(),
            "final_state2": final2.as_dict(),
            "Context1": rendered1.input_prefix,
            "Context2": rendered2.input_prefix,
            "Target1": rendered1.completion,
            "Target2": rendered2.completion,
            "target_items1": list(rendered1.completion_items),
            "target_items2": list(rendered2.completion_items),
            "text1": rendered1.text,
            "text2": rendered2.text,
            "word_count": len(WORD_RE.findall(rendered1.text))
            + len(WORD_RE.findall(rendered2.text)),
            "audit_status": "pass",
        }
    raise RuntimeError(f"Could not construct a valid counterfactual pair: {pair_id}")


def _generate_pairs(
    *,
    name: str,
    split: str,
    seed: int,
    lexicon_name: str,
    lexicon: LexiconSplit,
    renderers: Sequence[str],
    count_weights: Mapping[Any, Any],
    min_items: int,
    max_items: int,
    same_type_ratio: float,
    common_count_weights: Mapping[Any, Any],
    common_kind_weights: Mapping[Any, Any],
    target_words: int | None = None,
    target_pairs: int | None = None,
) -> tuple[dict[str, Any], ...]:
    if (target_words is None) == (target_pairs is None):
        raise ValueError("Set exactly one of target_words or target_pairs.")
    pairs: list[dict[str, Any]] = []
    words = 0
    index = 0
    while (target_words is not None and words < target_words) or (
        target_pairs is not None and index < target_pairs
    ):
        pair_seed = seed * 1_000_003 + index * 10_007
        rng = random.Random(pair_seed)
        pair = _build_pair(
            pair_id=f"{name}-{split}-{index:06d}",
            seed=pair_seed,
            lexicon_name=lexicon_name,
            lexicon=lexicon,
            renderer=renderers[index % len(renderers)],
            container_count=int(_weighted_choice(count_weights, rng)),
            min_items=min_items,
            max_items=max_items,
            same_type_ratio=same_type_ratio,
            common_operation_count=int(_weighted_choice(common_count_weights, rng)),
            common_kind_weights=common_kind_weights,
            counterfactual_kind=OPERATION_KINDS[index % len(OPERATION_KINDS)],
        )
        pairs.append(pair)
        words += int(pair["word_count"])
        index += 1
    return tuple(pairs)


def _completion_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        for side, correct, negative in ((1, 1, 2), (2, 2, 1)):
            rows.append(
                {
                    "rank_objective": "completion_rank",
                    "item_id": f"{pair['pair_id']}-side{side}",
                    "template_family": pair["renderer"],
                    "context": pair[f"Context{side}"],
                    "completion": pair[f"Target{correct}"],
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
            "Context1": pair["Context1"],
            "Context2": pair["Context2"],
            "Target1": pair["Target1"],
            "Target2": pair["Target2"],
        }
        for pair in pairs
    ]


def _write_training_text(path: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(f"{str(pair['text1']).strip()}\n")
            f.write(f"{str(pair['text2']).strip()}\n")


def _counts(pairs: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(pair[field]) for pair in pairs).items()))


def build_counterfactual_corpus(
    mapping: Mapping[str, Any],
    *,
    base_dir: Path,
    config_path: Path | None = None,
) -> CounterfactualCorpusResult:
    """Build ordinary text and two rank views from the same paired worlds."""

    name = str(mapping["name"])
    seed = int(mapping.get("seed", 42))
    output_dir = resolve_path(mapping["output_dir"], base_dir=base_dir)
    lexicon_path = resolve_path(mapping["lexicons"], base_dir=base_dir)
    lexicons: LexiconBundle = _load_lexicons(lexicon_path)
    pool_min_words = int(mapping.get("pool_min_words", 130_000))
    eval_pairs = int(mapping.get("eval_pairs", 1_000))
    world_config = mapping.get("world") or {}
    operation_config = mapping.get("operations") or {}
    rendering_config = mapping.get("rendering") or {}

    count_weights = world_config.get("container_count_weights") or {
        3: 0.5,
        4: 0.3,
        5: 0.2,
    }
    min_items = int(world_config.get("min_items_per_container", 2))
    max_items = int(world_config.get("max_items_per_container", 3))
    same_type_ratio = float(world_config.get("same_type_ratio", 0.5))
    common_count_weights = operation_config.get("common_count_weights") or {
        0: 0.34,
        1: 0.33,
        2: 0.33,
    }
    common_kind_weights = operation_config.get("common_kind_weights") or {
        kind: 1.0 for kind in OPERATION_KINDS
    }
    train_renderers = tuple(
        rendering_config.get("training_templates") or TRAIN_RENDERERS
    )
    eval_renderers = tuple(
        rendering_config.get("heldout_templates") or HELDOUT_RENDERERS
    )
    if min_items < 2:
        raise ValueError("counterfactual generation requires min_items_per_container >= 2.")
    if pool_min_words <= 0 or eval_pairs <= 0:
        raise ValueError("pool_min_words and eval_pairs must be positive.")

    train = _generate_pairs(
        name=name,
        split="train",
        seed=seed,
        lexicon_name="train",
        lexicon=lexicons.train,
        renderers=train_renderers,
        count_weights=count_weights,
        min_items=min_items,
        max_items=max_items,
        same_type_ratio=same_type_ratio,
        common_count_weights=common_count_weights,
        common_kind_weights=common_kind_weights,
        target_words=pool_min_words,
    )
    evaluation = _generate_pairs(
        name=name,
        split="eval",
        seed=seed + 2_000_003,
        lexicon_name="validation",
        lexicon=lexicons.validation,
        renderers=eval_renderers,
        count_weights=count_weights,
        min_items=min_items,
        max_items=max_items,
        same_type_ratio=same_type_ratio,
        common_count_weights=common_count_weights,
        common_kind_weights=common_kind_weights,
        target_pairs=eval_pairs,
    )
    train_signatures = {
        json.dumps(pair["initial_state"], sort_keys=True) for pair in train
    }
    eval_signatures = {
        json.dumps(pair["initial_state"], sort_keys=True) for pair in evaluation
    }
    overlap = train_signatures & eval_signatures
    if overlap:
        raise RuntimeError("Train/eval initial-world signatures overlap.")

    auxiliary_dir = output_dir / "auxiliary"
    outputs = {
        "train": output_dir / "train.txt",
        "train_pairs": output_dir / "train_pairs.jsonl",
        "eval_pairs": output_dir / "eval_pairs.jsonl",
        "train_completion_rank": auxiliary_dir / "train_completion_rank.jsonl",
        "eval_completion_rank": auxiliary_dir / "eval_completion_rank.jsonl",
        "train_counterfactual_rank": auxiliary_dir / "train_counterfactual_rank.jsonl",
        "eval_counterfactual_rank": auxiliary_dir / "eval_counterfactual_rank.jsonl",
        "audit_report": output_dir / "audit_report.json",
        "manifest": output_dir / "manifest.json",
        "config": output_dir / "counterfactual_config.yaml",
        "lexicons": output_dir / "lexicons.yaml",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_training_text(outputs["train"], train)
    _write_jsonl(outputs["train_pairs"], train)
    _write_jsonl(outputs["eval_pairs"], evaluation)
    _write_jsonl(outputs["train_completion_rank"], _completion_rows(train))
    _write_jsonl(outputs["eval_completion_rank"], _completion_rows(evaluation))
    _write_jsonl(outputs["train_counterfactual_rank"], _counterfactual_rows(train))
    _write_jsonl(outputs["eval_counterfactual_rank"], _counterfactual_rows(evaluation))

    audit = {
        "name": name,
        "counts": {
            "train_pairs": len(train),
            "train_documents": 2 * len(train),
            "train_words": sum(int(pair["word_count"]) for pair in train),
            "eval_pairs": len(evaluation),
            "empty_targets": 0,
            "identical_targets": 0,
            "train_eval_world_overlap": len(overlap),
        },
        "operation_count_counts": _counts(train, "operation_count"),
        "counterfactual_kind_counts": _counts(train, "counterfactual_kind"),
        "renderer_counts": _counts(train, "renderer"),
        "lexical_split": {"train": "train", "eval": "validation"},
    }
    _write_json(outputs["audit_report"], audit)
    manifest = {
        "name": name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "design": "paired counterfactual contexts with nonempty differing targets",
        "outputs": {key: str(path) for key, path in outputs.items()},
        "audit": audit,
    }
    _write_json(outputs["manifest"], manifest)
    if config_path is not None:
        shutil.copy2(config_path, outputs["config"])
    else:
        with outputs["config"].open("w", encoding="utf-8") as f:
            yaml.safe_dump(dict(mapping), f, sort_keys=False)
    shutil.copy2(lexicon_path, outputs["lexicons"])
    return CounterfactualCorpusResult(
        name=name,
        output_dir=output_dir,
        train_pairs=train,
        eval_pairs=evaluation,
        audit_report=audit,
        outputs=outputs,
    )


__all__ = ["CounterfactualCorpusResult", "build_counterfactual_corpus"]
