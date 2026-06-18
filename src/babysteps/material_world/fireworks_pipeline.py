"""Readable material-world generation pipeline for Fireworks chat models."""

from __future__ import annotations

import json
import os
import random
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from babysteps.material_world.audit import audit_jsonl_file, read_jsonl, write_json, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "material_world"
DEFAULT_PIPELINE_CONFIG_PATH = CONFIG_DIR / "fireworks_probe.json"
DEFAULT_AUDIT_CONFIG_PATH = CONFIG_DIR / "audit_config.json"
DEFAULT_COVERAGE_SETS_CONFIG_PATH = CONFIG_DIR / "coverage_sets.json"

ASSIGNMENT_FIELDS = [
    "coverage_set_id",
    "coverage_domain",
    "coverage_role",
    "target_property_concept",
    "target_dynamic_concept",
    "target_object_id",
    "target_material",
    "target_style_id",
    "target_anchor",
    "target_concept_pair",
    "object_sampling_mode",
    "object_sampling_scope",
    "object_sampling_role",
    "style_sampling_mode",
    "style_sampling_scope",
    "style_sampling_role",
]

DEFAULT_SYSTEM_PROMPT = """You write ordinary, concrete English about everyday objects and materials.
Prefer visible behavior and cautious words like often, usually, can, and may.
Avoid rare technical details, microscopic explanations, benchmark-style formatting, and question-answer format.
Write the answer directly. Do not include reasoning, analysis, scratch work, or hidden planning. /no_think"""

JSON_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT + "\nReturn valid JSON only. Do not wrap it in Markdown."

EDITOR_SYSTEM_PROMPT = """You lightly edit generated prose about everyday objects and materials.
Preserve the original meaning and metadata. Prefer concrete, visible behavior and cautious wording.
Only fix factual overreach, repetition, awkward synthetic phrasing, or flagged row-level issues."""

DEFAULT_DISCOURSE_STYLES: list[dict[str, Any]] = [
    {
        "style_id": "everyday_narrative",
        "style_name": "Everyday narrative",
        "description": "A small ordinary scene where a person uses, drops, washes, carries, or notices an object.",
        "avoid": ["explicit lesson labels", "technical explanation"],
    },
    {
        "style_id": "comparison_paragraph",
        "style_name": "Comparison paragraph",
        "description": "A natural paragraph comparing how two material choices change the same object or event.",
        "avoid": ["quiz wording", "C1/T1 labels", "overusing unlike"],
    },
    {
        "style_id": "child_science_explanation",
        "style_name": "Child science explanation",
        "description": "Simple explanatory prose with child-friendly vocabulary and visible causes.",
        "avoid": ["microscopic mechanisms", "absolute claims"],
    },
    {
        "style_id": "repair_or_craft_note",
        "style_name": "Repair/craft note",
        "description": "A practical note from fixing, making, cleaning, or choosing an everyday object.",
        "avoid": ["engineering jargon", "instruction-manual stiffness"],
    },
    {
        "style_id": "observation_log",
        "style_name": "Observation log",
        "description": "Brief, plain observations of what happened before and after an everyday event.",
        "avoid": ["lab jargon", "numbered benchmark cases"],
    },
    {
        "style_id": "short_dialogue",
        "style_name": "Short dialogue",
        "description": "Two or three short turns of natural dialogue about a material behavior.",
        "avoid": ["quiz-answer format", "over-explaining"],
    },
    {
        "style_id": "application_tradeoff",
        "style_name": "Application/tradeoff paragraph",
        "description": "A practical paragraph about why a material helps one use case but has a tradeoff.",
        "avoid": ["marketing language", "grand claims"],
    },
    {
        "style_id": "simple_encyclopedia",
        "style_name": "Simple encyclopedia paragraph",
        "description": "A calm factual paragraph about a common object-material pairing.",
        "avoid": ["dense terminology", "overbroad universals"],
    },
]

ROW_LEVEL_AUDIT_EDIT_FLAGS = {
    "repetitive_opening_5",
    "repetitive_opening_8",
    "many_connectives",
    "because_overuse",
    "contrast_phrase_overuse",
    "too_short",
    "too_long",
    "template_phrase_overuse",
    "repeated_first_12_words",
    "repeated_8gram",
    "near_duplicate_candidate",
}

NO_THINK_SUFFIX = "/no_think"
CHECK_CONTEXT_METADATA_FIELDS = [
    "style_id",
    "material",
    "material_class",
    "material_class_name",
    "object",
    "object_id",
    "event",
    "properties",
    "dynamics",
    "coverage_role",
    "target_property_concept",
    "target_dynamic_concept",
    "target_object_id",
    "target_material",
    "target_style_id",
    "target_anchor",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: str | Path, *, root: Path = REPO_ROOT) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return root / resolved


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pipeline_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_json(path or DEFAULT_PIPELINE_CONFIG_PATH)


def load_discourse_styles(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path is None:
        return deepcopy(DEFAULT_DISCOURSE_STYLES)
    style_config = load_json(path)
    styles = style_config.get("discourse_styles", style_config.get("styles", []))
    if not isinstance(styles, list):
        raise ValueError(f"Discourse styles file must contain a list, got: {path}")
    return styles


def _filter_by_ids(items: list[dict[str, Any]], key: str, ids: list[str] | None) -> list[dict[str, Any]]:
    if not ids:
        return items
    by_id = {str(item.get(key)): item for item in items}
    missing = [item_id for item_id in ids if item_id not in by_id]
    if missing:
        raise ValueError(f"Unknown {key} values: {missing}")
    return [by_id[item_id] for item_id in ids]


def _coverage_sets_by_ids(coverage_config: dict[str, Any], ids: list[str] | None) -> list[dict[str, Any]]:
    if not ids:
        return []
    return _filter_by_ids(coverage_config.get("coverage_sets", []), "set_id", ids)


def _unique_in_order(items: list[str]) -> list[str]:
    seen = set()
    unique = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _all_ids(items: list[dict[str, Any]], key: str) -> list[str]:
    return [str(item[key]) for item in items if item.get(key) is not None]


def selection_seed(config: dict[str, Any]) -> str | None:
    seed = config.get("selection_seed", config.get("seed"))
    if seed is None:
        return None
    return str(seed)


def seeded_shuffle(items: list[dict[str, Any]], *, seed: str | None, salt: str) -> list[dict[str, Any]]:
    shuffled = list(items)
    if seed is None:
        return shuffled
    random.Random(f"{seed}:{salt}").shuffle(shuffled)
    return shuffled


def _coverage_set_ids_from_config(config: dict[str, Any]) -> list[str]:
    ids = as_list(config.get("coverage_set_ids"))
    coverage_mix = config.get("coverage_mix") or {}
    if isinstance(coverage_mix, dict):
        ids.extend(str(key) for key in coverage_mix.keys())
    return _unique_in_order(ids)


def _fraction(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        value = default
    resolved = float(value)
    if resolved < 0.0 or resolved > 1.0:
        raise ValueError(f"Coverage mix fractions must be in [0, 1], got {value}.")
    return resolved


def coverage_mix_settings(
    config: dict[str, Any],
    coverage_sets: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Return validated soft coverage mix settings keyed by coverage set id."""

    coverage_mix = config.get("coverage_mix") or {}
    if not coverage_mix:
        return {}
    if not isinstance(coverage_mix, dict):
        raise ValueError("coverage_mix must be an object keyed by coverage set id.")
    settings: dict[str, dict[str, float]] = {}
    for coverage_set in coverage_sets:
        set_id = str(coverage_set.get("set_id"))
        raw = coverage_mix.get(set_id)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"coverage_mix.{set_id} must be an object.")
        settings[set_id] = {
            "row_fraction": _fraction(raw.get("row_fraction")),
            "object_fraction": _fraction(raw.get("object_fraction")),
            "material_fraction": _fraction(raw.get("material_fraction")),
            "dynamic_fraction": _fraction(raw.get("dynamic_fraction")),
            "anchor_fraction": _fraction(raw.get("anchor_fraction")),
        }
    return settings


def object_sampling_settings(config: dict[str, Any]) -> dict[str, str]:
    """Return validated object sampling settings."""

    raw = config.get("object_sampling") or {}
    if isinstance(raw, str):
        raw = {"mode": raw}
    if not isinstance(raw, dict):
        raise ValueError("object_sampling must be an object or mode string.")
    mode = str(raw.get("mode", "model_choice"))
    scope = str(raw.get("scope", "selected_objects"))
    if mode not in {"model_choice", "balanced_round_robin"}:
        raise ValueError(
            "object_sampling.mode must be 'model_choice' or 'balanced_round_robin'."
        )
    if scope != "selected_objects":
        raise ValueError("object_sampling.scope currently only supports 'selected_objects'.")
    return {"mode": mode, "scope": scope}


def style_sampling_settings(config: dict[str, Any]) -> dict[str, str]:
    """Return validated style sampling settings."""

    raw = config.get("style_sampling") or {}
    if isinstance(raw, str):
        raw = {"mode": raw}
    if not isinstance(raw, dict):
        raise ValueError("style_sampling must be an object or mode string.")
    mode = str(raw.get("mode", "model_choice"))
    scope = str(raw.get("scope", "selected_styles"))
    if mode not in {"model_choice", "balanced_round_robin"}:
        raise ValueError(
            "style_sampling.mode must be 'model_choice' or 'balanced_round_robin'."
        )
    if scope != "selected_styles":
        raise ValueError("style_sampling.scope currently only supports 'selected_styles'.")
    return {"mode": mode, "scope": scope}


def _domain_for_coverage_set(coverage_set: dict[str, Any]) -> str:
    set_id = str(coverage_set.get("set_id", ""))
    domain = coverage_set.get("domain")
    if domain:
        return str(domain)
    if "dynamic" in set_id:
        return "material-dynamics"
    return "material-properties"


def _flatten_pair_items(coverage_set: dict[str, Any], key: str) -> list[dict[str, Any]]:
    items = []
    seen = set()
    for pair in coverage_set.get(key, []):
        values = as_list(pair)
        if not values:
            continue
        for concept in values:
            if concept in seen:
                continue
            seen.add(concept)
            items.append({"concept": concept, "pair": values})
    return items


def _coverage_anchors(coverage_set: dict[str, Any]) -> list[str]:
    """Return lexical anchors for row-level assignment."""

    return _unique_in_order(
        as_list(coverage_set.get("lexical_anchors"))
        or as_list(coverage_set.get("anchors"))
    )


def _count_from_fraction(fraction: float, total: int) -> int:
    return max(0, min(int(total), int(round(float(fraction) * int(total)))))


def _cycle_value(items: list[str], index: int) -> str | None:
    if not items:
        return None
    return items[index % len(items)]


def _cycle_item(items: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    if not items:
        return None
    return items[index % len(items)]


def _selected_material_names(selected_material_classes: list[dict[str, Any]]) -> set[str]:
    return {
        material.lower()
        for material_class in selected_material_classes
        for material in as_list(material_class.get("examples"))
    }


def _seeded_order(items: list[str], *, seed: str | None, salt: str) -> list[str]:
    ordered = list(items)
    random.Random(f"{seed or 'default'}:{salt}").shuffle(ordered)
    return ordered


def build_coverage_assignments(
    *,
    config: dict[str, Any],
    coverage_sets: list[dict[str, Any]],
    coverage_mix: dict[str, dict[str, float]],
    selected: dict[str, list[dict[str, Any]]],
    scenario_limit: int,
    seed: str | None,
) -> list[dict[str, Any]]:
    """Create deterministic row-level coverage assignments before LLM planning."""

    del config
    if not coverage_sets or not coverage_mix or scenario_limit <= 0:
        return []

    selected_object_ids = {str(item.get("object_id")) for item in selected["objects"] if item.get("object_id")}
    selected_materials = _selected_material_names(selected["material_classes"])
    available_indices = list(range(int(scenario_limit)))
    random.Random(f"{seed or 'default'}:coverage_rows").shuffle(available_indices)
    cursor = 0
    assignments: list[dict[str, Any]] = []

    def take_indices(count: int) -> list[int]:
        nonlocal cursor
        count = max(0, min(count, len(available_indices) - cursor))
        indices = available_indices[cursor : cursor + count]
        cursor += count
        return indices

    for coverage_set in coverage_sets:
        set_id = str(coverage_set.get("set_id"))
        settings = coverage_mix.get(set_id)
        if not settings:
            continue
        domain = _domain_for_coverage_set(coverage_set)
        coverage_object_ids = _seeded_order(
            [
                object_id
                for object_id in as_list(coverage_set.get("object_ids"))
                if object_id in selected_object_ids
            ],
            seed=seed,
            salt=f"{set_id}:objects",
        )
        coverage_materials = _seeded_order(
            [
                material
                for material in as_list(coverage_set.get("materials"))
                if material.lower() in selected_materials
            ],
            seed=seed,
            salt=f"{set_id}:materials",
        )
        property_items = _flatten_pair_items(coverage_set, "concept_pairs")
        dynamic_items = _flatten_pair_items(coverage_set, "dynamic_pairs")
        lexical_anchors = _seeded_order(
            _coverage_anchors(coverage_set),
            seed=seed,
            salt=f"{set_id}:anchors",
        )

        specs: list[tuple[str, int]] = []
        if settings.get("row_fraction", 0.0):
            if domain == "material-dynamics" and dynamic_items:
                specs.append(("ewok_dynamic", _count_from_fraction(settings["row_fraction"], scenario_limit)))
            elif domain != "material-dynamics" and property_items:
                specs.append(("ewok_property", _count_from_fraction(settings["row_fraction"], scenario_limit)))
        if domain == "material-properties":
            if coverage_object_ids:
                specs.append(("ewok_object", _count_from_fraction(settings["object_fraction"], scenario_limit)))
            if coverage_materials:
                specs.append(("ewok_material", _count_from_fraction(settings["material_fraction"], scenario_limit)))
        if domain == "material-dynamics" and dynamic_items:
            specs.append(("ewok_dynamic", _count_from_fraction(settings["dynamic_fraction"], scenario_limit)))

        set_assignments: list[dict[str, Any]] = []
        role_offsets: dict[str, int] = {}
        for role, count in specs:
            for offset, row_index in enumerate(take_indices(count)):
                role_index = role_offsets.get(role, 0) + offset
                if domain == "material-dynamics":
                    concept_item = _cycle_item(dynamic_items, role_index) or {}
                    target_dynamic = concept_item.get("concept")
                    target_property = None
                else:
                    concept_item = _cycle_item(property_items, role_index) or {}
                    target_property = concept_item.get("concept")
                    target_dynamic = None
                target_object_id = _cycle_value(coverage_object_ids, role_index) if role == "ewok_object" else None
                target_material = _cycle_value(coverage_materials, role_index) if role == "ewok_material" else None
                if role == "ewok_property":
                    target_object_id = _cycle_value(coverage_object_ids, role_index)
                    target_material = _cycle_value(coverage_materials, role_index)
                set_assignments.append(
                    {
                        "scenario_id": f"mw_probe_{row_index:04d}",
                        "coverage_set_id": set_id,
                        "coverage_domain": domain,
                        "coverage_role": role,
                        "target_property_concept": target_property,
                        "target_dynamic_concept": target_dynamic,
                        "target_object_id": target_object_id,
                        "target_material": target_material,
                        "target_anchor": None,
                        "target_concept_pair": concept_item.get("pair"),
                    }
                )
            role_offsets[role] = role_offsets.get(role, 0) + count

        anchor_count = _count_from_fraction(settings.get("anchor_fraction", 0.0), scenario_limit)
        if lexical_anchors and anchor_count:
            by_row = {str(item["scenario_id"]): item for item in set_assignments}
            for anchor_index, row_index in enumerate(available_indices[:anchor_count]):
                scenario_id = f"mw_probe_{row_index:04d}"
                anchor = _cycle_value(lexical_anchors, anchor_index)
                existing = by_row.get(scenario_id)
                if existing is not None:
                    existing["target_anchor"] = anchor
                    role = str(existing.get("coverage_role") or "")
                    if "ewok_anchor" not in role:
                        existing["coverage_role"] = (
                            f"{role}+ewok_anchor" if role else "ewok_anchor"
                        )
                    continue
                anchor_assignment = {
                    "scenario_id": scenario_id,
                    "coverage_set_id": set_id,
                    "coverage_domain": domain,
                    "coverage_role": "ewok_anchor",
                    "target_property_concept": None,
                    "target_dynamic_concept": None,
                    "target_object_id": None,
                    "target_material": None,
                    "target_anchor": anchor,
                    "target_concept_pair": None,
                }
                by_row[scenario_id] = anchor_assignment
                set_assignments.append(anchor_assignment)

        assignments.extend(set_assignments)

    return sorted(assignments, key=lambda item: item["scenario_id"])


def _selected_object_ids(selected: dict[str, list[dict[str, Any]]]) -> list[str]:
    return [
        str(item.get("object_id"))
        for item in selected["objects"]
        if item.get("object_id") is not None
    ]


def build_object_sampling_assignments(
    *,
    object_sampling: dict[str, str],
    selected: dict[str, list[dict[str, Any]]],
    scenario_limit: int,
    seed: str | None,
    existing_assignments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create deterministic object targets across selected objects."""

    if object_sampling.get("mode") != "balanced_round_robin" or scenario_limit <= 0:
        return []

    object_ids = _seeded_order(
        _selected_object_ids(selected),
        seed=seed,
        salt="object_sampling:selected_objects",
    )
    if not object_ids:
        return []

    counts = {object_id: 0 for object_id in object_ids}
    existing_by_id = {
        str(item["scenario_id"]): item
        for item in existing_assignments
        if item.get("scenario_id") is not None
    }
    for assignment in existing_assignments:
        target_object_id = assignment.get("target_object_id")
        if target_object_id in counts:
            counts[str(target_object_id)] += 1

    assignments = []
    for row_index in range(int(scenario_limit)):
        scenario_id = f"mw_probe_{row_index:04d}"
        existing_target = existing_by_id.get(scenario_id, {}).get("target_object_id")
        if existing_target in counts:
            target_object_id = str(existing_target)
            role = "preserved_target"
        else:
            min_count = min(counts.values())
            target_object_id = next(
                object_id
                for object_id in object_ids
                if counts[object_id] == min_count
            )
            counts[target_object_id] += 1
            role = "balanced_target"
        assignments.append(
            {
                "scenario_id": scenario_id,
                "target_object_id": target_object_id,
                "object_sampling_mode": "balanced_round_robin",
                "object_sampling_scope": object_sampling.get("scope", "selected_objects"),
                "object_sampling_role": role,
            }
        )
    return assignments


def _selected_style_ids(selected: dict[str, list[dict[str, Any]]]) -> list[str]:
    return [
        str(item.get("style_id"))
        for item in selected["styles"]
        if item.get("style_id") is not None
    ]


def build_style_sampling_assignments(
    *,
    style_sampling: dict[str, str],
    selected: dict[str, list[dict[str, Any]]],
    scenario_limit: int,
    seed: str | None,
) -> list[dict[str, Any]]:
    """Create deterministic discourse-style targets across selected styles."""

    if style_sampling.get("mode") != "balanced_round_robin" or scenario_limit <= 0:
        return []

    style_ids = _seeded_order(
        _selected_style_ids(selected),
        seed=seed,
        salt="style_sampling:selected_styles",
    )
    if not style_ids:
        return []

    assignments = []
    for row_index in range(int(scenario_limit)):
        target_style_id = style_ids[row_index % len(style_ids)]
        assignments.append(
            {
                "scenario_id": f"mw_probe_{row_index:04d}",
                "target_style_id": target_style_id,
                "style_sampling_mode": "balanced_round_robin",
                "style_sampling_scope": style_sampling.get("scope", "selected_styles"),
                "style_sampling_role": "balanced_target",
            }
        )
    return assignments


def _complete_assignment(assignment: dict[str, Any]) -> dict[str, Any]:
    completed = dict(assignment)
    for field in ASSIGNMENT_FIELDS:
        completed.setdefault(field, None)
    return completed


def merge_row_assignments(*assignment_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge compatible row assignment layers by scenario id."""

    by_id: dict[str, dict[str, Any]] = {}
    for assignments in assignment_groups:
        for assignment in assignments:
            scenario_id = str(assignment.get("scenario_id"))
            if not scenario_id:
                continue
            row = by_id.setdefault(scenario_id, {"scenario_id": scenario_id})
            for key, value in assignment.items():
                if key == "scenario_id":
                    continue
                if value is not None or key not in row:
                    row[key] = value
    return [
        _complete_assignment(assignment)
        for assignment in sorted(by_id.values(), key=lambda item: item["scenario_id"])
    ]


def material_class_ids_for_materials(material_config: dict[str, Any], materials: list[str]) -> list[str]:
    """Find material classes whose examples include any of the requested material names."""

    requested = {material.lower() for material in materials}
    class_ids = []
    for material_class in material_config.get("material_classes", []):
        examples = {example.lower() for example in as_list(material_class.get("examples"))}
        if examples & requested:
            class_id = material_class.get("class_id")
            if class_id:
                class_ids.append(str(class_id))
    return class_ids


def coverage_selection_overrides(
    config: dict[str, Any],
    material_config: dict[str, Any],
    object_config: dict[str, Any],
    coverage_sets: list[dict[str, Any]],
    coverage_mix: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Expand optional coverage sets into class/object ids without changing default runs."""

    if not coverage_sets:
        return {
            "material_class_ids": config.get("material_class_ids"),
            "object_ids": config.get("object_ids"),
            "material_class_limit": int(config.get("material_class_limit", 4)),
            "object_limit": int(config.get("object_limit", 3)),
        }

    coverage_materials = _unique_in_order(
        [
            material
            for coverage_set in coverage_sets
            for material in as_list(coverage_set.get("materials"))
        ]
    )
    coverage_object_ids = _unique_in_order(
        [
            object_id
            for coverage_set in coverage_sets
            for object_id in as_list(coverage_set.get("object_ids"))
        ]
    )
    material_class_ids = config.get("material_class_ids") or material_class_ids_for_materials(
        material_config,
        coverage_materials,
    )
    object_ids = config.get("object_ids") or coverage_object_ids
    if coverage_mix:
        broad_material_ids = _all_ids(
            material_config.get("material_classes", []),
            "class_id",
        )
        broad_object_ids = _all_ids(
            object_config.get("objects", []),
            "object_id",
        )
        base_material_ids = as_list(config.get("material_class_ids")) or broad_material_ids
        base_object_ids = as_list(config.get("object_ids")) or broad_object_ids
        material_class_ids = _unique_in_order(base_material_ids + material_class_ids_for_materials(
            material_config,
            coverage_materials,
        ))
        object_ids = _unique_in_order(base_object_ids + coverage_object_ids)
    return {
        "material_class_ids": material_class_ids,
        "object_ids": object_ids,
        "material_class_limit": max(int(config.get("material_class_limit", 4)), len(material_class_ids or [])),
        "object_limit": max(int(config.get("object_limit", 3)), len(object_ids or [])),
    }


def select_probe_items(
    material_config: dict[str, Any],
    object_config: dict[str, Any],
    discourse_styles: list[dict[str, Any]],
    *,
    material_class_limit: int = 4,
    object_limit: int = 3,
    style_limit: int = 4,
    material_class_ids: list[str] | None = None,
    object_ids: list[str] | None = None,
    style_ids: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Select a deterministic probe slice, optionally constrained by ids."""

    material_classes = _filter_by_ids(material_config.get("material_classes", []), "class_id", material_class_ids)
    objects = _filter_by_ids(object_config.get("objects", []), "object_id", object_ids)
    styles = _filter_by_ids(discourse_styles, "style_id", style_ids)
    return {
        "material_classes": material_classes[: int(material_class_limit)],
        "objects": objects[: int(object_limit)],
        "styles": styles[: int(style_limit)],
    }


def extract_json(text: str) -> Any:
    """Extract a JSON value from plain model output or a fenced block."""

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        fenced_text = fence.group(1).strip()
        try:
            return json.loads(fenced_text)
        except json.JSONDecodeError:
            pass

    starts = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0]
    if not starts:
        raise ValueError(f"Model output did not contain JSON: {stripped[:400]}")
    start = min(starts)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if end <= start:
        raise ValueError(f"Model output contained incomplete JSON: {stripped[:400]}")
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON from model output: {stripped[:800]}") from exc


def json_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, dict):
        for key in ("items", "examples", "scenario_plans", "rows", "data"):
            if isinstance(value.get(key), list):
                items = value[key]
                break
        else:
            raise ValueError("JSON object must include an items/examples/scenario_plans/rows list.")
    else:
        raise ValueError(f"Expected JSON list or object, got {type(value).__name__}.")

    if not all(isinstance(item, dict) for item in items):
        raise ValueError("JSON items must be objects.")
    return items


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[,;/|]", value) if part.strip()]
        return parts or [value]
    return [str(value)]


def normalize_scenario_plans(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plans = []
    for index, item in enumerate(items):
        plan = dict(item)
        scenario_id = str(plan.get("scenario_id") or plan.get("id") or f"mw_probe_{index:04d}")
        plan["scenario_id"] = scenario_id
        plan["dynamics"] = as_list(plan.get("dynamics") or plan.get("dynamic"))
        plan["properties"] = as_list(plan.get("properties") or plan.get("property"))
        plan["contains_negative_reporting"] = bool(plan.get("contains_negative_reporting", False))
        plans.append(plan)
    return plans


def metadata_from_plan(plan: dict[str, Any], rendered: dict[str, Any] | None = None) -> dict[str, Any]:
    rendered = rendered or {}
    style_id = (
        rendered.get("discourse_style_id")
        or rendered.get("style_id")
        or plan.get("discourse_style_id")
        or plan.get("style_id")
    )
    material_class = plan.get("material_class_id") or plan.get("material_class") or plan.get("class_id")
    object_name = plan.get("object") or plan.get("object_name") or plan.get("object_id")
    return {
        "style_id": str(style_id or "missing"),
        "material": str(plan.get("material") or "missing"),
        "material_class": str(material_class or "missing"),
        "material_class_name": str(plan.get("material_class_name") or ""),
        "object": str(object_name or "missing"),
        "object_id": str(plan.get("object_id") or ""),
        "event": str(plan.get("event") or ""),
        "properties": as_list(plan.get("properties")),
        "dynamics": as_list(plan.get("dynamics")),
        "contains_negative_reporting": bool(plan.get("contains_negative_reporting", False)),
        "coverage_set_id": plan.get("coverage_set_id"),
        "coverage_domain": plan.get("coverage_domain"),
        "coverage_role": plan.get("coverage_role"),
        "target_property_concept": plan.get("target_property_concept"),
        "target_dynamic_concept": plan.get("target_dynamic_concept"),
        "target_object_id": plan.get("target_object_id"),
        "target_material": plan.get("target_material"),
        "target_style_id": plan.get("target_style_id"),
        "target_anchor": plan.get("target_anchor"),
        "target_concept_pair": plan.get("target_concept_pair"),
        "object_sampling_mode": plan.get("object_sampling_mode"),
        "object_sampling_scope": plan.get("object_sampling_scope"),
        "object_sampling_role": plan.get("object_sampling_role"),
        "style_sampling_mode": plan.get("style_sampling_mode"),
        "style_sampling_scope": plan.get("style_sampling_scope"),
        "style_sampling_role": plan.get("style_sampling_role"),
    }


def material_class_for_material(
    material_config: dict[str, Any],
    material: str | None,
) -> dict[str, Any] | None:
    if not material:
        return None
    requested = str(material).lower()
    for material_class in material_config.get("material_classes", []):
        if requested in {example.lower() for example in as_list(material_class.get("examples"))}:
            return material_class
    return None


def object_card_by_id(object_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("object_id")): item
        for item in object_config.get("objects", [])
        if item.get("object_id") is not None
    }


def _append_unique(values: list[str], value: str | None) -> list[str]:
    result = list(values)
    if value and value not in result:
        result.append(value)
    return result


def apply_coverage_assignments_to_plans(
    plans: list[dict[str, Any]],
    coverage_assignments: list[dict[str, Any]],
    *,
    material_config: dict[str, Any],
    object_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge deterministic coverage targets into model-created scenario plans."""

    assignments_by_id = {item["scenario_id"]: item for item in coverage_assignments}
    objects_by_id = object_card_by_id(object_config)
    merged = []
    for plan in plans:
        scenario_id = str(plan.get("scenario_id"))
        assignment = assignments_by_id.get(scenario_id)
        if not assignment:
            clean_plan = dict(plan)
            for field in ASSIGNMENT_FIELDS:
                clean_plan.setdefault(field, None)
            merged.append(clean_plan)
            continue

        clean_plan = dict(plan)
        clean_plan.update(assignment)
        target_object_id = assignment.get("target_object_id")
        if target_object_id and target_object_id in objects_by_id:
            object_card = objects_by_id[target_object_id]
            clean_plan["object_id"] = target_object_id
            clean_plan["object"] = object_card.get("object_name", target_object_id)

        target_material = assignment.get("target_material")
        if target_material:
            clean_plan["material"] = target_material
            material_class = material_class_for_material(material_config, target_material)
            if material_class:
                clean_plan["material_class_id"] = material_class.get("class_id")
                clean_plan["material_class_name"] = material_class.get("class_name")

        target_style_id = assignment.get("target_style_id")
        if target_style_id:
            clean_plan["discourse_style_id"] = target_style_id
            clean_plan["style_id"] = target_style_id

        target_anchor = assignment.get("target_anchor")
        if target_anchor:
            clean_plan["target_anchor"] = target_anchor
            plan_notes = str(clean_plan.get("plan_notes") or "").strip()
            anchor_note = (
                f"Make the exact lexical anchor phrase '{target_anchor}' a central participant in the event, "
                "not a nearby prop, absence statement, unaffected aside, or loose comparison."
            )
            clean_plan["plan_notes"] = f"{plan_notes} {anchor_note}".strip()

        clean_plan["properties"] = _append_unique(
            as_list(clean_plan.get("properties")),
            assignment.get("target_property_concept"),
        )
        clean_plan["dynamics"] = _append_unique(
            as_list(clean_plan.get("dynamics")),
            assignment.get("target_dynamic_concept"),
        )
        merged.append(clean_plan)
    return merged


def build_scenario_plan_prompt(
    *,
    selected_material_classes: list[dict[str, Any]],
    selected_objects: list[dict[str, Any]],
    selected_styles: list[dict[str, Any]],
    scenario_limit: int,
    scenario_start_index: int = 0,
    coverage_sets: list[dict[str, Any]] | None = None,
    coverage_mix: dict[str, dict[str, float]] | None = None,
    selection_seed_value: str | None = None,
    coverage_assignments: list[dict[str, Any]] | None = None,
) -> str:
    scenario_end_index = scenario_start_index + scenario_limit - 1
    coverage_sets = coverage_sets or []
    coverage_mix = coverage_mix or {}
    coverage_assignments = coverage_assignments or []
    assignment_guidance = ""
    if coverage_assignments:
        assignment_guidance = f"""

Row assignments for this batch:
{json.dumps(coverage_assignments, indent=2)}

For each listed scenario_id, preserve the assignment fields exactly in the returned plan.
If target_object_id is set, use that object_id.
If target_material is set, use that material as the object material.
If target_style_id is set, use that value as discourse_style_id.
If target_anchor is set, make that exact phrase a central participant in the event and make the rendered prose use the exact phrase.
The anchor may be the object, material, contents, substance, fuel, liquid, fabric, food, gas, or surface involved in the visible behavior.
Do not satisfy target_anchor by saying it is absent, unrelated, merely nearby, unaffected, or only a loose comparison or metaphor.
If target_property_concept is set, include that exact concept in properties and make it visible in the plan.
If target_dynamic_concept is set, include that exact concept in dynamics and make it visible in the plan.
If object_sampling_mode is set, keep the row on its assigned object so object exposure stays balanced.
If style_sampling_mode is set, keep the row on its assigned discourse style so style exposure stays balanced.
For scenario_id values without an assignment, set all assignment fields to null and keep the row broad/general.
""".rstrip()
    seed_guidance = ""
    if selection_seed_value is not None:
        seed_guidance = f"""

Run selection seed:
{selection_seed_value}

Use the selected material/object/style lists in their given order as the deterministic run order.
Vary choices across that order instead of repeatedly using only the first few entries.
""".rstrip()
    coverage_focus = ""
    if coverage_sets:
        mix_guidance = ""
        if coverage_mix:
            mix_guidance = f"""

Soft coverage mix targets:
{json.dumps(coverage_mix, indent=2)}

When coverage assignments are listed for this batch, those rows are the exact quota-backed targets.
Use this mix block as dataset-level guidance for the run as a whole.
Count a row toward object_fraction when its object_id is in that coverage set.
Count a row toward material_fraction when its material is in that coverage set.
Count a row toward anchor_fraction when target_anchor is set and the final prose includes that exact phrase.
Keep the remaining rows broad and ordinary; spread them across the selected material classes and objects.
Do not make every example coverage-specific or reuse only the first few non-coverage objects.
""".rstrip()
        else:
            mix_guidance = """

Use these coverage materials and objects as anchors when they fit.
""".rstrip()
        coverage_focus = f"""

Optional coverage focus:
{json.dumps(coverage_sets, indent=2)}
{mix_guidance}
Keep the examples natural.
Do not mention coverage sets, EWoK, benchmarks, concept labels, or test names in the generated prose.
For intentionally unusual object-material pairings, make the scenario clearly hypothetical or pretend.
""".rstrip()
    return f"""
Return final JSON only. Do not explain, analyze, draft, revise, or show reasoning.
Create exactly {scenario_limit} object-material scenario plans.
Use scenario_id values from mw_probe_{scenario_start_index:04d} through mw_probe_{scenario_end_index:04d}, in order.

Use only the selected material classes, objects, and styles below.
Use ordinary visible dynamics such as bending, soaking, denting, cracking, chipping, compressing, softening,
warping, pouring, spilling, or crumpling.
For this v0 probe, "material" must mean what the object is made of. Do not use material for the floor,
tool, liquid, surface, or surrounding object. The plan should fit the phrase "a/an {{material}} {{object}}".

Required schema:
{{
  "items": [
    {{
      "scenario_id": "mw_probe_{scenario_start_index:04d}",
      "material_class_id": "...",
      "material_class_name": "...",
      "material": "...",
      "object_id": "...",
      "object": "...",
      "event": "...",
      "properties": ["..."],
      "dynamics": ["..."],
      "discourse_style_id": "...",
      "contains_negative_reporting": false,
      "coverage_set_id": null,
      "coverage_domain": null,
      "coverage_role": null,
      "target_property_concept": null,
      "target_dynamic_concept": null,
      "target_object_id": null,
      "target_material": null,
      "target_style_id": null,
      "target_anchor": null,
      "target_concept_pair": null,
      "object_sampling_mode": null,
      "object_sampling_scope": null,
      "object_sampling_role": null,
      "style_sampling_mode": null,
      "style_sampling_scope": null,
      "style_sampling_role": null,
      "plan_notes": "short everyday note"
    }}
  ]
}}

Selected material classes:
{json.dumps(selected_material_classes, indent=2)}

Selected objects:
{json.dumps(selected_objects, indent=2)}

Selected discourse styles:
{json.dumps(selected_styles, indent=2)}
{seed_guidance}
{assignment_guidance}
{coverage_focus}
""".strip()


def build_render_prompt(scenario_plans: list[dict[str, Any]]) -> str:
    return f"""
Render each scenario plan into natural prose.

Rules:
- Return exactly {len(scenario_plans)} JSON objects inside an "items" list, one per scenario.
- Preserve scenario_id and discourse_style_id.
- The text should be prose only, not bullet points or Q&A.
- Avoid labels like C1/T1, benchmark wording, and repeated template openings.
- Use cautious everyday wording such as often, usually, can, or may.
- Prefer visible material behavior over rare facts or microscopic explanations.
- If a scenario has target_anchor set, include that exact phrase at least once as a central participant in the event, naturally and without calling it an anchor.
- Do not mention target_anchor only as absent, unrelated, nearby, unaffected, or as a loose comparison.
- Target roughly 70-120 words per example unless the style naturally wants slightly shorter text.

Return valid JSON only. Start with "{{" and end with "}}".
{{
  "items": [
    {{"scenario_id": "...", "discourse_style_id": "...", "text": "..."}}
  ]
}}

Scenario plans:
{json.dumps(scenario_plans, indent=2)}
""".strip()


def build_check_prompt(examples: list[dict[str, Any]]) -> str:
    compact_rows = []
    for row in examples:
        metadata = row.get("metadata", {})
        compact_rows.append(
            {
                "id": row.get("id"),
                "text": row.get("text", ""),
                "context": {
                    key: metadata.get(key)
                    for key in CHECK_CONTEXT_METADATA_FIELDS
                    if metadata.get(key) is not None
                },
            }
        )
    return f"""
Lightly check and edit these generated examples.

Return all {len(examples)} rows, in the same order.
Each returned item must contain exactly two keys: "id" and "text".
Do not return metadata, context, comments, explanations, Markdown, or extra keys.
The context below is only for checking whether the text still fits the intended material, object, event, properties, dynamics, style, and anchor target.
If context.target_anchor is set, the text must keep that exact phrase naturally as a central participant in the event.
Do not leave context.target_anchor only as an absent, unrelated, nearby, unaffected, or comparison-only mention.
Only revise text when it is too broad, too technical, factually dubious, impossible without being framed as pretend,
too repetitive, or too template-like. Soften absolute claims with often, usually, can, or may.

Return valid JSON only. Start with "{{" and end with "}}".
{{
  "items": [
    {{"id": "...", "text": "..."}}
  ]
}}

Rows:
{json.dumps(compact_rows, indent=2)}
""".strip()


def build_audit_edit_prompt(row: dict[str, Any]) -> str:
    flags = row.get("audit_flags", [])
    return f"""
Revise this one example to address only the row-level audit flags.

Keep the same material, object, event, dynamics, style, and target_anchor if one is present.
If target_anchor is present, keep it as a central participant in the event, not as a nearby, absent, unrelated, or comparison-only mention.
Do not add rare facts.
Use natural prose only. Avoid making the paragraph much longer.

Return valid JSON only:
{{"id": "{row.get('id')}", "text": "..."}}

Audit flags:
{json.dumps(flags, indent=2)}

Example:
{json.dumps({"id": row.get("id"), "metadata": row.get("metadata", {}), "text": row.get("text", "")}, indent=2)}
""".strip()


def build_audit_edit_batch_prompt(rows: list[dict[str, Any]]) -> str:
    compact_rows = [
        {
            "id": row.get("id"),
            "metadata": row.get("metadata", {}),
            "audit_flags": row.get("audit_flags", []),
            "text": row.get("text", ""),
        }
        for row in rows
    ]
    return f"""
Revise these {len(rows)} examples to address only their row-level audit flags.

Keep the same material, object, event, dynamics, style, and target_anchor if one is present.
If target_anchor is present, keep it as a central participant in the event, not as a nearby, absent, unrelated, or comparison-only mention.
Do not add rare facts.
Use natural prose only. Avoid making any paragraph much longer.

Return valid JSON only:
{{
  "items": [
    {{"id": "...", "text": "..."}}
  ]
}}

Examples:
{json.dumps(compact_rows, indent=2)}
""".strip()


def _by_scenario_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = {}
    for item in items:
        item_id = item.get("scenario_id") or item.get("id")
        if item_id is not None:
            by_id[str(item_id)] = item
    return by_id


def build_prose_rows(
    scenario_plans: list[dict[str, Any]],
    rendered_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rendered_by_id = _by_scenario_id(rendered_items)
    rows = []
    for index, plan in enumerate(scenario_plans):
        scenario_id = str(plan["scenario_id"])
        rendered = rendered_by_id.get(scenario_id)
        if rendered is None and index < len(rendered_items):
            rendered = rendered_items[index]
        if not rendered or not str(rendered.get("text", "")).strip():
            raise ValueError(f"Missing rendered text for scenario {scenario_id}")
        rows.append(
            {
                "id": scenario_id,
                "text": str(rendered["text"]).strip(),
                "metadata": metadata_from_plan(plan, rendered),
            }
        )
    return rows


def merge_text_edits(base_rows: list[dict[str, Any]], edited_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(item.get("id") or item.get("scenario_id")): item for item in edited_items}
    merged = []
    for index, row in enumerate(base_rows):
        item = by_id.get(str(row["id"]))
        if item is None and index < len(edited_items):
            item = edited_items[index]
        clean_row = {"id": row["id"], "text": row["text"], "metadata": deepcopy(row.get("metadata", {}))}
        if item and str(item.get("text", "")).strip():
            clean_row["text"] = str(item["text"]).strip()
        if item and isinstance(item.get("metadata"), dict):
            clean_row["metadata"].update(item["metadata"])
        merged.append(clean_row)
    return merged


def strip_audit_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "text": row.get("text", ""),
        "metadata": deepcopy(row.get("metadata", {})),
    }


def has_row_level_edit_flag(row: dict[str, Any]) -> bool:
    for flag in row.get("audit_flags", []):
        if flag.get("code") in ROW_LEVEL_AUDIT_EDIT_FLAGS:
            return True
    return False


def _counter_dict(values: list[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _coverage_observed_for_set(rows: list[dict[str, Any]], coverage_set: dict[str, Any]) -> dict[str, Any]:
    set_id = str(coverage_set.get("set_id"))
    materials = {material.lower() for material in as_list(coverage_set.get("materials"))}
    object_ids = set(as_list(coverage_set.get("object_ids")))
    metadata_rows = [row.get("metadata", {}) for row in rows]
    assigned_rows = [
        row
        for row in rows
        if row.get("metadata", {}).get("coverage_set_id") == set_id
    ]
    assigned = [row.get("metadata", {}) for row in assigned_rows]
    anchor_hits = [
        metadata.get("target_anchor")
        for row, metadata in zip(assigned_rows, assigned)
        if metadata.get("target_anchor")
        and str(metadata.get("target_anchor")).lower() in str(row.get("text", "")).lower()
    ]
    anchor_misses = [
        metadata.get("target_anchor")
        for row, metadata in zip(assigned_rows, assigned)
        if metadata.get("target_anchor")
        and str(metadata.get("target_anchor")).lower() not in str(row.get("text", "")).lower()
    ]
    return {
        "assigned_rows": len(assigned),
        "ewok_object_rows": sum(1 for metadata in metadata_rows if metadata.get("object_id") in object_ids),
        "ewok_material_rows": sum(
            1
            for metadata in metadata_rows
            if str(metadata.get("material", "")).lower() in materials
        ),
        "target_property_concepts": _counter_dict(
            [metadata.get("target_property_concept") for metadata in assigned]
        ),
        "target_dynamic_concepts": _counter_dict(
            [metadata.get("target_dynamic_concept") for metadata in assigned]
        ),
        "target_anchors": _counter_dict([metadata.get("target_anchor") for metadata in assigned]),
        "target_anchor_text_hits": _counter_dict(anchor_hits),
        "target_anchor_text_misses": _counter_dict(anchor_misses),
        "coverage_roles": _counter_dict([metadata.get("coverage_role") for metadata in assigned]),
    }


def _object_counts_for_rows(rows: list[dict[str, Any]], selected_object_ids: list[str]) -> dict[str, int]:
    counts = {object_id: 0 for object_id in selected_object_ids}
    for row in rows:
        metadata = row.get("metadata", row)
        if not isinstance(metadata, dict):
            continue
        object_id = metadata.get("object_id")
        if object_id in counts:
            counts[str(object_id)] += 1
    return dict(sorted(counts.items()))


def _style_counts_for_rows(rows: list[dict[str, Any]], selected_style_ids: list[str]) -> dict[str, int]:
    counts = {style_id: 0 for style_id in selected_style_ids}
    for row in rows:
        metadata = row.get("metadata", row)
        if not isinstance(metadata, dict):
            continue
        style_id = metadata.get("style_id") or metadata.get("discourse_style_id")
        if style_id in counts:
            counts[str(style_id)] += 1
    return dict(sorted(counts.items()))


def _count_spread(counts: dict[str, int]) -> dict[str, int]:
    if not counts:
        return {"min": 0, "max": 0, "spread": 0}
    values = list(counts.values())
    return {
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def build_object_sampling_report(
    *,
    object_sampling: dict[str, str],
    selected_objects: list[dict[str, Any]],
    object_sampling_assignments: list[dict[str, Any]],
    checked_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_object_ids = [
        str(item.get("object_id"))
        for item in selected_objects
        if item.get("object_id") is not None
    ]
    assigned_counts = {object_id: 0 for object_id in selected_object_ids}
    for assignment in object_sampling_assignments:
        target_object_id = assignment.get("target_object_id")
        if target_object_id in assigned_counts:
            assigned_counts[str(target_object_id)] += 1
    assigned_counts = dict(sorted(assigned_counts.items()))
    checked_counts = _object_counts_for_rows(checked_rows, selected_object_ids)
    final_counts = _object_counts_for_rows(final_rows, selected_object_ids)
    return {
        "requested": object_sampling,
        "selected_object_count": len(selected_object_ids),
        "selected_object_ids": selected_object_ids,
        "assigned": {
            "rows": len(object_sampling_assignments),
            "target_object_ids": assigned_counts,
            "object_sampling_roles": _counter_dict(
                [item.get("object_sampling_role") for item in object_sampling_assignments]
            ),
            "spread": _count_spread(assigned_counts),
        },
        "observed_checked": {
            "object_ids": checked_counts,
            "spread": _count_spread(checked_counts),
        },
        "observed_final": {
            "object_ids": final_counts,
            "spread": _count_spread(final_counts),
        },
    }


def build_style_sampling_report(
    *,
    style_sampling: dict[str, str],
    selected_styles: list[dict[str, Any]],
    style_sampling_assignments: list[dict[str, Any]],
    checked_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_style_ids = [
        str(item.get("style_id"))
        for item in selected_styles
        if item.get("style_id") is not None
    ]
    assigned_counts = {style_id: 0 for style_id in selected_style_ids}
    for assignment in style_sampling_assignments:
        target_style_id = assignment.get("target_style_id")
        if target_style_id in assigned_counts:
            assigned_counts[str(target_style_id)] += 1
    assigned_counts = dict(sorted(assigned_counts.items()))
    checked_counts = _style_counts_for_rows(checked_rows, selected_style_ids)
    final_counts = _style_counts_for_rows(final_rows, selected_style_ids)
    return {
        "requested": style_sampling,
        "selected_style_count": len(selected_style_ids),
        "selected_style_ids": selected_style_ids,
        "assigned": {
            "rows": len(style_sampling_assignments),
            "target_style_ids": assigned_counts,
            "style_sampling_roles": _counter_dict(
                [item.get("style_sampling_role") for item in style_sampling_assignments]
            ),
            "spread": _count_spread(assigned_counts),
        },
        "observed_checked": {
            "style_ids": checked_counts,
            "spread": _count_spread(checked_counts),
        },
        "observed_final": {
            "style_ids": final_counts,
            "spread": _count_spread(final_counts),
        },
    }


def build_coverage_report(
    *,
    scenario_limit: int,
    coverage_sets: list[dict[str, Any]],
    coverage_mix: dict[str, dict[str, float]],
    coverage_assignments: list[dict[str, Any]],
    checked_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    assignments_by_set: dict[str, list[dict[str, Any]]] = {}
    for assignment in coverage_assignments:
        assignments_by_set.setdefault(str(assignment.get("coverage_set_id")), []).append(assignment)

    assigned_summary = {}
    for coverage_set in coverage_sets:
        set_id = str(coverage_set.get("set_id"))
        assignments = assignments_by_set.get(set_id, [])
        assigned_summary[set_id] = {
            "rows": len(assignments),
            "coverage_roles": _counter_dict([item.get("coverage_role") for item in assignments]),
            "target_property_concepts": _counter_dict(
                [item.get("target_property_concept") for item in assignments]
            ),
            "target_dynamic_concepts": _counter_dict(
                [item.get("target_dynamic_concept") for item in assignments]
            ),
            "target_object_ids": _counter_dict([item.get("target_object_id") for item in assignments]),
            "target_materials": _counter_dict([item.get("target_material") for item in assignments]),
            "target_anchors": _counter_dict([item.get("target_anchor") for item in assignments]),
        }

    return {
        "scenario_limit": int(scenario_limit),
        "requested": coverage_mix,
        "assigned": assigned_summary,
        "observed_checked": {
            str(coverage_set.get("set_id")): _coverage_observed_for_set(checked_rows, coverage_set)
            for coverage_set in coverage_sets
        },
        "observed_final": {
            str(coverage_set.get("set_id")): _coverage_observed_for_set(final_rows, coverage_set)
            for coverage_set in coverage_sets
        },
    }


def export_train_text(audited_jsonl: Path, output_txt: Path, drop_ids: set[str] | None = None) -> int:
    drop_ids = drop_ids or set()
    rows = read_jsonl(audited_jsonl)
    kept_texts = [
        str(row.get("text", "")).strip()
        for row in rows
        if str(row.get("id")) not in drop_ids and str(row.get("text", "")).strip()
    ]
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n\n".join(kept_texts) + ("\n" if kept_texts else ""), encoding="utf-8")
    return len(kept_texts)


def _stage_params(config: dict[str, Any], name: str) -> dict[str, Any]:
    stage = config.get("stages", {}).get(name, {})
    return {
        "temperature": float(stage.get("temperature", 0.2)),
        "max_tokens": int(stage.get("max_tokens", 1024)),
    }


def _call_generator(
    generator: Any,
    *,
    prompt: str,
    system_prompt: str,
    stage: str,
    params: dict[str, Any],
    response_format: dict[str, Any] | None = None,
) -> str:
    if NO_THINK_SUFFIX not in prompt:
        prompt = f"{prompt}\n\n{NO_THINK_SUFFIX}"
    return generator(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=params["temperature"],
        max_tokens=params["max_tokens"],
        stage=stage,
        response_format=response_format,
    )


def _safe_file_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "batch"


def _save_invalid_model_output(
    *,
    raw_failure_dir: Path,
    stage: str,
    batch_label: str,
    attempt: int,
    raw_output: str,
    error: Exception,
) -> Path:
    raw_failure_dir.mkdir(parents=True, exist_ok=True)
    path = raw_failure_dir / (
        f"{_safe_file_label(stage)}_{_safe_file_label(batch_label)}_"
        f"attempt_{attempt:02d}.txt"
    )
    path.write_text(
        f"Error: {error}\n\nRaw model output:\n{raw_output}",
        encoding="utf-8",
    )
    return path


def _expect_json_items(
    raw_output: str,
    *,
    expected_count: int,
    stage: str,
    batch_label: str,
) -> list[dict[str, Any]]:
    items = json_items(extract_json(raw_output))
    if len(items) != expected_count:
        raise ValueError(
            f"{stage} returned {len(items)} items for {batch_label}; expected {expected_count}."
        )
    return items


def _request_json_items(
    generator: Any,
    *,
    prompt: str,
    system_prompt: str,
    stage: str,
    params: dict[str, Any],
    response_format: dict[str, Any] | None,
    expected_count: int,
    batch_label: str,
    raw_failure_dir: Path,
    max_attempts: int = 2,
) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    last_saved_path: Path | None = None
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        raw_output = _call_generator(
            generator,
            prompt=prompt,
            system_prompt=system_prompt,
            stage=stage,
            params=params,
            response_format=response_format,
        )
        try:
            return _expect_json_items(
                raw_output,
                expected_count=expected_count,
                stage=stage,
                batch_label=batch_label,
            )
        except ValueError as exc:
            last_error = exc
            last_saved_path = _save_invalid_model_output(
                raw_failure_dir=raw_failure_dir,
                stage=stage,
                batch_label=batch_label,
                attempt=attempt,
                raw_output=raw_output,
                error=exc,
            )
    detail = f" Last raw output saved to {last_saved_path}." if last_saved_path else ""
    raise ValueError(
        f"{stage} returned invalid JSON for {batch_label} after {max_attempts} attempt(s).{detail}"
    ) from last_error


def _request_check_edit_items(
    generator: Any,
    *,
    row_chunk: list[dict[str, Any]],
    chunk_index: int,
    total_chunks: int,
    params: dict[str, Any],
    raw_failure_dir: Path,
    progress: Any | None = None,
) -> list[dict[str, Any]]:
    batch_label = f"batch_{chunk_index:04d}_of_{total_chunks:04d}"
    try:
        return _request_json_items(
            generator,
            prompt=build_check_prompt(row_chunk),
            system_prompt=EDITOR_SYSTEM_PROMPT,
            stage="check_edit",
            params=params,
            response_format={"type": "json_object"},
            expected_count=len(row_chunk),
            batch_label=batch_label,
            raw_failure_dir=raw_failure_dir,
            max_attempts=2,
        )
    except ValueError:
        if len(row_chunk) == 1:
            row = row_chunk[0]
            _emit_progress(
                progress,
                f"Check/edit {batch_label} failed JSON parsing; keeping original text for {row.get('id')}",
            )
            return [{"id": row.get("id"), "text": row.get("text", "")}]

    _emit_progress(
        progress,
        f"Check/edit {batch_label} returned invalid JSON; retrying one row at a time",
    )
    checked_items: list[dict[str, Any]] = []
    for row in row_chunk:
        row_id = str(row.get("id") or "unknown")
        try:
            checked_items.extend(
                _request_json_items(
                    generator,
                    prompt=build_check_prompt([row]),
                    system_prompt=EDITOR_SYSTEM_PROMPT,
                    stage="check_edit",
                    params=params,
                    response_format={"type": "json_object"},
                    expected_count=1,
                    batch_label=f"{batch_label}_{row_id}",
                    raw_failure_dir=raw_failure_dir,
                    max_attempts=2,
                )
            )
        except ValueError:
            _emit_progress(
                progress,
                f"Check/edit failed JSON parsing for {row_id}; keeping original text",
            )
            checked_items.append({"id": row_id, "text": row.get("text", "")})
    return checked_items


def _emit_progress(progress: Any | None, message: str, *, advance: bool = False) -> None:
    if progress is None:
        return
    progress(message, advance=advance)


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError(f"Batch size must be positive, got {size}.")
    return [items[index : index + size] for index in range(0, len(items), size)]


def batching_value(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get("batching", {}).get(key, default)
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"batching.{key} must be positive, got {value}.")
    return resolved


def run_fireworks_pipeline(
    *,
    config: dict[str, Any],
    generator: Any,
    model_id: str,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Run the probe pipeline and return the manifest payload."""

    _emit_progress(progress, "Loading material/object/style configs")
    material_config = load_json(resolve_path(config.get("material_classes_path", "configs/material_world/material_classes.json")))
    object_config = load_json(resolve_path(config.get("material_objects_path", "configs/material_world/material_objects.json")))
    coverage_set_ids = _coverage_set_ids_from_config(config)
    coverage_sets = []
    if coverage_set_ids:
        coverage_config = load_json(
            resolve_path(config.get("coverage_sets_path", str(DEFAULT_COVERAGE_SETS_CONFIG_PATH)))
        )
        coverage_sets = _coverage_sets_by_ids(coverage_config, coverage_set_ids)
    active_coverage_mix = coverage_mix_settings(config, coverage_sets)
    active_object_sampling = object_sampling_settings(config)
    active_style_sampling = style_sampling_settings(config)
    coverage_overrides = coverage_selection_overrides(
        config,
        material_config,
        object_config,
        coverage_sets,
        active_coverage_mix,
    )
    style_path = config.get("discourse_styles_path")
    discourse_styles = load_discourse_styles(resolve_path(style_path) if style_path else None)
    active_selection_seed = selection_seed(config)

    selected = select_probe_items(
        material_config,
        object_config,
        discourse_styles,
        material_class_limit=coverage_overrides["material_class_limit"],
        object_limit=coverage_overrides["object_limit"],
        style_limit=int(config.get("style_limit", 4)),
        material_class_ids=coverage_overrides["material_class_ids"],
        object_ids=coverage_overrides["object_ids"],
        style_ids=config.get("style_ids"),
    )
    if active_selection_seed is not None:
        selected = {
            "material_classes": seeded_shuffle(
                selected["material_classes"],
                seed=active_selection_seed,
                salt="material_classes",
            ),
            "objects": seeded_shuffle(
                selected["objects"],
                seed=active_selection_seed,
                salt="objects",
            ),
            "styles": seeded_shuffle(
                selected["styles"],
                seed=active_selection_seed,
                salt="styles",
            ),
        }
    scenario_limit = int(config.get("scenario_limit", 12))
    coverage_assignments = build_coverage_assignments(
        config=config,
        coverage_sets=coverage_sets,
        coverage_mix=active_coverage_mix,
        selected=selected,
        scenario_limit=scenario_limit,
        seed=active_selection_seed,
    )
    object_sampling_assignments = build_object_sampling_assignments(
        object_sampling=active_object_sampling,
        selected=selected,
        scenario_limit=scenario_limit,
        seed=active_selection_seed,
        existing_assignments=coverage_assignments,
    )
    style_sampling_assignments = build_style_sampling_assignments(
        style_sampling=active_style_sampling,
        selected=selected,
        scenario_limit=scenario_limit,
        seed=active_selection_seed,
    )
    row_assignments = merge_row_assignments(
        coverage_assignments,
        object_sampling_assignments,
        style_sampling_assignments,
    )
    coverage_assignments_by_id = {
        item["scenario_id"]: item
        for item in coverage_assignments
    }
    row_assignments_by_id = {
        item["scenario_id"]: item
        for item in row_assignments
    }
    audit_edit_limit = int(config.get("audit_edit_limit", 20))
    scenario_batch_size = batching_value(config, "scenario_plans_per_request", scenario_limit)
    render_batch_size = batching_value(config, "render_examples_per_request", scenario_limit)
    check_batch_size = batching_value(config, "check_examples_per_request", scenario_limit)
    audit_edit_batch_size = batching_value(config, "audit_edit_examples_per_request", 1)
    resume_existing = bool(config.get("resume_existing", False))

    if hasattr(generator, "configure_probe"):
        generator.configure_probe(
            selected_material_classes=selected["material_classes"],
            selected_objects=selected["objects"],
            selected_styles=selected["styles"],
            scenario_limit=scenario_limit,
        )

    out_dir = resolve_path(output_dir or config.get("output_dir", "data/material_world/fireworks_probe_v0"))
    out_dir.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress, f"Output directory: {out_dir}", advance=True)

    scenario_path = out_dir / "scenario_plans.jsonl"
    draft_path = out_dir / "prose_draft.jsonl"
    checked_path = out_dir / "prose_checked.jsonl"
    audited_path = out_dir / "examples_audited.jsonl"
    audit_report_path = out_dir / "audit_report.json"
    drop_ids_path = out_dir / "drop_ids.txt"
    audit_edited_path = out_dir / "prose_audit_edited.jsonl"
    reaudited_path = out_dir / "examples_reaudited.jsonl"
    audit_report_after_path = out_dir / "audit_report_after_edit.json"
    drop_ids_after_path = out_dir / "drop_ids_after_edit.txt"
    train_path = out_dir / "train_audited.txt"
    coverage_report_path = out_dir / "coverage_report.json"
    object_sampling_report_path = out_dir / "object_sampling_report.json"
    style_sampling_report_path = out_dir / "style_sampling_report.json"
    manifest_path = out_dir / "manifest.json"
    raw_failure_dir = out_dir / "_raw_model_failures"

    scenario_plans: list[dict[str, Any]] = []
    if resume_existing and scenario_path.exists():
        existing_scenario_plans = read_jsonl(scenario_path)
        if len(existing_scenario_plans) == scenario_limit:
            scenario_plans = existing_scenario_plans
            _emit_progress(
                progress,
                f"Reused {len(scenario_plans)} existing scenario plans",
                advance=True,
            )
        else:
            _emit_progress(
                progress,
                (
                    f"Ignoring existing scenario plans with {len(existing_scenario_plans)} rows; "
                    f"expected {scenario_limit}"
                ),
            )
    if not scenario_plans:
        scenario_params = _stage_params(config, "scenario_plans")
        for start_index in range(0, scenario_limit, scenario_batch_size):
            batch_size = min(scenario_batch_size, scenario_limit - start_index)
            batch_assignment_ids = {
                f"mw_probe_{index:04d}"
                for index in range(start_index, start_index + batch_size)
            }
            batch_row_assignments = [
                assignment
                for assignment in row_assignments
                if assignment["scenario_id"] in batch_assignment_ids
            ]
            scenario_prompt = build_scenario_plan_prompt(
                selected_material_classes=selected["material_classes"],
                selected_objects=selected["objects"],
                selected_styles=selected["styles"],
                scenario_limit=batch_size,
                scenario_start_index=start_index,
                coverage_sets=coverage_sets,
                coverage_mix=active_coverage_mix,
                selection_seed_value=active_selection_seed,
                coverage_assignments=batch_row_assignments,
            )
            started_at = time.perf_counter()
            _emit_progress(
                progress,
                (
                    f"Requesting scenario plans {start_index + 1}-"
                    f"{start_index + batch_size}/{scenario_limit} from Fireworks"
                ),
            )
            scenario_items = _request_json_items(
                generator,
                prompt=scenario_prompt,
                system_prompt=JSON_SYSTEM_PROMPT,
                stage="scenario_plans",
                params=scenario_params,
                response_format={"type": "json_object"},
                expected_count=batch_size,
                batch_label=f"{start_index + 1}-{start_index + batch_size}",
                raw_failure_dir=raw_failure_dir,
            )
            batch_plans = normalize_scenario_plans(scenario_items)
            for offset, plan in enumerate(batch_plans):
                plan["scenario_id"] = f"mw_probe_{start_index + offset:04d}"
            batch_plans = apply_coverage_assignments_to_plans(
                batch_plans,
                batch_row_assignments,
                material_config=material_config,
                object_config=object_config,
            )
            scenario_plans.extend(batch_plans)
            _emit_progress(
                progress,
                (
                    f"Received scenario plans {start_index + 1}-"
                    f"{start_index + batch_size}/{scenario_limit} in {time.perf_counter() - started_at:.1f}s"
                ),
            )
        write_jsonl(scenario_path, scenario_plans)
        _emit_progress(
            progress,
            f"Wrote {len(scenario_plans)} scenario plans",
            advance=True,
        )

    prose_draft_rows: list[dict[str, Any]] = []
    if resume_existing and draft_path.exists():
        existing_draft_rows = read_jsonl(draft_path)
        if len(existing_draft_rows) == scenario_limit:
            prose_draft_rows = existing_draft_rows
            _emit_progress(
                progress,
                f"Reused {len(prose_draft_rows)} existing draft prose examples",
                advance=True,
            )
        else:
            _emit_progress(
                progress,
                (
                    f"Ignoring existing draft prose with {len(existing_draft_rows)} rows; "
                    f"expected {scenario_limit}"
                ),
            )
    if not prose_draft_rows:
        rendered_items: list[dict[str, Any]] = []
        render_params = _stage_params(config, "prose_render")
        scenario_chunks = chunked(scenario_plans, render_batch_size)
        for chunk_index, plan_chunk in enumerate(scenario_chunks, start=1):
            render_prompt = build_render_prompt(plan_chunk)
            started_at = time.perf_counter()
            _emit_progress(
                progress,
                f"Rendering prose batch {chunk_index}/{len(scenario_chunks)} ({len(plan_chunk)} examples)",
            )
            rendered_batch = _request_json_items(
                generator,
                prompt=render_prompt,
                system_prompt=JSON_SYSTEM_PROMPT,
                stage="prose_render",
                params=render_params,
                response_format={"type": "json_object"},
                expected_count=len(plan_chunk),
                batch_label=f"batch_{chunk_index:04d}_of_{len(scenario_chunks):04d}",
                raw_failure_dir=raw_failure_dir,
            )
            rendered_items.extend(rendered_batch)
            _emit_progress(
                progress,
                f"Finished prose batch {chunk_index}/{len(scenario_chunks)} in {time.perf_counter() - started_at:.1f}s",
            )
        prose_draft_rows = build_prose_rows(scenario_plans, rendered_items)
        write_jsonl(draft_path, prose_draft_rows)
        _emit_progress(
            progress,
            f"Wrote {len(prose_draft_rows)} draft prose examples",
            advance=True,
        )

    checked_rows: list[dict[str, Any]] = []
    if resume_existing and checked_path.exists():
        existing_checked_rows = read_jsonl(checked_path)
        if len(existing_checked_rows) == scenario_limit:
            checked_rows = existing_checked_rows
            _emit_progress(
                progress,
                f"Reused {len(checked_rows)} existing checked examples",
                advance=True,
            )
        else:
            _emit_progress(
                progress,
                (
                    f"Ignoring existing checked prose with {len(existing_checked_rows)} rows; "
                    f"expected {scenario_limit}"
                ),
            )
    if not checked_rows:
        checked_items: list[dict[str, Any]] = []
        check_params = _stage_params(config, "check_edit")
        draft_chunks = chunked(prose_draft_rows, check_batch_size)
        for chunk_index, row_chunk in enumerate(draft_chunks, start=1):
            started_at = time.perf_counter()
            _emit_progress(
                progress,
                f"Running lightweight Qwen check/edit batch {chunk_index}/{len(draft_chunks)} ({len(row_chunk)} rows)",
            )
            checked_batch = _request_check_edit_items(
                generator,
                row_chunk=row_chunk,
                chunk_index=chunk_index,
                total_chunks=len(draft_chunks),
                params=check_params,
                raw_failure_dir=raw_failure_dir,
                progress=progress,
            )
            checked_items.extend(checked_batch)
            _emit_progress(
                progress,
                f"Finished check/edit batch {chunk_index}/{len(draft_chunks)} in {time.perf_counter() - started_at:.1f}s",
            )
        checked_rows = merge_text_edits(prose_draft_rows, checked_items)
        write_jsonl(checked_path, checked_rows)
        _emit_progress(
            progress,
            f"Wrote {len(checked_rows)} checked examples",
            advance=True,
        )

    started_at = time.perf_counter()
    _emit_progress(progress, "Running programmatic audit")
    audit_result = audit_jsonl_file(
        checked_path,
        output_jsonl=audited_path,
        report_json=audit_report_path,
        drop_ids_output=drop_ids_path,
        config_path=resolve_path(config.get("audit_config_path", str(DEFAULT_AUDIT_CONFIG_PATH))),
    )
    _emit_progress(
        progress,
        (
            "Audit complete "
            f"(warned={audit_result.report['examples_warned']}, dropped={audit_result.report['examples_dropped']}) "
            f"in {time.perf_counter() - started_at:.1f}s"
        ),
        advance=True,
    )

    audited_rows = read_jsonl(audited_path)
    drop_ids = set(audit_result.drop_ids)
    edit_candidates = [
        row
        for row in audited_rows
        if str(row.get("id")) not in drop_ids and has_row_level_edit_flag(row)
    ][:audit_edit_limit]
    edited_by_id: dict[str, str] = {}
    _emit_progress(progress, f"Running audit-guided edits for {len(edit_candidates)} flagged rows")
    edit_chunks = chunked(edit_candidates, audit_edit_batch_size) if edit_candidates else []
    edited_count = 0
    for chunk_index, row_chunk in enumerate(edit_chunks, start=1):
        first_row_index = edited_count + 1
        last_row_index = edited_count + len(row_chunk)
        started_at = time.perf_counter()
        if len(row_chunk) == 1:
            prompt = build_audit_edit_prompt(row_chunk[0])
            progress_label = f"flagged row {first_row_index}/{len(edit_candidates)}: {row_chunk[0].get('id')}"
        else:
            prompt = build_audit_edit_batch_prompt(row_chunk)
            progress_label = f"flagged rows {first_row_index}-{last_row_index}/{len(edit_candidates)}"
        _emit_progress(progress, f"Editing {progress_label}")
        raw_edit = _call_generator(
            generator,
            prompt=prompt,
            system_prompt=EDITOR_SYSTEM_PROMPT,
            stage="audit_edit",
            params=_stage_params(config, "audit_edit"),
            response_format={"type": "json_object"},
        )
        try:
            parsed_edit = extract_json(raw_edit)
        except ValueError:
            edited_count += len(row_chunk)
            continue
        if len(row_chunk) == 1 and isinstance(parsed_edit, dict) and "items" not in parsed_edit:
            edit_items = [parsed_edit]
        else:
            edit_items = json_items(parsed_edit)
            if len(edit_items) != len(row_chunk):
                raise ValueError(
                    f"Audit edit returned {len(edit_items)} items for a batch of {len(row_chunk)} rows."
                )
        for item in edit_items:
            if isinstance(item, dict) and str(item.get("id", "")).strip() and str(item.get("text", "")).strip():
                edited_by_id[str(item["id"])] = str(item["text"]).strip()
        edited_count += len(row_chunk)
        _emit_progress(
            progress,
            f"Finished {progress_label} in {time.perf_counter() - started_at:.1f}s",
        )

    audit_edited_rows = []
    for row in audited_rows:
        if str(row.get("id")) in drop_ids:
            continue
        clean = strip_audit_fields(row)
        if str(clean["id"]) in edited_by_id:
            clean["text"] = edited_by_id[str(clean["id"])]
        audit_edited_rows.append(clean)
    write_jsonl(audit_edited_path, audit_edited_rows)
    _emit_progress(progress, f"Wrote {len(audit_edited_rows)} audit-edited rows", advance=True)

    started_at = time.perf_counter()
    _emit_progress(progress, "Running re-audit after edits")
    reaudited_result = audit_jsonl_file(
        audit_edited_path,
        output_jsonl=reaudited_path,
        report_json=audit_report_after_path,
        drop_ids_output=drop_ids_after_path,
        config_path=resolve_path(config.get("audit_config_path", str(DEFAULT_AUDIT_CONFIG_PATH))),
    )
    _emit_progress(
        progress,
        (
            "Re-audit complete "
            f"(warned={reaudited_result.report['examples_warned']}, dropped={reaudited_result.report['examples_dropped']}) "
            f"in {time.perf_counter() - started_at:.1f}s"
        ),
        advance=True,
    )
    final_count = export_train_text(reaudited_path, train_path, set(reaudited_result.drop_ids))
    _emit_progress(progress, f"Exported {final_count} final prose examples", advance=True)

    reaudited_rows = read_jsonl(reaudited_path)
    final_rows = [
        row
        for row in reaudited_rows
        if str(row.get("id")) not in set(reaudited_result.drop_ids)
    ]
    coverage_report = build_coverage_report(
        scenario_limit=scenario_limit,
        coverage_sets=coverage_sets,
        coverage_mix=active_coverage_mix,
        coverage_assignments=coverage_assignments,
        checked_rows=checked_rows,
        final_rows=final_rows,
    )
    write_json(coverage_report_path, coverage_report)
    object_sampling_report = build_object_sampling_report(
        object_sampling=active_object_sampling,
        selected_objects=selected["objects"],
        object_sampling_assignments=object_sampling_assignments,
        checked_rows=checked_rows,
        final_rows=final_rows,
    )
    write_json(object_sampling_report_path, object_sampling_report)
    style_sampling_report = build_style_sampling_report(
        style_sampling=active_style_sampling,
        selected_styles=selected["styles"],
        style_sampling_assignments=style_sampling_assignments,
        checked_rows=checked_rows,
        final_rows=final_rows,
    )
    write_json(style_sampling_report_path, style_sampling_report)

    manifest = {
        "version": config.get("version", "material_world_fireworks_probe_v0"),
        "created_at": utc_now_iso(),
        "model_id": model_id,
        "config_path": str(config_path) if config_path else None,
        "output_dir": str(out_dir),
        "resume_existing": resume_existing,
        "selected": {
            "material_classes": [item.get("class_id") for item in selected["material_classes"]],
            "objects": [item.get("object_id") for item in selected["objects"]],
            "styles": [item.get("style_id") for item in selected["styles"]],
            "coverage_sets": [item.get("set_id") for item in coverage_sets],
            "coverage_mix": active_coverage_mix,
            "selection_seed": active_selection_seed,
            "object_sampling": active_object_sampling,
            "style_sampling": active_style_sampling,
            "coverage_assignments": len(coverage_assignments_by_id),
            "row_assignments": len(row_assignments_by_id),
        },
        "batching": {
            "scenario_plans_per_request": scenario_batch_size,
            "render_examples_per_request": render_batch_size,
            "check_examples_per_request": check_batch_size,
            "audit_edit_examples_per_request": audit_edit_batch_size,
        },
        "counts": {
            "scenario_plans": len(scenario_plans),
            "prose_draft": len(prose_draft_rows),
            "prose_checked": len(checked_rows),
            "audit_edit_candidates": len(edit_candidates),
            "after_first_audit_dropped": len(audit_result.drop_ids),
            "after_second_audit_dropped": len(reaudited_result.drop_ids),
            "final_train_examples": final_count,
            "coverage_assignments": len(coverage_assignments),
            "object_sampling_assignments": len(object_sampling_assignments),
            "style_sampling_assignments": len(style_sampling_assignments),
            "row_assignments": len(row_assignments),
        },
        "paths": {
            "scenario_plans": str(scenario_path),
            "prose_draft": str(draft_path),
            "prose_checked": str(checked_path),
            "examples_audited": str(audited_path),
            "audit_report": str(audit_report_path),
            "drop_ids": str(drop_ids_path),
            "prose_audit_edited": str(audit_edited_path),
            "examples_reaudited": str(reaudited_path),
            "audit_report_after_edit": str(audit_report_after_path),
            "drop_ids_after_edit": str(drop_ids_after_path),
            "train_audited": str(train_path),
            "coverage_report": str(coverage_report_path),
            "object_sampling_report": str(object_sampling_report_path),
            "style_sampling_report": str(style_sampling_report_path),
            "manifest": str(manifest_path),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


class DeterministicMaterialWorldMockGenerator:
    """No-network generator for smoke tests and CLI demos."""

    def __init__(self) -> None:
        self.selected_material_classes: list[dict[str, Any]] = []
        self.selected_objects: list[dict[str, Any]] = []
        self.selected_styles: list[dict[str, Any]] = []
        self.scenario_limit = 12
        self._plans: list[dict[str, Any]] = []

    def configure_probe(
        self,
        *,
        selected_material_classes: list[dict[str, Any]],
        selected_objects: list[dict[str, Any]],
        selected_styles: list[dict[str, Any]],
        scenario_limit: int,
    ) -> None:
        self.selected_material_classes = selected_material_classes
        self.selected_objects = selected_objects
        self.selected_styles = selected_styles
        self.scenario_limit = scenario_limit

    def _make_plans(self) -> list[dict[str, Any]]:
        if self._plans:
            return self._plans
        plans = []
        material_classes = self.selected_material_classes or [{}]
        objects = self.selected_objects or [{}]
        styles = self.selected_styles or [{}]
        for index in range(self.scenario_limit):
            object_index = index % len(objects)
            material_index = (index // len(objects)) % len(material_classes)
            material_class = material_classes[material_index]
            object_card = objects[object_index]
            examples = as_list(material_class.get("examples"))
            dynamics = as_list(material_class.get("core_dynamics"))
            properties = as_list(material_class.get("core_properties"))
            events = as_list(object_card.get("useful_events"))
            style = styles[index % len(styles)]
            material = examples[object_index % max(1, len(examples))] if examples else "material"
            dynamic = dynamics[object_index % max(1, len(dynamics))] if dynamics else "changes"
            event = events[material_index % max(1, len(events))] if events else "used"
            plans.append(
                {
                    "scenario_id": f"mw_probe_{index:04d}",
                    "material_class_id": material_class.get("class_id"),
                    "material_class_name": material_class.get("class_name"),
                    "material": material,
                    "object_id": object_card.get("object_id"),
                    "object": object_card.get("object_name"),
                    "event": event,
                    "properties": properties[:2],
                    "dynamics": [dynamic],
                    "discourse_style_id": style.get("style_id"),
                    "contains_negative_reporting": index % 4 == 1,
                    "plan_notes": "Mock probe scenario for testing the script path.",
                }
            )
        self._plans = plans
        return plans

    def _text_for_plan(self, plan: dict[str, Any]) -> str:
        material = plan.get("material", "material")
        obj = plan.get("object", "object")
        event = plan.get("event", "used")
        dynamic = ", ".join(as_list(plan.get("dynamics"))) or "changes"
        prop = ", ".join(as_list(plan.get("properties"))[:2]) or "has useful properties"
        style = plan.get("discourse_style_id")
        anchor = plan.get("target_anchor")
        anchor_text = (
            f" The {anchor} is part of the event itself, so its visible behavior helps show the material change."
            if anchor
            else ""
        )
        contrast = (
            f" It is not described as doing every possible thing; the scene focuses on how {material} can {dynamic}."
            if plan.get("contains_negative_reporting")
            else ""
        )

        if style == "comparison_paragraph":
            return (
                f"A {material} {obj} handles being {event} in a material-specific way. "
                f"The useful point is not just the object name, but the material: {material} is often {prop}, "
                f"so the {obj} may show behavior such as {dynamic} during the event. Another material could change the outcome, "
                f"even when the same {obj} is involved.{anchor_text}{contrast}"
            )
        if style == "child_science_explanation":
            return (
                f"You can learn something from a {material} {obj} when it is {event}. "
                f"The {material} is often {prop}, and that helps explain why the {obj} may show behavior such as {dynamic}. "
                f"This is an everyday kind of change that can be seen without special tools. "
                f"The object still matters, but the material helps decide what happens.{anchor_text}{contrast}"
            )
        if style == "repair_or_craft_note":
            return (
                f"For a simple repair or craft choice, a {material} {obj} should be judged by what happens when it is {event}. "
                f"Since {material} is often {prop}, the part may show behavior such as {dynamic} instead of behaving like a stiffer or softer material. "
                f"That makes the material useful in some situations and less useful in others.{anchor_text}{contrast}"
            )
        return (
            f"During an ordinary day, someone notices a {material} {obj} after it is {event}. "
            f"The {material} is often {prop}, so the {obj} may show behavior such as {dynamic} rather than act like every other version of the same object. "
            f"The small event makes the material behavior easy to see in a familiar setting.{anchor_text}{contrast}"
        )

    def _prompt_plan_ids(self, prompt: str) -> list[str]:
        seen = set()
        ids = []
        for item_id in re.findall(r"mw_probe_\d{4}", prompt):
            if item_id in seen:
                continue
            seen.add(item_id)
            ids.append(item_id)
        return ids

    def _prompt_row_ids(self, prompt: str) -> list[str]:
        seen = set()
        ids = []
        for item_id in re.findall(r'"id":\s*"(mw_probe_\d{4})"', prompt):
            if item_id in seen:
                continue
            seen.add(item_id)
            ids.append(item_id)
        return ids

    def _plans_for_prompt(self, prompt: str) -> list[dict[str, Any]]:
        marker = "Scenario plans:"
        if marker in prompt:
            raw_plans = prompt.rsplit(marker, 1)[1].strip()
            if raw_plans.endswith(NO_THINK_SUFFIX):
                raw_plans = raw_plans[: -len(NO_THINK_SUFFIX)].strip()
            try:
                parsed = json.loads(raw_plans)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                return parsed
        plans = self._make_plans()
        by_id = {str(plan["scenario_id"]): plan for plan in plans}
        ids = [item_id for item_id in self._prompt_plan_ids(prompt) if item_id in by_id]
        if ids:
            return [by_id[item_id] for item_id in ids]
        return plans

    def _check_rows_for_prompt(self, prompt: str) -> list[dict[str, Any]]:
        marker = "Rows:"
        if marker not in prompt:
            return []
        raw_rows = prompt.rsplit(marker, 1)[1].strip()
        if raw_rows.endswith(NO_THINK_SUFFIX):
            raw_rows = raw_rows[: -len(NO_THINK_SUFFIX)].strip()
        try:
            parsed = json.loads(raw_rows)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            return parsed
        return []

    def __call__(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        stage: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        del system_prompt, temperature, max_tokens, response_format
        plans = self._make_plans()
        if stage == "scenario_plans":
            count_match = re.search(r"Create exactly (\d+)", prompt)
            start_match = re.search(r"scenario_id values from mw_probe_(\d{4})", prompt)
            count = int(count_match.group(1)) if count_match else len(plans)
            start = int(start_match.group(1)) if start_match else 0
            return json.dumps({"items": plans[start : start + count]})
        if stage == "prose_render":
            prompt_plans = self._plans_for_prompt(prompt)
            return json.dumps(
                {
                    "items": [
                        {
                            "scenario_id": plan["scenario_id"],
                            "id": plan["scenario_id"],
                            "discourse_style_id": plan["discourse_style_id"],
                            "text": self._text_for_plan(plan),
                        }
                        for plan in prompt_plans
                    ]
                }
            )
        if stage == "check_edit":
            prompt_rows = self._check_rows_for_prompt(prompt)
            if prompt_rows:
                items = []
                for row in prompt_rows:
                    text = str(row.get("text", ""))
                    context = row.get("context", {})
                    anchor = context.get("target_anchor") if isinstance(context, dict) else None
                    if anchor and str(anchor).lower() not in text.lower():
                        text = (
                            f"{text} The {anchor} is part of the event itself, so its visible behavior helps show the material change."
                        )
                    items.append({"id": row.get("id"), "text": text})
                return json.dumps({"items": items})
            prompt_plans = self._plans_for_prompt(prompt)
            return json.dumps(
                {
                    "items": [
                        {
                            "scenario_id": plan["scenario_id"],
                            "id": plan["scenario_id"],
                            "discourse_style_id": plan["discourse_style_id"],
                            "text": self._text_for_plan(plan),
                        }
                        for plan in prompt_plans
                    ]
                }
            )
        if stage == "audit_edit":
            by_id = {str(plan["scenario_id"]): plan for plan in plans}
            ids = self._prompt_row_ids(prompt) or ["unknown"]
            items = []
            for row_id in ids:
                plan = by_id.get(row_id, {})
                material = plan.get("material", "material")
                obj = plan.get("object", "object")
                event = plan.get("event", "used")
                dynamic = ", ".join(as_list(plan.get("dynamics"))) or "change"
                items.append(
                    {
                        "id": row_id,
                        "text": (
                            f"After a closer look, the {material} {obj} keeps the same everyday scene when it is {event}. "
                            f"The wording stays cautious: this material can show behavior such as {dynamic}, and the example "
                            "focuses on that visible change without turning into a quiz or a technical explanation."
                        ),
                    }
                )
            if len(items) == 1:
                return json.dumps(items[0])
            return json.dumps({"items": items})
        raise ValueError(f"Unknown mock generation stage: {stage}")
