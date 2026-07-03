"""Small concept-curriculum generation pipeline.

This pipeline is intentionally lighter than the material-world generator. It
uses a hand-authored concept profile to plan a small set of teaching documents,
renders them with a chat model, audits the result deterministically, and exports
plain text for token-block mixing.
"""

from __future__ import annotations

import json
import random
import re
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from babysteps.concept_curriculum.audit import audit_documents


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = (
    ROOT
    / "configs"
    / "concept_curriculum"
    / "physical_dynamics"
    / "physical_dynamics_ewok_micro_500.json"
)
NO_THINK_SUFFIX = "/no_think"


@dataclass(frozen=True)
class ScenarioSpec:
    """One concept-curriculum document request."""

    id: str
    concept_id: str
    domain: str
    coverage_pack_id: str
    document_type: str
    definition: str
    centrality_rule: str
    lexical_forms: list[str]
    positive_subtype_id: str | None = None
    positive_subtype_label: str | None = None
    positive_material: str | None = None
    positive_mechanism: str | None = None
    negative_subtype_id: str | None = None
    negative_subtype_label: str | None = None
    negative_material: str | None = None
    negative_reason: str | None = None
    state_case_id: str | None = None
    before_material: str | None = None
    after_material: str | None = None
    state_lesson: str | None = None
    neighbor: str | None = None
    neighbor_relation: str | None = None
    style_id: str | None = None
    style_name: str | None = None
    style_description: str | None = None
    style_guidance: list[str] | None = None
    style_required_terms: list[str] | None = None
    style_min_required_terms: int | None = None


CONCEPT_DOCUMENT_TYPES = (
    "direct_positive",
    "subtype_explanation",
    "state_contrast",
    "neighbor_contrast",
)
CORRELATION_DOCUMENT_TYPES = (
    "material_affordance_cluster",
    "single_material_actions",
    "simple_scene_sequence",
    "class_action_map",
)
PHYSICAL_DYNAMICS_DOCUMENT_TYPES = (
    "causal_scene",
    "state_transition",
    "contrast_pair",
    "observation_log",
)
PHYSICAL_PROMPT_METADATA_KEYS = frozenset(
    {
        "canonical_concept_id",
        "canonical_contrast_concept_id",
        "canonical_balance_role",
        "canonical_inventory_version",
    }
)
PHYSICAL_CONTRAST_PROMPT_KEYS = frozenset(
    {
        "contrast_process",
        "contrast_process_terms",
        "preferred_contrast_terms",
    }
)
PHYSICAL_METADATA_ONLY_PROMPT_KEYS = frozenset(
    {
        "concept_pair",
        "family",
        "physical_family",
        "family_label",
        "family_schema",
        "scenario_id",
    }
)
PHYSICAL_RENDER_PROMPT_KEYS = (
    "id",
    "document_type",
    "style_id",
    "style_name",
    "style_description",
    "style_guidance",
    "style_required_terms",
    "style_min_required_terms",
    "render_template",
    "template_sequence",
    "cause_type",
    "object_set_id",
    "object_type",
    "object",
    "object_shape",
    "body_pose",
    "material_or_surface",
    "contact_surface",
    "surface_friction",
    "rotation_allowed",
    "rotation_evidence",
    "motion_pattern",
    "geometry_or_medium",
    "timescale",
    "path_shape",
    "start_position",
    "end_position",
    "direction_evidence",
    "vertical_relation_language",
    "directional_polarity",
    "measurement_frame",
    "initial_speed_evidence",
    "later_speed_evidence",
    "initial_size_evidence",
    "later_size_evidence",
    "size_change_evidence",
    "quantity_change",
    "rate_evidence",
    "endpoint",
    "endpoint_tendency",
    "endpoint_risk",
    "stopping_evidence",
    "speed_geometry",
    "bridge_type",
    "bridge_clause",
    "evidence_clause",
    "same_sentence_evidence_terms",
    "same_sentence_target_terms",
    "contrast_evidence_bans",
    "reference_frame",
    "landmark",
    "initial_relation_to_landmark",
    "later_relation_to_landmark",
    "crossing_direction",
    "verb_type",
    "event_verb",
    "mechanism",
    "expected_process",
    "cause_evidence",
    "outcome_evidence",
    "required_cause_terms",
    "required_process_terms",
    "required_outcome_terms",
    "lexical_targets",
    "lexical_target",
    "preferred_process_terms",
    "shortcut_terms",
)
PHYSICAL_EXPLICIT_CONTRAST_RENDER_PROMPT_KEYS = (
    "family",
    "physical_family",
    "concept_pair",
    "contrast_process",
    "preferred_contrast_terms",
)
PHYSICAL_CONTRAST_VERBALIZATION_MODES = frozenset({"metadata_only", "implicit", "explicit"})
PHYSICAL_COMPARISON_FRAME_BANS = (
    "instead of",
    "rather than",
    "not X but Y",
    "unlike",
    "as opposed to",
)
PHYSICAL_POSITIVE_ONLY_LEXICAL_HYGIENE = (
    "Positive-only lexical hygiene: do not use negated opposite-event wording, "
    "even when it seems explanatory. Avoid phrases such as \"without sinking\", "
    "\"did not sink\", \"did not go under\", \"instead of floating\", "
    "\"not floating\", \"rather than slowing down\", or \"did not accelerate\". "
    "For floating examples, use positive surface evidence such as stayed on "
    "the surface, stayed visible, stayed level, or partly above the water. "
    "For sinking examples, use positive downward evidence such as below the "
    "surface, fully covered, deeper, lower, or reached the bottom. For speed "
    "examples, use positive speed evidence such as increasing speed, moved "
    "faster, decreasing speed, moved slower, or covered less distance.\n"
)
PHYSICAL_SPEED_CHANGE_FAMILIES = frozenset(
    {
        "force_speed",
        "speed_change_language_expansion",
        "speed_quantity_change_cues",
        "speed_endpoint_tendency_cues",
        "speed_rate_definition_bridge",
    }
)


def load_pipeline_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a JSON or YAML concept-curriculum config."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        return json.load(handle)


def _cycle(items: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot sample from an empty profile list.")
    return items[index % len(items)]


def _cycle_string(items: list[str], index: int) -> str:
    if not items:
        raise ValueError("Cannot sample from an empty string list.")
    return str(items[index % len(items)])


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower() or "concept"


def _active_profile(config: dict[str, Any]) -> dict[str, Any]:
    profile = (
        config.get("concept_profile")
        or config.get("material_class_profile")
        or config.get("physical_dynamics_profile")
    )
    if not isinstance(profile, dict):
        raise ValueError(
            "Config must include concept_profile, material_class_profile, or "
            "physical_dynamics_profile."
        )
    return dict(profile)


def _profile_id(profile: dict[str, Any]) -> str:
    return str(profile.get("concept_id") or profile.get("profile_id") or "material_affordance")


def _make_spec(
    *,
    index: int,
    document_type: str,
    coverage_pack_id: str,
    concept_profile: dict[str, Any],
    subtype_offset: int,
    state_offset: int,
    neighbor_offset: int,
) -> ScenarioSpec:
    concept_id = str(concept_profile["concept_id"])
    positive_subtypes = list(concept_profile.get("positive_subtypes", []))
    negative_subtypes = list(concept_profile.get("negative_subtypes", []))
    state_contrasts = list(concept_profile.get("state_contrasts", []))
    neighbors = list(concept_profile.get("neighbors", []))
    lexical_forms = [str(item) for item in concept_profile.get("lexical_forms", [])]

    positive = _cycle(positive_subtypes, subtype_offset)
    positive_materials = [str(item) for item in positive.get("materials", [])]
    negative = _cycle(negative_subtypes, neighbor_offset)
    negative_materials = [str(item) for item in negative.get("materials", [])]
    state = _cycle(state_contrasts, state_offset)
    neighbor = _cycle(neighbors, neighbor_offset)

    spec = {
        "id": f"cc_{_safe_id(concept_id)}_{index:04d}",
        "concept_id": concept_id,
        "domain": str(concept_profile.get("domain", "")),
        "coverage_pack_id": coverage_pack_id,
        "document_type": document_type,
        "definition": str(concept_profile.get("definition", "")),
        "centrality_rule": str(concept_profile.get("centrality_rule", "")),
        "lexical_forms": lexical_forms,
    }

    if document_type in {"direct_positive", "subtype_explanation", "neighbor_contrast"}:
        spec.update(
            {
                "positive_subtype_id": positive.get("id"),
                "positive_subtype_label": positive.get("label"),
                "positive_material": _cycle_string(positive_materials, subtype_offset),
                "positive_mechanism": positive.get("mechanism"),
            }
        )

    if document_type == "state_contrast":
        spec.update(
            {
                "state_case_id": state.get("id"),
                "before_material": state.get("before_material"),
                "after_material": state.get("after_material"),
                "state_lesson": state.get("lesson"),
            }
        )

    if document_type == "neighbor_contrast":
        spec.update(
            {
                "negative_subtype_id": negative.get("id"),
                "negative_subtype_label": negative.get("label"),
                "negative_material": _cycle_string(negative_materials, neighbor_offset),
                "negative_reason": negative.get("reason"),
                "neighbor": neighbor.get("label") or neighbor.get("id"),
                "neighbor_relation": neighbor.get("relation"),
            }
        )

    return ScenarioSpec(**spec)


def _cycle_values(items: list[str], start: int, count: int) -> list[str]:
    if not items:
        raise ValueError("Cannot sample from an empty string list.")
    return [str(items[(start + offset) % len(items)]) for offset in range(count)]


def _class_tags(row: dict[str, Any]) -> set[str]:
    return {str(tag) for tag in row.get("tags", []) if str(tag).strip()}


def _filtered_material_classes(config: dict[str, Any]) -> list[dict[str, Any]]:
    profile = dict(config["material_class_profile"])
    classes = [dict(row) for row in profile.get("material_classes", [])]
    focus_tags = {str(tag) for tag in config.get("focus_tags", []) if str(tag).strip()}
    if focus_tags:
        classes = [row for row in classes if focus_tags & _class_tags(row)]
    if not classes:
        tag_text = ", ".join(sorted(focus_tags)) if focus_tags else "configured classes"
        raise ValueError(f"No material classes matched {tag_text}.")
    return classes


def _target_affordances(profile: dict[str, Any]) -> list[str]:
    return [str(item) for item in profile.get("target_affordances", []) if str(item).strip()]


def _class_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("label") or "material_class")


def _supports_affordance(row: dict[str, Any], affordance: str) -> bool:
    return affordance in {str(item) for item in row.get("affordances", [])}


def _choose_class_for_affordance(
    classes: list[dict[str, Any]],
    *,
    affordance: str,
    action_class_counts: dict[tuple[str, str], int],
    class_counts: dict[str, int],
    tie_offset: int,
) -> dict[str, Any]:
    del tie_offset
    supporting = [row for row in classes if _supports_affordance(row, affordance)]
    if not supporting:
        raise ValueError(f"No material class supports target affordance {affordance!r}.")

    min_action_count = min(
        action_class_counts.get((affordance, _class_id(row)), 0) for row in supporting
    )
    candidates = [
        row
        for row in supporting
        if action_class_counts.get((affordance, _class_id(row)), 0) == min_action_count
    ]

    min_global_count = min(class_counts.get(_class_id(row), 0) for row in candidates)
    candidates = [
        row for row in candidates if class_counts.get(_class_id(row), 0) == min_global_count
    ]
    return candidates[0]


def _make_material_class_spec(
    *,
    index: int,
    document_type: str,
    coverage_pack_id: str,
    profile: dict[str, Any],
    material_class: dict[str, Any],
    material_offset: int,
    affordance_offset: int,
    target_affordance: str | None = None,
) -> dict[str, Any]:
    profile_id = _profile_id(profile)
    materials = [str(item) for item in material_class.get("materials", [])]
    affordances = [str(item) for item in material_class.get("affordances", [])]
    required_count = min(
        len(affordances),
        int(profile.get("required_affordances_per_document", 3)),
    )
    if target_affordance:
        if target_affordance not in affordances:
            raise ValueError(
                f"Material class {_class_id(material_class)!r} does not support "
                f"target affordance {target_affordance!r}."
            )
        remaining_affordances = [item for item in affordances if item != target_affordance]
        required_affordances = [target_affordance]
        if required_count > 1 and remaining_affordances:
            required_affordances.extend(
                _cycle_values(remaining_affordances, affordance_offset, required_count - 1)
            )
    else:
        required_affordances = _cycle_values(affordances, affordance_offset, max(required_count, 1))
    material = _cycle_string(materials, material_offset)

    spec = {
        "id": f"cc_{_safe_id(profile_id)}_{index:04d}",
        "concept_id": profile_id,
        "profile_id": profile_id,
        "domain": str(profile.get("domain", "")),
        "coverage_pack_id": coverage_pack_id,
        "document_type": document_type,
        "definition": str(profile.get("definition", "")),
        "centrality_rule": str(profile.get("centrality_rule", "")),
        "lexical_forms": [],
        "material_class_id": material_class.get("id"),
        "material_class_label": material_class.get("label"),
        "material": material,
        "materials": materials,
        "affordances": affordances,
        "target_affordance": target_affordance,
        "required_affordances": required_affordances,
        "required_materials": [material],
        "class_notes": material_class.get("notes"),
        "tags": list(material_class.get("tags", [])),
        "ewok_anchors": list(material_class.get("ewok_anchors", [])),
    }

    return spec


def build_material_class_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build correlation-first specs from material class profiles."""

    profile = dict(config["material_class_profile"])
    classes = _filtered_material_classes(config)
    specs: list[dict[str, Any]] = []
    class_cursor = 0
    affordance_cursor = 0
    material_cursors: dict[tuple[str | None, str], int] = {}
    action_class_counts: dict[tuple[str, str], int] = {}
    class_counts: dict[str, int] = {}
    target_affordances = _target_affordances(profile)

    for pack in config.get("coverage_packs", []):
        coverage_pack_id = str(pack["id"])
        counts = dict(pack.get("document_counts", {}))
        for document_type in CORRELATION_DOCUMENT_TYPES:
            count = int(counts.get(document_type, 0))
            for _ in range(count):
                target_affordance = (
                    target_affordances[len(specs) % len(target_affordances)]
                    if target_affordances
                    else None
                )
                material_class = (
                    _choose_class_for_affordance(
                        classes,
                        affordance=target_affordance,
                        action_class_counts=action_class_counts,
                        class_counts=class_counts,
                        tie_offset=class_cursor,
                    )
                    if target_affordance
                    else _cycle(classes, class_cursor)
                )
                material_class_id = _class_id(material_class)
                material_cursor_key = (target_affordance, material_class_id)
                material_offset = material_cursors.get(material_cursor_key, 0)
                specs.append(
                    _make_material_class_spec(
                        index=len(specs),
                        document_type=document_type,
                        coverage_pack_id=coverage_pack_id,
                        profile=profile,
                        material_class=material_class,
                        material_offset=material_offset,
                        affordance_offset=affordance_cursor,
                        target_affordance=target_affordance,
                    )
                )
                material_cursors[material_cursor_key] = material_offset + 1
                if target_affordance:
                    action_count_key = (target_affordance, material_class_id)
                    action_class_counts[action_count_key] = (
                        action_class_counts.get(action_count_key, 0) + 1
                    )
                class_counts[material_class_id] = class_counts.get(material_class_id, 0) + 1
                class_cursor += 1
                affordance_cursor += 1

    specs = _assign_discourse_styles(specs, list(config.get("discourse_styles", [])))
    return _assign_signal_orders(specs, list(config.get("signal_orders", [])))


def _physical_family_id(family: dict[str, Any]) -> str:
    return str(family.get("family_id") or family.get("id") or family.get("label") or "family")


def _physical_document_types(profile: dict[str, Any]) -> list[str]:
    configured = profile.get("document_types", [])
    document_types = []
    for item in configured:
        if isinstance(item, dict):
            value = item.get("id") or item.get("document_type")
        else:
            value = item
        if value is not None and str(value).strip():
            document_types.append(str(value))
    return document_types or list(PHYSICAL_DYNAMICS_DOCUMENT_TYPES)


def _physical_family_source_entry(source: Any) -> tuple[Path, set[str]]:
    if isinstance(source, str):
        return Path(source), set()
    if not isinstance(source, dict):
        raise TypeError("physical_dynamics family_source_configs entries must be strings or objects.")

    path_value = source.get("path") or source.get("config")
    if not path_value:
        raise ValueError("physical_dynamics family source entry must include path or config.")

    family_ids = {str(item) for item in source.get("family_ids", []) if str(item).strip()}
    return Path(str(path_value)), family_ids


def _resolve_physical_family_source(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def _resolve_physical_inventory_source(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def _physical_profile_families(
    profile: dict[str, Any],
    *,
    _seen: set[Path] | None = None,
) -> list[dict[str, Any]]:
    """Return inline and imported physical-dynamics families."""

    families = [dict(row) for row in profile.get("families", [])]
    seen = set(_seen or set())

    for source in profile.get("family_source_configs", []):
        source_path, family_ids = _physical_family_source_entry(source)
        resolved = _resolve_physical_family_source(source_path).resolve()
        if resolved in seen:
            raise ValueError(f"Recursive physical-dynamics family source: {resolved}")
        seen.add(resolved)
        source_config = load_pipeline_config(resolved)
        source_profile = dict(source_config.get("physical_dynamics_profile", {}))
        if not source_profile:
            raise ValueError(f"Family source {resolved} has no physical_dynamics_profile.")
        source_families = _physical_profile_families(source_profile, _seen=seen)
        for family in source_families:
            if family_ids and _physical_family_id(family) not in family_ids:
                continue
            families.append(dict(family))

    return families


def _physical_terms(
    scenario: dict[str, Any],
    key: str,
    fallback: str | None = None,
) -> list[str]:
    terms = [str(item) for item in scenario.get(key, []) if str(item).strip()]
    if terms:
        return terms
    if fallback and scenario.get(fallback):
        return [str(scenario[fallback])]
    return []


def _physical_contrast_verbalization(
    *,
    physical_dynamics_profile: dict[str, Any],
    constraints: dict[str, Any],
) -> str:
    """Return how much the renderer should verbalize contrast-process metadata."""

    raw_mode = constraints.get(
        "contrast_verbalization",
        physical_dynamics_profile.get("contrast_verbalization", "explicit"),
    )
    mode = str(raw_mode).strip().lower()
    if mode not in PHYSICAL_CONTRAST_VERBALIZATION_MODES:
        allowed = ", ".join(sorted(PHYSICAL_CONTRAST_VERBALIZATION_MODES))
        raise ValueError(f"contrast_verbalization must be one of: {allowed}.")
    return mode


def _physical_speed_direction(scenario: dict[str, Any]) -> str:
    """Return speed-increase or speed-decrease polarity for speed scenarios."""

    polarity = str(scenario.get("directional_polarity") or "").strip().lower()
    expected_process = str(scenario.get("expected_process") or "").strip().lower()
    if polarity == "speed_increase" or expected_process in {
        "accelerating",
        "accelerated",
        "speeding up",
        "increasing speed",
    }:
        return "increase"
    if polarity == "speed_decrease" or expected_process in {
        "slowing down",
        "slowed down",
        "losing speed",
        "decreasing speed",
    }:
        return "decrease"
    return ""


def _physical_speed_geometry(scenario: dict[str, Any]) -> dict[str, str] | None:
    """Build the reusable distance-over-time evidence block for speed change."""

    configured = scenario.get("speed_geometry")
    if isinstance(configured, dict) and configured:
        return {str(key): str(value) for key, value in configured.items() if str(value).strip()}

    direction = _physical_speed_direction(scenario)
    if not direction:
        return None

    object_name = str(scenario.get("object") or scenario.get("object_type") or "object").strip()
    measurement_frame = str(
        scenario.get("measurement_frame") or "distance covered during equal time intervals"
    )
    if direction == "increase":
        initial_rate = str(
            scenario.get("initial_speed_evidence")
            or f"at first, the {object_name} covered less distance in each equal time interval"
        )
        later_rate = str(
            scenario.get("later_speed_evidence")
            or f"later, the {object_name} covered more distance in each equal time interval"
        )
        same_time_comparison = str(
            scenario.get("quantity_change") or "more distance in the same amount of time"
        )
        rate_evidence = str(
            scenario.get("rate_evidence") or "distance per equal time interval increased"
        )
        definition_bridge = str(
            scenario.get("speed_definition_bridge")
            or "covering more distance in the same amount of time means speed increased"
        )
        process_bridge = str(
            scenario.get("speed_process_bridge")
            or f"increasing speed means the {object_name} was accelerating"
        )
    else:
        initial_rate = str(
            scenario.get("initial_speed_evidence")
            or f"at first, the {object_name} covered more distance in each equal time interval"
        )
        later_rate = str(
            scenario.get("later_speed_evidence")
            or f"later, the {object_name} covered less distance in each equal time interval"
        )
        same_time_comparison = str(
            scenario.get("quantity_change") or "less distance in the same amount of time"
        )
        rate_evidence = str(
            scenario.get("rate_evidence") or "distance per equal time interval decreased"
        )
        definition_bridge = str(
            scenario.get("speed_definition_bridge")
            or "covering less distance in the same amount of time means speed decreased"
        )
        process_bridge = str(
            scenario.get("speed_process_bridge")
            or f"decreasing speed means the {object_name} was slowing down"
        )

    speed_geometry = {
        "measurement_frame": measurement_frame,
        "initial_rate": initial_rate,
        "later_rate": later_rate,
        "same_time_comparison": same_time_comparison,
        "rate_evidence": rate_evidence,
        "definition_bridge": definition_bridge,
        "process_bridge": process_bridge,
    }
    optional_consequence = str(
        scenario.get("optional_consequence")
        or scenario.get("endpoint_tendency")
        or scenario.get("endpoint_risk")
        or ""
    ).strip()
    if optional_consequence:
        speed_geometry["optional_consequence"] = optional_consequence
    return speed_geometry


def _physical_render_family(spec: dict[str, Any]) -> str:
    """Return a positive-only family label that avoids naming contrast pairs."""

    family = str(spec.get("family") or spec.get("physical_family") or "").strip()
    if family in {"roll_slide", "roll_slide_surface_rotation_cues"}:
        return "surface_motion"
    if family in {
        "sink_float",
        "directional_motion_sink_expansion",
        "sink_float_surface_cues",
    }:
        return "fluid_motion"
    if family in {"landmark_direction"}:
        return "landmark_motion"
    if family in {"directional_motion", "directional_motion_lift_expansion"}:
        return "directional_motion"
    if family in {"support_stability"}:
        return "support_stability"
    if family in PHYSICAL_SPEED_CHANGE_FAMILIES:
        return "speed_change"
    if family in {"force_deformation_expansion"}:
        return "shape_change"
    if family in {"contact_collision", "contact_collision_expansion"}:
        return "contact_response"
    if family in {"containment_flow", "containment_flow_expansion"}:
        return "containment_flow"
    if family in {"containment_absorb_expansion"}:
        return "absorption"
    return "physical_event"


def _metadata_only_style_fields(spec: dict[str, Any]) -> dict[str, Any]:
    """Return renderer-facing style fields without contrast-oriented labels."""

    style_id = str(spec.get("style_id") or "")
    if style_id == "comparison_contrast":
        return {
            "style_id": "practical_handling_note",
            "style_name": "Practical handling note",
            "style_description": "A practical note about arranging or checking one observed physical event.",
            "style_guidance": [
                "Write from the perspective of someone handling the object, arranging the setup, or checking the result.",
                "Focus on the condition that produced the observed process and the visible trace that followed.",
                "Describe what the object or person actively did.",
                "Do not discuss alternate outcomes, competing processes, or what did not happen.",
            ],
        }

    row: dict[str, Any] = {}
    for key in ("style_id", "style_name", "style_description"):
        if spec.get(key):
            row[key] = spec.get(key)

    guidance = [
        str(item)
        for item in spec.get("style_guidance", [])
        if str(item).strip()
        and "contrast" not in str(item).lower()
        and "alternate" not in str(item).lower()
        and "other outcome" not in str(item).lower()
    ]
    if guidance:
        row["style_guidance"] = guidance
    return row


def _prompt_value_is_empty(value: Any) -> bool:
    if value is None:
        return True
    if value == "":
        return True
    if isinstance(value, (list, dict)) and not value:
        return True
    return False


def _compact_prompt_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not _prompt_value_is_empty(value)
    }


def _physical_prompt_spec(
    spec: dict[str, Any],
    *,
    contrast_verbalization: str,
) -> dict[str, Any]:
    row = {key: spec.get(key) for key in PHYSICAL_RENDER_PROMPT_KEYS if key in spec}
    if contrast_verbalization == "metadata_only":
        row["render_family"] = _physical_render_family(spec)
        if row.get("document_type") == "contrast_pair":
            row["document_type"] = "positive_event"
        for key in ("style_id", "style_name", "style_description", "style_guidance"):
            row.pop(key, None)
        row.update(_metadata_only_style_fields(spec))
    else:
        row.update(
            {
                key: spec.get(key)
                for key in PHYSICAL_EXPLICIT_CONTRAST_RENDER_PROMPT_KEYS
                if key in spec
            }
        )
    return _compact_prompt_row(row)


def _physical_lexical_targets(
    *,
    family: dict[str, Any],
    scenario: dict[str, Any],
) -> list[str]:
    """Return opt-in lexical targets for one physical-dynamics scenario."""

    targets = _physical_terms(scenario, "lexical_targets")
    if targets:
        return targets

    target_sets = family.get("lexical_target_sets", {})
    if not isinstance(target_sets, dict):
        return []
    expected_process = str(scenario.get("expected_process") or "").strip()
    configured = target_sets.get(expected_process, [])
    if isinstance(configured, str):
        configured = [configured]
    return [str(target) for target in configured if str(target).strip()]


def _physical_canonical_counts(pack: dict[str, Any]) -> dict[str, int]:
    raw_counts = pack.get("canonical_concept_counts", pack.get("canonical_document_counts", {}))
    if not raw_counts:
        return {}
    if not isinstance(raw_counts, dict):
        raise TypeError("canonical_concept_counts must be an object mapping concept ids to counts.")
    return {
        str(concept_id): int(count)
        for concept_id, count in raw_counts.items()
        if str(concept_id).strip() and int(count) > 0
    }


def _physical_canonical_inventory(
    config: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    inventory = profile.get("canonical_inventory") or config.get("canonical_inventory")
    inventory_path = profile.get("canonical_inventory_path") or config.get("canonical_inventory_path")
    if inventory and inventory_path:
        raise ValueError("Use either canonical_inventory or canonical_inventory_path, not both.")
    if inventory_path:
        resolved = _resolve_physical_inventory_source(Path(str(inventory_path))).resolve()
        inventory = load_pipeline_config(resolved)
    if not isinstance(inventory, dict):
        raise ValueError("canonical_concept_counts requires canonical_inventory_path or canonical_inventory.")
    if not isinstance(inventory.get("surface_to_concept"), dict):
        raise ValueError("canonical inventory must include a surface_to_concept object.")
    return inventory


def _physical_canonical_concept(
    process: Any,
    surface_to_concept: dict[str, Any],
) -> str | None:
    label = str(process or "").strip()
    if not label:
        return None
    concept_id = surface_to_concept.get(label)
    if concept_id is None:
        concept_id = surface_to_concept.get(label.lower())
    return str(concept_id) if concept_id is not None else None


def _physical_canonical_candidates(
    families_by_id: dict[str, dict[str, Any]],
    surface_to_concept: dict[str, Any],
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any], str | None]]]:
    candidates: dict[str, list[tuple[dict[str, Any], dict[str, Any], str | None]]] = {}
    for family in families_by_id.values():
        for scenario in family.get("scenarios", []):
            scenario_row = dict(scenario)
            concept_id = _physical_canonical_concept(
                scenario_row.get("expected_process"),
                surface_to_concept,
            )
            if concept_id is None:
                continue
            contrast_concept_id = _physical_canonical_concept(
                scenario_row.get("contrast_process"),
                surface_to_concept,
            )
            candidates.setdefault(concept_id, []).append((family, scenario_row, contrast_concept_id))
    return candidates


def _select_physical_canonical_candidate(
    candidates: list[tuple[dict[str, Any], dict[str, Any], str | None]],
    *,
    offset: int,
    total_count: int,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    if not candidates:
        raise ValueError("Cannot select from an empty canonical candidate list.")
    if total_count <= len(candidates):
        return candidates[(offset * len(candidates)) // total_count]
    return candidates[offset % len(candidates)]


def _select_physical_canonical_sequence(
    candidates: list[tuple[dict[str, Any], dict[str, Any], str | None]],
    *,
    total_count: int,
    candidate_balance_axes: dict[str, tuple[str, ...]] | None = None,
    candidate_balance_counts: dict[tuple[str, str, str], int] | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any], str | None]]:
    balance_axes = candidate_balance_axes or {}
    balance_keys_by_index = [
        _physical_candidate_balance_keys(candidate, balance_axes)
        for candidate in candidates
    ]
    if balance_axes and candidate_balance_counts is not None and any(balance_keys_by_index):
        selected: list[tuple[dict[str, Any], dict[str, Any], str | None]] = []
        candidate_counts: dict[int, int] = {}
        for _ in range(total_count):
            selected_index = min(
                range(len(candidates)),
                key=lambda index: (
                    0 if balance_keys_by_index[index] else 1,
                    sum(
                        candidate_balance_counts.get(key, 0)
                        for key in balance_keys_by_index[index]
                    ),
                    tuple(
                        candidate_balance_counts.get(key, 0)
                        for key in balance_keys_by_index[index]
                    ),
                    candidate_counts.get(index, 0),
                    index,
                ),
            )
            selected.append(candidates[selected_index])
            candidate_counts[selected_index] = (
                candidate_counts.get(selected_index, 0) + 1
            )
            for key in balance_keys_by_index[selected_index]:
                candidate_balance_counts[key] = candidate_balance_counts.get(key, 0) + 1
        return selected

    return [
        _select_physical_canonical_candidate(
            candidates,
            offset=offset,
            total_count=total_count,
        )
        for offset in range(total_count)
    ]


def _physical_candidate_family_share_rules(
    profile: dict[str, Any],
    pack: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_rules = (
        pack.get("canonical_candidate_family_shares")
        or profile.get("canonical_candidate_family_shares")
        or {}
    )
    if not raw_rules:
        return []
    if not isinstance(raw_rules, dict):
        raise TypeError("canonical_candidate_family_shares must be an object.")

    rules = []
    for family_id, raw_rule in raw_rules.items():
        if isinstance(raw_rule, (int, float)):
            share = float(raw_rule)
            concept_ids: set[str] = set()
        elif isinstance(raw_rule, dict):
            share = float(raw_rule.get("share", 0))
            concept_ids = {
                str(concept_id)
                for concept_id in raw_rule.get("concept_ids", [])
                if str(concept_id).strip()
            }
        else:
            raise TypeError(
                "canonical_candidate_family_shares entries must be numbers or objects."
            )
        if share <= 0:
            continue
        rules.append(
            {
                "family_id": str(family_id),
                "share": min(share, 1.0),
                "concept_ids": concept_ids,
            }
        )
    return rules


def _physical_candidate_balance_axes(
    profile: dict[str, Any],
    pack: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    raw_axes = (
        pack.get("canonical_candidate_balance_axes")
        or profile.get("canonical_candidate_balance_axes")
        or {}
    )
    if not raw_axes:
        return {}
    if not isinstance(raw_axes, dict):
        raise TypeError("canonical_candidate_balance_axes must be an object.")

    axes_by_family: dict[str, tuple[str, ...]] = {}
    for family_id, raw_family_axes in raw_axes.items():
        if isinstance(raw_family_axes, str):
            family_axes = [raw_family_axes]
        elif isinstance(raw_family_axes, list):
            family_axes = raw_family_axes
        else:
            raise TypeError(
                "canonical_candidate_balance_axes entries must be strings or lists."
            )
        axes = tuple(
            str(axis).strip()
            for axis in family_axes
            if str(axis).strip()
        )
        if axes:
            axes_by_family[str(family_id)] = axes
    return axes_by_family


def _physical_candidate_balance_keys(
    candidate: tuple[dict[str, Any], dict[str, Any], str | None],
    balance_axes: dict[str, tuple[str, ...]],
) -> tuple[tuple[str, str, str], ...]:
    family, scenario, _ = candidate
    family_id = _physical_family_id(family)
    keys = []
    for axis in balance_axes.get(family_id, ()):
        value = str(scenario.get(axis) or "").strip()
        if value:
            keys.append((family_id, axis, value))
    return tuple(keys)


def _select_physical_canonical_candidates_for_concept(
    candidates: list[tuple[dict[str, Any], dict[str, Any], str | None]],
    *,
    concept_id: str,
    total_count: int,
    family_share_rules: list[dict[str, Any]],
    candidate_balance_axes: dict[str, tuple[str, ...]] | None = None,
    candidate_balance_counts: dict[tuple[str, str, str], int] | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any], str | None]]:
    if not family_share_rules:
        return _select_physical_canonical_sequence(
            candidates,
            total_count=total_count,
            candidate_balance_axes=candidate_balance_axes,
            candidate_balance_counts=candidate_balance_counts,
        )

    selected: list[tuple[dict[str, Any], dict[str, Any], str | None]] = []
    reserved_family_ids: set[str] = set()
    remaining_count = total_count
    for rule in family_share_rules:
        rule_concepts = rule["concept_ids"]
        if rule_concepts and concept_id not in rule_concepts:
            continue
        family_id = str(rule["family_id"])
        family_candidates = [
            candidate
            for candidate in candidates
            if _physical_family_id(candidate[0]) == family_id
        ]
        if not family_candidates:
            continue
        quota = min(remaining_count, int(round(total_count * float(rule["share"]))))
        if quota <= 0:
            continue
        selected.extend(
            _select_physical_canonical_sequence(
                family_candidates,
                total_count=quota,
                candidate_balance_axes=candidate_balance_axes,
                candidate_balance_counts=candidate_balance_counts,
            )
        )
        reserved_family_ids.add(family_id)
        remaining_count -= quota
        if remaining_count <= 0:
            break

    if remaining_count <= 0:
        return selected[:total_count]

    remainder_candidates = [
        candidate
        for candidate in candidates
        if _physical_family_id(candidate[0]) not in reserved_family_ids
    ]
    if not remainder_candidates:
        remainder_candidates = candidates
    selected.extend(
        _select_physical_canonical_sequence(
            remainder_candidates,
            total_count=remaining_count,
            candidate_balance_axes=candidate_balance_axes,
            candidate_balance_counts=candidate_balance_counts,
        )
    )
    return selected


def _make_physical_dynamics_spec(
    *,
    index: int,
    document_type: str,
    coverage_pack_id: str,
    profile: dict[str, Any],
    family: dict[str, Any],
    scenario: dict[str, Any],
    scenario_repeat_index: int = 0,
    canonical_concept_id: str | None = None,
    canonical_contrast_concept_id: str | None = None,
    canonical_inventory_version: str | None = None,
) -> dict[str, Any]:
    profile_id = _profile_id(profile)
    family_id = _physical_family_id(family)
    scenario_id = str(scenario.get("scenario_id") or scenario.get("id") or f"{family_id}_{index}")
    concept_pair = scenario.get("concept_pair", [])
    if isinstance(concept_pair, str):
        concept_pair = [concept_pair]
    lexical_targets = _physical_lexical_targets(family=family, scenario=scenario)
    lexical_target = _cycle_string(lexical_targets, scenario_repeat_index) if lexical_targets else None
    render_template = str(scenario.get("render_template") or "")
    speed_geometry = (
        None
        if render_template.startswith(("lexical_shortcut_", "lexical_bridge_"))
        else (
            _physical_speed_geometry(scenario)
            if family_id in PHYSICAL_SPEED_CHANGE_FAMILIES or _physical_speed_direction(scenario)
            else None
        )
    )

    spec = {
        "id": f"cc_{_safe_id(profile_id)}_{index:04d}",
        "concept_id": profile_id,
        "profile_id": profile_id,
        "domain": str(profile.get("domain", "")),
        "coverage_pack_id": coverage_pack_id,
        "document_type": document_type,
        "definition": str(profile.get("definition", "")),
        "centrality_rule": str(profile.get("centrality_rule", "")),
        "lexical_forms": [],
        "profile_type": "physical_dynamics",
        "scenario_id": scenario_id,
        "family": family_id,
        "physical_family": family_id,
        "render_template": scenario.get("render_template"),
        "template_sequence": scenario.get("template_sequence"),
        "family_label": family.get("label") or family_id,
        "family_schema": family.get("schema"),
        "concept_pair": [str(item) for item in concept_pair if str(item).strip()],
        "cause_type": scenario.get("cause_type"),
        "object_set_id": scenario.get("object_set_id"),
        "object_type": scenario.get("object_type"),
        "object": scenario.get("object"),
        "object_shape": scenario.get("object_shape"),
        "body_pose": scenario.get("body_pose"),
        "material_or_surface": scenario.get("material_or_surface"),
        "contact_surface": scenario.get("contact_surface"),
        "surface_friction": scenario.get("surface_friction"),
        "rotation_allowed": scenario.get("rotation_allowed"),
        "rotation_evidence": scenario.get("rotation_evidence"),
        "motion_pattern": scenario.get("motion_pattern"),
        "geometry_or_medium": scenario.get("geometry_or_medium"),
        "timescale": scenario.get("timescale"),
        "path_shape": str(scenario.get("path_shape") or "none"),
        "start_position": scenario.get("start_position"),
        "end_position": scenario.get("end_position"),
        "direction_evidence": scenario.get("direction_evidence"),
        "vertical_relation_language": scenario.get("vertical_relation_language"),
        "directional_polarity": scenario.get("directional_polarity"),
        "measurement_frame": scenario.get("measurement_frame"),
        "initial_speed_evidence": scenario.get("initial_speed_evidence"),
        "later_speed_evidence": scenario.get("later_speed_evidence"),
        "initial_size_evidence": scenario.get("initial_size_evidence"),
        "later_size_evidence": scenario.get("later_size_evidence"),
        "size_change_evidence": scenario.get("size_change_evidence"),
        "quantity_change": scenario.get("quantity_change"),
        "rate_evidence": scenario.get("rate_evidence"),
        "endpoint": scenario.get("endpoint"),
        "endpoint_tendency": scenario.get("endpoint_tendency"),
        "endpoint_risk": scenario.get("endpoint_risk"),
        "stopping_evidence": scenario.get("stopping_evidence"),
        "speed_geometry": speed_geometry,
        "bridge_type": scenario.get("bridge_type"),
        "bridge_clause": scenario.get("bridge_clause"),
        "evidence_clause": scenario.get("evidence_clause"),
        "same_sentence_evidence_terms": _physical_terms(
            scenario,
            "same_sentence_evidence_terms",
        ),
        "same_sentence_target_terms": _physical_terms(
            scenario,
            "same_sentence_target_terms",
        ),
        "contrast_evidence_bans": _physical_terms(
            scenario,
            "contrast_evidence_bans",
        ),
        "reference_frame": scenario.get("reference_frame"),
        "landmark": scenario.get("landmark"),
        "initial_relation_to_landmark": scenario.get("initial_relation_to_landmark"),
        "later_relation_to_landmark": scenario.get("later_relation_to_landmark"),
        "crossing_direction": scenario.get("crossing_direction"),
        "verb_type": scenario.get("verb_type"),
        "event_verb": scenario.get("event_verb"),
        "mechanism": scenario.get("mechanism"),
        "expected_process": scenario.get("expected_process"),
        "contrast_process": scenario.get("contrast_process"),
        "cause_evidence": scenario.get("cause_evidence"),
        "outcome_evidence": scenario.get("outcome_evidence"),
        "required_cause_terms": _physical_terms(
            scenario,
            "required_cause_terms",
            fallback="cause_evidence",
        ),
        "required_process_terms": _physical_terms(
            scenario,
            "required_process_terms",
            fallback="expected_process",
        ),
        "required_outcome_terms": _physical_terms(
            scenario,
            "required_outcome_terms",
            fallback="outcome_evidence",
        ),
        "lexical_targets": lexical_targets,
        "lexical_target": lexical_target,
        "preferred_process_terms": _physical_terms(scenario, "preferred_process_terms"),
        "preferred_contrast_terms": _physical_terms(scenario, "preferred_contrast_terms"),
        "shortcut_terms": _physical_terms(scenario, "shortcut_terms"),
    }
    if canonical_concept_id:
        spec["canonical_concept_id"] = canonical_concept_id
        spec["canonical_balance_role"] = "expected_process"
    if canonical_contrast_concept_id:
        spec["canonical_contrast_concept_id"] = canonical_contrast_concept_id
    if canonical_inventory_version:
        spec["canonical_inventory_version"] = canonical_inventory_version
    return spec


def build_physical_dynamics_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build physical-dynamics specs from family quotas and scenario templates."""

    profile = dict(config["physical_dynamics_profile"])
    families = _physical_profile_families(profile)
    if not families:
        raise ValueError("physical_dynamics_profile must include at least one family.")

    families_by_id: dict[str, dict[str, Any]] = {}
    for family in families:
        family_id = _physical_family_id(family)
        if family_id in families_by_id:
            raise ValueError(f"Duplicate physical-dynamics family id: {family_id!r}.")
        families_by_id[family_id] = family
    document_types = _physical_document_types(profile)
    scenario_cursors = {family_id: 0 for family_id in families_by_id}
    scenario_repeat_counts: dict[tuple[str, str], int] = {}
    specs: list[dict[str, Any]] = []

    for pack in config.get("coverage_packs", []):
        coverage_pack_id = str(pack["id"])
        counts = dict(pack.get("document_counts", {}))
        canonical_counts = _physical_canonical_counts(pack)
        if counts and canonical_counts:
            raise ValueError("Use document_counts or canonical_concept_counts, not both.")

        if canonical_counts:
            inventory = _physical_canonical_inventory(config, profile)
            surface_to_concept = dict(inventory["surface_to_concept"])
            known_concepts = {
                str(row["concept_id"])
                for row in inventory.get("canonical_concepts", [])
                if row.get("concept_id")
            }
            unknown_concepts = set(canonical_counts) - known_concepts if known_concepts else set()
            if unknown_concepts:
                unknown = ", ".join(sorted(unknown_concepts))
                raise ValueError(f"Unknown canonical concept id(s): {unknown}.")

            candidates_by_concept = _physical_canonical_candidates(
                families_by_id,
                surface_to_concept,
            )
            family_share_rules = _physical_candidate_family_share_rules(profile, pack)
            candidate_balance_axes = _physical_candidate_balance_axes(profile, pack)
            candidate_balance_counts: dict[tuple[str, str, str], int] = {}
            for concept_id, count in canonical_counts.items():
                candidates = candidates_by_concept.get(concept_id, [])
                if not candidates:
                    raise ValueError(
                        f"No physical-dynamics scenarios map to canonical concept {concept_id!r}."
                    )
                selected_candidates = _select_physical_canonical_candidates_for_concept(
                    candidates,
                    concept_id=concept_id,
                    total_count=count,
                    family_share_rules=family_share_rules,
                    candidate_balance_axes=candidate_balance_axes,
                    candidate_balance_counts=candidate_balance_counts,
                )
                for offset, (family, scenario, contrast_concept_id) in enumerate(selected_candidates):
                    family_id = _physical_family_id(family)
                    scenario_id = str(
                        scenario.get("scenario_id")
                        or scenario.get("id")
                        or f"{family_id}_{offset}"
                    )
                    repeat_key = (family_id, scenario_id)
                    repeat_index = scenario_repeat_counts.get(repeat_key, 0)
                    specs.append(
                        _make_physical_dynamics_spec(
                            index=len(specs),
                            document_type=_cycle_string(document_types, len(specs)),
                            coverage_pack_id=coverage_pack_id,
                            profile=profile,
                            family=family,
                            scenario=scenario,
                            scenario_repeat_index=offset,
                            canonical_concept_id=concept_id,
                            canonical_contrast_concept_id=contrast_concept_id,
                            canonical_inventory_version=inventory.get("version"),
                        )
                    )
                    scenario_repeat_counts[repeat_key] = repeat_index + 1
            continue

        for family_id, family in families_by_id.items():
            count = int(counts.get(family_id, 0))
            scenarios = [dict(row) for row in family.get("scenarios", [])]
            if count and not scenarios:
                raise ValueError(f"Physical-dynamics family {family_id!r} has no scenarios.")
            for _ in range(count):
                scenario_offset = scenario_cursors[family_id]
                scenario_repeat_index = scenario_offset // len(scenarios)
                specs.append(
                    _make_physical_dynamics_spec(
                        index=len(specs),
                        document_type=_cycle_string(document_types, len(specs)),
                        coverage_pack_id=coverage_pack_id,
                        profile=profile,
                        family=family,
                        scenario=_cycle(scenarios, scenario_offset),
                        scenario_repeat_index=scenario_repeat_index,
                    )
                )
                scenario_cursors[family_id] = scenario_offset + 1

    specs = _assign_discourse_styles(specs, list(config.get("discourse_styles", [])))
    return _assign_signal_orders(specs, list(config.get("signal_orders", [])))


def _assign_discourse_styles(specs: list[dict[str, Any]], styles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach balanced discourse-style targets to specs."""

    if not styles:
        return specs

    styled_specs = []
    for index, spec in enumerate(specs):
        style = styles[index % len(styles)]
        row = dict(spec)
        row.update(
            {
                "style_id": style.get("style_id"),
                "style_name": style.get("style_name"),
                "style_description": style.get("description"),
                "style_guidance": list(style.get("guidance", [])),
                "style_required_terms": list(style.get("required_terms", [])),
                "style_min_required_terms": style.get("min_required_terms"),
            }
        )
        styled_specs.append(row)
    return styled_specs


def _assign_signal_orders(specs: list[dict[str, Any]], signal_orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach balanced positive/boundary ordering targets to specs."""

    if not signal_orders:
        return specs

    ordered_specs = []
    for index, spec in enumerate(specs):
        order = signal_orders[index % len(signal_orders)]
        row = dict(spec)
        row.update(
            {
                "signal_order_id": order.get("order_id") or order.get("id"),
                "signal_order_name": order.get("order_name") or order.get("name"),
                "signal_order_description": order.get("description"),
                "signal_order_guidance": list(order.get("guidance", [])),
            }
        )
        ordered_specs.append(row)
    return ordered_specs


def build_scenario_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build balanced scenario specs from concept profile coverage packs."""

    if "material_class_profile" in config:
        return build_material_class_specs(config)
    if "physical_dynamics_profile" in config:
        return build_physical_dynamics_specs(config)

    concept_profile = dict(config["concept_profile"])
    specs: list[ScenarioSpec] = []
    subtype_cursor = 0
    state_cursor = 0
    neighbor_cursor = 0

    for pack in config.get("coverage_packs", []):
        coverage_pack_id = str(pack["id"])
        counts = dict(pack.get("document_counts", {}))
        for document_type in CONCEPT_DOCUMENT_TYPES:
            count = int(counts.get(document_type, 0))
            for _ in range(count):
                spec = _make_spec(
                    index=len(specs),
                    document_type=document_type,
                    coverage_pack_id=coverage_pack_id,
                    concept_profile=concept_profile,
                    subtype_offset=subtype_cursor,
                    state_offset=state_cursor,
                    neighbor_offset=neighbor_cursor,
                )
                specs.append(spec)
                if document_type in {"direct_positive", "subtype_explanation", "neighbor_contrast"}:
                    subtype_cursor += 1
                if document_type == "state_contrast":
                    state_cursor += 1
                if document_type == "neighbor_contrast":
                    neighbor_cursor += 1

    spec_rows = [asdict(spec) for spec in specs]
    spec_rows = _assign_discourse_styles(spec_rows, list(config.get("discourse_styles", [])))
    return _assign_signal_orders(spec_rows, list(config.get("signal_orders", [])))


def extract_json(text: str) -> Any:
    """Extract JSON from raw model output or a fenced JSON block."""

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
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
    """Normalize a JSON object/list into a list of document rows."""

    if isinstance(value, list):
        items = value
    elif isinstance(value, dict):
        for key in ("items", "documents", "rows", "data"):
            if isinstance(value.get(key), list):
                items = value[key]
                break
        else:
            if isinstance(value.get("text"), str):
                items = [value]
            elif all(isinstance(item, (dict, str)) for item in value.values()):
                items = []
                for key, item in value.items():
                    if isinstance(item, dict):
                        row = dict(item)
                        row.setdefault("id", str(key))
                    else:
                        row = {"id": str(key), "text": item}
                    items.append(row)
            else:
                raise ValueError(
                    "JSON object must include an items/documents/rows/data list "
                    "or be an id-to-document mapping."
                )
    else:
        raise ValueError(f"Expected JSON list or object, got {type(value).__name__}.")

    normalized: list[dict[str, Any]] = []
    bad_types: list[str] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append({"text": item})
        else:
            bad_types.append(type(item).__name__)
    if bad_types:
        sample_types = ", ".join(sorted(set(bad_types)))
        raise ValueError(f"JSON items must be objects or strings, got {sample_types}.")
    return normalized


def _system_prompt() -> str:
    return (
        "You write short synthetic training documents for a small language model. "
        "Return only valid JSON. Do not mention benchmark names, tests, scoring, or hidden metadata."
    )


def _human_list(values: list[Any], *, limit: int | None = None) -> str:
    items = [str(value).strip() for value in values if str(value).strip()]
    if limit is not None:
        items = items[:limit]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", or " + items[-1]


def build_render_prompt(
    *,
    specs: list[dict[str, Any]],
    concept_profile: dict[str, Any],
    constraints: dict[str, Any],
) -> str:
    """Build the prompt for rendering a batch of concept-curriculum documents."""

    prompt_banned_terms = constraints.get("prompt_banned_terms", constraints.get("banned_terms", []))
    neighbor_reference = [
        {
            "label": item.get("label") or item.get("id"),
            "relation": item.get("relation"),
        }
        for item in concept_profile.get("neighbors", [])
    ]
    prompt_payload = {
        "concept": {
            "concept_id": concept_profile.get("concept_id"),
            "definition": concept_profile.get("definition"),
            "centrality_rule": concept_profile.get("centrality_rule"),
            "lexical_forms": concept_profile.get("lexical_forms", []),
            "related_terms": concept_profile.get("related_terms", []),
            "source_frame_terms": concept_profile.get("source_frame_terms", []),
            "generation_guidance": concept_profile.get("generation_guidance", []),
            "positive_subtype_labels": [
                item.get("label") for item in concept_profile.get("positive_subtypes", [])
            ],
            "negative_subtype_labels": [
                item.get("label") for item in concept_profile.get("negative_subtypes", [])
            ],
            "neighbor_actions": neighbor_reference,
        },
        "constraints": {
            "word_count_range": [
                int(constraints.get("min_words", 70)),
                int(constraints.get("max_words", 140)),
            ],
            "must_include_one_lexical_form": concept_profile.get("lexical_forms", []),
            "must_make_target_participant_central": True,
            "avoid_tool_centered_examples": True,
            "banned_terms": prompt_banned_terms,
        },
        "scenario_specs": specs,
    }
    concept_id = str(concept_profile.get("concept_id") or "target concept")
    lexical_examples = _human_list(list(concept_profile.get("lexical_forms", [])))
    related_examples = _human_list(list(concept_profile.get("related_terms", [])), limit=10)
    neighbor_examples = _human_list([item["label"] for item in neighbor_reference if item.get("label")], limit=8)

    exact_form_rule = (
        "Hard rule: every document must include at least one exact target lexical form from "
        "constraints.must_include_one_lexical_form."
    )
    if lexical_examples:
        exact_form_rule += f" For this concept, acceptable exact forms include {lexical_examples}."

    related_rule = "Use related terms naturally when helpful."
    if related_examples:
        related_rule += f" For this concept, related terms include {related_examples}."

    neighbor_rule = (
        "Hard rule: for any spec with document_type=\"neighbor_contrast\", the text must include "
        "the exact neighbor string from the spec."
    )
    if neighbor_examples:
        neighbor_rule += f" Possible neighbor strings include {neighbor_examples}."

    return (
        "Write one compact document for each scenario spec.\n"
        "Each document should be natural prose for pretraining, not a quiz or an instruction to the reader.\n"
        f"{exact_form_rule} Do not rely only on synonyms or related actions.\n"
        f"{related_rule}\n"
        "Follow concept.generation_guidance, subtype mechanisms, negative reasons, and state lessons when present.\n"
        "Use the specified material/substance names. Keep the target material central.\n"
        "Follow each spec's discourse style. Use style_guidance as instructions, not text to copy.\n"
        "Follow each spec's signal_order_guidance to decide whether the positive affordance, boundary case, neighbor action, or parallel material choice appears first. "
        "Vary the opening emphasis; do not always lead with the target concept or target lexical form.\n"
        "For concept_map_paragraph style, connect the target concept to nearby material behaviors and name 2-3 nearby actions when natural.\n"
        f"For state contrasts, clearly mark why one state fits {concept_id} and the other does not.\n"
        "For neighbor contrasts, explicitly name the neighbor concept word and explain how it differs from the target concept.\n"
        f"{neighbor_rule}\n"
        "Return JSON with this shape: {\"items\": [{\"id\": string, \"text\": string}]}.\n\n"
        "SCENARIO_SPECS_JSON:\n"
        f"{json.dumps(prompt_payload, indent=2)}"
    )


def _sanitize_correlation_specs_for_prompt(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove benchmark-facing metadata before sending specs to the generator."""

    hidden_keys = {"coverage_pack_id", "tags", "ewok_anchors"}
    return [{key: value for key, value in spec.items() if key not in hidden_keys} for spec in specs]


def build_material_class_render_prompt(
    *,
    specs: list[dict[str, Any]],
    material_class_profile: dict[str, Any],
    constraints: dict[str, Any],
) -> str:
    """Build a correlation-first prompt for material/action affordance classes."""

    prompt_banned_terms = constraints.get("prompt_banned_terms", constraints.get("banned_terms", []))
    negative_markers = constraints.get(
        "negative_markers",
        ["cannot", "can't", "does not", "do not", "not", "unlike", "instead"],
    )
    prompt_payload = {
        "profile": {
            "profile_id": material_class_profile.get("profile_id"),
            "domain": material_class_profile.get("domain"),
            "definition": material_class_profile.get("definition"),
            "centrality_rule": material_class_profile.get("centrality_rule"),
            "generation_guidance": material_class_profile.get("generation_guidance", []),
        },
        "constraints": {
            "word_count_range": [
                int(constraints.get("min_words", 35)),
                int(constraints.get("max_words", 100)),
            ],
            "required_affordances_are_positive": True,
            "keep_material_and_action_close": True,
            "avoid_negative_contrast_markers": negative_markers,
            "banned_terms": prompt_banned_terms,
        },
        "scenario_specs": _sanitize_correlation_specs_for_prompt(specs),
    }

    return (
        "Write one compact correlation-first training document for each scenario spec.\n"
        "Each document should teach positive material-action associations, not causal contrast.\n"
        "Hard rule: each document must focus on exactly one central material from required_materials.\n"
        "Hard rule: include every string in required_materials.\n"
        "Hard rule: include at least two actions from required_affordances, using the exact base word or a natural inflection.\n"
        "When target_affordance is present, make that action one of the visible material-action pairings.\n"
        "Do not introduce a second example material unless it is part of the required material phrase itself.\n"
        "Keep the central material near its plausible actions, preferably in the same sentence.\n"
        "Do not write negative contrasts. Avoid wording like cannot, can't, does not, not, unlike, or instead.\n"
        "Do not mention benchmark names, coverage labels, hidden tags, tests, or scoring.\n"
        "Return JSON with this shape: {\"items\": [{\"id\": string, \"text\": string}]}.\n\n"
        "SCENARIO_SPECS_JSON:\n"
        f"{json.dumps(prompt_payload, indent=2)}"
    )


def build_physical_dynamics_render_prompt(
    *,
    specs: list[dict[str, Any]],
    physical_dynamics_profile: dict[str, Any],
    constraints: dict[str, Any],
) -> str:
    """Build a prompt for physical-dynamics cause/process/outcome documents."""

    prompt_banned_terms = constraints.get("prompt_banned_terms", constraints.get("banned_terms", []))
    contrast_verbalization = _physical_contrast_verbalization(
        physical_dynamics_profile=physical_dynamics_profile,
        constraints=constraints,
    )
    if contrast_verbalization == "metadata_only":
        prompt_banned_terms = [
            term
            for term in prompt_banned_terms
            if str(term).strip().lower() != "contrast process"
        ]
    definition = str(
        physical_dynamics_profile.get(
            "definition",
            (
                "Can a model connect evidence about directional physical change, "
                "motion, and state/process change over time to the right "
                "event/process concept?"
            ),
        )
    )
    prompt_constraints = {
        "word_count_range": [
            int(constraints.get("min_words", 45)),
            int(constraints.get("max_words", 130)),
        ],
        "preserve_cause_process_outcome": True,
        "include_required_cause_evidence": True,
        "include_expected_process_evidence": True,
        "avoid_shortcut_only_object_outcome_correlation": True,
        "banned_terms": prompt_banned_terms,
    }
    if contrast_verbalization == "metadata_only":
        prompt_constraints["positive_only"] = True
        prompt_constraints["avoid_comparison_frames"] = list(PHYSICAL_COMPARISON_FRAME_BANS)
    else:
        prompt_constraints["contrast_verbalization"] = contrast_verbalization

    prompt_payload = {
        "profile": {
            "profile_id": physical_dynamics_profile.get("profile_id"),
            "domain": physical_dynamics_profile.get("domain"),
            "definition": definition,
            "centrality_rule": physical_dynamics_profile.get("centrality_rule"),
            "generation_guidance": physical_dynamics_profile.get("generation_guidance", []),
            "path_shape_policy": physical_dynamics_profile.get("path_shape_policy", {}),
        },
        "constraints": prompt_constraints,
        "scenario_specs": [
            _physical_prompt_spec(
                spec,
                contrast_verbalization=contrast_verbalization,
            )
            for spec in specs
        ],
    }
    if contrast_verbalization == "metadata_only":
        contrast_guidance = (
            "Positive-only render policy: describe only the observed event. Do not discuss alternate outcomes, competing processes, or what did not happen. Avoid comparison frames such as \"instead of\", \"rather than\", \"not X but Y\", \"unlike\", or \"as opposed to\". Write positive evidence for the cause, expected process, and outcome only.\n"
            f"{PHYSICAL_POSITIVE_ONLY_LEXICAL_HYGIENE}"
        )
    elif contrast_verbalization == "implicit":
        contrast_guidance = (
            "Contrast-verbalization policy: implicit. You may allude briefly to a different setup or different result, but do not name the contrast_process or use its lexical forms. Keep the expected process local context clean and central.\n"
        )
    else:
        contrast_guidance = (
            "Contrast-verbalization policy: explicit. You may mention contrast_process briefly when the style calls for it, but keep the expected process more central than the contrast.\n"
        )
    if contrast_verbalization == "metadata_only":
        controls_guidance = (
            "Use the configured render_family, render_template, cause_type, object_set_id, object_type, material_or_surface, geometry_or_medium, timescale, path_shape, and expected_process as controls. When present, also use template_sequence, start_position, end_position, direction_evidence, vertical_relation_language, directional_polarity, measurement_frame, initial_speed_evidence, later_speed_evidence, initial_size_evidence, later_size_evidence, size_change_evidence, quantity_change, rate_evidence, speed_geometry, endpoint, endpoint_tendency, endpoint_risk, stopping_evidence, bridge_type, bridge_clause, evidence_clause, same_sentence_evidence_terms, same_sentence_target_terms, contrast_evidence_bans, reference_frame, landmark, initial_relation_to_landmark, later_relation_to_landmark, crossing_direction, verb_type, event_verb, mechanism, object_shape, body_pose, contact_surface, surface_friction, rotation_allowed, rotation_evidence, and motion_pattern as concrete evidence fields.\n"
        )
        process_terms_guidance = (
            "When a spec includes preferred_process_terms, use one natural form from that list if it fits the style. Do not list every variant in one document.\n"
        )
        variation_guidance = (
            "Avoid robotic schema narration. Do not reuse the same sentence frame across items in a batch. Avoid boilerplate phrases such as \"In a simple physical scene\", \"The visible outcome was that\", \"The useful evidence links\", or repeated \"Over [time], X happened\" structures. Vary sentence order: some documents may start with the cause, some with the observation, and some with the outcome. Do not add summary/meta sentences about what the evidence teaches.\n"
        )
    else:
        controls_guidance = (
            "Use the configured family, concept_pair, render_template, cause_type, object_set_id, object_type, material_or_surface, geometry_or_medium, timescale, path_shape, expected_process, and, when provided in the render payload, contrast_process as controls. When present, also use template_sequence, start_position, end_position, direction_evidence, vertical_relation_language, directional_polarity, measurement_frame, initial_speed_evidence, later_speed_evidence, initial_size_evidence, later_size_evidence, size_change_evidence, quantity_change, rate_evidence, speed_geometry, endpoint, endpoint_tendency, endpoint_risk, stopping_evidence, bridge_type, bridge_clause, evidence_clause, same_sentence_evidence_terms, same_sentence_target_terms, contrast_evidence_bans, reference_frame, landmark, initial_relation_to_landmark, later_relation_to_landmark, crossing_direction, verb_type, event_verb, mechanism, object_shape, body_pose, contact_surface, surface_friction, rotation_allowed, rotation_evidence, and motion_pattern as concrete evidence fields.\n"
        )
        process_terms_guidance = (
            "When a spec includes preferred_process_terms or preferred_contrast_terms, use one natural form from those lists if it fits the style. Do not list every variant in one document.\n"
        )
        variation_guidance = (
            "Avoid robotic schema narration. Do not reuse the same sentence frame across items in a batch. Avoid boilerplate phrases such as \"In a simple physical scene\", \"The visible outcome was that\", \"The useful evidence links\", or repeated \"Over [time], X happened rather than Y\" structures. Vary sentence order: some documents may start with the cause, some with the observation, some with the outcome, and some with a contrast. Do not add summary/meta sentences about what the evidence teaches.\n"
        )
    if contrast_verbalization == "metadata_only":
        leakage_guidance = (
            "Avoid words that expose the data construction process, including benchmark, target, metadata, scenario spec, answer key, Context1, Context2, Target1, or Target2.\n"
        )
    else:
        leakage_guidance = (
            "Avoid words that expose the data construction process, including benchmark, target, metadata, scenario spec, expected process, or contrast process.\n"
        )

    return (
        "Write one compact physical-dynamics training document for each scenario spec.\n"
        f"Physical-dynamics definition: {definition}\n"
        "Each document should be natural prose for pretraining, not a quiz, answer key, or benchmark item.\n"
        "Hard rule: preserve the cause/process/outcome structure from each spec. The text must make the cause evidence, the expected process, and the outcome evidence visible in ordinary prose.\n"
        "Hard rule: do not solve the example only by object identity. The cause and physical change must carry the meaning, not shortcuts such as a familiar object always having one outcome.\n"
        "Hard rule: include at least one active event sentence where the object or person is the subject and the expected process is the main verb or verb phrase, such as \"The ball rolled across the mat\" or \"The skater slid across the ice.\" Avoid passive labels like \"the scene shows rolling.\"\n"
        "Hard rule: when a spec includes lexical_target, use that exact lexical_target naturally in the document. Treat it as the assigned surface form for this document; do not choose a different preferred form and do not list variants.\n"
        "Hard rule: for speed_change specs with speed_geometry, include the initial_rate, later_rate, same_time_comparison, definition_bridge, and process_bridge, unless render_template begins with \"lexical_shortcut_\" or \"lexical_bridge_\". Optional consequences may appear after that geometry, but must not replace the distance-over-time evidence.\n"
        "Hard rule: for render_template=\"speed_rate_bridge_5_sentence\", use the template_sequence order exactly: cause, initial_rate, later_rate, same_time_comparison, then definition_bridge plus process_bridge plus outcome_evidence. Keep these as short separate sentences, and copy definition_bridge and process_bridge wording from speed_geometry.\n"
        "Hard rule: for render_template values beginning with \"lexical_shortcut_\", write one or two short active sentences. Put the assigned lexical_target and at least one same_sentence_evidence_terms phrase in the same sentence. Use bridge_clause and evidence_clause naturally when present. Do not use long definition chains such as \"X means Y means Z\". Do not use any contrast_evidence_bans.\n"
        "Hard rule: for render_template values beginning with \"lexical_bridge_\", write simple active prose that puts the assigned lexical_target and at least one same_sentence_evidence_terms phrase in the same sentence. The bridge should feel like natural evidence in a broader event, not a glossary definition.\n"
        "Object-agnostic shortcut guidance: when object_set_id is present, treat repeated object sets as deliberate. Do not rely on object identity alone; let the cause, mechanism, and same-sentence evidence cue determine the process.\n"
        f"{contrast_guidance}"
        f"{controls_guidance}"
        "For landmark_direction specs, include the landmark in or near the active event sentence, and explicitly state both the initial relation and the later relation. For direction_neutral verbs such as passed, crossed, or went by, the before/after landmark relation and mechanism must make the direction clear.\n"
        f"{process_terms_guidance}"
        "Path-shape guidance: track path_shape as an axis in v1, but do not make path shape the main objective unless the spec naturally requires it. If path_shape is none, avoid forcing a trajectory description. Dedicated path-shape families such as roll/slide, swing/spin, drift, and curve/straight are v2 targets.\n"
        "Follow each spec's discourse style as a perspective or genre, not as a label. Use style_guidance as instructions, not text to copy. The batch should feel like different kinds of natural text: a child noticing an everyday event, a short simple wiki explanation, a practical handling note, or a compact observation record.\n"
        f"{variation_guidance}"
        f"{leakage_guidance}"
        "Return JSON with this shape: {\"items\": [{\"id\": string, \"text\": string}]}.\n\n"
        "SCENARIO_SPECS_JSON:\n"
        f"{json.dumps(prompt_payload, indent=2)}"
    )


def normalize_documents(items: list[dict[str, Any]], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach stable metadata from specs to rendered document rows."""

    specs_by_id = {str(spec["id"]): spec for spec in specs}
    normalized = []
    for index, item in enumerate(items):
        doc_id = str(item.get("id") or item.get("scenario_id") or specs[index]["id"])
        if doc_id not in specs_by_id and index < len(specs):
            doc_id = str(specs[index]["id"])
        spec = specs_by_id[doc_id]
        normalized.append(
            {
                "id": doc_id,
                "text": str(item.get("text", "")).strip(),
                "metadata": {
                    "concept_id": spec["concept_id"],
                    "profile_id": spec.get("profile_id"),
                    "domain": spec["domain"],
                    "coverage_pack_id": spec["coverage_pack_id"],
                    "document_type": spec["document_type"],
                    "style_id": spec.get("style_id"),
                    "style_name": spec.get("style_name"),
                    "signal_order_id": spec.get("signal_order_id"),
                    "signal_order_name": spec.get("signal_order_name"),
                    "positive_subtype_id": spec.get("positive_subtype_id"),
                    "positive_material": spec.get("positive_material"),
                    "negative_subtype_id": spec.get("negative_subtype_id"),
                    "negative_material": spec.get("negative_material"),
                    "state_case_id": spec.get("state_case_id"),
                    "before_material": spec.get("before_material"),
                    "after_material": spec.get("after_material"),
                    "neighbor": spec.get("neighbor"),
                    "lexical_forms": spec.get("lexical_forms", []),
                    "material_class_id": spec.get("material_class_id"),
                    "material_class_label": spec.get("material_class_label"),
                    "material": spec.get("material"),
                    "affordances": spec.get("affordances", []),
                    "target_affordance": spec.get("target_affordance"),
                    "required_affordances": spec.get("required_affordances", []),
                    "companion_material_class_id": spec.get("companion_material_class_id"),
                    "companion_material": spec.get("companion_material"),
                    "companion_required_affordances": spec.get("companion_required_affordances", []),
                    "tags": spec.get("tags", []),
                    "ewok_anchors": spec.get("ewok_anchors", []),
                    "profile_type": spec.get("profile_type"),
                    "scenario_id": spec.get("scenario_id"),
                    "family": spec.get("family"),
                    "physical_family": spec.get("physical_family"),
                    "render_template": spec.get("render_template"),
                    "template_sequence": spec.get("template_sequence"),
                    "family_label": spec.get("family_label"),
                    "concept_pair": spec.get("concept_pair", []),
                    "cause_type": spec.get("cause_type"),
                    "object_set_id": spec.get("object_set_id"),
                    "object_type": spec.get("object_type"),
                    "object": spec.get("object"),
                    "object_shape": spec.get("object_shape"),
                    "body_pose": spec.get("body_pose"),
                    "material_or_surface": spec.get("material_or_surface"),
                    "contact_surface": spec.get("contact_surface"),
                    "surface_friction": spec.get("surface_friction"),
                    "rotation_allowed": spec.get("rotation_allowed"),
                    "rotation_evidence": spec.get("rotation_evidence"),
                    "motion_pattern": spec.get("motion_pattern"),
                    "geometry_or_medium": spec.get("geometry_or_medium"),
                    "timescale": spec.get("timescale"),
                    "path_shape": spec.get("path_shape"),
                    "start_position": spec.get("start_position"),
                    "end_position": spec.get("end_position"),
                    "direction_evidence": spec.get("direction_evidence"),
                    "vertical_relation_language": spec.get("vertical_relation_language"),
                    "directional_polarity": spec.get("directional_polarity"),
                    "measurement_frame": spec.get("measurement_frame"),
                    "initial_speed_evidence": spec.get("initial_speed_evidence"),
                    "later_speed_evidence": spec.get("later_speed_evidence"),
                    "initial_size_evidence": spec.get("initial_size_evidence"),
                    "later_size_evidence": spec.get("later_size_evidence"),
                    "size_change_evidence": spec.get("size_change_evidence"),
                    "quantity_change": spec.get("quantity_change"),
                    "rate_evidence": spec.get("rate_evidence"),
                    "endpoint": spec.get("endpoint"),
                    "endpoint_tendency": spec.get("endpoint_tendency"),
                    "endpoint_risk": spec.get("endpoint_risk"),
                    "stopping_evidence": spec.get("stopping_evidence"),
                    "speed_geometry": spec.get("speed_geometry"),
                    "bridge_type": spec.get("bridge_type"),
                    "bridge_clause": spec.get("bridge_clause"),
                    "evidence_clause": spec.get("evidence_clause"),
                    "same_sentence_evidence_terms": spec.get("same_sentence_evidence_terms", []),
                    "same_sentence_target_terms": spec.get("same_sentence_target_terms", []),
                    "contrast_evidence_bans": spec.get("contrast_evidence_bans", []),
                    "reference_frame": spec.get("reference_frame"),
                    "landmark": spec.get("landmark"),
                    "initial_relation_to_landmark": spec.get("initial_relation_to_landmark"),
                    "later_relation_to_landmark": spec.get("later_relation_to_landmark"),
                    "crossing_direction": spec.get("crossing_direction"),
                    "verb_type": spec.get("verb_type"),
                    "event_verb": spec.get("event_verb"),
                    "mechanism": spec.get("mechanism"),
                    "expected_process": spec.get("expected_process"),
                    "contrast_process": spec.get("contrast_process"),
                    "canonical_concept_id": spec.get("canonical_concept_id"),
                    "canonical_contrast_concept_id": spec.get("canonical_contrast_concept_id"),
                    "canonical_balance_role": spec.get("canonical_balance_role"),
                    "canonical_inventory_version": spec.get("canonical_inventory_version"),
                    "cause_evidence": spec.get("cause_evidence"),
                    "outcome_evidence": spec.get("outcome_evidence"),
                    "required_cause_terms": spec.get("required_cause_terms", []),
                    "required_process_terms": spec.get("required_process_terms", []),
                    "required_outcome_terms": spec.get("required_outcome_terms", []),
                    "lexical_targets": spec.get("lexical_targets", []),
                    "lexical_target": spec.get("lexical_target"),
                    "preferred_process_terms": spec.get("preferred_process_terms", []),
                    "preferred_contrast_terms": spec.get("preferred_contrast_terms", []),
                    "shortcut_terms": spec.get("shortcut_terms", []),
                },
            }
        )
    return normalized


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_json_from_prompt(prompt: str) -> dict[str, Any]:
    marker = "SCENARIO_SPECS_JSON:"
    if marker not in prompt:
        return {}
    payload = prompt.split(marker, 1)[1].strip()
    if NO_THINK_SUFFIX in payload:
        payload = payload.split(NO_THINK_SUFFIX, 1)[0].strip()
    return extract_json(payload)


def _physical_process_clause(subject: str, process: str) -> str:
    process = str(process).strip()
    if not process:
        return f"{subject} changed"
    if process.lower().endswith("ing"):
        return f"{subject} was {process}"
    return f"{subject} {process}"


def _physical_lexical_clause(subject: str, lexical_target: str, fallback_process: str) -> str:
    lexical_target = str(lexical_target).strip()
    if not lexical_target:
        return _physical_process_clause(subject, fallback_process)
    lower = lexical_target.lower()
    if lower.startswith("speed is "):
        return f"{subject}'s {lexical_target}"
    if lower in {
        "speeding up",
        "slowing down",
        "losing speed",
        "gaining speed",
        "increasing speed",
        "decreasing speed",
    }:
        return f"{subject} was {lexical_target}"
    if " " in lexical_target:
        return f"{subject} {lexical_target}"
    if lower in {
        "sink",
        "float",
        "fall",
        "rise",
        "slide",
        "slip",
        "roll",
        "accelerate",
        "slow",
        "grow",
        "shrink",
    }:
        return f"{subject} began to {lexical_target}"
    if lower.endswith("ing"):
        return f"{subject} was {lexical_target}"
    return f"{subject} {lexical_target}"


def _physical_contrast_note(
    *,
    contrast_verbalization: str,
    explicit: str,
    implicit: str,
) -> str:
    if contrast_verbalization == "metadata_only":
        return ""
    if contrast_verbalization == "implicit":
        return implicit
    return explicit


def _physical_speed_geometry_note(spec: dict[str, Any]) -> str:
    speed_geometry = spec.get("speed_geometry")
    if not isinstance(speed_geometry, dict) or not speed_geometry:
        return ""

    parts = [
        str(speed_geometry.get("initial_rate") or "").strip(),
        str(speed_geometry.get("later_rate") or "").strip(),
        str(speed_geometry.get("same_time_comparison") or "").strip(),
        str(speed_geometry.get("definition_bridge") or "").strip(),
        str(speed_geometry.get("process_bridge") or "").strip(),
    ]
    consequence = str(speed_geometry.get("optional_consequence") or "").strip()
    if consequence:
        parts.append(f"A possible consequence was that {consequence}.")
    text = " ".join(part.rstrip(".") + "." for part in parts if part)
    return f" {text}" if text else ""


def _mock_text_for_spec(
    spec: dict[str, Any],
    *,
    contrast_verbalization: str = "explicit",
) -> str:
    document_type = str(spec.get("document_type"))
    style_id = str(spec.get("style_id") or "")
    if (
        spec.get("profile_type") == "physical_dynamics"
        or spec.get("physical_family")
        or spec.get("render_family")
        or spec.get("expected_process")
    ):
        object_name = str(spec.get("object") or spec.get("object_type") or "object")
        cause_evidence = str(spec.get("cause_evidence") or spec.get("cause_type") or "a cause changed it")
        expected_process = str(spec.get("expected_process") or "changed")
        lexical_target = str(spec.get("lexical_target") or "")
        contrast_process = str(spec.get("contrast_process") or "stayed the same")
        outcome_evidence = str(spec.get("outcome_evidence") or "the result changed visibly")
        timescale = str(spec.get("timescale") or "a short time")
        material_context = str(spec.get("material_or_surface") or object_name)
        geometry = str(spec.get("geometry_or_medium") or "the setup")
        path_shape = str(spec.get("path_shape") or "none")
        path_note = ""
        if path_shape and path_shape != "none":
            path_note = (
                f" The path detail was {path_shape.replace('_', ' ')}, which matched the way "
                f"{geometry} shaped the motion."
            )
        landmark = str(spec.get("landmark") or "")
        initial_relation = str(spec.get("initial_relation_to_landmark") or "")
        later_relation = str(spec.get("later_relation_to_landmark") or "")
        landmark_note = ""
        if landmark and initial_relation and later_relation:
            landmark_note = (
                f" The landmark clue was that the {object_name} started {initial_relation} "
                f"and later was {later_relation}."
            )
        speed_geometry_note = _physical_speed_geometry_note(spec)
        match = re.search(r"_(\d+)$", str(spec.get("id", "")))
        variant_index = int(match.group(1)) % 5 if match else 0

        if str(spec.get("render_template") or "").startswith(
            ("lexical_shortcut_", "lexical_bridge_")
        ):
            bridge_clause = str(spec.get("bridge_clause") or "").strip()
            evidence_terms = [
                str(term).strip()
                for term in spec.get("same_sentence_evidence_terms", [])
                if str(term).strip()
            ]
            evidence_clause = str(spec.get("evidence_clause") or "").strip()
            evidence = evidence_clause or (evidence_terms[0] if evidence_terms else outcome_evidence)
            lead = _physical_lexical_clause(
                f"The {object_name}",
                lexical_target,
                expected_process,
            )
            if bridge_clause:
                lead = f"{lead} {bridge_clause}"
            return (
                f"{lead}, {evidence}. "
                f"{cause_evidence.capitalize().rstrip('.')}, so {outcome_evidence}."
            )

        if spec.get("render_template") == "speed_rate_bridge_5_sentence":
            speed_geometry = spec.get("speed_geometry") if isinstance(spec.get("speed_geometry"), dict) else {}
            process_and_outcome = " ".join(
                part.strip()
                for part in (
                    str(speed_geometry.get("process_bridge") or ""),
                    f"and {outcome_evidence}",
                )
                if part.strip()
            )
            sentences = [
                cause_evidence,
                str(speed_geometry.get("initial_rate") or ""),
                str(speed_geometry.get("later_rate") or ""),
                str(speed_geometry.get("same_time_comparison") or ""),
                process_and_outcome,
            ]
            definition_bridge = str(speed_geometry.get("definition_bridge") or "").strip()
            if definition_bridge and process_and_outcome:
                sentences[-1] = f"{definition_bridge}, and {process_and_outcome}"
            elif definition_bridge:
                sentences[-1] = definition_bridge
            return " ".join(
                sentence.rstrip(".") + "."
                for sentence in sentences
                if sentence.strip()
            )

        if style_id == "cause_effect_explanation":
            openings = [
                "A short explanation starts with the condition around the object.",
                "One simple way to understand the change is to look at what happened first.",
                "This example shows how the process depends on the setup around an object.",
                "A plain account begins with the cause, not just the object name.",
                f"The process here is {expected_process}, and the condition around the object matters.",
            ]
            contrast_note = _physical_contrast_note(
                contrast_verbalization=contrast_verbalization,
                explicit=(
                    "With a different condition, the event could have shown "
                    f"{contrast_process} instead."
                ),
                implicit="With a different condition, the final trace would have changed.",
            )
            return (
                f"{openings[variant_index]} "
                f"In this example, {cause_evidence}, so {_physical_lexical_clause(f'the {object_name}', lexical_target, expected_process)} within {timescale}. "
                f"The central example uses {material_context}: {outcome_evidence}. "
                f"{contrast_note} "
                f"{path_note}{landmark_note}{speed_geometry_note}"
            )
        if style_id == "comparison_contrast":
            if contrast_verbalization == "explicit":
                openings = [
                    f"Someone trying to get the other outcome, {contrast_process}, would need a different setup.",
                    f"A careful helper would not expect {contrast_process} after this setup.",
                    f"For a practical contrast, imagine the event had ended with {contrast_process}.",
                    f"The setup makes {expected_process} more plausible than {contrast_process}.",
                    f"A person checking the result would compare {expected_process} with {contrast_process}.",
                ]
                closings = [
                    "The contrast would leave a different trace, so the actual change is easier to pick out.",
                    "That difference makes the cause worth noticing, not just the object.",
                    "The result fits this setup better than the alternative.",
                    "A different condition would be needed for the contrast to make sense.",
                    "The practical clue is the change that remains after contact or motion ends.",
                ]
            else:
                openings = [
                    "A practical note starts with the condition that shaped the event.",
                    "A careful helper would first check the setup before naming the result.",
                    "For handling this object, the important detail is the condition before motion.",
                    "The setup makes the actual change visible without relying on the object name.",
                    "A person checking the result would compare the starting clue with the final trace.",
                ]
                closings = [
                    "That trace makes the cause worth noticing, not just the object.",
                    "The result fits the setup and the final position.",
                    "A different setup would leave a different sign.",
                    "The practical clue is the change that remains after contact or motion ends.",
                    "The final trace matches the visible cause and motion.",
                ]
            return (
                f"{openings[variant_index]} "
                f"Here, {cause_evidence}, and {_physical_lexical_clause(f'the {object_name}', lexical_target, expected_process)}. "
                f"That practical difference matters around {geometry}, because {outcome_evidence}. "
                f"{closings[variant_index]}{path_note}{landmark_note}{speed_geometry_note}"
            )
        if style_id == "observation_log":
            openings = [
                f"Observation during {timescale}: {cause_evidence}.",
                f"Quick record from the setup: {cause_evidence}.",
                f"Noted over {timescale}: {cause_evidence}.",
                f"Field note for the setup: {cause_evidence}.",
                f"The first visible clue was that {cause_evidence}.",
            ]
            followups = [
                _physical_contrast_note(
                    contrast_verbalization=contrast_verbalization,
                    explicit=f"A different setup might have shown {contrast_process}, but this record points to the actual change.",
                    implicit="A different setup would have left a different sign.",
                ),
                _physical_contrast_note(
                    contrast_verbalization=contrast_verbalization,
                    explicit=f"The alternative, {contrast_process}, would have left a different sign.",
                    implicit="The note would read differently under another condition.",
                ),
                _physical_contrast_note(
                    contrast_verbalization=contrast_verbalization,
                    explicit=f"That makes {contrast_process} a weaker description of the event.",
                    implicit="That makes the actual physical change easier to identify.",
                ),
                _physical_contrast_note(
                    contrast_verbalization=contrast_verbalization,
                    explicit=f"The note would read differently if the result had been {contrast_process}.",
                    implicit="A changed setup would have changed the note.",
                ),
                _physical_contrast_note(
                    contrast_verbalization=contrast_verbalization,
                    explicit=f"Nothing in the final position fits {contrast_process} as well.",
                    implicit="The final position fits the recorded motion.",
                ),
            ]
            return (
                f"{openings[variant_index]} "
                f"{_physical_lexical_clause(f'The {object_name}', lexical_target, expected_process)}, and the clearest detail was that {outcome_evidence}. "
                f"{followups[variant_index]} "
                f"The material context was {material_context}, with {geometry} shaping the setup.{path_note}{landmark_note}{speed_geometry_note}"
            )

        openings = [
            f"A child watching the event might first notice the {object_name} and then the place around it.",
            f"During cleanup, a child could see the {object_name} before noticing what changed.",
            f"A curious student might point at the {object_name} as the event begins.",
            f"At the table, the first easy thing to notice is the {object_name}.",
            f"In an ordinary moment, a child might watch the {object_name} closely.",
        ]
        closings = [
            _physical_contrast_note(
                contrast_verbalization=contrast_verbalization,
                explicit=f"That is different from the other possible outcome, {contrast_process}, because the visible change happened in this setup.",
                implicit="Another setup would have made the child see a different result.",
            ),
            _physical_contrast_note(
                contrast_verbalization=contrast_verbalization,
                explicit=f"If the event had shown {contrast_process}, the child would have seen a different result.",
                implicit="The child could tell the result from the final position.",
            ),
            _physical_contrast_note(
                contrast_verbalization=contrast_verbalization,
                explicit=f"The other outcome, {contrast_process}, would not match what the child just saw.",
                implicit="What the child saw matched the cause and the motion.",
            ),
            _physical_contrast_note(
                contrast_verbalization=contrast_verbalization,
                explicit=f"So the noticed event fits {expected_process} better than {contrast_process}.",
                implicit=f"So the noticed event fits {expected_process}.",
            ),
            _physical_contrast_note(
                contrast_verbalization=contrast_verbalization,
                explicit=f"The everyday clue separates this result from {contrast_process}.",
                implicit="The everyday clue is the visible physical change.",
            ),
        ]
        return (
            f"{openings[variant_index]} "
            f"When {cause_evidence}, {_physical_lexical_clause(f'the {object_name}', lexical_target, expected_process)}. "
            f"The child could point to the result: {outcome_evidence}. "
            f"{closings[variant_index]}{path_note}{landmark_note}{speed_geometry_note}"
        )

    if spec.get("material_class_id"):
        material = str(spec.get("material") or "water")
        affordances = [str(item) for item in spec.get("required_affordances", []) if str(item).strip()]
        first = _human_list(affordances[:3]) or "move"
        text = (
            f"{material} can {first} in ordinary material scenes. "
            f"The sentence keeps {material} close to its actions so the association is easy to learn."
        )
        if style_id == "class_map_paragraph":
            text += " The class map links this one material word with several plausible actions."
        return text

    positive = spec.get("positive_material") or "soup"
    before = spec.get("before_material") or positive
    after = spec.get("after_material") or "a hard block"
    neighbor = spec.get("neighbor") or "mix"
    negative = spec.get("negative_material") or "cloth"
    lexical_forms = [str(item) for item in spec.get("lexical_forms", []) if str(item).strip()]
    target = lexical_forms[0] if lexical_forms else "stir"
    target_past = next((form for form in lexical_forms if form.endswith("ed")), f"{target}ed")
    target_ing = next((form for form in lexical_forms if form.endswith("ing")), f"{target}ing")

    if document_type == "state_contrast":
        text = (
            f"{before} is a material that can be {target_past} while it is in the right state. "
            f"The action changes how the material moves or sits, so its parts can shift in the expected way. "
            f"After it changes into {after}, the same motion no longer works. "
            f"The changed state holds together or moves differently, so it may crack, slump, stay fixed, or respond another way instead of {target_ing}. "
            f"This shows that {target_ing} depends on the material state, not just on the action of the hand."
        )
    elif document_type == "neighbor_contrast":
        text = (
            f"{positive} can be {target_past} because the material has the right kind of motion for that action. "
            f"That is different from {neighbor}, which describes a nearby action with a different boundary. "
            f"For contrast, {negative} does not behave like a material that is naturally {target_past}; it keeps its own form or responds in another way. "
            f"The important participant is the material being {target_past}, because its pieces or regions have to move in the right pattern."
        )
    elif document_type == "subtype_explanation":
        text = (
            f"{positive} belongs to a kind of material that can be {target_past}. "
            f"When {target_ing} happens, the material gives way and changes position instead of staying like one fixed object. "
            f"The motion can reveal whether the material flows, slides, tumbles, deforms, or remains stuck in place. "
            f"A rigid object would mostly stay in one piece, but {positive} can change arrangement under the action. "
            f"That is why {target_ing} is a useful action for this subtype of material."
        )
    else:
        text = (
            f"The {positive} was {target_past} in a simple everyday scene. "
            f"As the action happened, the material's surface and inner parts shifted in a visible way. "
            f"The {target_ing} showed that the {positive} had the right physical form for this event. "
            f"The tool or container helped, but the central participant was the {positive}, because it could flow, slide, tumble, or deform under the motion. "
            f"A single hard object would not respond in the same way."
        )

    if style_id == "concept_map_paragraph":
        text += (
            f" In the larger map of actions, {target} is near {neighbor} and mix, but it is different from "
            "shake or squeeze because each action has a different material boundary."
        )
    return text


class DeterministicConceptCurriculumMockGenerator:
    """Local generator used in tests and dry runs."""

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
        del system_prompt, temperature, max_tokens, stage, response_format
        payload = _read_json_from_prompt(prompt)
        specs = payload.get("scenario_specs", [])
        constraints = payload.get("constraints", {})
        if constraints.get("positive_only"):
            contrast_verbalization = "metadata_only"
        else:
            contrast_verbalization = str(constraints.get("contrast_verbalization") or "explicit")
        return json.dumps(
            {
                "items": [
                    {
                        "id": spec["id"],
                        "text": _mock_text_for_spec(
                            spec,
                            contrast_verbalization=contrast_verbalization,
                        ),
                    }
                    for spec in specs
                ]
            }
        )


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


def _document_render_prompt(
    *,
    config: dict[str, Any],
    profile: dict[str, Any],
    constraints: dict[str, Any],
    batch: list[dict[str, Any]],
) -> str:
    if "material_class_profile" in config:
        return build_material_class_render_prompt(
            specs=batch,
            material_class_profile=profile,
            constraints=constraints,
        )
    if "physical_dynamics_profile" in config:
        return build_physical_dynamics_render_prompt(
            specs=batch,
            physical_dynamics_profile=profile,
            constraints=constraints,
        )
    return build_render_prompt(
        specs=batch,
        concept_profile=profile,
        constraints=constraints,
    )


def _render_batch_stem(start: int, batch: list[dict[str, Any]]) -> str:
    return f"batch_{start:06d}_n{len(batch):03d}"


def _render_checkpoint_path(
    checkpoint_dir: Path | None,
    *,
    start: int,
    batch: list[dict[str, Any]],
) -> Path | None:
    if checkpoint_dir is None:
        return None
    return checkpoint_dir / f"{_render_batch_stem(start, batch)}.jsonl"


def _render_spec_fingerprint(spec: dict[str, Any]) -> str:
    """Return a stable hash for the render-relevant scenario spec."""

    payload = {
        key: value
        for key, value in spec.items()
        if key not in {"signal_order_guidance"}
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _render_checkpoint_fingerprints(batch: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(spec["id"]): _render_spec_fingerprint(spec)
        for spec in batch
    }


def _attach_render_checkpoint_metadata(
    documents: list[dict[str, Any]],
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fingerprints = _render_checkpoint_fingerprints(batch)
    rows = []
    for document in documents:
        row = dict(document)
        doc_id = str(row.get("id"))
        row["render_checkpoint"] = {
            "version": 1,
            "spec_fingerprint": fingerprints.get(doc_id),
        }
        rows.append(row)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_render_checkpoint(
    checkpoint_dir: Path | None,
    *,
    start: int,
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    path = _render_checkpoint_path(checkpoint_dir, start=start, batch=batch)
    if path is None or not path.exists():
        return None
    rows = _read_jsonl(path)
    expected_ids = [str(spec["id"]) for spec in batch]
    actual_ids = [str(row.get("id")) for row in rows]
    if actual_ids != expected_ids:
        raise ValueError(
            f"render checkpoint {path} does not match expected batch ids."
        )
    expected_fingerprints = _render_checkpoint_fingerprints(batch)
    actual_fingerprints = [
        str(
            (row.get("render_checkpoint") or {}).get("spec_fingerprint")
            or ""
        )
        for row in rows
    ]
    if actual_fingerprints != [expected_fingerprints[doc_id] for doc_id in expected_ids]:
        return None
    return rows


def _write_render_checkpoint(
    checkpoint_dir: Path | None,
    *,
    start: int,
    batch: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> None:
    path = _render_checkpoint_path(checkpoint_dir, start=start, batch=batch)
    if path is not None:
        _write_jsonl(documents, path)


def _write_render_failure(
    failure_dir: Path | None,
    *,
    start: int,
    batch: list[dict[str, Any]],
    attempt: int,
    error: Exception,
    raw: str,
) -> Path | None:
    if failure_dir is None:
        return None
    failure_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_render_batch_stem(start, batch)}_attempt{attempt:02d}"
    raw_path = failure_dir / f"{stem}.raw.txt"
    meta_path = failure_dir / f"{stem}.json"
    raw_path.write_text(raw, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "stage": "document_render",
                "start": start,
                "batch_size": len(batch),
                "attempt": attempt,
                "ids": [str(spec.get("id")) for spec in batch],
                "error": str(error),
                "raw_output_path": str(raw_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return raw_path


def _parse_render_documents(raw: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = json_items(extract_json(raw))
    if len(items) != len(batch):
        raise ValueError(f"document_render returned {len(items)} items; expected {len(batch)}.")
    return normalize_documents(items, batch)


def _render_document_batch(
    *,
    config: dict[str, Any],
    profile: dict[str, Any],
    constraints: dict[str, Any],
    generator: Any,
    params: dict[str, Any],
    start: int,
    batch: list[dict[str, Any]],
    checkpoint_dir: Path | None,
    failure_dir: Path | None,
    invalid_json_retries: int,
    min_fallback_batch_size: int,
) -> list[dict[str, Any]]:
    checkpointed = _load_render_checkpoint(checkpoint_dir, start=start, batch=batch)
    if checkpointed is not None:
        return checkpointed

    prompt = _document_render_prompt(
        config=config,
        profile=profile,
        constraints=constraints,
        batch=batch,
    )
    last_error: Exception | None = None
    raw_failure_paths: list[Path] = []
    for attempt in range(1, invalid_json_retries + 2):
        raw = _call_generator(
            generator,
            prompt=prompt,
            system_prompt=_system_prompt(),
            stage="document_render",
            params=params,
            response_format={"type": "json_object"},
        )
        try:
            documents = _parse_render_documents(raw, batch)
        except ValueError as exc:
            last_error = exc
            raw_path = _write_render_failure(
                failure_dir,
                start=start,
                batch=batch,
                attempt=attempt,
                error=exc,
                raw=raw,
            )
            if raw_path is not None:
                raw_failure_paths.append(raw_path)
            continue

        documents = _attach_render_checkpoint_metadata(documents, batch)
        _write_render_checkpoint(
            checkpoint_dir,
            start=start,
            batch=batch,
            documents=documents,
        )
        return documents

    if len(batch) > max(1, min_fallback_batch_size):
        mid = len(batch) // 2
        left = _render_document_batch(
            config=config,
            profile=profile,
            constraints=constraints,
            generator=generator,
            params=params,
            start=start,
            batch=batch[:mid],
            checkpoint_dir=checkpoint_dir,
            failure_dir=failure_dir,
            invalid_json_retries=invalid_json_retries,
            min_fallback_batch_size=min_fallback_batch_size,
        )
        right = _render_document_batch(
            config=config,
            profile=profile,
            constraints=constraints,
            generator=generator,
            params=params,
            start=start + mid,
            batch=batch[mid:],
            checkpoint_dir=checkpoint_dir,
            failure_dir=failure_dir,
            invalid_json_retries=invalid_json_retries,
            min_fallback_batch_size=min_fallback_batch_size,
        )
        documents = left + right
        documents = _attach_render_checkpoint_metadata(documents, batch)
        _write_render_checkpoint(
            checkpoint_dir,
            start=start,
            batch=batch,
            documents=documents,
        )
        return documents

    batch_ids = ", ".join(str(spec.get("id")) for spec in batch)
    failure_note = ""
    if raw_failure_paths:
        failure_note = " Raw output saved to: " + ", ".join(str(path) for path in raw_failure_paths)
    raise ValueError(
        f"document_render failed for batch starting at {start} "
        f"with ids [{batch_ids}]: {last_error}.{failure_note}"
    ) from last_error


def render_documents(
    *,
    config: dict[str, Any],
    specs: list[dict[str, Any]],
    generator: Any,
    checkpoint_dir: str | Path | None = None,
    failure_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Render all scenario specs with the configured generator."""

    profile = _active_profile(config)
    constraints = dict(config.get("document_constraints", {}))
    batch_size = int(config.get("batching", {}).get("documents_per_request", 10))
    request = config.get("request", {})
    invalid_json_retries = max(0, int(request.get("invalid_json_retries", 1)))
    min_fallback_batch_size = max(1, int(request.get("min_fallback_documents_per_request", 1)))
    params = _stage_params(config, "document_render")
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    failure_path = Path(failure_dir) if failure_dir is not None else None
    documents: list[dict[str, Any]] = []

    for start in range(0, len(specs), batch_size):
        batch = specs[start : start + batch_size]
        documents.extend(
            _render_document_batch(
                config=config,
                profile=profile,
                constraints=constraints,
                generator=generator,
                params=params,
                start=start,
                batch=batch,
                checkpoint_dir=checkpoint_path,
                failure_dir=failure_path,
                invalid_json_retries=invalid_json_retries,
                min_fallback_batch_size=min_fallback_batch_size,
            )
        )

    return documents


def export_train_text(rows: list[dict[str, Any]], path: Path) -> int:
    """Write pass/warn documents to a paragraph-separated train text file."""

    kept = [
        str(row.get("text", "")).strip()
        for row in rows
        if str(row.get("audit_status", "pass")) in {"pass", "warn"} and str(row.get("text", "")).strip()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return len(kept)


def build_audit_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"pass": 0, "warn": 0, "fail": 0}
    flag_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("audit_status", "pass"))
        counts[status] = counts.get(status, 0) + 1
        for flag in row.get("audit_flags", []):
            code = str(flag.get("code", "unknown"))
            flag_counts[code] = flag_counts.get(code, 0) + 1
    return {
        "counts": counts,
        "flag_counts": dict(sorted(flag_counts.items())),
    }


def run_concept_curriculum_pipeline(
    *,
    config: dict[str, Any],
    generator: Any,
    model_id: str,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Run the concept-curriculum generation pipeline."""

    resolved_output_dir = Path(output_dir or config["output_dir"])
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = ROOT / resolved_output_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    def emit(message: str) -> None:
        if progress is not None:
            progress(message)

    profile = _active_profile(config)
    profile_id = _profile_id(profile)
    specs = build_scenario_specs(config)
    seed = str(config.get("selection_seed", "concept-curriculum"))
    random.Random(seed).shuffle(specs)

    paths = {
        "scenario_plans": resolved_output_dir / "scenario_plans.jsonl",
        "render_checkpoints": resolved_output_dir / "render_checkpoints",
        "render_failures": resolved_output_dir / "render_failures",
        "documents_draft": resolved_output_dir / "documents_draft.jsonl",
        "documents_audited": resolved_output_dir / "documents_audited.jsonl",
        "audit_report": resolved_output_dir / "audit_report.json",
        "train": resolved_output_dir / "train.txt",
        "manifest": resolved_output_dir / "manifest.json",
    }

    emit(f"writing {len(specs)} scenario specs")
    _write_jsonl(specs, paths["scenario_plans"])

    emit("rendering concept documents")
    documents = render_documents(
        config=config,
        specs=specs,
        generator=generator,
        checkpoint_dir=paths["render_checkpoints"],
        failure_dir=paths["render_failures"],
    )
    _write_jsonl(documents, paths["documents_draft"])

    emit("auditing concept documents")
    specs_by_id = {str(spec["id"]): spec for spec in specs}
    audited = audit_documents(
        documents,
        specs_by_id=specs_by_id,
        concept_profile=profile,
        constraints=dict(config.get("document_constraints", {})),
    )
    _write_jsonl(audited, paths["documents_audited"])

    audit_report = build_audit_report(audited)
    paths["audit_report"].write_text(json.dumps(audit_report, indent=2), encoding="utf-8")

    train_count = export_train_text(audited, paths["train"])

    manifest = {
        "version": config.get("version"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "config_path": str(config_path) if config_path is not None else None,
        "output_dir": str(resolved_output_dir),
        "concept_id": profile_id,
        "profile_id": profile_id,
        "coverage_packs": config.get("coverage_packs", []),
        "counts": {
            "scenario_plans": len(specs),
            "documents_draft": len(documents),
            "documents_audited": len(audited),
            "train_documents": train_count,
            **audit_report["counts"],
        },
        "paths": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "CONCEPT_DOCUMENT_TYPES",
    "CORRELATION_DOCUMENT_TYPES",
    "DEFAULT_CONFIG_PATH",
    "DeterministicConceptCurriculumMockGenerator",
    "PHYSICAL_DYNAMICS_DOCUMENT_TYPES",
    "ScenarioSpec",
    "build_material_class_render_prompt",
    "build_material_class_specs",
    "build_physical_dynamics_render_prompt",
    "build_physical_dynamics_specs",
    "build_render_prompt",
    "build_scenario_specs",
    "extract_json",
    "json_items",
    "load_pipeline_config",
    "render_documents",
    "run_concept_curriculum_pipeline",
]
