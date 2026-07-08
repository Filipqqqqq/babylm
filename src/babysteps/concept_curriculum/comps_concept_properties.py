"""COMPS-oriented concept-property characterization data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from babysteps.material_world.fireworks_client import FireworksChatClient, load_api_key
from babysteps.paths import resolve_path


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
DEFAULT_MODEL_ID = "accounts/fireworks/models/qwen3p7-plus"
DEFAULT_TARGET_WORDS = 200
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 768
SINGLE_PROFILE = "single_profile"
CLEAN_MULTIDOC_NO_NEIGHBORS = "clean_multidoc_no_neighbors"
LIFE_CYCLE_AND_MOVEMENT = "life_cycle_and_movement"
BODY_AND_SIGNALS = "body_and_signals"
ECOLOGY_AND_GROWTH = "ecology_and_growth"
OPERATION_AND_COMPONENTS = "operation_and_components"
INGREDIENTS_AND_SERVING = "ingredients_and_serving"
FORMS_AND_USES = "forms_and_uses"
FOOD_PAIRINGS_FORMS_PACKAGING = "food_pairings_forms_packaging"
ANIMAL_HABITATS_VARIATION_INTERACTIONS = "animal_habitats_variation_interactions"
PLAY_OBJECT_GAMES_AND_VARIANTS = "play_object_games_and_variants"
BODY_PART_ACTIONS_INJURY_SENSATION = "body_part_actions_injury_sensation"
OBJECT_SETTINGS_AND_ACCESSORIES = "object_settings_and_accessories"
PROMPT_ONLY = "prompt_only"
FIREWORKS = "fireworks"
TEMPLATE = "template"
RENDERER_TYPES = {PROMPT_ONLY, FIREWORKS, TEMPLATE}

CLEAN_MULTIDOC_STYLES = (
    {
        "id": "core_profile",
        "min_words": 60,
        "max_words": 90,
        "focus": (
            "what the concept is, its broad category, its most typical appearance "
            "or form, and the central facts a reader would expect in ordinary "
            "educational text"
        ),
    },
    {
        "id": "parts_and_properties",
        "min_words": 60,
        "max_words": 90,
        "focus": (
            "important parts, physical properties, materials, textures, sounds, "
            "motions, or state changes when they are common and diagnostic"
        ),
    },
    {
        "id": "contexts_and_interactions",
        "min_words": 65,
        "max_words": 95,
        "focus": (
            "common locations, habitats, settings, biological needs for living "
            "things, human uses for human-made objects or foods, and ordinary "
            "interactions"
        ),
    },
    {
        "id": "natural_contrasts",
        "min_words": 70,
        "max_words": 100,
        "focus": (
            "brief natural comparisons to familiar related concepts from "
            "ordinary world knowledge"
        ),
    },
)
SPECIALIZED_CLEAN_MULTIDOC_STYLES = (
    {
        "id": LIFE_CYCLE_AND_MOVEMENT,
        "min_words": 70,
        "max_words": 105,
        "focus": (
            "how the animal moves, grows, reproduces, survives, and changes over "
            "its life. Include ordinary movement on land, in water, or in air when "
            "relevant; feeding and hunting behavior when central; young animals, "
            "eggs, nests, dens, or parental care when relevant; and common body "
            "or skin changes when they are typical"
        ),
    },
    {
        "id": BODY_AND_SIGNALS,
        "min_words": 70,
        "max_words": 105,
        "focus": (
            "the animal's important body parts, body covering, shape, limbs, mouth "
            "or beak, teeth, claws, feet, tail, eyes, ears, colors, sounds, displays, "
            "and social signals when they are typical and useful for recognizing it"
        ),
    },
    {
        "id": ECOLOGY_AND_GROWTH,
        "min_words": 70,
        "max_words": 105,
        "focus": (
            "how the plant grows, changes through seasons, reproduces or spreads, "
            "and fits into its environment. Include growth conditions such as soil, "
            "water, and sunlight when relevant; roots, trunk, bark, twigs, branches, "
            "leaves, flowers, or seeds when central; seasonal changes; and ordinary "
            "relationships with animals, people, soil, water, and nearby habitats"
        ),
    },
    {
        "id": OPERATION_AND_COMPONENTS,
        "min_words": 70,
        "max_words": 105,
        "focus": (
            "how the human-made object or device works, what major components it "
            "has, what powers or activates it, what it connects to, how people "
            "control it, and what changes when it is used or turned on, when relevant"
        ),
    },
    {
        "id": INGREDIENTS_AND_SERVING,
        "min_words": 70,
        "max_words": 105,
        "focus": (
            "what the food is made from, how it is prepared or packaged, how it "
            "tastes and feels, how people serve or eat it, common containers or "
            "portions, and ordinary effects such as spilling, leaving marks, "
            "melting, mixing, or changing texture when relevant"
        ),
    },
    {
        "id": FORMS_AND_USES,
        "min_words": 70,
        "max_words": 105,
        "focus": (
            "the substance's common forms, texture, source, packaging, storage, "
            "how it changes when mixed, heated, dried, dissolved, or spilled, "
            "and ordinary uses in foods, homes, crafts, tools, or industry when relevant"
        ),
    },
    {
        "id": FOOD_PAIRINGS_FORMS_PACKAGING,
        "min_words": 70,
        "max_words": 105,
        "focus": (
            "ordinary ways the food is served, paired with other foods, bought, "
            "stored, packaged, portioned, sliced, cooked, frozen, dried, canned, "
            "or used in everyday meals and recipes. Include common containers, "
            "packets, tins, jars, wrappers, and typical eating contexts when relevant"
        ),
    },
    {
        "id": ANIMAL_HABITATS_VARIATION_INTERACTIONS,
        "min_words": 70,
        "max_words": 105,
        "focus": (
            "where the animal commonly lives or is kept, how people encounter or "
            "interact with it, ordinary body variation such as colors, patterns, "
            "horns, spots, skin, fur, feathers, scales, or size, and everyday risks, "
            "benefits, work roles, farm roles, hunting, care, or disease when relevant"
        ),
    },
    {
        "id": PLAY_OBJECT_GAMES_AND_VARIANTS,
        "min_words": 70,
        "max_words": 105,
        "focus": (
            "how the play object is used across common games, sports, playgrounds, "
            "beaches, homes, schools, teams, and informal play. Include common "
            "game variants, motions such as kicking, throwing, catching, bouncing, "
            "hitting, spinning, or rolling, and ordinary materials or sizes when relevant"
        ),
    },
    {
        "id": BODY_PART_ACTIONS_INJURY_SENSATION,
        "min_words": 70,
        "max_words": 105,
        "focus": (
            "what the body part does, where it is located, how it moves or changes, "
            "what sensations it supports, how it helps eating, speech, balance, "
            "touch, taste, smell, sight, hearing, breathing, or movement when relevant, "
            "and ordinary injuries, bleeding, pain, flexibility, or visible marks"
        ),
    },
    {
        "id": OBJECT_SETTINGS_AND_ACCESSORIES,
        "min_words": 70,
        "max_words": 105,
        "focus": (
            "common places where the object is found, stored, rented, carried, "
            "displayed, repaired, bought, or used; nearby accessories, labels, lids, "
            "handles, covers, cushions, straps, cases, stands, containers, or spare "
            "parts; and ordinary size, portability, stability, comfort, or storage"
        ),
    },
)
CLEAN_MULTIDOC_STYLE_REGISTRY = {
    str(style["id"]): style
    for style in (*CLEAN_MULTIDOC_STYLES, *SPECIALIZED_CLEAN_MULTIDOC_STYLES)
}
DEFAULT_CLEAN_MULTIDOC_STYLE_IDS = tuple(
    str(style["id"]) for style in CLEAN_MULTIDOC_STYLES
)
CONCEPT_TYPE_STYLE_IDS = {
    "animal": (
        LIFE_CYCLE_AND_MOVEMENT,
        BODY_AND_SIGNALS,
        ANIMAL_HABITATS_VARIATION_INTERACTIONS,
    ),
    "bird": (
        LIFE_CYCLE_AND_MOVEMENT,
        BODY_AND_SIGNALS,
        ANIMAL_HABITATS_VARIATION_INTERACTIONS,
    ),
    "fish": (
        LIFE_CYCLE_AND_MOVEMENT,
        BODY_AND_SIGNALS,
        ANIMAL_HABITATS_VARIATION_INTERACTIONS,
    ),
    "insect": (
        LIFE_CYCLE_AND_MOVEMENT,
        BODY_AND_SIGNALS,
        ANIMAL_HABITATS_VARIATION_INTERACTIONS,
    ),
    "mammal": (
        LIFE_CYCLE_AND_MOVEMENT,
        BODY_AND_SIGNALS,
        ANIMAL_HABITATS_VARIATION_INTERACTIONS,
    ),
    "reptile": (
        LIFE_CYCLE_AND_MOVEMENT,
        BODY_AND_SIGNALS,
        ANIMAL_HABITATS_VARIATION_INTERACTIONS,
    ),
    "plant": (ECOLOGY_AND_GROWTH,),
    "tree": (ECOLOGY_AND_GROWTH,),
    "flower": (ECOLOGY_AND_GROWTH,),
    "food": (INGREDIENTS_AND_SERVING, FOOD_PAIRINGS_FORMS_PACKAGING),
    "fruit": (ECOLOGY_AND_GROWTH, INGREDIENTS_AND_SERVING, FOOD_PAIRINGS_FORMS_PACKAGING),
    "vegetable": (ECOLOGY_AND_GROWTH, INGREDIENTS_AND_SERVING, FOOD_PAIRINGS_FORMS_PACKAGING),
    "dessert": (INGREDIENTS_AND_SERVING, FOOD_PAIRINGS_FORMS_PACKAGING),
    "condiment": (INGREDIENTS_AND_SERVING, FOOD_PAIRINGS_FORMS_PACKAGING),
    "drink": (INGREDIENTS_AND_SERVING, FOOD_PAIRINGS_FORMS_PACKAGING),
    "beverage": (INGREDIENTS_AND_SERVING, FOOD_PAIRINGS_FORMS_PACKAGING),
    "ingredient": (FORMS_AND_USES,),
    "material": (FORMS_AND_USES,),
    "substance": (FORMS_AND_USES,),
    "human_made_object": (OPERATION_AND_COMPONENTS, OBJECT_SETTINGS_AND_ACCESSORIES),
    "object": (OPERATION_AND_COMPONENTS, OBJECT_SETTINGS_AND_ACCESSORIES),
    "device": (OPERATION_AND_COMPONENTS, OBJECT_SETTINGS_AND_ACCESSORIES),
    "furniture": (OPERATION_AND_COMPONENTS, OBJECT_SETTINGS_AND_ACCESSORIES),
    "tool": (OPERATION_AND_COMPONENTS, OBJECT_SETTINGS_AND_ACCESSORIES),
    "vehicle": (OPERATION_AND_COMPONENTS, OBJECT_SETTINGS_AND_ACCESSORIES),
    "clothing": (OPERATION_AND_COMPONENTS, OBJECT_SETTINGS_AND_ACCESSORIES),
    "kitchen_utensil": (OPERATION_AND_COMPONENTS, OBJECT_SETTINGS_AND_ACCESSORIES),
    "musical_instrument": (OPERATION_AND_COMPONENTS, OBJECT_SETTINGS_AND_ACCESSORIES),
    "weapon": (OPERATION_AND_COMPONENTS, OBJECT_SETTINGS_AND_ACCESSORIES),
    "toy": (
        OPERATION_AND_COMPONENTS,
        PLAY_OBJECT_GAMES_AND_VARIANTS,
        OBJECT_SETTINGS_AND_ACCESSORIES,
    ),
    "play_object": (
        OPERATION_AND_COMPONENTS,
        PLAY_OBJECT_GAMES_AND_VARIANTS,
        OBJECT_SETTINGS_AND_ACCESSORIES,
    ),
    "game": (
        OPERATION_AND_COMPONENTS,
        PLAY_OBJECT_GAMES_AND_VARIANTS,
        OBJECT_SETTINGS_AND_ACCESSORIES,
    ),
    "sports_equipment": (
        OPERATION_AND_COMPONENTS,
        PLAY_OBJECT_GAMES_AND_VARIANTS,
        OBJECT_SETTINGS_AND_ACCESSORIES,
    ),
    "body_part": (BODY_PART_ACTIONS_INJURY_SENSATION,),
    "organ": (BODY_PART_ACTIONS_INJURY_SENSATION,),
    "anatomy": (BODY_PART_ACTIONS_INJURY_SENSATION,),
}
ALLOWED_CONCEPT_TYPES = tuple(sorted(CONCEPT_TYPE_STYLE_IDS))


@dataclass(frozen=True)
class CompsConceptPlan:
    id: str
    index: int
    concept: str
    contrast_set: tuple[str, ...]
    source: str = "manual"
    prompt_styles: tuple[str, ...] = ()
    concept_types: tuple[str, ...] = ()
    concept_type: str = ""
    sense_hint: str = ""
    avoid_senses: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompsConceptPropertiesResult:
    name: str
    output_dir: Path
    prompts: tuple[dict[str, Any], ...]
    documents: tuple[dict[str, Any], ...]
    audit_report: dict[str, Any] | None
    manifest: dict[str, Any]
    outputs: dict[str, Path]


def load_comps_concept_properties_config(path: str | Path) -> dict[str, Any]:
    """Load a COMPS concept-properties YAML or JSON config."""

    config_path = Path(path)
    raw = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        config = json.loads(raw)
    else:
        import yaml

        config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError(f"Expected object config at {config_path}")
    return config


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "concept"


def word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def normalize_document_text(text: str) -> str:
    """Normalize a model completion into one trainable document line."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|text)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        stripped = payload["text"].strip()

    return re.sub(r"\s+", " ", stripped).strip()


def _string_list(value: Any, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of strings")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    return result


def normalize_concept_type(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return normalized


def normalize_concept_types(*concept_type_groups: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Normalize, validate, and de-duplicate concept type labels in order."""

    normalized_types: list[str] = []
    seen: set[str] = set()
    for group in concept_type_groups:
        raw_values: tuple[str, ...]
        if isinstance(group, str):
            raw_values = (group,)
        else:
            raw_values = tuple(str(item) for item in group)
        for raw_value in raw_values:
            concept_type = normalize_concept_type(raw_value)
            if not concept_type or concept_type in seen:
                continue
            resolve_concept_type_style_ids(concept_type)
            normalized_types.append(concept_type)
            seen.add(concept_type)
    return tuple(normalized_types)


def merge_style_ids(*style_id_groups: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in style_id_groups:
        for raw_style_id in group:
            style_id = str(raw_style_id).strip()
            if style_id and style_id not in seen:
                merged.append(style_id)
                seen.add(style_id)
    return tuple(merged)


def resolve_concept_type_style_ids(concept_type: str) -> tuple[str, ...]:
    if not concept_type:
        return ()
    if concept_type not in CONCEPT_TYPE_STYLE_IDS:
        raise ValueError(
            "unknown concept_type "
            f"{concept_type!r}; expected one of {sorted(CONCEPT_TYPE_STYLE_IDS)}"
        )
    return CONCEPT_TYPE_STYLE_IDS[concept_type]


def resolve_concept_types_style_ids(concept_types: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Resolve and merge all specialized style IDs for a concept's type labels."""

    style_groups = tuple(
        resolve_concept_type_style_ids(concept_type)
        for concept_type in normalize_concept_types(tuple(concept_types))
    )
    return merge_style_ids(*style_groups)


def display_concept_name(concept_id: str) -> str:
    """Convert a COMPS concept ID into a prose concept name."""

    return str(concept_id).replace("_", " ")


def load_comps_acceptable_concepts(path: str | Path) -> tuple[str, ...]:
    """Load sorted unique acceptable concept IDs from a COMPS JSONL file."""

    concepts: set[str] = set()
    source_path = Path(path)
    for line_number, raw_line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        concept = str(row.get("acceptable_concept", "")).strip()
        if not concept:
            raise ValueError(f"{source_path}:{line_number} is missing acceptable_concept")
        concepts.add(concept)
    if not concepts:
        raise ValueError(f"No acceptable concepts found in {source_path}")
    return tuple(sorted(concepts))


def load_concept_type_annotations(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Load concept type annotations from a plain map or metadata wrapper."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("concept type annotations must be a JSON object")
    raw_mapping = payload.get("concept_types", payload)
    if not isinstance(raw_mapping, dict):
        raise ValueError("concept type annotations must include an object concept_types field")
    annotations: dict[str, tuple[str, ...]] = {}
    for raw_concept, raw_types in raw_mapping.items():
        concept = str(raw_concept).strip()
        if not concept:
            raise ValueError("concept type annotation includes an empty concept key")
        annotations[concept] = normalize_concept_types(
            _string_list(raw_types, field=f"concept_types[{concept!r}]")
        )
    return annotations


def _normalize_sense_override(raw_value: Any, *, concept: str) -> dict[str, Any]:
    if isinstance(raw_value, str):
        sense_hint = raw_value.strip()
        avoid_senses: tuple[str, ...] = ()
    elif isinstance(raw_value, dict):
        sense_hint = str(
            raw_value.get("sense_hint", raw_value.get("generation_sense", ""))
        ).strip()
        avoid_senses = _string_list(
            raw_value.get("avoid_senses", raw_value.get("excluded_senses")),
            field=f"sense_overrides[{concept!r}].avoid_senses",
        )
    else:
        raise ValueError(
            f"sense_overrides[{concept!r}] must be a string or object"
        )
    if not sense_hint:
        raise ValueError(f"sense_overrides[{concept!r}] is missing sense_hint")
    result: dict[str, Any] = {"sense_hint": sense_hint}
    if avoid_senses:
        result["avoid_senses"] = list(avoid_senses)
    return result


def load_concept_sense_overrides(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load optional word-sense hints for ambiguous concept names."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("concept sense overrides must be a JSON object")
    raw_mapping = payload.get("sense_overrides", payload)
    if not isinstance(raw_mapping, dict):
        raise ValueError("concept sense overrides must include an object sense_overrides field")
    overrides: dict[str, dict[str, Any]] = {}
    for raw_concept, raw_override in raw_mapping.items():
        concept = str(raw_concept).strip()
        if not concept:
            raise ValueError("concept sense override includes an empty concept key")
        overrides[concept] = _normalize_sense_override(raw_override, concept=concept)
    return overrides


def build_full_comps_concept_properties_config(
    *,
    concepts: tuple[str, ...] | list[str],
    concept_type_annotations: dict[str, tuple[str, ...] | list[str]],
    concept_sense_overrides: dict[str, dict[str, Any]] | None = None,
    name: str,
    output_dir: str,
    renderer_type: str = FIREWORKS,
    model_id: str = DEFAULT_MODEL_ID,
    api_key_file: str = "fireworks_api.txt",
) -> dict[str, Any]:
    """Build a full clean-multidoc config from concept IDs and type annotations."""

    concept_rows: list[dict[str, Any]] = []
    sense_overrides = concept_sense_overrides or {}
    for concept_id in concepts:
        raw_concept_id = str(concept_id).strip()
        concept_types = normalize_concept_types(
            tuple(concept_type_annotations.get(raw_concept_id, ()))
        )
        row: dict[str, Any] = {
            "id": raw_concept_id,
            "concept": display_concept_name(raw_concept_id),
            "concept_types": list(concept_types),
        }
        if raw_concept_id in sense_overrides:
            row.update(_normalize_sense_override(
                sense_overrides[raw_concept_id],
                concept=raw_concept_id,
            ))
        concept_rows.append(row)
    return {
        "name": str(name),
        "prompt_variant": CLEAN_MULTIDOC_NO_NEIGHBORS,
        "output_dir": str(output_dir),
        "renderer": {
            "type": str(renderer_type),
            "model_id": str(model_id),
            "api_key_file": str(api_key_file),
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": 190,
            "reasoning_effort": "none",
            "reasoning_history": "disabled",
            "timeout_seconds": 600,
            "max_request_retries": 4,
        },
        "concepts": concept_rows,
    }


def build_comps_concept_plans(
    config: dict[str, Any],
    *,
    concept_limit: int | None = None,
) -> tuple[CompsConceptPlan, ...]:
    """Build concept plans from a config-owned concept inventory."""

    raw_concepts = config.get("concepts")
    if not isinstance(raw_concepts, list) or not raw_concepts:
        raise ValueError("COMPS concept-properties config must include a non-empty concepts list")

    plans: list[CompsConceptPlan] = []
    for index, raw in enumerate(raw_concepts):
        if isinstance(raw, str):
            concept = raw.strip()
            plan_id = slugify(concept)
            contrast_set: tuple[str, ...] = ()
            source = "manual"
            prompt_styles: tuple[str, ...] = ()
            concept_types: tuple[str, ...] = ()
            concept_type = ""
            sense_hint = ""
            avoid_senses: tuple[str, ...] = ()
        elif isinstance(raw, dict):
            concept = str(raw.get("concept", "")).strip()
            plan_id = str(raw.get("id") or slugify(concept)).strip()
            contrast_set = _string_list(
                raw.get("contrast_set", raw.get("contrasts")),
                field=f"concepts[{index}].contrast_set",
            )
            source = str(raw.get("source", "manual")).strip() or "manual"
            prompt_styles = _string_list(
                raw.get("prompt_styles", raw.get("style_ids")),
                field=f"concepts[{index}].prompt_styles",
            )
            singular_concept_type = normalize_concept_type(raw.get("concept_type", ""))
            plural_concept_types = _string_list(
                raw.get("concept_types"),
                field=f"concepts[{index}].concept_types",
            )
            concept_types = normalize_concept_types(
                singular_concept_type,
                plural_concept_types,
            )
            concept_type = concept_types[0] if concept_types else ""
            sense_hint = str(
                raw.get("sense_hint", raw.get("generation_sense", ""))
            ).strip()
            avoid_senses = _string_list(
                raw.get("avoid_senses", raw.get("excluded_senses")),
                field=f"concepts[{index}].avoid_senses",
            )
        else:
            raise ValueError(f"concepts[{index}] must be a string or object")

        if not concept:
            raise ValueError(f"concepts[{index}] is missing a concept string")
        if not plan_id:
            raise ValueError(f"concepts[{index}] has an empty id")
        plans.append(
            CompsConceptPlan(
                id=plan_id,
                index=index,
                concept=concept,
                contrast_set=contrast_set,
                source=source,
                prompt_styles=prompt_styles,
                concept_types=concept_types,
                concept_type=concept_type,
                sense_hint=sense_hint,
                avoid_senses=avoid_senses,
            )
        )

    if concept_limit is not None:
        plans = plans[: max(0, int(concept_limit))]
    return tuple(plans)


def build_sense_guidance(plan: CompsConceptPlan) -> str:
    """Return optional disambiguation text for prompts."""

    guidance = []
    if plan.sense_hint:
        guidance.append(
            f'Use this intended sense of "{plan.concept}": {plan.sense_hint}.'
        )
    if plan.avoid_senses:
        guidance.append(
            "Avoid shifting into these other meanings: "
            f"{', '.join(plan.avoid_senses)}."
        )
    if not guidance:
        return ""
    return (
        "\n\n"
        + " ".join(guidance)
        + " Stay with the intended sense throughout the prose, and do not discuss alternate word meanings."
    )


def build_comps_concept_prompt(
    plan: CompsConceptPlan,
    *,
    target_words: int = DEFAULT_TARGET_WORDS,
) -> str:
    """Build the Qwen prompt for one concept characterization document."""

    contrast_text = ", ".join(plan.contrast_set) if plan.contrast_set else "none provided"
    return f"""You are generating synthetic pretraining text for a small language model. Write a natural, wiki-style characterization document for the concept: **{plan.concept}**.
{build_sense_guidance(plan)}

The goal is not to list every true fact. The goal is to describe the concept using a salience-weighted profile: emphasize facts that are common, typical, diagnostic, and useful for distinguishing this concept from nearby concepts.

Before writing, internally decide which properties are high-salience, medium-salience, and low-salience.

High-salience facts should be frequent, central, and diagnostic. These include category, typical appearance, important parts, characteristic behaviors, common locations, common sounds, biological needs for living things, and human purpose for human-made objects.

Medium-salience facts should include useful but less central information: state changes, common roles, common human interactions, typical causes and effects, and nearby contrasts.

Low-salience facts should appear rarely or only briefly. These include generic affordances that are true but not distinctive, such as "can walk," "has eyes," "can be touched," or "can be moved," unless they are important for comparison.

Write the final output as natural prose, not as a bullet list. Use simple, clear sentences suitable for pretraining a small language model. The document should feel like a compact encyclopedia entry.

Include:

* what the concept is
* what category it belongs to
* important parts or physical properties
* typical behaviors or functions
* common locations or contexts
* important relations to people, other entities, or environments
* common state changes
* characteristic sounds, motions, or affordances when relevant
* contrasts with nearby concepts
* "usually," "often," "can," "may," and "not always" language where appropriate

For human-made objects, explain the human purpose of the object.
For animals or plants, explain biological needs, behaviors, habitats, reproduction, and human associations without saying the animal has a human-made purpose.
For foods, explain ingredients, parts, preparation, edibility, taste/texture, and common uses.
For social or cultural objects, explain physical form, social function, symbolic meaning, and context.

Avoid rare trivia, overly technical details, and facts that are true but not characteristic. Do not overrepresent generic facts. For example, if the concept is "beaver," mention that beavers can walk only briefly, while emphasizing swimming, flat tails, gnawing wood, dams, lodges, and water habitats.

Use nearby contrasts: **{contrast_text}**.

Write about {int(target_words)} words. This is a target length, not a hard maximum. Do not pad the document if the concept is naturally covered.

Mention the concept name "{plan.concept}" naturally several times, especially near important properties. Prefer repeating the noun over using only pronouns or generic category words. Do not keyword-stuff; the prose should still sound natural.

End with a final paragraph that summarizes the concept's most characteristic fingerprint in one or two sentences.

Return only the final prose document. Do not include headings, bullet lists, markdown labels, planning notes, or any mention that this text is synthetic pretraining data."""


def build_clean_multidoc_prompt(
    plan: CompsConceptPlan,
    *,
    style: dict[str, Any],
) -> str:
    """Build one clean, concept-only Qwen prompt for a short document."""

    style_id = str(style["id"])
    min_words = int(style.get("min_words", 60))
    max_words = int(style.get("max_words", 90))
    focus = str(style["focus"])
    contrast_guidance = ""
    if style_id == "natural_contrasts":
        contrast_guidance = (
            "\nIf a comparison is useful, use one familiar nearby concept from "
            "ordinary world knowledge and keep the comparison brief and natural."
        )
    return f"""Write one short, natural paragraph about the concept: **{plan.concept}**.
{build_sense_guidance(plan)}

The paragraph should read like ordinary educational corpus text. Focus on common, typical, diagnostic facts that would help a reader understand the concept.

Document focus: {focus}.{contrast_guidance}

Use simple, clear prose. Mention the concept name "{plan.concept}" naturally 3-5 times, especially near important properties. Prefer repeating the noun over using only pronouns or generic category words. Keep repetition natural.

Keep the paragraph about {min_words}-{max_words} words. Emphasize ordinary central facts over obscure trivia or highly specific facts.

Return only one paragraph of prose."""


def resolve_clean_multidoc_styles(style_ids: tuple[str, ...] | list[str]) -> tuple[dict[str, Any], ...]:
    styles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_style_id in style_ids:
        style_id = str(raw_style_id).strip()
        if not style_id or style_id in seen:
            continue
        if style_id not in CLEAN_MULTIDOC_STYLE_REGISTRY:
            raise ValueError(
                "unknown clean multidoc style "
                f"{style_id!r}; expected one of {sorted(CLEAN_MULTIDOC_STYLE_REGISTRY)}"
            )
        styles.append(CLEAN_MULTIDOC_STYLE_REGISTRY[style_id])
        seen.add(style_id)
    if not styles:
        raise ValueError("clean multidoc style list must not be empty")
    return tuple(styles)


def build_prompt_rows(
    plans: tuple[CompsConceptPlan, ...],
    *,
    target_words: int = DEFAULT_TARGET_WORDS,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for plan in plans:
        rows.append(
            {
                **plan.to_dict(),
                "concept_id": plan.id,
                "prompt_id": plan.id,
                "style_id": SINGLE_PROFILE,
                "target_words": int(target_words),
                "prompt": build_comps_concept_prompt(plan, target_words=target_words),
            }
        )
    return tuple(rows)


def build_clean_multidoc_prompt_rows(
    plans: tuple[CompsConceptPlan, ...],
    *,
    styles: tuple[dict[str, Any], ...] | None = None,
    style_ids: tuple[str, ...] = DEFAULT_CLEAN_MULTIDOC_STYLE_IDS,
) -> tuple[dict[str, Any], ...]:
    rows = []
    default_styles = styles or resolve_clean_multidoc_styles(style_ids)
    default_style_ids = tuple(str(style["id"]) for style in default_styles)
    for plan in plans:
        if plan.prompt_styles:
            plan_style_ids = plan.prompt_styles
        else:
            plan_style_ids = merge_style_ids(
                default_style_ids,
                resolve_concept_types_style_ids(plan.concept_types),
            )
        plan_styles = resolve_clean_multidoc_styles(plan_style_ids)
        for style in plan_styles:
            style_id = str(style["id"])
            target_words = int(
                (int(style.get("min_words", 60)) + int(style.get("max_words", 90))) / 2
            )
            prompt_id = f"{plan.id}__{style_id}"
            rows.append(
                {
                    **plan.to_dict(),
                    "id": prompt_id,
                    "concept_id": plan.id,
                    "prompt_id": prompt_id,
                    "style_id": style_id,
                    "target_words": target_words,
                    "prompt": build_clean_multidoc_prompt(plan, style=style),
                }
            )
    return tuple(rows)


def build_prompt_rows_from_config(
    config: dict[str, Any],
    plans: tuple[CompsConceptPlan, ...],
    *,
    target_words: int = DEFAULT_TARGET_WORDS,
) -> tuple[dict[str, Any], ...]:
    variant = str(config.get("prompt_variant", SINGLE_PROFILE))
    if variant == SINGLE_PROFILE:
        return build_prompt_rows(plans, target_words=target_words)
    if variant == CLEAN_MULTIDOC_NO_NEIGHBORS:
        configured_style_ids = _string_list(
            config.get("prompt_styles", config.get("clean_multidoc_style_ids")),
            field="prompt_styles",
        )
        return build_clean_multidoc_prompt_rows(
            plans,
            style_ids=configured_style_ids or DEFAULT_CLEAN_MULTIDOC_STYLE_IDS,
        )
    raise ValueError(
        f"prompt_variant must be one of {sorted({SINGLE_PROFILE, CLEAN_MULTIDOC_NO_NEIGHBORS})}"
    )


def render_template_document(
    plan: CompsConceptPlan,
    *,
    target_words: int = DEFAULT_TARGET_WORDS,
) -> str:
    """Render a local smoke-test document; Qwen should be used for real data."""

    contrasts = ", ".join(plan.contrast_set) if plan.contrast_set else "nearby concepts"
    text = (
        f"{plan.concept.capitalize()} is a concept chosen for a COMPS concept-property "
        f"characterization document. A real generated document should describe what "
        f"{plan.concept} is, its category, its common parts or physical properties, "
        f"and the behaviors, functions, locations, sounds, motions, and state changes "
        f"that are typical for it. It should use ordinary language, hedge with words "
        f"such as usually and often, and contrast {plan.concept} with {contrasts}. "
        f"The target document is about {int(target_words)} words, but the important "
        f"constraint is salience: central and diagnostic properties should appear "
        f"more strongly than generic properties."
    )
    return normalize_document_text(text)


def render_fireworks_document(
    *,
    plan: CompsConceptPlan,
    prompt: str,
    renderer_config: dict[str, Any],
    api_key: str,
) -> str:
    model_id = str(renderer_config.get("model_id", DEFAULT_MODEL_ID))
    client = FireworksChatClient(
        model=model_id,
        api_key=api_key,
        timeout_seconds=int(renderer_config.get("timeout_seconds", 600)),
        max_request_retries=int(renderer_config.get("max_request_retries", 4)),
        reasoning_effort=renderer_config.get("reasoning_effort", "none"),
        reasoning_history=renderer_config.get("reasoning_history", "disabled"),
    )
    system_prompt = str(
        renderer_config.get(
            "system_prompt",
            "You write clean, factual corpus documents. Return only the final prose text.",
        )
    )
    raw = client(
        prompt=f"{prompt}\n\n/no_think",
        system_prompt=system_prompt,
        temperature=float(renderer_config.get("temperature", DEFAULT_TEMPERATURE)),
        max_tokens=int(renderer_config.get("max_tokens", DEFAULT_MAX_TOKENS)),
        stage=f"comps_concept_properties:{plan.id}",
    )
    return normalize_document_text(raw)


def audit_rendered_documents(
    documents: tuple[dict[str, Any], ...],
    *,
    target_words: int,
) -> dict[str, Any]:
    rows = []
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for row in documents:
        warnings = []
        text = str(row.get("text", ""))
        count = word_count(text)
        if not text:
            status = "fail"
            warnings.append("empty_text")
        else:
            status = "pass"
            if count < max(20, int(target_words * 0.5)):
                status = "warn"
                warnings.append("short_document")
            if count > int(target_words * 1.8):
                status = "warn"
                warnings.append("long_document")
        counts[status] += 1
        rows.append(
            {
                "id": row["id"],
                "concept": row["concept"],
                "status": status,
                "word_count": count,
                "warnings": warnings,
            }
        )
    return {"counts": counts, "documents": rows}


def write_jsonl(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    write_json(tmp_path, payload)
    tmp_path.replace(path)


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def render_checkpoint_path(
    checkpoint_dir: Path,
    *,
    prompt_index: int,
    prompt_row: dict[str, Any],
) -> Path:
    prompt_id = str(prompt_row.get("prompt_id", prompt_row.get("id", "prompt")))
    return checkpoint_dir / f"{int(prompt_index):06d}__{slugify(prompt_id)}.json"


def build_rendered_document_row(
    *,
    plan: CompsConceptPlan,
    prompt_row: dict[str, Any],
    renderer: str,
    text: str,
) -> dict[str, Any]:
    return {
        **plan.to_dict(),
        "id": str(prompt_row["id"]),
        "concept_id": plan.id,
        "prompt_id": str(prompt_row["prompt_id"]),
        "style_id": str(prompt_row["style_id"]),
        "target_words": int(prompt_row.get("target_words", DEFAULT_TARGET_WORDS)),
        "renderer": renderer,
        "prompt_sha256": prompt_sha256(str(prompt_row["prompt"])),
        "text": text,
        "word_count": word_count(text),
    }


def load_render_checkpoint(
    path: Path,
    *,
    prompt_row: dict[str, Any],
    renderer: str,
) -> tuple[dict[str, Any] | None, str]:
    """Load a matching rendered-row checkpoint, returning a status string."""

    if not path.exists():
        return None, "missing"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"render checkpoint is not a JSON object: {path}")

    expected = {
        "id": str(prompt_row["id"]),
        "prompt_id": str(prompt_row["prompt_id"]),
        "concept_id": str(prompt_row["concept_id"]),
        "style_id": str(prompt_row["style_id"]),
        "renderer": str(renderer),
        "prompt_sha256": prompt_sha256(str(prompt_row["prompt"])),
    }
    for key, value in expected.items():
        if str(payload.get(key, "")) != value:
            return None, "stale"
    if not str(payload.get("text", "")).strip():
        return None, "empty"
    if int(payload.get("word_count", 0)) <= 0:
        payload["word_count"] = word_count(str(payload["text"]))
    return payload, "reused"


def build_comps_concept_properties(
    config: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    renderer_type: str | None = None,
    output_dir: str | Path | None = None,
    concept_limit: int | None = None,
    api_key: str | None = None,
    api_key_file: str | None = None,
    force_render: bool = False,
) -> CompsConceptPropertiesResult:
    """Generate prompts and, optionally, rendered COMPS concept-property documents."""

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    name = str(config.get("name", "comps_concept_properties")).strip()
    target_words = int(config.get("target_words_per_concept", DEFAULT_TARGET_WORDS))
    renderer_config = dict(config.get("renderer", {}))
    selected_renderer = str(renderer_type or renderer_config.get("type", PROMPT_ONLY))
    if selected_renderer not in RENDERER_TYPES:
        raise ValueError(f"renderer type must be one of {sorted(RENDERER_TYPES)}")

    out_dir_value = output_dir or config.get("output_dir") or f"data/concept_curriculum/{name}"
    out_dir = resolve_path(out_dir_value, base_dir=root)
    out_dir.mkdir(parents=True, exist_ok=True)

    plans = build_comps_concept_plans(config, concept_limit=concept_limit)
    prompts = build_prompt_rows_from_config(config, plans, target_words=target_words)
    prompts_path = out_dir / "prompts.jsonl"
    write_jsonl(prompts_path, prompts)

    outputs: dict[str, Path] = {"prompts": prompts_path}
    documents: tuple[dict[str, Any], ...] = ()
    audit_report: dict[str, Any] | None = None
    checkpoint_summary: dict[str, Any] = {
        "enabled": selected_renderer != PROMPT_ONLY,
        "missing": 0,
        "reused": 0,
        "written": 0,
        "stale": 0,
        "empty": 0,
        "forced": 0,
    }

    if selected_renderer != PROMPT_ONLY:
        resolved_api_key = None
        if selected_renderer == FIREWORKS:
            resolved_api_key = load_api_key(api_key=api_key, api_key_file=api_key_file)

        checkpoint_dir = out_dir / str(renderer_config.get("checkpoint_dir", "render_checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_summary["dir"] = str(checkpoint_dir)
        outputs["render_checkpoints"] = checkpoint_dir

        rendered_rows: list[dict[str, Any]] = []
        plans_by_id = {plan.id: plan for plan in plans}
        progress = tqdm(
            total=len(prompts),
            desc="COMPS render",
            unit="doc",
            dynamic_ncols=True,
            disable=selected_renderer != FIREWORKS,
        )
        try:
            for prompt_index, prompt_row in enumerate(prompts):
                prompt_id = str(prompt_row["prompt_id"])
                concept_id = str(prompt_row.get("concept_id", prompt_row["id"]))
                plan = plans_by_id[concept_id]
                checkpoint_path = render_checkpoint_path(
                    checkpoint_dir,
                    prompt_index=prompt_index,
                    prompt_row=prompt_row,
                )
                checkpoint_row: dict[str, Any] | None = None
                checkpoint_status = "forced" if force_render and checkpoint_path.exists() else "missing"
                if not force_render:
                    checkpoint_row, checkpoint_status = load_render_checkpoint(
                        checkpoint_path,
                        prompt_row=prompt_row,
                        renderer=selected_renderer,
                    )
                if checkpoint_status in checkpoint_summary:
                    checkpoint_summary[checkpoint_status] += 1
                if checkpoint_row is not None:
                    rendered_rows.append(checkpoint_row)
                    progress.set_postfix_str(f"reused {prompt_id[:48]}", refresh=False)
                    progress.update(1)
                    continue

                if selected_renderer == TEMPLATE:
                    text = render_template_document(
                        plan,
                        target_words=int(prompt_row.get("target_words", target_words)),
                    )
                else:
                    assert resolved_api_key is not None
                    text = render_fireworks_document(
                        plan=plan,
                        prompt=str(prompt_row["prompt"]),
                        renderer_config=renderer_config,
                        api_key=resolved_api_key,
                    )
                document_row = build_rendered_document_row(
                    plan=plan,
                    prompt_row=prompt_row,
                    renderer=selected_renderer,
                    text=text,
                )
                atomic_write_json(checkpoint_path, document_row)
                checkpoint_summary["written"] += 1
                rendered_rows.append(document_row)
                progress.set_postfix_str(f"written {prompt_id[:48]}", refresh=False)
                progress.update(1)
        finally:
            progress.close()
        documents = tuple(rendered_rows)
        documents_path = out_dir / "documents.jsonl"
        train_path = out_dir / "train.txt"
        audit_path = out_dir / "audit_report.json"
        write_jsonl(documents_path, documents)
        train_path.write_text(
            "".join(f"{row['text'].strip()}\n" for row in documents if row["text"].strip()),
            encoding="utf-8",
        )
        audit_report = audit_rendered_documents(documents, target_words=target_words)
        audit_path.write_text(json.dumps(audit_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.update({"documents": documents_path, "train": train_path, "audit_report": audit_path})

    if config_path is not None:
        copied_config_path = out_dir / Path(config_path).name
        if Path(config_path).resolve() != copied_config_path.resolve():
            shutil.copyfile(config_path, copied_config_path)
        outputs["config"] = copied_config_path

    manifest = {
        "name": name,
        "task": "comps_concept_properties",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_variant": str(config.get("prompt_variant", SINGLE_PROFILE)),
        "renderer": selected_renderer,
        "model_id": renderer_config.get("model_id", DEFAULT_MODEL_ID),
        "target_words_per_concept": target_words,
        "render_checkpoints": checkpoint_summary,
        "counts": {
            "concepts": len(plans),
            "prompts": len(prompts),
            "documents": len(documents),
            "train_words": sum(int(row.get("word_count", 0)) for row in documents),
        },
        "concept_ids": [plan.id for plan in plans],
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs["manifest"] = manifest_path

    return CompsConceptPropertiesResult(
        name=name,
        output_dir=out_dir,
        prompts=prompts,
        documents=documents,
        audit_report=audit_report,
        manifest=manifest,
        outputs=outputs,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML/JSON concept-properties config.")
    parser.add_argument("--renderer", choices=sorted(RENDERER_TYPES), help="Override renderer type.")
    parser.add_argument("--output-dir", help="Override config output_dir.")
    parser.add_argument("--concept-limit", type=int, help="Render only the first N concepts.")
    parser.add_argument("--api-key", help="Fireworks API key for --renderer fireworks.")
    parser.add_argument("--api-key-file", help="Plain-text Fireworks API key file.")
    parser.add_argument(
        "--force-render",
        action="store_true",
        help="Ignore existing render checkpoints and regenerate rendered documents.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = load_comps_concept_properties_config(config_path)
    repo_root = Path(__file__).resolve().parents[3]
    result = build_comps_concept_properties(
        config,
        base_dir=repo_root,
        config_path=config_path,
        renderer_type=args.renderer,
        output_dir=args.output_dir,
        concept_limit=args.concept_limit,
        api_key=args.api_key,
        api_key_file=args.api_key_file,
        force_render=bool(args.force_render),
    )
    print(
        f"COMPS concept properties: {result.name} "
        f"renderer={result.manifest['renderer']} "
        f"prompts={len(result.prompts)} "
        f"documents={len(result.documents)} "
        f"output_dir={result.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
