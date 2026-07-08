from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from babysteps.concept_curriculum.comps_concept_properties import (
    ANIMAL_HABITATS_VARIATION_INTERACTIONS,
    BODY_AND_SIGNALS,
    BODY_PART_ACTIONS_INJURY_SENSATION,
    CLEAN_MULTIDOC_NO_NEIGHBORS,
    ECOLOGY_AND_GROWTH,
    FOOD_PAIRINGS_FORMS_PACKAGING,
    FORMS_AND_USES,
    INGREDIENTS_AND_SERVING,
    LIFE_CYCLE_AND_MOVEMENT,
    OBJECT_SETTINGS_AND_ACCESSORIES,
    OPERATION_AND_COMPONENTS,
    PLAY_OBJECT_GAMES_AND_VARIANTS,
    TEMPLATE,
    build_full_comps_concept_properties_config,
    build_clean_multidoc_prompt_rows,
    build_comps_concept_plans,
    build_comps_concept_properties,
    build_comps_concept_prompt,
    load_concept_sense_overrides,
    load_comps_acceptable_concepts,
    load_concept_type_annotations,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(relative_path: str, module_name: str):
    script_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _toy_config(tmp_path: Path, *, renderer: str = "prompt_only") -> dict:
    return {
        "name": "comps_concept_properties_test",
        "target_words_per_concept": 200,
        "output_dir": str(tmp_path / "comps_concept_properties_test"),
        "renderer": {"type": renderer},
        "concepts": [
            {
                "id": "sheep",
                "concept": "sheep",
                "contrast_set": ["goat", "cow", "lamb"],
            },
            {
                "id": "toaster",
                "concept": "toaster",
                "contrast_set": ["oven", "microwave", "blender"],
            },
        ],
    }


def _clean_toy_config(tmp_path: Path, *, renderer: str = "prompt_only") -> dict:
    return {
        "name": "comps_concept_properties_clean_test",
        "prompt_variant": CLEAN_MULTIDOC_NO_NEIGHBORS,
        "output_dir": str(tmp_path / "comps_concept_properties_clean_test"),
        "renderer": {"type": renderer},
        "concepts": [
            {"id": "sheep", "concept": "sheep"},
            {"id": "toaster", "concept": "toaster"},
        ],
    }


def test_comps_concept_prompt_keeps_concept_first_shape(tmp_path: Path) -> None:
    plans = build_comps_concept_plans(_toy_config(tmp_path))

    assert [plan.id for plan in plans] == ["sheep", "toaster"]
    prompt = build_comps_concept_prompt(plans[0], target_words=200)

    assert "concept: **sheep**" in prompt
    assert "Use nearby contrasts: **goat, cow, lamb**." in prompt
    assert "Write about 200 words." in prompt
    assert 'Mention the concept name "sheep" naturally several times' in prompt
    assert "Return only the final prose document." in prompt
    assert "property_phrase" not in prompt
    assert "prefix_acceptable" not in prompt


def test_clean_multidoc_prompts_avoid_supplied_neighbors(tmp_path: Path) -> None:
    plans = build_comps_concept_plans(_clean_toy_config(tmp_path))
    rows = build_clean_multidoc_prompt_rows(plans)

    assert len(rows) == 8
    assert rows[0]["id"] == "sheep__core_profile"
    assert rows[0]["concept_id"] == "sheep"
    assert {row["style_id"] for row in rows} == {
        "core_profile",
        "parts_and_properties",
        "contexts_and_interactions",
        "natural_contrasts",
    }
    natural_contrast_prompt = [
        row["prompt"] for row in rows if row["style_id"] == "natural_contrasts"
    ][0]
    assert "ordinary world knowledge" in natural_contrast_prompt
    assert "brief and natural" in natural_contrast_prompt
    assert "goat" not in natural_contrast_prompt
    assert "supplied contrast" not in natural_contrast_prompt
    assert "benchmark" not in natural_contrast_prompt
    assert "evaluation" not in natural_contrast_prompt
    assert "minimal" not in natural_contrast_prompt
    assert "foil" not in natural_contrast_prompt
    assert "paired" not in natural_contrast_prompt
    assert "true/false" not in natural_contrast_prompt
    assert "quiz" not in natural_contrast_prompt
    assert "property_phrase" not in natural_contrast_prompt
    assert "prefix_acceptable" not in natural_contrast_prompt


def test_clean_multidoc_supports_opt_in_specialized_styles(tmp_path: Path) -> None:
    config = {
        "name": "comps_concept_properties_specialized_test",
        "prompt_variant": CLEAN_MULTIDOC_NO_NEIGHBORS,
        "output_dir": str(tmp_path / "comps_concept_properties_specialized_test"),
        "renderer": {"type": "prompt_only"},
        "concepts": [
            {
                "id": "crocodile",
                "concept": "crocodile",
                "prompt_styles": ["core_profile", LIFE_CYCLE_AND_MOVEMENT, BODY_AND_SIGNALS],
            },
            {
                "id": "willow_tree",
                "concept": "willow tree",
                "prompt_styles": ["core_profile", ECOLOGY_AND_GROWTH],
            },
            {
                "id": "television",
                "concept": "television",
                "prompt_styles": ["core_profile", OPERATION_AND_COMPONENTS],
            },
            {
                "id": "ketchup",
                "concept": "ketchup",
                "prompt_styles": ["core_profile", INGREDIENTS_AND_SERVING],
            },
        ],
    }
    plans = build_comps_concept_plans(config)
    rows = build_clean_multidoc_prompt_rows(plans)

    assert [row["id"] for row in rows] == [
        "crocodile__core_profile",
        "crocodile__life_cycle_and_movement",
        "crocodile__body_and_signals",
        "willow_tree__core_profile",
        "willow_tree__ecology_and_growth",
        "television__core_profile",
        "television__operation_and_components",
        "ketchup__core_profile",
        "ketchup__ingredients_and_serving",
    ]
    prompts = {row["id"]: row["prompt"] for row in rows}
    assert "how the animal moves, grows, reproduces" in prompts[
        "crocodile__life_cycle_and_movement"
    ]
    assert "body parts, body covering, shape, limbs" in prompts[
        "crocodile__body_and_signals"
    ]
    assert "sounds, displays, and social signals" in prompts["crocodile__body_and_signals"]
    assert "how the plant grows, changes through seasons" in prompts[
        "willow_tree__ecology_and_growth"
    ]
    assert "how the human-made object or device works" in prompts[
        "television__operation_and_components"
    ]
    assert "what powers or activates it" in prompts["television__operation_and_components"]
    assert "what the food is made from" in prompts["ketchup__ingredients_and_serving"]
    assert "how people serve or eat it" in prompts["ketchup__ingredients_and_serving"]
    assert "benchmark" not in prompts["television__operation_and_components"]
    assert "property_phrase" not in prompts["television__operation_and_components"]
    assert "benchmark" not in prompts["ketchup__ingredients_and_serving"]
    assert "property_phrase" not in prompts["ketchup__ingredients_and_serving"]


def test_clean_multidoc_concept_type_adds_specialized_styles(tmp_path: Path) -> None:
    config = {
        "name": "comps_concept_properties_concept_type_test",
        "prompt_variant": CLEAN_MULTIDOC_NO_NEIGHBORS,
        "output_dir": str(tmp_path / "comps_concept_properties_concept_type_test"),
        "renderer": {"type": "prompt_only"},
        "concepts": [
            {"id": "sugar", "concept": "sugar", "concept_type": "substance"},
            {"id": "parakeet", "concept": "parakeet", "concept_type": "bird"},
            {"id": "corn", "concept": "corn", "concept_type": "food"},
            {"id": "kiwi", "concept": "kiwi", "concept_type": "fruit"},
            {"id": "carrot", "concept": "carrot", "concept_type": "vegetable"},
            {"id": "ice_cream", "concept": "ice cream", "concept_type": "dessert"},
            {"id": "ball", "concept": "ball", "concept_type": "toy"},
            {"id": "tongue", "concept": "tongue", "concept_type": "body_part"},
            {"id": "deckchair", "concept": "deckchair", "concept_type": "furniture"},
            {"id": "bikini", "concept": "bikini", "concept_type": "clothing"},
            {"id": "clarinet", "concept": "clarinet", "concept_type": "musical_instrument"},
            {"id": "machete", "concept": "machete", "concept_type": "weapon"},
            {"id": "ladle", "concept": "ladle", "concept_type": "kitchen_utensil"},
            {
                "id": "ketchup",
                "concept": "ketchup",
                "concept_type": "condiment",
                "prompt_styles": ["core_profile", INGREDIENTS_AND_SERVING],
            },
        ],
    }
    plans = build_comps_concept_plans(config)
    rows = build_clean_multidoc_prompt_rows(plans)
    row_ids = [row["id"] for row in rows]

    assert plans[0].concept_type == "substance"
    assert plans[0].concept_types == ("substance",)
    assert f"sugar__{FORMS_AND_USES}" in row_ids
    assert "parakeet__life_cycle_and_movement" in row_ids
    assert "parakeet__body_and_signals" in row_ids
    assert f"parakeet__{ANIMAL_HABITATS_VARIATION_INTERACTIONS}" in row_ids
    assert f"corn__{FOOD_PAIRINGS_FORMS_PACKAGING}" in row_ids
    assert "kiwi__ecology_and_growth" in row_ids
    assert "kiwi__ingredients_and_serving" in row_ids
    assert f"kiwi__{FOOD_PAIRINGS_FORMS_PACKAGING}" in row_ids
    assert "carrot__ecology_and_growth" in row_ids
    assert "carrot__ingredients_and_serving" in row_ids
    assert "ice_cream__ingredients_and_serving" in row_ids
    assert f"ice_cream__{FOOD_PAIRINGS_FORMS_PACKAGING}" in row_ids
    assert f"ball__{PLAY_OBJECT_GAMES_AND_VARIANTS}" in row_ids
    assert f"ball__{OBJECT_SETTINGS_AND_ACCESSORIES}" in row_ids
    assert f"tongue__{BODY_PART_ACTIONS_INJURY_SENSATION}" in row_ids
    assert f"deckchair__{OBJECT_SETTINGS_AND_ACCESSORIES}" in row_ids
    assert "bikini__operation_and_components" in row_ids
    assert f"bikini__{OBJECT_SETTINGS_AND_ACCESSORIES}" in row_ids
    assert "clarinet__operation_and_components" in row_ids
    assert f"clarinet__{OBJECT_SETTINGS_AND_ACCESSORIES}" in row_ids
    assert "machete__operation_and_components" in row_ids
    assert f"machete__{OBJECT_SETTINGS_AND_ACCESSORIES}" in row_ids
    assert "ladle__operation_and_components" in row_ids
    assert f"ladle__{OBJECT_SETTINGS_AND_ACCESSORIES}" in row_ids
    assert "ketchup__core_profile" in row_ids
    assert "ketchup__ingredients_and_serving" in row_ids
    assert "ketchup__parts_and_properties" not in row_ids
    assert "ketchup__natural_contrasts" not in row_ids

    prompts = {row["id"]: row["prompt"] for row in rows}
    assert "substance's common forms, texture, source, packaging" in prompts[
        f"sugar__{FORMS_AND_USES}"
    ]
    assert "mixed, heated, dried, dissolved, or spilled" in prompts[
        f"sugar__{FORMS_AND_USES}"
    ]
    assert "where the animal commonly lives or is kept" in prompts[
        f"parakeet__{ANIMAL_HABITATS_VARIATION_INTERACTIONS}"
    ]
    assert "ordinary body variation" in prompts[
        f"parakeet__{ANIMAL_HABITATS_VARIATION_INTERACTIONS}"
    ]
    assert "served, paired with other foods, bought" in prompts[
        f"corn__{FOOD_PAIRINGS_FORMS_PACKAGING}"
    ]
    assert "playgrounds, beaches, homes, schools" in prompts[
        f"ball__{PLAY_OBJECT_GAMES_AND_VARIANTS}"
    ]
    assert "ordinary injuries, bleeding, pain, flexibility" in prompts[
        f"tongue__{BODY_PART_ACTIONS_INJURY_SENSATION}"
    ]
    assert "nearby accessories, labels, lids" in prompts[
        f"deckchair__{OBJECT_SETTINGS_AND_ACCESSORIES}"
    ]
    assert "benchmark" not in prompts[f"sugar__{FORMS_AND_USES}"]
    assert "property_phrase" not in prompts[f"sugar__{FORMS_AND_USES}"]
    assert "benchmark" not in prompts[f"corn__{FOOD_PAIRINGS_FORMS_PACKAGING}"]
    assert "property_phrase" not in prompts[f"ball__{PLAY_OBJECT_GAMES_AND_VARIANTS}"]


def test_clean_multidoc_concept_types_merge_mixed_roles(tmp_path: Path) -> None:
    config = {
        "name": "comps_concept_properties_concept_types_test",
        "prompt_variant": CLEAN_MULTIDOC_NO_NEIGHBORS,
        "output_dir": str(tmp_path / "comps_concept_properties_concept_types_test"),
        "renderer": {"type": "prompt_only"},
        "concepts": [
            {
                "id": "salmon",
                "concept": "salmon",
                "concept_type": "fish",
                "concept_types": ["fish", "food"],
            },
            {
                "id": "cotton",
                "concept": "cotton",
                "concept_types": ["plant", "material"],
            },
        ],
    }
    plans = build_comps_concept_plans(config)
    rows = build_clean_multidoc_prompt_rows(plans)
    row_ids = [row["id"] for row in rows]

    assert plans[0].concept_type == "fish"
    assert plans[0].concept_types == ("fish", "food")
    assert "salmon__life_cycle_and_movement" in row_ids
    assert "salmon__body_and_signals" in row_ids
    assert f"salmon__{ANIMAL_HABITATS_VARIATION_INTERACTIONS}" in row_ids
    assert "salmon__ingredients_and_serving" in row_ids
    assert f"salmon__{FOOD_PAIRINGS_FORMS_PACKAGING}" in row_ids
    assert row_ids.count("salmon__life_cycle_and_movement") == 1
    assert "cotton__ecology_and_growth" in row_ids
    assert f"cotton__{FORMS_AND_USES}" in row_ids


def test_sense_hint_disambiguates_clean_multidoc_prompt(tmp_path: Path) -> None:
    config = {
        "name": "comps_concept_properties_sense_hint_test",
        "prompt_variant": CLEAN_MULTIDOC_NO_NEIGHBORS,
        "output_dir": str(tmp_path / "comps_concept_properties_sense_hint_test"),
        "renderer": {"type": "prompt_only"},
        "concepts": [
            {
                "id": "coach",
                "concept": "coach",
                "concept_types": ["vehicle"],
                "sense_hint": "a coach, meaning a long passenger bus",
                "avoid_senses": ["a person who trains athletes"],
            },
        ],
    }

    plans = build_comps_concept_plans(config)
    rows = build_clean_multidoc_prompt_rows(plans)

    assert plans[0].sense_hint == "a coach, meaning a long passenger bus"
    assert plans[0].avoid_senses == ("a person who trains athletes",)
    assert rows[0]["sense_hint"] == "a coach, meaning a long passenger bus"
    assert 'Use this intended sense of "coach": a coach, meaning a long passenger bus.' in rows[0]["prompt"]
    assert "Avoid shifting into these other meanings: a person who trains athletes." in rows[0]["prompt"]
    assert "do not discuss alternate word meanings" in rows[0]["prompt"]
    assert "benchmark" not in rows[0]["prompt"]
    assert "property_phrase" not in rows[0]["prompt"]


def test_clean_multidoc_unknown_concept_type_rejected(tmp_path: Path) -> None:
    config = {
        "name": "comps_concept_properties_bad_type_test",
        "prompt_variant": CLEAN_MULTIDOC_NO_NEIGHBORS,
        "output_dir": str(tmp_path / "comps_concept_properties_bad_type_test"),
        "renderer": {"type": "prompt_only"},
        "concepts": [
            {"id": "salmon", "concept": "salmon", "concept_types": ["fish", "magic"]},
        ],
    }

    with pytest.raises(ValueError, match="unknown concept_type"):
        build_comps_concept_plans(config)


def test_load_comps_acceptable_concepts_from_jsonl(tmp_path: Path) -> None:
    comps_path = tmp_path / "comps.jsonl"
    comps_path.write_text(
        "\n".join(
            [
                json.dumps({"acceptable_concept": "salmon"}),
                json.dumps({"acceptable_concept": "jar"}),
                json.dumps({"acceptable_concept": "salmon"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_comps_acceptable_concepts(comps_path) == ("jar", "salmon")


def test_load_concept_sense_overrides(tmp_path: Path) -> None:
    overrides_path = tmp_path / "sense_overrides.json"
    overrides_path.write_text(
        json.dumps(
            {
                "sense_overrides": {
                    "coach": {
                        "sense_hint": "a coach, meaning a long passenger bus",
                        "avoid_senses": ["a person who trains athletes"],
                    },
                    "bat": "a bat, meaning the flying mammal",
                }
            }
        ),
        encoding="utf-8",
    )

    overrides = load_concept_sense_overrides(overrides_path)

    assert overrides["coach"] == {
        "sense_hint": "a coach, meaning a long passenger bus",
        "avoid_senses": ["a person who trains athletes"],
    }
    assert overrides["bat"] == {"sense_hint": "a bat, meaning the flying mammal"}


def test_less_repetitive_postprocess_prompt_and_parser() -> None:
    module = _load_script_module(
        "scripts/synthetic_data_generation/concept_curriculum/postprocess_comps_concept_properties.py",
        "postprocess_comps_concept_properties_prompt_test",
    )
    rows = [
        {
            "id": "monkey__core_profile",
            "concept_id": "monkey",
            "concept": "monkey",
            "style_id": "core_profile",
            "text": "A monkey is a primate. A monkey often has a tail.",
            "word_count": 10,
        },
        {
            "id": "monkey__parts_and_properties",
            "concept_id": "monkey",
            "concept": "monkey",
            "style_id": "parts_and_properties",
            "text": "A monkey is covered in fur and uses hands and feet to grasp.",
            "word_count": 13,
        },
    ]

    prompt = module.build_less_repetitive_prompt(
        concept_id="monkey",
        concept="monkey",
        rows=rows,
    )
    paragraphs = module.parse_revised_paragraphs(
        "Paragraph 1: A monkey is a primate, and a monkey often has a monkey tail.\n\n"
        "Paragraph 2: A monkey uses monkey fur, hands, and feet as a monkey grasps branches.",
        expected_count=2,
    )

    assert "Keep exactly 2 paragraphs" in prompt
    assert "Do not merge, split, summarize, or reorder paragraphs" in prompt
    assert "Preserve every distinct factual detail" in prompt
    assert "Do not shorten a paragraph just to make it smoother" in prompt
    assert "should not become a compressed summary" in prompt
    assert 'Every output paragraph must include the exact concept name "monkey" at least 3 times' in prompt
    assert 'Plural or inflected forms may appear naturally, but they do not replace the required exact "monkey" mentions' in prompt
    assert "usually ideal" not in prompt
    assert "co-occur repeatedly with varied characteristic properties" in prompt
    assert "Keep \"monkey\" frequent, but make the surrounding information diverse" in prompt
    assert "Avoid repeating the same category phrase or definition across paragraphs" in prompt
    assert "preserve the more concrete details from both" in prompt
    assert "keep any meaningful modifiers inside it" in prompt
    assert "After the broad category has been stated once" in prompt
    assert "rewrite that space around more specific input facts" in prompt
    assert "Do not drop concrete details" in prompt
    assert "Prefer varied starts like a property, location, action, contrast, material, or use" in prompt
    assert "avoid starting every paragraph with the same definition" in prompt
    assert "benchmark" not in prompt
    assert "property_phrase" not in prompt
    assert paragraphs == (
        "A monkey is a primate, and a monkey often has a monkey tail.",
        "A monkey uses monkey fur, hands, and feet as a monkey grasps branches.",
    )
    with pytest.raises(module.PostprocessError, match="at least 3 times"):
        module.validate_paragraph_concept_mentions(
            ("A monkey is a primate, and a monkey has a tail.",),
            concept="monkey",
        )
    with pytest.raises(module.PostprocessError, match="expected 2 paragraphs"):
        module.parse_revised_paragraphs("Only one paragraph.", expected_count=2)

    retry_prompt = module.build_retry_prompt(
        prompt=prompt,
        concept="monkey",
        error=module.PostprocessError("paragraph 2 had too few monkey mentions"),
        min_concept_mentions=3,
    )
    assert "The previous revision failed validation" in retry_prompt
    assert 'exact string "monkey" at least 3 times' in retry_prompt
    assert "Return the full revised paragraph set again" in retry_prompt
    assert retry_prompt.endswith("/no_think")

    reasoning_prompt = module.build_less_repetitive_prompt(
        concept_id="monkey",
        concept="monkey",
        rows=rows,
        disable_thinking=False,
    )
    assert "/no_think" not in reasoning_prompt
    reasoning_retry_prompt = module.build_retry_prompt(
        prompt=reasoning_prompt,
        concept="monkey",
        error=module.PostprocessError("paragraph 2 had too few monkey mentions"),
        min_concept_mentions=3,
        disable_thinking=False,
    )
    assert "/no_think" not in reasoning_retry_prompt


def test_less_repetitive_postprocess_template_writes_outputs(tmp_path: Path) -> None:
    module = _load_script_module(
        "scripts/synthetic_data_generation/concept_curriculum/postprocess_comps_concept_properties.py",
        "postprocess_comps_concept_properties_template_test",
    )
    input_documents = tmp_path / "documents.jsonl"
    rows = [
        {
            "id": "coach__core_profile",
            "concept_id": "coach",
            "concept": "coach",
            "style_id": "core_profile",
            "sense_hint": "a coach, meaning a long passenger bus",
            "text": "A coach is a long passenger bus, and a coach carries coach groups.",
            "word_count": 11,
        },
        {
            "id": "coach__parts_and_properties",
            "concept_id": "coach",
            "concept": "coach",
            "style_id": "parts_and_properties",
            "sense_hint": "a coach, meaning a long passenger bus",
            "text": "A coach has coach seats, windows, wheels, and coach luggage storage.",
            "word_count": 10,
        },
    ]
    input_documents.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = module.postprocess_documents(
        input_documents=input_documents,
        output_documents=tmp_path / "documents_less_repetitive.jsonl",
        output_train=tmp_path / "train_less_repetitive.txt",
        output_prompts=tmp_path / "postprocess_prompts.jsonl",
        output_manifest=tmp_path / "postprocess_manifest.json",
        raw_output_dir=tmp_path / "raw",
        renderer=module.TEMPLATE,
    )

    assert len(result.prompts) == 1
    assert len(result.documents) == 2
    assert result.manifest["counts"]["concepts"] == 1
    assert result.manifest["counts"]["documents"] == 2
    assert result.manifest["counts"]["leaky_documents"] == 0
    assert result.manifest["reasoning_effort"] is None
    assert result.manifest["max_tokens"] == module.DEFAULT_MAX_TOKENS
    assert result.manifest["max_attempts"] == module.DEFAULT_MAX_ATTEMPTS
    assert "long passenger bus" in result.prompts[0]["prompt"]
    assert result.outputs["documents"].exists()
    assert result.outputs["train"].exists()
    assert result.outputs["manifest"].exists()
    output_rows = [
        json.loads(line)
        for line in result.outputs["documents"].read_text(encoding="utf-8").splitlines()
    ]
    assert output_rows[0]["postprocess"]["method"] == "less_repetitive_paragraphs"
    assert output_rows[0]["source_text"] == rows[0]["text"]
    assert "A coach is a long passenger bus" in result.outputs["train"].read_text(encoding="utf-8")


def test_concept_type_classifier_batch_helpers() -> None:
    module = _load_script_module(
        "scripts/synthetic_data_generation/concept_curriculum/classify_comps_concept_types.py",
        "classify_comps_concept_types_test",
    )
    concepts = tuple(f"concept_{index}" for index in range(85))

    batches = module.split_batches(concepts, batch_size=40)

    assert [len(batch) for batch in batches] == [40, 40, 5]

    prompt = module.build_classifier_prompt(
        ("parsley", "unknown_word"),
        allowed_labels=module.COMPS.ALLOWED_CONCEPT_TYPES,
    )
    assert "choose the closest broader allowed label" in prompt
    assert 'parsley -> ["plant", "ingredient"]' in prompt
    assert 'ladle -> ["kitchen_utensil"]' in prompt
    assert 'strainer -> ["kitchen_utensil"]' in prompt
    assert "unknown_word -> []" in prompt


def test_concept_type_classifier_validates_fake_qwen_json() -> None:
    module = _load_script_module(
        "scripts/synthetic_data_generation/concept_curriculum/classify_comps_concept_types.py",
        "classify_comps_concept_types_validate_test",
    )
    payload = {
        "concept_types": {
            "salmon": ["fish", "food"],
            "jar": ["human_made_object"],
            "bikini": ["clothing"],
            "bomb": ["weapon"],
            "clarinet": ["musical_instrument"],
            "ladle": ["kitchen_utensil"],
            "ice_cream": ["dessert"],
            "kiwi": ["fruit"],
            "lemon": ["fruit"],
            "lime": ["fruit"],
            "mystery": [],
        },
        "uncertain": [{"concept": "mystery", "reason": "unclear ordinary category"}],
    }

    result = module.validate_classification_payload(
        payload,
        requested_concepts=(
            "salmon",
            "jar",
            "bikini",
            "bomb",
            "clarinet",
            "ladle",
            "ice_cream",
            "kiwi",
            "lemon",
            "lime",
            "mystery",
        ),
        allowed_labels=module.COMPS.ALLOWED_CONCEPT_TYPES,
    )

    assert result.concept_types["salmon"] == ("fish", "food")
    assert result.concept_types["jar"] == ("human_made_object",)
    assert result.concept_types["bikini"] == ("clothing",)
    assert result.concept_types["bomb"] == ("weapon",)
    assert result.concept_types["clarinet"] == ("musical_instrument",)
    assert result.concept_types["ladle"] == ("kitchen_utensil",)
    assert result.concept_types["ice_cream"] == ("dessert",)
    assert result.concept_types["kiwi"] == ("fruit",)
    assert result.empty_label_concepts == ("mystery",)
    assert result.uncertain[0]["concept"] == "mystery"


def test_concept_type_classifier_retries_incomplete_or_malformed_output() -> None:
    module = _load_script_module(
        "scripts/synthetic_data_generation/concept_curriculum/classify_comps_concept_types.py",
        "classify_comps_concept_types_retry_test",
    )

    with pytest.raises(module.ClassificationBatchError) as malformed:
        module.parse_json_object("{")
    assert module.should_retry_classification_failure(malformed.value)

    with pytest.raises(module.ClassificationBatchError) as incomplete:
        module.validate_classification_payload(
            {"concept_types": {"salmon": ["fish"]}, "uncertain": []},
            requested_concepts=("salmon", "jar"),
            allowed_labels=module.COMPS.ALLOWED_CONCEPT_TYPES,
        )
    assert module.should_retry_classification_failure(incomplete.value)


def test_build_full_comps_config_from_annotation_fixture(tmp_path: Path) -> None:
    annotations_path = tmp_path / "concept_types.json"
    annotations_path.write_text(
        json.dumps(
            {
                "concept_types": {
                    "coach": ["vehicle"],
                    "salmon": ["fish", "food"],
                    "jar": ["human_made_object"],
                }
            }
        ),
        encoding="utf-8",
    )
    annotations = load_concept_type_annotations(annotations_path)

    config = build_full_comps_concept_properties_config(
        concepts=("coach", "jar", "salmon"),
        concept_type_annotations=annotations,
        concept_sense_overrides={
            "coach": {
                "sense_hint": "a coach, meaning a long passenger bus",
            }
        },
        name="tiny_full_config",
        output_dir="data/concept_curriculum/tiny_full_config",
    )
    plans = build_comps_concept_plans(config)
    rows = build_clean_multidoc_prompt_rows(plans)

    assert [plan.id for plan in plans] == ["coach", "jar", "salmon"]
    assert plans[0].sense_hint == "a coach, meaning a long passenger bus"
    assert plans[0].concept_types == ("vehicle",)
    assert plans[2].concept == "salmon"
    assert plans[2].concept_types == ("fish", "food")
    assert config["concepts"][0]["sense_hint"] == "a coach, meaning a long passenger bus"
    assert len([row for row in rows if row["concept_id"] == "coach"]) == 6
    assert len([row for row in rows if row["concept_id"] == "jar"]) == 6
    assert len([row for row in rows if row["concept_id"] == "salmon"]) == 9
    assert 'Use this intended sense of "coach"' in rows[0]["prompt"]
    assert config["renderer"]["max_tokens"] == 190
    assert config["renderer"]["reasoning_effort"] == "none"


def test_full_config_builder_script_summarizes_tiny_config(tmp_path: Path) -> None:
    module = _load_script_module(
        "scripts/synthetic_data_generation/concept_curriculum/build_comps_concept_properties_full_config.py",
        "build_comps_concept_properties_full_config_test",
    )
    config = build_full_comps_concept_properties_config(
        concepts=("ball", "tongue"),
        concept_type_annotations={
            "ball": ("toy", "sports_equipment"),
            "tongue": ("body_part",),
        },
        name="tiny_summary_config",
        output_dir="data/concept_curriculum/tiny_summary_config",
    )

    summary = module.summarize_config(config)

    assert summary["concepts"] == 2
    assert summary["prompts"] == 12
    assert summary["untyped_count"] == 0
    assert summary["type_counts"]["toy"] == 1
    assert summary["type_counts"]["sports_equipment"] == 1
    assert summary["type_counts"]["body_part"] == 1


def test_comps_concept_properties_prompt_only_artifacts(tmp_path: Path) -> None:
    result = build_comps_concept_properties(_toy_config(tmp_path), base_dir=ROOT)

    assert len(result.prompts) == 2
    assert result.documents == ()
    assert result.outputs["prompts"].exists()
    assert result.outputs["manifest"].exists()
    assert "train" not in result.outputs

    rows = [
        json.loads(line)
        for line in result.outputs["prompts"].read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == ["sheep", "toaster"]
    assert all(row["target_words"] == 200 for row in rows)
    assert result.manifest["counts"] == {
        "concepts": 2,
        "documents": 0,
        "prompts": 2,
        "train_words": 0,
    }


def test_clean_multidoc_prompt_only_artifacts(tmp_path: Path) -> None:
    result = build_comps_concept_properties(_clean_toy_config(tmp_path), base_dir=ROOT)

    assert len(result.prompts) == 8
    assert result.documents == ()
    assert result.manifest["prompt_variant"] == CLEAN_MULTIDOC_NO_NEIGHBORS
    assert result.manifest["counts"] == {
        "concepts": 2,
        "documents": 0,
        "prompts": 8,
        "train_words": 0,
    }
    rows = [
        json.loads(line)
        for line in result.outputs["prompts"].read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["id"] == "sheep__core_profile"
    assert rows[-1]["id"] == "toaster__natural_contrasts"


def test_comps_concept_properties_template_renderer_writes_train(tmp_path: Path) -> None:
    result = build_comps_concept_properties(
        _toy_config(tmp_path, renderer=TEMPLATE),
        base_dir=ROOT,
    )

    assert len(result.documents) == 2
    assert result.outputs["documents"].exists()
    assert result.outputs["train"].exists()
    assert result.outputs["audit_report"].exists()
    assert result.audit_report is not None
    assert result.audit_report["counts"]["fail"] == 0

    train_lines = result.outputs["train"].read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 2
    assert "sheep" in train_lines[0].lower()


def test_comps_concept_properties_template_renderer_resumes_checkpoints(tmp_path: Path) -> None:
    config = _toy_config(tmp_path, renderer=TEMPLATE)
    first = build_comps_concept_properties(config, base_dir=ROOT)

    checkpoint_dir = first.outputs["render_checkpoints"]
    checkpoint_paths = sorted(checkpoint_dir.glob("*.json"))
    assert len(checkpoint_paths) == 2
    assert first.manifest["render_checkpoints"]["missing"] == 2
    assert first.manifest["render_checkpoints"]["written"] == 2
    assert first.manifest["render_checkpoints"]["reused"] == 0

    checkpoint_payload = json.loads(checkpoint_paths[0].read_text(encoding="utf-8"))
    checkpoint_payload["text"] = "Sentinel sheep checkpoint text."
    checkpoint_payload["word_count"] = 4
    checkpoint_paths[0].write_text(
        json.dumps(checkpoint_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    second = build_comps_concept_properties(config, base_dir=ROOT)

    assert second.manifest["render_checkpoints"]["reused"] == 2
    assert second.manifest["render_checkpoints"]["written"] == 0
    assert second.documents[0]["text"] == "Sentinel sheep checkpoint text."
    assert "Sentinel sheep checkpoint text." in second.outputs["train"].read_text(encoding="utf-8")


def test_comps_concept_properties_cli_prompt_only(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(
        "\n".join(
            [
                "name: comps_concept_properties_cli_test",
                "target_words_per_concept: 200",
                f"output_dir: {output_dir}",
                "renderer:",
                "  type: prompt_only",
                "concepts:",
                "  - id: sheep",
                "    concept: sheep",
                "    contrast_set: [goat, cow, lamb]",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "synthetic_data_generation"
                / "concept_curriculum"
                / "generate_comps_concept_properties.py"
            ),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "COMPS concept properties: comps_concept_properties_cli_test" in completed.stdout
    assert "prompts=1" in completed.stdout
    assert (output_dir / "prompts.jsonl").exists()
    assert (output_dir / "manifest.json").exists()
