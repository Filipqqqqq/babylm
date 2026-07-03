from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from babysteps.concept_curriculum.audit import audit_document, audit_documents
from babysteps.concept_curriculum.pipeline import (
    DeterministicConceptCurriculumMockGenerator,
    build_material_class_render_prompt,
    build_physical_dynamics_render_prompt,
    build_render_prompt,
    build_scenario_specs,
    json_items,
    load_pipeline_config,
    render_documents,
    run_concept_curriculum_pipeline,
)
from babysteps.cli.generate_concept_curriculum import (
    _apply_document_count_override,
    _apply_documents_per_request_override,
)
from babysteps.cli.audit_concept_curriculum import main as audit_main


ROOT = Path(__file__).resolve().parents[1]
CONCEPT_ARCHIVE = ROOT / "configs" / "concept_curriculum" / "archive"
PHYSICAL_ARCHIVE = (
    ROOT / "configs" / "concept_curriculum" / "physical_dynamics" / "archive"
)
CONFIG_PATH = CONCEPT_ARCHIVE / "stir_pilot_20.json"
POUR_CONFIG_PATH = CONCEPT_ARCHIVE / "pour_pilot_20.json"
SQUEEZE_CONFIG_PATH = CONCEPT_ARCHIVE / "squeeze_pilot_20.json"
MATERIAL_AFFORDANCE_EWOK_CONFIG_PATH = (
    CONCEPT_ARCHIVE / "material_affordance_ewok_20.json"
)
PHYSICAL_DYNAMICS_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_pilot_20.json"
)
PHYSICAL_DYNAMICS_CORE_80_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_core_80.json"
)
PHYSICAL_DYNAMICS_SINK_FLOAT_80_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_sink_float_80.json"
)
PHYSICAL_DYNAMICS_FALL_RISE_80_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_fall_rise_80.json"
)
PHYSICAL_DYNAMICS_LANDMARK_DIRECTION_60_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_landmark_direction_60.json"
)
PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_140_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_core_plus_landmark_140.json"
)
PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_CANONICAL_108_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_core_plus_landmark_canonical_108.json"
)
PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_PHYSICAL_120_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_core_plus_landmark_physical_120.json"
)
PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_BROAD_200_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_core_plus_landmark_broad_200.json"
)
PHYSICAL_DYNAMICS_SCENARIO_EXPANSIONS_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_scenario_expansions_v1.json"
)
PHYSICAL_DYNAMICS_W2V_SHORTCUTS_200_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_w2v_lexical_shortcuts_200.json"
)
PHYSICAL_DYNAMICS_W2V_OBJECT_AGNOSTIC_SHORTCUTS_200_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_w2v_object_agnostic_shortcuts_200.json"
)
PHYSICAL_DYNAMICS_EWOK_MICRO_500_CONFIG_PATH = (
    ROOT
    / "configs"
    / "concept_curriculum"
    / "physical_dynamics"
    / "physical_dynamics_ewok_micro_500.json"
)
PHYSICAL_DYNAMICS_BROAD_PLUS_BRIDGES_800_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_broad_plus_bridges_800.json"
)
PHYSICAL_DYNAMICS_BROAD_PLUS_BRIDGES_800_WEIGHTED_12P5_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_broad_plus_bridges_800_weighted_12p5.json"
)
PHYSICAL_DYNAMICS_BROAD_PLUS_BRIDGES_800_WEIGHTED_18P5_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_broad_plus_bridges_800_weighted_18p5.json"
)
PHYSICAL_DYNAMICS_BROAD_PLUS_BRIDGES_800_WEIGHTED_18P5_ALTITUDE_BRIDGE_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_broad_plus_bridges_800_weighted_18p5_altitude_bridge.json"
)
PHYSICAL_DYNAMICS_BROAD_PLUS_BRIDGES_800_WEIGHTED_18P5_OBJECT_AGNOSTIC_BG_CONFIG_PATH = (
    PHYSICAL_ARCHIVE
    / "physical_dynamics_broad_plus_bridges_800_weighted_18p5_object_agnostic_bg.json"
)
PHYSICAL_DYNAMICS_CANONICAL_INVENTORY_PATH = (
    ROOT
    / "configs"
    / "concept_curriculum"
    / "physical_dynamics"
    / "canonical_inventory.json"
)
EXPECTED_SIGNAL_ORDER_COUNTS_20 = {
    "positive_then_boundary": 4,
    "boundary_then_positive": 4,
    "neighbor_then_target": 4,
    "target_then_neighbor": 4,
    "material_choice_pair": 4,
}


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_json_items_accepts_text_only_rows_from_model():
    assert json_items({"items": ["First document.", "Second document."]}) == [
        {"text": "First document."},
        {"text": "Second document."},
    ]
    assert json_items({"doc_a": "First document.", "doc_b": "Second document."}) == [
        {"id": "doc_a", "text": "First document."},
        {"id": "doc_b", "text": "Second document."},
    ]
    assert json_items({"doc_a": {"text": "First document."}}) == [
        {"id": "doc_a", "text": "First document."},
    ]


def test_render_documents_checkpoints_and_falls_back_after_invalid_json(tmp_path):
    config = load_pipeline_config(CONFIG_PATH)
    config = dict(config)
    config["batching"] = {"documents_per_request": 2}
    config["request"] = {"invalid_json_retries": 1}
    specs = build_scenario_specs(config)[:2]
    checkpoint_dir = tmp_path / "render_checkpoints"
    failure_dir = tmp_path / "render_failures"

    class SplitFallbackGenerator:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def __call__(
            self,
            *,
            prompt: str,
            system_prompt: str,
            temperature: float,
            max_tokens: int,
            stage: str,
            response_format: dict | None = None,
        ) -> str:
            del system_prompt, temperature, max_tokens, stage, response_format
            payload = prompt.split("SCENARIO_SPECS_JSON:", 1)[1]
            payload = payload.split("/no_think", 1)[0].strip()
            prompt_specs = json.loads(payload)["scenario_specs"]
            self.batch_sizes.append(len(prompt_specs))
            if len(prompt_specs) > 1:
                return '{\n": [ [ ['
            return json.dumps(
                {
                    "items": [
                        {
                            "id": prompt_specs[0]["id"],
                            "text": f"Draft text for {prompt_specs[0]['id']}.",
                        }
                    ]
                }
            )

    generator = SplitFallbackGenerator()

    documents = render_documents(
        config=config,
        specs=specs,
        generator=generator,
        checkpoint_dir=checkpoint_dir,
        failure_dir=failure_dir,
    )

    assert [row["id"] for row in documents] == [spec["id"] for spec in specs]
    assert generator.batch_sizes == [2, 2, 1, 1]
    assert (checkpoint_dir / "batch_000000_n002.jsonl").exists()
    raw_failures = sorted(failure_dir.glob("*.raw.txt"))
    assert len(raw_failures) == 2
    assert all("[ [ [" in path.read_text(encoding="utf-8") for path in raw_failures)

    class ExplodingGenerator:
        def __call__(self, **kwargs: object) -> str:
            raise AssertionError("checkpointed render should not call the generator")

    resumed = render_documents(
        config=config,
        specs=specs,
        generator=ExplodingGenerator(),
        checkpoint_dir=checkpoint_dir,
        failure_dir=failure_dir,
    )

    assert resumed == documents


def test_render_documents_checkpoint_requires_matching_spec_fingerprint(tmp_path):
    config = load_pipeline_config(CONFIG_PATH)
    config = dict(config)
    config["batching"] = {"documents_per_request": 1}
    specs = build_scenario_specs(config)[:1]
    checkpoint_dir = tmp_path / "render_checkpoints"

    class EchoGenerator:
        def __init__(self, label: str) -> None:
            self.label = label
            self.calls = 0

        def __call__(
            self,
            *,
            prompt: str,
            system_prompt: str,
            temperature: float,
            max_tokens: int,
            stage: str,
            response_format: dict | None = None,
        ) -> str:
            del system_prompt, temperature, max_tokens, stage, response_format
            self.calls += 1
            payload = prompt.split("SCENARIO_SPECS_JSON:", 1)[1]
            payload = payload.split("/no_think", 1)[0].strip()
            prompt_specs = json.loads(payload)["scenario_specs"]
            return json.dumps(
                {
                    "items": [
                        {
                            "id": prompt_specs[0]["id"],
                            "text": f"{self.label} render for {prompt_specs[0]['id']}.",
                        }
                    ]
                }
            )

    first_generator = EchoGenerator("first")
    first_documents = render_documents(
        config=config,
        specs=specs,
        generator=first_generator,
        checkpoint_dir=checkpoint_dir,
    )

    checkpoint_rows = read_jsonl(checkpoint_dir / "batch_000000_n001.jsonl")
    assert first_generator.calls == 1
    assert checkpoint_rows[0]["render_checkpoint"]["spec_fingerprint"]

    stale_specs = deepcopy(specs)
    stale_specs[0]["positive_material"] = "changed material"
    second_generator = EchoGenerator("fresh")
    second_documents = render_documents(
        config=config,
        specs=stale_specs,
        generator=second_generator,
        checkpoint_dir=checkpoint_dir,
    )

    assert second_generator.calls == 1
    assert second_documents[0]["text"].startswith("fresh render")
    assert (
        second_documents[0]["render_checkpoint"]["spec_fingerprint"]
        != first_documents[0]["render_checkpoint"]["spec_fingerprint"]
    )


def test_stir_pilot_planner_balances_document_types():
    config = load_pipeline_config(CONFIG_PATH)

    specs = build_scenario_specs(config)

    assert len(specs) == 20
    assert Counter(spec["document_type"] for spec in specs) == {
        "direct_positive": 5,
        "subtype_explanation": 5,
        "state_contrast": 5,
        "neighbor_contrast": 5,
    }
    assert {spec["concept_id"] for spec in specs} == {"stir"}
    assert {spec["coverage_pack_id"] for spec in specs} == {"general_affordance"}
    assert all(spec["lexical_forms"] == ["stir", "stirs", "stirred", "stirring"] for spec in specs)
    assert Counter(spec["style_id"] for spec in specs) == {
        "concept_map_paragraph": 5,
        "observation_log": 5,
        "child_science_explanation": 5,
        "everyday_narrative": 5,
    }
    assert Counter(spec["signal_order_id"] for spec in specs) == EXPECTED_SIGNAL_ORDER_COUNTS_20


def test_stir_pilot_planner_covers_general_buckets():
    config = load_pipeline_config(CONFIG_PATH)
    specs = build_scenario_specs(config)

    positive_subtypes = {
        spec.get("positive_subtype_id")
        for spec in specs
        if spec["document_type"] in {"direct_positive", "subtype_explanation", "neighbor_contrast"}
    }
    state_cases = {spec.get("state_case_id") for spec in specs if spec["document_type"] == "state_contrast"}
    neighbors = {spec.get("neighbor") for spec in specs if spec["document_type"] == "neighbor_contrast"}

    assert "soft_unset_gels_foams" in positive_subtypes
    assert "gases_vapors" in positive_subtypes
    assert "loose_chunks_aggregates" in positive_subtypes
    assert state_cases == {
        "cement_sets",
        "chocolate_solidifies",
        "gelatin_firms",
        "clay_dries_hard",
        "grains_pack",
    }
    assert neighbors == {"mix", "swirl", "shake", "pour", "knead"}


def test_pour_pilot_planner_and_prompt_are_concept_specific():
    config = load_pipeline_config(POUR_CONFIG_PATH)

    specs = build_scenario_specs(config)
    prompt = build_render_prompt(
        specs=specs[:6],
        concept_profile=config["concept_profile"],
        constraints=config["document_constraints"],
    )

    assert len(specs) == 20
    assert Counter(spec["document_type"] for spec in specs) == {
        "direct_positive": 5,
        "subtype_explanation": 5,
        "state_contrast": 5,
        "neighbor_contrast": 5,
    }
    assert {spec["concept_id"] for spec in specs} == {"pour"}
    assert all(spec["lexical_forms"] == ["pour", "pours", "poured", "pouring"] for spec in specs)
    assert Counter(spec["style_id"] for spec in specs) == {
        "concept_map_paragraph": 4,
        "observation_log": 4,
        "child_science_explanation": 3,
        "everyday_narrative": 3,
        "source_frame_vignette": 3,
        "handling_note": 3,
    }
    assert Counter(spec["signal_order_id"] for spec in specs) == EXPECTED_SIGNAL_ORDER_COUNTS_20
    assert "pour, pours, poured, or pouring" in prompt
    assert "For state contrasts, clearly mark why one state fits pour" in prompt
    assert "spill, drip, dump, sprinkle, scoop, stream, drain, or flow" in prompt
    assert "source_frame_terms" in prompt
    assert "name a source, container, opening, or spout" in prompt
    assert "source_frame_vignette" in prompt
    assert "handling_note" in prompt
    assert "signal_order_guidance" in prompt
    assert "boundary_then_positive" in prompt
    assert "do not always lead with the target concept" in prompt
    assert "stir, stirs, stirred, or stirring" not in prompt


def test_pour_audit_warns_when_source_frame_is_missing():
    config = load_pipeline_config(POUR_CONFIG_PATH)
    spec = next(spec for spec in build_scenario_specs(config) if spec["document_type"] == "direct_positive")

    missing_source = audit_document(
        {
            "text": (
                "Water can pour in a smooth stream because it is a thin liquid. "
                "The material flows downward under gravity and does not hold one fixed shape. "
                "This behavior differs from a brick, which can only be dropped or placed. "
                "The water is central because its own flowing motion makes the event possible."
            )
        },
        spec=spec,
        concept_profile=config["concept_profile"],
        constraints={"min_words": 0, "max_words": 200, "audit_banned_terms": []},
    )
    with_source = audit_document(
        {
            "text": (
                "Water pours from a pitcher in a smooth stream because it is a thin liquid. "
                "The material flows downward under gravity and does not hold one fixed shape. "
                "This behavior differs from a brick, which can only be dropped or placed. "
                "The water is central because its own flowing motion makes the event possible."
            )
        },
        spec=spec,
        concept_profile=config["concept_profile"],
        constraints={"min_words": 0, "max_words": 200, "audit_banned_terms": []},
    )
    can_source = audit_document(
        {
            "text": (
                "Paint poured from the can in a heavy stream because it is a viscous liquid. "
                "The material moved slowly but still left the opening under gravity. "
                "A brick would not pour this way, because one rigid object must be placed or dropped."
            )
        },
        spec=spec,
        concept_profile=config["concept_profile"],
        constraints={"min_words": 0, "max_words": 200, "audit_banned_terms": []},
    )
    edge_source = audit_document(
        {
            "text": (
                "Melted wax began to pour from the edge of the candle after it softened. "
                "The warm material formed a thin stream down the side. "
                "Once the wax cooled, it held shape again and no longer poured like a liquid."
            )
        },
        spec=spec,
        concept_profile=config["concept_profile"],
        constraints={"min_words": 0, "max_words": 200, "audit_banned_terms": []},
    )

    assert any(flag["code"] == "missing_source_frame" for flag in missing_source["audit_flags"])
    assert not any(flag["code"] == "missing_source_frame" for flag in with_source["audit_flags"])
    assert not any(flag["code"] == "missing_source_frame" for flag in can_source["audit_flags"])
    assert not any(flag["code"] == "missing_source_frame" for flag in edge_source["audit_flags"])


def test_squeeze_pilot_covers_added_boundary_categories():
    config = load_pipeline_config(SQUEEZE_CONFIG_PATH)

    specs = build_scenario_specs(config)
    prompt = build_render_prompt(
        specs=specs[:6],
        concept_profile=config["concept_profile"],
        constraints=config["document_constraints"],
    )

    positive_subtypes = {
        spec.get("positive_subtype_id")
        for spec in specs
        if spec["document_type"] in {"direct_positive", "subtype_explanation", "neighbor_contrast"}
    }
    negative_subtypes = {
        spec.get("negative_subtype_id")
        for spec in specs
        if spec["document_type"] == "neighbor_contrast"
    }

    assert len(specs) == 20
    assert {spec["concept_id"] for spec in specs} == {"squeeze"}
    assert Counter(spec["signal_order_id"] for spec in specs) == EXPECTED_SIGNAL_ORDER_COUNTS_20
    assert all(
        spec["lexical_forms"] == ["squeeze", "squeezes", "squeezed", "squeezing"]
        for spec in specs
    )
    assert "granular_materials_in_flexible_bags" in positive_subtypes
    assert "crumplable_thin_solids" in positive_subtypes
    assert "soft_packed_masses" in positive_subtypes
    assert "rigid_but_grippable_boundary" in negative_subtypes
    assert "squeeze, squeezes, squeezed, or squeezing" in prompt
    assert "force_response_note" in prompt
    assert "material_choice_pair" in prompt
    assert "pour, pours, poured, or pouring" not in prompt


def test_material_affordance_ewok_planner_keeps_anchor_metadata_local():
    config = load_pipeline_config(MATERIAL_AFFORDANCE_EWOK_CONFIG_PATH)

    specs = build_scenario_specs(config)
    expected_document_counts = dict(config["coverage_packs"][0]["document_counts"])
    expected_scenario_count = sum(expected_document_counts.values())
    expected_style_counts = {
        style["style_id"]: expected_scenario_count // len(config["discourse_styles"])
        for style in config["discourse_styles"]
    }
    prompt = build_material_class_render_prompt(
        specs=specs[:6],
        material_class_profile=config["material_class_profile"],
        constraints=config["document_constraints"],
    )

    material_vocabulary = {
        material
        for spec in specs
        for material in spec.get("materials", [])
    }
    configured_material_vocabulary = {
        material
        for material_class in config["material_class_profile"]["material_classes"]
        for material in material_class.get("materials", [])
    }
    target_affordances = set(config["material_class_profile"]["target_affordances"])
    planned_targets = {spec.get("target_affordance") for spec in specs}

    assert len(specs) == expected_scenario_count
    assert Counter(spec["document_type"] for spec in specs) == expected_document_counts
    assert all("ewok_anchor" in spec["tags"] for spec in specs)
    assert all(not spec.get("companion_material") for spec in specs)
    assert all(len(spec["required_materials"]) == 1 for spec in specs)
    assert len({spec["material_class_id"] for spec in specs}) >= 10
    assert Counter(spec["style_id"] for spec in specs) == expected_style_counts
    assert target_affordances <= planned_targets
    assert all(spec["required_affordances"][0] == spec["target_affordance"] for spec in specs)
    assert {"ripple", "splash"} <= planned_targets
    assert {"play-doh", "terracotta", "terra-cotta"} <= material_vocabulary
    assert {
        "twill",
        "velvet",
        "play-doh",
        "jello",
        "helmet",
        "slipper",
        "feather",
        "terracotta",
        "terra-cotta",
    } <= configured_material_vocabulary
    assert "required_affordances" in prompt
    assert "target_affordance" in prompt
    assert "task_intention_vignette" in prompt
    assert "Include an explicit intention cue" in prompt
    assert "Do not write an imperative command" in prompt
    assert "twill" in prompt
    assert "exactly one central material" in prompt
    assert "companion_material" not in prompt
    assert "EWoK" not in prompt
    assert "ewok_anchor" not in prompt
    assert "ewok_anchor_correlations" not in prompt


def test_material_affordance_planner_balances_classes_within_target_actions():
    config = deepcopy(load_pipeline_config(MATERIAL_AFFORDANCE_EWOK_CONFIG_PATH))
    for pack in config["coverage_packs"]:
        for document_type in pack["document_counts"]:
            pack["document_counts"][document_type] = 125

    specs = build_scenario_specs(config)
    classes = config["material_class_profile"]["material_classes"]

    def supporting_class_ids(action: str) -> set[str]:
        return {
            row["id"]
            for row in classes
            if "ewok_anchor" in row.get("tags", []) and action in row.get("affordances", [])
        }

    def planned_class_counts(action: str) -> Counter:
        return Counter(
            spec["material_class_id"]
            for spec in specs
            if spec.get("target_affordance") == action
        )

    for action in ["squeeze", "stir", "pour", "tap"]:
        counts = planned_class_counts(action)
        assert set(counts) == supporting_class_ids(action)
        assert max(counts.values()) - min(counts.values()) <= 1

    materials_by_action = {
        action: {
            spec["material"]
            for spec in specs
            if spec.get("target_affordance") == action
        }
        for action in ["squeeze", "stir", "splash", "ripple", "pile"]
    }
    assert {"twill", "velvet", "fabric"} & materials_by_action["squeeze"]
    assert {"air", "water", "liquid"} <= materials_by_action["stir"]
    assert "water" in materials_by_action["splash"]
    assert "water" in materials_by_action["ripple"]
    assert "salt" in materials_by_action["pile"]


def test_material_affordance_audit_checks_correlation_fields():
    config = load_pipeline_config(MATERIAL_AFFORDANCE_EWOK_CONFIG_PATH)
    spec = next(
        spec for spec in build_scenario_specs(config) if spec["material_class_id"] == "gas_vapor_anchors"
    )
    material = spec["material"]
    affordances = spec["required_affordances"]

    good = audit_document(
        {
            "text": (
                f"{material} can {affordances[0]}, {affordances[1]}, and {affordances[2]} "
                "inside a sealed classroom demonstration tube. The moving gas can flow through the space."
            )
        },
        spec=spec,
        concept_profile=config["material_class_profile"],
        constraints={"min_words": 0, "max_words": 200, "min_required_affordances": 2},
    )
    missing_affordance = audit_document(
        {"text": f"{material} sits inside a sealed classroom demonstration tube."},
        spec=spec,
        concept_profile=config["material_class_profile"],
        constraints={"min_words": 0, "max_words": 200, "min_required_affordances": 2},
    )
    negative = audit_document(
        {
            "text": (
                f"{material} can {affordances[0]} and {affordances[1]} inside a sealed classroom demonstration tube, "
                "but it is not written as a contrast here."
            )
        },
        spec=spec,
        concept_profile=config["material_class_profile"],
        constraints={
            "min_words": 0,
            "max_words": 200,
            "min_required_affordances": 2,
            "negative_markers": ["not"],
        },
    )

    assert good["audit_status"] == "pass"
    assert any(flag["code"] == "missing_required_affordance" for flag in missing_affordance["audit_flags"])
    assert any(flag["code"] == "negative_marker" for flag in negative["audit_flags"])


def test_physical_dynamics_pilot_planner_balances_families_and_styles():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CONFIG_PATH)

    specs = build_scenario_specs(config)

    required_fields = {
        "family",
        "concept_pair",
        "cause_type",
        "object_type",
        "material_or_surface",
        "geometry_or_medium",
        "timescale",
        "path_shape",
        "style_id",
        "expected_process",
        "contrast_process",
        "cause_evidence",
        "outcome_evidence",
        "preferred_process_terms",
    }
    assert len(specs) == 20
    assert Counter(spec["family"] for spec in specs) == {
        "directional_motion": 4,
        "support_stability": 4,
        "force_speed": 4,
        "contact_collision": 4,
        "containment_flow": 4,
    }
    assert Counter(spec["style_id"] for spec in specs) == {
        "everyday_observation": 5,
        "cause_effect_explanation": 5,
        "comparison_contrast": 5,
        "observation_log": 5,
    }
    assert all(required_fields <= set(spec) for spec in specs)
    assert all(spec["path_shape"] for spec in specs)
    assert any(spec["path_shape"] == "none" for spec in specs)
    assert all(spec["required_cause_terms"] for spec in specs)
    assert all(spec["required_process_terms"] for spec in specs)
    assert all(spec["required_outcome_terms"] for spec in specs)
    directional_specs = [spec for spec in specs if spec["family"] == "directional_motion"]
    assert Counter(spec["expected_process"] for spec in directional_specs) == {
        "falling": 2,
        "rising": 2,
    }
    assert all(spec["cause_evidence"] for spec in directional_specs)
    force_speed_specs = [spec for spec in specs if spec["family"] == "force_speed"]
    assert {"accelerating", "slowing down"} <= {
        spec["expected_process"] for spec in force_speed_specs
    }
    preferred_terms = {
        term
        for spec in specs
        for term in spec.get("preferred_process_terms", [])
    }
    assert {
        "falling",
        "rising",
        "accelerating",
        "slowing down",
        "fell",
        "rose",
        "sped up",
        "slowed down",
    } <= preferred_terms


def test_physical_dynamics_core_80_balances_families_scenarios_and_styles():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CORE_80_CONFIG_PATH)

    specs = build_scenario_specs(config)
    covered_families = set(config["coverage_packs"][0]["document_counts"])
    covered_scenarios = [
        scenario["scenario_id"]
        for family in config["physical_dynamics_profile"]["families"]
        if family["family_id"] in covered_families
        for scenario in family["scenarios"]
    ]
    directional_scenarios = [
        scenario
        for family in config["physical_dynamics_profile"]["families"]
        if family["family_id"] == "directional_motion"
        for scenario in family["scenarios"]
    ]

    assert config["document_constraints"]["min_words"] == 35
    assert len(specs) == 80
    assert Counter(spec["family"] for spec in specs) == {
        "directional_motion": 16,
        "support_stability": 16,
        "force_speed": 16,
        "contact_collision": 16,
        "containment_flow": 16,
    }
    assert Counter(spec["style_id"] for spec in specs) == {
        "everyday_observation": 20,
        "cause_effect_explanation": 20,
        "comparison_contrast": 20,
        "observation_log": 20,
    }
    assert Counter(covered_scenarios).most_common(1)[0][1] == 1
    assert len(covered_scenarios) == 40
    assert Counter(spec["scenario_id"] for spec in specs) == {
        scenario_id: 2 for scenario_id in covered_scenarios
    }
    assert all(spec["family"] in covered_families for spec in specs)
    assert all(spec["required_cause_terms"] for spec in specs)
    assert all(spec["required_process_terms"] for spec in specs)
    assert all(spec["required_outcome_terms"] for spec in specs)

    upward = {
        "rising",
        "lifted",
    }
    downward = {
        "falling",
        "sinking",
    }
    assert sum(scenario["expected_process"] in upward for scenario in directional_scenarios) == 4
    assert sum(scenario["expected_process"] in downward for scenario in directional_scenarios) == 4


def test_physical_dynamics_sink_float_80_balances_lexical_scenarios():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_SINK_FLOAT_80_CONFIG_PATH)

    specs = build_scenario_specs(config)
    family = config["physical_dynamics_profile"]["families"][0]
    scenarios = family["scenarios"]
    sink_forms = {"sink", "sinks", "sank", "sinking"}
    float_forms = {"float", "floats", "floated", "floating"}
    avoided_objects = {"football", "volleyball", "wheel", "rock"}
    expected_lexical_targets = {
        "sink": 10,
        "sinks": 10,
        "sank": 10,
        "sinking": 10,
        "float": 10,
        "floats": 10,
        "floated": 10,
        "floating": 10,
    }

    assert config["document_constraints"]["min_words"] == 35
    assert family["family_id"] == "sink_float"
    assert len(scenarios) == 20
    assert len(specs) == 80
    assert Counter(spec["family"] for spec in specs) == {"sink_float": 80}
    assert Counter(spec["style_id"] for spec in specs) == {
        "everyday_observation": 20,
        "cause_effect_explanation": 20,
        "comparison_contrast": 20,
        "observation_log": 20,
    }
    assert Counter(spec["scenario_id"] for spec in specs) == {
        scenario["scenario_id"]: 4 for scenario in scenarios
    }
    assert Counter(spec["lexical_target"] for spec in specs) == expected_lexical_targets
    assert all(spec["lexical_target"] in spec["lexical_targets"] for spec in specs)
    assert sum(scenario["expected_process"] in sink_forms for scenario in scenarios) == 10
    assert sum(scenario["expected_process"] in float_forms for scenario in scenarios) == 10
    assert all(
        set(scenario["preferred_process_terms"]) == sink_forms
        or set(scenario["preferred_process_terms"]) == float_forms
        for scenario in scenarios
    )
    assert not any(
        avoided_object in str(scenario.get("object", "")).lower()
        for scenario in scenarios
        for avoided_object in avoided_objects
    )
    assert all(spec["required_cause_terms"] for spec in specs)
    assert all(spec["required_process_terms"] for spec in specs)
    assert all(spec["required_outcome_terms"] for spec in specs)


def test_physical_dynamics_fall_rise_80_balances_directional_evidence():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_FALL_RISE_80_CONFIG_PATH)

    specs = build_scenario_specs(config)
    family = config["physical_dynamics_profile"]["families"][0]
    scenarios = family["scenarios"]
    fall_forms = {"fall", "falls", "fell", "falling"}
    rise_forms = {"rise", "rises", "rose", "rising"}
    avoided_objects = {"football", "volleyball", "wheel", "rock"}
    expected_lexical_targets = {
        "fall": 10,
        "falls": 10,
        "fell": 10,
        "falling": 10,
        "rise": 10,
        "rises": 10,
        "rose": 10,
        "rising": 10,
    }

    assert config["document_constraints"]["min_words"] == 35
    assert family["family_id"] == "fall_rise"
    assert len(scenarios) == 20
    assert len(specs) == 80
    assert Counter(spec["family"] for spec in specs) == {"fall_rise": 80}
    assert Counter(spec["style_id"] for spec in specs) == {
        "everyday_observation": 20,
        "cause_effect_explanation": 20,
        "comparison_contrast": 20,
        "observation_log": 20,
    }
    assert Counter(spec["scenario_id"] for spec in specs) == {
        scenario["scenario_id"]: 4 for scenario in scenarios
    }
    assert Counter(spec["lexical_target"] for spec in specs) == expected_lexical_targets
    assert all(spec["lexical_target"] in spec["lexical_targets"] for spec in specs)
    assert Counter(scenario["expected_process"] for scenario in scenarios) == {
        "falling": 10,
        "rising": 10,
    }
    assert all(
        set(scenario["preferred_process_terms"]) == fall_forms
        or set(scenario["preferred_process_terms"]) == rise_forms
        for scenario in scenarios
    )
    assert all(
        scenario.get("start_position")
        and scenario.get("end_position")
        and scenario.get("direction_evidence")
        for scenario in scenarios
    )
    assert all(
        spec.get("start_position")
        and spec.get("end_position")
        and spec.get("direction_evidence")
        for spec in specs
    )
    assert not any(
        avoided_object in str(scenario.get("object", "")).lower()
        for scenario in scenarios
        for avoided_object in avoided_objects
    )
    assert all(spec["required_cause_terms"] for spec in specs)
    assert all(spec["required_process_terms"] for spec in specs)
    assert all(spec["required_outcome_terms"] for spec in specs)


def test_physical_dynamics_landmark_direction_60_balances_grid():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_LANDMARK_DIRECTION_60_CONFIG_PATH)

    specs = build_scenario_specs(config)
    family = config["physical_dynamics_profile"]["families"][0]
    scenarios = family["scenarios"]
    required_fields = {
        "landmark",
        "initial_relation_to_landmark",
        "later_relation_to_landmark",
        "crossing_direction",
        "verb_type",
        "event_verb",
        "mechanism",
        "lexical_target",
    }
    expected_landmarks = {
        "roofline",
        "water surface",
        "ledge",
        "window",
        "rim",
    }
    expected_verb_types = {
        "explicit",
        "direction_biased",
        "direction_neutral",
    }
    avoided_objects = {"football", "volleyball", "wheel", "rock"}

    assert config["document_constraints"]["min_words"] == 35
    assert family["family_id"] == "landmark_direction"
    assert len(scenarios) == 30
    assert len(specs) == 60
    assert Counter(spec["family"] for spec in specs) == {"landmark_direction": 60}
    assert Counter(spec["style_id"] for spec in specs) == {
        "everyday_observation": 15,
        "cause_effect_explanation": 15,
        "comparison_contrast": 15,
        "observation_log": 15,
    }
    assert Counter(spec["scenario_id"] for spec in specs) == {
        scenario["scenario_id"]: 2 for scenario in scenarios
    }
    assert Counter(scenario["landmark"] for scenario in scenarios) == {
        landmark: 6 for landmark in expected_landmarks
    }
    assert Counter(scenario["crossing_direction"] for scenario in scenarios) == {
        "upward": 15,
        "downward": 15,
    }
    assert Counter(scenario["verb_type"] for scenario in scenarios) == {
        verb_type: 10 for verb_type in expected_verb_types
    }
    assert Counter(spec["landmark"] for spec in specs) == {
        landmark: 12 for landmark in expected_landmarks
    }
    assert Counter(spec["crossing_direction"] for spec in specs) == {
        "upward": 30,
        "downward": 30,
    }
    assert Counter(spec["verb_type"] for spec in specs) == {
        verb_type: 20 for verb_type in expected_verb_types
    }
    assert Counter(
        (scenario["landmark"], scenario["crossing_direction"], scenario["verb_type"])
        for scenario in scenarios
    ) == {
        (landmark, direction, verb_type): 1
        for landmark in expected_landmarks
        for direction in {"upward", "downward"}
        for verb_type in expected_verb_types
    }
    assert all(required_fields <= set(spec) for spec in specs)
    assert all(spec["event_verb"] == spec["lexical_target"] for spec in specs)
    assert all(spec["lexical_target"] in spec["lexical_targets"] for spec in specs)
    assert all(spec["initial_relation_to_landmark"] != spec["later_relation_to_landmark"] for spec in specs)
    assert all(spec["required_cause_terms"] for spec in specs)
    assert all(spec["required_process_terms"] for spec in specs)
    assert all(spec["required_outcome_terms"] for spec in specs)
    assert not any(
        avoided_object in str(scenario.get("object", "")).lower()
        for scenario in scenarios
        for avoided_object in avoided_objects
    )


def test_physical_dynamics_core_plus_landmark_140_imports_landmark_family():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_140_CONFIG_PATH)

    specs = build_scenario_specs(config)
    landmark_specs = [spec for spec in specs if spec["family"] == "landmark_direction"]

    assert config["model_id"] == "accounts/fireworks/models/qwen3p7-plus"
    assert len(specs) == 140
    assert Counter(spec["family"] for spec in specs) == {
        "directional_motion": 16,
        "support_stability": 16,
        "force_speed": 16,
        "contact_collision": 16,
        "containment_flow": 16,
        "landmark_direction": 60,
    }
    assert Counter(spec["style_id"] for spec in specs) == {
        "everyday_observation": 35,
        "cause_effect_explanation": 35,
        "comparison_contrast": 35,
        "observation_log": 35,
    }
    assert len(landmark_specs) == 60
    assert Counter(spec["crossing_direction"] for spec in landmark_specs) == {
        "upward": 30,
        "downward": 30,
    }
    assert Counter(spec["verb_type"] for spec in landmark_specs) == {
        "explicit": 20,
        "direction_biased": 20,
        "direction_neutral": 20,
    }
    assert Counter(spec["landmark"] for spec in landmark_specs) == {
        "roofline": 12,
        "water surface": 12,
        "ledge": 12,
        "window": 12,
        "rim": 12,
    }
    assert all(spec["event_verb"] == spec["lexical_target"] for spec in landmark_specs)


def test_physical_dynamics_core_plus_landmark_canonical_108_balances_concepts():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_CANONICAL_108_CONFIG_PATH)
    expected_counts = Counter(config["coverage_packs"][0]["canonical_concept_counts"])

    specs = build_scenario_specs(config)
    landmark_specs = [spec for spec in specs if spec["family"] == "landmark_direction"]

    assert len(specs) == 108
    assert Counter(spec["canonical_concept_id"] for spec in specs) == expected_counts
    assert Counter(spec["style_id"] for spec in specs) == {
        "everyday_observation": 27,
        "cause_effect_explanation": 27,
        "comparison_contrast": 27,
        "observation_log": 27,
    }
    assert Counter(spec["crossing_direction"] for spec in landmark_specs) == {
        "upward": 4,
        "downward": 4,
    }
    assert len({spec["landmark"] for spec in landmark_specs}) >= 4
    assert all(spec["canonical_balance_role"] == "expected_process" for spec in specs)
    assert all(spec["canonical_inventory_version"] == "physical_dynamics_canonical_inventory_v1" for spec in specs)
    for concept_id in {
        "sink",
        "spill",
        "drain",
        "leak",
        "overflow",
        "flow_down",
        "lift_up",
        "absorb",
        "compress",
        "stretch",
        "crack",
        "dent",
        "scatter",
        "stick",
    }:
        concept_specs = [
            spec for spec in specs if spec["canonical_concept_id"] == concept_id
        ]
        assert len({spec["scenario_id"] for spec in concept_specs}) == 4

    prompt = build_physical_dynamics_render_prompt(
        specs=specs[:8],
        physical_dynamics_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert "canonical_concept_id" not in prompt
    assert "canonical_contrast_concept_id" not in prompt
    assert "canonical_balance_role" not in prompt


def test_physical_dynamics_core_plus_landmark_physical_120_targets_ewok_physical_concepts():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_PHYSICAL_120_CONFIG_PATH)
    expected_counts = Counter(config["coverage_packs"][0]["canonical_concept_counts"])

    specs = build_scenario_specs(config)

    assert len(specs) == 120
    assert Counter(spec["canonical_concept_id"] for spec in specs) == expected_counts
    assert Counter(spec["style_id"] for spec in specs) == {
        "everyday_observation": 30,
        "cause_effect_explanation": 30,
        "comparison_contrast": 30,
        "observation_log": 30,
    }
    for concept_id in {"sink", "float", "roll", "slide"}:
        concept_specs = [
            spec for spec in specs if spec["canonical_concept_id"] == concept_id
        ]
        assert len(concept_specs) == 4
        assert len({spec["scenario_id"] for spec in concept_specs}) == 4

    assert Counter(
        spec["lexical_target"]
        for spec in specs
        if spec["canonical_concept_id"] == "float"
    ) == {
        "float": 1,
        "floats": 1,
        "floated": 1,
        "floating": 1,
    }
    assert Counter(
        spec["lexical_target"]
        for spec in specs
        if spec["canonical_concept_id"] == "roll"
    ) == {
        "roll": 1,
        "rolls": 1,
        "rolled": 1,
        "rolling": 1,
    }
    assert Counter(
        spec["lexical_target"]
        for spec in specs
        if spec["canonical_concept_id"] == "slide"
    ) == {
        "slide": 1,
        "slides": 1,
        "slid": 1,
        "sliding": 1,
    }
    sink_targets = {
        spec["lexical_target"]
        for spec in specs
        if spec["canonical_concept_id"] == "sink"
    }
    assert sink_targets == {"sink", "sinks", "sank", "sinking"}


def test_physical_dynamics_broad_200_balances_concepts_and_hides_contrast_in_prompt():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_BROAD_200_CONFIG_PATH)
    expected_counts = Counter(config["coverage_packs"][0]["canonical_concept_counts"])

    specs = build_scenario_specs(config)

    assert config["physical_dynamics_profile"]["canonical_candidate_balance_axes"] == {
        "landmark_direction": ["landmark"],
        "vertical_relation_language_expansion": ["landmark"],
    }
    assert config["batching"]["documents_per_request"] == 10
    assert config["stages"]["document_render"]["max_tokens"] == 5000
    assert len(specs) == 200
    assert Counter(spec["canonical_concept_id"] for spec in specs) == expected_counts
    assert set(expected_counts.values()) == {6, 7}
    assert Counter(expected_counts.values()) == {7: 20, 6: 10}
    assert Counter(spec["style_id"] for spec in specs) == {
        "everyday_observation": 50,
        "cause_effect_explanation": 50,
        "comparison_contrast": 50,
        "observation_log": 50,
    }
    assert all(spec.get("contrast_process") for spec in specs)
    fall_rise_specs = [
        spec
        for spec in specs
        if spec["family"] == "fall_rise_direction_language_expansion"
    ]
    assert len(fall_rise_specs) == 4
    assert Counter(spec["canonical_concept_id"] for spec in fall_rise_specs) == {
        "fall": 2,
        "rise": 2,
    }
    assert Counter(spec["directional_polarity"] for spec in fall_rise_specs) == {
        "downward": 2,
        "upward": 2,
    }
    vertical_specs = [
        spec
        for spec in specs
        if spec["family"] == "vertical_relation_language_expansion"
    ]
    assert len(vertical_specs) == 8
    assert Counter(spec["canonical_concept_id"] for spec in vertical_specs) == {
        "fall": 2,
        "rise": 2,
        "sink": 2,
        "float": 2,
    }
    assert Counter(spec["directional_polarity"] for spec in vertical_specs) == {
        "downward": 2,
        "upward": 2,
        "downward_in_medium": 2,
        "same_level": 2,
    }
    vertical_fall_rise_landmarks = [
        spec["landmark"]
        for spec in vertical_specs
        if spec["canonical_concept_id"] in {"fall", "rise"}
    ]
    assert len(set(vertical_fall_rise_landmarks)) == len(vertical_fall_rise_landmarks)
    landmark_direction_specs = [
        spec
        for spec in specs
        if spec["family"] == "landmark_direction"
    ]
    landmark_counts = Counter(spec["landmark"] for spec in landmark_direction_specs)
    assert set(landmark_counts) == {
        "roofline",
        "water surface",
        "ledge",
        "window",
        "rim",
    }
    assert max(landmark_counts.values()) - min(landmark_counts.values()) <= 1
    surface_cue_specs = [
        spec
        for spec in specs
        if spec["family"] == "sink_float_surface_cues"
    ]
    assert len(surface_cue_specs) == 4
    assert Counter(spec["canonical_concept_id"] for spec in surface_cue_specs) == {
        "sink": 2,
        "float": 2,
    }
    assert Counter(spec["directional_polarity"] for spec in surface_cue_specs) == {
        "downward_in_medium": 2,
        "same_level": 2,
    }
    speed_specs = [
        spec
        for spec in specs
        if spec["family"] == "speed_change_language_expansion"
    ]
    assert len(speed_specs) == 4
    assert Counter(spec["canonical_concept_id"] for spec in speed_specs) == {
        "speed_up": 2,
        "slow_down": 2,
    }
    assert Counter(spec["directional_polarity"] for spec in speed_specs) == {
        "speed_increase": 2,
        "speed_decrease": 2,
    }
    assert {
        spec["lexical_target"] for spec in speed_specs
    } == {"accelerating", "accelerated", "slowing down", "slowed down"}
    speed_quantity_specs = [
        spec
        for spec in specs
        if spec["family"] == "speed_quantity_change_cues"
    ]
    assert len(speed_quantity_specs) == 4
    assert Counter(spec["canonical_concept_id"] for spec in speed_quantity_specs) == {
        "speed_up": 2,
        "slow_down": 2,
    }
    assert Counter(spec["directional_polarity"] for spec in speed_quantity_specs) == {
        "speed_increase": 2,
        "speed_decrease": 2,
    }
    assert {
        spec["lexical_target"] for spec in speed_quantity_specs
    } == {"speeding up", "increasing speed", "losing speed", "decreasing speed"}
    assert all(spec["measurement_frame"] for spec in speed_quantity_specs)
    assert all(spec["initial_speed_evidence"] for spec in speed_quantity_specs)
    assert all(spec["later_speed_evidence"] for spec in speed_quantity_specs)
    assert all(spec["quantity_change"] for spec in speed_quantity_specs)
    assert all(spec["rate_evidence"] for spec in speed_quantity_specs)
    assert all(spec["speed_geometry"] for spec in speed_quantity_specs)
    assert all(spec["speed_geometry"]["initial_rate"] for spec in speed_quantity_specs)
    assert all(spec["speed_geometry"]["later_rate"] for spec in speed_quantity_specs)
    assert all(spec["speed_geometry"]["same_time_comparison"] for spec in speed_quantity_specs)
    assert all(spec["speed_geometry"]["definition_bridge"] for spec in speed_quantity_specs)
    assert all(spec["speed_geometry"]["process_bridge"] for spec in speed_quantity_specs)
    speed_endpoint_specs = [
        spec
        for spec in specs
        if spec["family"] == "speed_endpoint_tendency_cues"
    ]
    assert len(speed_endpoint_specs) == 4
    assert Counter(spec["canonical_concept_id"] for spec in speed_endpoint_specs) == {
        "speed_up": 2,
        "slow_down": 2,
    }
    assert Counter(spec["directional_polarity"] for spec in speed_endpoint_specs) == {
        "speed_increase": 2,
        "speed_decrease": 2,
    }
    assert {
        spec["lexical_target"] for spec in speed_endpoint_specs
    } == {"accelerating", "accelerated", "slowing down", "slowed down"}
    assert all(spec["endpoint"] for spec in speed_endpoint_specs)
    assert all(spec["endpoint_tendency"] for spec in speed_endpoint_specs)
    assert all(spec["endpoint_risk"] for spec in speed_endpoint_specs)
    assert all(spec["stopping_evidence"] for spec in speed_endpoint_specs)
    assert all(spec["speed_geometry"] for spec in speed_endpoint_specs)
    speed_geometry_payload = json.dumps(
        [spec["speed_geometry"] for spec in speed_endpoint_specs]
    ).lower()
    assert "same amount of time" in speed_geometry_payload
    assert "speed increased" in speed_geometry_payload
    assert "speed decreased" in speed_geometry_payload
    speed_bridge_specs = [
        spec
        for spec in specs
        if spec["family"] == "speed_rate_definition_bridge"
    ]
    assert len(speed_bridge_specs) == 2
    assert Counter(spec["canonical_concept_id"] for spec in speed_bridge_specs) == {
        "speed_up": 1,
        "slow_down": 1,
    }
    assert Counter(spec["directional_polarity"] for spec in speed_bridge_specs) == {
        "speed_increase": 1,
        "speed_decrease": 1,
    }
    assert all(
        spec["render_template"] == "speed_rate_bridge_5_sentence"
        for spec in speed_bridge_specs
    )
    assert all(spec["template_sequence"] for spec in speed_bridge_specs)
    assert all(spec["speed_geometry"]["definition_bridge"] for spec in speed_bridge_specs)
    assert all(spec["speed_geometry"]["process_bridge"] for spec in speed_bridge_specs)
    roll_slide_surface_specs = [
        spec
        for spec in specs
        if spec["family"] == "roll_slide_surface_rotation_cues"
    ]
    assert len(roll_slide_surface_specs) == 4
    assert Counter(spec["canonical_concept_id"] for spec in roll_slide_surface_specs) == {
        "roll": 2,
        "slide": 2,
    }
    assert {
        spec["lexical_target"] for spec in roll_slide_surface_specs
    } == {"roll", "rolls", "slide", "slides"}
    assert all(spec["surface_friction"] for spec in roll_slide_surface_specs)
    assert all(spec["rotation_evidence"] for spec in roll_slide_surface_specs)

    prompt_specs = [
        spec
        for spec in specs
        if spec["canonical_concept_id"] in {"roll", "slide", "sink", "float"}
    ][:8]
    prompt = build_physical_dynamics_render_prompt(
        specs=prompt_specs,
        physical_dynamics_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )

    prompt_payload = json.loads(prompt.split("SCENARIO_SPECS_JSON:", 1)[1])
    assert prompt_payload["constraints"]["positive_only"] is True
    assert "contrast_verbalization" not in prompt_payload["constraints"]
    assert all("render_family" in spec for spec in prompt_payload["scenario_specs"])
    for prompt_spec in prompt_payload["scenario_specs"]:
        assert "concept_id" not in prompt_spec
        assert "profile_id" not in prompt_spec
        assert "coverage_pack_id" not in prompt_spec
        assert "canonical_concept_id" not in prompt_spec
        assert "contrast_process" not in prompt_spec
        assert "preferred_contrast_terms" not in prompt_spec
        assert "concept_pair" not in prompt_spec
        assert "family" not in prompt_spec
        assert "physical_family" not in prompt_spec
        assert "family_label" not in prompt_spec
        assert "family_schema" not in prompt_spec
        assert "scenario_id" not in prompt_spec
        assert None not in prompt_spec.values()
        assert [] not in prompt_spec.values()
        assert {} not in prompt_spec.values()
        assert prompt_spec.get("document_type") != "contrast_pair"
        assert prompt_spec.get("style_id") != "comparison_contrast"

    assert '"roll_slide"' not in prompt
    assert '"roll_slide_surface_rotation_cues"' not in prompt
    assert '"sink_float"' not in prompt
    assert '"concept_pair"' not in prompt
    assert '"speed_geometry": null' not in prompt
    assert '"start_position": null' not in prompt
    assert '"contrast_pair"' not in prompt
    assert '"comparison_contrast"' not in prompt
    assert "Positive-only render policy" in prompt
    assert "Positive-only lexical hygiene" in prompt
    assert "without sinking" in prompt
    assert "did not go under" in prompt
    assert "instead of floating" in prompt
    assert "For floating examples, use positive surface evidence" in prompt
    assert "For sinking examples, use positive downward evidence" in prompt
    assert "For speed examples, use positive speed evidence" in prompt
    assert "contrast process" not in prompt.lower()


def test_physical_dynamics_broad_plus_bridges_800_adds_controlled_bridge_slices():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_BROAD_PLUS_BRIDGES_800_CONFIG_PATH)
    expected_counts = Counter(config["coverage_packs"][0]["canonical_concept_counts"])

    specs = build_scenario_specs(config)

    assert len(specs) == 800
    assert Counter(spec["canonical_concept_id"] for spec in specs) == expected_counts
    assert set(expected_counts.values()) == {26, 27}
    assert Counter(expected_counts.values()) == {27: 20, 26: 10}
    assert Counter(spec["style_id"] for spec in specs) == {
        "everyday_observation": 200,
        "cause_effect_explanation": 200,
        "comparison_contrast": 200,
        "observation_log": 200,
    }
    assert config["batching"]["documents_per_request"] == 8

    bridge_specs = [
        spec
        for spec in specs
        if str(spec.get("render_template") or "").startswith("lexical_bridge_")
    ]
    assert len(bridge_specs) == 50
    assert Counter(spec["family"] for spec in bridge_specs) == {
        "fall_rise_landmark_lexical_bridge": 10,
        "sink_float_surface_lexical_bridge": 10,
        "float_rise_disambiguation_bridge": 6,
        "roll_slide_rotation_surface_bridge": 14,
        "speed_change_lexical_bridge": 10,
    }
    assert Counter(spec["canonical_concept_id"] for spec in bridge_specs) == {
        "fall": 5,
        "rise": 8,
        "sink": 5,
        "float": 8,
        "speed_up": 5,
        "slow_down": 5,
        "roll": 7,
        "slide": 7,
    }
    assert all(spec["bridge_type"] for spec in bridge_specs)
    assert all(spec["lexical_target"] for spec in bridge_specs)
    assert all(spec["same_sentence_evidence_terms"] for spec in bridge_specs)
    assert all(spec["evidence_clause"] for spec in bridge_specs)
    assert all(spec["contrast_evidence_bans"] for spec in bridge_specs)
    assert all(not spec.get("speed_geometry") for spec in bridge_specs)

    landmark_bridge_specs = [
        spec
        for spec in bridge_specs
        if spec["family"] == "fall_rise_landmark_lexical_bridge"
    ]
    assert {"ground", "floor", "tree branch", "roofline"} <= {
        spec["landmark"] for spec in landmark_bridge_specs
    }
    assert {"sky", "ceiling", "clouds", "stars"} <= {
        spec["landmark"] for spec in landmark_bridge_specs
    }

    float_rise_specs = [
        spec
        for spec in bridge_specs
        if spec["family"] == "float_rise_disambiguation_bridge"
    ]
    assert Counter(spec["canonical_concept_id"] for spec in float_rise_specs) == {
        "rise": 3,
        "float": 3,
    }
    assert all(spec["bridge_type"] == "float_rise_disambiguation" for spec in float_rise_specs)

    speed_bridge_specs = [
        spec
        for spec in bridge_specs
        if spec["family"] == "speed_change_lexical_bridge"
    ]
    assert Counter(spec["directional_polarity"] for spec in speed_bridge_specs) == {
        "speed_increase": 5,
        "speed_decrease": 5,
    }
    assert {
        spec["lexical_target"]
        for spec in speed_bridge_specs
        if spec["canonical_concept_id"] == "speed_up"
    } <= {"accelerating", "accelerated", "speeding up", "sped up"}
    assert {
        spec["lexical_target"]
        for spec in speed_bridge_specs
        if spec["canonical_concept_id"] == "slow_down"
    } <= {"slowing down", "slowed down", "lost speed", "decelerated"}

    prompt = build_physical_dynamics_render_prompt(
        specs=bridge_specs[:8],
        physical_dynamics_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    prompt_payload = json.loads(prompt.split("SCENARIO_SPECS_JSON:", 1)[1])
    assert 'render_template values beginning with "lexical_bridge_"' in prompt
    assert "bridge_type" in prompt
    assert "same_sentence_evidence_terms" in prompt
    assert "canonical_concept_id" not in prompt
    assert all("bridge_type" in spec for spec in prompt_payload["scenario_specs"])


def test_physical_dynamics_broad_plus_bridges_weighted_configs_upweight_bridges():
    cases = [
        (
            PHYSICAL_DYNAMICS_BROAD_PLUS_BRIDGES_800_WEIGHTED_12P5_CONFIG_PATH,
            100,
            {
                "fall_rise_landmark_lexical_bridge": 22,
                "sink_float_surface_lexical_bridge": 17,
                "float_rise_disambiguation_bridge": 17,
                "roll_slide_rotation_surface_bridge": 22,
                "speed_change_lexical_bridge": 22,
            },
            {
                "fall": 11,
                "rise": 17,
                "sink": 11,
                "float": 17,
                "speed_up": 11,
                "slow_down": 11,
                "roll": 11,
                "slide": 11,
            },
        ),
        (
            PHYSICAL_DYNAMICS_BROAD_PLUS_BRIDGES_800_WEIGHTED_18P5_CONFIG_PATH,
            148,
            {
                "fall_rise_landmark_lexical_bridge": 34,
                "sink_float_surface_lexical_bridge": 23,
                "float_rise_disambiguation_bridge": 23,
                "roll_slide_rotation_surface_bridge": 34,
                "speed_change_lexical_bridge": 34,
            },
            {
                "fall": 17,
                "rise": 23,
                "sink": 17,
                "float": 23,
                "speed_up": 17,
                "slow_down": 17,
                "roll": 17,
                "slide": 17,
            },
        ),
    ]

    for config_path, bridge_count, family_counts, concept_counts in cases:
        config = load_pipeline_config(config_path)
        expected_counts = Counter(config["coverage_packs"][0]["canonical_concept_counts"])

        specs = build_scenario_specs(config)
        bridge_specs = [
            spec
            for spec in specs
            if str(spec.get("render_template") or "").startswith("lexical_bridge_")
        ]

        assert len(specs) == 800
        assert Counter(spec["canonical_concept_id"] for spec in specs) == expected_counts
        assert set(expected_counts.values()) == {26, 27}
        assert Counter(expected_counts.values()) == {27: 20, 26: 10}
        assert Counter(spec["style_id"] for spec in specs) == {
            "everyday_observation": 200,
            "cause_effect_explanation": 200,
            "comparison_contrast": 200,
            "observation_log": 200,
        }
        assert len(bridge_specs) == bridge_count
        assert Counter(spec["family"] for spec in bridge_specs) == family_counts
        assert Counter(spec["canonical_concept_id"] for spec in bridge_specs) == concept_counts
        assert all(spec["lexical_target"] for spec in bridge_specs)
        assert all(spec["same_sentence_evidence_terms"] for spec in bridge_specs)


def test_physical_dynamics_weighted_18p5_altitude_bridge_balances_height_language():
    config = load_pipeline_config(
        PHYSICAL_DYNAMICS_BROAD_PLUS_BRIDGES_800_WEIGHTED_18P5_ALTITUDE_BRIDGE_CONFIG_PATH
    )
    expected_counts = Counter(config["coverage_packs"][0]["canonical_concept_counts"])

    specs = build_scenario_specs(config)
    bridge_specs = [
        spec
        for spec in specs
        if str(spec.get("render_template") or "").startswith("lexical_bridge_")
    ]
    altitude_specs = [
        spec
        for spec in bridge_specs
        if spec["family"] == "sink_float_altitude_height_bridge"
    ]

    assert len(specs) == 800
    assert Counter(spec["canonical_concept_id"] for spec in specs) == expected_counts
    assert len(bridge_specs) == 148
    assert Counter(spec["family"] for spec in bridge_specs) == {
        "fall_rise_landmark_lexical_bridge": 34,
        "sink_float_surface_lexical_bridge": 13,
        "sink_float_altitude_height_bridge": 10,
        "float_rise_disambiguation_bridge": 23,
        "roll_slide_rotation_surface_bridge": 34,
        "speed_change_lexical_bridge": 34,
    }
    assert Counter(spec["canonical_concept_id"] for spec in bridge_specs) == {
        "fall": 17,
        "rise": 23,
        "sink": 17,
        "float": 23,
        "speed_up": 17,
        "slow_down": 17,
        "roll": 17,
        "slide": 17,
    }

    assert len(altitude_specs) == 10
    assert Counter(spec["canonical_concept_id"] for spec in altitude_specs) == {
        "sink": 5,
        "float": 5,
    }
    assert Counter(spec["bridge_type"] for spec in altitude_specs) == {
        "losing_altitude": 3,
        "losing_height": 2,
        "same_altitude": 3,
        "same_height": 2,
    }

    evidence_terms = {
        term
        for spec in altitude_specs
        for term in spec["same_sentence_evidence_terms"]
    }
    assert {"losing altitude", "losing height", "same altitude", "same height"} <= evidence_terms
    assert {spec["object_set_id"] for spec in altitude_specs}.isdisjoint(
        {"balloon", "football", "rock", "volleyball", "wheel"}
    )


def test_physical_dynamics_broad_plus_bridges_object_agnostic_background_variant():
    config = load_pipeline_config(
        PHYSICAL_DYNAMICS_BROAD_PLUS_BRIDGES_800_WEIGHTED_18P5_OBJECT_AGNOSTIC_BG_CONFIG_PATH
    )
    expected_counts = Counter(config["coverage_packs"][0]["canonical_concept_counts"])

    specs = build_scenario_specs(config)
    bridge_specs = [
        spec
        for spec in specs
        if str(spec.get("render_template") or "").startswith("lexical_bridge_")
    ]
    shortcut_specs = [
        spec
        for spec in specs
        if str(spec.get("render_template") or "").startswith("lexical_shortcut_")
    ]

    assert len(specs) == 800
    assert Counter(spec["canonical_concept_id"] for spec in specs) == expected_counts
    assert set(expected_counts.values()) == {26, 27}
    assert Counter(expected_counts.values()) == {27: 20, 26: 10}
    assert Counter(spec["style_id"] for spec in specs) == {
        "everyday_observation": 200,
        "cause_effect_explanation": 200,
        "comparison_contrast": 200,
        "observation_log": 200,
    }

    assert len(bridge_specs) == 148
    assert Counter(spec["family"] for spec in bridge_specs) == {
        "fall_rise_landmark_lexical_bridge": 36,
        "float_rise_disambiguation_bridge": 20,
        "sink_float_surface_lexical_bridge": 20,
        "roll_slide_rotation_surface_bridge": 36,
        "speed_change_lexical_bridge": 36,
    }
    assert Counter(spec["canonical_concept_id"] for spec in bridge_specs) == {
        "fall": 18,
        "rise": 20,
        "sink": 18,
        "float": 20,
        "roll": 18,
        "slide": 18,
        "speed_up": 18,
        "slow_down": 18,
    }

    assert len(shortcut_specs) == 56
    assert Counter(spec["family"] for spec in shortcut_specs) == {
        "w2v_object_agnostic_shortcuts": 56
    }
    assert Counter(spec["canonical_concept_id"] for spec in shortcut_specs) == {
        "fall": 7,
        "rise": 7,
        "sink": 7,
        "float": 7,
        "speed_up": 7,
        "slow_down": 7,
        "roll": 7,
        "slide": 7,
    }
    assert all(spec["object_set_id"] for spec in shortcut_specs)
    assert all(spec["lexical_target"] for spec in shortcut_specs)
    assert all(spec["same_sentence_evidence_terms"] for spec in shortcut_specs)
    assert all(spec["contrast_evidence_bans"] for spec in shortcut_specs)

    concepts_by_object_set = {}
    for spec in shortcut_specs:
        concepts_by_object_set.setdefault(spec["object_set_id"], set()).add(
            spec["canonical_concept_id"]
        )
    assert {"button", "washer_ring", "toy_block"} <= set(concepts_by_object_set)
    for object_set_id in ("button", "washer_ring", "toy_block"):
        assert len(concepts_by_object_set[object_set_id]) >= 3


def test_physical_dynamics_ewok_micro_500_balances_pair_specific_bridges():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_EWOK_MICRO_500_CONFIG_PATH)
    expected_counts = Counter(config["coverage_packs"][0]["canonical_concept_counts"])

    specs = build_scenario_specs(config)
    bridge_specs = [
        spec
        for spec in specs
        if str(spec.get("render_template") or "").startswith("lexical_bridge_")
    ]

    assert len(specs) == 500
    assert len(bridge_specs) == 500
    assert Counter(spec["canonical_concept_id"] for spec in specs) == expected_counts
    assert Counter(spec["family"] for spec in specs) == {
        "ewok_fall_rise_micro_bridge": 100,
        "ewok_sink_float_micro_bridge": 100,
        "ewok_speed_micro_bridge": 100,
        "ewok_roll_slide_micro_bridge": 100,
        "ewok_grow_shrink_micro_bridge": 100,
    }
    assert Counter(spec["style_id"] for spec in specs) == {
        "micro_active_observation": 125,
        "micro_play_by_play": 125,
        "micro_measurement_note": 125,
        "micro_handling_note": 125,
    }
    assert Counter(spec["document_type"] for spec in specs) == {
        "micro_observation": 125,
        "micro_play_by_play": 125,
        "micro_measurement_note": 125,
        "micro_handling_note": 125,
    }
    assert all(spec["lexical_target"] for spec in bridge_specs)
    assert all(spec["same_sentence_evidence_terms"] for spec in bridge_specs)
    assert all(spec["contrast_evidence_bans"] for spec in bridge_specs)
    assert all(not spec.get("speed_geometry") for spec in bridge_specs)

    speed_up_targets = {
        spec["lexical_target"]
        for spec in specs
        if spec["canonical_concept_id"] == "speed_up"
    }
    slow_down_targets = {
        spec["lexical_target"]
        for spec in specs
        if spec["canonical_concept_id"] == "slow_down"
    }
    assert {"accelerating", "speeding up", "moved faster", "went faster"} <= speed_up_targets
    assert {"slowing down", "moved slower", "went slower", "losing speed"} <= slow_down_targets
    speed_endpoint_risk_specs = [
        spec
        for spec in specs
        if spec["family"] == "ewok_speed_micro_bridge"
        and spec["bridge_type"] == "speed_endpoint_risk"
    ]
    assert Counter(spec["canonical_concept_id"] for spec in speed_endpoint_risk_specs) == {
        "speed_up": 10,
        "slow_down": 10,
    }
    speed_endpoint_evidence = {
        term
        for spec in speed_endpoint_risk_specs
        for term in spec["same_sentence_evidence_terms"]
    }
    assert {"about to crash", "almost crashed", "came to a stop", "stopped beside the porch"} <= speed_endpoint_evidence
    speed_endpoint_bans = {
        term
        for spec in speed_endpoint_risk_specs
        for term in spec["contrast_evidence_bans"]
    }
    assert "crash" in speed_endpoint_bans
    assert "came to a stop" in speed_endpoint_bans

    fall_rise_specs = [
        spec for spec in specs if spec["family"] == "ewok_fall_rise_micro_bridge"
    ]
    fall_rise_evidence = {
        term
        for spec in fall_rise_specs
        for term in spec["same_sentence_evidence_terms"]
    }
    assert {
        "closer to the ground",
        "toward the ground",
        "closer to the sky",
        "toward the sky",
    } <= fall_rise_evidence
    sky_ground_specs = [
        spec for spec in fall_rise_specs if spec["bridge_type"] == "sky_ground_landmark"
    ]
    assert Counter(spec["canonical_concept_id"] for spec in sky_ground_specs) == {
        "fall": 7,
        "rise": 9,
    }
    for spec in sky_ground_specs:
        bans = set(spec["contrast_evidence_bans"])
        evidence = set(spec["same_sentence_evidence_terms"])
        if spec["canonical_concept_id"] == "fall":
            assert "ground" in evidence
            assert "ground" not in bans
            assert "sky" in bans
        else:
            assert "sky" in evidence
            assert "sky" not in bans
            assert "ground" in bans
    fall_rise_bans = {
        term
        for spec in fall_rise_specs
        for term in spec["contrast_evidence_bans"]
    }
    assert "above the branch" not in fall_rise_bans
    assert "below the branch" not in fall_rise_bans
    direct_fall_rise_specs = [
        spec for spec in fall_rise_specs if spec["bridge_type"] == "direct_motion_language"
    ]
    assert Counter(spec["canonical_concept_id"] for spec in direct_fall_rise_specs) == {
        "fall": 8,
        "rise": 11,
    }
    direct_fall_rise_evidence = {
        term
        for spec in direct_fall_rise_specs
        for term in spec["same_sentence_evidence_terms"]
    }
    assert {
        "falling downward",
        "dropped lower",
        "rising higher",
        "climbed upward",
        "rose higher",
    } <= direct_fall_rise_evidence

    slide_specs = [spec for spec in specs if spec["canonical_concept_id"] == "slide"]
    slide_evidence = {
        term
        for spec in slide_specs
        for term in spec["same_sentence_evidence_terms"]
    }
    assert {"same face stayed down", "wheels stayed still", "straight skid mark"} <= slide_evidence
    assert {
        "sliding forward on the slippery banana peel",
        "sliding across the wet slippery tile",
        "sliding across the surface",
        "slid forward across the icy patch",
        "sliding forward across the icy patch",
    } <= slide_evidence
    assert not any(term.startswith("did not roll") for term in slide_evidence)
    slide_targets = Counter(spec["lexical_target"] for spec in slide_specs)
    assert {
        "slip": 8,
        "slips": 8,
        "slipped": 8,
    }.items() <= slide_targets.items()
    roll_slide_orientation_specs = [
        spec
        for spec in specs
        if spec["family"] == "ewok_roll_slide_micro_bridge"
        and spec["bridge_type"] == "top_bottom_orientation"
    ]
    assert Counter(spec["canonical_concept_id"] for spec in roll_slide_orientation_specs) == {
        "roll": 10,
        "slide": 4,
    }
    roll_slide_orientation_evidence = {
        term
        for spec in roll_slide_orientation_specs
        for term in spec["same_sentence_evidence_terms"]
    }
    assert {
        "turning over as it rolled",
        "different side faced up",
        "cover stayed on top",
        "same side stayed facing up",
        "without turning over",
    } <= roll_slide_orientation_evidence
    slip_slide_specs = [
        spec for spec in slide_specs if spec["lexical_target"] in {"slip", "slips", "slipped"}
    ]
    assert all(
        any("slid" in term or "sliding" in term for term in spec["same_sentence_evidence_terms"])
        for spec in slip_slide_specs
    )

    grow_shrink_specs = [
        spec
        for spec in specs
        if spec["canonical_concept_id"] in {"grow", "shrink"}
    ]
    assert len(grow_shrink_specs) == 100
    assert all(spec["initial_size_evidence"] for spec in grow_shrink_specs)
    assert all(spec["later_size_evidence"] for spec in grow_shrink_specs)
    assert all(spec["size_change_evidence"] for spec in grow_shrink_specs)
    assert {"grew", "growing"} <= {
        spec["lexical_target"]
        for spec in grow_shrink_specs
        if spec["canonical_concept_id"] == "grow"
    }
    assert {"shrank", "shrunk", "shrinking"} <= {
        spec["lexical_target"]
        for spec in grow_shrink_specs
        if spec["canonical_concept_id"] == "shrink"
    }
    sponge_grow = next(
        spec for spec in grow_shrink_specs if spec["scenario_id"] == "sponge_grows_when_soaked"
    )
    foam_pellet_grow = next(
        spec for spec in grow_shrink_specs if spec["scenario_id"] == "foam_pellet_grows_when_warmed"
    )
    balloon_grow = next(
        spec for spec in grow_shrink_specs if spec["scenario_id"] == "balloon_grows_when_inflated"
    )
    assert "drying" not in sponge_grow["contrast_evidence_bans"]
    assert "drying out" in sponge_grow["contrast_evidence_bans"]
    assert foam_pellet_grow["required_cause_terms"] == ["warm"]
    assert "wide measuring ring" in balloon_grow["same_sentence_evidence_terms"]

    prompt = build_physical_dynamics_render_prompt(
        specs=bridge_specs[:8],
        physical_dynamics_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    prompt_payload = json.loads(prompt.split("SCENARIO_SPECS_JSON:", 1)[1])
    assert 'render_template values beginning with "lexical_bridge_"' in prompt
    assert "initial_size_evidence" in prompt
    assert "size_change_evidence" in prompt
    assert all(
        "canonical_concept_id" not in spec
        for spec in prompt_payload["scenario_specs"]
    )
    assert all(
        "contrast_process" not in spec
        for spec in prompt_payload["scenario_specs"]
    )


def test_document_count_override_scales_physical_canonical_counts():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_BROAD_200_CONFIG_PATH)

    scaled_config = _apply_document_count_override(config, 400)
    scaled_counts = Counter(
        scaled_config["coverage_packs"][0]["canonical_concept_counts"]
    )
    specs = build_scenario_specs(scaled_config)

    assert sum(scaled_counts.values()) == 400
    assert len(specs) == 400
    assert Counter(spec["canonical_concept_id"] for spec in specs) == scaled_counts
    assert {scaled_counts[concept_id] for concept_id in {"sink", "float", "roll", "slide"}} == {
        14
    }
    assert set(scaled_counts.values()) == {12, 14}
    fall_rise_specs = [
        spec
        for spec in specs
        if spec["family"] == "fall_rise_direction_language_expansion"
    ]
    assert len(fall_rise_specs) == 8
    assert Counter(spec["canonical_concept_id"] for spec in fall_rise_specs) == {
        "fall": 4,
        "rise": 4,
    }
    assert Counter(spec["directional_polarity"] for spec in fall_rise_specs) == {
        "downward": 4,
        "upward": 4,
    }
    assert {
        spec["lexical_target"] for spec in fall_rise_specs
    } == {"fall", "falls", "fell", "falling", "rise", "rises", "rose", "rising"}
    vertical_specs = [
        spec
        for spec in specs
        if spec["family"] == "vertical_relation_language_expansion"
    ]
    assert len(vertical_specs) == 16
    assert Counter(spec["canonical_concept_id"] for spec in vertical_specs) == {
        "fall": 4,
        "rise": 4,
        "sink": 4,
        "float": 4,
    }
    assert Counter(spec["directional_polarity"] for spec in vertical_specs) == {
        "downward": 4,
        "upward": 4,
        "downward_in_medium": 4,
        "same_level": 4,
    }
    assert all(spec["vertical_relation_language"] for spec in vertical_specs)
    assert all(spec["directional_polarity"] for spec in vertical_specs)
    assert all(spec["reference_frame"] for spec in vertical_specs)
    vertical_fall_rise_specs = [
        spec for spec in vertical_specs if spec["canonical_concept_id"] in {"fall", "rise"}
    ]
    assert Counter(spec["landmark"] for spec in vertical_fall_rise_specs) == {
        "ground": 1,
        "clouds": 1,
        "sky": 1,
        "floor": 1,
        "stars": 1,
        "roofline": 1,
        "ceiling": 1,
        "skyline": 1,
    }
    assert all(spec["initial_relation_to_landmark"] for spec in vertical_fall_rise_specs)
    assert all(spec["later_relation_to_landmark"] for spec in vertical_fall_rise_specs)
    assert all(spec["mechanism"] for spec in vertical_fall_rise_specs)
    surface_cue_specs = [
        spec
        for spec in specs
        if spec["family"] == "sink_float_surface_cues"
    ]
    assert len(surface_cue_specs) == 8
    assert Counter(spec["canonical_concept_id"] for spec in surface_cue_specs) == {
        "sink": 4,
        "float": 4,
    }
    assert Counter(spec["directional_polarity"] for spec in surface_cue_specs) == {
        "downward_in_medium": 4,
        "same_level": 4,
    }
    assert {
        spec["lexical_target"] for spec in surface_cue_specs
    } == {
        "sink",
        "sinks",
        "sank",
        "sinking",
        "float",
        "floats",
        "floated",
        "floating",
    }
    speed_specs = [
        spec
        for spec in specs
        if spec["family"] == "speed_change_language_expansion"
    ]
    assert len(speed_specs) == 8
    assert Counter(spec["canonical_concept_id"] for spec in speed_specs) == {
        "speed_up": 4,
        "slow_down": 4,
    }
    assert Counter(spec["directional_polarity"] for spec in speed_specs) == {
        "speed_increase": 4,
        "speed_decrease": 4,
    }
    assert {
        spec["lexical_target"] for spec in speed_specs
    } == {
        "accelerating",
        "accelerated",
        "speeding up",
        "increasing speed",
        "slowing down",
        "slowed down",
        "losing speed",
        "decreasing speed",
    }
    speed_quantity_specs = [
        spec
        for spec in specs
        if spec["family"] == "speed_quantity_change_cues"
    ]
    assert len(speed_quantity_specs) == 8
    assert Counter(spec["canonical_concept_id"] for spec in speed_quantity_specs) == {
        "speed_up": 4,
        "slow_down": 4,
    }
    assert Counter(spec["directional_polarity"] for spec in speed_quantity_specs) == {
        "speed_increase": 4,
        "speed_decrease": 4,
    }
    assert {
        spec["lexical_target"] for spec in speed_quantity_specs
    } == {
        "accelerating",
        "accelerated",
        "speeding up",
        "increasing speed",
        "slowing down",
        "slowed down",
        "losing speed",
        "decreasing speed",
    }
    assert all(spec["measurement_frame"] for spec in speed_quantity_specs)
    assert all(spec["initial_speed_evidence"] for spec in speed_quantity_specs)
    assert all(spec["later_speed_evidence"] for spec in speed_quantity_specs)
    assert all(spec["quantity_change"] for spec in speed_quantity_specs)
    assert all(spec["rate_evidence"] for spec in speed_quantity_specs)
    assert all(spec["speed_geometry"] for spec in speed_quantity_specs)
    assert all(spec["speed_geometry"]["definition_bridge"] for spec in speed_quantity_specs)
    assert all(spec["speed_geometry"]["process_bridge"] for spec in speed_quantity_specs)
    speed_endpoint_specs = [
        spec
        for spec in specs
        if spec["family"] == "speed_endpoint_tendency_cues"
    ]
    assert len(speed_endpoint_specs) == 8
    assert Counter(spec["canonical_concept_id"] for spec in speed_endpoint_specs) == {
        "speed_up": 4,
        "slow_down": 4,
    }
    assert Counter(spec["directional_polarity"] for spec in speed_endpoint_specs) == {
        "speed_increase": 4,
        "speed_decrease": 4,
    }
    assert {
        spec["lexical_target"] for spec in speed_endpoint_specs
    } == {
        "accelerating",
        "accelerated",
        "speeding up",
        "increasing speed",
        "slowing down",
        "slowed down",
        "losing speed",
        "decreasing speed",
    }
    assert all(spec["endpoint"] for spec in speed_endpoint_specs)
    assert all(spec["endpoint_tendency"] for spec in speed_endpoint_specs)
    assert all(spec["endpoint_risk"] for spec in speed_endpoint_specs)
    assert all(spec["stopping_evidence"] for spec in speed_endpoint_specs)
    assert all(spec["speed_geometry"] for spec in speed_endpoint_specs)
    assert all(spec["speed_geometry"]["same_time_comparison"] for spec in speed_endpoint_specs)
    assert all(spec["speed_geometry"]["optional_consequence"] for spec in speed_endpoint_specs)
    speed_bridge_specs = [
        spec
        for spec in specs
        if spec["family"] == "speed_rate_definition_bridge"
    ]
    assert len(speed_bridge_specs) == 4
    assert Counter(spec["canonical_concept_id"] for spec in speed_bridge_specs) == {
        "speed_up": 2,
        "slow_down": 2,
    }
    assert Counter(spec["directional_polarity"] for spec in speed_bridge_specs) == {
        "speed_increase": 2,
        "speed_decrease": 2,
    }
    assert all(
        spec["render_template"] == "speed_rate_bridge_5_sentence"
        for spec in speed_bridge_specs
    )
    assert all(spec["speed_geometry"] for spec in speed_bridge_specs)
    assert all(spec["speed_geometry"]["initial_rate"] for spec in speed_bridge_specs)
    assert all(spec["speed_geometry"]["later_rate"] for spec in speed_bridge_specs)
    assert all(spec["speed_geometry"]["definition_bridge"] for spec in speed_bridge_specs)
    assert all(spec["speed_geometry"]["process_bridge"] for spec in speed_bridge_specs)
    roll_slide_surface_specs = [
        spec
        for spec in specs
        if spec["family"] == "roll_slide_surface_rotation_cues"
    ]
    assert len(roll_slide_surface_specs) == 8
    assert Counter(spec["canonical_concept_id"] for spec in roll_slide_surface_specs) == {
        "roll": 4,
        "slide": 4,
    }
    assert {
        spec["lexical_target"] for spec in roll_slide_surface_specs
    } == {
        "roll",
        "rolls",
        "rolled",
        "rolling",
        "slide",
        "slides",
        "slid",
        "sliding",
    }
    assert all(spec["surface_friction"] for spec in roll_slide_surface_specs)
    assert all(spec["rotation_evidence"] for spec in roll_slide_surface_specs)

    scaled_800_config = _apply_document_count_override(config, 800)
    specs_800 = build_scenario_specs(scaled_800_config)
    vertical_fall_rise_specs_800 = [
        spec
        for spec in specs_800
        if spec["family"] == "vertical_relation_language_expansion"
        and spec["canonical_concept_id"] in {"fall", "rise"}
    ]
    vertical_landmark_counts_800 = Counter(
        spec["landmark"] for spec in vertical_fall_rise_specs_800
    )
    assert "sky" in vertical_landmark_counts_800
    assert set(vertical_landmark_counts_800) == {
        "ground",
        "clouds",
        "sky",
        "floor",
        "tree branch",
        "stars",
        "roofline",
        "ceiling",
        "skyline",
    }
    assert (
        max(vertical_landmark_counts_800.values())
        - min(vertical_landmark_counts_800.values())
        <= 1
    )
    landmark_counts_800 = Counter(
        spec["landmark"]
        for spec in specs_800
        if spec["family"] == "landmark_direction"
    )
    assert max(landmark_counts_800.values()) - min(landmark_counts_800.values()) <= 1
    speed_family_counts_800 = Counter(
        (spec["canonical_concept_id"], spec["family"])
        for spec in specs_800
        if spec["canonical_concept_id"] in {"speed_up", "slow_down"}
    )
    for concept_id in ("speed_up", "slow_down"):
        assert {
            speed_family_counts_800[(concept_id, family_id)]
            for family_id in (
                "speed_change_language_expansion",
                "speed_quantity_change_cues",
                "speed_endpoint_tendency_cues",
                "speed_rate_definition_bridge",
            )
        } == {7}


def test_documents_per_request_override_updates_batching():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_BROAD_200_CONFIG_PATH)

    updated = _apply_documents_per_request_override(config, 4)

    assert config["batching"]["documents_per_request"] == 10
    assert updated["batching"]["documents_per_request"] == 4


def test_physical_dynamics_w2v_shortcuts_are_balanced_and_prompted():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_W2V_SHORTCUTS_200_CONFIG_PATH)

    specs = build_scenario_specs(config)

    assert len(specs) == 200
    assert Counter(spec["canonical_concept_id"] for spec in specs) == {
        "fall": 25,
        "rise": 25,
        "sink": 25,
        "float": 25,
        "speed_up": 25,
        "slow_down": 25,
        "roll": 25,
        "slide": 25,
    }
    assert Counter(spec["style_id"] for spec in specs) == {
        "simple_active_observation": 100,
        "play_by_play": 100,
    }
    assert Counter(spec["document_type"] for spec in specs) == {
        "shortcut_observation": 100,
        "play_by_play": 100,
    }
    assert all(
        str(spec["render_template"]).startswith("lexical_shortcut_")
        for spec in specs
    )
    assert all(spec["lexical_target"] for spec in specs)
    assert all(spec["same_sentence_evidence_terms"] for spec in specs)
    assert all(spec["contrast_evidence_bans"] for spec in specs)
    assert all(not spec.get("speed_geometry") for spec in specs)

    speed_specs = [
        spec for spec in specs if spec["canonical_concept_id"] in {"speed_up", "slow_down"}
    ]
    assert {
        spec["lexical_target"]
        for spec in speed_specs
        if spec["canonical_concept_id"] == "speed_up"
    } == {"accelerated", "accelerating", "sped up", "speeding up"}
    assert {
        spec["lexical_target"]
        for spec in speed_specs
        if spec["canonical_concept_id"] == "slow_down"
    } == {"slowed down", "slowing down", "lost speed", "decelerated"}

    prompt = build_physical_dynamics_render_prompt(
        specs=specs[:8],
        physical_dynamics_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert 'render_template values beginning with "lexical_shortcut_"' in prompt
    assert "same_sentence_evidence_terms" in prompt
    assert "Put the assigned lexical_target and at least one" in prompt
    assert "Do not use long definition chains" in prompt


def test_physical_dynamics_object_agnostic_shortcuts_share_objects_and_prompt():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_W2V_OBJECT_AGNOSTIC_SHORTCUTS_200_CONFIG_PATH)

    specs = build_scenario_specs(config)

    assert len(specs) == 200
    assert Counter(spec["canonical_concept_id"] for spec in specs) == {
        "fall": 25,
        "rise": 25,
        "sink": 25,
        "float": 25,
        "speed_up": 25,
        "slow_down": 25,
        "roll": 25,
        "slide": 25,
    }
    assert Counter(spec["style_id"] for spec in specs) == {
        "simple_active_observation": 100,
        "play_by_play": 100,
    }
    assert Counter(spec["document_type"] for spec in specs) == {
        "shortcut_observation": 100,
        "play_by_play": 100,
    }
    assert len({spec["scenario_id"] for spec in specs}) == 56
    assert all(spec["object_set_id"] for spec in specs)
    assert all(
        str(spec["render_template"]).startswith("lexical_shortcut_")
        for spec in specs
    )
    assert all(spec["lexical_target"] for spec in specs)
    assert all(spec["same_sentence_evidence_terms"] for spec in specs)
    assert all(spec["contrast_evidence_bans"] for spec in specs)

    concepts_by_object_set = {}
    for spec in specs:
        concepts_by_object_set.setdefault(spec["object_set_id"], set()).add(
            spec["canonical_concept_id"]
        )
    assert all(len(concepts) >= 3 for concepts in concepts_by_object_set.values())
    assert concepts_by_object_set["paper_scrap"] >= {
        "fall",
        "rise",
        "sink",
        "float",
        "speed_up",
        "slow_down",
    }
    assert concepts_by_object_set["bead"] == {
        "fall",
        "rise",
        "sink",
        "float",
        "speed_up",
        "slow_down",
        "roll",
        "slide",
    }
    assert concepts_by_object_set["toy_cart"] >= {
        "fall",
        "rise",
        "float",
        "speed_up",
        "slow_down",
        "roll",
        "slide",
    }
    for object_set_id in ("button", "washer_ring", "toy_block"):
        assert concepts_by_object_set[object_set_id] == {
            "fall",
            "rise",
            "sink",
            "float",
            "speed_up",
            "slow_down",
            "roll",
            "slide",
        }

    prompt = build_physical_dynamics_render_prompt(
        specs=specs[:8],
        physical_dynamics_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    prompt_payload = json.loads(prompt.split("SCENARIO_SPECS_JSON:", 1)[1])
    assert "Object-agnostic shortcut guidance" in prompt
    assert "object_set_id" in prompt
    assert all("object_set_id" in spec for spec in prompt_payload["scenario_specs"])
    assert all("canonical_concept_id" not in spec for spec in prompt_payload["scenario_specs"])

    documents = render_documents(
        config=config,
        specs=specs[:2],
        generator=DeterministicConceptCurriculumMockGenerator(),
    )
    assert [row["metadata"]["object_set_id"] for row in documents] == [
        spec["object_set_id"] for spec in specs[:2]
    ]


def test_physical_dynamics_w2v_shortcut_audit_requires_local_evidence():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_W2V_SHORTCUTS_200_CONFIG_PATH)
    specs = build_scenario_specs(config)
    spec = next(spec for spec in specs if spec["canonical_concept_id"] == "speed_up")
    slide_spec = next(
        spec
        for spec in specs
        if spec["scenario_id"] == "book_slid_smooth_table_play_by_play"
    )

    good = audit_document(
        {
            "id": spec["id"],
            "text": (
                "A hand kept pushing the toy cart. "
                "The toy cart accelerated as its speed increased, moving faster."
            ),
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert {
        flag["code"] for flag in good["audit_flags"]
    }.isdisjoint({"missing_same_sentence_shortcut_evidence"})

    missing_cause_only = audit_document(
        {
            "id": spec["id"],
            "text": (
                f"The toy cart {spec['lexical_target']} as its speed increased, "
                "moving faster along the track."
            ),
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert missing_cause_only["audit_status"] == "warn"
    assert {
        flag["code"]: flag["severity"]
        for flag in missing_cause_only["audit_flags"]
    }["missing_physical_cause"] == "warning"
    assert {
        flag["code"] for flag in missing_cause_only["audit_flags"]
    }.isdisjoint({"missing_same_sentence_shortcut_evidence"})

    missing_process = audit_document(
        {
            "id": spec["id"],
            "text": "A hand kept pushing the toy cart as its speed increased and it moved faster.",
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert missing_process["audit_status"] == "fail"
    assert {
        flag["code"]: flag["severity"]
        for flag in missing_process["audit_flags"]
    }["missing_lexical_target"] == "error"

    separated = audit_document(
        {
            "id": spec["id"],
            "text": (
                "A hand kept pushing the toy cart. "
                "The toy cart accelerated. "
                "Its speed increased and it moved faster."
            ),
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert "missing_same_sentence_shortcut_evidence" in {
        flag["code"] for flag in separated["audit_flags"]
    }

    contrast = audit_document(
        {
            "id": spec["id"],
            "text": (
                "A hand kept pushing the toy cart. "
                "The toy cart accelerated as its speed increased, moving faster, "
                "then the speed decreased."
            ),
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert "shortcut_contrast_evidence_violation" in {
        flag["code"] for flag in contrast["audit_flags"]
    }

    negated_rotation = audit_document(
        {
            "id": slide_spec["id"],
            "text": (
                "A hand pushed the flat book across the smooth table. "
                f"The small book {slide_spec['lexical_target']} across the smooth table without rotating."
            ),
        },
        spec=slide_spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert {
        flag["code"] for flag in negated_rotation["audit_flags"]
    }.isdisjoint({"shortcut_contrast_evidence_violation"})

    positive_rotation = audit_document(
        {
            "id": slide_spec["id"],
            "text": (
                "A hand pushed the flat book across the smooth table. "
                f"The small book {slide_spec['lexical_target']} across the smooth table while rotating."
            ),
        },
        spec=slide_spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert "shortcut_contrast_evidence_violation" in {
        flag["code"] for flag in positive_rotation["audit_flags"]
    }


def test_physical_dynamics_ewok_micro_bridge_audit_requires_local_evidence():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_EWOK_MICRO_500_CONFIG_PATH)
    specs = build_scenario_specs(config)
    speed_spec = next(
        spec for spec in specs if spec["canonical_concept_id"] == "speed_up"
    )
    slide_spec = next(
        spec for spec in specs if spec["canonical_concept_id"] == "slide"
    )
    slipped_spec = next(
        spec for spec in specs if spec.get("lexical_target") == "slipped"
    )
    slips_spec = next(spec for spec in specs if spec.get("lexical_target") == "slips")

    good = audit_document(
        {
            "id": speed_spec["id"],
            "text": (
                "A hand kept pushing the toy cart. "
                "The toy cart was accelerating as its speed increased and it went faster."
            ),
        },
        spec=speed_spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert {
        flag["code"] for flag in good["audit_flags"]
    }.isdisjoint({"missing_same_sentence_bridge_evidence"})

    separated = audit_document(
        {
            "id": speed_spec["id"],
            "text": (
                "A hand kept pushing the toy cart. "
                "The toy cart was accelerating. "
                "Its speed increased and it went faster."
            ),
        },
        spec=speed_spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert "missing_same_sentence_bridge_evidence" in {
        flag["code"] for flag in separated["audit_flags"]
    }

    contrast = audit_document(
        {
            "id": speed_spec["id"],
            "text": (
                "A hand kept pushing the toy cart. "
                "The toy cart was accelerating as its speed increased and it went faster, "
                "then the speed decreased."
            ),
        },
        spec=speed_spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert "bridge_contrast_evidence_violation" in {
        flag["code"] for flag in contrast["audit_flags"]
    }

    negated_rotation = audit_document(
        {
            "id": slide_spec["id"],
            "text": (
                "The flat-sided bead rested on its flat side. "
                "The flat-sided bead began to slide across the slick tray while "
                "the same flat side stayed down without rotating."
            ),
        },
        spec=slide_spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert {
        flag["code"] for flag in negated_rotation["audit_flags"]
    }.isdisjoint({"bridge_contrast_evidence_violation"})

    positive_rotation = audit_document(
        {
            "id": slide_spec["id"],
            "text": (
                "The flat-sided bead rested on its flat side. "
                "The flat-sided bead began to slide across the slick tray while "
                "the same flat side stayed down and it was rotating."
            ),
        },
        spec=slide_spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert "bridge_contrast_evidence_violation" in {
        flag["code"] for flag in positive_rotation["audit_flags"]
    }

    local_slip_slide = audit_document(
        {
            "id": slipped_spec["id"],
            "text": (
                "Ice made the floor patch slippery. "
                "The toy shoe slipped on the icy patch and slid forward across the icy patch with its sole down."
            ),
        },
        spec=slipped_spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert {
        flag["code"] for flag in local_slip_slide["audit_flags"]
    }.isdisjoint({"missing_same_sentence_bridge_evidence"})

    comma_slip_slide = audit_document(
        {
            "id": slipped_spec["id"],
            "text": (
                "Ice made the floor patch slippery. "
                "The toy shoe slipped, sliding forward across the icy patch with its sole down."
            ),
        },
        spec=slipped_spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert {
        flag["code"] for flag in comma_slip_slide["audit_flags"]
    }.isdisjoint({"missing_same_sentence_bridge_evidence"})

    icy_slip_slide = audit_document(
        {
            "id": slipped_spec["id"],
            "text": (
                "The toy shoe slipped and slid forward across the icy patch with its sole down."
            ),
        },
        spec=slipped_spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert {
        flag["code"] for flag in icy_slip_slide["audit_flags"]
    }.isdisjoint({"missing_physical_cause", "missing_same_sentence_bridge_evidence"})

    wet_tile_slips = audit_document(
        {
            "id": slips_spec["id"],
            "text": (
                "Water made the tile slippery. "
                "The button slips across the wet slippery tile, sliding with the top side still on top."
            ),
        },
        spec=slips_spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert {
        flag["code"] for flag in wet_tile_slips["audit_flags"]
    }.isdisjoint({"missing_expected_outcome", "missing_same_sentence_bridge_evidence"})

    pure_slip = audit_document(
        {
            "id": slipped_spec["id"],
            "text": (
                "Ice made the floor patch slippery. "
                "The toy shoe slipped forward across the icy patch with its sole down."
            ),
        },
        spec=slipped_spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert "missing_same_sentence_bridge_evidence" in {
        flag["code"] for flag in pure_slip["audit_flags"]
    }

    separated_slip_slide = audit_document(
        {
            "id": slipped_spec["id"],
            "text": (
                "Ice made the floor patch slippery. "
                "The toy shoe slipped on the icy patch. "
                "It slid forward across the icy patch with its sole down."
            ),
        },
        spec=slipped_spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert "missing_same_sentence_bridge_evidence" in {
        flag["code"] for flag in separated_slip_slide["audit_flags"]
    }


def test_physical_dynamics_ewok_micro_grow_audit_accepts_starting_state_and_variants():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_EWOK_MICRO_500_CONFIG_PATH)
    specs = build_scenario_specs(config)
    sponge_grow = next(
        spec
        for spec in specs
        if spec["scenario_id"] == "sponge_grows_when_soaked"
        and spec["lexical_target"] == "grow"
    )
    foam_pellet_grow = next(
        spec
        for spec in specs
        if spec["scenario_id"] == "foam_pellet_grows_when_warmed"
        and spec["lexical_target"] == "growing"
    )
    balloon_grow = next(
        spec
        for spec in specs
        if spec["scenario_id"] == "balloon_grows_when_inflated"
        and spec["lexical_target"] == "growing"
    )

    sponge = audit_document(
        {
            "id": sponge_grow["id"],
            "text": (
                "Water soaked into the dry sponge, causing it to grow by "
                "swelling past the pencil outline."
            ),
        },
        spec=sponge_grow,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert "bridge_contrast_evidence_violation" not in {
        flag["code"] for flag in sponge["audit_flags"]
    }

    foam = audit_document(
        {
            "id": foam_pellet_grow["id"],
            "text": (
                "Warmth expanded the foam pellet on the tray. "
                "The pellet was growing, expanding to the larger diameter mark."
            ),
        },
        spec=foam_pellet_grow,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert {
        flag["code"] for flag in foam["audit_flags"]
    }.isdisjoint({"missing_physical_cause", "physical_shortcut_only"})

    balloon = audit_document(
        {
            "id": balloon_grow["id"],
            "text": (
                "Air filled the rubber balloon, making it keep growing until "
                "it pressed against the wide measuring ring."
            ),
        },
        spec=balloon_grow,
        concept_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )
    assert "missing_same_sentence_bridge_evidence" not in {
        flag["code"] for flag in balloon["audit_flags"]
    }


def test_physical_dynamics_mock_uses_slim_render_fields():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_W2V_SHORTCUTS_200_CONFIG_PATH)
    specs = build_scenario_specs(config)[:2]

    documents = render_documents(
        config=config,
        specs=specs,
        generator=DeterministicConceptCurriculumMockGenerator(),
    )

    assert len(documents) == 2
    assert "soup was stired" not in " ".join(row["text"].lower() for row in documents)
    assert all(spec["object"] in document["text"] for spec, document in zip(specs, documents))


def test_physical_dynamics_scenario_expansions_target_repetitive_concepts():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_SCENARIO_EXPANSIONS_CONFIG_PATH)
    families = config["physical_dynamics_profile"]["families"]
    scenarios = [
        scenario
        for family in families
        for scenario in family["scenarios"]
    ]

    assert len(scenarios) == 116
    assert Counter(scenario["expected_process"] for scenario in scenarios) == {
        "falling": 9,
        "rising": 9,
        "sinking": 11,
        "floating": 8,
        "accelerating": 16,
        "slowing down": 16,
        "rolling": 4,
        "sliding": 4,
        "spilled": 3,
        "drained": 3,
        "leaked": 3,
        "overflowed": 3,
        "flowed downward": 3,
        "lifted": 3,
        "soaked up": 3,
        "compressed": 3,
        "stretched": 3,
        "cracked": 3,
        "dented": 3,
        "scattered": 3,
        "stuck": 3,
    }
    assert all(scenario["required_cause_terms"] for scenario in scenarios)
    assert all(scenario["required_process_terms"] for scenario in scenarios)
    assert all(scenario["required_outcome_terms"] for scenario in scenarios)
    speed_family = next(
        family
        for family in families
        if family["family_id"] == "speed_change_language_expansion"
    )
    speed_scenarios = speed_family["scenarios"]
    assert len(speed_scenarios) == 8
    assert Counter(scenario["expected_process"] for scenario in speed_scenarios) == {
        "accelerating": 4,
        "slowing down": 4,
    }
    assert Counter(scenario["directional_polarity"] for scenario in speed_scenarios) == {
        "speed_increase": 4,
        "speed_decrease": 4,
    }
    speed_payload = json.dumps(speed_family).lower()
    assert "increasing speed" in speed_payload
    assert "decreasing speed" in speed_payload
    assert "losing speed" in speed_payload
    assert "speeding up" in speed_payload
    speed_quantity_family = next(
        family
        for family in families
        if family["family_id"] == "speed_quantity_change_cues"
    )
    speed_quantity_scenarios = speed_quantity_family["scenarios"]
    assert len(speed_quantity_scenarios) == 8
    assert Counter(
        scenario["expected_process"] for scenario in speed_quantity_scenarios
    ) == {
        "accelerating": 4,
        "slowing down": 4,
    }
    assert Counter(
        scenario["directional_polarity"] for scenario in speed_quantity_scenarios
    ) == {
        "speed_increase": 4,
        "speed_decrease": 4,
    }
    assert all(
        scenario["measurement_frame"] for scenario in speed_quantity_scenarios
    )
    assert all(
        scenario["initial_speed_evidence"] for scenario in speed_quantity_scenarios
    )
    assert all(
        scenario["later_speed_evidence"] for scenario in speed_quantity_scenarios
    )
    assert all(scenario["quantity_change"] for scenario in speed_quantity_scenarios)
    assert all(scenario["rate_evidence"] for scenario in speed_quantity_scenarios)
    speed_quantity_payload = json.dumps(speed_quantity_family).lower()
    assert "per second" in speed_quantity_payload
    assert "per beat" in speed_quantity_payload
    assert "distance per" in speed_quantity_payload
    assert "speed increased" in speed_quantity_payload
    assert "speed decreased" in speed_quantity_payload
    speed_endpoint_family = next(
        family
        for family in families
        if family["family_id"] == "speed_endpoint_tendency_cues"
    )
    speed_endpoint_scenarios = speed_endpoint_family["scenarios"]
    assert len(speed_endpoint_scenarios) == 8
    assert Counter(
        scenario["expected_process"] for scenario in speed_endpoint_scenarios
    ) == {
        "accelerating": 4,
        "slowing down": 4,
    }
    assert Counter(
        scenario["directional_polarity"] for scenario in speed_endpoint_scenarios
    ) == {
        "speed_increase": 4,
        "speed_decrease": 4,
    }
    assert all(scenario["endpoint"] for scenario in speed_endpoint_scenarios)
    assert all(
        scenario["endpoint_tendency"] for scenario in speed_endpoint_scenarios
    )
    assert all(scenario["endpoint_risk"] for scenario in speed_endpoint_scenarios)
    assert all(
        scenario["stopping_evidence"] for scenario in speed_endpoint_scenarios
    )
    speed_endpoint_payload = json.dumps(speed_endpoint_family).lower()
    assert "crash" in speed_endpoint_payload
    assert "hit" in speed_endpoint_payload
    assert "overshoot" in speed_endpoint_payload
    assert "nearly stopped" in speed_endpoint_payload
    assert "stop short" in speed_endpoint_payload
    speed_bridge_family = next(
        family
        for family in families
        if family["family_id"] == "speed_rate_definition_bridge"
    )
    speed_bridge_scenarios = speed_bridge_family["scenarios"]
    assert len(speed_bridge_scenarios) == 8
    assert Counter(
        scenario["expected_process"] for scenario in speed_bridge_scenarios
    ) == {
        "accelerating": 4,
        "slowing down": 4,
    }
    assert Counter(
        scenario["directional_polarity"] for scenario in speed_bridge_scenarios
    ) == {
        "speed_increase": 4,
        "speed_decrease": 4,
    }
    assert all(
        scenario["render_template"] == "speed_rate_bridge_5_sentence"
        for scenario in speed_bridge_scenarios
    )
    assert all(scenario["template_sequence"] for scenario in speed_bridge_scenarios)
    assert all(
        scenario["speed_definition_bridge"] for scenario in speed_bridge_scenarios
    )
    assert all(
        scenario["speed_process_bridge"] for scenario in speed_bridge_scenarios
    )
    speed_bridge_payload = json.dumps(speed_bridge_family).lower()
    assert "same amount of time" in speed_bridge_payload
    assert "speed increased" in speed_bridge_payload
    assert "speed decreased" in speed_bridge_payload
    assert "speed_rate_bridge_5_sentence" in speed_bridge_payload
    fall_rise_family = next(
        family
        for family in families
        if family["family_id"] == "fall_rise_direction_language_expansion"
    )
    fall_rise_scenarios = fall_rise_family["scenarios"]
    assert len(fall_rise_scenarios) == 8
    assert Counter(scenario["expected_process"] for scenario in fall_rise_scenarios) == {
        "falling": 4,
        "rising": 4,
    }
    assert Counter(
        scenario["directional_polarity"] for scenario in fall_rise_scenarios
    ) == {
        "downward": 4,
        "upward": 4,
    }
    fall_rise_payload = json.dumps(fall_rise_family).lower()
    assert "moving down" in fall_rise_payload
    assert "moving up" in fall_rise_payload
    assert "getting closer" not in fall_rise_payload
    assert "altitude" not in fall_rise_payload
    vertical_family = next(
        family
        for family in families
        if family["family_id"] == "vertical_relation_language_expansion"
    )
    vertical_scenarios = vertical_family["scenarios"]
    assert len(vertical_scenarios) == 18
    assert Counter(scenario["expected_process"] for scenario in vertical_scenarios) == {
        "falling": 5,
        "rising": 5,
        "sinking": 4,
        "floating": 4,
        }
    assert Counter(
        scenario["directional_polarity"] for scenario in vertical_scenarios
    ) == {
        "downward": 5,
        "upward": 5,
        "downward_in_medium": 4,
        "same_level": 4,
    }
    assert Counter(
        scenario["reference_frame"] for scenario in vertical_scenarios
    ) == {
        "earth_space": 2,
        "clouds_ground": 1,
        "ceiling_floor": 2,
        "tree_branch_ground": 1,
        "sky_ground": 2,
        "roofline_sky": 1,
        "skyline_ground": 1,
        "fluid_height": 3,
        "water_surface": 4,
        "liquid_surface": 1,
    }
    assert all(scenario["vertical_relation_language"] for scenario in vertical_scenarios)
    assert all(scenario["directional_polarity"] for scenario in vertical_scenarios)
    vertical_fall_rise_scenarios = [
        scenario
        for scenario in vertical_scenarios
        if scenario["expected_process"] in {"falling", "rising"}
    ]
    assert Counter(scenario["landmark"] for scenario in vertical_fall_rise_scenarios) == {
        "ground": 1,
        "clouds": 1,
        "sky": 2,
        "floor": 1,
        "tree branch": 1,
        "stars": 1,
        "roofline": 1,
        "ceiling": 1,
        "skyline": 1,
    }
    assert all(
        scenario["initial_relation_to_landmark"]
        for scenario in vertical_fall_rise_scenarios
    )
    assert all(
        scenario["later_relation_to_landmark"]
        for scenario in vertical_fall_rise_scenarios
    )
    assert all(scenario["mechanism"] for scenario in vertical_fall_rise_scenarios)
    vertical_payload = json.dumps(vertical_family).lower()
    assert "outer space" in vertical_payload
    assert "earth" in vertical_payload
    assert "sky" in vertical_payload
    assert "stars" in vertical_payload
    assert "getting closer" not in vertical_payload
    assert "altitude" not in vertical_payload
    surface_cue_family = next(
        family
        for family in families
        if family["family_id"] == "sink_float_surface_cues"
    )
    surface_cue_scenarios = surface_cue_family["scenarios"]
    assert len(surface_cue_scenarios) == 8
    assert Counter(scenario["expected_process"] for scenario in surface_cue_scenarios) == {
        "sinking": 4,
        "floating": 4,
    }
    assert Counter(
        scenario["directional_polarity"] for scenario in surface_cue_scenarios
    ) == {
        "downward_in_medium": 4,
        "same_level": 4,
    }
    surface_cue_payload = json.dumps(surface_cue_family).lower()
    assert "below the waterline" in surface_cue_payload
    assert "underwater" in surface_cue_payload
    assert "disappeared from view" in surface_cue_payload
    assert "stayed visible" in surface_cue_payload
    assert "partly above the surface" in surface_cue_payload
    assert "football" not in surface_cue_payload
    assert "volleyball" not in surface_cue_payload
    roll_slide_surface_family = next(
        family
        for family in families
        if family["family_id"] == "roll_slide_surface_rotation_cues"
    )
    roll_slide_surface_scenarios = roll_slide_surface_family["scenarios"]
    assert len(roll_slide_surface_scenarios) == 8
    assert Counter(
        scenario["expected_process"] for scenario in roll_slide_surface_scenarios
    ) == {
        "rolling": 4,
        "sliding": 4,
    }
    assert all(
        scenario["surface_friction"] for scenario in roll_slide_surface_scenarios
    )
    assert all(
        scenario["rotation_evidence"] for scenario in roll_slide_surface_scenarios
    )
    roll_slide_payload = json.dumps(roll_slide_surface_family).lower()
    assert "firm flat floor" in roll_slide_payload
    assert "slick tabletop" in roll_slide_payload
    assert "same face upward" in roll_slide_payload
    assert "speck" not in roll_slide_payload
    assert "top of the" not in roll_slide_payload


def test_physical_dynamics_canonical_inventory_covers_current_process_labels():
    inventory = load_pipeline_config(PHYSICAL_DYNAMICS_CANONICAL_INVENTORY_PATH)
    concept_ids = {
        str(concept["concept_id"])
        for concept in inventory["canonical_concepts"]
    }
    surface_to_concept = dict(inventory["surface_to_concept"])
    config_paths = [
        PHYSICAL_DYNAMICS_CONFIG_PATH,
        PHYSICAL_DYNAMICS_CORE_80_CONFIG_PATH,
        PHYSICAL_DYNAMICS_SINK_FLOAT_80_CONFIG_PATH,
        PHYSICAL_DYNAMICS_FALL_RISE_80_CONFIG_PATH,
        PHYSICAL_DYNAMICS_LANDMARK_DIRECTION_60_CONFIG_PATH,
        PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_140_CONFIG_PATH,
        PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_CANONICAL_108_CONFIG_PATH,
        PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_PHYSICAL_120_CONFIG_PATH,
        PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_BROAD_200_CONFIG_PATH,
        PHYSICAL_DYNAMICS_EWOK_MICRO_500_CONFIG_PATH,
    ]

    assert len(concept_ids) == len(inventory["canonical_concepts"])
    assert set(surface_to_concept.values()) <= concept_ids
    assert inventory["do_not_map_by_surface_alone"] == [
        "cleared",
        "crossed",
        "passed",
        "went by",
    ]

    missing_expected = set()
    missing_contrast = set()
    for config_path in config_paths:
        specs = build_scenario_specs(load_pipeline_config(config_path))
        for spec in specs:
            expected_process = str(spec.get("expected_process") or "").strip()
            contrast_process = str(spec.get("contrast_process") or "").strip()
            if expected_process and expected_process not in surface_to_concept:
                missing_expected.add(expected_process)
            if contrast_process and contrast_process not in surface_to_concept:
                missing_contrast.add(contrast_process)

    assert missing_expected == set()
    assert missing_contrast == set()


def test_physical_dynamics_roll_slide_v2_is_balanced_and_pose_aware():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CONFIG_PATH)
    roll_slide = next(
        family
        for family in config["physical_dynamics_profile"]["families"]
        if family["family_id"] == "roll_slide"
    )

    scenarios = roll_slide["scenarios"]
    axis_fields = {
        "object_shape",
        "contact_surface",
        "surface_friction",
        "rotation_allowed",
        "rotation_evidence",
        "motion_pattern",
    }
    assert roll_slide["stage"] == "v2"
    assert len(scenarios) == 8
    assert Counter(scenario["expected_process"] for scenario in scenarios) == {
        "rolling": 4,
        "sliding": 4,
    }
    assert Counter(scenario["object_type"] for scenario in scenarios)["person_body"] == 4
    assert all(axis_fields <= set(scenario) for scenario in scenarios)
    assert all(
        scenario.get("body_pose")
        for scenario in scenarios
        if scenario["object_type"] == "person_body"
    )

    v2_config = deepcopy(config)
    v2_config["coverage_packs"] = [
        {"id": "roll_slide_v2_probe", "document_counts": {"roll_slide": 8}}
    ]
    specs = build_scenario_specs(v2_config)

    assert len(specs) == 8
    assert {spec["family"] for spec in specs} == {"roll_slide"}
    assert Counter(spec["expected_process"] for spec in specs) == {
        "rolling": 4,
        "sliding": 4,
    }
    assert Counter(spec["object_type"] for spec in specs)["person_body"] == 4
    assert all(axis_fields <= set(spec) for spec in specs)
    assert all(spec["rotation_evidence"] for spec in specs)


def test_physical_dynamics_prompt_states_domain_contract():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CONFIG_PATH)
    specs = build_scenario_specs(config)

    prompt = build_physical_dynamics_render_prompt(
        specs=specs[:4],
        physical_dynamics_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )

    assert (
        "Can a model connect evidence about directional physical change, motion, "
        "and state/process change over time to the right event/process concept?"
    ) in prompt
    assert "preserve the cause/process/outcome structure" in prompt
    assert "cause evidence, the expected process, and the outcome evidence" in prompt
    assert "path_shape" in prompt
    assert "render_template" in prompt
    assert "template_sequence" in prompt
    assert "start_position, end_position, direction_evidence" in prompt
    assert "vertical_relation_language, directional_polarity, measurement_frame" in prompt
    assert "initial_speed_evidence, later_speed_evidence" in prompt
    assert "quantity_change, rate_evidence, speed_geometry, endpoint" in prompt
    assert "endpoint_tendency, endpoint_risk, stopping_evidence" in prompt
    assert "for speed_change specs with speed_geometry" in prompt
    assert "initial_rate, later_rate, same_time_comparison" in prompt
    assert "definition_bridge, and process_bridge" in prompt
    assert "distance-over-time evidence" in prompt
    assert 'render_template="speed_rate_bridge_5_sentence"' in prompt
    assert "template_sequence order exactly" in prompt
    assert "landmark, initial_relation_to_landmark, later_relation_to_landmark" in prompt
    assert "For landmark_direction specs" in prompt
    assert "direction_neutral verbs such as passed, crossed, or went by" in prompt
    assert "preferred_process_terms" in prompt
    assert "use one natural form from those lists" in prompt
    assert "active event sentence where the object or person is the subject" in prompt
    assert "Contrast-verbalization policy" in prompt
    assert "roll/slide, swing/spin, drift, and curve/straight are v2 targets" in prompt
    assert "Follow each spec's discourse style as a perspective or genre" in prompt
    assert "a child noticing an everyday event" in prompt
    assert "a short simple wiki explanation" in prompt


def test_physical_dynamics_prompt_includes_assigned_lexical_targets():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_SINK_FLOAT_80_CONFIG_PATH)
    specs = [
        spec
        for spec in build_scenario_specs(config)
        if spec["scenario_id"] == "metal_key_sank_in_water"
    ]

    prompt = build_physical_dynamics_render_prompt(
        specs=specs,
        physical_dynamics_profile=config["physical_dynamics_profile"],
        constraints=config["document_constraints"],
    )

    assert "when a spec includes lexical_target" in prompt
    assert "use that exact lexical_target naturally" in prompt
    assert "do not choose a different preferred form" in prompt
    assert '"lexical_target": "sink"' in prompt
    assert '"lexical_target": "sinks"' in prompt
    assert '"lexical_target": "sank"' in prompt
    assert '"lexical_target": "sinking"' in prompt


def test_physical_dynamics_audit_checks_cause_process_and_shortcuts():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CONFIG_PATH)
    spec = next(
        spec
        for spec in build_scenario_specs(config)
        if spec["scenario_id"] == "support_removed_block_falls"
    )
    constraints = {"min_words": 0, "max_words": 200, "audit_banned_terms": []}

    good = audit_document(
        {
            "text": (
                "The wood block moved because the support was removed. "
                "The block fell instead of staying in place, and the final clue was lower on the floor."
            )
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=constraints,
    )
    missing_cause = audit_document(
        {
            "text": (
                "The wood block fell instead of staying in place, and the final clue was lower on the floor."
            )
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=constraints,
    )
    missing_process = audit_document(
        {
            "text": (
                "The wood block changed because the support was removed, and the final clue was lower on the floor."
            )
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=constraints,
    )
    mismatch = audit_document(
        {
            "text": (
                "The wood block changed because the support was removed, but the block stayed in place. "
                "The final clue was lower on the floor."
            )
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=constraints,
    )
    shortcut_only = audit_document(
        {"text": "The wood block fell, and the final clue was lower on the floor."},
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=constraints,
    )

    assert good["audit_status"] == "pass"
    assert any(flag["code"] == "missing_physical_cause" for flag in missing_cause["audit_flags"])
    assert any(flag["code"] == "missing_expected_process" for flag in missing_process["audit_flags"])
    assert any(flag["code"] == "physical_process_mismatch" for flag in mismatch["audit_flags"])
    assert any(flag["code"] == "physical_shortcut_only" for flag in shortcut_only["audit_flags"])


def test_physical_dynamics_audit_warns_on_missing_speed_geometry():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_BROAD_200_CONFIG_PATH)
    spec = next(
        spec
        for spec in build_scenario_specs(config)
        if spec["family"] == "speed_quantity_change_cues"
        and spec["canonical_concept_id"] == "speed_up"
    )
    speed_geometry = spec["speed_geometry"]
    constraints = {"min_words": 0, "max_words": 200, "audit_banned_terms": []}

    good = audit_document(
        {
            "text": (
                f"{spec['cause_evidence']}. "
                f"{speed_geometry['initial_rate']}. "
                f"{speed_geometry['later_rate']}. "
                f"{speed_geometry['same_time_comparison']}. "
                f"{speed_geometry['definition_bridge']}. "
                f"{speed_geometry['process_bridge']}. "
                f"The cart was {spec['lexical_target']}. "
                f"{spec['outcome_evidence']}."
            )
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=constraints,
    )
    missing_geometry = audit_document(
        {
            "text": (
                f"{spec['cause_evidence']}. "
                f"The cart was {spec['lexical_target']}. "
                f"{spec['outcome_evidence']}."
            )
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=constraints,
    )

    assert not any(flag["code"] == "missing_speed_geometry" for flag in good["audit_flags"])
    assert any(
        flag["code"] == "missing_speed_geometry"
        for flag in missing_geometry["audit_flags"]
    )
    assert missing_geometry["audit_status"] == "warn"

    bridge_spec = next(
        spec
        for spec in build_scenario_specs(config)
        if spec["family"] == "speed_rate_definition_bridge"
        and spec["canonical_concept_id"] == "speed_up"
    )
    bridge_missing_geometry = audit_document(
        {
            "text": (
                f"{bridge_spec['cause_evidence']}. "
                f"The cart was {bridge_spec['expected_process']}. "
                f"{bridge_spec['outcome_evidence']}."
            )
        },
        spec=bridge_spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=constraints,
    )

    assert any(
        flag["code"] == "missing_speed_geometry" and flag["severity"] == "error"
        for flag in bridge_missing_geometry["audit_flags"]
    )
    assert bridge_missing_geometry["audit_status"] == "fail"


def test_physical_dynamics_audit_accepts_process_and_outcome_variants():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CONFIG_PATH)
    specs_by_scenario = {
        spec["scenario_id"]: spec
        for spec in build_scenario_specs(
            {
                **config,
                "coverage_packs": [
                    {"id": "roll_slide_v2_probe", "document_counts": {"roll_slide": 8}}
                ],
            }
        )
    }
    constraints = {"min_words": 0, "max_words": 200, "audit_banned_terms": []}

    examples = {
        "child_rolls_down_hill": (
            "I watched my friend curl up sideways on the grassy hill. "
            "As soon as they leaned into the slope, their body started to turn over and over. "
            "The grass provided enough grip for them to roll downhill instead of just sliding down. "
            "They kept turning until they reached the bottom."
        ),
        "locked_wheel_slides": (
            "Rolling requires an object to turn as it moves. "
            "If a toy wheel has a locked axle, it cannot turn. "
            "When pushed across a smooth floor, the locked wheel slides forward instead of rolling. "
            "The lack of rotation means the wheel skids along the surface."
        ),
        "rubber_ball_rolls_on_mat": (
            "A round rubber ball placed on a grippy mat will roll when pushed. "
            "The friction between the rubber and the mat allows the ball to grip the surface and rotate. "
            "As it moves forward, the ball spins across the mat."
        ),
        "gymnast_rolls_forward_on_mat": (
            "When teaching a forward roll, the gymnast must tuck their body tightly. "
            "This shape allows them to rotate over their shoulders and move forward across the mat. "
            "By keeping the tuck, the gymnast completes the roll while turning over."
        ),
    }

    audits = {
        scenario_id: audit_document(
            {"text": text},
            spec=specs_by_scenario[scenario_id],
            concept_profile=config["physical_dynamics_profile"],
            constraints=constraints,
        )
        for scenario_id, text in examples.items()
    }

    assert {scenario_id: audit["audit_status"] for scenario_id, audit in audits.items()} == {
        scenario_id: "pass" for scenario_id in examples
    }


def test_physical_dynamics_audit_accepts_removed_and_downward_variants():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CORE_80_CONFIG_PATH)
    specs_by_scenario = {
        spec["scenario_id"]: spec for spec in build_scenario_specs(config)
    }
    constraints = {"min_words": 0, "max_words": 200, "audit_banned_terms": []}

    examples = {
        "drain_plug_removed_water_drains": (
            "I pulled the plug out of the basin. "
            "The water drained quickly, swirling downward through the opening. "
            "It moved into the pipe, leaving the basin empty."
        ),
        "brace_removed_sign_collapses": (
            "I watched the folding sign stand on the sidewalk. "
            "Then someone pulled out the brace. "
            "The sign wobbled and collapsed, folding down onto the pavement."
        ),
    }

    audits = {
        scenario_id: audit_document(
            {"text": text},
            spec=specs_by_scenario[scenario_id],
            concept_profile=config["physical_dynamics_profile"],
            constraints=constraints,
        )
        for scenario_id, text in examples.items()
    }

    assert {scenario_id: audit["audit_status"] for scenario_id, audit in audits.items()} == {
        scenario_id: "pass" for scenario_id in examples
    }


def test_physical_dynamics_audit_accepts_speed_change_language_variants():
    config = _apply_document_count_override(
        load_pipeline_config(PHYSICAL_DYNAMICS_CORE_PLUS_LANDMARK_BROAD_200_CONFIG_PATH),
        400,
    )
    specs_by_scenario = {
        spec["scenario_id"]: spec for spec in build_scenario_specs(config)
    }
    constraints = {"min_words": 0, "max_words": 200, "audit_banned_terms": []}

    examples = {
        "marble_accelerates_on_down_slope": (
            "Gravity pulled the glass marble down the smooth slope. "
            "The marble's increasing speed showed the speed-up process as it moved down the track."
        ),
        "cart_slows_on_rough_strip": (
            "The rough rubber strip added friction against the cart wheels. "
            "The cart's decreasing speed showed it moving slower on the rough strip."
        ),
    }

    audits = {}
    for scenario_id, text in examples.items():
        spec = dict(specs_by_scenario[scenario_id])
        spec["lexical_target"] = None
        audits[scenario_id] = audit_document(
            {"text": text},
            spec=spec,
            concept_profile=config["physical_dynamics_profile"],
            constraints=constraints,
        )

    assert all(
        not any(flag["severity"] == "error" for flag in audit["audit_flags"])
        for audit in audits.values()
    )
    assert all(
        any(flag["code"] == "missing_speed_geometry" for flag in audit["audit_flags"])
        for audit in audits.values()
    )


def test_physical_dynamics_audit_enforces_assigned_lexical_target():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_SINK_FLOAT_80_CONFIG_PATH)
    spec = next(
        spec
        for spec in build_scenario_specs(config)
        if spec["scenario_id"] == "metal_key_sank_in_water" and spec["lexical_target"] == "sank"
    )
    constraints = {"min_words": 0, "max_words": 200, "audit_banned_terms": []}

    good = audit_document(
        {
            "text": (
                "The metal key was denser than the water. "
                "The key sank through the cup and reached the bottom."
            )
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=constraints,
    )
    wrong_form = audit_document(
        {
            "text": (
                "The metal key was denser than the water. "
                "The key was sinking through the cup and reached the bottom."
            )
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=constraints,
    )

    assert good["audit_status"] == "pass"
    assert any(flag["code"] == "missing_lexical_target" for flag in wrong_form["audit_flags"])


def test_physical_dynamics_audit_can_warn_on_missing_assigned_lexical_target():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_SINK_FLOAT_80_CONFIG_PATH)
    spec = next(
        spec
        for spec in build_scenario_specs(config)
        if spec["scenario_id"] == "metal_key_sank_in_water" and spec["lexical_target"] == "sank"
    )
    constraints = {
        "min_words": 0,
        "max_words": 200,
        "audit_banned_terms": [],
        "missing_lexical_target_severity": "warning",
    }

    audit = audit_document(
        {
            "text": (
                "The metal key was denser than the water. "
                "The key was sinking through the cup and reached the bottom."
            )
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=constraints,
    )

    assert audit["audit_status"] == "warn"
    assert any(
        flag["code"] == "missing_lexical_target" and flag["severity"] == "warning"
        for flag in audit["audit_flags"]
    )


def test_physical_dynamics_audit_flags_metadata_only_contrast_mentions():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CONFIG_PATH)
    spec = next(
        spec
        for spec in build_scenario_specs(config)
        if spec["scenario_id"] == "support_removed_block_falls"
    )
    constraints = {
        "min_words": 0,
        "max_words": 200,
        "audit_banned_terms": [],
        "contrast_verbalization": "metadata_only",
    }

    audit = audit_document(
        {
            "text": (
                "The wood block moved because the support was removed. "
                "The block fell, not stayed in place, and the final clue was lower on the floor."
            )
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=constraints,
    )

    assert any(
        flag["code"] == "contrast_verbalization_violation"
        and flag["severity"] == "error"
        for flag in audit["audit_flags"]
    )


def test_physical_dynamics_audit_accepts_release_variants():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_LANDMARK_DIRECTION_60_CONFIG_PATH)
    spec = next(
        spec
        for spec in build_scenario_specs(config)
        if spec["scenario_id"] == "water_surface_foam_crossed_upward"
    )
    constraints = {"min_words": 0, "max_words": 200, "audit_banned_terms": []}

    audit = audit_document(
        {
            "text": (
                "I pushed the foam square under the water and let go. "
                "Buoyancy pushed it up. "
                "The foam crossed the water surface, going from below the surface to above the surface."
            )
        },
        spec=spec,
        concept_profile=config["physical_dynamics_profile"],
        constraints=constraints,
    )

    assert not any(flag["code"] == "missing_physical_cause" for flag in audit["audit_flags"])


def test_physical_dynamics_corpus_audit_flags_boilerplate_sentences():
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CONFIG_PATH)
    base_spec = next(
        spec
        for spec in build_scenario_specs(config)
        if spec["scenario_id"] == "support_removed_block_falls"
    )
    specs = [
        {**base_spec, "id": "physical_boilerplate_1"},
        {**base_spec, "id": "physical_boilerplate_2"},
    ]
    repeated_sentence = (
        "The useful evidence links the cause, motion, and result instead of relying only on the object name."
    )
    documents = [
        {
            "id": specs[0]["id"],
            "text": (
                "The wood block moved because the support was removed. "
                "The wood block fell rather than stayed in place. "
                "The visible outcome was that the block was lower on the floor. "
                f"{repeated_sentence}"
            ),
        },
        {
            "id": specs[1]["id"],
            "text": (
                "After the shelf changed, the wood block moved because the support was removed. "
                "The wood block fell rather than stayed in place. "
                "The visible outcome was that the block was lower on the floor. "
                f"{repeated_sentence}"
            ),
        },
    ]

    audited = audit_documents(
        documents,
        specs_by_id={spec["id"]: spec for spec in specs},
        concept_profile=config["physical_dynamics_profile"],
        constraints={
            "min_words": 0,
            "max_words": 200,
            "audit_banned_terms": [],
            "audit_repeated_sentences": True,
            "max_repeated_sentence_reuse": 1,
            "audit_overused_sentence_frames": True,
            "max_sentence_frame_reuse": 1,
        },
    )

    second_codes = {flag["code"] for flag in audited[1]["audit_flags"]}
    assert audited[0]["audit_status"] == "pass"
    assert audited[1]["audit_status"] == "warn"
    assert "repeated_sentence" in second_codes
    assert "overused_sentence_frame" in second_codes


def test_audit_catches_missing_target_and_banned_terms():
    config = load_pipeline_config(CONFIG_PATH)
    spec = build_scenario_specs(config)[0]

    audit = audit_document(
        {"text": "This benchmark Context1 item talks about soup but never uses the target verb."},
        spec=spec,
        concept_profile=config["concept_profile"],
        constraints=config["document_constraints"],
    )

    assert audit["audit_status"] == "fail"
    assert {flag["code"] for flag in audit["audit_flags"]} >= {
        "missing_target_lexeme",
        "banned_term",
    }


def test_prompt_hides_audit_only_banned_terms():
    config = load_pipeline_config(CONFIG_PATH)
    specs = build_scenario_specs(config)[:1]

    prompt = build_render_prompt(
        specs=specs,
        concept_profile=config["concept_profile"],
        constraints=config["document_constraints"],
    )

    assert "EWoK" not in prompt
    assert "every document must include at least one exact target lexical form" in prompt
    assert "stir, stirs, stirred, or stirring" in prompt
    assert "gases and vapors" in prompt
    assert "loose chunks and aggregates" in prompt
    assert "name 2-3 nearby actions" in prompt
    assert "related terms include mix, mixing, blend" in prompt
    assert "agitate" in prompt

    audit = audit_document(
        {"text": "EWoK should not appear in generated training text, even when soup is stirred."},
        spec=specs[0],
        concept_profile=config["concept_profile"],
        constraints=config["document_constraints"],
    )

    assert audit["audit_status"] == "fail"
    assert any(flag["code"] == "banned_term" for flag in audit["audit_flags"])


def test_concept_map_style_requires_neighbor_connections():
    config = load_pipeline_config(CONFIG_PATH)
    spec = next(spec for spec in build_scenario_specs(config) if spec["style_id"] == "concept_map_paragraph")

    audit = audit_document(
        {
            "text": (
                "Soup can be stirred because the liquid moves around the spoon and becomes even. "
                "The material is loose enough for its parts to trade places inside the bowl. "
                "A hard block would stay together as one object, so the same hand motion would not make it circulate."
            )
        },
        spec=spec,
        concept_profile=config["concept_profile"],
        constraints={"min_words": 0, "max_words": 200, "audit_banned_terms": []},
    )

    assert audit["audit_status"] == "warn"
    assert any(flag["code"] == "weak_concept_map" for flag in audit["audit_flags"])


def test_neighbor_contrast_requires_neighbor_word():
    config = load_pipeline_config(CONFIG_PATH)
    spec = next(
        spec for spec in build_scenario_specs(config) if spec["document_type"] == "neighbor_contrast"
    )

    audit = audit_document(
        {
            "text": (
                "Mashed potatoes can be stirred because the soft paste moves around the spoon. "
                "A wood block stays rigid and does not circulate as a mixture. "
                "The stirred material changes position inside the bowl, while the solid keeps its shape."
            )
        },
        spec=spec,
        concept_profile=config["concept_profile"],
        constraints={"min_words": 0, "max_words": 200, "banned_terms": []},
    )

    assert audit["audit_status"] == "fail"
    assert any(flag["code"] == "missing_neighbor" for flag in audit["audit_flags"])


def test_neighbor_and_concept_map_audit_accept_inflections():
    config = load_pipeline_config(CONFIG_PATH)
    neighbor_spec = next(
        spec
        for spec in build_scenario_specs(config)
        if spec["document_type"] == "neighbor_contrast" and spec["neighbor"] == "pour"
    )
    concept_map_spec = next(spec for spec in build_scenario_specs(config) if spec["style_id"] == "concept_map_paragraph")

    neighbor_audit = audit_document(
        {
            "text": (
                "Syrup can be stirred because it flows around the spoon and blends slowly. "
                "This differs from pouring, which moves the syrup from one cup to another. "
                "A ceramic plate cannot be stirred because the brittle solid does not flow or mix."
            )
        },
        spec=neighbor_spec,
        concept_profile=config["concept_profile"],
        constraints={"min_words": 0, "max_words": 200, "audit_banned_terms": []},
    )
    concept_map_audit = audit_document(
        {
            "text": (
                "Cake batter can be stirred because the thick paste turns over under the spoon. "
                "It is near swirling and shaking, but stirring moves through the material directly. "
                "The batter circulates in place instead of leaving the bowl."
            )
        },
        spec=concept_map_spec,
        concept_profile=config["concept_profile"],
        constraints={"min_words": 0, "max_words": 200, "audit_banned_terms": []},
    )

    assert not any(flag["code"] == "missing_neighbor" for flag in neighbor_audit["audit_flags"])
    assert not any(flag["code"] == "weak_concept_map" for flag in concept_map_audit["audit_flags"])


def test_mock_pipeline_writes_inspectable_outputs(tmp_path):
    config = load_pipeline_config(CONFIG_PATH)
    config["output_dir"] = str(tmp_path / "stir_pilot")

    manifest = run_concept_curriculum_pipeline(
        config=config,
        generator=DeterministicConceptCurriculumMockGenerator(),
        model_id="mock-fireworks",
        output_dir=config["output_dir"],
        config_path=CONFIG_PATH,
    )

    assert manifest["counts"]["scenario_plans"] == 20
    assert manifest["counts"]["documents_draft"] == 20
    assert manifest["counts"]["documents_audited"] == 20
    assert manifest["counts"]["train_documents"] == 20
    assert manifest["concept_id"] == "stir"

    scenario_rows = read_jsonl(manifest["paths"]["scenario_plans"])
    audited_rows = read_jsonl(manifest["paths"]["documents_audited"])
    train_text = Path(manifest["paths"]["train"]).read_text(encoding="utf-8")

    assert len(scenario_rows) == 20
    assert len(audited_rows) == 20
    assert "stir" in train_text.lower()
    assert all(row["metadata"]["concept_id"] == "stir" for row in audited_rows)
    assert {row["metadata"]["style_id"] for row in audited_rows} == {
        "concept_map_paragraph",
        "observation_log",
        "child_science_explanation",
        "everyday_narrative",
    }
    assert {row["metadata"]["signal_order_id"] for row in audited_rows} == set(
        EXPECTED_SIGNAL_ORDER_COUNTS_20
    )
    assert all(row["audit_status"] in {"pass", "warn"} for row in audited_rows)


def test_mock_pour_pipeline_writes_pour_outputs(tmp_path):
    config = load_pipeline_config(POUR_CONFIG_PATH)
    config["output_dir"] = str(tmp_path / "pour_pilot")

    manifest = run_concept_curriculum_pipeline(
        config=config,
        generator=DeterministicConceptCurriculumMockGenerator(),
        model_id="mock-fireworks",
        output_dir=config["output_dir"],
        config_path=POUR_CONFIG_PATH,
    )

    assert manifest["counts"]["scenario_plans"] == 20
    assert manifest["counts"]["train_documents"] == 20
    assert manifest["concept_id"] == "pour"

    train_text = Path(manifest["paths"]["train"]).read_text(encoding="utf-8").lower()
    assert "pour" in train_text
    assert "stirred" not in train_text


def test_mock_material_affordance_pipeline_writes_anchor_outputs(tmp_path):
    config = load_pipeline_config(MATERIAL_AFFORDANCE_EWOK_CONFIG_PATH)
    expected_scenario_count = sum(config["coverage_packs"][0]["document_counts"].values())
    config["output_dir"] = str(tmp_path / "material_affordance_ewok")

    manifest = run_concept_curriculum_pipeline(
        config=config,
        generator=DeterministicConceptCurriculumMockGenerator(),
        model_id="mock-fireworks",
        output_dir=config["output_dir"],
        config_path=MATERIAL_AFFORDANCE_EWOK_CONFIG_PATH,
    )

    audited_rows = read_jsonl(manifest["paths"]["documents_audited"])
    train_text = Path(manifest["paths"]["train"]).read_text(encoding="utf-8").lower()

    assert manifest["counts"]["scenario_plans"] == expected_scenario_count
    assert manifest["counts"]["train_documents"] == expected_scenario_count
    assert manifest["profile_id"] == "material_affordance_correlations"
    assert all("ewok_anchor" in row["metadata"]["tags"] for row in audited_rows)
    assert any(row["metadata"]["material"] == "twill" for row in audited_rows)
    assert "ewok" not in train_text
    assert all(row["audit_status"] in {"pass", "warn"} for row in audited_rows)


def test_mock_physical_dynamics_pipeline_writes_pilot_outputs(tmp_path):
    config = load_pipeline_config(PHYSICAL_DYNAMICS_CONFIG_PATH)
    config["output_dir"] = str(tmp_path / "physical_dynamics_pilot")

    manifest = run_concept_curriculum_pipeline(
        config=config,
        generator=DeterministicConceptCurriculumMockGenerator(),
        model_id="mock-fireworks",
        output_dir=config["output_dir"],
        config_path=PHYSICAL_DYNAMICS_CONFIG_PATH,
    )

    scenario_rows = read_jsonl(manifest["paths"]["scenario_plans"])
    audited_rows = read_jsonl(manifest["paths"]["documents_audited"])
    train_text = Path(manifest["paths"]["train"]).read_text(encoding="utf-8").lower()

    assert manifest["counts"]["scenario_plans"] == 20
    assert manifest["counts"]["documents_audited"] == 20
    assert manifest["counts"]["train_documents"] == 20
    assert manifest["profile_id"] == "physical_dynamics"
    assert len(scenario_rows) == 20
    assert {row["family"] for row in scenario_rows} == {
        "directional_motion",
        "support_stability",
        "force_speed",
        "contact_collision",
        "containment_flow",
    }
    assert all(row["metadata"]["path_shape"] for row in audited_rows)
    assert all(row["metadata"]["expected_process"] for row in audited_rows)
    assert all(row["audit_status"] in {"pass", "warn"} for row in audited_rows)
    assert "benchmark" not in train_text
    assert "expected process" not in train_text


def test_standalone_audit_cli_reaudits_existing_documents(tmp_path):
    config = load_pipeline_config(CONFIG_PATH)
    config["output_dir"] = str(tmp_path / "stir_pilot")
    manifest = run_concept_curriculum_pipeline(
        config=config,
        generator=DeterministicConceptCurriculumMockGenerator(),
        model_id="mock-fireworks",
        output_dir=config["output_dir"],
        config_path=CONFIG_PATH,
    )
    output_path = tmp_path / "reaudited.jsonl"
    report_path = tmp_path / "reaudit_report.json"
    train_path = tmp_path / "train_reaudited.txt"

    audit_main(
        [
            "--config",
            str(CONFIG_PATH),
            "--input",
            manifest["paths"]["documents_draft"],
            "--scenario-plans",
            manifest["paths"]["scenario_plans"],
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--train-output",
            str(train_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"]["fail"] == 0
    assert report["counts"]["pass"] + report["counts"]["warn"] == 20
    assert len(read_jsonl(output_path)) == 20
    assert "stir" in train_path.read_text(encoding="utf-8").lower()
