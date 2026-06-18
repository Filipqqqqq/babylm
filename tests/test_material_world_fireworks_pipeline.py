from __future__ import annotations

import json
from pathlib import Path

from babysteps.cli.generate_material_world_fireworks import main as generate_main
from babysteps.material_world.audit import read_jsonl
from babysteps.material_world.fireworks_pipeline import (
    DeterministicMaterialWorldMockGenerator,
    build_check_prompt,
    build_render_prompt,
    build_scenario_plan_prompt,
    load_json,
    load_pipeline_config,
    run_fireworks_pipeline,
    select_probe_items,
)


ROOT = Path(__file__).resolve().parents[1]


class MalformedFirstCheckGenerator(DeterministicMaterialWorldMockGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.return_malformed_check = True

    def __call__(self, **kwargs):
        if kwargs.get("stage") == "check_edit" and self.return_malformed_check:
            self.return_malformed_check = False
            return '{"items": [{"id": "mw_probe_0000", "text": "unterminated"}'
        return super().__call__(**kwargs)


def test_probe_selection_returns_expected_counts():
    material_config = load_json(ROOT / "configs" / "material_world" / "material_classes.json")
    object_config = load_json(ROOT / "configs" / "material_world" / "material_objects.json")
    styles = [
        {"style_id": "a"},
        {"style_id": "b"},
        {"style_id": "c"},
        {"style_id": "d"},
        {"style_id": "e"},
    ]

    selected = select_probe_items(
        material_config,
        object_config,
        styles,
        material_class_limit=4,
        object_limit=3,
        style_limit=4,
    )

    assert len(selected["material_classes"]) == 4
    assert len(selected["objects"]) == 3
    assert len(selected["styles"]) == 4
    assert selected["material_classes"][0]["class_id"] == "metals"
    assert selected["objects"][0]["object_id"] == "cup"


def test_mock_pipeline_writes_audit_ready_metadata(tmp_path):
    config = load_pipeline_config(ROOT / "configs" / "material_world" / "fireworks_probe.json")
    config["output_dir"] = str(tmp_path / "probe")
    config["scenario_limit"] = 12

    manifest = run_fireworks_pipeline(
        config=config,
        generator=DeterministicMaterialWorldMockGenerator(),
        model_id="mock-fireworks",
        output_dir=config["output_dir"],
        config_path=ROOT / "configs" / "material_world" / "fireworks_probe.json",
    )

    assert manifest["counts"]["scenario_plans"] == 12
    assert manifest["selected"]["material_classes"] == [
        "metals",
        "brittle_hard_solids",
        "stone_masonry_mineral_solids",
        "rigid_plant_solids",
    ]
    assert manifest["selected"]["objects"] == ["cup", "bottle", "bowl"]
    assert manifest["selected"]["styles"] == [
        "everyday_narrative",
        "comparison_paragraph",
        "child_science_explanation",
        "repair_or_craft_note",
    ]

    rows = read_jsonl(manifest["paths"]["prose_checked"])
    assert len(rows) == 12
    for row in rows:
        assert row["id"]
        assert row["text"]
        metadata = row["metadata"]
        assert metadata["style_id"]
        assert metadata["material"]
        assert metadata["material_class"]
        assert metadata["object"]
        assert isinstance(metadata["dynamics"], list)
        assert "contains_negative_reporting" in metadata

    assert Path(manifest["paths"]["train_audited"]).exists()


def test_mock_pipeline_respects_explicit_stage_batch_sizes(tmp_path):
    config = load_pipeline_config(ROOT / "configs" / "material_world" / "fireworks_probe.json")
    config["output_dir"] = str(tmp_path / "batched_probe")
    config["scenario_limit"] = 10
    config["audit_edit_limit"] = 5
    config["batching"] = {
        "scenario_plans_per_request": 3,
        "render_examples_per_request": 4,
        "check_examples_per_request": 5,
        "audit_edit_examples_per_request": 2,
    }

    manifest = run_fireworks_pipeline(
        config=config,
        generator=DeterministicMaterialWorldMockGenerator(),
        model_id="mock-fireworks",
        output_dir=config["output_dir"],
        config_path=ROOT / "configs" / "material_world" / "fireworks_probe.json",
    )

    assert manifest["counts"]["scenario_plans"] == 10
    assert manifest["counts"]["prose_draft"] == 10
    assert manifest["counts"]["prose_checked"] == 10
    assert manifest["batching"] == config["batching"]
    assert [row["scenario_id"] for row in read_jsonl(manifest["paths"]["scenario_plans"])] == [
        f"mw_probe_{index:04d}" for index in range(10)
    ]
    assert len(read_jsonl(manifest["paths"]["examples_reaudited"])) == 10


def test_mock_pipeline_retries_malformed_check_json(tmp_path):
    config = load_pipeline_config(ROOT / "configs" / "material_world" / "fireworks_probe.json")
    config["output_dir"] = str(tmp_path / "retry_check_probe")
    config["scenario_limit"] = 4
    config["audit_edit_limit"] = 0

    manifest = run_fireworks_pipeline(
        config=config,
        generator=MalformedFirstCheckGenerator(),
        model_id="mock-fireworks",
        output_dir=config["output_dir"],
        config_path=ROOT / "configs" / "material_world" / "fireworks_probe.json",
    )

    assert manifest["counts"]["prose_checked"] == 4
    raw_failures = list((Path(manifest["output_dir"]) / "_raw_model_failures").glob("check_edit_*.txt"))
    assert raw_failures


def test_mock_pipeline_can_target_coverage_set(tmp_path):
    config = load_pipeline_config(ROOT / "configs" / "material_world" / "fireworks_probe.json")
    config["output_dir"] = str(tmp_path / "ewok_coverage_probe")
    config["coverage_set_ids"] = ["ewok_material_properties"]
    config["scenario_limit"] = 6
    config["style_limit"] = 1

    manifest = run_fireworks_pipeline(
        config=config,
        generator=DeterministicMaterialWorldMockGenerator(),
        model_id="mock-fireworks",
        output_dir=config["output_dir"],
        config_path=ROOT / "configs" / "material_world" / "fireworks_probe.json",
    )

    assert manifest["selected"]["coverage_sets"] == ["ewok_material_properties"]
    assert manifest["selected"]["objects"] == [
        "balloon",
        "bin",
        "football",
        "volleyball",
        "wheel",
    ]
    assert "transparent_crystalline_solids" in manifest["selected"]["material_classes"]
    assert "textiles_fibers" in manifest["selected"]["material_classes"]

    scenario_rows = read_jsonl(manifest["paths"]["scenario_plans"])
    assert {row["object_id"] for row in scenario_rows} <= set(manifest["selected"]["objects"])


def test_mock_pipeline_soft_coverage_mix_keeps_general_and_coverage_items(tmp_path):
    config = load_pipeline_config(ROOT / "configs" / "material_world" / "fireworks_probe.json")
    material_config = load_json(ROOT / "configs" / "material_world" / "material_classes.json")
    object_config = load_json(ROOT / "configs" / "material_world" / "material_objects.json")
    config["output_dir"] = str(tmp_path / "ewok_softmix_probe")
    config["coverage_mix"] = {
        "ewok_material_properties": {
            "object_fraction": 0.35,
            "material_fraction": 0.35,
        }
    }
    config["scenario_limit"] = 8
    config["style_limit"] = 1

    manifest = run_fireworks_pipeline(
        config=config,
        generator=DeterministicMaterialWorldMockGenerator(),
        model_id="mock-fireworks",
        output_dir=config["output_dir"],
        config_path=ROOT / "configs" / "material_world" / "fireworks_probe.json",
    )

    assert manifest["selected"]["coverage_sets"] == ["ewok_material_properties"]
    assert manifest["selected"]["coverage_mix"] == {
        "ewok_material_properties": {
            "row_fraction": 0.0,
            "object_fraction": 0.35,
            "material_fraction": 0.35,
            "dynamic_fraction": 0.0,
            "anchor_fraction": 0.0,
        }
    }
    assert manifest["counts"]["coverage_assignments"] == 6
    coverage_report = json.loads(Path(manifest["paths"]["coverage_report"]).read_text(encoding="utf-8"))
    assert coverage_report["assigned"]["ewok_material_properties"]["coverage_roles"] == {
        "ewok_material": 3,
        "ewok_object": 3,
    }
    assert manifest["selected"]["objects"][:3] == ["cup", "bottle", "bowl"]
    assert len(manifest["selected"]["objects"]) == len(object_config["objects"])
    assert "balloon" in manifest["selected"]["objects"]
    assert "wheel" in manifest["selected"]["objects"]
    assert manifest["selected"]["material_classes"][:4] == [
        "metals",
        "brittle_hard_solids",
        "stone_masonry_mineral_solids",
        "rigid_plant_solids",
    ]
    assert len(manifest["selected"]["material_classes"]) == len(material_config["material_classes"])
    assert "textiles_fibers" in manifest["selected"]["material_classes"]
    assert "transparent_crystalline_solids" in manifest["selected"]["material_classes"]


def test_mock_pipeline_assigns_material_dynamics_targets(tmp_path):
    config = load_pipeline_config(ROOT / "configs" / "material_world" / "fireworks_probe.json")
    config["output_dir"] = str(tmp_path / "ewok_dynamic_probe")
    config["coverage_mix"] = {
        "ewok_material_dynamics": {
            "dynamic_fraction": 1.0,
        }
    }
    config["scenario_limit"] = 15
    config["style_limit"] = 1

    manifest = run_fireworks_pipeline(
        config=config,
        generator=DeterministicMaterialWorldMockGenerator(),
        model_id="mock-fireworks",
        output_dir=config["output_dir"],
        config_path=ROOT / "configs" / "material_world" / "fireworks_probe.json",
    )

    coverage_report = json.loads(Path(manifest["paths"]["coverage_report"]).read_text(encoding="utf-8"))
    assert manifest["selected"]["coverage_sets"] == ["ewok_material_dynamics"]
    assert manifest["counts"]["coverage_assignments"] == 15
    assert coverage_report["assigned"]["ewok_material_dynamics"]["coverage_roles"] == {
        "ewok_dynamic": 15
    }
    assert set(coverage_report["assigned"]["ewok_material_dynamics"]["target_dynamic_concepts"]) >= {
        "stir",
        "tap",
        "flap",
        "ripple",
    }
    scenario_rows = read_jsonl(manifest["paths"]["scenario_plans"])
    assert all(row["coverage_domain"] == "material-dynamics" for row in scenario_rows)
    assert all(row["target_dynamic_concept"] in row["dynamics"] for row in scenario_rows)


def test_mock_pipeline_assigns_lexical_anchors_round_robin(tmp_path):
    config = load_pipeline_config(
        ROOT / "configs" / "material_world" / "fireworks_probe_ewok_dyn_anchor_v2.json"
    )
    config["output_dir"] = str(tmp_path / "ewok_dynamic_anchor_probe")
    config["scenario_limit"] = 32
    config["style_limit"] = 1
    config["audit_edit_limit"] = 0

    manifest = run_fireworks_pipeline(
        config=config,
        generator=DeterministicMaterialWorldMockGenerator(),
        model_id="mock-fireworks",
        output_dir=config["output_dir"],
        config_path=ROOT / "configs" / "material_world" / "fireworks_probe_ewok_dyn_anchor_v2.json",
    )

    coverage_report = json.loads(Path(manifest["paths"]["coverage_report"]).read_text(encoding="utf-8"))
    assigned = coverage_report["assigned"]["ewok_material_dynamics_anchor_v2"]
    scenario_rows = read_jsonl(manifest["paths"]["scenario_plans"])
    target_anchor_counts = assigned["target_anchors"]

    assert manifest["selected"]["coverage_mix"] == {
        "ewok_material_dynamics_anchor_v2": {
            "row_fraction": 0.0,
            "object_fraction": 0.0,
            "material_fraction": 0.0,
            "dynamic_fraction": 0.70,
            "anchor_fraction": 1.0,
        }
    }
    assert manifest["counts"]["coverage_assignments"] == 32
    assert sum(target_anchor_counts.values()) == 32
    assert max(target_anchor_counts.values()) - min(target_anchor_counts.values()) <= 1
    assert all(row["target_anchor"] for row in scenario_rows)
    assert all("central participant" in row.get("plan_notes", "") for row in scenario_rows)
    assert sum(1 for row in scenario_rows if row["target_dynamic_concept"]) == 22


def test_anchor_prompts_require_central_participant():
    scenario_prompt = build_scenario_plan_prompt(
        selected_material_classes=[
            {
                "class_id": "textiles_fibers",
                "class_name": "Textiles and fibers",
                "examples": ["twill"],
            }
        ],
        selected_objects=[
            {
                "object_id": "shirt_jacket",
                "object_name": "shirt",
                "useful_events": ["folded"],
            }
        ],
        selected_styles=[{"style_id": "everyday_narrative"}],
        scenario_limit=1,
        coverage_assignments=[{"scenario_id": "mw_probe_0000", "target_anchor": "twill"}],
    )
    render_prompt = build_render_prompt(
        [
            {
                "scenario_id": "mw_probe_0000",
                "discourse_style_id": "everyday_narrative",
                "target_anchor": "twill",
            }
        ]
    )
    check_prompt = build_check_prompt(
        [
            {
                "id": "mw_probe_0000",
                "text": "The twill wrinkles when the shirt is folded.",
                "metadata": {"target_anchor": "twill"},
            }
        ]
    )

    for prompt in [scenario_prompt, render_prompt, check_prompt]:
        assert "central participant" in prompt
        assert "absent" in prompt
        assert "nearby" in prompt


def test_mock_pipeline_balances_selected_object_exposure(tmp_path):
    config = load_pipeline_config(ROOT / "configs" / "material_world" / "fireworks_probe.json")
    config["output_dir"] = str(tmp_path / "balanced_object_probe")
    config["scenario_limit"] = 13
    config["object_limit"] = 5
    config["style_limit"] = 1
    config["selection_seed"] = "object-balance"
    config["object_sampling"] = {
        "mode": "balanced_round_robin",
        "scope": "selected_objects",
    }

    manifest = run_fireworks_pipeline(
        config=config,
        generator=DeterministicMaterialWorldMockGenerator(),
        model_id="mock-fireworks",
        output_dir=config["output_dir"],
        config_path=ROOT / "configs" / "material_world" / "fireworks_probe.json",
    )

    scenario_rows = read_jsonl(manifest["paths"]["scenario_plans"])
    selected_object_ids = set(manifest["selected"]["objects"])
    object_counts = {
        object_id: sum(1 for row in scenario_rows if row["object_id"] == object_id)
        for object_id in selected_object_ids
    }
    assert manifest["selected"]["object_sampling"] == {
        "mode": "balanced_round_robin",
        "scope": "selected_objects",
    }
    assert manifest["counts"]["object_sampling_assignments"] == 13
    assert all(row["object_sampling_mode"] == "balanced_round_robin" for row in scenario_rows)
    assert max(object_counts.values()) - min(object_counts.values()) <= 1

    object_report = json.loads(Path(manifest["paths"]["object_sampling_report"]).read_text(encoding="utf-8"))
    assert object_report["assigned"]["rows"] == 13
    assert object_report["assigned"]["spread"]["spread"] <= 1
    assert set(object_report["assigned"]["target_object_ids"]) == selected_object_ids


def test_mock_pipeline_balances_selected_style_exposure(tmp_path):
    config = load_pipeline_config(ROOT / "configs" / "material_world" / "fireworks_probe.json")
    config["output_dir"] = str(tmp_path / "balanced_style_probe")
    config["scenario_limit"] = 19
    config["style_limit"] = 8
    config["selection_seed"] = "style-balance"
    config["style_sampling"] = {
        "mode": "balanced_round_robin",
        "scope": "selected_styles",
    }

    manifest = run_fireworks_pipeline(
        config=config,
        generator=DeterministicMaterialWorldMockGenerator(),
        model_id="mock-fireworks",
        output_dir=config["output_dir"],
        config_path=ROOT / "configs" / "material_world" / "fireworks_probe.json",
    )

    scenario_rows = read_jsonl(manifest["paths"]["scenario_plans"])
    selected_style_ids = set(manifest["selected"]["styles"])
    style_counts = {
        style_id: sum(1 for row in scenario_rows if row["discourse_style_id"] == style_id)
        for style_id in selected_style_ids
    }
    assert manifest["selected"]["style_sampling"] == {
        "mode": "balanced_round_robin",
        "scope": "selected_styles",
    }
    assert manifest["counts"]["style_sampling_assignments"] == 19
    assert all(row["style_sampling_mode"] == "balanced_round_robin" for row in scenario_rows)
    assert all(row["target_style_id"] == row["discourse_style_id"] for row in scenario_rows)
    assert max(style_counts.values()) - min(style_counts.values()) <= 1

    style_report = json.loads(Path(manifest["paths"]["style_sampling_report"]).read_text(encoding="utf-8"))
    assert style_report["assigned"]["rows"] == 19
    assert style_report["assigned"]["spread"]["spread"] <= 1
    assert set(style_report["assigned"]["target_style_ids"]) == selected_style_ids


def test_mock_pipeline_selection_seed_reorders_soft_mix_reproducibly(tmp_path):
    base_config = load_pipeline_config(ROOT / "configs" / "material_world" / "fireworks_probe.json")
    base_config["coverage_mix"] = {
        "ewok_material_properties": {
            "object_fraction": 0.35,
            "material_fraction": 0.35,
        }
    }
    base_config["scenario_limit"] = 4

    first_config = dict(base_config, output_dir=str(tmp_path / "seed_17_a"), selection_seed="17")
    second_config = dict(base_config, output_dir=str(tmp_path / "seed_17_b"), selection_seed="17")
    third_config = dict(base_config, output_dir=str(tmp_path / "seed_18"), selection_seed="18")

    first = run_fireworks_pipeline(
        config=first_config,
        generator=DeterministicMaterialWorldMockGenerator(),
        model_id="mock-fireworks",
        output_dir=first_config["output_dir"],
        config_path=ROOT / "configs" / "material_world" / "fireworks_probe.json",
    )
    second = run_fireworks_pipeline(
        config=second_config,
        generator=DeterministicMaterialWorldMockGenerator(),
        model_id="mock-fireworks",
        output_dir=second_config["output_dir"],
        config_path=ROOT / "configs" / "material_world" / "fireworks_probe.json",
    )
    third = run_fireworks_pipeline(
        config=third_config,
        generator=DeterministicMaterialWorldMockGenerator(),
        model_id="mock-fireworks",
        output_dir=third_config["output_dir"],
        config_path=ROOT / "configs" / "material_world" / "fireworks_probe.json",
    )

    assert first["selected"]["selection_seed"] == "17"
    assert first["selected"]["objects"] == second["selected"]["objects"]
    assert first["selected"]["material_classes"] == second["selected"]["material_classes"]
    assert first["selected"]["objects"] != third["selected"]["objects"]


def test_cli_mock_fireworks_smoke(tmp_path):
    output_dir = tmp_path / "cli_probe"

    generate_main(["--mock-fireworks", "--output-dir", str(output_dir)])

    assert (output_dir / "scenario_plans.jsonl").exists()
    assert (output_dir / "examples_reaudited.jsonl").exists()
    assert (output_dir / "train_audited.txt").read_text(encoding="utf-8").strip()


def test_cli_mock_fireworks_accepts_coverage_set(tmp_path):
    output_dir = tmp_path / "cli_ewok_probe"

    generate_main(
        [
            "--mock-fireworks",
            "--coverage-set",
            "ewok_material_properties",
            "--scenario-limit",
            "3",
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected"]["coverage_sets"] == ["ewok_material_properties"]
    assert manifest["selected"]["objects"] == [
        "balloon",
        "bin",
        "football",
        "volleyball",
        "wheel",
    ]


def test_cli_mock_fireworks_accepts_soft_coverage_mix(tmp_path):
    output_dir = tmp_path / "cli_ewok_softmix_probe"
    object_config = load_json(ROOT / "configs" / "material_world" / "material_objects.json")

    generate_main(
        [
            "--mock-fireworks",
            "--coverage-set",
            "ewok_material_properties",
            "--coverage-object-fraction",
            "0.4",
            "--coverage-material-fraction",
            "0.25",
            "--seed",
            "softmix-demo",
            "--object-sampling-mode",
            "balanced_round_robin",
            "--style-sampling-mode",
            "balanced_round_robin",
            "--scenario-limit",
            "4",
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected"]["coverage_sets"] == ["ewok_material_properties"]
    assert manifest["selected"]["coverage_mix"] == {
        "ewok_material_properties": {
            "row_fraction": 0.0,
            "object_fraction": 0.4,
            "material_fraction": 0.25,
            "dynamic_fraction": 0.0,
            "anchor_fraction": 0.0,
        }
    }
    assert manifest["selected"]["selection_seed"] == "softmix-demo"
    assert manifest["selected"]["object_sampling"] == {
        "mode": "balanced_round_robin",
        "scope": "selected_objects",
    }
    assert manifest["selected"]["style_sampling"] == {
        "mode": "balanced_round_robin",
        "scope": "selected_styles",
    }
    assert len(manifest["selected"]["objects"]) == len(object_config["objects"])
    assert "cup" in manifest["selected"]["objects"]
    assert "balloon" in manifest["selected"]["objects"]
