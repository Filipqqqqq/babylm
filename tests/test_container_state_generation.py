from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from babysteps.container_state import build_container_state_corpus, load_yaml_config
from babysteps.container_state.generation import HELDOUT_RENDERERS, TRAIN_RENDERERS
from babysteps.container_state.world import Operation, World, apply_operations
from babysteps.evaluation.entity_tracking_likelihood import read_entity_tracking_items


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "container_state" / "container_state_v1_seed42.yaml"


def small_config(tmp_path: Path, *, name: str = "container_state_test") -> dict:
    config = load_yaml_config(CONFIG_PATH)
    config["name"] = name
    config["output_dir"] = str(tmp_path / name)
    config["pool_min_words"] = 4_000
    validation = dict(config["validation"])
    validation["items_per_slice"] = 12
    config["validation"] = validation
    return config


def replay(row: dict) -> World:
    initial = World.from_mapping(row["initial_state"])
    operations = tuple(
        Operation(
            kind=operation["kind"],
            item=operation["item"],
            source=operation["source"],
            destination=operation["destination"],
        )
        for operation in row["operations"]
    )
    return apply_operations(initial, operations)


def test_generation_writes_audited_training_and_validation_artifacts(tmp_path: Path) -> None:
    result = build_container_state_corpus(
        small_config(tmp_path),
        base_dir=ROOT,
    )

    assert result.audit_report["counts"]["fail"] == 0
    assert result.audit_report["counts"]["words"] >= 4_000
    assert result.audit_report["operation_count_counts"].keys() == {"0", "1"}
    assert set(result.audit_report["operation_kind_counts"]) == {"add", "move", "remove"}
    assert set(result.audit_report["renderer_counts"]) == set(TRAIN_RENDERERS)
    assert set(result.validation_items) == {
        "iid",
        "lexical_ood",
        "template_ood",
        "two_operations",
    }
    assert all(len(rows) == 12 for rows in result.validation_items.values())

    for key in ("train", "documents", "audit_report", "manifest", "lexicons"):
        assert result.outputs[key].exists()
    assert len(result.outputs["train"].read_text(encoding="utf-8").splitlines()) == len(
        result.documents
    )


def test_generated_documents_replay_and_avoid_benchmark_surface_forms(tmp_path: Path) -> None:
    result = build_container_state_corpus(small_config(tmp_path), base_dir=ROOT)

    for row in result.documents:
        assert replay(row).as_dict() == row["final_state"]
        assert set(row["recap_items"]) == set(
            row["final_state"][row["target_container"]]
        )
        assert row["text"] == row["input_prefix"] + row["recap_completion"]
        assert "Box " not in row["text"]
        assert " contains " not in row["text"]
        assert "\n" not in row["text"]

    assert result.audit_report["lexical_checks"] == {
        "benchmark_object_overlap": [],
        "container_object_overlap": [],
        "train_test_object_overlap": [],
        "train_validation_object_overlap": [],
    }


def test_validation_slices_hold_out_lexicons_templates_and_length(tmp_path: Path) -> None:
    result = build_container_state_corpus(small_config(tmp_path), base_dir=ROOT)

    assert {row["lexicon_split"] for row in result.validation_items["iid"]} == {"train"}
    assert {row["lexicon_split"] for row in result.validation_items["lexical_ood"]} == {
        "validation"
    }
    assert {row["renderer"] for row in result.validation_items["template_ood"]} == set(
        HELDOUT_RENDERERS
    )
    assert {row["numops"] for row in result.validation_items["two_operations"]} == {2}

    train_signatures = {
        json.dumps(row["initial_state"], sort_keys=True)
        for row in result.documents
    }
    validation_signatures = set()
    for rows in result.validation_items.values():
        for row in rows:
            assert len(row["options"]) == 5
            assert len(set(row["options"])) == 5
            assert len(row["distractor_kinds"]) == 4
            signature = json.dumps(row["initial_state"], sort_keys=True)
            assert signature not in train_signatures
            assert signature not in validation_signatures
            validation_signatures.add(signature)

    likelihood_items = read_entity_tracking_items(result.outputs["validation_dir"])
    assert len(likelihood_items) == 48
    assert {item.family for item in likelihood_items} == {
        "iid",
        "lexical_ood",
        "template_ood",
        "two_operations",
    }
    assert all(len(item.options) == 5 for item in likelihood_items)


def test_one_and_two_percent_mix_and_training_configs_keep_the_baseline_budget() -> None:
    one_percent_mix = load_yaml_config(
        ROOT
        / "configs"
        / "data_mixes"
        / "babylm2026_strict_small_replace1pct_container_state_v1_seed42.yaml"
    )
    two_percent_mix = load_yaml_config(
        ROOT
        / "configs"
        / "data_mixes"
        / "babylm2026_strict_small_replace2pct_container_state_v1_seed42.yaml"
    )
    assert one_percent_mix["max_words"] == two_percent_mix["max_words"] == 10_000_000
    assert [source["target_words"] for source in one_percent_mix["sources"]] == [
        9_900_000,
        100_000,
    ]
    assert [source["target_words"] for source in two_percent_mix["sources"]] == [
        9_800_000,
        200_000,
    ]
    assert all(
        mix["sources"][1]["sampling"] == "random_without_replacement_exact_words"
        for mix in (one_percent_mix, two_percent_mix)
    )

    expanded_pool = load_yaml_config(
        ROOT
        / "configs"
        / "container_state"
        / "container_state_v1_pool260k_seed42.yaml"
    )
    assert expanded_pool["pool_min_words"] == 260_000

    baseline = load_yaml_config(
        ROOT
        / "configs"
        / "train"
        / "gpt_bert_baseline_mixed_babylm2026_strict_small_muon.yaml"
    )
    experiment_names = (
        "gpt_bert_mixed_babylm2026_strict_small_replace1pct_container_state_v1_muon.yaml",
        "gpt_bert_mixed_babylm2026_strict_small_replace2pct_container_state_v1_muon.yaml",
    )
    for experiment_name in experiment_names:
        experiment = load_yaml_config(
            ROOT / "configs" / "train" / experiment_name
        )
        assert experiment["model"] == baseline["model"]
        assert experiment["hybrid"] == baseline["hybrid"]
        assert experiment["training"] == baseline["training"]
        assert experiment["logging"] == baseline["logging"]
        assert experiment["checkpointing"] == baseline["checkpointing"]


def test_generation_is_reproducible_for_the_same_seed(tmp_path: Path) -> None:
    first = build_container_state_corpus(
        small_config(tmp_path, name="first"),
        base_dir=ROOT,
    )
    second_config = small_config(tmp_path, name="second")
    second_config["name"] = "first"
    second = build_container_state_corpus(second_config, base_dir=ROOT)

    assert first.documents == second.documents
    assert first.validation_items == second.validation_items
    assert first.audit_report == second.audit_report


def test_container_state_cli_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "synthetic_data_generation"
                / "container_state"
                / "generate_container_state.py"
            ),
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(output_dir),
            "--pool-min-words",
            "800",
            "--validation-items-per-slice",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Container-state corpus:" in completed.stdout
    assert (output_dir / "train.txt").exists()
    assert (output_dir / "validation" / "entity_tracking" / "iid.jsonl").exists()
