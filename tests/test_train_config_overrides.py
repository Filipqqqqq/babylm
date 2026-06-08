from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_train_script_module():
    spec = importlib.util.spec_from_file_location(
        "train_gpt2_small_script",
        ROOT / "scripts" / "train_gpt2_small.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_config_overrides_update_nested_values() -> None:
    train_script = load_train_script_module()
    config = {
        "run": {"name": "base", "seed": 42},
        "training": {"learning_rate": 5e-5},
    }

    train_script.apply_config_overrides(
        config,
        [
            "run.name=bpe8192_lr1e-4_seed43",
            "run.seed=43",
            "training.learning_rate=1e-4",
            "logging.eval_max_batches=null",
        ],
    )

    assert config["run"]["name"] == "bpe8192_lr1e-4_seed43"
    assert config["run"]["seed"] == 43
    assert config["training"]["learning_rate"] == 1e-4
    assert config["logging"]["eval_max_batches"] is None


def test_cli_config_override_rejects_non_mapping_path() -> None:
    train_script = load_train_script_module()
    config = {"training": {"learning_rate": 5e-5}}

    try:
        train_script.apply_config_override(config, "training.learning_rate.value=1e-4")
    except ValueError as exc:
        assert "is not a mapping" in str(exc)
    else:
        raise AssertionError("Expected non-mapping override path to fail.")


def test_training_input_validation_catches_short_context() -> None:
    train_script = load_train_script_module()
    config = {"model": {"context_length": 2, "vocab_size": 10}}
    tokenizer = SimpleNamespace()
    train_dataset = SimpleNamespace(chunks=[[0, 1, 2, 3]])
    val_dataset = SimpleNamespace(chunks=[[0, 1, 2]])

    try:
        train_script.validate_training_inputs(config, tokenizer, train_dataset, val_dataset)
    except ValueError as exc:
        assert "input positions" in str(exc)
    else:
        raise AssertionError("Expected too-short model context to fail.")


def test_training_input_validation_catches_small_vocab() -> None:
    train_script = load_train_script_module()
    config = {"model": {"context_length": 8, "vocab_size": 3}}
    tokenizer = SimpleNamespace()
    train_dataset = SimpleNamespace(chunks=[[0, 1, 3]])
    val_dataset = SimpleNamespace(chunks=[[0, 1, 2]])

    try:
        train_script.validate_training_inputs(config, tokenizer, train_dataset, val_dataset)
    except ValueError as exc:
        assert "model.vocab_size" in str(exc)
    else:
        raise AssertionError("Expected too-small model vocab to fail.")
