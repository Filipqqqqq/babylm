from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from babysteps.cli import build_token_shards as token_shard_builder
from babysteps.cli import train_causal_lm as train_script
from babysteps.training.causal_lm_loop import (
    advance_word_interval,
    validate_every_words,
    word_interval_due,
)

ROOT = Path(__file__).resolve().parents[1]


def load_train_script_module():
    return train_script


def load_token_shard_builder_module():
    return token_shard_builder


def test_env_expansion_supports_shell_style_defaults(monkeypatch) -> None:
    train_script = load_train_script_module()
    monkeypatch.delenv("BABYSTEPS_CACHE_DIR", raising=False)
    monkeypatch.delenv("BABYSTEPS_BABYLM2026_TOKENIZER_DIR", raising=False)

    value = (
        "${BABYSTEPS_BABYLM2026_TOKENIZER_DIR:-"
        "${BABYSTEPS_CACHE_DIR:-/default/cache}/tokenizers/babylm2026}"
    )

    assert train_script.expand_env_string(value) == "/default/cache/tokenizers/babylm2026"

    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")
    assert train_script.expand_env_string(value) == "/cache/tokenizers/babylm2026"

    monkeypatch.setenv("BABYSTEPS_BABYLM2026_TOKENIZER_DIR", "/exact/tokenizer")
    assert train_script.expand_env_string(value) == "/exact/tokenizer"


def test_babylm2026_train_config_paths_can_use_env_roots(monkeypatch) -> None:
    train_script = load_train_script_module()
    monkeypatch.delenv("BABYSTEPS_BABYLM2026_TOKENIZER_DIR", raising=False)
    monkeypatch.delenv("BABYSTEPS_BABYLM2026_SHARDS_DIR", raising=False)
    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")
    monkeypatch.setenv("BABYSTEPS_RUNS_DIR", "/runs")

    config = train_script.load_yaml_config(
        ROOT
        / "configs"
        / "train"
        / "llama_12l896_bpe16384_babylm2026_strict_muon_liger.yaml"
    )

    assert config["run"]["output_root"] == "/runs/training"
    assert (
        config["data"]["data_dir"]
        == "/cache/token_shards/babylm2026_strict_byte_bpe_16384_bos_eos"
    )
    assert (
        config["tokenizer"]["name"]
        == "/cache/tokenizers/babylm2026_strict_byte_bpe_16384"
    )


def test_llama_muon_sweep_defaults_to_current_babylm2026_config() -> None:
    script = (
        ROOT / "scripts" / "sweeps" / "run_llama_muon_lr_seed_sweep.sh"
    ).read_text(encoding="utf-8")

    assert (
        'CONFIG="${BABYSTEPS_CONFIG:-configs/train/'
        'llama_12l896_bpe16384_babylm2026_strict_muon_liger.yaml}"'
    ) in script
    assert "MUON_LRS=(${BABYSTEPS_MUON_LRS:-0.01})" in script
    assert "BATCH_SIZES=(${BABYSTEPS_BATCH_SIZES:-${BABYSTEPS_BATCH_SIZE:-64}})" in script
    assert 'MICRO_BATCH_SIZE="${BABYSTEPS_MICRO_BATCH_SIZE:-16}"' in script


def test_babylm2026_llama_wrapper_defaults_to_current_liger_recipe() -> None:
    script = (
        ROOT
        / "scripts"
        / "sweeps"
        / "run_llama_babylm2026_muon_lr_seed_sweep.sh"
    ).read_text(encoding="utf-8")

    assert (
        'BABYSTEPS_CONFIG="${BABYSTEPS_CONFIG:-configs/train/'
        'llama_12l896_bpe16384_babylm2026_strict_muon_liger.yaml}"'
    ) in script
    assert 'BABYSTEPS_HIDDEN_SIZE="${BABYSTEPS_HIDDEN_SIZE:-896}"' in script
    assert 'BABYSTEPS_N_HEAD="${BABYSTEPS_N_HEAD:-14}"' in script
    assert 'BABYSTEPS_LLAMA_INTERMEDIATE_SIZE="${BABYSTEPS_LLAMA_INTERMEDIATE_SIZE:-2304}"' in script
    assert (
        'BABYSTEPS_LLAMA_NUM_KEY_VALUE_HEADS="'
        '${BABYSTEPS_LLAMA_NUM_KEY_VALUE_HEADS:-14}"'
    ) in script
    assert 'BABYSTEPS_BATCH_SIZE="${BABYSTEPS_BATCH_SIZE:-64}"' in script
    assert (
        'BABYSTEPS_MICRO_BATCH_SIZE="${BABYSTEPS_MICRO_BATCH_SIZE:-16}"'
    ) in script
    assert 'BABYSTEPS_MUON_LRS="${BABYSTEPS_MUON_LRS:-0.01}"' in script


def test_babylm2026_build_config_paths_can_use_exact_env_overrides(monkeypatch) -> None:
    builder = load_token_shard_builder_module()
    monkeypatch.setenv("BABYSTEPS_BABYLM2026_TOKENIZER_DIR", "/exact/tokenizer")
    monkeypatch.setenv("BABYSTEPS_BABYLM2026_SHARDS_DIR", "/exact/shards")

    config = builder.load_yaml_config(
        ROOT
        / "configs"
        / "tokenization"
        / "babylm2026_strict_byte_bpe_16384_shards.yaml"
    )

    assert config["tokenizer"]["output_dir"] == "/exact/tokenizer"
    assert config["shards"]["output_dir"] == "/exact/shards"


def test_babylm2026_strict_small_build_config_paths_can_use_env_roots(monkeypatch) -> None:
    builder = load_token_shard_builder_module()
    monkeypatch.delenv("BABYSTEPS_BABYLM2026_STRICT_SMALL_TOKENIZER_DIR", raising=False)
    monkeypatch.delenv("BABYSTEPS_BABYLM2026_STRICT_SMALL_SHARDS_DIR", raising=False)
    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")

    config = builder.load_yaml_config(
        ROOT
        / "configs"
        / "tokenization"
        / "babylm2026_strict_small_byte_bpe_16384_shards.yaml"
    )

    assert config["dataset"]["name"] == "BabyLM-community/BabyLM-2026-Strict-Small"
    assert (
        config["tokenizer"]["output_dir"]
        == "/cache/tokenizers/babylm2026_strict_small_byte_bpe_16384"
    )
    assert (
        config["shards"]["output_dir"]
        == "/cache/token_shards/babylm2026_strict_small_byte_bpe_16384_bos_eos"
    )


def test_gpt2_strict_small_train_config_paths_can_use_env_roots(monkeypatch) -> None:
    train_script = load_train_script_module()
    monkeypatch.delenv("BABYSTEPS_BABYLM2026_STRICT_SMALL_TOKENIZER_DIR", raising=False)
    monkeypatch.delenv("BABYSTEPS_BABYLM2026_STRICT_SMALL_SHARDS_DIR", raising=False)
    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")
    monkeypatch.setenv("BABYSTEPS_RUNS_DIR", "/runs")

    config = train_script.load_yaml_config(
        ROOT
        / "configs"
        / "train"
        / "gpt2_8l512_bpe16384_babylm2026_strict_small_muon.yaml"
    )

    assert config["run"]["output_root"] == "/runs/training"
    assert config["data"]["format"] == "token_shards"
    assert config["training"]["batch_size"] == 32
    assert config["training"]["mixed_precision"] == "bf16"
    assert (
        config["data"]["data_dir"]
        == "/cache/token_shards/babylm2026_strict_small_byte_bpe_16384_bos_eos"
    )
    assert (
        config["tokenizer"]["name"]
        == "/cache/tokenizers/babylm2026_strict_small_byte_bpe_16384"
    )


def test_gpt2_slip_bridge_mix_configs_use_strict_small_tokenizer_and_mix_shards(monkeypatch) -> None:
    train_script = load_train_script_module()
    token_shards = load_token_shard_builder_module()
    monkeypatch.delenv("BABYSTEPS_BABYLM2026_STRICT_SMALL_TOKENIZER_DIR", raising=False)
    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")
    monkeypatch.setenv("BABYSTEPS_RUNS_DIR", "/runs")

    tokenization_config = token_shards.load_yaml_config(
        ROOT
        / "configs"
        / "tokenization"
        / "archive"
        / "babylm2026_strict_small_plus_physical_dynamics_ewok_micro_1000_slip_bridge_1pct_byte_bpe_16384_shards.yaml"
    )
    train_config = train_script.load_yaml_config(
        ROOT
        / "configs"
        / "train"
        / "archive"
        / "gpt2_8l512_bpe16384_babylm2026_strict_small_plus_physical_dynamics_ewok_micro_1000_slip_bridge_1pct_muon.yaml"
    )

    expected_shards = (
        "/cache/token_shards/"
        "babylm2026_strict_small_plus_physical_dynamics_ewok_micro_1000_"
        "slip_bridge_1pct_byte_bpe_16384_bos_eos"
    )
    assert tokenization_config["dataset"]["kind"] == "local_text"
    assert tokenization_config["dataset"]["files"] == [
        "data/analysis/word2vec/base_text/babylm2026_strict_small/train.txt",
        (
            "data/analysis/word2vec/text_mixes/"
            "babylm2026_strict_small_plus_physical_dynamics_ewok_micro_1000_"
            "slip_bridge_1pct/synthetic_repeated.txt"
        ),
    ]
    assert (
        tokenization_config["tokenizer"]["source"]
        == "/cache/tokenizers/babylm2026_strict_small_byte_bpe_16384"
    )
    assert tokenization_config["shards"]["output_dir"] == expected_shards
    assert train_config["run"]["output_root"] == "/runs/training"
    assert train_config["data"]["data_dir"] == expected_shards
    assert train_config["model"]["n_layer"] == 8
    assert train_config["model"]["n_embd"] == 512
    assert train_config["training"]["epochs"] == 10
    assert train_config["training"]["optimizer"] == "muon_pe"
    assert train_config["training"]["muon_lr"] == 0.01
    assert train_config["training"]["mixed_precision"] == "bf16"


def test_gpt2_ewok_physical_dynamics_leak_configs_repeat_correct_pairs(monkeypatch) -> None:
    train_script = load_train_script_module()
    token_shards = load_token_shard_builder_module()
    monkeypatch.delenv("BABYSTEPS_BABYLM2026_STRICT_SMALL_TOKENIZER_DIR", raising=False)
    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")
    monkeypatch.setenv("BABYSTEPS_RUNS_DIR", "/runs")

    tokenization_config = token_shards.load_yaml_config(
        ROOT
        / "configs"
        / "tokenization"
        / "archive"
        / "babylm2026_strict_small_plus_ewok_physical_dynamics_correct_pairs_repeat4_byte_bpe_16384_shards.yaml"
    )
    train_config = train_script.load_yaml_config(
        ROOT
        / "configs"
        / "train"
        / "archive"
        / "gpt2_8l512_bpe16384_babylm2026_strict_small_plus_ewok_physical_dynamics_correct_pairs_repeat4_muon.yaml"
    )

    expected_shards = (
        "/cache/token_shards/"
        "babylm2026_strict_small_plus_ewok_physical_dynamics_correct_pairs_"
        "repeat4_byte_bpe_16384_bos_eos"
    )
    assert tokenization_config["dataset"]["kind"] == "local_text"
    assert tokenization_config["dataset"]["files"] == [
        "data/analysis/word2vec/base_text/babylm2026_strict_small/train.txt",
        "data/analysis/ewok_leakage/archive/physical_dynamics_correct_pairs_repeat4/train.txt",
    ]
    assert tokenization_config["shards"]["output_dir"] == expected_shards
    assert train_config["run"]["output_root"] == "/runs/training"
    assert train_config["run"]["name"].endswith("correct_pairs_repeat4_muon")
    assert train_config["data"]["data_dir"] == expected_shards
    assert train_config["model"]["n_layer"] == 8
    assert train_config["model"]["n_embd"] == 512
    assert train_config["training"]["epochs"] == 10
    assert train_config["training"]["optimizer"] == "muon_pe"
    assert train_config["training"]["mixed_precision"] == "bf16"


def test_gpt2_ewok_physical_dynamics_leak_configs_repeat8_correct_pairs(monkeypatch) -> None:
    train_script = load_train_script_module()
    token_shards = load_token_shard_builder_module()
    monkeypatch.delenv("BABYSTEPS_BABYLM2026_STRICT_SMALL_TOKENIZER_DIR", raising=False)
    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")
    monkeypatch.setenv("BABYSTEPS_RUNS_DIR", "/runs")

    tokenization_config = token_shards.load_yaml_config(
        ROOT
        / "configs"
        / "tokenization"
        / "archive"
        / "babylm2026_strict_small_plus_ewok_physical_dynamics_correct_pairs_repeat8_byte_bpe_16384_shards.yaml"
    )
    train_config = train_script.load_yaml_config(
        ROOT
        / "configs"
        / "train"
        / "archive"
        / "gpt2_8l512_bpe16384_babylm2026_strict_small_plus_ewok_physical_dynamics_correct_pairs_repeat8_muon.yaml"
    )

    expected_shards = (
        "/cache/token_shards/"
        "babylm2026_strict_small_plus_ewok_physical_dynamics_correct_pairs_"
        "repeat8_byte_bpe_16384_bos_eos"
    )
    assert tokenization_config["dataset"]["kind"] == "local_text"
    assert tokenization_config["dataset"]["files"] == [
        "data/analysis/word2vec/base_text/babylm2026_strict_small/train.txt",
        "data/analysis/ewok_leakage/archive/physical_dynamics_correct_pairs_repeat8/train.txt",
    ]
    assert tokenization_config["shards"]["output_dir"] == expected_shards
    assert train_config["run"]["output_root"] == "/runs/training"
    assert train_config["run"]["name"].endswith("correct_pairs_repeat8_muon")
    assert train_config["data"]["data_dir"] == expected_shards
    assert train_config["model"]["n_layer"] == 8
    assert train_config["model"]["n_embd"] == 512
    assert train_config["training"]["epochs"] == 10
    assert train_config["training"]["optimizer"] == "muon_pe"
    assert train_config["training"]["mixed_precision"] == "bf16"


@pytest.mark.parametrize("repeat", [4, 8, 16])
def test_gpt2_ewok_full_leak_configs_repeat_correct_pairs(monkeypatch, repeat: int) -> None:
    train_script = load_train_script_module()
    token_shards = load_token_shard_builder_module()
    monkeypatch.delenv("BABYSTEPS_BABYLM2026_STRICT_SMALL_TOKENIZER_DIR", raising=False)
    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")
    monkeypatch.setenv("BABYSTEPS_RUNS_DIR", "/runs")

    tokenization_config = token_shards.load_yaml_config(
        ROOT
        / "configs"
        / "tokenization"
        / "archive"
        / f"babylm2026_strict_small_plus_ewok_full_correct_pairs_repeat{repeat}_byte_bpe_16384_shards.yaml"
    )
    train_config = train_script.load_yaml_config(
        ROOT
        / "configs"
        / "train"
        / "archive"
        / f"gpt2_8l512_bpe16384_babylm2026_strict_small_plus_ewok_full_correct_pairs_repeat{repeat}_muon.yaml"
    )

    expected_shards = (
        "/cache/token_shards/"
        "babylm2026_strict_small_plus_ewok_full_correct_pairs_"
        f"repeat{repeat}_byte_bpe_16384_bos_eos"
    )
    assert tokenization_config["dataset"]["kind"] == "local_text"
    assert tokenization_config["dataset"]["files"] == [
        "data/analysis/word2vec/base_text/babylm2026_strict_small/train.txt",
        f"data/analysis/ewok_leakage/archive/full_ewok_correct_pairs_repeat{repeat}/train.txt",
    ]
    assert tokenization_config["shards"]["output_dir"] == expected_shards
    assert train_config["run"]["output_root"] == "/runs/training"
    assert train_config["run"]["name"].endswith(f"ewok_full_correct_pairs_repeat{repeat}_muon")
    assert train_config["data"]["data_dir"] == expected_shards
    assert train_config["model"]["n_layer"] == 8
    assert train_config["model"]["n_embd"] == 512
    assert train_config["training"]["epochs"] == 10
    assert train_config["training"]["optimizer"] == "muon_pe"
    assert train_config["training"]["mixed_precision"] == "bf16"


def test_modern_gpt2_smoke_config_uses_strict_small_shards(monkeypatch) -> None:
    train_script = load_train_script_module()
    monkeypatch.delenv("BABYSTEPS_BABYLM2026_STRICT_SMALL_TOKENIZER_DIR", raising=False)
    monkeypatch.delenv("BABYSTEPS_BABYLM2026_STRICT_SMALL_SHARDS_DIR", raising=False)
    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")
    monkeypatch.setenv("BABYSTEPS_RUNS_DIR", "/runs")

    config = train_script.load_yaml_config(
        ROOT
        / "configs"
        / "train"
        / "smoke"
        / "gpt2_bpe16384_babylm2026_strict_small_smoke.yaml"
    )

    assert config["run"]["name"] == "gpt2_bpe16384_babylm2026_strict_small_smoke"
    assert config["run"]["output_root"] == "/runs/training"
    assert config["data"]["format"] == "token_shards"
    assert config["data"]["max_train_blocks"] == 32
    assert config["data"]["max_val_blocks"] == 4
    assert config["training"]["batch_size"] == 8
    assert (
        config["data"]["data_dir"]
        == "/cache/token_shards/babylm2026_strict_small_byte_bpe_16384_bos_eos"
    )
    assert (
        config["tokenizer"]["name"]
        == "/cache/tokenizers/babylm2026_strict_small_byte_bpe_16384"
    )


def test_default_babylm_eval_dir_prefers_full_eval_when_present(tmp_path: Path) -> None:
    train_script = load_train_script_module()
    full_dir = (
        tmp_path
        / "external"
        / "babylm-eval"
        / "strict"
        / "evaluation_data"
        / "full_eval"
        / "blimp_filtered"
    )
    full_dir.mkdir(parents=True)

    path = train_script.default_babylm_eval_dir(
        "full_eval",
        "blimp_filtered",
        root=tmp_path,
    )

    assert path == full_dir


def test_default_babylm_eval_dir_keeps_full_eval_path_when_full_missing(
    tmp_path: Path,
) -> None:
    train_script = load_train_script_module()
    full_dir = (
        tmp_path
        / "external"
        / "babylm-eval"
        / "strict"
        / "evaluation_data"
        / "full_eval"
        / "blimp_filtered"
    )

    path = train_script.default_babylm_eval_dir(
        "full_eval",
        "blimp_filtered",
        root=tmp_path,
    )

    assert path == full_dir


class FakeTokenizer:
    all_special_ids = [0, 1, 2]
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        pieces = {3: "hello", 4: " world", 5: " again"}
        return "".join(pieces[token_id] for token_id in token_ids)


def test_estimate_words_per_token_uses_raw_chunk_token_count() -> None:
    train_script = load_train_script_module()
    dataset = SimpleNamespace(chunks=[[1, 3, 4, 2, 5]])

    estimate = train_script.estimate_words_per_token_from_dataset(
        dataset,
        FakeTokenizer(),
        sample_tokens=5,
    )

    assert estimate["sample_tokens"] == 5
    assert estimate["sample_words"] == 3
    assert estimate["words_per_token"] == 3 / 5


def test_default_words_per_token_sample_tokens_is_4096() -> None:
    train_script = load_train_script_module()

    assert train_script.DEFAULT_WORDS_PER_TOKEN_SAMPLE_TOKENS == 4096


def test_read_every_words_accepts_positive_values() -> None:
    train_script = load_train_script_module()

    assert train_script.read_every_words(
        {"every_words": 1_000_000},
        "entity_tracking_likelihood",
    ) == 1_000_000.0
    assert train_script.read_every_words({}, "entity_tracking_likelihood") is None

    with pytest.raises(ValueError, match="every_words must be positive"):
        train_script.read_every_words({"every_words": 0}, "entity_tracking_likelihood")


def test_word_interval_helpers_advance_past_current_progress() -> None:
    assert validate_every_words("toy_every_words", 1000) == 1000.0
    assert word_interval_due(words_seen=999.0, next_words=1000.0) is False
    assert word_interval_due(words_seen=1000.0, next_words=1000.0) is True
    assert (
        advance_word_interval(
            next_words=1000.0,
            every_words=1000.0,
            words_seen=2500.0,
        )
        == 3000.0
    )

    with pytest.raises(ValueError, match="toy_every_words must be positive"):
        validate_every_words("toy_every_words", -1)


def test_sample_training_token_ids_reads_token_shard_prefix(tmp_path: Path) -> None:
    train_script = load_train_script_module()
    shard = tmp_path / "train_000000.bin"
    np.asarray([10, 11, 12, 13], dtype=np.uint16).tofile(shard)
    dataset = SimpleNamespace(shards=[shard])

    assert train_script.sample_training_token_ids(dataset, sample_tokens=3) == [
        10,
        11,
        12,
    ]


def test_default_babylm_eval_dir_without_fallback_returns_full_eval_path(
    tmp_path: Path,
) -> None:
    train_script = load_train_script_module()
    full_dir = (
        tmp_path
        / "external"
        / "babylm-eval"
        / "strict"
        / "evaluation_data"
        / "full_eval"
        / "ewok_filtered"
    )

    path = train_script.default_babylm_eval_dir(
        "full_eval",
        "ewok_filtered",
        root=tmp_path,
    )

    assert path == full_dir


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


def test_cli_config_override_keeps_yaml_bool_aliases_as_strings() -> None:
    train_script = load_train_script_module()
    config = {"training": {"mixed_precision": "bf16"}}

    train_script.apply_config_overrides(
        config,
        [
            "training.mixed_precision=no",
            "model.note=off",
            "model.enabled=false",
        ],
    )

    assert config["training"]["mixed_precision"] == "no"
    assert config["model"]["note"] == "off"
    assert config["model"]["enabled"] is False


def test_read_score_reductions_accepts_single_both_and_lists() -> None:
    train_script = load_train_script_module()

    assert train_script.read_score_reductions({}, "blimp_likelihood") == ("sum",)
    assert train_script.read_score_reductions(
        {"score_reduction": "mean"},
        "blimp_likelihood",
    ) == ("mean",)
    assert train_script.read_score_reductions(
        {"score_reduction": "both"},
        "blimp_likelihood",
    ) == ("sum", "mean")
    assert train_script.read_score_reductions(
        {"score_reduction": ["mean", "sum", "mean"]},
        "blimp_likelihood",
    ) == ("mean", "sum")


def test_benchmark_eval_at_start_default_is_enabled() -> None:
    train_script = load_train_script_module()
    config = {"logging": {}}

    assert config["logging"].get("benchmark_eval_at_start", True) is True
    train_script.apply_config_override(config, "logging.benchmark_eval_at_start=false")
    assert config["logging"]["benchmark_eval_at_start"] is False


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


def test_training_input_validation_accepts_token_shard_metadata() -> None:
    train_script = load_train_script_module()
    config = {"model": {"context_length": 8, "vocab_size": 16}}
    tokenizer = SimpleNamespace()
    train_dataset = SimpleNamespace(seq_len=8, meta={"vocab_size": 16})
    val_dataset = SimpleNamespace(seq_len=8, meta={"vocab_size": 16})

    train_script.validate_training_inputs(config, tokenizer, train_dataset, val_dataset)


def test_training_input_validation_catches_token_shard_context() -> None:
    train_script = load_train_script_module()
    config = {"model": {"context_length": 4, "vocab_size": 16}}
    tokenizer = SimpleNamespace()
    train_dataset = SimpleNamespace(seq_len=8, meta={"vocab_size": 16})
    val_dataset = SimpleNamespace(seq_len=4, meta={"vocab_size": 16})

    try:
        train_script.validate_training_inputs(config, tokenizer, train_dataset, val_dataset)
    except ValueError as exc:
        assert "input positions" in str(exc)
    else:
        raise AssertionError("Expected too-short shard context to fail.")


def test_training_script_builds_llama_model_config() -> None:
    train_script = load_train_script_module()
    tokenizer = SimpleNamespace(
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config = {
        "model": {
            "architecture": "llama",
            "vocab_size": 128,
            "context_length": 32,
            "n_layer": 1,
            "n_head": 4,
            "n_embd": 32,
            "llama_intermediate_size": 64,
            "llama_num_key_value_heads": 2,
        }
    }

    hf_config = train_script.build_model_config(config, tokenizer)

    assert hf_config.model_type == "llama"
    assert hf_config.hidden_size == 32
    assert hf_config.intermediate_size == 64
    assert hf_config.num_key_value_heads == 2
