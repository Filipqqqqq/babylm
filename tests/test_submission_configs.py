from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "configs" / "train" / "submission"
TOKENIZATION_DIR = ROOT / "configs" / "tokenization" / "submission"
MURPHYSLAW_CONFIG = TRAIN_DIR / "murphyslaw_strict_small.yaml"
CHATTY_BERT_CONFIG = TRAIN_DIR / "chatty_bert_strict.yaml"
SIDBERT_CONFIG = TRAIN_DIR / "sidbert_strict.yaml"
STRICT_SMALL_MILESTONES = (*range(1, 11), *range(20, 101, 10))
STRICT_MILESTONES = (*STRICT_SMALL_MILESTONES, *range(200, 1001, 100))


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_murphyslaw_exact_public_recipe() -> None:
    config = _load(MURPHYSLAW_CONFIG)
    training = config["training"]
    checkpointing = config["checkpointing"]

    assert config["model"] == {
        "architecture": "gpt_bert",
        "repo": "BabyLM-community/babylm-baseline-10m-gpt-bert-mixed",
        "context_length": 512,
    }
    assert config["data"]["format"] == "token_shards"
    assert config["data"]["seq_len"] == 512
    assert config["hybrid"]["causal_steps"] == 1
    assert config["hybrid"]["masked_steps"] == 1
    assert training["optimizer"] == "adamuon"
    assert training["muon_lr"] == 0.02
    assert training["learning_rate"] == 0.006
    assert training["batch_size"] == 32
    assert training["gradient_accumulation_steps"] == 1
    assert training["sequence_length_curriculum"] == {
        "enabled": True,
        "phases": [
            {"until_fraction": 0.60, "seq_len": 128},
            {"until_fraction": 0.80, "seq_len": 256},
            {"until_fraction": 1.00, "seq_len": 512},
        ],
    }
    assert config["progress"]["estimate_words_per_token"] is False
    assert config["progress"]["words_per_token"] == 0.615646119124470
    assert tuple(checkpointing["word_milestones_millions"]) == STRICT_SMALL_MILESTONES
    assert checkpointing["save_final"] is False
    assert checkpointing["tail_average"] == {
        "enabled": True,
        "fraction": 0.20,
        "every_steps": 10,
        "name": "final",
    }
    assert config["logging"]["benchmark_eval_mode"] == "final_only"


def test_chatty_bert_exact_public_recipe_and_batch_accounting() -> None:
    config = _load(CHATTY_BERT_CONFIG)
    model = config["model"]
    training = config["training"]

    assert model["architecture"] == "llama"
    assert model["vocab_size"] == 16_384
    assert model["context_length"] == 1_024
    assert model["n_layer"] == 12
    assert model["n_head"] == 14
    assert model["n_embd"] == 896
    assert model["llama_intermediate_size"] == 2_304
    assert model["use_liger_kernel"] is True
    assert training["optimizer"] == "muon_pe"
    assert training["muon_lr"] == 0.01
    assert training["learning_rate"] == 0.0001

    effective_batch = training["batch_size"] * training["gradient_accumulation_steps"]
    tokens_per_update = effective_batch * config["data"]["seq_len"]
    assert effective_batch == 64
    assert tokens_per_update == 65_536
    assert tuple(config["checkpointing"]["word_milestones_millions"]) == STRICT_MILESTONES
    assert config["checkpointing"]["save_final"] is True


def test_sidbert_exact_public_recipe_and_distributed_batch_accounting() -> None:
    config = _load(SIDBERT_CONFIG)
    training = config["training"]
    hybrid = config["hybrid"]

    assert config["run"]["seed"] == 42
    assert config["run"]["expected_world_size"] == 2
    assert config["model"] == {
        "architecture": "gpt_bert",
        "repo": "BabyLM-community/babylm-baseline-100m-gpt-bert-mixed",
        "context_length": 512,
    }
    assert config["data"]["packing_mode"] == "sid_wrapped"
    assert hybrid["causal_steps"] == hybrid["masked_steps"] == 1
    assert hybrid["objective_granularity"] == "microbatch"
    assert hybrid["objective_cycle_start"] == "masked"
    assert hybrid["objective_micro_batch_size"] == 8
    assert training["optimizer"] == "muon_pe"
    assert training["muon_lr"] == 0.005
    assert training["learning_rate"] == 0.007
    assert training["betas"] == [0.9, 0.98]
    assert training["warmup_ratio"] == 0.016
    assert training["total_steps"] == 12_330
    assert training["sequence_length_curriculum"] == {
        "enabled": True,
        "phases": [
            {"until_fraction": 0.60, "seq_len": 128},
            {"until_fraction": 0.80, "seq_len": 256},
            {"until_fraction": 1.00, "seq_len": 512},
        ],
    }

    effective_batch = training["batch_size"] * config["run"]["expected_world_size"]
    tokens_per_update = effective_batch * config["data"]["seq_len"]
    words_seen = training["total_steps"] * tokens_per_update * config["progress"]["words_per_token"]
    assert effective_batch == 256
    assert tokens_per_update == 131_072
    assert words_seen == 1_000_000_000
    assert tuple(config["checkpointing"]["word_milestones_millions"]) == STRICT_MILESTONES
    assert config["checkpointing"]["save_final"] is True
    assert config["logging"]["benchmark_eval_mode"] == "final_only"


def test_matching_public_tokenization_recipes() -> None:
    murphyslaw = _load(TOKENIZATION_DIR / "murphyslaw_strict_small_shards.yaml")
    sidbert = _load(TOKENIZATION_DIR / "sidbert_strict_shards.yaml")
    chatty = _load(TOKENIZATION_DIR / "chatty_bert_strict_shards.yaml")

    assert murphyslaw["dataset"]["name"] == "BabyLM-community/BabyLM-2026-Strict-Small"
    assert murphyslaw["tokenizer"]["kind"] == "pretrained"
    assert murphyslaw["tokenizer"]["source"] == config_repo(MURPHYSLAW_CONFIG)
    assert sidbert["dataset"]["name"] == "BabyLM-community/BabyLM-2026-Strict"
    assert sidbert["tokenizer"]["kind"] == "pretrained"
    assert sidbert["tokenizer"]["source"] == config_repo(SIDBERT_CONFIG)
    assert sidbert["shards"]["document_format"] == "eos_only"
    assert chatty["dataset"]["name"] == "BabyLM-community/BabyLM-2026-Strict"
    assert chatty["tokenizer"]["kind"] == "byte_bpe"
    assert chatty["tokenizer"]["vocab_size"] == 16_384
    assert chatty["tokenizer"]["bos_token"] == "<s>"


def config_repo(path: Path) -> str:
    return _load(path)["model"]["repo"]


def test_readme_publishes_all_three_submission_recipes() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert SIDBERT_CONFIG.exists()
    assert "configs/train/submission/sidbert_strict.yaml" in readme
    assert "SidBert release gate" not in readme


def test_public_config_surface_is_deliberately_small() -> None:
    yaml_paths = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "configs").rglob("*.yaml")
    }
    assert yaml_paths == {
        "configs/eval/full_eval.yaml",
        "configs/tokenization/submission/chatty_bert_strict_shards.yaml",
        "configs/tokenization/submission/murphyslaw_strict_small_shards.yaml",
        "configs/tokenization/submission/sidbert_strict_shards.yaml",
        "configs/train/smoke/llama_tiny_bpe16384_10m_muon_liger_smoke.yaml",
        "configs/train/submission/chatty_bert_strict.yaml",
        "configs/train/submission/murphyslaw_strict_small.yaml",
        "configs/train/submission/sidbert_strict.yaml",
    }
    for removed_surface in ("archive", "docs", "scripts/sweeps", "scripts/analysis"):
        assert not (ROOT / removed_surface).exists()
