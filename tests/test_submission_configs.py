from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MURPHYSLAW_CONFIG = (
    ROOT
    / "configs"
    / "train"
    / "gpt_bert_baseline_mixed_babylm2026_strict_small_adamuon_lr0p02_aux0p006_tailavg_seed42.yaml"
)
CHATTY_BERT_CONFIG = (
    ROOT
    / "configs"
    / "train"
    / "llama_12l896_bpe16384_babylm2026_strict_muon_liger_submission_auxlr1e4_bos_seed42.yaml"
)


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_murphyslaw_submission_recipe() -> None:
    config = _load(MURPHYSLAW_CONFIG)

    assert config["model"]["architecture"] == "gpt_bert"
    assert config["data"]["format"] == "token_shards"
    assert config["hybrid"]["causal_steps"] == 1
    assert config["hybrid"]["masked_steps"] == 1
    assert config["training"]["optimizer"] == "adamuon"
    assert config["training"]["muon_lr"] == 0.02
    assert config["training"]["learning_rate"] == 0.006
    assert config["training"]["batch_size"] == 32
    assert config["checkpointing"]["tail_average"] == {
        "enabled": True,
        "fraction": 0.20,
        "every_steps": 10,
        "name": "model_tail_avg",
    }


def test_chatty_bert_submission_recipe_and_batch_accounting() -> None:
    config = _load(CHATTY_BERT_CONFIG)
    model = config["model"]
    training = config["training"]

    assert model["architecture"] == "llama"
    assert model["n_layer"] == 12
    assert model["n_embd"] == 896
    assert model["use_liger_kernel"] is True
    assert training["optimizer"] == "muon_pe"
    assert training["muon_lr"] == 0.01
    assert training["learning_rate"] == 0.0001

    effective_batch_size = (
        training["batch_size"] * training["gradient_accumulation_steps"]
    )
    tokens_per_update = effective_batch_size * config["data"]["seq_len"]

    assert effective_batch_size == 64
    assert tokens_per_update == 65_536


def test_sidbert_recipe_is_still_marked_as_unresolved() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "| SidBert |" in readme
    assert "Exact standalone 100M submission config still needs consolidation" in readme
