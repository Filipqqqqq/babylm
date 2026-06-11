from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
import yaml
from transformers import AutoModelForCausalLM

from babysteps.models import (
    build_causal_lm_config,
    build_causal_lm_config_from_dict,
    build_causal_lm_model,
    default_llama_intermediate_size,
    normalize_model_arch,
    resolve_model_vocab_size,
)


class _Tokenizer(SimpleNamespace):
    def __len__(self) -> int:
        return int(self.vocab_size)


def _tokenizer(vocab_size: int = 128) -> _Tokenizer:
    return _Tokenizer(
        vocab_size=vocab_size,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def test_normalize_model_arch_accepts_expected_aliases() -> None:
    assert normalize_model_arch("gpt-2") == "gpt2"
    assert normalize_model_arch("llama_3") == "llama"

    with pytest.raises(ValueError, match="Unsupported model architecture"):
        normalize_model_arch("mamba")


def test_build_gpt2_config_from_dict_keeps_existing_surface() -> None:
    config = {
        "model": {
            "architecture": "gpt2",
            "vocab_size": 16384,
            "context_length": 512,
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 256,
            "resid_pdrop": 0.2,
            "embd_pdrop": 0.3,
            "attn_pdrop": 0.4,
        }
    }

    hf_config = build_causal_lm_config_from_dict(config, _tokenizer())

    assert hf_config.model_type == "gpt2"
    assert hf_config.vocab_size == 16384
    assert hf_config.n_ctx == 512
    assert hf_config.n_positions == 512
    assert hf_config.n_layer == 4
    assert hf_config.n_head == 4
    assert hf_config.n_embd == 256
    assert hf_config.resid_pdrop == 0.2
    assert hf_config.embd_pdrop == 0.3
    assert hf_config.attn_pdrop == 0.4
    assert hf_config.bos_token_id == 1
    assert hf_config.eos_token_id == 2
    assert hf_config.pad_token_id == 0
    assert hf_config.use_cache is False


def test_build_llama_config_from_dict_uses_babysteps_shape_keys() -> None:
    config = {
        "model": {
            "architecture": "llama",
            "context_length": 1024,
            "n_layer": 2,
            "n_head": 4,
            "n_embd": 128,
            "llama_num_key_value_heads": 2,
            "llama_tie_word_embeddings": True,
            "rope_theta": 500000.0,
        }
    }

    hf_config = build_causal_lm_config_from_dict(config, _tokenizer(vocab_size=32000))

    assert hf_config.model_type == "llama"
    assert hf_config.vocab_size == 32000
    assert hf_config.max_position_embeddings == 1024
    assert hf_config.num_hidden_layers == 2
    assert hf_config.num_attention_heads == 4
    assert hf_config.num_key_value_heads == 2
    assert hf_config.hidden_size == 128
    assert hf_config.intermediate_size == default_llama_intermediate_size(128)
    assert hf_config.tie_word_embeddings is True
    assert hf_config.rope_theta == 500000.0
    assert hf_config.bos_token_id == 1
    assert hf_config.eos_token_id == 2
    assert hf_config.pad_token_id == 0
    assert hf_config.use_cache is False


def test_tiny_llama_model_instantiates_and_runs_forward() -> None:
    hf_config = build_causal_lm_config(
        architecture="llama",
        vocab_size=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        context_length=16,
        n_embd=32,
        n_head=4,
        n_layer=1,
        llama_intermediate_size=64,
        llama_num_key_value_heads=2,
    )

    model = AutoModelForCausalLM.from_config(hf_config)
    input_ids = torch.tensor([[1, 3, 4, 2]])
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs.logits.shape == (1, 4, 64)
    assert model.config.model_type == "llama"


def test_tiny_gpt2_model_helper_instantiates_and_runs_forward() -> None:
    config = {
        "model": {
            "architecture": "gpt2",
            "context_length": 16,
            "n_layer": 1,
            "n_head": 4,
            "n_embd": 32,
            "vocab_size": 64,
            "resid_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "attn_pdrop": 0.0,
        }
    }

    model = build_causal_lm_model(config, _tokenizer())
    input_ids = torch.tensor([[1, 3, 4, 2]])
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs.logits.shape == (1, 4, 64)
    assert model.config.model_type == "gpt2"


def test_llama_config_rejects_invalid_attention_shapes() -> None:
    with pytest.raises(ValueError, match="hidden size"):
        build_causal_lm_config(
            architecture="llama",
            vocab_size=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            context_length=16,
            n_embd=30,
            n_head=8,
            n_layer=1,
        )

    with pytest.raises(ValueError, match="key/value heads"):
        build_causal_lm_config(
            architecture="llama",
            vocab_size=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            context_length=16,
            n_embd=32,
            n_head=8,
            n_layer=1,
            llama_num_key_value_heads=3,
        )


def test_resolve_vocab_size_defaults_to_tokenizer_length() -> None:
    assert resolve_model_vocab_size({}, _tokenizer(vocab_size=321)) == 321
    assert resolve_model_vocab_size({"vocab_size": 0}, _tokenizer(vocab_size=321)) == 321
    assert resolve_model_vocab_size({"vocab_size": 123}, _tokenizer(vocab_size=321)) == 123


def test_llama_tiny_smoke_config_builds_llama_adamw_surface() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "train"
        / "llama_tiny_bpe16384_10m_smoke.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    hf_config = build_causal_lm_config_from_dict(config, _tokenizer(vocab_size=16384))

    assert config["training"]["optimizer"] == "adamw"
    assert hf_config.model_type == "llama"
    assert hf_config.vocab_size == 16384
    assert hf_config.hidden_size == 256
    assert hf_config.num_hidden_layers == 4
    assert hf_config.num_attention_heads == 4
    assert hf_config.num_key_value_heads == 4


def test_llama_tiny_muon_smoke_config_uses_moonshot_muon_defaults() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "train"
        / "llama_tiny_bpe16384_10m_muon_smoke.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    hf_config = build_causal_lm_config_from_dict(config, _tokenizer(vocab_size=16384))
    training_config = config["training"]

    assert hf_config.model_type == "llama"
    assert hf_config.hidden_size == 256
    assert training_config["optimizer"] == "muon_pe"
    assert training_config["betas"] == [0.9, 0.95]
    assert training_config["weight_decay"] == 0.1
    assert training_config["muon_lr"] == 0.02
    assert training_config["muon_weight_decay"] == 0.1
    assert training_config["muon_momentum"] == 0.95
    assert training_config["muon_ns_steps"] == 5
    assert training_config["muon_nesterov"] is True
    assert training_config["muon_split_qkv"] is True
    assert training_config["muon_batch_updates"] is True


def test_llama_tiny_muon_liger_smoke_config_enables_liger_only() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "train"
        / "llama_tiny_bpe16384_10m_muon_liger_smoke.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    hf_config = build_causal_lm_config_from_dict(config, _tokenizer(vocab_size=16384))

    assert hf_config.model_type == "llama"
    assert config["model"]["use_liger_kernel"] is True
    assert config["training"]["optimizer"] == "muon_pe"
    assert config["logging"]["profile_optimizer_steps"] is True
