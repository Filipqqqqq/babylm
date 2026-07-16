from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
import yaml
from transformers import AutoModelForCausalLM

import babysteps.models.causal_lm as causal_lm_module
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


def _config_rope_theta(hf_config) -> float:
    if hasattr(hf_config, "rope_theta"):
        return float(hf_config.rope_theta)

    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if rope_parameters is None:
        rope_parameters = getattr(hf_config, "rope_scaling", None)
    return float(rope_parameters["rope_theta"])


def test_normalize_model_arch_accepts_expected_aliases() -> None:
    assert normalize_model_arch("gpt-2") == "gpt2"
    assert normalize_model_arch("llama_3") == "llama"
    assert normalize_model_arch("gpt-bert") == "gpt_bert"

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
    assert _config_rope_theta(hf_config) == 500000.0
    assert hf_config.bos_token_id == 1
    assert hf_config.eos_token_id == 2
    assert hf_config.pad_token_id == 0
    assert hf_config.use_cache is False


def test_build_llama_config_from_dict_supports_optional_dropout() -> None:
    config = {
        "model": {
            "architecture": "llama",
            "context_length": 32,
            "n_layer": 1,
            "n_head": 4,
            "n_embd": 32,
            "llama_dropout": 0.1,
            "llama_resid_pdrop": 0.2,
            "llama_attention_dropout": 0.3,
        }
    }

    hf_config = build_causal_lm_config_from_dict(config, _tokenizer(vocab_size=128))

    assert hf_config.model_type == "llama"
    assert hf_config.attention_dropout == pytest.approx(0.3)
    assert hf_config.babysteps_llama_embd_pdrop == pytest.approx(0.1)
    assert hf_config.babysteps_llama_resid_pdrop == pytest.approx(0.2)


def test_llama_dropout_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="llama_embd_pdrop"):
        build_causal_lm_config(
            architecture="llama",
            vocab_size=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            context_length=16,
            n_embd=32,
            n_head=4,
            n_layer=1,
            llama_embd_pdrop=1.0,
        )


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


def test_tiny_llama_model_applies_optional_training_dropout() -> None:
    config = {
        "model": {
            "architecture": "llama",
            "context_length": 16,
            "n_layer": 1,
            "n_head": 4,
            "n_embd": 32,
            "vocab_size": 64,
            "llama_intermediate_size": 64,
            "llama_embd_pdrop": 0.8,
            "llama_resid_pdrop": 0.8,
            "llama_attention_dropout": 0.0,
        }
    }

    model = build_causal_lm_model(config, _tokenizer())
    input_ids = torch.tensor([[1, 3, 4, 2]])
    attention_mask = torch.ones_like(input_ids)

    assert hasattr(model.model, "babysteps_embed_dropout")
    assert hasattr(model.model.layers[0], "babysteps_resid_dropout")
    assert len(model._babysteps_dropout_hook_handles) == 2

    model.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        eval_logits_1 = model(input_ids=input_ids, attention_mask=attention_mask).logits
        torch.manual_seed(1)
        eval_logits_2 = model(input_ids=input_ids, attention_mask=attention_mask).logits
    assert torch.allclose(eval_logits_1, eval_logits_2)

    model.train()
    with torch.no_grad():
        torch.manual_seed(0)
        train_logits_1 = model(input_ids=input_ids, attention_mask=attention_mask).logits
        torch.manual_seed(1)
        train_logits_2 = model(input_ids=input_ids, attention_mask=attention_mask).logits
    assert not torch.allclose(train_logits_1, train_logits_2)


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


def test_gpt_bert_config_from_dict_uses_baseline_repo(monkeypatch) -> None:
    calls = {}

    def fake_from_pretrained(repo: str, *, trust_remote_code: bool):
        calls["repo"] = repo
        calls["trust_remote_code"] = trust_remote_code
        return SimpleNamespace(vocab_size=0, use_cache=True)

    monkeypatch.setattr(causal_lm_module.AutoConfig, "from_pretrained", fake_from_pretrained)
    config = {
        "model": {
            "architecture": "gpt_bert",
            "repo": "example/gpt-bert",
        }
    }
    tokenizer = _tokenizer(vocab_size=123)
    tokenizer.bos_token_id = None
    tokenizer.cls_token_id = 1
    tokenizer.eos_token_id = None
    tokenizer.sep_token_id = 2

    hf_config = build_causal_lm_config_from_dict(config, tokenizer)

    assert calls == {"repo": "example/gpt-bert", "trust_remote_code": True}
    assert hf_config.vocab_size == 123
    assert hf_config.bos_token_id == 1
    assert hf_config.eos_token_id == 2
    assert hf_config.pad_token_id == 0
    assert hf_config.use_cache is False
    assert hf_config.babysteps_gpt_bert_repo == "example/gpt-bert"


def test_gpt_bert_model_builder_uses_masked_lm_class(monkeypatch) -> None:
    sentinel_config = SimpleNamespace(vocab_size=64, use_cache=False)
    sentinel_model = object()
    calls = {}

    monkeypatch.setattr(
        causal_lm_module.AutoConfig,
        "from_pretrained",
        lambda repo, *, trust_remote_code: sentinel_config,
    )

    def fake_from_config(hf_config, *, trust_remote_code: bool):
        calls["hf_config"] = hf_config
        calls["trust_remote_code"] = trust_remote_code
        return sentinel_model

    monkeypatch.setattr(causal_lm_module.AutoModelForMaskedLM, "from_config", fake_from_config)
    model = build_causal_lm_model(
        {"model": {"architecture": "gpt_bert", "repo": "example/gpt-bert"}},
        _tokenizer(vocab_size=64),
    )

    assert model is sentinel_model
    assert calls == {"hf_config": sentinel_config, "trust_remote_code": True}


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
        / "smoke"
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
        / "smoke"
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
        / "smoke"
        / "llama_tiny_bpe16384_10m_muon_liger_smoke.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    hf_config = build_causal_lm_config_from_dict(config, _tokenizer(vocab_size=16384))

    assert hf_config.model_type == "llama"
    assert config["model"]["use_liger_kernel"] is True
    assert config["training"]["optimizer"] == "muon_pe"
    assert config["logging"]["profile_optimizer_steps"] is True


def test_llama_100m_muon_liger_config_is_current_default() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "train"
        / "llama_12l896_bpe16384_babylm2026_strict_muon_liger.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    hf_config = build_causal_lm_config_from_dict(config, _tokenizer(vocab_size=16384))
    training_config = config["training"]

    assert config["data"]["format"] == "token_shards"
    assert config["data"].get("max_train_blocks") is None
    assert config["data"].get("max_val_blocks") is None
    assert hf_config.model_type == "llama"
    assert hf_config.hidden_size == 896
    assert hf_config.intermediate_size == 2304
    assert hf_config.num_hidden_layers == 12
    assert hf_config.num_attention_heads == 14
    assert hf_config.num_key_value_heads == 14
    assert hf_config.babysteps_llama_embd_pdrop == pytest.approx(0.1)
    assert config["model"]["use_liger_kernel"] is True
    assert training_config["optimizer"] == "muon_pe"
    assert training_config["muon_lr"] == 0.01
    assert training_config["batch_size"] == 16
    assert training_config["gradient_accumulation_steps"] == 4
    assert training_config["warmup_ratio"] == 0.03
    assert training_config["mixed_precision"] == "bf16"


def test_llama_100m_muon_liger_causal_diagnostics_config_enables_benchmarks() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "train"
        / "llama_12l896_bpe16384_babylm2026_strict_muon_liger_causal_diagnostics.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    hf_config = build_causal_lm_config_from_dict(config, _tokenizer(vocab_size=16384))

    assert hf_config.model_type == "llama"
    assert hf_config.hidden_size == 896
    assert hf_config.num_hidden_layers == 12
    assert config["progress"]["benchmark_plot_x_axis"] == "words_seen"
    assert config["progress"]["estimate_words_per_token"] is True
    assert config["progress"]["words_per_token_sample_tokens"] == 8192
    assert config["logging"]["benchmark_eval_at_start"] is True
    assert "mntp_likelihood" not in config
    assert config["ewok_context_sensitivity_full"]["enabled"] is True
    assert config["ewok_context_sensitivity_full"]["every_words"] == 50_000_000
    assert config["ewok_context_sensitivity_full"]["score_reduction"] == "both"
    for section_name in (
        "blimp_likelihood",
        "entity_tracking_likelihood",
        "comps_likelihood",
    ):
        section = config[section_name]
        assert section["enabled"] is True
        assert section["every_words"] == 50_000_000
        assert section["score_reduction"] == "sum"
        assert section["plot"] is True
