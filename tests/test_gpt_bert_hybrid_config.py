from __future__ import annotations

import json
from pathlib import Path

from babysteps.cli import build_token_shards as token_shard_builder
from babysteps.cli import train_causal_lm as train_script
from babysteps.cli import train_hybrid_lm as hybrid_train_script


ROOT = Path(__file__).resolve().parents[1]


class _FakeTokenizer:
    unk_token_id = 0
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 3
    mask_token_id = 4
    bos_token_id = None
    eos_token_id = None
    model_max_length = 512

    def __len__(self) -> int:
        return 32

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tokenizer_config.json").write_text(
            json.dumps({"model_max_length": self.model_max_length}, indent=2),
            encoding="utf-8",
        )


def test_gpt_bert_tokenization_config_uses_cache_roots(monkeypatch) -> None:
    monkeypatch.delenv("BABYSTEPS_GPT_BERT_BASELINE_CAUSAL_FOCUS_TOKENIZER_DIR", raising=False)
    monkeypatch.delenv(
        "BABYSTEPS_BABYLM2026_STRICT_SMALL_GPT_BERT_BASELINE_CAUSAL_FOCUS_SHARDS_DIR",
        raising=False,
    )
    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")

    config = token_shard_builder.load_yaml_config(
        ROOT
        / "configs"
        / "tokenization"
        / "babylm2026_strict_small_gpt_bert_baseline_causal_focus_shards.yaml"
    )

    assert config["tokenizer"]["kind"] == "pretrained"
    assert (
        config["tokenizer"]["output_dir"]
        == "/cache/tokenizers/gpt_bert_baseline_causal_focus"
    )
    assert (
        config["shards"]["output_dir"]
        == "/cache/token_shards/babylm2026_strict_small_gpt_bert_baseline_causal_focus"
    )


def test_gpt_bert_100m_masked_focus_tokenization_config_uses_full_strict_dataset(
    monkeypatch,
) -> None:
    monkeypatch.delenv("BABYSTEPS_GPT_BERT_BASELINE_100M_MASKED_FOCUS_TOKENIZER_DIR", raising=False)
    monkeypatch.delenv(
        "BABYSTEPS_BABYLM2026_STRICT_GPT_BERT_BASELINE_100M_MASKED_FOCUS_SHARDS_DIR",
        raising=False,
    )
    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")

    config = token_shard_builder.load_yaml_config(
        ROOT
        / "configs"
        / "tokenization"
        / "babylm2026_strict_gpt_bert_baseline_100m_masked_focus_shards.yaml"
    )

    assert config["dataset"]["name"] == "BabyLM-community/BabyLM-2026-Strict"
    assert (
        config["tokenizer"]["source"]
        == "BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus"
    )
    assert (
        config["tokenizer"]["output_dir"]
        == "/cache/tokenizers/gpt_bert_baseline_100m_masked_focus"
    )
    assert (
        config["shards"]["output_dir"]
        == "/cache/token_shards/babylm2026_strict_gpt_bert_baseline_100m_masked_focus"
    )
    assert config["shards"]["shard_tokens"] == 1_000_000
    assert config["shards"]["val_shards"] == 1
    assert config["shards"]["max_shards"] is None


def test_gpt_bert_train_config_uses_hybrid_defaults(monkeypatch) -> None:
    monkeypatch.delenv("BABYSTEPS_GPT_BERT_BASELINE_CAUSAL_FOCUS_TOKENIZER_DIR", raising=False)
    monkeypatch.delenv(
        "BABYSTEPS_BABYLM2026_STRICT_SMALL_GPT_BERT_BASELINE_CAUSAL_FOCUS_SHARDS_DIR",
        raising=False,
    )
    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")
    monkeypatch.setenv("BABYSTEPS_RUNS_DIR", "/runs")

    config = train_script.load_yaml_config(
        ROOT / "configs" / "train" / "gpt_bert_baseline_causal_focus_babylm2026_strict_small.yaml"
    )

    assert config["run"]["name"] == "gpt_bert_baseline_causal_focus_babylm2026_strict_small"
    assert config["run"]["output_root"] == "/runs/training"
    assert config["model"]["architecture"] == "gpt_bert"
    assert config["training"]["optimizer"] == "lamb"
    assert config["training"]["learning_rate"] == 0.007
    assert config["data"]["seq_len"] == 512
    assert config["hybrid"]["causal_steps"] == 15
    assert config["hybrid"]["masked_steps"] == 1
    assert config["hybrid"]["mask_start"] == 0.30
    assert config["hybrid"]["mask_end"] == 0.15
    assert config["progress"]["benchmark_plot_x_axis"] == "words_seen"
    assert config["progress"]["estimate_words_per_token"] is True
    assert config["ewok_context_sensitivity_full"]["enabled"] is True
    assert config["ewok_context_sensitivity_full"]["every_words"] == 10_000_000
    assert config["ewok_context_sensitivity_full"]["score_reduction"] == "both"
    assert config["ewok_context_sensitivity_full"]["plot"] is True
    assert config["ewok_mntp_context_sensitivity"]["enabled"] is False
    assert config["ewok_mntp_context_sensitivity"]["every_words"] is None
    assert config["ewok_mntp_context_sensitivity"]["score_reduction"] == "both"
    assert config["ewok_mntp_context_sensitivity"]["plot"] is False
    assert config["mntp_likelihood"]["enabled"] is True
    assert config["mntp_likelihood"]["every_words"] == 10_000_000
    assert config["mntp_likelihood"]["score_reduction"] == "sum"
    assert config["mntp_likelihood"]["plot"] is True
    assert config["mntp_likelihood"]["plot_every_evals"] == 1
    assert config["loss_dynamics"]["plot"] is True
    assert config["loss_dynamics"]["plot_every_evals"] == 1
    assert config["loss_dynamics"]["x_axis"] == "words_seen"
    assert config["logging"]["benchmark_eval_at_start"] is True
    assert (
        config["tokenizer"]["name"]
        == "/cache/tokenizers/gpt_bert_baseline_causal_focus"
    )


def test_gpt_bert_100m_masked_focus_train_config_uses_100m_paths(
    monkeypatch,
) -> None:
    monkeypatch.delenv("BABYSTEPS_GPT_BERT_BASELINE_100M_MASKED_FOCUS_TOKENIZER_DIR", raising=False)
    monkeypatch.delenv(
        "BABYSTEPS_BABYLM2026_STRICT_GPT_BERT_BASELINE_100M_MASKED_FOCUS_SHARDS_DIR",
        raising=False,
    )
    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")
    monkeypatch.setenv("BABYSTEPS_RUNS_DIR", "/runs")

    config = train_script.load_yaml_config(
        ROOT
        / "configs"
        / "train"
        / "gpt_bert_baseline_100m_masked_focus_babylm2026_strict.yaml"
    )

    assert config["run"]["name"] == "gpt_bert_baseline_100m_masked_focus_babylm2026_strict"
    assert config["data"]["data_dir"] == (
        "/cache/token_shards/babylm2026_strict_gpt_bert_baseline_100m_masked_focus"
    )
    assert config["tokenizer"]["name"] == "/cache/tokenizers/gpt_bert_baseline_100m_masked_focus"
    assert config["model"]["repo"] == "BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus"
    assert config["hybrid"]["causal_steps"] == 1
    assert config["hybrid"]["masked_steps"] == 15
    assert config["training"]["epochs"] == 10
    assert config["training"]["batch_size"] == 16
    assert config["training"]["gradient_accumulation_steps"] == 16
    assert config["training"]["mixed_precision"] == "bf16"
    assert config["ewok_context_sensitivity_full"]["every_words"] == config["mntp_likelihood"]["every_words"]
    assert config["mntp_likelihood"]["every_words"] > 0
    assert config["logging"]["benchmark_eval_at_start"] is True
    assert config["loss_dynamics"]["plot"] is True


def test_gpt_bert_masked_focus_train_config_uses_masked_schedule(monkeypatch) -> None:
    monkeypatch.delenv("BABYSTEPS_GPT_BERT_BASELINE_CAUSAL_FOCUS_TOKENIZER_DIR", raising=False)
    monkeypatch.delenv("BABYSTEPS_GPT_BERT_BASELINE_MASKED_FOCUS_TOKENIZER_DIR", raising=False)
    monkeypatch.delenv(
        "BABYSTEPS_BABYLM2026_STRICT_SMALL_GPT_BERT_BASELINE_CAUSAL_FOCUS_SHARDS_DIR",
        raising=False,
    )
    monkeypatch.delenv(
        "BABYSTEPS_BABYLM2026_STRICT_SMALL_GPT_BERT_BASELINE_MASKED_FOCUS_SHARDS_DIR",
        raising=False,
    )
    monkeypatch.setenv("BABYSTEPS_CACHE_DIR", "/cache")
    monkeypatch.setenv("BABYSTEPS_RUNS_DIR", "/runs")

    config = train_script.load_yaml_config(
        ROOT / "configs" / "train" / "gpt_bert_baseline_masked_focus_babylm2026_strict_small.yaml"
    )

    assert config["run"]["name"] == "gpt_bert_baseline_masked_focus_babylm2026_strict_small"
    assert config["run"]["output_root"] == "/runs/training"
    assert config["model"]["architecture"] == "gpt_bert"
    assert config["model"]["repo"] == "BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus"
    assert config["hybrid"]["causal_steps"] == 1
    assert config["hybrid"]["masked_steps"] == 15
    assert config["ewok_context_sensitivity_full"]["enabled"] is True
    assert config["ewok_context_sensitivity_full"]["every_words"] == 10_000_000
    assert config["ewok_context_sensitivity_full"]["plot"] is True
    assert config["ewok_mntp_context_sensitivity"]["enabled"] is False
    assert config["ewok_mntp_context_sensitivity"]["every_words"] is None
    assert config["ewok_mntp_context_sensitivity"]["plot"] is False
    assert (
        config["data"]["data_dir"]
        == "/cache/token_shards/babylm2026_strict_small_gpt_bert_baseline_causal_focus"
    )
    assert (
        config["tokenizer"]["name"]
        == "/cache/tokenizers/gpt_bert_baseline_causal_focus"
    )


def test_word_interval_eval_step_estimate_uses_global_update_size() -> None:
    steps = hybrid_train_script.estimate_word_interval_eval_steps(
        every_words=25,
        words_per_token=0.5,
        seq_len=10,
        batch_size=2,
        grad_accum=1,
        world_size=1,
        total_update_steps=10,
    )

    assert steps == [3, 5, 8, 10]


def test_eval_step_preview_formats_count_preview_and_last_step() -> None:
    steps = list(range(10, 260, 10))

    assert (
        hybrid_train_script.format_eval_step_preview(steps, preview=3)
        == "25 evals at update steps: 10, 20, 30, ... last=250"
    )
    assert hybrid_train_script.format_eval_step_preview([]) == "0 evals"


def test_pretrained_tokenizer_config_copies_tokenizer_and_manifest(tmp_path, monkeypatch) -> None:
    fake_tokenizer = _FakeTokenizer()

    def fake_from_pretrained(_source, **_kwargs):
        return fake_tokenizer

    monkeypatch.setattr(token_shard_builder.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    config = {
        "name": "test_gpt_bert_tokenizer",
        "dataset": {"name": "example/dataset"},
        "tokenizer": {
            "kind": "pretrained",
            "source": "example/gpt-bert",
            "trust_remote_code": True,
            "model_max_length": 128,
            "output_dir": str(tmp_path / "tokenizer"),
        },
    }

    tokenizer = token_shard_builder.train_or_load_tokenizer(
        config,
        overwrite=False,
        max_docs=None,
    )
    manifest = json.loads((tmp_path / "tokenizer" / "manifest.json").read_text())

    assert tokenizer is fake_tokenizer
    assert fake_tokenizer.model_max_length == 128
    assert manifest["source"] == "example/gpt-bert"
    assert manifest["actual_vocab_size"] == 32
    assert manifest["cls_token_id"] == 1
    assert manifest["sep_token_id"] == 2
    assert manifest["mask_token_id"] == 4
