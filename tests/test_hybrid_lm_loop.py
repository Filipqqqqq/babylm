from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader, Dataset

import babysteps.training.hybrid_lm_loop as hybrid_lm_loop
from babysteps.evaluation.mntp_likelihood import MntpBenchmarkResult
from babysteps.training.hybrid_lm_loop import (
    build_special_mask,
    compute_gpt_bert_loss,
    make_causal_batch,
    make_masked_batch,
    objective_schedule_from_config,
    objective_for,
    set_gpt_bert_is_causal,
    train_hybrid_lm,
)
from babysteps.training.optimizers import build_optimizer


class _Tokenizer:
    all_special_ids = [0, 1, 2, 3, 9]
    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2
    bos_token_id = None
    eos_token_id = None
    mask_token_id = 9

    def __len__(self) -> int:
        return 16

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text(
            json.dumps({"model_max_length": 8}, indent=2),
            encoding="utf-8",
        )


class _TinyGptBert(nn.Module):
    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.model = SimpleNamespace(is_causal=True)
        self.embedding = nn.Embedding(vocab_size, 8)
        self.lm_head = nn.Linear(8, vocab_size)

    def forward(self, *, input_ids, attention_mask, return_dict=True):
        del attention_mask, return_dict
        logits = self.lm_head(self.embedding(input_ids))
        return SimpleNamespace(logits=logits)


class _BatchDataset(Dataset):
    def __init__(self, batches) -> None:
        self.batches = list(batches)

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index):
        return self.batches[index]


def _batch(tokens: torch.Tensor):
    labels = torch.zeros_like(tokens)
    loss_mask = torch.ones_like(tokens, dtype=torch.bool)
    attention_mask = torch.ones_like(tokens, dtype=torch.bool)
    return tokens, labels, loss_mask, attention_mask


def test_objective_schedule_uses_micro_steps() -> None:
    assert [objective_for(step, causal_steps=3, masked_steps=1) for step in range(8)] == [
        "causal",
        "causal",
        "causal",
        "masked",
        "causal",
        "causal",
        "causal",
        "masked",
    ]


def test_objective_schedule_supports_masked_focus() -> None:
    assert [objective_for(step, causal_steps=1, masked_steps=3) for step in range(8)] == [
        "causal",
        "masked",
        "masked",
        "masked",
        "causal",
        "masked",
        "masked",
        "masked",
    ]


def test_objective_schedule_config_replaces_causal_per_masked() -> None:
    assert objective_schedule_from_config({"causal_steps": 15, "masked_steps": 1}) == (15, 1)
    assert objective_schedule_from_config({"causal_steps": 1, "masked_steps": 15}) == (1, 15)

    try:
        objective_schedule_from_config({"causal_per_masked": 15})
    except ValueError as exc:
        assert "causal_per_masked was removed" in str(exc)
    else:
        raise AssertionError("Expected causal_per_masked to be rejected.")

    try:
        objective_schedule_from_config({"causal_steps": 0, "masked_steps": 0})
    except ValueError as exc:
        assert "At least one" in str(exc)
    else:
        raise AssertionError("Expected an empty objective schedule to be rejected.")


def test_make_causal_batch_shifts_labels() -> None:
    input_ids = torch.tensor([[1, 4, 5, 2]])
    causal_input_ids, labels = make_causal_batch(input_ids)

    assert causal_input_ids is input_ids
    assert labels.tolist() == [[4, 5, 2, -100]]


def test_make_masked_batch_scores_next_token_and_skips_specials() -> None:
    input_ids = torch.tensor([[1, 4, 5, 2]])
    special_mask = build_special_mask(input_ids, {1, 2, 9})
    gen = torch.Generator().manual_seed(0)

    corrupted, labels = make_masked_batch(
        input_ids,
        special_mask,
        mask_ratio=1.0,
        vocab_size=16,
        mask_id=9,
        p_random=0.0,
        p_keep=0.0,
        gen=gen,
    )

    assert corrupted.tolist() == [[1, 9, 9, 2]]
    assert labels.tolist() == [[4, 5, -100, -100]]


def test_compute_gpt_bert_loss_uses_only_valid_labels() -> None:
    model = _TinyGptBert()
    input_ids = torch.tensor([[1, 4, 5, 2]])
    labels = torch.tensor([[4, -100, 2, -100]])
    attention_mask = torch.ones_like(input_ids)

    loss, ce_loss, token_count = compute_gpt_bert_loss(
        model,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        z_loss=0.01,
    )

    assert token_count.item() == 2
    assert torch.isfinite(loss)
    assert torch.isfinite(ce_loss)
    assert loss.item() > ce_loss.item()


def test_set_gpt_bert_is_causal_updates_backbone() -> None:
    model = _TinyGptBert()

    set_gpt_bert_is_causal(model, False)
    assert model.model.is_causal is False

    set_gpt_bert_is_causal(model, True)
    assert model.model.is_causal is True


def test_train_hybrid_lm_smoke_logs_validation_and_saves(tmp_path: Path) -> None:
    tokenizer = _Tokenizer()
    model = _TinyGptBert()
    optimizer, info = build_optimizer(
        model,
        {
            "optimizer": "lamb",
            "learning_rate": 1e-3,
            "betas": [0.9, 0.98],
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
    )
    assert info.name == "lamb"

    batches = [
        _batch(torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])),
        _batch(torch.tensor([[1, 5, 4, 2], [1, 7, 6, 2]])),
    ]
    train_loader = DataLoader(_BatchDataset(batches), batch_size=None)
    val_loader = DataLoader(_BatchDataset(batches[:1]), batch_size=None)
    accelerator = Accelerator(cpu=True)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
    )

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    for filename in ("configuration_gpt_bert.py", "modeling_gpt_bert.py", "config.json"):
        (repo_dir / filename).write_text("{}", encoding="utf-8")

    history = train_hybrid_lm(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        accelerator=accelerator,
        output_dir=tmp_path,
        epochs=1,
        total_update_steps=2,
        hybrid_config={
            "causal_steps": 1,
            "masked_steps": 1,
            "mask_start": 1.0,
            "mask_end": 1.0,
            "random_ratio": 0.0,
            "keep_ratio": 0.0,
            "z_loss": 0.0,
        },
        model_repo=str(repo_dir),
        log_every_steps=1,
        eval_every_steps=1,
        eval_max_batches=1,
        save_every_epochs=1,
        save_final=True,
    )

    assert any(record["type"] == "validation" for record in history)
    assert any("val_causal_loss" in record for record in history)
    assert any("val_mntp_loss" in record for record in history)
    train_records = [record for record in history if record["type"] == "train"]
    assert any("train_causal_loss" in record for record in train_records)
    assert any("train_mntp_loss" in record for record in train_records)
    assert (tmp_path / "checkpoint-epoch-0001" / "pytorch_model.bin").exists()
    assert (tmp_path / "checkpoint-epoch-0001" / "modeling_gpt_bert.py").exists()
    assert (tmp_path / "final" / "pytorch_model.bin").exists()


def test_train_hybrid_lm_runs_mntp_benchmarks_on_word_interval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []
    plot_calls = []
    loss_plot_calls = []

    def fake_evaluate_mntp_likelihood_benchmark(**kwargs):
        calls.append(kwargs)
        assert kwargs["score_reduction"] == "sum"
        assert kwargs["benchmark_name"] == "blimp"
        assert kwargs["model"].model.is_causal is False
        return MntpBenchmarkResult(
            item_records=[
                {
                    "uid": "item-1",
                    "field": "toy",
                    "linguistics_term": "toy",
                    "good_score": 1.0,
                    "bad_score": 0.0,
                    "correct": True,
                    "margin": 1.0,
                    "tie": False,
                }
            ],
            summary_rows=[
                {
                    "group_scope": "overall",
                    "group_name": "all",
                    "score_reduction": "sum",
                    "items": 1,
                    "correct": 1,
                    "score": 1.0,
                    "mean_margin": 1.0,
                    "ties": 0,
                }
            ],
        )

    def fake_plot_mntp_likelihood_summaries(**kwargs):
        plot_calls.append(kwargs)
        assert kwargs["summary_paths"]["blimp"].exists()
        assert kwargs["x_axis"] == "words_seen"

    def fake_write_training_loss_plots(**kwargs):
        loss_plot_calls.append(kwargs)
        assert kwargs["metrics_paths"] == [tmp_path / "metrics_history.json"]
        assert kwargs["output_dir"] == tmp_path / "loss_dynamics"
        assert kwargs["labels"] == ["toy_run"]
        assert kwargs["x_axis"] == "words_seen"
        assert kwargs["metrics_paths"][0].exists()
        return [tmp_path / "loss_dynamics" / "loss_by_words_seen.png"]

    monkeypatch.setattr(
        hybrid_lm_loop,
        "evaluate_mntp_likelihood_benchmark",
        fake_evaluate_mntp_likelihood_benchmark,
    )
    monkeypatch.setattr(
        hybrid_lm_loop,
        "plot_mntp_likelihood_summaries",
        fake_plot_mntp_likelihood_summaries,
    )
    monkeypatch.setattr(
        hybrid_lm_loop,
        "write_training_loss_plots",
        fake_write_training_loss_plots,
    )

    tokenizer = _Tokenizer()
    model = _TinyGptBert()
    optimizer, _info = build_optimizer(
        model,
        {
            "optimizer": "lamb",
            "learning_rate": 1e-3,
            "betas": [0.9, 0.98],
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
    )
    batches = [
        _batch(torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])),
        _batch(torch.tensor([[1, 5, 4, 2], [1, 7, 6, 2]])),
    ]
    train_loader = DataLoader(_BatchDataset(batches), batch_size=None)
    val_loader = DataLoader(_BatchDataset(batches[:1]), batch_size=None)
    accelerator = Accelerator(cpu=True)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
    )

    history = train_hybrid_lm(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        accelerator=accelerator,
        output_dir=tmp_path,
        epochs=1,
        total_update_steps=2,
        hybrid_config={
            "causal_steps": 1,
            "masked_steps": 1,
            "mask_start": 1.0,
            "mask_end": 1.0,
            "random_ratio": 0.0,
            "keep_ratio": 0.0,
            "z_loss": 0.0,
        },
        benchmark_eval_at_start=False,
        mntp_likelihood_items={"blimp": [object()]},
        mntp_likelihood_every_words=10,
        mntp_likelihood_score_reductions=("sum",),
        mntp_likelihood_plot=True,
        mntp_likelihood_run_name="toy_run",
        loss_dynamics_plot=True,
        loss_dynamics_run_name="toy_run",
        benchmark_plot_x_axis="words_seen",
        words_per_token=1.0,
        log_every_steps=1,
        eval_every_steps=0,
        save_every_epochs=0,
        save_final=False,
    )

    records = [record for record in history if record["type"] == "mntp_likelihood"]
    assert len(calls) == 1
    assert len(plot_calls) == 1
    assert len(loss_plot_calls) == 1
    assert len(records) == 1
    assert records[0]["benchmark"] == "blimp"
    assert records[0]["score_reduction"] == "sum"
    assert records[0]["score"] == 1.0
    assert records[0]["words_seen"] == 16.0
    assert (tmp_path / "benchmarks" / "mntp_likelihood" / "likelihood_summary.csv").exists()
    assert (tmp_path / "benchmarks" / "mntp_likelihood" / "likelihood_items.jsonl").exists()
    assert (
        tmp_path
        / "benchmarks"
        / "mntp_likelihood"
        / "blimp"
        / "likelihood_summary.csv"
    ).exists()
    assert (
        tmp_path
        / "benchmarks"
        / "mntp_likelihood"
        / "blimp"
        / "likelihood_items.jsonl"
    ).exists()
