from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader, Dataset

from babysteps.evaluation.ewok_context_sensitivity import EwokItem
from babysteps.training.causal_lm_loop import EwokContextEvalSpec, train_causal_lm
from babysteps.training.spatial_auxiliary import (
    ContextRankExample,
    SpatialAuxiliarySampler,
)


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "c1": 3,
            "c2": 4,
            "t1": 5,
            "t2": 6,
        }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [self.vocab[token] for token in text.split()]

    def save_pretrained(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


class TinyCausalModel(nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 8)
        self.lm_head = nn.Linear(8, vocab_size)

    def forward(self, *, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(logits=self.lm_head(self.embedding(input_ids)))


class BatchDataset(Dataset):
    def __init__(self, batches) -> None:
        self.batches = list(batches)

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index):
        return self.batches[index]


def make_batch(tokens: torch.Tensor):
    labels = torch.zeros_like(tokens)
    labels[:, :-1] = tokens[:, 1:]
    loss_mask = torch.ones_like(tokens, dtype=torch.bool)
    loss_mask[:, -1] = False
    attention_mask = torch.ones_like(tokens, dtype=torch.bool)
    return tokens, labels, loss_mask, attention_mask


def test_train_causal_lm_logs_spatial_auxiliary_loss_and_eval(tmp_path: Path) -> None:
    tokenizer = TinyTokenizer()
    model = TinyCausalModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = [
        make_batch(torch.tensor([[1, 3, 5, 2], [1, 4, 6, 2]])),
        make_batch(torch.tensor([[1, 3, 5, 2], [1, 4, 6, 2]])),
    ]
    train_loader = DataLoader(BatchDataset(batches), batch_size=None)
    val_loader = DataLoader(BatchDataset(batches[:1]), batch_size=None)
    accelerator = Accelerator(cpu=True)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
    )
    examples = [ContextRankExample("1", "c1", "c2", "t1", "t2")]
    sampler = SpatialAuxiliarySampler(examples, batch_size=1, seed=0)
    ewok_item = EwokItem(
        item_id="toy:1",
        source_file="toy.jsonl",
        line_number=1,
        domain="toy",
        concept_a="t1",
        concept_b="t2",
        context_type="direct",
        context_diff="toy",
        target_diff="toy",
        context1="c1",
        context2="c2",
        target1="t1",
        target2="t2",
    )

    history = train_causal_lm(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        accelerator=accelerator,
        output_dir=tmp_path,
        epochs=1,
        log_every_steps=1,
        eval_every_steps=0,
        save_every_epochs=None,
        save_final=False,
        spatial_auxiliary_sampler=sampler,
        spatial_auxiliary_loss_weight=0.03,
        spatial_auxiliary_eval_items=examples,
        spatial_auxiliary_eval_every_steps=1,
        spatial_auxiliary_eval_batch_size=1,
        extra_ewok_context_evals=[
            EwokContextEvalSpec(
                name="ewok_context_sensitivity_full",
                items=[ewok_item],
                every_steps=1,
                batch_size=1,
                score_reductions=("mean",),
                output_subdir="ewok_full",
            )
        ],
    )

    train_records = [record for record in history if record["type"] == "train"]
    aux_eval_records = [
        record for record in history if record["type"] == "spatial_auxiliary_eval"
    ]

    assert any("spatial_auxiliary_loss" in record for record in train_records)
    assert aux_eval_records
    assert aux_eval_records[0]["step"] == 0
    assert aux_eval_records[-1]["step"] == 2
    assert any(
        record["type"] == "ewok_context_sensitivity_full"
        for record in history
    )
    assert (tmp_path / "benchmarks" / "ewok_full" / "context_summary.csv").exists()
