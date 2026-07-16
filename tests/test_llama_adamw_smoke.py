from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from babysteps.data.token_chunks import TokenChunkDataset, collate_token_chunks
from babysteps.models import build_causal_lm_model
from babysteps.training.causal_lm_loop import train_causal_lm
from babysteps.training.optimizers import build_optimizer


class _TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __len__(self) -> int:
        return 64

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text(
            json.dumps({"model_max_length": 16}, indent=2),
            encoding="utf-8",
        )
        (path / "special_tokens_map.json").write_text(
            json.dumps(
                {
                    "bos_token": "<bos>",
                    "eos_token": "<eos>",
                    "pad_token": "<pad>",
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _loader(chunks: list[list[int]]) -> DataLoader:
    dataset = TokenChunkDataset(chunks)
    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: collate_token_chunks(batch, pad_id=0),
    )


def test_llama_adamw_smoke_trains_evals_and_saves(tmp_path: Path) -> None:
    tokenizer = _TinyTokenizer()
    config = {
        "model": {
            "architecture": "llama",
            "vocab_size": 64,
            "context_length": 16,
            "n_layer": 1,
            "n_head": 4,
            "n_embd": 32,
            "llama_intermediate_size": 64,
            "llama_num_key_value_heads": 2,
            "llama_tie_word_embeddings": True,
        }
    }
    training_config = {
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
    }

    torch.manual_seed(123)
    model = build_causal_lm_model(config, tokenizer)
    optimizer, optimizer_info = build_optimizer(model, training_config)
    assert optimizer_info.name == "adamw"

    train_loader = _loader(
        [
            [1, 3, 4, 5, 2],
            [1, 6, 7, 8, 2],
            [1, 9, 10, 11, 2],
            [1, 12, 13, 14, 2],
        ]
    )
    val_loader = _loader(
        [
            [1, 15, 16, 17, 2],
            [1, 18, 19, 20, 2],
        ]
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader),
    )
    accelerator = Accelerator(cpu=True)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
    )

    history = train_causal_lm(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        output_dir=tmp_path,
        epochs=1,
        log_every_steps=1,
        eval_every_steps=1,
        eval_max_batches=1,
        save_every_epochs=1,
        save_final=True,
    )

    assert any(record["type"] == "train" for record in history)
    assert any(record["type"] == "validation" for record in history)
    assert any(record["type"] == "epoch" for record in history)
    assert (tmp_path / "metrics_history.json").exists()
    assert (tmp_path / "checkpoint-epoch-0001" / "config.json").exists()
    assert (tmp_path / "checkpoint-epoch-0001" / "model.safetensors").exists()
    assert (tmp_path / "final" / "config.json").exists()
    assert (tmp_path / "final" / "model.safetensors").exists()


def test_word_based_validation_schedule(tmp_path: Path) -> None:
    tokenizer = _TinyTokenizer()
    config = {
        "model": {
            "architecture": "llama",
            "vocab_size": 64,
            "context_length": 16,
            "n_layer": 1,
            "n_head": 4,
            "n_embd": 32,
            "llama_intermediate_size": 64,
            "llama_num_key_value_heads": 2,
            "llama_tie_word_embeddings": True,
        }
    }
    training_config = {
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
    }

    torch.manual_seed(123)
    model = build_causal_lm_model(config, tokenizer)
    optimizer, _ = build_optimizer(model, training_config)
    train_loader = _loader(
        [
            [1, 3, 4, 5, 2],
            [1, 6, 7, 8, 2],
            [1, 9, 10, 11, 2],
            [1, 12, 13, 14, 2],
        ]
    )
    val_loader = _loader(
        [
            [1, 15, 16, 17, 2],
            [1, 18, 19, 20, 2],
        ]
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader),
    )
    accelerator = Accelerator(cpu=True)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
    )

    history = train_causal_lm(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        output_dir=tmp_path,
        epochs=1,
        log_every_steps=0,
        eval_every_steps=0,
        eval_every_words=4.0,
        eval_max_batches=1,
        save_every_epochs=None,
        save_final=False,
        words_per_token=0.5,
    )

    validation_records = [
        record for record in history if record["type"] == "validation"
    ]
    assert [record["step"] for record in validation_records] == [1, 2]
    assert [record["words_seen"] for record in validation_records] == [4.0, 8.0]
