#!/usr/bin/env python3
"""Train a GPT-2 small causal LM from token-chunk pickle files."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from transformers import get_cosine_schedule_with_warmup

from babysteps.data.token_chunks import build_token_chunk_loader
from babysteps.training.causal_lm_loop import (
    save_json_atomic,
    set_all_seeds,
    train_causal_lm,
)


def expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR} references in loaded YAML values."""

    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return expand_env_vars(config)


def resolve_output_dir(config: dict[str, Any]) -> Path:
    run = config["run"]
    output_dir = run.get("output_dir")
    if output_dir:
        path = Path(output_dir)
    else:
        output_root = Path(run.get("output_root", ROOT / "runs" / "training"))
        path = output_root / run["name"]

    if not path.is_absolute():
        path = ROOT / path

    if run.get("append_timestamp", False):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path / stamp

    return path


def build_gpt2_config(config: dict[str, Any], tokenizer) -> GPT2Config:
    model_config = config["model"]
    context_length = int(model_config.get("context_length", 1024))
    vocab_size = int(model_config.get("vocab_size", len(tokenizer)))

    return GPT2Config(
        vocab_size=vocab_size,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        n_ctx=context_length,
        n_positions=context_length,
        n_embd=int(model_config.get("n_embd", 768)),
        n_head=int(model_config.get("n_head", 12)),
        n_layer=int(model_config.get("n_layer", 12)),
        resid_pdrop=float(model_config.get("resid_pdrop", 0.1)),
        embd_pdrop=float(model_config.get("embd_pdrop", 0.1)),
        attn_pdrop=float(model_config.get("attn_pdrop", 0.1)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "train" / "gpt2_small_10m_smoke.yaml"),
        help="Path to a YAML training config.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    run_config = config["run"]
    data_config = config["data"]
    training_config = config["training"]
    logging_config = config.get("logging", {})
    checkpoint_config = config.get("checkpointing", {})
    generation_config = config.get("generation", {})

    output_dir = resolve_output_dir(config)
    seed = int(run_config.get("seed", 42))
    set_all_seeds(seed, deterministic=bool(run_config.get("deterministic", False)))

    accelerator = Accelerator(
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 1)),
        mixed_precision=training_config.get("mixed_precision", "no"),
    )

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json_atomic(config, output_dir / "config.json")
        print(f"Output dir: {output_dir}")

    tokenizer_name = config["tokenizer"].get("name", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, train_dataset = build_token_chunk_loader(
        data_config["train_chunks"],
        batch_size=int(training_config.get("batch_size", 16)),
        pad_id=int(tokenizer.pad_token_id),
        shuffle=True,
        max_chunks=data_config.get("max_train_chunks"),
        num_workers=int(data_config.get("num_workers", 0)),
        seed=seed,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader, val_dataset = build_token_chunk_loader(
        data_config["val_chunks"],
        batch_size=int(training_config.get("batch_size", 16)),
        pad_id=int(tokenizer.pad_token_id),
        shuffle=False,
        max_chunks=data_config.get("max_val_chunks"),
        num_workers=int(data_config.get("num_workers", 0)),
        seed=seed,
        pin_memory=torch.cuda.is_available(),
    )

    if accelerator.is_main_process:
        print(f"Train chunks: {len(train_dataset):,}")
        print(f"Val chunks: {len(val_dataset):,}")

    model = GPT2LMHeadModel(build_gpt2_config(config, tokenizer))
    if accelerator.is_main_process:
        params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 5e-5)),
        betas=tuple(training_config.get("betas", [0.9, 0.999])),
        eps=float(training_config.get("eps", 1e-8)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )

    epochs = int(training_config.get("epochs", 1))
    grad_accum = int(training_config.get("gradient_accumulation_steps", 1))
    updates_per_epoch = max(1, (len(train_loader) + grad_accum - 1) // grad_accum)
    total_steps = epochs * updates_per_epoch
    warmup_steps = training_config.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = int(float(training_config.get("warmup_ratio", 0.1)) * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_steps),
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
    )

    train_causal_lm(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        output_dir=output_dir,
        epochs=epochs,
        log_every_steps=int(logging_config.get("log_every_steps", 50)),
        eval_every_steps=int(logging_config.get("eval_every_steps", 250)),
        eval_max_batches=logging_config.get("eval_max_batches", 20),
        generate_every_steps=int(generation_config.get("every_steps", 250)),
        generation_prompts=generation_config.get("prompts"),
        max_new_tokens=int(generation_config.get("max_new_tokens", 40)),
        gradient_clip_norm=training_config.get("gradient_clip_norm"),
        save_every_epochs=checkpoint_config.get("save_every_epochs", 1),
        save_final=bool(checkpoint_config.get("save_final", True)),
    )


if __name__ == "__main__":
    main()
