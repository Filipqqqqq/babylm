#!/usr/bin/env python3
"""Train a GPT-2 or LLaMA-style causal LM from token chunks or token shards."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import yaml
from accelerate import Accelerator
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

from babysteps.data.token_chunks import build_token_chunk_loader
from babysteps.data.token_shards import build_token_shard_loader
from babysteps.models import (
    build_causal_lm_model,
    build_causal_lm_config_from_dict,
    normalize_model_arch,
    resolve_model_vocab_size,
)
from babysteps.training.causal_lm_loop import (
    save_json_atomic,
    set_all_seeds,
    train_causal_lm,
)
from babysteps.training.liger import apply_liger_kernel_if_requested
from babysteps.training.optimizers import build_optimizer
from babysteps.paths import expand_env_string, expand_env_vars, validate_tokenizer_dir


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return expand_env_vars(config)


def parse_override_value(raw_value: str) -> Any:
    """Parse one CLI override value with YAML scalar/list/dict semantics."""

    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse override value {raw_value!r}: {exc}") from exc

    # PyYAML follows YAML 1.1 boolean aliases, so unquoted values like "no" and
    # "off" become False. Preserve those as strings for CLI ergonomics; callers
    # can still use true/false for booleans.
    if isinstance(value, bool) and raw_value.strip() in {
        "y",
        "Y",
        "yes",
        "Yes",
        "YES",
        "n",
        "N",
        "no",
        "No",
        "NO",
        "on",
        "On",
        "ON",
        "off",
        "Off",
        "OFF",
    }:
        value = raw_value.strip()

    # PyYAML keeps values like "1e-4" as strings, but those are convenient for
    # learning-rate sweeps. Convert only when Python can clearly parse a number.
    if isinstance(value, str) and any(char in value for char in ("e", "E")):
        try:
            value = float(value)
        except ValueError:
            pass

    return expand_env_vars(value)


def apply_config_override(config: dict[str, Any], override: str) -> None:
    """Apply a dotted ``section.key=value`` override to a loaded config."""

    if "=" not in override:
        raise ValueError(f"Override must look like section.key=value, got {override!r}.")

    dotted_key, raw_value = override.split("=", 1)
    keys = [key.strip() for key in dotted_key.split(".") if key.strip()]
    if not keys:
        raise ValueError(f"Override key cannot be empty: {override!r}.")

    cursor: dict[str, Any] = config
    for key in keys[:-1]:
        next_value = cursor.get(key)
        if next_value is None:
            next_value = {}
            cursor[key] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(
                f"Cannot set {override!r}: {'.'.join(keys[:-1])} is not a mapping."
            )
        cursor = next_value

    cursor[keys[-1]] = parse_override_value(raw_value)


def apply_config_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    for override in overrides:
        apply_config_override(config, override)


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


def build_model_config(config: dict[str, Any], tokenizer):
    """Build the selected causal-LM config from a loaded training config."""

    return build_causal_lm_config_from_dict(config, tokenizer)


def build_gpt2_config(config: dict[str, Any], tokenizer):
    """Backward-compatible alias for older tests/imports."""

    return build_model_config(config, tokenizer)


def token_chunk_summary(dataset) -> dict[str, int]:
    """Return small compatibility stats for an already-loaded chunk dataset."""

    max_length = 0
    max_token_id = -1
    for chunk in dataset.chunks:
        if not chunk:
            continue
        max_length = max(max_length, len(chunk))
        max_token_id = max(max_token_id, max(chunk))
    return {"max_length": max_length, "max_token_id": max_token_id}


def token_shard_summary(dataset) -> dict[str, int]:
    """Return small compatibility stats for a token-shard dataset."""

    vocab_size = dataset.meta.get("vocab_size")
    max_token_id = int(vocab_size) - 1 if vocab_size is not None else -1
    return {"max_input_length": int(dataset.seq_len), "max_token_id": max_token_id}


def validate_training_inputs(
    config: dict[str, Any],
    tokenizer,
    train_dataset,
    val_dataset,
) -> None:
    """Fail early for config/data mismatches that otherwise crash on GPU."""

    model_config = config["model"]
    context_length = int(model_config.get("context_length", model_config.get("seq_len", 1024)))
    vocab_size = resolve_model_vocab_size(model_config, tokenizer)

    for split_name, dataset in (("train", train_dataset), ("val", val_dataset)):
        if hasattr(dataset, "chunks"):
            summary = token_chunk_summary(dataset)
            max_input_length = max(0, summary["max_length"] - 1)
        else:
            summary = token_shard_summary(dataset)
            max_input_length = summary["max_input_length"]
        if max_input_length > context_length:
            raise ValueError(
                f"{split_name} chunks produce {max_input_length} input positions, "
                f"but model.context_length is {context_length}. Increase "
                "model.context_length or rebuild chunks with a smaller chunk_size."
            )
        if summary["max_token_id"] >= 0 and summary["max_token_id"] >= vocab_size:
            raise ValueError(
                f"{split_name} chunks contain token id {summary['max_token_id']}, "
                f"but model.vocab_size is {vocab_size}. Increase model.vocab_size "
                "or rebuild chunks with the matching tokenizer."
            )


def release_cuda_memory() -> None:
    """Best-effort CUDA cache cleanup for interrupted direct runs."""

    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        # ipc_collect can fail if CUDA is already tearing down after an error.
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "train" / "smoke" / "gpt2_small_10m_smoke.yaml"),
        help="Path to a YAML training config.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help=(
            "Override a config value after loading YAML. "
            "Can be passed more than once, e.g. "
            "--set run.seed=43 --set training.learning_rate=1e-4."
        ),
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    try:
        apply_config_overrides(config, args.overrides)
    except ValueError as exc:
        parser.error(str(exc))

    run_config = config["run"]
    data_config = config["data"]
    training_config = config["training"]
    logging_config = config.get("logging", {})
    checkpoint_config = config.get("checkpointing", {})

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
        print(f"Output dir: {output_dir}", flush=True)

    tokenizer_name = config["tokenizer"].get("name", "gpt2")
    tokenizer_path = validate_tokenizer_dir(tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path or tokenizer_name))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_format = str(data_config.get("format", "token_chunks")).strip().lower()
    batch_size = int(training_config.get("batch_size", 16))
    num_workers = int(data_config.get("num_workers", 0))
    if data_format in {"token_chunks", "chunks", "pickle"}:
        train_loader, train_dataset = build_token_chunk_loader(
            data_config["train_chunks"],
            batch_size=batch_size,
            pad_id=int(tokenizer.pad_token_id),
            shuffle=True,
            max_chunks=data_config.get("max_train_chunks"),
            num_workers=num_workers,
            seed=seed,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader, val_dataset = build_token_chunk_loader(
            data_config["val_chunks"],
            batch_size=batch_size,
            pad_id=int(tokenizer.pad_token_id),
            shuffle=False,
            max_chunks=data_config.get("max_val_chunks"),
            num_workers=num_workers,
            seed=seed,
            pin_memory=torch.cuda.is_available(),
        )
    elif data_format in {"token_shards", "shards", "bin"}:
        model_config = config["model"]
        seq_len = int(
            data_config.get(
                "seq_len",
                model_config.get("context_length", model_config.get("seq_len", 1024)),
            )
        )
        data_dir = data_config["data_dir"]
        train_loader, train_dataset = build_token_shard_loader(
            data_dir,
            split=str(data_config.get("train_split", "train")),
            batch_size=batch_size,
            seq_len=seq_len,
            shuffle=True,
            max_blocks=data_config.get("max_train_blocks", data_config.get("max_train_chunks")),
            max_shards=data_config.get("max_train_shards"),
            num_workers=num_workers,
            seed=seed,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader, val_dataset = build_token_shard_loader(
            data_dir,
            split=str(data_config.get("val_split", "val")),
            batch_size=batch_size,
            seq_len=seq_len,
            shuffle=False,
            max_blocks=data_config.get("max_val_blocks", data_config.get("max_val_chunks")),
            max_shards=data_config.get("max_val_shards"),
            num_workers=num_workers,
            seed=seed,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        raise ValueError(
            f"Unsupported data.format={data_format!r}; expected token_chunks or token_shards."
        )

    if accelerator.is_main_process:
        print(f"Data format: {data_format}", flush=True)
        if data_format in {"token_chunks", "chunks", "pickle"}:
            print(f"Train chunks: {len(train_dataset):,}", flush=True)
            print(f"Val chunks: {len(val_dataset):,}", flush=True)
        else:
            print(f"Train shard blocks: {len(train_dataset):,}", flush=True)
            print(f"Val shard blocks: {len(val_dataset):,}", flush=True)

    validate_training_inputs(config, tokenizer, train_dataset, val_dataset)

    architecture = normalize_model_arch(
        config["model"].get("architecture", config["model"].get("model_arch", "gpt2"))
    )
    liger_applied = apply_liger_kernel_if_requested(
        use_liger_kernel=bool(config["model"].get("use_liger_kernel", False)),
        model_arch=architecture,
    )
    model = build_causal_lm_model(config, tokenizer)
    if accelerator.is_main_process:
        print(f"Model architecture: {architecture}", flush=True)
        print(f"Liger kernel applied: {liger_applied}", flush=True)
        if architecture == "llama":
            print(
                "LLaMA dropout: "
                f"attention={float(getattr(model.config, 'attention_dropout', 0.0))} "
                f"embd={float(getattr(model.config, 'babysteps_llama_embd_pdrop', 0.0))} "
                f"resid={float(getattr(model.config, 'babysteps_llama_resid_pdrop', 0.0))}",
                flush=True,
            )
        params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {params:,}", flush=True)

    optimizer, optimizer_info = build_optimizer(
        model,
        training_config,
        device=accelerator.device,
    )
    if accelerator.is_main_process:
        print(f"Optimizer: {optimizer_info.name}", flush=True)
        if optimizer_info.name == "muon_pe":
            print(f"Muon hidden matrix tensors: {optimizer_info.primary_tensors}", flush=True)
            print(f"Auxiliary AdamW tensors: {optimizer_info.auxiliary_tensors}", flush=True)
            print(f"Muon split-QKV tensors: {optimizer_info.qkv_split_tensors}", flush=True)
            print(
                "Muon batch same-shape updates: "
                f"{bool(training_config.get('muon_batch_updates', True))}",
                flush=True,
            )
        else:
            print(f"AdamW decayed tensors: {optimizer_info.primary_tensors}", flush=True)
            print(f"AdamW non-decayed tensors: {optimizer_info.auxiliary_tensors}", flush=True)
            print(f"Using fused AdamW: {optimizer_info.use_fused}", flush=True)

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
        gradient_clip_norm=training_config.get("gradient_clip_norm"),
        save_every_epochs=checkpoint_config.get("save_every_epochs", 1),
        save_final=bool(checkpoint_config.get("save_final", True)),
        profile_optimizer_steps=bool(logging_config.get("profile_optimizer_steps", False)),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        release_cuda_memory()
        sys.exit(130)
