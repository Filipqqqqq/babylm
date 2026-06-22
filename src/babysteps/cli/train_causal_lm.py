#!/usr/bin/env python3
"""Train a GPT-2 or LLaMA-style causal LM from token chunks or token shards."""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

from babysteps.data.token_chunks import build_token_chunk_loader
from babysteps.data.token_shards import build_token_shard_loader
from babysteps.evaluation.blimp_likelihood import read_blimp_items
from babysteps.evaluation.comps_likelihood import read_comps_items
from babysteps.evaluation.entity_tracking_likelihood import read_entity_tracking_items
from babysteps.evaluation.ewok_context_sensitivity import read_ewok_items
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
from babysteps.paths import (
    expand_env_string,
    expand_env_vars,
    resolve_path,
    validate_tokenizer_dir,
)


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


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
        output_root = Path(
            run.get(
                "output_root",
                Path(
                    os.environ.get("BABYSTEPS_RUNS_DIR", "~/babylm-baby-steps-runs")
                ).expanduser()
                / "training",
            )
        )
        path = output_root / run["name"]

    if not path.is_absolute():
        path = ROOT / path

    if run.get("append_timestamp", False):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path / stamp

    return path


def default_babylm_eval_dir(
    split: str,
    *relative_parts: str,
    fallback_split: str | None = None,
    fallback_relative_parts: tuple[str, ...] | None = None,
    root: Path | None = None,
) -> Path:
    """Return a default BabyLM eval-data path, preferring full eval when present."""

    root = ROOT if root is None else root
    eval_root = root / "external" / "babylm-eval" / "strict" / "evaluation_data"
    candidate = eval_root / split / Path(*relative_parts)
    if candidate.exists() or fallback_split is None:
        return candidate

    fallback_parts = fallback_relative_parts or relative_parts
    fallback = eval_root / fallback_split / Path(*fallback_parts)
    return fallback if fallback.exists() else candidate


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


def sample_training_token_ids(dataset, *, sample_tokens: int) -> list[int]:
    """Take a deterministic prefix sample from a token-chunk or token-shard dataset."""

    if int(sample_tokens) <= 0:
        raise ValueError("sample_tokens must be positive.")

    token_ids: list[int] = []
    if hasattr(dataset, "chunks"):
        for chunk in dataset.chunks:
            remaining = int(sample_tokens) - len(token_ids)
            if remaining <= 0:
                break
            token_ids.extend(int(token_id) for token_id in chunk[:remaining])
    elif hasattr(dataset, "shards"):
        for shard_path in dataset.shards:
            remaining = int(sample_tokens) - len(token_ids)
            if remaining <= 0:
                break
            shard = np.memmap(Path(shard_path), dtype=np.uint16, mode="r")
            take = min(remaining, int(shard.size))
            token_ids.extend(
                int(token_id) for token_id in np.asarray(shard[:take], dtype=np.int64)
            )
    else:
        raise TypeError("Cannot sample tokens from this dataset type.")

    if not token_ids:
        raise ValueError("Cannot estimate words_per_token from an empty token sample.")
    return token_ids


def count_decoded_words(text: str) -> int:
    """Count word-like spans in decoded training text."""

    return len(WORD_RE.findall(text))


def estimate_words_per_token_from_dataset(
    dataset,
    tokenizer,
    *,
    sample_tokens: int = 2048,
) -> dict[str, float | int]:
    """Estimate a run-level words/token conversion from decoded training tokens."""

    token_ids = sample_training_token_ids(dataset, sample_tokens=sample_tokens)
    raw_token_count = len(token_ids)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_ids.add(int(token_id))

    decode_ids = [token_id for token_id in token_ids if token_id not in special_ids]
    try:
        text = tokenizer.decode(
            decode_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        text = tokenizer.decode(decode_ids, skip_special_tokens=True)

    word_count = count_decoded_words(text)
    if word_count <= 0:
        raise ValueError(
            "Cannot estimate words_per_token because the decoded token sample had no "
            "word-like spans."
        )

    return {
        "words_per_token": word_count / raw_token_count,
        "sample_tokens": raw_token_count,
        "sample_words": word_count,
    }


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
        default=str(
            ROOT
            / "configs"
            / "train"
            / "smoke"
            / "gpt2_bpe16384_babylm2026_strict_small_smoke.yaml"
        ),
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
    ewok_context_config = config.get("ewok_context_sensitivity", {})
    blimp_likelihood_config = config.get("blimp_likelihood", {})
    entity_tracking_config = config.get("entity_tracking_likelihood", {})
    comps_config = config.get("comps_likelihood", {})
    progress_config = config.get("progress", {})

    words_per_token = progress_config.get(
        "words_per_token",
        logging_config.get("words_per_token"),
    )
    tokens_per_word = progress_config.get(
        "tokens_per_word",
        logging_config.get("tokens_per_word"),
    )
    if words_per_token is not None and tokens_per_word is not None:
        raise ValueError("Set only one of words_per_token or tokens_per_word.")
    if tokens_per_word is not None:
        tokens_per_word = float(tokens_per_word)
        if tokens_per_word <= 0:
            raise ValueError("tokens_per_word must be positive.")
        words_per_token = 1.0 / tokens_per_word
    elif words_per_token is not None:
        words_per_token = float(words_per_token)
        if words_per_token <= 0:
            raise ValueError("words_per_token must be positive.")

    benchmark_plot_x_axis = str(
        progress_config.get(
            "benchmark_plot_x_axis",
            logging_config.get("benchmark_plot_x_axis", "step"),
        )
    ).strip()
    estimate_words_per_token = bool(
        progress_config.get(
            "estimate_words_per_token",
            logging_config.get(
                "estimate_words_per_token",
                benchmark_plot_x_axis == "words_seen" and words_per_token is None,
            ),
        )
    )
    words_per_token_sample_tokens = int(
        progress_config.get(
            "words_per_token_sample_tokens",
            logging_config.get("words_per_token_sample_tokens", 2048),
        )
        or 2048
    )
    if words_per_token_sample_tokens <= 0:
        raise ValueError("words_per_token_sample_tokens must be positive.")

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
        if words_per_token is None and estimate_words_per_token:
            print(
                "Words/token estimation: "
                f"enabled, sample_tokens={words_per_token_sample_tokens:,}",
                flush=True,
            )
        elif words_per_token is not None:
            print(
                f"Words/token conversion: words_per_token={words_per_token:.6f}",
                flush=True,
            )

    ewok_context_items = None
    ewok_context_every_steps = int(
        ewok_context_config.get(
            "every_steps",
            ewok_context_config.get("ewok_every_steps", 0),
        )
        or 0
    )
    ewok_context_enabled = bool(
        ewok_context_config.get("enabled", ewok_context_every_steps > 0)
    )
    if ewok_context_enabled and ewok_context_every_steps <= 0:
        raise ValueError(
            "ewok_context_sensitivity.every_steps must be positive when diagnostics "
            "are enabled."
        )
    ewok_context_score_reduction = str(
        ewok_context_config.get("score_reduction", "sum")
    ).strip().lower()
    if ewok_context_score_reduction not in {"sum", "mean"}:
        raise ValueError(
            "ewok_context_sensitivity.score_reduction must be 'sum' or 'mean', "
            f"got {ewok_context_score_reduction!r}."
        )
    ewok_context_plot = bool(ewok_context_config.get("plot", False))
    ewok_context_plot_every_evals = int(
        ewok_context_config.get("plot_every_evals", 1) or 1
    )
    if ewok_context_plot and ewok_context_plot_every_evals <= 0:
        raise ValueError("ewok_context_sensitivity.plot_every_evals must be positive.")
    if ewok_context_enabled:
        default_ewok_data_dir = default_babylm_eval_dir(
            "full_eval",
            "ewok_filtered",
        )
        ewok_data_dir = resolve_path(
            ewok_context_config.get(
                "data_dir",
                default_ewok_data_dir,
            ),
            base_dir=ROOT,
        )
        ewok_context_items = read_ewok_items(ewok_data_dir)
        max_ewok_items = ewok_context_config.get("max_items")
        if max_ewok_items:
            ewok_context_items = ewok_context_items[: int(max_ewok_items)]
        if accelerator.is_main_process:
            print(
                "EWoK context-sensitivity diagnostics: "
                f"{len(ewok_context_items):,} items, "
                f"every {ewok_context_every_steps} steps, "
                f"reduction={ewok_context_score_reduction}",
                flush=True,
            )
            if ewok_context_plot:
                print(
                    "EWoK context-sensitivity plot: "
                    f"enabled every {ewok_context_plot_every_evals} eval(s)",
                    flush=True,
                )

    blimp_likelihood_items = None
    blimp_likelihood_every_steps = int(blimp_likelihood_config.get("every_steps", 0) or 0)
    blimp_likelihood_enabled = bool(
        blimp_likelihood_config.get("enabled", blimp_likelihood_every_steps > 0)
    )
    if blimp_likelihood_enabled and blimp_likelihood_every_steps <= 0:
        raise ValueError(
            "blimp_likelihood.every_steps must be positive when diagnostics are enabled."
        )
    blimp_likelihood_plot = bool(blimp_likelihood_config.get("plot", False))
    blimp_likelihood_plot_every_evals = int(
        blimp_likelihood_config.get("plot_every_evals", 1) or 1
    )
    if blimp_likelihood_plot and blimp_likelihood_plot_every_evals <= 0:
        raise ValueError("blimp_likelihood.plot_every_evals must be positive.")
    if blimp_likelihood_enabled:
        default_eval_dir = (
            ROOT
            / "external"
            / "babylm-eval"
            / "strict"
            / "evaluation_data"
            / "full_eval"
        )
        default_blimp_data_dir = default_babylm_eval_dir(
            "full_eval",
            "blimp_filtered",
            fallback_split="fast_eval",
            fallback_relative_parts=("blimp_fast",),
        )
        default_supplement_data_dir = default_babylm_eval_dir(
            "full_eval",
            "supplement_filtered",
            fallback_split="fast_eval",
            fallback_relative_parts=("supplement_fast",),
        )
        raw_benchmarks = blimp_likelihood_config.get("benchmarks")
        if raw_benchmarks is None:
            raw_benchmarks = {
                "blimp": {
                    "data_dir": blimp_likelihood_config.get(
                        "blimp_data_dir",
                        default_blimp_data_dir,
                    ),
                    "max_items": blimp_likelihood_config.get("max_blimp_items"),
                },
                "blimp_supplement": {
                    "data_dir": blimp_likelihood_config.get(
                        "supplement_data_dir",
                        default_supplement_data_dir,
                    ),
                    "max_items": blimp_likelihood_config.get("max_supplement_items"),
                },
            }
        if not isinstance(raw_benchmarks, dict):
            raise ValueError("blimp_likelihood.benchmarks must be a mapping.")

        blimp_likelihood_items = {}
        for benchmark_name, benchmark_config in raw_benchmarks.items():
            if benchmark_config is None:
                benchmark_config = {}
            if not isinstance(benchmark_config, dict):
                raise ValueError(
                    "Each blimp_likelihood.benchmarks entry must be a mapping, "
                    f"got {benchmark_name!r}."
                )
            if benchmark_config.get("enabled", True) is False:
                continue
            default_data_dir = default_eval_dir / (
                "supplement_filtered"
                if benchmark_name == "blimp_supplement"
                else "blimp_filtered"
            )
            data_dir = resolve_path(
                benchmark_config.get("data_dir", default_data_dir),
                base_dir=ROOT,
            )
            items = read_blimp_items(data_dir)
            max_items = benchmark_config.get("max_items")
            if max_items:
                items = items[: int(max_items)]
            blimp_likelihood_items[str(benchmark_name)] = items

        if not blimp_likelihood_items:
            raise ValueError("blimp_likelihood is enabled but no benchmarks were loaded.")
        if accelerator.is_main_process:
            print(
                "BLiMP likelihood diagnostics: "
                + ", ".join(
                    f"{name}={len(items):,} items"
                    for name, items in blimp_likelihood_items.items()
                )
                + f", every {blimp_likelihood_every_steps} steps",
                flush=True,
            )
            if blimp_likelihood_plot:
                print(
                    "BLiMP likelihood plots: "
                    f"enabled every {blimp_likelihood_plot_every_evals} eval(s)",
                    flush=True,
                )

    entity_tracking_items = None
    entity_tracking_every_steps = int(entity_tracking_config.get("every_steps", 0) or 0)
    entity_tracking_enabled = bool(
        entity_tracking_config.get("enabled", entity_tracking_every_steps > 0)
    )
    if entity_tracking_enabled and entity_tracking_every_steps <= 0:
        raise ValueError(
            "entity_tracking_likelihood.every_steps must be positive when diagnostics "
            "are enabled."
        )
    entity_tracking_plot = bool(entity_tracking_config.get("plot", False))
    entity_tracking_plot_every_evals = int(
        entity_tracking_config.get("plot_every_evals", 1) or 1
    )
    if entity_tracking_plot and entity_tracking_plot_every_evals <= 0:
        raise ValueError("entity_tracking_likelihood.plot_every_evals must be positive.")
    if entity_tracking_enabled:
        default_entity_tracking_data_dir = default_babylm_eval_dir(
            "full_eval",
            "entity_tracking",
            fallback_split="fast_eval",
            fallback_relative_parts=("entity_tracking_fast",),
        )
        entity_tracking_data_dir = resolve_path(
            entity_tracking_config.get(
                "data_dir",
                default_entity_tracking_data_dir,
            ),
            base_dir=ROOT,
        )
        entity_tracking_items = read_entity_tracking_items(entity_tracking_data_dir)
        max_entity_tracking_items = entity_tracking_config.get("max_items")
        if max_entity_tracking_items:
            entity_tracking_items = entity_tracking_items[: int(max_entity_tracking_items)]
        if accelerator.is_main_process:
            print(
                "Entity Tracking likelihood diagnostics: "
                f"{len(entity_tracking_items):,} items, "
                f"every {entity_tracking_every_steps} steps",
                flush=True,
            )
            if entity_tracking_plot:
                print(
                    "Entity Tracking likelihood plots: "
                    f"enabled every {entity_tracking_plot_every_evals} eval(s)",
                    flush=True,
                )

    comps_items = None
    comps_every_steps = int(comps_config.get("every_steps", 0) or 0)
    comps_enabled = bool(comps_config.get("enabled", comps_every_steps > 0))
    if comps_enabled and comps_every_steps <= 0:
        raise ValueError(
            "comps_likelihood.every_steps must be positive when diagnostics are enabled."
        )
    comps_plot = bool(comps_config.get("plot", False))
    comps_plot_every_evals = int(comps_config.get("plot_every_evals", 1) or 1)
    if comps_plot and comps_plot_every_evals <= 0:
        raise ValueError("comps_likelihood.plot_every_evals must be positive.")
    if comps_enabled:
        default_comps_data_dir = default_babylm_eval_dir(
            "full_eval",
            "comps",
        )
        comps_data_dir = resolve_path(
            comps_config.get(
                "data_dir",
                default_comps_data_dir,
            ),
            base_dir=ROOT,
        )
        comps_items = read_comps_items(comps_data_dir)
        max_comps_items = comps_config.get("max_items")
        if max_comps_items:
            comps_items = comps_items[: int(max_comps_items)]
        if accelerator.is_main_process:
            print(
                "COMPS likelihood diagnostics: "
                f"{len(comps_items):,} items, "
                f"every {comps_every_steps} steps",
                flush=True,
            )
            if comps_plot:
                print(
                    "COMPS likelihood plots: "
                    f"enabled every {comps_plot_every_evals} eval(s)",
                    flush=True,
                )

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

    if words_per_token is None and estimate_words_per_token:
        estimate = estimate_words_per_token_from_dataset(
            train_dataset,
            tokenizer,
            sample_tokens=words_per_token_sample_tokens,
        )
        words_per_token = float(estimate["words_per_token"])
        progress_config = config.setdefault("progress", {})
        progress_config["words_per_token"] = words_per_token
        progress_config["words_per_token_estimated"] = True
        progress_config["words_per_token_sample_tokens"] = int(
            estimate["sample_tokens"]
        )
        progress_config["words_per_token_sample_words"] = int(estimate["sample_words"])
        if accelerator.is_main_process:
            save_json_atomic(config, output_dir / "config.json")
            print(
                "Estimated words_per_token: "
                f"{words_per_token:.6f} "
                f"({int(estimate['sample_words']):,} words / "
                f"{int(estimate['sample_tokens']):,} tokens)",
                flush=True,
            )

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
        ewok_context_items=ewok_context_items,
        ewok_context_every_steps=ewok_context_every_steps,
        ewok_context_batch_size=int(ewok_context_config.get("batch_size", 64)),
        ewok_context_score_reduction=ewok_context_score_reduction,  # type: ignore[arg-type]
        ewok_context_run_name=str(run_config.get("name", "")),
        ewok_context_plot=ewok_context_plot,
        ewok_context_plot_every_evals=ewok_context_plot_every_evals,
        blimp_likelihood_items=blimp_likelihood_items,
        blimp_likelihood_every_steps=blimp_likelihood_every_steps,
        blimp_likelihood_batch_size=int(blimp_likelihood_config.get("batch_size", 64)),
        blimp_likelihood_run_name=str(run_config.get("name", "")),
        blimp_likelihood_plot=blimp_likelihood_plot,
        blimp_likelihood_plot_every_evals=blimp_likelihood_plot_every_evals,
        entity_tracking_items=entity_tracking_items,
        entity_tracking_every_steps=entity_tracking_every_steps,
        entity_tracking_batch_size=int(entity_tracking_config.get("batch_size", 64)),
        entity_tracking_run_name=str(run_config.get("name", "")),
        entity_tracking_plot=entity_tracking_plot,
        entity_tracking_plot_every_evals=entity_tracking_plot_every_evals,
        comps_items=comps_items,
        comps_every_steps=comps_every_steps,
        comps_batch_size=int(comps_config.get("batch_size", 64)),
        comps_run_name=str(run_config.get("name", "")),
        comps_plot=comps_plot,
        comps_plot_every_evals=comps_plot_every_evals,
        words_per_token=words_per_token,
        benchmark_plot_x_axis=benchmark_plot_x_axis,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        release_cuda_memory()
        sys.exit(130)
