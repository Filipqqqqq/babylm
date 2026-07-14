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
    input_embedding_size,
    model_hidden_size,
    normalize_model_arch,
    resolve_model_vocab_size,
)
from babysteps.training.causal_lm_loop import (
    EwokContextEvalSpec,
    save_json_atomic,
    set_all_seeds,
    train_causal_lm,
)
from babysteps.training.spatial_auxiliary import (
    SpatialAuxiliarySampler,
    read_spatial_auxiliary_examples,
    split_spatial_auxiliary_examples,
)
from babysteps.training.liger import apply_liger_kernel_if_requested
from babysteps.training.nextlat import LatentDynamicsMLP, nextlat_config_from_dict
from babysteps.training.optimizers import build_optimizer
from babysteps.paths import (
    expand_env_string,
    expand_env_vars,
    resolve_path,
    validate_tokenizer_dir,
)


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
DEFAULT_WORDS_PER_TOKEN_SAMPLE_TOKENS = 4096


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


VALID_SCORE_REDUCTIONS = ("sum", "mean")


def read_score_reductions(config: dict[str, Any], section_name: str) -> tuple[str, ...]:
    """Read benchmark score reductions from config."""

    raw_value = config.get("score_reduction", "sum")
    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        raw_reductions = VALID_SCORE_REDUCTIONS if value == "both" else (value,)
    elif isinstance(raw_value, (list, tuple)):
        raw_reductions = tuple(str(value).strip().lower() for value in raw_value)
    else:
        raise ValueError(
            f"{section_name}.score_reduction must be 'sum', 'mean', 'both', "
            "or a list of those reductions."
        )

    reductions: list[str] = []
    for score_reduction in raw_reductions:
        if score_reduction not in VALID_SCORE_REDUCTIONS:
            raise ValueError(
                f"{section_name}.score_reduction must contain only 'sum' or 'mean', "
                f"got {score_reduction!r}."
            )
        if score_reduction not in reductions:
            reductions.append(score_reduction)
    if not reductions:
        raise ValueError(f"{section_name}.score_reduction must not be empty.")
    return tuple(reductions)


def read_optional_positive_float(
    config: dict[str, Any],
    key: str,
    setting_name: str,
) -> float | None:
    """Read an optional positive float config value."""

    raw_value = config.get(key)
    if raw_value is None or raw_value == "":
        return None
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"{setting_name} must be positive when set.")
    return value


def read_every_words(config: dict[str, Any], section_name: str) -> float | None:
    """Read an optional benchmark word-count evaluation interval from config."""

    return read_optional_positive_float(
        config,
        "every_words",
        f"{section_name}.every_words",
    )


def format_eval_interval(every_steps: int, every_words: float | None) -> str:
    """Return a compact human-readable evaluation interval."""

    pieces: list[str] = []
    if every_steps > 0:
        pieces.append(f"every {every_steps} steps")
    if every_words is not None:
        pieces.append(f"every {every_words:g} words")
    return " or ".join(pieces) if pieces else "not scheduled"


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
    root: Path | None = None,
) -> Path:
    """Return a default BabyLM eval-data path."""

    root = ROOT if root is None else root
    eval_root = root / "external" / "babylm-eval" / "strict" / "evaluation_data"
    return eval_root / split / Path(*relative_parts)


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
    if "ewok_context_sensitivity" in config:
        raise ValueError(
            "ewok_context_sensitivity has been removed from causal LM configs. "
            "Use ewok_context_sensitivity_spatial or ewok_context_sensitivity_full."
        )
    ewok_context_spatial_config = config.get("ewok_context_sensitivity_spatial", {})
    ewok_context_full_config = config.get("ewok_context_sensitivity_full", {})
    blimp_likelihood_config = config.get("blimp_likelihood", {})
    entity_tracking_config = config.get("entity_tracking_likelihood", {})
    comps_config = config.get("comps_likelihood", {})
    spatial_auxiliary_config = config.get("spatial_auxiliary", {})
    nextlat_config = nextlat_config_from_dict(config.get("nextlat", {}))
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

    eval_every_words = read_optional_positive_float(
        logging_config,
        "eval_every_words",
        "logging.eval_every_words",
    )
    ewok_context_spatial_every_words = read_every_words(
        ewok_context_spatial_config,
        "ewok_context_sensitivity_spatial",
    )
    ewok_context_full_every_words = read_every_words(
        ewok_context_full_config,
        "ewok_context_sensitivity_full",
    )
    blimp_likelihood_every_words = read_every_words(
        blimp_likelihood_config,
        "blimp_likelihood",
    )
    entity_tracking_every_words = read_every_words(
        entity_tracking_config,
        "entity_tracking_likelihood",
    )
    comps_every_words = read_every_words(comps_config, "comps_likelihood")
    word_interval_requested = any(
        value is not None
        for value in (
            eval_every_words,
            ewok_context_spatial_every_words,
            ewok_context_full_every_words,
            blimp_likelihood_every_words,
            entity_tracking_every_words,
            comps_every_words,
        )
    )

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
                (
                    benchmark_plot_x_axis == "words_seen"
                    or word_interval_requested
                )
                and words_per_token is None,
            ),
        )
    )
    words_per_token_sample_tokens = int(
        progress_config.get(
            "words_per_token_sample_tokens",
            logging_config.get(
                "words_per_token_sample_tokens",
                DEFAULT_WORDS_PER_TOKEN_SAMPLE_TOKENS,
            ),
        )
        or DEFAULT_WORDS_PER_TOKEN_SAMPLE_TOKENS
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
    benchmark_eval_at_start = bool(logging_config.get("benchmark_eval_at_start", True))

    extra_ewok_context_evals: list[EwokContextEvalSpec] = []
    ewok_context_spatial_every_steps = int(
        ewok_context_spatial_config.get("every_steps", 0) or 0
    )
    ewok_context_spatial_enabled = bool(
        ewok_context_spatial_config.get(
            "enabled",
            ewok_context_spatial_every_steps > 0
            or ewok_context_spatial_every_words is not None,
        )
    )
    if (
        ewok_context_spatial_enabled
        and ewok_context_spatial_every_steps <= 0
        and ewok_context_spatial_every_words is None
    ):
        raise ValueError(
            "Set ewok_context_sensitivity_spatial.every_steps or "
            "ewok_context_sensitivity_spatial.every_words when diagnostics are enabled."
        )
    ewok_context_spatial_score_reductions = read_score_reductions(
        ewok_context_spatial_config,
        "ewok_context_sensitivity_spatial",
    )
    ewok_context_spatial_plot = bool(
        ewok_context_spatial_config.get("plot", False)
    )
    ewok_context_spatial_metric_name = str(
        ewok_context_spatial_config.get(
            "metric_name",
            "ewok_context_sensitivity_spatial",
        )
    ).strip()
    if not ewok_context_spatial_metric_name:
        raise ValueError(
            "ewok_context_sensitivity_spatial.metric_name must not be empty."
        )
    ewok_context_spatial_plot_every_evals = int(
        ewok_context_spatial_config.get("plot_every_evals", 1) or 1
    )
    if ewok_context_spatial_plot and ewok_context_spatial_plot_every_evals <= 0:
        raise ValueError(
            "ewok_context_sensitivity_spatial.plot_every_evals must be positive."
        )
    if ewok_context_spatial_enabled:
        default_ewok_spatial_data_dir = default_babylm_eval_dir(
            "full_eval",
            "ewok_filtered",
            "spatial-relations.jsonl",
        )
        ewok_spatial_data_dir = resolve_path(
            ewok_context_spatial_config.get(
                "data_dir",
                default_ewok_spatial_data_dir,
            ),
            base_dir=ROOT,
        )
        ewok_context_spatial_items = read_ewok_items(ewok_spatial_data_dir)
        max_ewok_spatial_items = ewok_context_spatial_config.get("max_items")
        if max_ewok_spatial_items:
            ewok_context_spatial_items = ewok_context_spatial_items[
                : int(max_ewok_spatial_items)
            ]
        extra_ewok_context_evals.append(
            EwokContextEvalSpec(
                name=ewok_context_spatial_metric_name,
                items=ewok_context_spatial_items,
                every_steps=ewok_context_spatial_every_steps,
                every_words=ewok_context_spatial_every_words,
                batch_size=int(ewok_context_spatial_config.get("batch_size", 64)),
                score_reductions=ewok_context_spatial_score_reductions,  # type: ignore[arg-type]
                run_name=str(run_config.get("name", "")),
                plot=ewok_context_spatial_plot,
                plot_every_evals=ewok_context_spatial_plot_every_evals,
                output_subdir=str(
                    ewok_context_spatial_config.get(
                        "output_subdir",
                        "ewok_spatial",
                    )
                ),
            )
        )
        if accelerator.is_main_process:
            print(
                "Spatial EWoK context-sensitivity diagnostics: "
                f"{len(ewok_context_spatial_items):,} items, "
                f"{format_eval_interval(ewok_context_spatial_every_steps, ewok_context_spatial_every_words)}, "
                f"reductions={','.join(ewok_context_spatial_score_reductions)}, "
                f"metric_name={ewok_context_spatial_metric_name}",
                flush=True,
            )
            if ewok_context_spatial_plot:
                print(
                    "Spatial EWoK context-sensitivity plot: "
                    f"enabled every {ewok_context_spatial_plot_every_evals} eval(s)",
                    flush=True,
                )

    ewok_context_full_every_steps = int(
        ewok_context_full_config.get("every_steps", 0) or 0
    )
    ewok_context_full_enabled = bool(
        ewok_context_full_config.get(
            "enabled",
            ewok_context_full_every_steps > 0
            or ewok_context_full_every_words is not None,
        )
    )
    if (
        ewok_context_full_enabled
        and ewok_context_full_every_steps <= 0
        and ewok_context_full_every_words is None
    ):
        raise ValueError(
            "Set ewok_context_sensitivity_full.every_steps or "
            "ewok_context_sensitivity_full.every_words when diagnostics are enabled."
        )
    ewok_context_full_score_reductions = read_score_reductions(
        ewok_context_full_config,
        "ewok_context_sensitivity_full",
    )
    ewok_context_full_plot = bool(ewok_context_full_config.get("plot", False))
    ewok_context_full_metric_name = str(
        ewok_context_full_config.get(
            "metric_name",
            "ewok_context_sensitivity_full",
        )
    ).strip()
    if not ewok_context_full_metric_name:
        raise ValueError("ewok_context_sensitivity_full.metric_name must not be empty.")
    ewok_context_full_plot_every_evals = int(
        ewok_context_full_config.get("plot_every_evals", 1) or 1
    )
    if ewok_context_full_plot and ewok_context_full_plot_every_evals <= 0:
        raise ValueError(
            "ewok_context_sensitivity_full.plot_every_evals must be positive."
        )
    if ewok_context_full_enabled:
        default_ewok_full_data_dir = default_babylm_eval_dir(
            "full_eval",
            "ewok_filtered",
        )
        ewok_full_data_dir = resolve_path(
            ewok_context_full_config.get(
                "data_dir",
                default_ewok_full_data_dir,
            ),
            base_dir=ROOT,
        )
        ewok_context_full_items = read_ewok_items(ewok_full_data_dir)
        max_ewok_full_items = ewok_context_full_config.get("max_items")
        if max_ewok_full_items:
            ewok_context_full_items = ewok_context_full_items[: int(max_ewok_full_items)]
        extra_ewok_context_evals.append(
            EwokContextEvalSpec(
                name=ewok_context_full_metric_name,
                items=ewok_context_full_items,
                every_steps=ewok_context_full_every_steps,
                every_words=ewok_context_full_every_words,
                batch_size=int(ewok_context_full_config.get("batch_size", 64)),
                score_reductions=ewok_context_full_score_reductions,  # type: ignore[arg-type]
                run_name=str(run_config.get("name", "")),
                plot=ewok_context_full_plot,
                plot_every_evals=ewok_context_full_plot_every_evals,
                output_subdir=str(
                    ewok_context_full_config.get("output_subdir", "ewok_full")
                ),
            )
        )
        if accelerator.is_main_process:
            print(
                "Full EWoK context-sensitivity diagnostics: "
                f"{len(ewok_context_full_items):,} items, "
                f"{format_eval_interval(ewok_context_full_every_steps, ewok_context_full_every_words)}, "
                f"reductions={','.join(ewok_context_full_score_reductions)}, "
                f"metric_name={ewok_context_full_metric_name}",
                flush=True,
            )
            if ewok_context_full_plot:
                print(
                    "Full EWoK context-sensitivity plot: "
                    f"enabled every {ewok_context_full_plot_every_evals} eval(s)",
                    flush=True,
                )

    blimp_likelihood_items = None
    blimp_likelihood_every_steps = int(blimp_likelihood_config.get("every_steps", 0) or 0)
    blimp_likelihood_enabled = bool(
        blimp_likelihood_config.get(
            "enabled",
            blimp_likelihood_every_steps > 0
            or blimp_likelihood_every_words is not None,
        )
    )
    if (
        blimp_likelihood_enabled
        and blimp_likelihood_every_steps <= 0
        and blimp_likelihood_every_words is None
    ):
        raise ValueError(
            "Set blimp_likelihood.every_steps or blimp_likelihood.every_words "
            "when diagnostics are enabled."
        )
    blimp_likelihood_score_reductions = read_score_reductions(
        blimp_likelihood_config,
        "blimp_likelihood",
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
        )
        default_supplement_data_dir = default_babylm_eval_dir(
            "full_eval",
            "supplement_filtered",
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
                + ", "
                + format_eval_interval(
                    blimp_likelihood_every_steps,
                    blimp_likelihood_every_words,
                )
                + ", "
                + f"reductions={','.join(blimp_likelihood_score_reductions)}",
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
        entity_tracking_config.get(
            "enabled",
            entity_tracking_every_steps > 0 or entity_tracking_every_words is not None,
        )
    )
    if (
        entity_tracking_enabled
        and entity_tracking_every_steps <= 0
        and entity_tracking_every_words is None
    ):
        raise ValueError(
            "Set entity_tracking_likelihood.every_steps or "
            "entity_tracking_likelihood.every_words when diagnostics are enabled."
        )
    entity_tracking_score_reductions = read_score_reductions(
        entity_tracking_config,
        "entity_tracking_likelihood",
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
                f"{format_eval_interval(entity_tracking_every_steps, entity_tracking_every_words)}, "
                f"reductions={','.join(entity_tracking_score_reductions)}",
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
    comps_enabled = bool(
        comps_config.get(
            "enabled",
            comps_every_steps > 0 or comps_every_words is not None,
        )
    )
    if comps_enabled and comps_every_steps <= 0 and comps_every_words is None:
        raise ValueError(
            "Set comps_likelihood.every_steps or comps_likelihood.every_words "
            "when diagnostics are enabled."
        )
    comps_score_reductions = read_score_reductions(comps_config, "comps_likelihood")
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
                f"{format_eval_interval(comps_every_steps, comps_every_words)}, "
                f"reductions={','.join(comps_score_reductions)}",
                flush=True,
            )
            if comps_plot:
                print(
                    "COMPS likelihood plots: "
                    f"enabled every {comps_plot_every_evals} eval(s)",
                    flush=True,
                )

    spatial_auxiliary_sampler = None
    spatial_auxiliary_eval_items = None
    spatial_auxiliary_enabled = bool(spatial_auxiliary_config.get("enabled", False))
    spatial_auxiliary_objective = str(
        spatial_auxiliary_config.get("objective", "context_rank")
    ).strip()
    if spatial_auxiliary_objective not in {"completion_rank", "context_rank"}:
        raise ValueError(
            "spatial_auxiliary.objective must be 'completion_rank' or 'context_rank'."
        )
    spatial_auxiliary_loss_weight = float(
        spatial_auxiliary_config.get("loss_weight", 0.03)
    )
    if spatial_auxiliary_loss_weight < 0.0:
        raise ValueError("spatial_auxiliary.loss_weight must be non-negative.")
    spatial_auxiliary_temperature = float(
        spatial_auxiliary_config.get("temperature", 1.0)
    )
    if spatial_auxiliary_temperature <= 0.0:
        raise ValueError("spatial_auxiliary.temperature must be positive.")
    spatial_auxiliary_score_reduction = str(
        spatial_auxiliary_config.get("score_reduction", "mean")
    ).strip()
    if spatial_auxiliary_score_reduction not in VALID_SCORE_REDUCTIONS:
        raise ValueError("spatial_auxiliary.score_reduction must be 'sum' or 'mean'.")
    spatial_auxiliary_batch_size = int(spatial_auxiliary_config.get("batch_size", 16))
    if spatial_auxiliary_batch_size <= 0:
        raise ValueError("spatial_auxiliary.batch_size must be positive.")
    spatial_auxiliary_every_steps = int(spatial_auxiliary_config.get("every_steps", 1))
    if spatial_auxiliary_enabled and spatial_auxiliary_every_steps <= 0:
        raise ValueError("spatial_auxiliary.every_steps must be positive when enabled.")
    spatial_auxiliary_eval_every_steps = int(
        spatial_auxiliary_config.get("eval_every_steps", 0) or 0
    )
    if spatial_auxiliary_eval_every_steps < 0:
        raise ValueError("spatial_auxiliary.eval_every_steps must be non-negative.")
    spatial_auxiliary_eval_batch_size = int(
        spatial_auxiliary_config.get(
            "eval_batch_size",
            spatial_auxiliary_config.get("batch_size", 64),
        )
    )
    if spatial_auxiliary_eval_batch_size <= 0:
        raise ValueError("spatial_auxiliary.eval_batch_size must be positive.")
    spatial_auxiliary_eval_at_start = bool(
        spatial_auxiliary_config.get("eval_at_start", True)
    )

    if spatial_auxiliary_enabled:
        if not spatial_auxiliary_config.get("path"):
            raise ValueError("Set spatial_auxiliary.path when spatial auxiliary is enabled.")
        spatial_auxiliary_path = resolve_path(
            spatial_auxiliary_config["path"],
            base_dir=ROOT,
        )
        max_spatial_auxiliary_items = spatial_auxiliary_config.get("max_items")
        spatial_auxiliary_examples = read_spatial_auxiliary_examples(
            spatial_auxiliary_path,
            objective=spatial_auxiliary_objective,  # type: ignore[arg-type]
            max_items=(
                int(max_spatial_auxiliary_items)
                if max_spatial_auxiliary_items is not None
                else None
            ),
        )
        if not spatial_auxiliary_examples:
            raise ValueError("spatial_auxiliary loaded zero examples.")

        eval_path = spatial_auxiliary_config.get("eval_path")
        if eval_path:
            spatial_auxiliary_train_items = spatial_auxiliary_examples
            spatial_auxiliary_eval_items = read_spatial_auxiliary_examples(
                resolve_path(eval_path, base_dir=ROOT),
                objective=spatial_auxiliary_objective,  # type: ignore[arg-type]
                max_items=(
                    int(spatial_auxiliary_config["max_eval_items"])
                    if spatial_auxiliary_config.get("max_eval_items") is not None
                    else None
                ),
            )
        else:
            spatial_auxiliary_train_items, spatial_auxiliary_eval_items = (
                split_spatial_auxiliary_examples(
                    spatial_auxiliary_examples,
                    eval_items=int(spatial_auxiliary_config.get("eval_items", 0) or 0),
                    eval_fraction=float(
                        spatial_auxiliary_config.get("eval_fraction", 0.0) or 0.0
                    ),
                    seed=int(spatial_auxiliary_config.get("seed", seed)),
                )
            )

        spatial_auxiliary_sampler = SpatialAuxiliarySampler(
            spatial_auxiliary_train_items,
            batch_size=spatial_auxiliary_batch_size,
            seed=int(spatial_auxiliary_config.get("seed", seed)),
        )
        if not spatial_auxiliary_eval_items:
            spatial_auxiliary_eval_items = None

        if accelerator.is_main_process:
            eval_description = (
                f", eval_items={len(spatial_auxiliary_eval_items):,}, "
                f"eval_every_steps={spatial_auxiliary_eval_every_steps}"
                if spatial_auxiliary_eval_items is not None
                else ""
            )
            print(
                "Spatial auxiliary loss: "
                f"objective={spatial_auxiliary_objective}, "
                f"train_items={len(spatial_auxiliary_train_items):,}, "
                f"batch_size={spatial_auxiliary_batch_size}, "
                f"loss_weight={spatial_auxiliary_loss_weight:g}, "
                f"temperature={spatial_auxiliary_temperature:g}, "
                f"score_reduction={spatial_auxiliary_score_reduction}, "
                f"every_steps={spatial_auxiliary_every_steps}"
                f"{eval_description}",
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

    if words_per_token is None and word_interval_requested:
        raise ValueError(
            "Word-count evaluation intervals require words_per_token. Set "
            "logging.words_per_token, logging.tokens_per_word, or "
            "logging.estimate_words_per_token=true."
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
    nextlat_dynamics = None
    if nextlat_config.enabled:
        nextlat_dynamics = LatentDynamicsMLP(
            hidden_size=model_hidden_size(model),
            token_embedding_size=input_embedding_size(model),
            mlp_hidden_dim=nextlat_config.mlp_hidden_dim,
            mlp_layers=nextlat_config.mlp_layers,
        )
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
        if nextlat_config.enabled:
            nextlat_params = (
                sum(p.numel() for p in nextlat_dynamics.parameters())
                if nextlat_dynamics is not None
                else 0
            )
            print(
                "NextLat prototype enabled: "
                f"horizon={nextlat_config.horizon} "
                f"lambda_next_h={nextlat_config.lambda_next_h:g} "
                f"lambda_kl={nextlat_config.lambda_kl:g} "
                f"kl_mode={nextlat_config.kl_mode} "
                f"kl_top_k={nextlat_config.kl_top_k} "
                f"kl_warmup_start_words={nextlat_config.kl_warmup_start_words} "
                f"kl_warmup_end_words={nextlat_config.kl_warmup_end_words} "
                f"mlp_hidden_dim={nextlat_config.mlp_hidden_dim} "
                f"mlp_layers={nextlat_config.mlp_layers} "
                f"optimizer_lr={training_config.get('extra_learning_rate')} "
                f"rollout_eval_horizons={nextlat_config.rollout_eval_horizons} "
                f"rollout_eval_words={nextlat_config.rollout_eval_words} "
                f"params={nextlat_params:,}",
                flush=True,
            )

    optimizer, optimizer_info = build_optimizer(
        model,
        training_config,
        device=accelerator.device,
        extra_modules=((nextlat_dynamics,) if nextlat_dynamics is not None else ()),
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

    if nextlat_dynamics is not None:
        (
            model,
            nextlat_dynamics,
            optimizer,
            train_loader,
            val_loader,
            scheduler,
        ) = accelerator.prepare(
            model,
            nextlat_dynamics,
            optimizer,
            train_loader,
            val_loader,
            scheduler,
        )
    else:
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
        eval_every_words=eval_every_words,
        eval_max_batches=logging_config.get("eval_max_batches", 20),
        gradient_clip_norm=training_config.get("gradient_clip_norm"),
        save_every_epochs=checkpoint_config.get("save_every_epochs", 1),
        save_final=bool(checkpoint_config.get("save_final", True)),
        profile_optimizer_steps=bool(logging_config.get("profile_optimizer_steps", False)),
        nextlat_dynamics=nextlat_dynamics,
        nextlat_config=nextlat_config,
        nextlat_bos_token_id=(
            None if tokenizer.bos_token_id is None else int(tokenizer.bos_token_id)
        ),
        nextlat_eos_token_id=(
            None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)
        ),
        spatial_auxiliary_sampler=spatial_auxiliary_sampler,
        spatial_auxiliary_loss_weight=spatial_auxiliary_loss_weight,
        spatial_auxiliary_temperature=spatial_auxiliary_temperature,
        spatial_auxiliary_score_reduction=spatial_auxiliary_score_reduction,  # type: ignore[arg-type]
        spatial_auxiliary_every_steps=spatial_auxiliary_every_steps,
        spatial_auxiliary_eval_items=spatial_auxiliary_eval_items,
        spatial_auxiliary_eval_every_steps=spatial_auxiliary_eval_every_steps,
        spatial_auxiliary_eval_batch_size=spatial_auxiliary_eval_batch_size,
        spatial_auxiliary_eval_at_start=spatial_auxiliary_eval_at_start,
        benchmark_eval_at_start=benchmark_eval_at_start,
        extra_ewok_context_evals=extra_ewok_context_evals,
        blimp_likelihood_items=blimp_likelihood_items,
        blimp_likelihood_every_steps=blimp_likelihood_every_steps,
        blimp_likelihood_every_words=blimp_likelihood_every_words,
        blimp_likelihood_batch_size=int(blimp_likelihood_config.get("batch_size", 64)),
        blimp_likelihood_score_reductions=blimp_likelihood_score_reductions,  # type: ignore[arg-type]
        blimp_likelihood_run_name=str(run_config.get("name", "")),
        blimp_likelihood_plot=blimp_likelihood_plot,
        blimp_likelihood_plot_every_evals=blimp_likelihood_plot_every_evals,
        entity_tracking_items=entity_tracking_items,
        entity_tracking_every_steps=entity_tracking_every_steps,
        entity_tracking_every_words=entity_tracking_every_words,
        entity_tracking_batch_size=int(entity_tracking_config.get("batch_size", 64)),
        entity_tracking_score_reductions=entity_tracking_score_reductions,  # type: ignore[arg-type]
        entity_tracking_run_name=str(run_config.get("name", "")),
        entity_tracking_plot=entity_tracking_plot,
        entity_tracking_plot_every_evals=entity_tracking_plot_every_evals,
        comps_items=comps_items,
        comps_every_steps=comps_every_steps,
        comps_every_words=comps_every_words,
        comps_batch_size=int(comps_config.get("batch_size", 64)),
        comps_score_reductions=comps_score_reductions,  # type: ignore[arg-type]
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
