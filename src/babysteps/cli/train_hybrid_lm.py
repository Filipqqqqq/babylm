#!/usr/bin/env python3
"""Train GPT-BERT with a hybrid causal + masked-next-token objective."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from babysteps.analysis.loss_dynamics import DEFAULT_LOSS_KEYS
from babysteps.cli.train_causal_lm import (
    apply_config_overrides,
    default_babylm_eval_dir,
    estimate_words_per_token_from_dataset,
    load_yaml_config,
    resolve_output_dir,
    validate_training_inputs,
)
from babysteps.data.token_shards import build_token_shard_loader
from babysteps.evaluation.blimp_likelihood import read_blimp_items
from babysteps.evaluation.comps_likelihood import read_comps_items
from babysteps.evaluation.entity_tracking_likelihood import read_entity_tracking_items
from babysteps.models import build_causal_lm_model, normalize_model_arch
from babysteps.models.causal_lm import DEFAULT_GPT_BERT_REPO
from babysteps.paths import resolve_path, validate_tokenizer_dir
from babysteps.training.causal_lm_loop import save_json_atomic, set_all_seeds
from babysteps.training.hybrid_lm_loop import MNTP_LIKELIHOOD_BENCHMARKS
from babysteps.training.hybrid_lm_loop import train_hybrid_lm
from babysteps.training.optimizers import build_optimizer


DEFAULT_WORDS_PER_TOKEN_SAMPLE_TOKENS = 4096
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


def read_every_words(config: dict[str, Any], section_name: str) -> float | None:
    """Read an optional positive word-count evaluation interval."""

    raw_value = config.get("every_words")
    if raw_value is None or raw_value == "":
        return None
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"{section_name}.every_words must be positive when set.")
    return value


def format_eval_interval(every_steps: int, every_words: float | None) -> str:
    """Return a compact human-readable evaluation interval."""

    pieces: list[str] = []
    if every_steps > 0:
        pieces.append(f"every {every_steps} steps")
    if every_words is not None:
        pieces.append(f"every {every_words:g} words")
    return " or ".join(pieces) if pieces else "not scheduled"


def estimate_word_interval_eval_steps(
    *,
    every_words: float | None,
    words_per_token: float | None,
    seq_len: int,
    batch_size: int,
    grad_accum: int,
    world_size: int,
    total_update_steps: int,
) -> list[int]:
    """Estimate optimizer-update steps that will cross word-count thresholds."""

    if every_words is None or words_per_token is None or total_update_steps <= 0:
        return []

    tokens_per_update = (
        int(seq_len)
        * int(batch_size)
        * max(1, int(grad_accum))
        * max(1, int(world_size))
    )
    words_per_update = tokens_per_update * float(words_per_token)
    if words_per_update <= 0:
        return []

    threshold_count = int(math.floor(total_update_steps * words_per_update / every_words))
    steps: list[int] = []
    for threshold_index in range(1, threshold_count + 1):
        step = int(math.ceil(threshold_index * every_words / words_per_update))
        step = min(max(1, step), int(total_update_steps))
        if not steps or step != steps[-1]:
            steps.append(step)
    return steps


def estimate_step_interval_eval_steps(every_steps: int, total_update_steps: int) -> list[int]:
    """Return update steps triggered by a fixed step interval."""

    if every_steps <= 0 or total_update_steps <= 0:
        return []
    return list(range(int(every_steps), int(total_update_steps) + 1, int(every_steps)))


def merge_eval_steps(*step_lists: list[int]) -> list[int]:
    """Merge scheduled update-step lists while preserving ascending order."""

    return sorted({step for steps in step_lists for step in steps})


def format_eval_step_preview(steps: list[int], *, preview: int = 20) -> str:
    """Format a concise preview of scheduled evaluation update steps."""

    if not steps:
        return "0 evals"
    shown = ", ".join(str(step) for step in steps[:preview])
    if len(steps) > preview:
        shown += f", ... last={steps[-1]}"
    return f"{len(steps)} evals at update steps: {shown}"


def read_loss_keys(config: dict[str, Any]) -> tuple[str, ...]:
    """Read loss-dynamics metric keys from config."""

    raw_value = config.get("loss_keys")
    if raw_value is None:
        return DEFAULT_LOSS_KEYS
    if isinstance(raw_value, str):
        loss_keys = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    elif isinstance(raw_value, (list, tuple)):
        loss_keys = tuple(str(part).strip() for part in raw_value if str(part).strip())
    else:
        raise ValueError("loss_dynamics.loss_keys must be a string or list of strings.")
    if not loss_keys:
        raise ValueError("loss_dynamics.loss_keys must not be empty.")
    return loss_keys


def resolve_padding_for_gpt_bert(tokenizer: Any) -> None:
    """Set a padding token when the baseline tokenizer omits one."""

    if tokenizer.pad_token is not None:
        return
    if getattr(tokenizer, "sep_token", None) is not None:
        tokenizer.pad_token = tokenizer.sep_token
    elif getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError("Hybrid GPT-BERT training requires a pad, sep, or eos token.")


def validate_hybrid_config(config: dict[str, Any], tokenizer: Any) -> None:
    """Fail early for GPT-BERT-specific config/tokenizer mismatches."""

    architecture = normalize_model_arch(config["model"].get("architecture", "gpt_bert"))
    if architecture != "gpt_bert":
        raise ValueError(
            f"train_hybrid_lm.py supports only model.architecture=gpt_bert, got {architecture!r}."
        )
    if tokenizer.mask_token_id is None:
        raise ValueError("GPT-BERT hybrid training requires tokenizer.mask_token_id.")
    data_format = str(config["data"].get("format", "token_shards")).strip().lower()
    if data_format not in {"token_shards", "shards", "bin"}:
        raise ValueError("Hybrid GPT-BERT training currently supports only token_shards data.")


def default_mntp_benchmark_config(name: str) -> dict[str, Any]:
    """Return default BabyLM eval-data settings for one MNTP benchmark."""

    if name == "blimp":
        return {
            "data_dir": default_babylm_eval_dir(
                "full_eval",
                "blimp_filtered",
                fallback_split="fast_eval",
                fallback_relative_parts=("blimp_fast",),
            )
        }
    if name == "blimp_supplement":
        return {
            "data_dir": default_babylm_eval_dir(
                "full_eval",
                "supplement_filtered",
                fallback_split="fast_eval",
                fallback_relative_parts=("supplement_fast",),
            )
        }
    if name == "entity_tracking":
        return {
            "data_dir": default_babylm_eval_dir(
                "full_eval",
                "entity_tracking",
                fallback_split="fast_eval",
                fallback_relative_parts=("entity_tracking_fast",),
            )
        }
    if name == "comps":
        return {"data_dir": default_babylm_eval_dir("full_eval", "comps")}
    raise ValueError(f"Unsupported MNTP benchmark: {name!r}.")


def read_mntp_likelihood_items(config: dict[str, Any]) -> dict[str, list[Any]]:
    """Load configured MNTP BabyLM benchmark items."""

    raw_benchmarks = config.get("benchmarks")
    if raw_benchmarks is None:
        raw_benchmarks = {
            name: default_mntp_benchmark_config(name)
            for name in MNTP_LIKELIHOOD_BENCHMARKS
        }
    if not isinstance(raw_benchmarks, dict):
        raise ValueError("mntp_likelihood.benchmarks must be a mapping.")

    items_by_benchmark: dict[str, list[Any]] = {}
    global_max_items = config.get("max_items")
    for benchmark_name, benchmark_config in raw_benchmarks.items():
        benchmark_name = str(benchmark_name)
        if benchmark_config is None:
            benchmark_config = {}
        if not isinstance(benchmark_config, dict):
            raise ValueError(
                "Each mntp_likelihood.benchmarks entry must be a mapping, "
                f"got {benchmark_name!r}."
            )
        if benchmark_config.get("enabled", True) is False:
            continue
        if benchmark_name not in MNTP_LIKELIHOOD_BENCHMARKS:
            raise ValueError(f"Unsupported MNTP benchmark: {benchmark_name!r}.")

        merged = {**default_mntp_benchmark_config(benchmark_name), **benchmark_config}
        data_dir = resolve_path(merged["data_dir"], base_dir=ROOT)
        max_items = merged.get("max_items", global_max_items)
        if benchmark_name in {"blimp", "blimp_supplement"}:
            items = read_blimp_items(data_dir)
        elif benchmark_name == "entity_tracking":
            items = read_entity_tracking_items(data_dir)
        elif benchmark_name == "comps":
            items = read_comps_items(data_dir)
        else:
            raise ValueError(f"Unsupported MNTP benchmark: {benchmark_name!r}.")
        items_by_benchmark[benchmark_name] = items[: int(max_items)] if max_items else items

    if not items_by_benchmark:
        raise ValueError("mntp_likelihood is enabled but no benchmarks were loaded.")
    return items_by_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            ROOT
            / "configs"
            / "train"
            / "gpt_bert_baseline_causal_focus_babylm2026_strict_small.yaml"
        ),
        help="Path to a YAML hybrid training config.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Override a config value after loading YAML.",
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
    hybrid_config = config.get("hybrid", {})
    logging_config = config.get("logging", {})
    checkpoint_config = config.get("checkpointing", {})
    progress_config = config.get("progress", {})
    mntp_likelihood_config = config.get("mntp_likelihood", {})
    loss_dynamics_config = config.get("loss_dynamics", {})

    mntp_likelihood_every_steps = int(mntp_likelihood_config.get("every_steps", 0) or 0)
    mntp_likelihood_every_words = read_every_words(
        mntp_likelihood_config,
        "mntp_likelihood",
    )
    mntp_likelihood_enabled = bool(
        mntp_likelihood_config.get(
            "enabled",
            mntp_likelihood_every_steps > 0 or mntp_likelihood_every_words is not None,
        )
    )
    if (
        mntp_likelihood_enabled
        and mntp_likelihood_every_steps <= 0
        and mntp_likelihood_every_words is None
    ):
        raise ValueError(
            "Set mntp_likelihood.every_steps or mntp_likelihood.every_words "
            "when MNTP benchmarks are enabled."
        )
    mntp_likelihood_score_reductions = read_score_reductions(
        mntp_likelihood_config,
        "mntp_likelihood",
    )

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

    word_interval_requested = mntp_likelihood_every_words is not None
    estimate_words_per_token = bool(
        progress_config.get(
            "estimate_words_per_token",
            logging_config.get(
                "estimate_words_per_token",
                word_interval_requested and words_per_token is None,
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
    benchmark_plot_x_axis = str(
        progress_config.get(
            "benchmark_plot_x_axis",
            logging_config.get("benchmark_plot_x_axis", "step"),
        )
    )

    mntp_likelihood_plot = bool(mntp_likelihood_config.get("plot", False))
    mntp_likelihood_plot_every_evals = int(
        mntp_likelihood_config.get("plot_every_evals", 1) or 1
    )
    if mntp_likelihood_plot and mntp_likelihood_plot_every_evals <= 0:
        raise ValueError("mntp_likelihood.plot_every_evals must be positive.")
    loss_dynamics_plot = bool(
        loss_dynamics_config.get("plot", mntp_likelihood_plot)
    )
    loss_dynamics_plot_every_evals = int(
        loss_dynamics_config.get(
            "plot_every_evals",
            mntp_likelihood_plot_every_evals,
        )
        or 1
    )
    if loss_dynamics_plot and loss_dynamics_plot_every_evals <= 0:
        raise ValueError("loss_dynamics.plot_every_evals must be positive.")
    loss_dynamics_x_axis = str(
        loss_dynamics_config.get("x_axis", benchmark_plot_x_axis)
    )
    loss_dynamics_loss_keys = read_loss_keys(loss_dynamics_config)

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

    tokenizer_name = config["tokenizer"].get("name")
    if not tokenizer_name:
        raise ValueError("Hybrid GPT-BERT configs must set tokenizer.name.")
    tokenizer_path = validate_tokenizer_dir(tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path or tokenizer_name),
        trust_remote_code=bool(config["tokenizer"].get("trust_remote_code", False)),
    )
    resolve_padding_for_gpt_bert(tokenizer)
    validate_hybrid_config(config, tokenizer)

    model_config = config["model"]
    seq_len = int(data_config.get("seq_len", model_config.get("context_length", 512)))
    batch_size = int(training_config.get("batch_size", 16))
    num_workers = int(data_config.get("num_workers", 0))
    data_dir = data_config["data_dir"]
    train_loader, train_dataset = build_token_shard_loader(
        data_dir,
        split=str(data_config.get("train_split", "train")),
        batch_size=batch_size,
        seq_len=seq_len,
        shuffle=True,
        max_blocks=data_config.get("max_train_blocks"),
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
        max_blocks=data_config.get("max_val_blocks"),
        max_shards=data_config.get("max_val_shards"),
        num_workers=num_workers,
        seed=seed,
        pin_memory=torch.cuda.is_available(),
    )
    validate_training_inputs(config, tokenizer, train_dataset, val_dataset)

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
        progress_config["words_per_token_sample_tokens"] = int(estimate["sample_tokens"])
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
            "Word-count MNTP benchmark intervals require words_per_token. Set "
            "progress.words_per_token, progress.tokens_per_word, or "
            "progress.estimate_words_per_token=true."
        )

    mntp_likelihood_items = None
    if mntp_likelihood_enabled:
        mntp_likelihood_items = read_mntp_likelihood_items(mntp_likelihood_config)
        if accelerator.is_main_process:
            print(
                "MNTP likelihood benchmarks: "
                + ", ".join(
                    f"{name}={len(items):,} items"
                    for name, items in mntp_likelihood_items.items()
                )
                + ", "
                + format_eval_interval(
                    mntp_likelihood_every_steps,
                    mntp_likelihood_every_words,
                )
                + ", "
                + f"reductions={','.join(mntp_likelihood_score_reductions)}",
                flush=True,
            )
            if mntp_likelihood_plot:
                print(
                    "MNTP likelihood plots: "
                    f"enabled every {mntp_likelihood_plot_every_evals} eval(s), "
                    f"x_axis={benchmark_plot_x_axis}",
                    flush=True,
                )
            if loss_dynamics_plot:
                print(
                    "Training loss dynamics plots: "
                    f"enabled every {loss_dynamics_plot_every_evals} benchmark eval(s), "
                    f"x_axis={loss_dynamics_x_axis}",
                    flush=True,
                )

    model = build_causal_lm_model(config, tokenizer)
    if accelerator.is_main_process:
        params = sum(param.numel() for param in model.parameters())
        print("Model architecture: gpt_bert", flush=True)
        print(f"Model parameters: {params:,}", flush=True)
        print(f"Train shard blocks: {len(train_dataset):,}", flush=True)
        print(f"Val shard blocks: {len(val_dataset):,}", flush=True)

    optimizer, optimizer_info = build_optimizer(
        model,
        training_config,
        device=accelerator.device,
    )
    if accelerator.is_main_process:
        print(f"Optimizer: {optimizer_info.name}", flush=True)
        print(f"Decayed tensors: {optimizer_info.primary_tensors}", flush=True)
        print(f"Non-decayed tensors: {optimizer_info.auxiliary_tensors}", flush=True)

    epochs = int(training_config.get("epochs", 1))
    grad_accum = int(training_config.get("gradient_accumulation_steps", 1))
    updates_per_epoch = max(1, (len(train_loader) + grad_accum - 1) // grad_accum)
    total_update_steps = epochs * updates_per_epoch
    warmup_steps = training_config.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = int(float(training_config.get("warmup_ratio", 0.016)) * total_update_steps)

    if accelerator.is_main_process:
        tokens_per_update = (
            seq_len
            * batch_size
            * max(1, grad_accum)
            * max(1, accelerator.num_processes)
        )
        print(
            "Training schedule estimate: "
            f"updates_per_epoch={updates_per_epoch:,}, "
            f"total_update_steps={total_update_steps:,}, "
            f"tokens/update~={tokens_per_update:,}",
            flush=True,
        )
        if words_per_token is not None:
            print(
                "Word-progress estimate: "
                f"words/update~={tokens_per_update * words_per_token:,.1f}",
                flush=True,
            )

        validation_steps = estimate_step_interval_eval_steps(
            int(logging_config.get("eval_every_steps", 500)),
            total_update_steps,
        )
        print(
            "Validation loss eval estimate: "
            f"{format_eval_step_preview(validation_steps)}",
            flush=True,
        )

        if mntp_likelihood_enabled:
            mntp_step_steps = estimate_step_interval_eval_steps(
                mntp_likelihood_every_steps,
                total_update_steps,
            )
            mntp_word_steps = estimate_word_interval_eval_steps(
                every_words=mntp_likelihood_every_words,
                words_per_token=words_per_token,
                seq_len=seq_len,
                batch_size=batch_size,
                grad_accum=grad_accum,
                world_size=accelerator.num_processes,
                total_update_steps=total_update_steps,
            )
            mntp_eval_steps = merge_eval_steps(mntp_step_steps, mntp_word_steps)
            startup_eval = bool(logging_config.get("benchmark_eval_at_start", True))
            suffix = "; plus startup eval at step 0" if startup_eval else ""
            print(
                "MNTP benchmark eval estimate: "
                f"{format_eval_step_preview(mntp_eval_steps)}{suffix}",
                flush=True,
            )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_steps),
        num_training_steps=total_update_steps,
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
    )

    train_hybrid_lm(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        output_dir=output_dir,
        epochs=epochs,
        total_update_steps=total_update_steps,
        hybrid_config=hybrid_config,
        model_repo=str(model_config.get("repo", DEFAULT_GPT_BERT_REPO)),
        seed=seed,
        log_every_steps=int(logging_config.get("log_every_steps", 50)),
        eval_every_steps=int(logging_config.get("eval_every_steps", 500)),
        eval_max_batches=logging_config.get("eval_max_batches", 20),
        gradient_clip_norm=training_config.get("gradient_clip_norm"),
        save_every_epochs=checkpoint_config.get("save_every_epochs", 1),
        save_final=bool(checkpoint_config.get("save_final", True)),
        benchmark_eval_at_start=bool(logging_config.get("benchmark_eval_at_start", True)),
        mntp_likelihood_items=mntp_likelihood_items,
        mntp_likelihood_every_steps=mntp_likelihood_every_steps,
        mntp_likelihood_every_words=mntp_likelihood_every_words,
        mntp_likelihood_batch_size=int(mntp_likelihood_config.get("batch_size", 64)),
        mntp_likelihood_score_reductions=mntp_likelihood_score_reductions,
        mntp_likelihood_run_name=str(run_config.get("name", "")),
        mntp_likelihood_plot=mntp_likelihood_plot,
        mntp_likelihood_plot_every_evals=mntp_likelihood_plot_every_evals,
        benchmark_plot_x_axis=benchmark_plot_x_axis,
        loss_dynamics_plot=loss_dynamics_plot,
        loss_dynamics_plot_every_evals=loss_dynamics_plot_every_evals,
        loss_dynamics_x_axis=loss_dynamics_x_axis,
        loss_dynamics_loss_keys=loss_dynamics_loss_keys,
        loss_dynamics_run_name=str(run_config.get("name", "")),
        words_per_token=words_per_token,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
