"""A small reusable causal language-model training loop."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from babysteps.analysis.blimp_likelihood_dynamics import (
    plot_blimp_likelihood_summary,
    plot_overall_accuracy,
)
from babysteps.analysis.comps_dynamics import plot_comps_summary
from babysteps.analysis.entity_tracking_dynamics import plot_entity_tracking_summary
from babysteps.analysis.ewok_context_dynamics import plot_ewok_context_summary
from babysteps.evaluation.blimp_likelihood import (
    BlimpItem,
    append_item_records_jsonl as append_blimp_item_records_jsonl,
    append_summary_csv as append_blimp_summary_csv,
    evaluate_blimp_likelihood,
)
from babysteps.evaluation.comps_likelihood import (
    CompsItem,
    append_item_records_jsonl as append_comps_item_records_jsonl,
    append_summary_csv as append_comps_summary_csv,
    evaluate_comps_likelihood,
)
from babysteps.evaluation.entity_tracking_likelihood import (
    EntityTrackingItem,
    append_item_records_jsonl as append_entity_tracking_item_records_jsonl,
    append_summary_csv as append_entity_tracking_summary_csv,
    evaluate_entity_tracking_likelihood,
)
from babysteps.evaluation.ewok_context_sensitivity import (
    EwokItem,
    ScoreReduction,
    append_item_records_jsonl,
    append_summary_csv,
    evaluate_ewok_context_sensitivity,
)
from babysteps.training.branch_scoring import BranchScoreReduction
from babysteps.training.spatial_auxiliary import (
    SpatialAuxiliaryExample,
    SpatialAuxiliarySampler,
    compute_spatial_auxiliary_loss,
    evaluate_spatial_auxiliary,
)


def set_all_seeds(seed: int, *, deterministic: bool = False) -> None:
    """Set the common random seeds used during training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_json_atomic(data: Any, path: str | Path) -> None:
    """Write JSON through a temporary file and replace atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def compute_causal_lm_loss(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute next-token cross-entropy over non-padding targets."""

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    vocab_size = logits.size(-1)

    # The dataloader has already shifted labels one token to the right, so each
    # logit position predicts the matching label position.
    token_losses = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)

    # Average by real target tokens instead of by sequence length so padding
    # does not change the loss scale across batches.
    mask = loss_mask.to(dtype=token_losses.dtype)
    token_count = mask.sum().clamp_min(1)
    loss = (token_losses * mask).sum() / token_count
    return loss, token_count.detach()


def evaluate_causal_lm_loss(
    model: torch.nn.Module,
    dataloader: Iterable,
    accelerator,
    *,
    max_batches: int | None = None,
) -> float:
    """Evaluate token-averaged causal LM loss."""

    model.eval()
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_tokens = torch.tensor(0.0, device=accelerator.device)

    batches = dataloader if max_batches is None else islice(dataloader, max_batches)
    with torch.no_grad():
        for input_ids, labels, loss_mask, attention_mask in batches:
            with accelerator.autocast():
                loss, token_count = compute_causal_lm_loss(
                    model,
                    input_ids=input_ids,
                    labels=labels,
                    loss_mask=loss_mask,
                    attention_mask=attention_mask,
                )
            total_loss += loss.detach() * token_count
            total_tokens += token_count

    if accelerator.num_processes > 1:
        # Each process sees a shard of the validation loader; reduce token sums
        # before forming the final token-averaged loss.
        total_loss = accelerator.reduce(total_loss, reduction="sum")
        total_tokens = accelerator.reduce(total_tokens, reduction="sum")

    model.train()
    return (total_loss / total_tokens.clamp_min(1)).item()


def blimp_benchmark_label(benchmark_name: str) -> str:
    if benchmark_name == "blimp":
        return "BLiMP"
    if benchmark_name == "blimp_supplement":
        return "BLiMP Supplement"
    return benchmark_name.replace("_", " ").title()


def blimp_benchmark_group_scopes(benchmark_name: str) -> tuple[str, ...]:
    if benchmark_name == "blimp":
        return ("field", "linguistics_term")
    if benchmark_name == "blimp_supplement":
        return ("uid",)
    return ("uid", "field", "linguistics_term")


BENCHMARK_X_AXES = {"step", "tokens_seen", "words_seen"}


@dataclass
class EwokContextEvalSpec:
    """One scheduled EWoK context-sensitivity diagnostic."""

    name: str
    items: list[EwokItem]
    every_steps: int = 0
    every_words: float | None = None
    batch_size: int = 64
    score_reductions: tuple[ScoreReduction, ...] = ("sum",)
    run_name: str | None = None
    plot: bool = False
    plot_every_evals: int = 1
    output_subdir: str | None = None


def validate_every_words(name: str, every_words: float | int | None) -> float | None:
    """Normalize an optional word-count evaluation interval."""

    if every_words is None:
        return None
    every_words = float(every_words)
    if every_words <= 0:
        raise ValueError(f"{name} must be positive when set.")
    return every_words


def word_interval_due(
    *,
    words_seen: float,
    next_words: float | None,
) -> bool:
    """Return whether a word-count scheduled event is due."""

    return next_words is not None and words_seen >= next_words


def advance_word_interval(
    *,
    next_words: float | None,
    every_words: float | None,
    words_seen: float,
) -> float | None:
    """Advance a word-count threshold past the current progress point."""

    if next_words is None or every_words is None:
        return next_words
    while next_words <= words_seen:
        next_words += every_words
    return next_words


def progress_metadata(
    *,
    epoch: int,
    step: int,
    tokens_seen: float,
    words_per_token: float | None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Build training-progress metadata shared by benchmark outputs."""

    metadata: dict[str, Any] = {
        "epoch": epoch,
        "step": step,
        "tokens_seen": int(tokens_seen),
    }
    if words_per_token is not None:
        metadata["words_seen"] = float(tokens_seen) * words_per_token
    if run_name:
        metadata["run_name"] = run_name
    return metadata


def save_hf_checkpoint(
    model: torch.nn.Module,
    tokenizer,
    accelerator,
    output_dir: str | Path,
    *,
    name: str,
) -> None:
    """Save a Hugging Face model/tokenizer checkpoint."""

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    # Only the main process writes files, while accelerator.save handles the
    # distributed-safe tensor serialization details.
    checkpoint_dir = Path(output_dir) / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(checkpoint_dir, save_function=accelerator.save)
    tokenizer.save_pretrained(checkpoint_dir)


def train_causal_lm(
    *,
    model: torch.nn.Module,
    tokenizer,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    accelerator,
    output_dir: str | Path,
    epochs: int,
    log_every_steps: int = 50,
    eval_every_steps: int = 500,
    eval_every_words: float | None = None,
    eval_max_batches: int | None = 20,
    gradient_clip_norm: float | None = None,
    save_every_epochs: int | None = 1,
    save_final: bool = True,
    profile_optimizer_steps: bool = False,
    spatial_auxiliary_sampler: SpatialAuxiliarySampler | None = None,
    spatial_auxiliary_loss_weight: float = 0.03,
    spatial_auxiliary_temperature: float = 1.0,
    spatial_auxiliary_score_reduction: BranchScoreReduction = "mean",
    spatial_auxiliary_every_steps: int = 1,
    spatial_auxiliary_eval_items: list[SpatialAuxiliaryExample] | None = None,
    spatial_auxiliary_eval_every_steps: int = 0,
    spatial_auxiliary_eval_batch_size: int = 64,
    spatial_auxiliary_eval_at_start: bool = True,
    benchmark_eval_at_start: bool = True,
    extra_ewok_context_evals: list[EwokContextEvalSpec] | None = None,
    blimp_likelihood_items: dict[str, list[BlimpItem]] | None = None,
    blimp_likelihood_every_steps: int = 0,
    blimp_likelihood_every_words: float | None = None,
    blimp_likelihood_batch_size: int = 64,
    blimp_likelihood_score_reductions: tuple[ScoreReduction, ...] = ("sum",),
    blimp_likelihood_run_name: str | None = None,
    blimp_likelihood_plot: bool = False,
    blimp_likelihood_plot_every_evals: int = 1,
    entity_tracking_items: list[EntityTrackingItem] | None = None,
    entity_tracking_every_steps: int = 0,
    entity_tracking_every_words: float | None = None,
    entity_tracking_batch_size: int = 64,
    entity_tracking_score_reductions: tuple[ScoreReduction, ...] = ("sum",),
    entity_tracking_run_name: str | None = None,
    entity_tracking_plot: bool = False,
    entity_tracking_plot_every_evals: int = 1,
    comps_items: list[CompsItem] | None = None,
    comps_every_steps: int = 0,
    comps_every_words: float | None = None,
    comps_batch_size: int = 64,
    comps_score_reductions: tuple[ScoreReduction, ...] = ("sum",),
    comps_run_name: str | None = None,
    comps_plot: bool = False,
    comps_plot_every_evals: int = 1,
    words_per_token: float | None = None,
    benchmark_plot_x_axis: str = "step",
) -> list[dict[str, Any]]:
    """Train a causal LM and return the metric history."""

    if benchmark_plot_x_axis not in BENCHMARK_X_AXES:
        raise ValueError(
            "benchmark_plot_x_axis must be one of "
            f"{sorted(BENCHMARK_X_AXES)}, got {benchmark_plot_x_axis!r}."
        )
    if benchmark_plot_x_axis == "words_seen" and words_per_token is None:
        raise ValueError("benchmark_plot_x_axis='words_seen' requires words_per_token.")
    if spatial_auxiliary_loss_weight < 0.0:
        raise ValueError("spatial_auxiliary_loss_weight must be non-negative.")
    if spatial_auxiliary_temperature <= 0.0:
        raise ValueError("spatial_auxiliary_temperature must be positive.")
    if spatial_auxiliary_sampler is not None and spatial_auxiliary_every_steps <= 0:
        raise ValueError("spatial_auxiliary_every_steps must be positive when enabled.")
    if spatial_auxiliary_eval_every_steps < 0:
        raise ValueError("spatial_auxiliary_eval_every_steps must be non-negative.")
    ewok_context_eval_specs: list[EwokContextEvalSpec] = []
    if extra_ewok_context_evals:
        ewok_context_eval_specs.extend(extra_ewok_context_evals)
    seen_ewok_metric_names: set[str] = set()
    for spec in ewok_context_eval_specs:
        spec.name = str(spec.name).strip()
        if not spec.name:
            raise ValueError("EWoK context eval spec names must not be empty.")
        if spec.name in seen_ewok_metric_names:
            raise ValueError(f"Duplicate EWoK context eval metric name: {spec.name!r}.")
        seen_ewok_metric_names.add(spec.name)
        if spec.every_steps < 0:
            raise ValueError(f"{spec.name}.every_steps must be non-negative.")
        spec.every_words = validate_every_words(
            f"{spec.name}.every_words",
            spec.every_words,
        )
        if spec.every_steps <= 0 and spec.every_words is None:
            raise ValueError(
                f"Set {spec.name}.every_steps or {spec.name}.every_words when enabled."
            )
        if spec.batch_size <= 0:
            raise ValueError(f"{spec.name}.batch_size must be positive.")
        if not spec.score_reductions:
            raise ValueError(f"{spec.name}.score_reductions must not be empty.")
        if spec.plot_every_evals <= 0:
            raise ValueError(f"{spec.name}.plot_every_evals must be positive.")

    eval_every_words = validate_every_words("eval_every_words", eval_every_words)
    blimp_likelihood_every_words = validate_every_words(
        "blimp_likelihood_every_words",
        blimp_likelihood_every_words,
    )
    entity_tracking_every_words = validate_every_words(
        "entity_tracking_every_words",
        entity_tracking_every_words,
    )
    comps_every_words = validate_every_words("comps_every_words", comps_every_words)
    if (
        words_per_token is None
        and any(
            value is not None
            for value in (
                eval_every_words,
                *(spec.every_words for spec in ewok_context_eval_specs),
                blimp_likelihood_every_words,
                entity_tracking_every_words,
                comps_every_words,
            )
        )
    ):
        raise ValueError("Word-count evaluation intervals require words_per_token.")

    for name, score_reductions in (
        ("blimp_likelihood_score_reductions", blimp_likelihood_score_reductions),
        ("entity_tracking_score_reductions", entity_tracking_score_reductions),
        ("comps_score_reductions", comps_score_reductions),
    ):
        if not score_reductions:
            raise ValueError(f"{name} must not be empty.")

    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics_history.json"
    benchmark_output_dir = output_dir / "benchmarks"
    blimp_combined_plot_path = benchmark_output_dir / "blimp_likelihood_overall_accuracy.png"
    entity_tracking_output_dir = benchmark_output_dir / "entity_tracking"
    entity_tracking_items_path = entity_tracking_output_dir / "likelihood_items.jsonl"
    entity_tracking_summary_path = entity_tracking_output_dir / "likelihood_summary.csv"
    comps_output_dir = benchmark_output_dir / "comps"
    comps_items_path = comps_output_dir / "likelihood_items.jsonl"
    comps_summary_path = comps_output_dir / "likelihood_summary.csv"
    history: list[dict[str, Any]] = []
    global_step = 0
    cumulative_tokens_seen = 0.0
    next_validation_words = eval_every_words
    next_ewok_context_words = {
        spec.name: spec.every_words for spec in ewok_context_eval_specs
    }
    next_blimp_likelihood_words = blimp_likelihood_every_words
    next_entity_tracking_words = entity_tracking_every_words
    next_comps_words = comps_every_words
    ewok_eval_counts = {spec.name: 0 for spec in ewok_context_eval_specs}
    blimp_eval_count = 0
    entity_tracking_eval_count = 0
    comps_eval_count = 0

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    def reduction_filter(score_reductions: tuple[ScoreReduction, ...]) -> ScoreReduction | None:
        return score_reductions[0] if len(score_reductions) == 1 else None

    def run_ewok_context_eval(
        spec: EwokContextEvalSpec,
        *,
        epoch: int,
        step: int,
        tokens_seen: float,
    ) -> None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            output_subdir = spec.output_subdir or spec.name
            ewok_output_dir = benchmark_output_dir / output_subdir
            ewok_items_path = ewok_output_dir / "context_items.jsonl"
            ewok_summary_path = ewok_output_dir / "context_summary.csv"
            ewok_plot_path = ewok_output_dir / "context_domain_dynamics.png"
            metadata = progress_metadata(
                epoch=epoch,
                step=step,
                tokens_seen=tokens_seen,
                words_per_token=words_per_token,
                run_name=spec.run_name,
            )
            for score_reduction in spec.score_reductions:
                result = evaluate_ewok_context_sensitivity(
                    model=unwrapped_model,
                    tokenizer=tokenizer,
                    items=spec.items,
                    batch_size=spec.batch_size,
                    score_reduction=score_reduction,
                    device=accelerator.device,
                )
                append_item_records_jsonl(
                    result.item_records,
                    ewok_items_path,
                    metadata=metadata,
                )
                append_summary_csv(
                    result.summary_rows,
                    ewok_summary_path,
                    metadata=metadata,
                )
                overall = next(
                    row
                    for row in result.summary_rows
                    if row["group_scope"] == "overall" and row["group_name"] == "all"
                )
                record = {
                    "type": spec.name,
                    **metadata,
                    "score_reduction": score_reduction,
                    "score": overall["score"],
                    "mean_k_margin": overall["mean_k_margin"],
                    "items": overall["items"],
                    "target_token_count_mismatches": overall[
                        "target_token_count_mismatches"
                    ],
                    "sum_mean_sign_disagreements": overall[
                        "sum_mean_sign_disagreements"
                    ],
                }
                history.append(record)
                print(
                    f"[step {step:06d}] "
                    f"{spec.name}_score={overall['score']:.4f} "
                    f"reduction={score_reduction} "
                    f"mean_k_margin={overall['mean_k_margin']:.4f} "
                    "sum_mean_sign_disagreements="
                    f"{overall['sum_mean_sign_disagreements']}",
                    flush=True,
                )
            ewok_eval_counts[spec.name] += 1
            if (
                spec.plot
                and ewok_eval_counts[spec.name] % max(1, spec.plot_every_evals) == 0
            ):
                plot_ewok_context_summary(
                    summary_csv=ewok_summary_path,
                    output_path=ewok_plot_path,
                    run_name=spec.run_name,
                    score_reduction=reduction_filter(spec.score_reductions),
                    x_axis=benchmark_plot_x_axis,
                )
            save_json_atomic(history, metrics_path)
        accelerator.wait_for_everyone()

    def run_spatial_auxiliary_eval(*, epoch: int, step: int, tokens_seen: float) -> None:
        if spatial_auxiliary_eval_items is None:
            return
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            metadata = progress_metadata(
                epoch=epoch,
                step=step,
                tokens_seen=tokens_seen,
                words_per_token=words_per_token,
            )
            result = evaluate_spatial_auxiliary(
                model=unwrapped_model,
                tokenizer=tokenizer,
                examples=spatial_auxiliary_eval_items,
                batch_size=spatial_auxiliary_eval_batch_size,
                temperature=spatial_auxiliary_temperature,
                score_reduction=spatial_auxiliary_score_reduction,
                device=accelerator.device,
            )
            record = {
                "type": "spatial_auxiliary_eval",
                **metadata,
                "score_reduction": spatial_auxiliary_score_reduction,
                "temperature": spatial_auxiliary_temperature,
                **result,
            }
            history.append(record)
            message = (
                f"[step {step:06d}] "
                f"spatial_aux_score={float(result['score']):.4f} "
                f"loss={float(result['loss']):.4f} "
                f"objective={result['objective']} "
                f"mean_margin={float(result['mean_margin']):.4f}"
            )
            if result["objective"] == "context_rank":
                message += (
                    f" k1={float(result['k1_score']):.4f} "
                    f"k2={float(result['k2_score']):.4f}"
                )
            print(message, flush=True)
            save_json_atomic(history, metrics_path)
        accelerator.wait_for_everyone()

    def run_blimp_likelihood_eval(*, epoch: int, step: int, tokens_seen: float) -> None:
        nonlocal blimp_eval_count

        if blimp_likelihood_items is None:
            return
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            metadata = progress_metadata(
                epoch=epoch,
                step=step,
                tokens_seen=tokens_seen,
                words_per_token=words_per_token,
                run_name=blimp_likelihood_run_name,
            )

            summary_csvs: dict[str, Path] = {}
            for benchmark_name, items in blimp_likelihood_items.items():
                benchmark_dir = benchmark_output_dir / benchmark_name
                items_path = benchmark_dir / "likelihood_items.jsonl"
                summary_path = benchmark_dir / "likelihood_summary.csv"
                summary_csvs[benchmark_name] = summary_path
                for score_reduction in blimp_likelihood_score_reductions:
                    result = evaluate_blimp_likelihood(
                        model=unwrapped_model,
                        tokenizer=tokenizer,
                        items=items,
                        batch_size=blimp_likelihood_batch_size,
                        score_reduction=score_reduction,
                        device=accelerator.device,
                    )
                    append_blimp_item_records_jsonl(
                        result.item_records,
                        items_path,
                        metadata=metadata,
                    )
                    append_blimp_summary_csv(
                        result.summary_rows,
                        summary_path,
                        metadata=metadata,
                    )

                    overall = next(
                        row
                        for row in result.summary_rows
                        if row["group_scope"] == "overall" and row["group_name"] == "all"
                    )
                    record = {
                        "type": "blimp_likelihood",
                        "benchmark": benchmark_name,
                        **metadata,
                        "score_reduction": score_reduction,
                        "score": overall["score"],
                        "correct": overall["correct"],
                        "items": overall["items"],
                        "ties": overall["ties"],
                        "mean_margin": overall["mean_margin"],
                    }
                    history.append(record)
                    print(
                        f"[step {step:06d}] "
                        f"{benchmark_name}_score={overall['score']:.4f} "
                        f"reduction={score_reduction} "
                        f"correct={overall['correct']}/{overall['items']} "
                        f"ties={overall['ties']}",
                        flush=True,
                    )

            blimp_eval_count += 1
            if (
                blimp_likelihood_plot
                and blimp_eval_count % max(1, blimp_likelihood_plot_every_evals) == 0
            ):
                score_reduction = reduction_filter(blimp_likelihood_score_reductions)
                for benchmark_name, summary_path in summary_csvs.items():
                    benchmark_dir = benchmark_output_dir / benchmark_name
                    plot_blimp_likelihood_summary(
                        summary_csv=summary_path,
                        output_dir=benchmark_dir,
                        benchmark_label=blimp_benchmark_label(benchmark_name),
                        run_name=blimp_likelihood_run_name,
                        group_scopes=blimp_benchmark_group_scopes(benchmark_name),
                        score_reduction=score_reduction,
                        x_axis=benchmark_plot_x_axis,
                    )
                plot_overall_accuracy(
                    summary_csvs={
                        blimp_benchmark_label(name): path
                        for name, path in summary_csvs.items()
                    },
                    output_path=blimp_combined_plot_path,
                    run_name=blimp_likelihood_run_name,
                    score_reduction=score_reduction,
                    x_axis=benchmark_plot_x_axis,
                )
            save_json_atomic(history, metrics_path)
        accelerator.wait_for_everyone()

    def run_entity_tracking_eval(*, epoch: int, step: int, tokens_seen: float) -> None:
        nonlocal entity_tracking_eval_count

        if entity_tracking_items is None:
            return
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            metadata = progress_metadata(
                epoch=epoch,
                step=step,
                tokens_seen=tokens_seen,
                words_per_token=words_per_token,
                run_name=entity_tracking_run_name,
            )
            for score_reduction in entity_tracking_score_reductions:
                result = evaluate_entity_tracking_likelihood(
                    model=unwrapped_model,
                    tokenizer=tokenizer,
                    items=entity_tracking_items,
                    batch_size=entity_tracking_batch_size,
                    score_reduction=score_reduction,
                    device=accelerator.device,
                )
                append_entity_tracking_item_records_jsonl(
                    result.item_records,
                    entity_tracking_items_path,
                    metadata=metadata,
                )
                append_entity_tracking_summary_csv(
                    result.summary_rows,
                    entity_tracking_summary_path,
                    metadata=metadata,
                )
                overall = next(
                    row
                    for row in result.summary_rows
                    if row["group_scope"] == "overall" and row["group_name"] == "all"
                )
                record = {
                    "type": "entity_tracking_likelihood",
                    **metadata,
                    "score_reduction": score_reduction,
                    "score": overall["score"],
                    "correct": overall["correct"],
                    "items": overall["items"],
                    "ties": overall["ties"],
                    "mean_margin": overall["mean_margin"],
                }
                history.append(record)
                print(
                    f"[step {step:06d}] "
                    f"entity_tracking_score={overall['score']:.4f} "
                    f"reduction={score_reduction} "
                    f"correct={overall['correct']}/{overall['items']} "
                    f"ties={overall['ties']}",
                    flush=True,
                )
            entity_tracking_eval_count += 1
            if (
                entity_tracking_plot
                and entity_tracking_eval_count
                % max(1, entity_tracking_plot_every_evals)
                == 0
            ):
                plot_entity_tracking_summary(
                    summary_csv=entity_tracking_summary_path,
                    output_dir=entity_tracking_output_dir,
                    run_name=entity_tracking_run_name,
                    score_reduction=reduction_filter(entity_tracking_score_reductions),
                    x_axis=benchmark_plot_x_axis,
                )
            save_json_atomic(history, metrics_path)
        accelerator.wait_for_everyone()

    def run_comps_eval(*, epoch: int, step: int, tokens_seen: float) -> None:
        nonlocal comps_eval_count

        if comps_items is None:
            return
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            metadata = progress_metadata(
                epoch=epoch,
                step=step,
                tokens_seen=tokens_seen,
                words_per_token=words_per_token,
                run_name=comps_run_name,
            )
            for score_reduction in comps_score_reductions:
                result = evaluate_comps_likelihood(
                    model=unwrapped_model,
                    tokenizer=tokenizer,
                    items=comps_items,
                    batch_size=comps_batch_size,
                    score_reduction=score_reduction,
                    device=accelerator.device,
                )
                append_comps_item_records_jsonl(
                    result.item_records,
                    comps_items_path,
                    metadata=metadata,
                )
                append_comps_summary_csv(
                    result.summary_rows,
                    comps_summary_path,
                    metadata=metadata,
                )
                overall = next(
                    row
                    for row in result.summary_rows
                    if row["group_scope"] == "overall" and row["group_name"] == "all"
                )
                record = {
                    "type": "comps_likelihood",
                    **metadata,
                    "score_reduction": score_reduction,
                    "score": overall["score"],
                    "score_kind": overall["score_kind"],
                    "micro_score": overall["micro_score"],
                    "correct": overall["correct"],
                    "items": overall["items"],
                    "ties": overall["ties"],
                    "mean_margin": overall["mean_margin"],
                }
                history.append(record)
                print(
                    f"[step {step:06d}] "
                    f"comps_macro_score={overall['score']:.4f} "
                    f"reduction={score_reduction} "
                    f"micro_score={overall['micro_score']:.4f} "
                    f"correct={overall['correct']}/{overall['items']} "
                    f"ties={overall['ties']}",
                    flush=True,
                )
            comps_eval_count += 1
            if (
                comps_plot
                and comps_eval_count % max(1, comps_plot_every_evals) == 0
            ):
                plot_comps_summary(
                    summary_csv=comps_summary_path,
                    output_dir=comps_output_dir,
                    run_name=comps_run_name,
                    score_reduction=reduction_filter(comps_score_reductions),
                    x_axis=benchmark_plot_x_axis,
                )
            save_json_atomic(history, metrics_path)
        accelerator.wait_for_everyone()

    if benchmark_eval_at_start:
        if spatial_auxiliary_eval_at_start:
            run_spatial_auxiliary_eval(epoch=0, step=0, tokens_seen=0.0)
        for spec in ewok_context_eval_specs:
            run_ewok_context_eval(spec, epoch=0, step=0, tokens_seen=0.0)
        run_blimp_likelihood_eval(epoch=0, step=0, tokens_seen=0.0)
        run_entity_tracking_eval(epoch=0, step=0, tokens_seen=0.0)
        run_comps_eval(epoch=0, step=0, tokens_seen=0.0)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_tokens = 0.0
        window_loss_sum = 0.0
        window_tokens = 0.0
        window_spatial_auxiliary_loss_sum = 0.0
        window_spatial_auxiliary_batches = 0
        window_optimizer_step_seconds = 0.0
        window_optimizer_steps = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for input_ids, labels, loss_mask, attention_mask in progress:
            # accelerator.accumulate handles no_sync during gradient
            # accumulation and flips sync_gradients on optimizer-update steps.
            spatial_auxiliary_loss = None
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss, token_count = compute_causal_lm_loss(
                        model,
                        input_ids=input_ids,
                        labels=labels,
                        loss_mask=loss_mask,
                        attention_mask=attention_mask,
                    )
                accelerator.backward(loss)

                should_apply_spatial_auxiliary = (
                    spatial_auxiliary_sampler is not None
                    and spatial_auxiliary_loss_weight > 0.0
                    and (global_step + 1) % spatial_auxiliary_every_steps == 0
                )
                if should_apply_spatial_auxiliary:
                    # Backprop the auxiliary branch after the main NTP backward
                    # so we do not keep the long-sequence activation graph alive
                    # while scoring the short contrastive branches.
                    with accelerator.autocast():
                        spatial_auxiliary_batch = spatial_auxiliary_sampler.next_batch()
                        spatial_auxiliary_loss = compute_spatial_auxiliary_loss(
                            model=model,
                            tokenizer=tokenizer,
                            examples=spatial_auxiliary_batch,
                            temperature=spatial_auxiliary_temperature,
                            score_reduction=spatial_auxiliary_score_reduction,
                            device=accelerator.device,
                        )
                        weighted_spatial_auxiliary_loss = (
                            spatial_auxiliary_loss_weight * spatial_auxiliary_loss
                        )
                    accelerator.backward(weighted_spatial_auxiliary_loss)

                if gradient_clip_norm and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), gradient_clip_norm)

                optimizer_step_start = None
                if profile_optimizer_steps and accelerator.sync_gradients:
                    if accelerator.device.type == "cuda":
                        torch.cuda.synchronize(accelerator.device)
                    optimizer_step_start = time.perf_counter()
                optimizer.step()
                if optimizer_step_start is not None:
                    if accelerator.device.type == "cuda":
                        torch.cuda.synchronize(accelerator.device)
                    window_optimizer_step_seconds += time.perf_counter() - optimizer_step_start
                    window_optimizer_steps += 1
                if scheduler is not None and accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_sum = (loss.detach() * token_count).float()
            token_count = token_count.float()
            if accelerator.num_processes > 1:
                loss_sum = accelerator.reduce(loss_sum, reduction="sum")
                token_count = accelerator.reduce(token_count, reduction="sum")

            # Count only optimizer-update steps, not every microbatch.
            if accelerator.sync_gradients:
                global_step += 1

            loss_sum_value = loss_sum.item()
            token_count_value = token_count.item()
            cumulative_tokens_seen += token_count_value
            current_words_seen = (
                cumulative_tokens_seen * words_per_token
                if words_per_token is not None
                else None
            )
            epoch_loss_sum += loss_sum_value
            epoch_tokens += token_count_value
            window_loss_sum += loss_sum_value
            window_tokens += token_count_value
            if spatial_auxiliary_loss is not None:
                spatial_auxiliary_loss_value = spatial_auxiliary_loss.detach().float()
                if accelerator.num_processes > 1:
                    spatial_auxiliary_loss_value = accelerator.reduce(
                        spatial_auxiliary_loss_value,
                        reduction="mean",
                    )
                window_spatial_auxiliary_loss_sum += spatial_auxiliary_loss_value.item()
                window_spatial_auxiliary_batches += 1

            if accelerator.is_local_main_process:
                progress.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            should_log = (
                log_every_steps > 0
                and accelerator.sync_gradients
                and global_step % log_every_steps == 0
            )
            if should_log and accelerator.is_main_process:
                window_loss = window_loss_sum / max(1.0, window_tokens)
                record = {
                    "type": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "tokens_seen": int(cumulative_tokens_seen),
                    "loss": window_loss,
                }
                if window_spatial_auxiliary_batches:
                    spatial_auxiliary_window_loss = (
                        window_spatial_auxiliary_loss_sum
                        / window_spatial_auxiliary_batches
                    )
                    record["spatial_auxiliary_loss"] = spatial_auxiliary_window_loss
                    record["spatial_auxiliary_weighted_loss"] = (
                        spatial_auxiliary_loss_weight * spatial_auxiliary_window_loss
                    )
                    record["spatial_auxiliary_loss_weight"] = spatial_auxiliary_loss_weight
                if words_per_token is not None:
                    record["words_seen"] = cumulative_tokens_seen * words_per_token
                message = f"[step {global_step:06d}] train_loss={window_loss:.4f}"
                if window_spatial_auxiliary_batches:
                    message += (
                        " spatial_aux_loss="
                        f"{record['spatial_auxiliary_loss']:.4f}"
                    )
                if profile_optimizer_steps and window_optimizer_steps:
                    optimizer_step_ms = (
                        1000.0 * window_optimizer_step_seconds / window_optimizer_steps
                    )
                    record["optimizer_step_ms"] = optimizer_step_ms
                    record["optimizer_steps_timed"] = window_optimizer_steps
                    message += f" optimizer_step_ms={optimizer_step_ms:.2f}"
                print(message, flush=True)
                history.append(record)
                save_json_atomic(history, metrics_path)
                window_loss_sum = 0.0
                window_tokens = 0.0
                window_spatial_auxiliary_loss_sum = 0.0
                window_spatial_auxiliary_batches = 0
                window_optimizer_step_seconds = 0.0
                window_optimizer_steps = 0

            validation_word_due = (
                current_words_seen is not None
                and word_interval_due(
                    words_seen=current_words_seen,
                    next_words=next_validation_words,
                )
            )
            should_eval = (
                val_loader is not None
                and accelerator.sync_gradients
                and (
                    (
                        eval_every_steps > 0
                        and global_step % eval_every_steps == 0
                    )
                    or validation_word_due
                )
            )
            if should_eval:
                # Evaluation is intentionally inside the loop but outside the
                # gradient path; it restores train mode before returning.
                val_loss = evaluate_causal_lm_loss(
                    model,
                    val_loader,
                    accelerator,
                    max_batches=eval_max_batches,
                )
                if accelerator.is_main_process:
                    print(f"[step {global_step:06d}] val_loss={val_loss:.4f}", flush=True)
                    history.append(
                        {
                            "type": "validation",
                            "epoch": epoch,
                            "step": global_step,
                            "tokens_seen": int(cumulative_tokens_seen),
                            "loss": val_loss,
                            "max_batches": eval_max_batches,
                            **(
                                {"words_seen": cumulative_tokens_seen * words_per_token}
                                if words_per_token is not None
                                else {}
                            ),
                        }
                    )
                    save_json_atomic(history, metrics_path)
                if validation_word_due:
                    next_validation_words = advance_word_interval(
                        next_words=next_validation_words,
                        every_words=eval_every_words,
                        words_seen=current_words_seen or 0.0,
                    )

            should_spatial_auxiliary_eval = (
                spatial_auxiliary_eval_items is not None
                and spatial_auxiliary_eval_every_steps > 0
                and accelerator.sync_gradients
                and global_step % spatial_auxiliary_eval_every_steps == 0
            )
            if should_spatial_auxiliary_eval:
                run_spatial_auxiliary_eval(
                    epoch=epoch,
                    step=global_step,
                    tokens_seen=cumulative_tokens_seen,
                )

            for spec in ewok_context_eval_specs:
                ewok_word_due = (
                    current_words_seen is not None
                    and word_interval_due(
                        words_seen=current_words_seen,
                        next_words=next_ewok_context_words.get(spec.name),
                    )
                )
                should_ewok_eval = (
                    accelerator.sync_gradients
                    and (
                        (
                            spec.every_steps > 0
                            and global_step % spec.every_steps == 0
                        )
                        or ewok_word_due
                    )
                )
                if should_ewok_eval:
                    run_ewok_context_eval(
                        spec,
                        epoch=epoch,
                        step=global_step,
                        tokens_seen=cumulative_tokens_seen,
                    )
                    if ewok_word_due:
                        next_ewok_context_words[spec.name] = advance_word_interval(
                            next_words=next_ewok_context_words.get(spec.name),
                            every_words=spec.every_words,
                            words_seen=current_words_seen or 0.0,
                        )

            blimp_word_due = (
                current_words_seen is not None
                and word_interval_due(
                    words_seen=current_words_seen,
                    next_words=next_blimp_likelihood_words,
                )
            )
            should_blimp_eval = (
                blimp_likelihood_items is not None
                and accelerator.sync_gradients
                and (
                    (
                        blimp_likelihood_every_steps > 0
                        and global_step % blimp_likelihood_every_steps == 0
                    )
                    or blimp_word_due
                )
            )
            if should_blimp_eval:
                run_blimp_likelihood_eval(
                    epoch=epoch,
                    step=global_step,
                    tokens_seen=cumulative_tokens_seen,
                )
                if blimp_word_due:
                    next_blimp_likelihood_words = advance_word_interval(
                        next_words=next_blimp_likelihood_words,
                        every_words=blimp_likelihood_every_words,
                        words_seen=current_words_seen or 0.0,
                    )

            entity_tracking_word_due = (
                current_words_seen is not None
                and word_interval_due(
                    words_seen=current_words_seen,
                    next_words=next_entity_tracking_words,
                )
            )
            should_entity_tracking_eval = (
                entity_tracking_items is not None
                and accelerator.sync_gradients
                and (
                    (
                        entity_tracking_every_steps > 0
                        and global_step % entity_tracking_every_steps == 0
                    )
                    or entity_tracking_word_due
                )
            )
            if should_entity_tracking_eval:
                run_entity_tracking_eval(
                    epoch=epoch,
                    step=global_step,
                    tokens_seen=cumulative_tokens_seen,
                )
                if entity_tracking_word_due:
                    next_entity_tracking_words = advance_word_interval(
                        next_words=next_entity_tracking_words,
                        every_words=entity_tracking_every_words,
                        words_seen=current_words_seen or 0.0,
                    )

            comps_word_due = (
                current_words_seen is not None
                and word_interval_due(
                    words_seen=current_words_seen,
                    next_words=next_comps_words,
                )
            )
            should_comps_eval = (
                comps_items is not None
                and accelerator.sync_gradients
                and (
                    (
                        comps_every_steps > 0
                        and global_step % comps_every_steps == 0
                    )
                    or comps_word_due
                )
            )
            if should_comps_eval:
                run_comps_eval(
                    epoch=epoch,
                    step=global_step,
                    tokens_seen=cumulative_tokens_seen,
                )
                if comps_word_due:
                    next_comps_words = advance_word_interval(
                        next_words=next_comps_words,
                        every_words=comps_every_words,
                        words_seen=current_words_seen or 0.0,
                    )

        epoch_loss = epoch_loss_sum / max(1.0, epoch_tokens)
        if accelerator.is_main_process:
            print(f"Epoch {epoch} train_loss={epoch_loss:.4f}", flush=True)
            history.append(
                {
                    "type": "epoch",
                    "epoch": epoch,
                    "step": global_step,
                    "tokens_seen": int(cumulative_tokens_seen),
                    "train_loss": epoch_loss,
                    **(
                        {"words_seen": cumulative_tokens_seen * words_per_token}
                        if words_per_token is not None
                        else {}
                    ),
                }
            )
            save_json_atomic(history, metrics_path)

        if save_every_epochs and epoch % save_every_epochs == 0:
            save_hf_checkpoint(
                model,
                tokenizer,
                accelerator,
                output_dir,
                name=f"checkpoint-epoch-{epoch:04d}",
            )

    if save_final:
        save_hf_checkpoint(model, tokenizer, accelerator, output_dir, name="final")

    accelerator.wait_for_everyone()
    return history
