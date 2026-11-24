"""A small reusable causal language-model training loop."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from itertools import chain, islice
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
from babysteps.training.moe import compute_moe_router_losses, moe_settings_from_model
from babysteps.training.nextlat import (
    LatentDynamicsMLP,
    NextLatConfig,
    NextLatLossResult,
    compute_nextlat_loss,
    compute_nextlat_rollout_losses,
    nextlat_kl_weight,
)
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
    return compute_causal_lm_loss_from_logits(
        logits=outputs.logits,
        labels=labels,
        loss_mask=loss_mask,
    )


def compute_causal_lm_loss_from_logits(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute next-token cross-entropy from precomputed logits."""

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
BENCHMARK_EVAL_MODES = {"scheduled", "final_only"}


def normalize_benchmark_eval_mode(value: Any) -> str:
    """Normalize the zero-shot benchmark schedule mode."""

    mode = str(value or "scheduled").strip().lower()
    if mode not in BENCHMARK_EVAL_MODES:
        raise ValueError(
            "logging.benchmark_eval_mode must be 'scheduled' or 'final_only', "
            f"got {value!r}."
        )
    return mode


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
    nextlat_dynamics: LatentDynamicsMLP | None = None,
) -> None:
    """Save a Hugging Face model/tokenizer and optional NextLat dynamics."""

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
    if nextlat_dynamics is not None:
        unwrapped_dynamics = accelerator.unwrap_model(nextlat_dynamics)
        accelerator.save(
            unwrapped_dynamics.state_dict(),
            checkpoint_dir / "nextlat_dynamics.pt",
        )


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
    checkpoint_word_milestones: tuple[tuple[float, str], ...] = (),
    save_final: bool = True,
    profile_optimizer_steps: bool = False,
    nextlat_dynamics: LatentDynamicsMLP | None = None,
    nextlat_config: NextLatConfig | None = None,
    nextlat_bos_token_id: int | None = None,
    nextlat_eos_token_id: int | None = None,
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
    benchmark_eval_mode: str = "scheduled",
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

    benchmark_eval_mode = normalize_benchmark_eval_mode(benchmark_eval_mode)
    scheduled_benchmark_evals = benchmark_eval_mode == "scheduled"
    if benchmark_plot_x_axis not in BENCHMARK_X_AXES:
        raise ValueError(
            "benchmark_plot_x_axis must be one of "
            f"{sorted(BENCHMARK_X_AXES)}, got {benchmark_plot_x_axis!r}."
        )
    if benchmark_plot_x_axis == "words_seen" and words_per_token is None:
        raise ValueError("benchmark_plot_x_axis='words_seen' requires words_per_token.")
    normalized_checkpoint_word_milestones: list[tuple[float, str]] = []
    previous_checkpoint_words: float | None = None
    for raw_words, raw_name in checkpoint_word_milestones:
        milestone_words = float(raw_words)
        checkpoint_name = str(raw_name).strip()
        if milestone_words <= 0:
            raise ValueError("Checkpoint word milestones must be positive.")
        if (
            previous_checkpoint_words is not None
            and milestone_words <= previous_checkpoint_words
        ):
            raise ValueError("Checkpoint word milestones must be strictly increasing.")
        if (
            not checkpoint_name
            or checkpoint_name in {".", ".."}
            or Path(checkpoint_name).name != checkpoint_name
        ):
            raise ValueError("Checkpoint milestone names must be simple directory names.")
        normalized_checkpoint_word_milestones.append(
            (milestone_words, checkpoint_name)
        )
        previous_checkpoint_words = milestone_words
    if normalized_checkpoint_word_milestones and words_per_token is None:
        raise ValueError("Checkpoint word milestones require words_per_token.")
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
    checkpoint_milestone_index = 0
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
    # Sparse MoE models need their router losses added to the LM loss here,
    # because this loop computes cross-entropy itself from the logits.
    moe_settings = moe_settings_from_model(accelerator.unwrap_model(model))
    nextlat_enabled = (
        nextlat_dynamics is not None
        and nextlat_config is not None
        and nextlat_config.enabled
        and (nextlat_config.lambda_next_h > 0.0 or nextlat_config.lambda_kl > 0.0)
    )
    nextlat_input_embeddings = None
    nextlat_output_embeddings = None
    nextlat_rollout_eval_index = 0
    if nextlat_enabled:
        if nextlat_config is None or nextlat_dynamics is None:
            raise ValueError("NextLat requires both nextlat_config and nextlat_dynamics.")
        if (
            nextlat_config.lambda_kl > 0.0
            and nextlat_config.kl_warmup_start_words is not None
            and words_per_token is None
        ):
            raise ValueError("NextLat KL word warmup requires words_per_token.")
        unwrapped_model = accelerator.unwrap_model(model)
        if not hasattr(unwrapped_model, "get_input_embeddings"):
            raise ValueError("NextLat requires a model with get_input_embeddings().")
        nextlat_input_embeddings = unwrapped_model.get_input_embeddings()
        if nextlat_config.lambda_kl > 0.0:
            if not hasattr(unwrapped_model, "get_output_embeddings"):
                raise ValueError("NextLat KL requires a model with get_output_embeddings().")
            nextlat_output_embeddings = unwrapped_model.get_output_embeddings()
            if nextlat_output_embeddings is None:
                raise ValueError("NextLat KL requires output embeddings.")
        if nextlat_config.rollout_eval_words:
            if val_loader is None:
                raise ValueError("NextLat rollout evaluation requires a val_loader.")
            if words_per_token is None:
                raise ValueError(
                    "NextLat rollout word checkpoints require words_per_token."
                )

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    def run_nextlat_rollout_eval(
        *,
        epoch: int,
        step: int,
        tokens_seen: float,
        requested_words: float,
    ) -> None:
        if (
            nextlat_dynamics is None
            or nextlat_config is None
            or nextlat_input_embeddings is None
            or val_loader is None
        ):
            return

        model.eval()
        nextlat_dynamics.eval()
        horizons = nextlat_config.rollout_eval_horizons
        loss_sums = {
            horizon: torch.zeros((), device=accelerator.device, dtype=torch.float32)
            for horizon in horizons
        }
        transition_counts = {
            horizon: torch.zeros((), device=accelerator.device, dtype=torch.float32)
            for horizon in horizons
        }
        persistence_sums = {
            horizon: torch.zeros((), device=accelerator.device, dtype=torch.float32)
            for horizon in horizons
        }
        persistence_counts = {
            horizon: torch.zeros((), device=accelerator.device, dtype=torch.float32)
            for horizon in horizons
        }

        batches = islice(val_loader, nextlat_config.rollout_eval_max_batches)
        with torch.no_grad():
            for input_ids, _labels, loss_mask, attention_mask in batches:
                max_seq_len = nextlat_config.rollout_eval_max_seq_len
                if max_seq_len is not None:
                    input_ids = input_ids[:, :max_seq_len]
                    loss_mask = loss_mask[:, :max_seq_len]
                    attention_mask = attention_mask[:, :max_seq_len]
                with accelerator.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    if outputs.hidden_states is None:
                        raise RuntimeError(
                            "NextLat rollout evaluation expected hidden states."
                        )
                    rollout = compute_nextlat_rollout_losses(
                        dynamics=nextlat_dynamics,
                        input_embeddings=nextlat_input_embeddings,
                        hidden_states=outputs.hidden_states[-1],
                        input_ids=input_ids,
                        loss_mask=loss_mask,
                        attention_mask=attention_mask,
                        horizons=horizons,
                        bos_token_id=nextlat_bos_token_id,
                        eos_token_id=nextlat_eos_token_id,
                        mask_eos_bos_boundaries=(
                            nextlat_config.mask_eos_bos_boundaries
                        ),
                    )
                for horizon in horizons:
                    if horizon not in rollout.loss_by_horizon:
                        continue
                    count = rollout.transition_count_by_horizon[horizon].float()
                    persistence_count = (
                        rollout.persistence_transition_count_by_horizon[horizon].float()
                    )
                    loss_sums[horizon] += (
                        rollout.loss_by_horizon[horizon].detach().float() * count
                    )
                    transition_counts[horizon] += count
                    persistence_sums[horizon] += (
                        rollout.persistence_loss_by_horizon[horizon].detach().float()
                        * persistence_count
                    )
                    persistence_counts[horizon] += persistence_count

        if accelerator.num_processes > 1:
            for horizon in horizons:
                loss_sums[horizon] = accelerator.reduce(
                    loss_sums[horizon], reduction="sum"
                )
                transition_counts[horizon] = accelerator.reduce(
                    transition_counts[horizon], reduction="sum"
                )
                persistence_sums[horizon] = accelerator.reduce(
                    persistence_sums[horizon], reduction="sum"
                )
                persistence_counts[horizon] = accelerator.reduce(
                    persistence_counts[horizon], reduction="sum"
                )

        if accelerator.is_main_process:
            for horizon in horizons:
                transition_count = transition_counts[horizon].item()
                persistence_count = persistence_counts[horizon].item()
                if transition_count <= 0.0 or persistence_count <= 0.0:
                    continue
                rollout_loss = (loss_sums[horizon] / transition_counts[horizon]).item()
                persistence_loss = (
                    persistence_sums[horizon] / persistence_counts[horizon]
                ).item()
                record = {
                    "type": "nextlat_rollout",
                    "epoch": epoch,
                    "step": step,
                    "tokens_seen": int(tokens_seen),
                    "words_seen": float(tokens_seen) * float(words_per_token),
                    "requested_words": float(requested_words),
                    "horizon": horizon,
                    "smooth_l1_loss": rollout_loss,
                    "persistence_loss": persistence_loss,
                    "improvement_over_persistence": persistence_loss - rollout_loss,
                    "transition_count": int(transition_count),
                    "max_batches": nextlat_config.rollout_eval_max_batches,
                    "max_seq_len": nextlat_config.rollout_eval_max_seq_len,
                }
                history.append(record)
                print(
                    f"[step {step:06d}] nextlat_rollout_h{horizon}="
                    f"{rollout_loss:.4f} persistence={persistence_loss:.4f} "
                    f"improvement={persistence_loss - rollout_loss:.4f}",
                    flush=True,
                )
            save_json_atomic(history, metrics_path)

        model.train()
        nextlat_dynamics.train()

    def reduction_filter(score_reductions: tuple[ScoreReduction, ...]) -> ScoreReduction | None:
        return score_reductions[0] if len(score_reductions) == 1 else None

    def run_ewok_context_eval(
        spec: EwokContextEvalSpec,
        *,
        epoch: int,
        step: int,
        tokens_seen: float,
        at_end: bool = False,
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
            if at_end:
                metadata["at_end"] = True
                metadata["checkpoint_name"] = "final"
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

    def run_blimp_likelihood_eval(
        *,
        epoch: int,
        step: int,
        tokens_seen: float,
        at_end: bool = False,
    ) -> None:
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
            if at_end:
                metadata["at_end"] = True
                metadata["checkpoint_name"] = "final"

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

    def run_entity_tracking_eval(
        *,
        epoch: int,
        step: int,
        tokens_seen: float,
        at_end: bool = False,
    ) -> None:
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
            if at_end:
                metadata["at_end"] = True
                metadata["checkpoint_name"] = "final"
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

    def run_comps_eval(
        *,
        epoch: int,
        step: int,
        tokens_seen: float,
        at_end: bool = False,
    ) -> None:
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
            if at_end:
                metadata["at_end"] = True
                metadata["checkpoint_name"] = "final"
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

    if scheduled_benchmark_evals and benchmark_eval_at_start:
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
        epoch_total_loss_sum = 0.0
        epoch_tokens = 0.0
        epoch_nextlat_loss_sum = 0.0
        epoch_nextlat_transition_count = 0.0
        epoch_nextlat_persistence_loss_sum = 0.0
        epoch_nextlat_persistence_transition_count = 0.0
        epoch_nextlat_kl_loss_sum = 0.0
        epoch_nextlat_kl_transition_count = 0.0
        epoch_nextlat_weighted_kl_loss_sum = 0.0
        epoch_nextlat_kl_weight_sum = 0.0
        epoch_nextlat_kl_weight_count = 0.0
        window_loss_sum = 0.0
        window_total_loss_sum = 0.0
        window_tokens = 0.0
        window_nextlat_loss_sum = 0.0
        window_nextlat_transition_count = 0.0
        window_nextlat_persistence_loss_sum = 0.0
        window_nextlat_persistence_transition_count = 0.0
        window_nextlat_kl_loss_sum = 0.0
        window_nextlat_kl_transition_count = 0.0
        window_nextlat_weighted_kl_loss_sum = 0.0
        window_nextlat_kl_weight_sum = 0.0
        window_nextlat_kl_weight_count = 0.0
        window_spatial_auxiliary_loss_sum = 0.0
        window_spatial_auxiliary_batches = 0
        window_optimizer_step_seconds = 0.0
        window_optimizer_steps = 0
        window_moe_aux_loss_sum = 0.0
        window_moe_z_loss_sum = 0.0
        window_moe_max_load_ratio = 0.0
        window_moe_batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for input_ids, labels, loss_mask, attention_mask in progress:
            # accelerator.accumulate handles no_sync during gradient
            # accumulation and flips sync_gradients on optimizer-update steps.
            spatial_auxiliary_loss = None
            nextlat_result: NextLatLossResult | None = None
            batch_words_seen = (
                cumulative_tokens_seen * words_per_token
                if words_per_token is not None
                else None
            )
            batch_nextlat_kl_weight = (
                nextlat_kl_weight(nextlat_config, words_seen=batch_words_seen)
                if nextlat_config is not None
                else 0.0
            )
            accumulate_models = (
                (model, nextlat_dynamics)
                if nextlat_dynamics is not None
                else (model,)
            )
            moe_result = None
            with accelerator.accumulate(*accumulate_models):
                with accelerator.autocast():
                    if nextlat_enabled:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                        )
                        loss, token_count = compute_causal_lm_loss_from_logits(
                            logits=outputs.logits,
                            labels=labels,
                            loss_mask=loss_mask,
                        )
                        if outputs.hidden_states is None:
                            raise RuntimeError(
                                "NextLat expected hidden states from the model forward."
                            )
                        if (
                            nextlat_input_embeddings is None
                            or nextlat_config is None
                        ):
                            raise RuntimeError("NextLat was enabled without setup.")
                        nextlat_result = compute_nextlat_loss(
                            dynamics=nextlat_dynamics,
                            input_embeddings=nextlat_input_embeddings,
                            output_embeddings=nextlat_output_embeddings,
                            hidden_states=outputs.hidden_states[-1],
                            input_ids=input_ids,
                            loss_mask=loss_mask,
                            attention_mask=attention_mask,
                            config=nextlat_config,
                            bos_token_id=nextlat_bos_token_id,
                            eos_token_id=nextlat_eos_token_id,
                            kl_weight=batch_nextlat_kl_weight,
                        )
                        weighted_nextlat_loss = (
                            nextlat_config.lambda_next_h * nextlat_result.next_h_loss
                        )
                        weighted_nextlat_kl_loss = (
                            batch_nextlat_kl_weight * nextlat_result.kl_loss
                        )
                        total_loss = loss + weighted_nextlat_loss + weighted_nextlat_kl_loss
                        if moe_settings is not None:
                            moe_result = compute_moe_router_losses(
                                outputs,
                                moe_settings,
                                attention_mask=attention_mask,
                            )
                            total_loss = total_loss + moe_result.weighted_loss
                    elif moe_settings is not None:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss, token_count = compute_causal_lm_loss_from_logits(
                            logits=outputs.logits,
                            labels=labels,
                            loss_mask=loss_mask,
                        )
                        moe_result = compute_moe_router_losses(
                            outputs,
                            moe_settings,
                            attention_mask=attention_mask,
                        )
                        total_loss = loss + moe_result.weighted_loss
                    else:
                        loss, token_count = compute_causal_lm_loss(
                            model,
                            input_ids=input_ids,
                            labels=labels,
                            loss_mask=loss_mask,
                            attention_mask=attention_mask,
                        )
                        total_loss = loss
                accelerator.backward(total_loss)

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
                    clip_parameters = model.parameters()
                    if nextlat_dynamics is not None:
                        clip_parameters = chain(
                            clip_parameters,
                            nextlat_dynamics.parameters(),
                        )
                    accelerator.clip_grad_norm_(clip_parameters, gradient_clip_norm)

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
            total_loss_sum = (total_loss.detach() * token_count).float()
            token_count = token_count.float()
            nextlat_loss_sum = None
            nextlat_transition_count = None
            nextlat_persistence_loss_sum = None
            nextlat_persistence_transition_count = None
            nextlat_kl_loss_sum = None
            nextlat_kl_transition_count = None
            nextlat_weighted_kl_loss_sum = None
            nextlat_kl_weight_sum = None
            nextlat_kl_weight_count = None
            if nextlat_result is not None:
                nextlat_transition_count = nextlat_result.transition_count.float()
                nextlat_loss_sum = (
                    nextlat_result.next_h_loss.detach().float()
                    * nextlat_transition_count
                )
                nextlat_persistence_transition_count = (
                    nextlat_result.persistence_transition_count.float()
                )
                nextlat_persistence_loss_sum = (
                    nextlat_result.persistence_loss.detach().float()
                    * nextlat_persistence_transition_count
                )
                if nextlat_config is not None and nextlat_config.lambda_kl > 0.0:
                    nextlat_kl_transition_count = (
                        nextlat_result.kl_transition_count.float()
                    )
                    nextlat_kl_loss_sum = (
                        nextlat_result.kl_loss.detach().float()
                        * nextlat_kl_transition_count
                    )
                    nextlat_weighted_kl_loss_sum = (
                        nextlat_result.kl_weight.float()
                        * nextlat_result.kl_loss.detach().float()
                        * nextlat_kl_transition_count
                    )
                    nextlat_kl_weight_count = nextlat_transition_count
                    nextlat_kl_weight_sum = (
                        nextlat_result.kl_weight.float()
                        * nextlat_kl_weight_count
                    )
            if accelerator.num_processes > 1:
                loss_sum = accelerator.reduce(loss_sum, reduction="sum")
                total_loss_sum = accelerator.reduce(total_loss_sum, reduction="sum")
                token_count = accelerator.reduce(token_count, reduction="sum")
                if nextlat_loss_sum is not None and nextlat_transition_count is not None:
                    nextlat_loss_sum = accelerator.reduce(
                        nextlat_loss_sum,
                        reduction="sum",
                    )
                    nextlat_transition_count = accelerator.reduce(
                        nextlat_transition_count,
                        reduction="sum",
                    )
                if (
                    nextlat_persistence_loss_sum is not None
                    and nextlat_persistence_transition_count is not None
                ):
                    nextlat_persistence_loss_sum = accelerator.reduce(
                        nextlat_persistence_loss_sum,
                        reduction="sum",
                    )
                    nextlat_persistence_transition_count = accelerator.reduce(
                        nextlat_persistence_transition_count,
                        reduction="sum",
                    )
                if nextlat_kl_loss_sum is not None and nextlat_kl_transition_count is not None:
                    nextlat_kl_loss_sum = accelerator.reduce(
                        nextlat_kl_loss_sum,
                        reduction="sum",
                    )
                    nextlat_kl_transition_count = accelerator.reduce(
                        nextlat_kl_transition_count,
                        reduction="sum",
                    )
                if (
                    nextlat_weighted_kl_loss_sum is not None
                    and nextlat_kl_weight_sum is not None
                    and nextlat_kl_weight_count is not None
                ):
                    nextlat_weighted_kl_loss_sum = accelerator.reduce(
                        nextlat_weighted_kl_loss_sum,
                        reduction="sum",
                    )
                    nextlat_kl_weight_sum = accelerator.reduce(
                        nextlat_kl_weight_sum,
                        reduction="sum",
                    )
                    nextlat_kl_weight_count = accelerator.reduce(
                        nextlat_kl_weight_count,
                        reduction="sum",
                    )

            # Count only optimizer-update steps, not every microbatch.
            if accelerator.sync_gradients:
                global_step += 1

            loss_sum_value = loss_sum.item()
            total_loss_sum_value = total_loss_sum.item()
            token_count_value = token_count.item()
            cumulative_tokens_seen += token_count_value
            current_words_seen = (
                cumulative_tokens_seen * words_per_token
                if words_per_token is not None
                else None
            )
            if accelerator.sync_gradients and current_words_seen is not None:
                while checkpoint_milestone_index < len(
                    normalized_checkpoint_word_milestones
                ):
                    milestone_words, checkpoint_name = (
                        normalized_checkpoint_word_milestones[
                            checkpoint_milestone_index
                        ]
                    )
                    if current_words_seen < milestone_words:
                        break
                    save_hf_checkpoint(
                        model,
                        tokenizer,
                        accelerator,
                        output_dir,
                        name=checkpoint_name,
                        nextlat_dynamics=nextlat_dynamics,
                    )
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        checkpoint_record = {
                            "type": "checkpoint",
                            "epoch": epoch,
                            "step": global_step,
                            "tokens_seen": int(cumulative_tokens_seen),
                            "words_seen": current_words_seen,
                            "milestone_words": milestone_words,
                            "checkpoint_name": checkpoint_name,
                        }
                        history.append(checkpoint_record)
                        print(
                            f"[step {global_step:06d}] saved {checkpoint_name} "
                            f"at {current_words_seen:,.0f} words",
                            flush=True,
                        )
                        save_json_atomic(history, metrics_path)
                    checkpoint_milestone_index += 1
            epoch_loss_sum += loss_sum_value
            epoch_total_loss_sum += total_loss_sum_value
            epoch_tokens += token_count_value
            window_loss_sum += loss_sum_value
            window_total_loss_sum += total_loss_sum_value
            window_tokens += token_count_value
            if moe_result is not None:
                # Local (per-process) router statistics; BabySteps runs single-GPU.
                window_moe_aux_loss_sum += float(moe_result.aux_loss.detach().item())
                window_moe_z_loss_sum += float(moe_result.z_loss.detach().item())
                window_moe_max_load_ratio = max(
                    window_moe_max_load_ratio,
                    float(moe_result.max_load_ratio.item()),
                )
                window_moe_batches += 1
            if nextlat_loss_sum is not None and nextlat_transition_count is not None:
                nextlat_loss_sum_value = nextlat_loss_sum.item()
                nextlat_transition_count_value = nextlat_transition_count.item()
                epoch_nextlat_loss_sum += nextlat_loss_sum_value
                epoch_nextlat_transition_count += nextlat_transition_count_value
                window_nextlat_loss_sum += nextlat_loss_sum_value
                window_nextlat_transition_count += nextlat_transition_count_value
            if (
                nextlat_persistence_loss_sum is not None
                and nextlat_persistence_transition_count is not None
            ):
                nextlat_persistence_loss_sum_value = (
                    nextlat_persistence_loss_sum.item()
                )
                nextlat_persistence_transition_count_value = (
                    nextlat_persistence_transition_count.item()
                )
                epoch_nextlat_persistence_loss_sum += (
                    nextlat_persistence_loss_sum_value
                )
                epoch_nextlat_persistence_transition_count += (
                    nextlat_persistence_transition_count_value
                )
                window_nextlat_persistence_loss_sum += (
                    nextlat_persistence_loss_sum_value
                )
                window_nextlat_persistence_transition_count += (
                    nextlat_persistence_transition_count_value
                )
            if (
                nextlat_kl_loss_sum is not None
                and nextlat_kl_transition_count is not None
            ):
                nextlat_kl_loss_sum_value = nextlat_kl_loss_sum.item()
                nextlat_kl_transition_count_value = nextlat_kl_transition_count.item()
                epoch_nextlat_kl_loss_sum += nextlat_kl_loss_sum_value
                epoch_nextlat_kl_transition_count += nextlat_kl_transition_count_value
                window_nextlat_kl_loss_sum += nextlat_kl_loss_sum_value
                window_nextlat_kl_transition_count += nextlat_kl_transition_count_value
            if (
                nextlat_weighted_kl_loss_sum is not None
                and nextlat_kl_weight_sum is not None
                and nextlat_kl_weight_count is not None
            ):
                nextlat_weighted_kl_loss_sum_value = (
                    nextlat_weighted_kl_loss_sum.item()
                )
                nextlat_kl_weight_sum_value = nextlat_kl_weight_sum.item()
                nextlat_kl_weight_count_value = nextlat_kl_weight_count.item()
                epoch_nextlat_weighted_kl_loss_sum += (
                    nextlat_weighted_kl_loss_sum_value
                )
                window_nextlat_weighted_kl_loss_sum += (
                    nextlat_weighted_kl_loss_sum_value
                )
                epoch_nextlat_kl_weight_sum += nextlat_kl_weight_sum_value
                window_nextlat_kl_weight_sum += nextlat_kl_weight_sum_value
                epoch_nextlat_kl_weight_count += nextlat_kl_weight_count_value
                window_nextlat_kl_weight_count += nextlat_kl_weight_count_value
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
                postfix = {"loss": f"{loss.item():.4f}", "step": global_step}
                if nextlat_result is not None:
                    postfix["nextlat"] = f"{nextlat_result.next_h_loss.item():.4f}"
                progress.set_postfix(**postfix)

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
                if nextlat_enabled:
                    record["total_loss"] = (
                        window_total_loss_sum / max(1.0, window_tokens)
                    )
                if window_nextlat_transition_count:
                    nextlat_window_loss = (
                        window_nextlat_loss_sum / window_nextlat_transition_count
                    )
                    nextlat_weight = (
                        nextlat_config.lambda_next_h
                        if nextlat_config is not None
                        else 0.0
                    )
                    record["nextlat_next_h_loss"] = nextlat_window_loss
                    record["nextlat_weighted_next_h_loss"] = (
                        nextlat_weight * nextlat_window_loss
                    )
                    record["nextlat_transition_count"] = int(
                        window_nextlat_transition_count
                    )
                if window_nextlat_persistence_transition_count:
                    nextlat_persistence_window_loss = (
                        window_nextlat_persistence_loss_sum
                        / window_nextlat_persistence_transition_count
                    )
                    record["nextlat_persistence_loss"] = (
                        nextlat_persistence_window_loss
                    )
                    if "nextlat_next_h_loss" in record:
                        record["nextlat_improvement_over_persistence"] = (
                            nextlat_persistence_window_loss
                            - record["nextlat_next_h_loss"]
                        )
                if window_nextlat_kl_weight_count:
                    record["nextlat_kl_weight"] = (
                        window_nextlat_kl_weight_sum
                        / window_nextlat_kl_weight_count
                    )
                if window_nextlat_kl_transition_count:
                    nextlat_window_kl_loss = (
                        window_nextlat_kl_loss_sum
                        / window_nextlat_kl_transition_count
                    )
                    record["nextlat_kl_loss"] = nextlat_window_kl_loss
                    record["nextlat_weighted_kl_loss"] = (
                        window_nextlat_weighted_kl_loss_sum
                        / window_nextlat_kl_transition_count
                    )
                    record["nextlat_kl_transition_count"] = int(
                        window_nextlat_kl_transition_count
                    )
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
                if window_nextlat_transition_count:
                    message += (
                        " total_loss="
                        f"{record['total_loss']:.4f} "
                        "nextlat_next_h_loss="
                        f"{record['nextlat_next_h_loss']:.4f}"
                    )
                    if "nextlat_persistence_loss" in record:
                        message += (
                            " nextlat_persistence_loss="
                            f"{record['nextlat_persistence_loss']:.4f}"
                        )
                    if "nextlat_improvement_over_persistence" in record:
                        message += (
                            " nextlat_improvement="
                            f"{record['nextlat_improvement_over_persistence']:.4f}"
                        )
                    if "nextlat_kl_weight" in record:
                        message += (
                            " nextlat_kl_weight="
                            f"{record['nextlat_kl_weight']:.4f}"
                        )
                    if "nextlat_kl_loss" in record:
                        message += (
                            " nextlat_kl_loss="
                            f"{record['nextlat_kl_loss']:.4f}"
                        )
                if window_spatial_auxiliary_batches:
                    message += (
                        " spatial_aux_loss="
                        f"{record['spatial_auxiliary_loss']:.4f}"
                    )
                if window_moe_batches:
                    record["moe_aux_loss"] = window_moe_aux_loss_sum / window_moe_batches
                    record["moe_z_loss"] = window_moe_z_loss_sum / window_moe_batches
                    record["moe_max_load_ratio"] = window_moe_max_load_ratio
                    message += (
                        f" moe_aux_loss={record['moe_aux_loss']:.4f}"
                        f" moe_z_loss={record['moe_z_loss']:.4f}"
                        f" moe_max_load_ratio={window_moe_max_load_ratio:.2f}"
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
                window_total_loss_sum = 0.0
                window_tokens = 0.0
                window_nextlat_loss_sum = 0.0
                window_nextlat_transition_count = 0.0
                window_nextlat_persistence_loss_sum = 0.0
                window_nextlat_persistence_transition_count = 0.0
                window_nextlat_kl_loss_sum = 0.0
                window_nextlat_kl_transition_count = 0.0
                window_nextlat_weighted_kl_loss_sum = 0.0
                window_nextlat_kl_weight_sum = 0.0
                window_nextlat_kl_weight_count = 0.0
                window_spatial_auxiliary_loss_sum = 0.0
                window_spatial_auxiliary_batches = 0
                window_optimizer_step_seconds = 0.0
                window_optimizer_steps = 0
                window_moe_aux_loss_sum = 0.0
                window_moe_z_loss_sum = 0.0
                window_moe_max_load_ratio = 0.0
                window_moe_batches = 0

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

            while (
                nextlat_enabled
                and nextlat_config is not None
                and accelerator.sync_gradients
                and current_words_seen is not None
                and nextlat_rollout_eval_index
                < len(nextlat_config.rollout_eval_words)
                and current_words_seen
                >= nextlat_config.rollout_eval_words[nextlat_rollout_eval_index]
            ):
                requested_words = nextlat_config.rollout_eval_words[
                    nextlat_rollout_eval_index
                ]
                run_nextlat_rollout_eval(
                    epoch=epoch,
                    step=global_step,
                    tokens_seen=cumulative_tokens_seen,
                    requested_words=requested_words,
                )
                nextlat_rollout_eval_index += 1

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
                    scheduled_benchmark_evals
                    and accelerator.sync_gradients
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
                scheduled_benchmark_evals
                and blimp_likelihood_items is not None
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
                scheduled_benchmark_evals
                and entity_tracking_items is not None
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
                scheduled_benchmark_evals
                and comps_items is not None
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
            epoch_record = {
                "type": "epoch",
                "epoch": epoch,
                "step": global_step,
                "tokens_seen": int(cumulative_tokens_seen),
                "train_loss": epoch_loss,
            }
            epoch_message = f"Epoch {epoch} train_loss={epoch_loss:.4f}"
            if nextlat_enabled:
                epoch_record["train_total_loss"] = (
                    epoch_total_loss_sum / max(1.0, epoch_tokens)
                )
                epoch_message += (
                    f" train_total_loss={epoch_record['train_total_loss']:.4f}"
                )
            if epoch_nextlat_transition_count:
                epoch_nextlat_loss = (
                    epoch_nextlat_loss_sum / epoch_nextlat_transition_count
                )
                nextlat_weight = (
                    nextlat_config.lambda_next_h
                    if nextlat_config is not None
                    else 0.0
                )
                epoch_record["nextlat_next_h_loss"] = epoch_nextlat_loss
                epoch_record["nextlat_weighted_next_h_loss"] = (
                    nextlat_weight * epoch_nextlat_loss
                )
                epoch_record["nextlat_transition_count"] = int(
                    epoch_nextlat_transition_count
                )
                epoch_message += f" nextlat_next_h_loss={epoch_nextlat_loss:.4f}"
            if epoch_nextlat_persistence_transition_count:
                epoch_nextlat_persistence_loss = (
                    epoch_nextlat_persistence_loss_sum
                    / epoch_nextlat_persistence_transition_count
                )
                epoch_record["nextlat_persistence_loss"] = (
                    epoch_nextlat_persistence_loss
                )
                if "nextlat_next_h_loss" in epoch_record:
                    epoch_record["nextlat_improvement_over_persistence"] = (
                        epoch_nextlat_persistence_loss
                        - epoch_record["nextlat_next_h_loss"]
                    )
                epoch_message += (
                    " nextlat_persistence_loss="
                    f"{epoch_nextlat_persistence_loss:.4f}"
                )
                if "nextlat_improvement_over_persistence" in epoch_record:
                    epoch_message += (
                        " nextlat_improvement="
                        f"{epoch_record['nextlat_improvement_over_persistence']:.4f}"
                    )
            if epoch_nextlat_kl_weight_count:
                epoch_record["nextlat_kl_weight"] = (
                    epoch_nextlat_kl_weight_sum / epoch_nextlat_kl_weight_count
                )
                epoch_message += (
                    " nextlat_kl_weight="
                    f"{epoch_record['nextlat_kl_weight']:.4f}"
                )
            if epoch_nextlat_kl_transition_count:
                epoch_nextlat_kl_loss = (
                    epoch_nextlat_kl_loss_sum / epoch_nextlat_kl_transition_count
                )
                epoch_record["nextlat_kl_loss"] = epoch_nextlat_kl_loss
                epoch_record["nextlat_weighted_kl_loss"] = (
                    epoch_nextlat_weighted_kl_loss_sum
                    / epoch_nextlat_kl_transition_count
                )
                epoch_record["nextlat_kl_transition_count"] = int(
                    epoch_nextlat_kl_transition_count
                )
                epoch_message += f" nextlat_kl_loss={epoch_nextlat_kl_loss:.4f}"
            if words_per_token is not None:
                epoch_record["words_seen"] = cumulative_tokens_seen * words_per_token
            print(epoch_message, flush=True)
            history.append(epoch_record)
            save_json_atomic(history, metrics_path)

        if save_every_epochs and epoch % save_every_epochs == 0:
            save_hf_checkpoint(
                model,
                tokenizer,
                accelerator,
                output_dir,
                name=f"checkpoint-epoch-{epoch:04d}",
                nextlat_dynamics=nextlat_dynamics,
            )

    if save_final:
        save_hf_checkpoint(
            model,
            tokenizer,
            accelerator,
            output_dir,
            name="final",
            nextlat_dynamics=nextlat_dynamics,
        )

    if benchmark_eval_mode == "final_only":
        for spec in ewok_context_eval_specs:
            run_ewok_context_eval(
                spec,
                epoch=epochs,
                step=global_step,
                tokens_seen=cumulative_tokens_seen,
                at_end=True,
            )
        run_blimp_likelihood_eval(
            epoch=epochs,
            step=global_step,
            tokens_seen=cumulative_tokens_seen,
            at_end=True,
        )
        run_entity_tracking_eval(
            epoch=epochs,
            step=global_step,
            tokens_seen=cumulative_tokens_seen,
            at_end=True,
        )
        run_comps_eval(
            epoch=epochs,
            step=global_step,
            tokens_seen=cumulative_tokens_seen,
            at_end=True,
        )

    accelerator.wait_for_everyone()
    return history
