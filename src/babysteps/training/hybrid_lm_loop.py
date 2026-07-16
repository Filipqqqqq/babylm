"""Hybrid GPT-BERT causal + masked-next-token training loop."""

from __future__ import annotations

import shutil
import time
from itertools import chain, islice
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from babysteps.analysis.blimp_likelihood_dynamics import (
    plot_blimp_likelihood_summary,
    plot_overall_accuracy as plot_blimp_overall_accuracy,
)
from babysteps.analysis.comps_dynamics import plot_comps_summary
from babysteps.analysis.entity_tracking_dynamics import plot_entity_tracking_summary
from babysteps.analysis.ewok_context_dynamics import (
    plot_ewok_context_summary,
    plot_full_vs_spatial_summary,
)
from babysteps.analysis.loss_dynamics import DEFAULT_LOSS_KEYS
from babysteps.analysis.loss_dynamics import write_training_loss_plots
from babysteps.evaluation import ewok_context_sensitivity
from babysteps.evaluation import mntp_likelihood
from babysteps.models.causal_lm import DEFAULT_GPT_BERT_REPO
from babysteps.training.branch_scoring import BranchScoreReduction
from babysteps.training.causal_lm_loop import save_json_atomic
from babysteps.training.nextlat import (
    LatentDynamicsMLP,
    NextLatConfig,
    NextLatLossResult,
    compute_nextlat_loss,
    nextlat_kl_weight,
)
from babysteps.training.spatial_auxiliary import (
    SpatialAuxiliaryExample,
    SpatialAuxiliarySampler,
    compute_spatial_auxiliary_loss,
    evaluate_spatial_auxiliary,
)
from babysteps.training.tail_averaging import TailAverager


MNTP_LIKELIHOOD_BENCHMARKS = (
    "blimp",
    "blimp_supplement",
    "entity_tracking",
    "comps",
)
BENCHMARK_X_AXES = {"step", "tokens_seen", "words_seen"}


def benchmark_label(benchmark_name: str) -> str:
    """Return the display label used for one BabyLM benchmark."""

    return {
        "blimp": "BLiMP",
        "blimp_supplement": "BLiMP Supplement",
        "entity_tracking": "Entity Tracking",
        "comps": "COMPS",
    }.get(benchmark_name, benchmark_name.replace("_", " ").title())


def benchmark_group_scopes(benchmark_name: str) -> tuple[str, ...]:
    """Return useful plot groupings for one BabyLM likelihood benchmark."""

    if benchmark_name == "blimp":
        return ("field", "linguistics_term")
    if benchmark_name == "blimp_supplement":
        return ("uid",)
    return ("uid", "field", "linguistics_term")


def word_interval_due(*, words_seen: float, next_words: float | None) -> bool:
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
    run_name: str | None,
) -> dict[str, Any]:
    """Build shared metadata for benchmark records."""

    metadata: dict[str, Any] = {
        "epoch": epoch,
        "step": step,
        "tokens_seen": int(tokens_seen),
    }
    if words_per_token is not None:
        metadata["words_seen"] = tokens_seen * words_per_token
    if run_name:
        metadata["run_name"] = run_name
    return metadata


def unwrap_module(model: torch.nn.Module) -> torch.nn.Module:
    """Peel common distributed wrappers without requiring an Accelerator."""

    current = model
    while hasattr(current, "module"):
        current = current.module
    return current


def set_gpt_bert_is_causal(model: torch.nn.Module, is_causal: bool) -> None:
    """Set the GPT-BERT backbone attention mode on wrapped or unwrapped models."""

    base = unwrap_module(model)
    backbone = getattr(base, "model", None)
    if backbone is None:
        raise ValueError("Expected a GPT-BERT model with a .model backbone.")
    backbone.is_causal = bool(is_causal)


def make_causal_batch(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build standard next-token labels for already-packed token sequences."""

    labels = torch.full_like(input_ids, -100)
    labels[:, :-1] = input_ids[:, 1:]
    return input_ids, labels


def make_masked_batch(
    input_ids: torch.Tensor,
    special_mask: torch.Tensor,
    mask_ratio: float,
    vocab_size: int,
    mask_id: int,
    p_random: float,
    p_keep: float,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build MNTP inputs/labels following the GPT-BERT baseline convention."""

    batch_size, seq_len = input_ids.shape
    del batch_size, seq_len
    orig = input_ids
    eligible = ~special_mask
    selected = torch.bernoulli(
        torch.full(orig.shape, float(mask_ratio), device=orig.device),
        generator=gen,
    ).bool()
    selected = selected & eligible

    corrupted = orig.clone()
    rand = torch.rand(orig.shape, device=orig.device, generator=gen)
    do_mask = selected & (rand < (1.0 - p_random - p_keep))
    do_rand = selected & (rand >= (1.0 - p_random - p_keep)) & (rand < (1.0 - p_keep))
    corrupted[do_mask] = int(mask_id)

    n_rand = int(do_rand.sum().item())
    if n_rand > 0:
        rand_tok = torch.randint(0, int(vocab_size), (n_rand,), device=orig.device, generator=gen)
        corrupted[do_rand] = rand_tok

    labels = torch.full_like(orig, -100)
    labels[:, :-1] = torch.where(
        selected[:, 1:],
        orig[:, 1:],
        torch.full_like(orig[:, 1:], -100),
    )
    return corrupted, labels


def objective_schedule_from_config(hybrid_config: dict[str, Any]) -> tuple[int, int]:
    """Read the causal/MNTP microstep schedule from config."""

    if "causal_per_masked" in hybrid_config:
        raise ValueError(
            "hybrid.causal_per_masked was removed; use hybrid.causal_steps "
            "and hybrid.masked_steps instead."
        )
    causal_steps = int(hybrid_config.get("causal_steps", 15))
    masked_steps = int(hybrid_config.get("masked_steps", 1))
    if causal_steps < 0 or masked_steps < 0:
        raise ValueError("hybrid.causal_steps and hybrid.masked_steps must be non-negative.")
    if causal_steps + masked_steps <= 0:
        raise ValueError(
            "At least one of hybrid.causal_steps or hybrid.masked_steps must be positive."
        )
    return causal_steps, masked_steps


def objective_for(global_micro_step: int, causal_steps: int, masked_steps: int) -> str:
    """Return the objective for a microstep within a causal/MNTP cycle."""

    period = int(causal_steps) + int(masked_steps)
    if period <= 0:
        raise ValueError("causal_steps + masked_steps must be positive.")
    phase = int(global_micro_step) % period
    return "causal" if phase < int(causal_steps) else "masked"


def mask_ratio_for_step(global_update_step: int, total_update_steps: int, start: float, end: float) -> float:
    """Linearly decay the MNTP mask ratio over optimizer-update steps."""

    frac = int(global_update_step) / max(1, int(total_update_steps) - 1)
    return float(start) + (float(end) - float(start)) * frac


def compute_gpt_bert_loss(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    z_loss: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute selected-token cross-entropy plus optional z-loss."""

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    return compute_gpt_bert_loss_from_logits(
        logits=outputs.logits,
        labels=labels,
        z_loss=z_loss,
    )


def compute_gpt_bert_loss_from_logits(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    z_loss: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute selected-token cross-entropy from precomputed GPT-BERT logits."""

    vocab_size = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)
    valid = flat_labels != -100
    token_count = valid.sum()
    if token_count == 0:
        zero = logits.sum() * 0.0
        return zero, zero.detach(), token_count.detach()

    selected_logits = flat_logits[valid]
    selected_labels = flat_labels[valid]
    ce_loss = F.cross_entropy(selected_logits, selected_labels)
    z = torch.logsumexp(selected_logits, dim=-1)
    total_loss = ce_loss + float(z_loss) * (z**2).mean()
    return total_loss, ce_loss.detach(), token_count.detach()


def special_token_ids(tokenizer: Any) -> set[int]:
    """Collect token ids that should not be selected as MNTP targets."""

    ids = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) or [])
    for attr in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "cls_token_id",
        "sep_token_id",
        "mask_token_id",
    ):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            ids.add(int(token_id))
    return ids


def build_special_mask(input_ids: torch.Tensor, special_ids: set[int]) -> torch.Tensor:
    """Return true where input ids are special tokens."""

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        mask |= input_ids == int(token_id)
    return mask


def make_torch_generator(device: torch.device, seed: int) -> torch.Generator:
    """Create a torch RNG on the current training device when supported."""

    try:
        return torch.Generator(device=device).manual_seed(int(seed))
    except RuntimeError:
        return torch.Generator().manual_seed(int(seed))


@torch.no_grad()
def evaluate_hybrid_lm_loss(
    model: torch.nn.Module,
    dataloader: Iterable,
    accelerator,
    *,
    tokenizer: Any,
    hybrid_config: dict[str, Any],
    vocab_size: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate causal CE and fixed-ratio MNTP CE separately."""

    model.eval()
    special_ids = special_token_ids(tokenizer)
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Hybrid GPT-BERT validation requires tokenizer.mask_token_id.")

    causal_loss_sum = torch.tensor(0.0, device=accelerator.device)
    causal_tokens = torch.tensor(0.0, device=accelerator.device)
    mntp_loss_sum = torch.tensor(0.0, device=accelerator.device)
    mntp_tokens = torch.tensor(0.0, device=accelerator.device)
    gen = make_torch_generator(accelerator.device, int(hybrid_config.get("validation_seed", 1234)))
    batches = dataloader if max_batches is None else islice(dataloader, int(max_batches))

    for input_ids, _labels, _loss_mask, attention_mask in batches:
        with accelerator.autocast():
            set_gpt_bert_is_causal(model, True)
            causal_input_ids, causal_labels = make_causal_batch(input_ids)
            _loss, ce_loss, token_count = compute_gpt_bert_loss(
                model,
                input_ids=causal_input_ids,
                labels=causal_labels,
                attention_mask=attention_mask,
                z_loss=0.0,
            )
        causal_loss_sum += ce_loss.float() * token_count.float()
        causal_tokens += token_count.float()

        special_mask = build_special_mask(input_ids, special_ids)
        masked_input_ids, masked_labels = make_masked_batch(
            input_ids,
            special_mask,
            float(hybrid_config.get("mask_end", 0.15)),
            vocab_size,
            int(mask_id),
            float(hybrid_config.get("random_ratio", 0.10)),
            float(hybrid_config.get("keep_ratio", 0.10)),
            gen,
        )
        with accelerator.autocast():
            set_gpt_bert_is_causal(model, False)
            _loss, ce_loss, token_count = compute_gpt_bert_loss(
                model,
                input_ids=masked_input_ids,
                labels=masked_labels,
                attention_mask=attention_mask,
                z_loss=0.0,
            )
        mntp_loss_sum += ce_loss.float() * token_count.float()
        mntp_tokens += token_count.float()

    if accelerator.num_processes > 1:
        causal_loss_sum = accelerator.reduce(causal_loss_sum, reduction="sum")
        causal_tokens = accelerator.reduce(causal_tokens, reduction="sum")
        mntp_loss_sum = accelerator.reduce(mntp_loss_sum, reduction="sum")
        mntp_tokens = accelerator.reduce(mntp_tokens, reduction="sum")

    model.train()
    return {
        "val_causal_loss": (causal_loss_sum / causal_tokens.clamp_min(1)).item(),
        "val_mntp_loss": (mntp_loss_sum / mntp_tokens.clamp_min(1)).item(),
    }


def copy_gpt_bert_repo_file(repo: str, filename: str, output_dir: Path) -> None:
    """Copy remote-code files from a local repo path or the Hugging Face cache."""

    repo_path = Path(repo)
    destination = output_dir / filename
    if repo_path.is_dir():
        source = repo_path / filename
        if source.exists():
            shutil.copyfile(source, destination)
        return

    from huggingface_hub import hf_hub_download

    source = hf_hub_download(repo, filename)
    shutil.copyfile(source, destination)


def save_gpt_bert_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    accelerator,
    output_dir: str | Path,
    *,
    name: str,
    repo: str = DEFAULT_GPT_BERT_REPO,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> None:
    """Save GPT-BERT in the trust_remote_code format expected by HF loading."""

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    checkpoint_dir = Path(output_dir) / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    checkpoint_state = unwrapped_model.state_dict() if state_dict is None else state_dict
    torch.save(checkpoint_state, checkpoint_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(checkpoint_dir)

    for filename in ("configuration_gpt_bert.py", "modeling_gpt_bert.py", "config.json"):
        try:
            copy_gpt_bert_repo_file(repo, filename, checkpoint_dir)
        except Exception as exc:
            print(
                f"   (warning) could not refresh {filename} in {checkpoint_dir}: {exc}",
                flush=True,
            )


def evaluate_mntp_likelihood_benchmark(
    *,
    benchmark_name: str,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[Any],
    batch_size: int,
    device: torch.device,
    score_reduction: str,
) -> mntp_likelihood.MntpBenchmarkResult:
    """Evaluate one BabyLM benchmark with the MNTP scoring backend."""

    if benchmark_name in {"blimp", "blimp_supplement"}:
        return mntp_likelihood.evaluate_blimp_mntp_likelihood(
            model=model,
            tokenizer=tokenizer,
            items=items,
            batch_size=batch_size,
            device=device,
            score_reduction=score_reduction,  # type: ignore[arg-type]
        )
    if benchmark_name == "entity_tracking":
        return mntp_likelihood.evaluate_entity_tracking_mntp_likelihood(
            model=model,
            tokenizer=tokenizer,
            items=items,
            batch_size=batch_size,
            device=device,
            score_reduction=score_reduction,  # type: ignore[arg-type]
        )
    if benchmark_name == "comps":
        return mntp_likelihood.evaluate_comps_mntp_likelihood(
            model=model,
            tokenizer=tokenizer,
            items=items,
            batch_size=batch_size,
            device=device,
            score_reduction=score_reduction,  # type: ignore[arg-type]
        )
    raise ValueError(f"Unsupported MNTP benchmark: {benchmark_name!r}.")


def overall_summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the overall/all row emitted by BabyLM benchmark aggregators."""

    return next(
        row
        for row in rows
        if row["group_scope"] == "overall" and row["group_name"] == "all"
    )


def summary_row_for_group(
    rows: list[dict[str, Any]],
    *,
    group_scope: str,
    group_name: str,
) -> dict[str, Any] | None:
    """Return one grouped summary row when present."""

    return next(
        (
            row
            for row in rows
            if row["group_scope"] == group_scope and row["group_name"] == group_name
        ),
        None,
    )


def add_objective_loss_metrics(
    record: dict[str, Any],
    *,
    prefix: str,
    loss_sum: float,
    ce_sum: float,
    tokens: float,
) -> None:
    """Add token-averaged objective loss metrics when the window has targets."""

    record[f"{prefix}_target_tokens"] = tokens
    if tokens > 0:
        record[f"{prefix}_loss"] = loss_sum / tokens
        record[f"{prefix}_ce_loss"] = ce_sum / tokens


def has_training_loss_plot_rows(
    history: list[dict[str, Any]],
    *,
    x_axis: str,
    loss_keys: tuple[str, ...],
) -> bool:
    """Return whether the in-memory metrics can produce a loss plot."""

    for record in history:
        if record.get("type") != "train" or record.get(x_axis) is None:
            continue
        if any(record.get(loss_key) is not None for loss_key in loss_keys):
            return True
    return False


def plot_mntp_likelihood_summaries(
    *,
    summary_paths: dict[str, Path],
    output_dir: Path,
    run_name: str | None,
    score_reduction: str | None,
    x_axis: str,
) -> None:
    """Write MNTP benchmark dynamics plots from per-benchmark summary files."""

    blimp_summary_paths: dict[str, Path] = {}
    for benchmark_name, summary_path in summary_paths.items():
        benchmark_output_dir = output_dir / benchmark_name
        if benchmark_name in {"blimp", "blimp_supplement"}:
            label = benchmark_label(benchmark_name)
            plot_blimp_likelihood_summary(
                summary_csv=summary_path,
                output_dir=benchmark_output_dir,
                benchmark_label=label,
                run_name=run_name,
                score_reduction=score_reduction,
                group_scopes=benchmark_group_scopes(benchmark_name),
                x_axis=x_axis,
            )
            blimp_summary_paths[label] = summary_path
        elif benchmark_name == "entity_tracking":
            plot_entity_tracking_summary(
                summary_csv=summary_path,
                output_dir=benchmark_output_dir,
                run_name=run_name,
                score_reduction=score_reduction,
                x_axis=x_axis,
            )
        elif benchmark_name == "comps":
            plot_comps_summary(
                summary_csv=summary_path,
                output_dir=benchmark_output_dir,
                run_name=run_name,
                score_reduction=score_reduction,
                x_axis=x_axis,
            )

    if blimp_summary_paths:
        plot_blimp_overall_accuracy(
            summary_csvs=blimp_summary_paths,
            output_path=output_dir / "blimp_mntp_likelihood_overall_accuracy.png",
            run_name=run_name,
            score_reduction=score_reduction,
            title="BLiMP MNTP Likelihood Overall Accuracy",
            x_axis=x_axis,
        )


def train_hybrid_lm(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    accelerator,
    output_dir: str | Path,
    epochs: int,
    total_update_steps: int,
    hybrid_config: dict[str, Any],
    model_repo: str = DEFAULT_GPT_BERT_REPO,
    seed: int = 42,
    log_every_steps: int = 50,
    eval_every_steps: int = 500,
    eval_max_batches: int | None = 20,
    gradient_clip_norm: float | None = None,
    save_every_epochs: int | None = 1,
    save_final: bool = True,
    tail_average_fraction: float = 0.0,
    tail_average_every_steps: int = 10,
    tail_average_name: str = "model_tail_avg",
    benchmark_eval_at_start: bool = True,
    nextlat_dynamics: LatentDynamicsMLP | None = None,
    nextlat_config: NextLatConfig | None = None,
    nextlat_bos_token_id: int | None = None,
    nextlat_eos_token_id: int | None = None,
    nextlat_apply_on: str = "causal",
    ewok_context_items: list[ewok_context_sensitivity.EwokItem] | None = None,
    ewok_context_every_steps: int = 0,
    ewok_context_every_words: float | None = None,
    ewok_context_batch_size: int = 64,
    ewok_context_score_reductions: tuple[
        ewok_context_sensitivity.ScoreReduction,
        ...,
    ] = ("sum",),
    ewok_context_run_name: str | None = None,
    ewok_context_plot: bool = False,
    ewok_context_plot_every_evals: int = 1,
    ewok_mntp_items: list[ewok_context_sensitivity.EwokItem] | None = None,
    ewok_mntp_every_steps: int = 0,
    ewok_mntp_every_words: float | None = None,
    ewok_mntp_batch_size: int = 64,
    ewok_mntp_score_reductions: tuple[
        ewok_context_sensitivity.ScoreReduction,
        ...,
    ] = ("sum",),
    ewok_mntp_run_name: str | None = None,
    ewok_mntp_plot: bool = False,
    ewok_mntp_plot_every_evals: int = 1,
    mntp_likelihood_items: dict[str, list[Any]] | None = None,
    mntp_likelihood_every_steps: int = 0,
    mntp_likelihood_every_words: float | None = None,
    mntp_likelihood_batch_size: int = 64,
    mntp_likelihood_score_reductions: tuple[str, ...] = ("sum",),
    mntp_likelihood_run_name: str | None = None,
    mntp_likelihood_plot: bool = False,
    mntp_likelihood_plot_every_evals: int = 1,
    spatial_auxiliary_sampler: SpatialAuxiliarySampler | None = None,
    spatial_auxiliary_loss_weight: float = 0.03,
    spatial_auxiliary_temperature: float = 1.0,
    spatial_auxiliary_score_reduction: BranchScoreReduction = "mean",
    spatial_auxiliary_every_steps: int = 1,
    spatial_auxiliary_every_causal_steps: int | None = None,
    spatial_auxiliary_apply_on: str = "causal",
    spatial_auxiliary_eval_items: list[SpatialAuxiliaryExample] | None = None,
    spatial_auxiliary_eval_every_steps: int = 0,
    spatial_auxiliary_eval_batch_size: int = 64,
    spatial_auxiliary_eval_at_start: bool = True,
    benchmark_plot_x_axis: str = "step",
    loss_dynamics_plot: bool = False,
    loss_dynamics_plot_every_evals: int = 1,
    loss_dynamics_x_axis: str = "words_seen",
    loss_dynamics_loss_keys: tuple[str, ...] = DEFAULT_LOSS_KEYS,
    loss_dynamics_run_name: str | None = None,
    words_per_token: float | None = None,
) -> list[dict[str, Any]]:
    """Train GPT-BERT with a configurable CLM/MNTP hybrid objective."""

    if benchmark_plot_x_axis not in BENCHMARK_X_AXES:
        raise ValueError(
            "benchmark_plot_x_axis must be one of "
            f"{sorted(BENCHMARK_X_AXES)}, got {benchmark_plot_x_axis!r}."
        )
    ewok_context_word_interval = (
        ewok_context_items is not None and ewok_context_every_words is not None
    )
    ewok_mntp_word_interval = (
        ewok_mntp_items is not None and ewok_mntp_every_words is not None
    )
    mntp_likelihood_word_interval = (
        bool(mntp_likelihood_items) and mntp_likelihood_every_words is not None
    )
    active_ewok_context_plot = ewok_context_items is not None and ewok_context_plot
    active_ewok_mntp_plot = ewok_mntp_items is not None and ewok_mntp_plot
    active_mntp_likelihood_plot = bool(mntp_likelihood_items) and mntp_likelihood_plot
    active_loss_dynamics_plot = bool(mntp_likelihood_items) and loss_dynamics_plot

    any_benchmark_plot = (
        active_ewok_context_plot
        or active_ewok_mntp_plot
        or active_mntp_likelihood_plot
    )
    if (
        any_benchmark_plot
        and benchmark_plot_x_axis == "words_seen"
        and words_per_token is None
    ):
        raise ValueError(
            "Benchmark plots with x_axis='words_seen' require words_per_token."
        )
    if active_ewok_context_plot and ewok_context_plot_every_evals <= 0:
        raise ValueError("ewok_context_plot_every_evals must be positive.")
    if active_ewok_mntp_plot and ewok_mntp_plot_every_evals <= 0:
        raise ValueError("ewok_mntp_plot_every_evals must be positive.")
    if active_mntp_likelihood_plot and mntp_likelihood_plot_every_evals <= 0:
        raise ValueError("mntp_likelihood_plot_every_evals must be positive.")
    if loss_dynamics_x_axis not in BENCHMARK_X_AXES:
        raise ValueError(
            "loss_dynamics_x_axis must be one of "
            f"{sorted(BENCHMARK_X_AXES)}, got {loss_dynamics_x_axis!r}."
        )
    if (
        active_loss_dynamics_plot
        and loss_dynamics_x_axis == "words_seen"
        and words_per_token is None
    ):
        raise ValueError(
            "Loss dynamics plots with x_axis='words_seen' require words_per_token."
        )
    if active_loss_dynamics_plot and loss_dynamics_plot_every_evals <= 0:
        raise ValueError("loss_dynamics_plot_every_evals must be positive.")
    if not loss_dynamics_loss_keys:
        raise ValueError("loss_dynamics_loss_keys must not be empty.")
    if ewok_context_word_interval and words_per_token is None:
        raise ValueError("EWoK causal word-count benchmark intervals require words_per_token.")
    if ewok_mntp_word_interval and words_per_token is None:
        raise ValueError("EWoK MNTP word-count benchmark intervals require words_per_token.")
    if mntp_likelihood_word_interval and words_per_token is None:
        raise ValueError("MNTP word-count benchmark intervals require words_per_token.")
    if not ewok_context_score_reductions:
        raise ValueError("ewok_context_score_reductions must not be empty.")
    if not ewok_mntp_score_reductions:
        raise ValueError("ewok_mntp_score_reductions must not be empty.")
    if spatial_auxiliary_loss_weight < 0.0:
        raise ValueError("spatial_auxiliary_loss_weight must be non-negative.")
    if spatial_auxiliary_temperature <= 0.0:
        raise ValueError("spatial_auxiliary_temperature must be positive.")
    if spatial_auxiliary_score_reduction not in {"sum", "mean"}:
        raise ValueError("spatial_auxiliary_score_reduction must be 'sum' or 'mean'.")
    if spatial_auxiliary_sampler is not None and spatial_auxiliary_every_steps <= 0:
        raise ValueError("spatial_auxiliary_every_steps must be positive when enabled.")
    if (
        spatial_auxiliary_sampler is not None
        and spatial_auxiliary_every_causal_steps is not None
        and spatial_auxiliary_every_causal_steps <= 0
    ):
        raise ValueError(
            "spatial_auxiliary_every_causal_steps must be positive when set."
        )
    if spatial_auxiliary_apply_on != "causal":
        raise ValueError("spatial_auxiliary_apply_on currently supports only 'causal'.")
    if nextlat_apply_on != "causal":
        raise ValueError("nextlat_apply_on currently supports only 'causal'.")
    if spatial_auxiliary_eval_every_steps < 0:
        raise ValueError("spatial_auxiliary_eval_every_steps must be non-negative.")
    if spatial_auxiliary_eval_batch_size <= 0:
        raise ValueError("spatial_auxiliary_eval_batch_size must be positive.")
    if not 0.0 <= float(tail_average_fraction) <= 1.0:
        raise ValueError("tail_average_fraction must be in [0, 1].")
    if tail_average_fraction > 0.0 and int(tail_average_every_steps) <= 0:
        raise ValueError("tail_average_every_steps must be positive when enabled.")
    if tail_average_fraction > 0.0 and not str(tail_average_name).strip():
        raise ValueError("tail_average_name must not be empty when enabled.")

    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics_history.json"
    ewok_context_dir = output_dir / "benchmarks" / "ewok_full"
    ewok_context_items_path = ewok_context_dir / "context_items.jsonl"
    ewok_context_summary_path = ewok_context_dir / "context_summary.csv"
    ewok_context_plot_path = ewok_context_dir / "context_domain_dynamics.png"
    ewok_context_full_spatial_plot_path = (
        ewok_context_dir / "full_vs_spatial_over_steps.png"
    )
    ewok_mntp_dir = output_dir / "benchmarks" / "ewok_mntp"
    ewok_mntp_items_path = ewok_mntp_dir / "context_items.jsonl"
    ewok_mntp_summary_path = ewok_mntp_dir / "context_summary.csv"
    ewok_mntp_plot_path = ewok_mntp_dir / "context_domain_dynamics.png"
    benchmark_dir = output_dir / "benchmarks" / "mntp_likelihood"
    loss_dynamics_dir = output_dir / "loss_dynamics"
    mntp_items_path = benchmark_dir / "likelihood_items.jsonl"
    mntp_summary_path = benchmark_dir / "likelihood_summary.csv"
    history: list[dict[str, Any]] = []
    global_micro_step = 0
    global_causal_update_step = 0
    global_update_step = 0
    cumulative_tokens_seen = 0.0
    next_ewok_context_words = ewok_context_every_words
    next_ewok_mntp_words = ewok_mntp_every_words
    next_mntp_likelihood_words = mntp_likelihood_every_words
    ewok_context_eval_count = 0
    ewok_mntp_eval_count = 0
    mntp_likelihood_eval_count = 0
    vocab_size = int(getattr(accelerator.unwrap_model(model).config, "vocab_size", len(tokenizer)))
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Hybrid GPT-BERT training requires tokenizer.mask_token_id.")
    if not mntp_likelihood_score_reductions:
        raise ValueError("mntp_likelihood_score_reductions must not be empty.")
    nextlat_enabled = (
        nextlat_dynamics is not None
        and nextlat_config is not None
        and nextlat_config.enabled
        and (nextlat_config.lambda_next_h > 0.0 or nextlat_config.lambda_kl > 0.0)
    )
    nextlat_input_embeddings = None
    nextlat_output_embeddings = None
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

    special_ids = special_token_ids(tokenizer)
    causal_steps, masked_steps = objective_schedule_from_config(hybrid_config)
    mask_start = float(hybrid_config.get("mask_start", 0.30))
    mask_end = float(hybrid_config.get("mask_end", 0.15))
    random_ratio = float(hybrid_config.get("random_ratio", 0.10))
    keep_ratio = float(hybrid_config.get("keep_ratio", 0.10))
    z_loss = float(hybrid_config.get("z_loss", 1e-4))
    train_gen = make_torch_generator(accelerator.device, int(seed) + 1000 + accelerator.process_index)

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    tail_averager = None
    if tail_average_fraction > 0.0 and accelerator.is_main_process:
        tail_averager = TailAverager(
            accelerator.unwrap_model(model),
            total_steps=total_update_steps,
            fraction=tail_average_fraction,
            every_steps=tail_average_every_steps,
        )
        print(
            "Tail averaging enabled: "
            f"start_step={tail_averager.start_step} "
            f"every_steps={tail_averager.every_steps} "
            f"artifact={tail_average_name}",
            flush=True,
        )

    def reduction_filter(score_reductions: tuple[Any, ...]) -> Any | None:
        return score_reductions[0] if len(score_reductions) == 1 else None

    def append_ewok_context_outputs(
        *,
        result: ewok_context_sensitivity.EwokContextResult,
        items_path: Path,
        summary_path: Path,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        ewok_context_sensitivity.append_item_records_jsonl(
            result.item_records,
            items_path,
            metadata=metadata,
        )
        ewok_context_sensitivity.append_summary_csv(
            result.summary_rows,
            summary_path,
            metadata=metadata,
        )
        return overall_summary_row(result.summary_rows)

    def run_ewok_context_eval(*, epoch: int, step: int, tokens_seen: float) -> None:
        nonlocal ewok_context_eval_count

        if ewok_context_items is None:
            return
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            set_gpt_bert_is_causal(unwrapped_model, True)
            metadata = progress_metadata(
                epoch=epoch,
                step=step,
                tokens_seen=tokens_seen,
                words_per_token=words_per_token,
                run_name=ewok_context_run_name,
            )
            for score_reduction in ewok_context_score_reductions:
                result = ewok_context_sensitivity.evaluate_ewok_context_sensitivity(
                    model=unwrapped_model,
                    tokenizer=tokenizer,
                    items=ewok_context_items,
                    batch_size=ewok_context_batch_size,
                    score_reduction=score_reduction,
                    device=accelerator.device,
                )
                overall = append_ewok_context_outputs(
                    result=result,
                    items_path=ewok_context_items_path,
                    summary_path=ewok_context_summary_path,
                    metadata=metadata,
                )
                spatial_relations = summary_row_for_group(
                    result.summary_rows,
                    group_scope="domain",
                    group_name="spatial-relations",
                )
                record = {
                    "type": "ewok_context_sensitivity_full",
                    "scoring_backend": "causal",
                    **metadata,
                    "score_reduction": score_reduction,
                    "score": overall["score"],
                    "ewok_context_sensitivity_full_score": overall["score"],
                    "mean_k_margin": overall["mean_k_margin"],
                    "items": overall["items"],
                    "target_token_count_mismatches": overall[
                        "target_token_count_mismatches"
                    ],
                    "sum_mean_sign_disagreements": overall[
                        "sum_mean_sign_disagreements"
                    ],
                }
                if spatial_relations is not None:
                    record["ewok_spatial_relations_score"] = spatial_relations["score"]
                    record["ewok_spatial_relations_mean_k_margin"] = spatial_relations[
                        "mean_k_margin"
                    ]
                    record["ewok_spatial_relations_items"] = spatial_relations["items"]
                history.append(record)
                message = (
                    f"[step {step:06d}] "
                    f"ewok_context_sensitivity_full_score={overall['score']:.4f} "
                    f"reduction={score_reduction} "
                    f"mean_k_margin={overall['mean_k_margin']:.4f}"
                )
                if spatial_relations is not None:
                    message += (
                        " ewok_spatial_relations_score="
                        f"{float(spatial_relations['score']):.4f}"
                    )
                print(message, flush=True)
            ewok_context_eval_count += 1
            if (
                ewok_context_plot
                and ewok_context_eval_count % max(1, ewok_context_plot_every_evals) == 0
            ):
                plot_ewok_context_summary(
                    summary_csv=ewok_context_summary_path,
                    output_path=ewok_context_plot_path,
                    run_name=ewok_context_run_name,
                    score_reduction=reduction_filter(ewok_context_score_reductions),
                    x_axis=benchmark_plot_x_axis,
                )
                plot_full_vs_spatial_summary(
                    summary_csv=ewok_context_summary_path,
                    output_path=ewok_context_full_spatial_plot_path,
                    run_name=ewok_context_run_name,
                    score_reduction=reduction_filter(ewok_context_score_reductions),
                    title="EWoK Full vs Spatial Relations",
                    x_axis=benchmark_plot_x_axis,
                )
            save_json_atomic(history, metrics_path)
        accelerator.wait_for_everyone()

    def run_ewok_mntp_eval(*, epoch: int, step: int, tokens_seen: float) -> None:
        nonlocal ewok_mntp_eval_count

        if ewok_mntp_items is None:
            return
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            set_gpt_bert_is_causal(unwrapped_model, False)
            metadata = progress_metadata(
                epoch=epoch,
                step=step,
                tokens_seen=tokens_seen,
                words_per_token=words_per_token,
                run_name=ewok_mntp_run_name,
            )
            for score_reduction in ewok_mntp_score_reductions:
                result = mntp_likelihood.evaluate_ewok_mntp_context_sensitivity(
                    model=unwrapped_model,
                    tokenizer=tokenizer,
                    items=ewok_mntp_items,
                    batch_size=ewok_mntp_batch_size,
                    score_reduction=score_reduction,
                    device=accelerator.device,
                )
                overall = append_ewok_context_outputs(
                    result=result,
                    items_path=ewok_mntp_items_path,
                    summary_path=ewok_mntp_summary_path,
                    metadata=metadata,
                )
                record = {
                    "type": "ewok_mntp_context_sensitivity",
                    "scoring_backend": "mntp",
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
                    f"ewok_mntp_context_score={overall['score']:.4f} "
                    f"reduction={score_reduction} "
                    f"mean_k_margin={overall['mean_k_margin']:.4f}",
                    flush=True,
                )
            ewok_mntp_eval_count += 1
            if (
                ewok_mntp_plot
                and ewok_mntp_eval_count % max(1, ewok_mntp_plot_every_evals) == 0
            ):
                plot_ewok_context_summary(
                    summary_csv=ewok_mntp_summary_path,
                    output_path=ewok_mntp_plot_path,
                    run_name=ewok_mntp_run_name,
                    score_reduction=reduction_filter(ewok_mntp_score_reductions),
                    x_axis=benchmark_plot_x_axis,
                )
            save_json_atomic(history, metrics_path)
        accelerator.wait_for_everyone()

    def run_mntp_likelihood_eval(*, epoch: int, step: int, tokens_seen: float) -> None:
        nonlocal mntp_likelihood_eval_count

        if not mntp_likelihood_items:
            return
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            set_gpt_bert_is_causal(unwrapped_model, False)
            metadata = progress_metadata(
                epoch=epoch,
                step=step,
                tokens_seen=tokens_seen,
                words_per_token=words_per_token,
                run_name=mntp_likelihood_run_name,
            )
            summary_paths: dict[str, Path] = {}
            for benchmark_name, items in mntp_likelihood_items.items():
                benchmark_output_dir = benchmark_dir / benchmark_name
                benchmark_items_path = benchmark_output_dir / "likelihood_items.jsonl"
                benchmark_summary_path = benchmark_output_dir / "likelihood_summary.csv"
                summary_paths[benchmark_name] = benchmark_summary_path
                for score_reduction in mntp_likelihood_score_reductions:
                    result = evaluate_mntp_likelihood_benchmark(
                        benchmark_name=benchmark_name,
                        model=unwrapped_model,
                        tokenizer=tokenizer,
                        items=items,
                        batch_size=mntp_likelihood_batch_size,
                        device=accelerator.device,
                        score_reduction=score_reduction,
                    )
                    mntp_likelihood.append_item_records_jsonl(
                        result.item_records,
                        mntp_items_path,
                        benchmark=benchmark_name,
                        metadata=metadata,
                    )
                    mntp_likelihood.append_item_records_jsonl(
                        result.item_records,
                        benchmark_items_path,
                        benchmark=benchmark_name,
                        metadata=metadata,
                    )
                    mntp_likelihood.append_summary_csv(
                        result.summary_rows,
                        mntp_summary_path,
                        benchmark=benchmark_name,
                        metadata=metadata,
                    )
                    mntp_likelihood.append_summary_csv(
                        result.summary_rows,
                        benchmark_summary_path,
                        benchmark=benchmark_name,
                        metadata=metadata,
                    )
                    overall = overall_summary_row(result.summary_rows)
                    record = {
                        "type": "mntp_likelihood",
                        "benchmark": benchmark_name,
                        **metadata,
                        "score_reduction": score_reduction,
                        "score": overall.get("score"),
                        "correct": overall.get("correct"),
                        "items": overall.get("items"),
                        "ties": overall.get("ties"),
                        "mean_margin": overall.get("mean_margin"),
                        "score_kind": overall.get("score_kind"),
                        "micro_score": overall.get("micro_score"),
                    }
                    history.append(record)
                    print(
                        f"[step {step:06d}] "
                        f"{benchmark_name}_mntp_score={float(overall.get('score', 0.0)):.4f} "
                        f"reduction={score_reduction} "
                        f"items={overall.get('items')}",
                        flush=True,
                    )
            mntp_likelihood_eval_count += 1
            if (
                mntp_likelihood_plot
                and mntp_likelihood_eval_count % max(1, mntp_likelihood_plot_every_evals) == 0
            ):
                plot_mntp_likelihood_summaries(
                    summary_paths=summary_paths,
                    output_dir=benchmark_dir,
                    run_name=mntp_likelihood_run_name,
                    score_reduction=reduction_filter(mntp_likelihood_score_reductions),
                    x_axis=benchmark_plot_x_axis,
                )
            save_json_atomic(history, metrics_path)
            if (
                loss_dynamics_plot
                and mntp_likelihood_eval_count % max(1, loss_dynamics_plot_every_evals) == 0
                and has_training_loss_plot_rows(
                    history,
                    x_axis=loss_dynamics_x_axis,
                    loss_keys=loss_dynamics_loss_keys,
                )
            ):
                write_training_loss_plots(
                    metrics_paths=[metrics_path],
                    output_dir=loss_dynamics_dir,
                    labels=[loss_dynamics_run_name] if loss_dynamics_run_name else None,
                    loss_keys=loss_dynamics_loss_keys,
                    x_axis=loss_dynamics_x_axis,
                )
        accelerator.wait_for_everyone()

    def run_spatial_auxiliary_eval(*, epoch: int, step: int, tokens_seen: float) -> None:
        if spatial_auxiliary_eval_items is None:
            return
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            set_gpt_bert_is_causal(unwrapped_model, True)
            metadata = progress_metadata(
                epoch=epoch,
                step=step,
                tokens_seen=tokens_seen,
                words_per_token=words_per_token,
                run_name=None,
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
                "scoring_backend": "causal",
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

    if benchmark_eval_at_start:
        run_ewok_context_eval(epoch=0, step=0, tokens_seen=0.0)
        run_ewok_mntp_eval(epoch=0, step=0, tokens_seen=0.0)
        run_mntp_likelihood_eval(epoch=0, step=0, tokens_seen=0.0)
    if spatial_auxiliary_eval_at_start:
        run_spatial_auxiliary_eval(epoch=0, step=0, tokens_seen=0.0)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_ce_sum = 0.0
        epoch_tokens = 0.0
        epoch_causal_loss_sum = 0.0
        epoch_causal_ce_sum = 0.0
        epoch_causal_tokens = 0.0
        epoch_mntp_loss_sum = 0.0
        epoch_mntp_ce_sum = 0.0
        epoch_mntp_tokens = 0.0
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
        window_ce_sum = 0.0
        window_tokens = 0.0
        window_causal_loss_sum = 0.0
        window_causal_ce_sum = 0.0
        window_causal_tokens = 0.0
        window_mntp_loss_sum = 0.0
        window_mntp_ce_sum = 0.0
        window_mntp_tokens = 0.0
        window_causal_microsteps = 0
        window_masked_microsteps = 0
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
        started = time.perf_counter()

        progress = tqdm(
            train_loader,
            desc=f"Hybrid epoch {epoch}/{epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for input_ids, _loader_labels, _loss_mask, attention_mask in progress:
            objective = objective_for(global_micro_step, causal_steps, masked_steps)
            mask_ratio = mask_ratio_for_step(
                global_update_step,
                total_update_steps,
                mask_start,
                mask_end,
            )
            global_micro_step += 1

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
            with accelerator.accumulate(*accumulate_models):
                with accelerator.autocast():
                    if objective == "masked":
                        special_mask = build_special_mask(input_ids, special_ids)
                        batch_input_ids, labels = make_masked_batch(
                            input_ids,
                            special_mask,
                            mask_ratio,
                            vocab_size,
                            int(mask_id),
                            random_ratio,
                            keep_ratio,
                            train_gen,
                        )
                        set_gpt_bert_is_causal(model, False)
                    else:
                        batch_input_ids, labels = make_causal_batch(input_ids)
                        set_gpt_bert_is_causal(model, True)

                    should_apply_nextlat = nextlat_enabled and objective == nextlat_apply_on
                    if should_apply_nextlat:
                        outputs = model(
                            input_ids=batch_input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                            output_hidden_states=True,
                        )
                        loss, ce_loss, target_tokens = compute_gpt_bert_loss_from_logits(
                            logits=outputs.logits,
                            labels=labels,
                            z_loss=z_loss,
                        )
                        if outputs.hidden_states is None:
                            raise RuntimeError(
                                "NextLat expected hidden states from the GPT-BERT forward."
                            )
                        if nextlat_input_embeddings is None or nextlat_config is None:
                            raise RuntimeError("NextLat was enabled without setup.")
                        nextlat_result = compute_nextlat_loss(
                            dynamics=nextlat_dynamics,
                            input_embeddings=nextlat_input_embeddings,
                            output_embeddings=nextlat_output_embeddings,
                            hidden_states=outputs.hidden_states[-1],
                            input_ids=batch_input_ids,
                            loss_mask=labels != -100,
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
                        loss = loss + weighted_nextlat_loss + weighted_nextlat_kl_loss
                    else:
                        loss, ce_loss, target_tokens = compute_gpt_bert_loss(
                            model,
                            input_ids=batch_input_ids,
                            labels=labels,
                            attention_mask=attention_mask,
                            z_loss=z_loss,
                        )

                accelerator.backward(loss)
                spatial_schedule_due = (
                    accelerator.sync_gradients
                    and (global_causal_update_step + 1)
                    % spatial_auxiliary_every_causal_steps
                    == 0
                    if spatial_auxiliary_every_causal_steps is not None
                    else (global_update_step + 1) % spatial_auxiliary_every_steps == 0
                )
                should_apply_spatial_auxiliary = (
                    objective == spatial_auxiliary_apply_on
                    and spatial_auxiliary_sampler is not None
                    and spatial_auxiliary_loss_weight > 0.0
                    and spatial_schedule_due
                )
                if should_apply_spatial_auxiliary:
                    with accelerator.autocast():
                        set_gpt_bert_is_causal(model, True)
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
                optimizer.step()
                if scheduler is not None and accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients and tail_averager is not None:
                    tail_averager.update(
                        accelerator.unwrap_model(model),
                        step=global_update_step + 1,
                    )

            target_tokens_float = target_tokens.float()
            loss_sum = (loss.detach() * target_tokens_float).float()
            ce_sum = (ce_loss.detach() * target_tokens_float).float()
            input_tokens = torch.tensor(float(input_ids.numel()), device=accelerator.device)
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
                ce_sum = accelerator.reduce(ce_sum, reduction="sum")
                target_tokens_float = accelerator.reduce(target_tokens_float, reduction="sum")
                input_tokens = accelerator.reduce(input_tokens, reduction="sum")
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

            if accelerator.sync_gradients:
                global_update_step += 1
                if objective == "causal":
                    global_causal_update_step += 1

            loss_sum_value = loss_sum.item()
            ce_sum_value = ce_sum.item()
            target_token_value = target_tokens_float.item()
            cumulative_tokens_seen += input_tokens.item()
            current_words_seen = (
                cumulative_tokens_seen * words_per_token
                if words_per_token is not None
                else None
            )
            epoch_loss_sum += loss_sum_value
            epoch_ce_sum += ce_sum_value
            epoch_tokens += target_token_value
            window_loss_sum += loss_sum_value
            window_ce_sum += ce_sum_value
            window_tokens += target_token_value
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
            if objective == "masked":
                epoch_mntp_loss_sum += loss_sum_value
                epoch_mntp_ce_sum += ce_sum_value
                epoch_mntp_tokens += target_token_value
                window_mntp_loss_sum += loss_sum_value
                window_mntp_ce_sum += ce_sum_value
                window_mntp_tokens += target_token_value
                window_masked_microsteps += 1
            else:
                epoch_causal_loss_sum += loss_sum_value
                epoch_causal_ce_sum += ce_sum_value
                epoch_causal_tokens += target_token_value
                window_causal_loss_sum += loss_sum_value
                window_causal_ce_sum += ce_sum_value
                window_causal_tokens += target_token_value
                window_causal_microsteps += 1

            if accelerator.is_local_main_process:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    obj=objective,
                    step=global_update_step,
                )

            should_log = (
                log_every_steps > 0
                and accelerator.sync_gradients
                and global_update_step % int(log_every_steps) == 0
            )
            if should_log and accelerator.is_main_process:
                record = {
                    "type": "train",
                    "epoch": epoch,
                    "step": global_update_step,
                    "global_micro_step": global_micro_step,
                    "tokens_seen": int(cumulative_tokens_seen),
                    "loss": window_loss_sum / max(1.0, window_tokens),
                    "ce_loss": window_ce_sum / max(1.0, window_tokens),
                    "mask_ratio": mask_ratio,
                    "causal_microsteps": window_causal_microsteps,
                    "masked_microsteps": window_masked_microsteps,
                    "seconds": time.perf_counter() - started,
                }
                add_objective_loss_metrics(
                    record,
                    prefix="train_causal",
                    loss_sum=window_causal_loss_sum,
                    ce_sum=window_causal_ce_sum,
                    tokens=window_causal_tokens,
                )
                add_objective_loss_metrics(
                    record,
                    prefix="train_mntp",
                    loss_sum=window_mntp_loss_sum,
                    ce_sum=window_mntp_ce_sum,
                    tokens=window_mntp_tokens,
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
                    record["spatial_auxiliary_batches"] = window_spatial_auxiliary_batches
                if current_words_seen is not None:
                    record["words_seen"] = current_words_seen
                history.append(record)
                objective_pieces = []
                if "train_causal_loss" in record:
                    objective_pieces.append(f"causal_loss={record['train_causal_loss']:.4f}")
                if "train_mntp_loss" in record:
                    objective_pieces.append(f"mntp_loss={record['train_mntp_loss']:.4f}")
                if window_nextlat_transition_count:
                    objective_pieces.append(
                        "nextlat_next_h_loss="
                        f"{record['nextlat_next_h_loss']:.4f}"
                    )
                    if "nextlat_kl_weight" in record:
                        objective_pieces.append(
                            "nextlat_kl_weight="
                            f"{record['nextlat_kl_weight']:.4f}"
                        )
                    if "nextlat_kl_loss" in record:
                        objective_pieces.append(
                            "nextlat_kl_loss="
                            f"{record['nextlat_kl_loss']:.4f}"
                        )
                if window_spatial_auxiliary_batches:
                    objective_pieces.append(
                        f"spatial_aux_loss={record['spatial_auxiliary_loss']:.4f}"
                    )
                print(
                    f"[step {global_update_step:06d}] "
                    f"train_loss={record['loss']:.4f} "
                    f"ce_loss={record['ce_loss']:.4f} "
                    f"mask={mask_ratio:.3f} "
                    f"micro={global_micro_step}"
                    + (f" {' '.join(objective_pieces)}" if objective_pieces else ""),
                    flush=True,
                )
                save_json_atomic(history, metrics_path)
                window_loss_sum = 0.0
                window_ce_sum = 0.0
                window_tokens = 0.0
                window_causal_loss_sum = 0.0
                window_causal_ce_sum = 0.0
                window_causal_tokens = 0.0
                window_mntp_loss_sum = 0.0
                window_mntp_ce_sum = 0.0
                window_mntp_tokens = 0.0
                window_causal_microsteps = 0
                window_masked_microsteps = 0
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

            should_eval = (
                val_loader is not None
                and eval_every_steps > 0
                and accelerator.sync_gradients
                and global_update_step % int(eval_every_steps) == 0
            )
            if should_eval:
                val_losses = evaluate_hybrid_lm_loss(
                    model,
                    val_loader,
                    accelerator,
                    tokenizer=tokenizer,
                    hybrid_config=hybrid_config,
                    vocab_size=vocab_size,
                    max_batches=eval_max_batches,
                )
                if accelerator.is_main_process:
                    record = {
                        "type": "validation",
                        "epoch": epoch,
                        "step": global_update_step,
                        "global_micro_step": global_micro_step,
                        "tokens_seen": int(cumulative_tokens_seen),
                        "max_batches": eval_max_batches,
                        **val_losses,
                    }
                    if current_words_seen is not None:
                        record["words_seen"] = current_words_seen
                    history.append(record)
                    print(
                        f"[step {global_update_step:06d}] "
                        f"val_causal_loss={val_losses['val_causal_loss']:.4f} "
                        f"val_mntp_loss={val_losses['val_mntp_loss']:.4f}",
                        flush=True,
                    )
                    save_json_atomic(history, metrics_path)

            ewok_context_word_due = (
                current_words_seen is not None
                and word_interval_due(
                    words_seen=current_words_seen,
                    next_words=next_ewok_context_words,
                )
            )
            should_ewok_context_eval = (
                ewok_context_items is not None
                and accelerator.sync_gradients
                and (
                    (
                        ewok_context_every_steps > 0
                        and global_update_step % int(ewok_context_every_steps) == 0
                    )
                    or ewok_context_word_due
                )
            )
            if should_ewok_context_eval:
                run_ewok_context_eval(
                    epoch=epoch,
                    step=global_update_step,
                    tokens_seen=cumulative_tokens_seen,
                )
                if ewok_context_word_due:
                    next_ewok_context_words = advance_word_interval(
                        next_words=next_ewok_context_words,
                        every_words=ewok_context_every_words,
                        words_seen=current_words_seen or 0.0,
                    )

            ewok_mntp_word_due = (
                current_words_seen is not None
                and word_interval_due(
                    words_seen=current_words_seen,
                    next_words=next_ewok_mntp_words,
                )
            )
            should_ewok_mntp_eval = (
                ewok_mntp_items is not None
                and accelerator.sync_gradients
                and (
                    (
                        ewok_mntp_every_steps > 0
                        and global_update_step % int(ewok_mntp_every_steps) == 0
                    )
                    or ewok_mntp_word_due
                )
            )
            if should_ewok_mntp_eval:
                run_ewok_mntp_eval(
                    epoch=epoch,
                    step=global_update_step,
                    tokens_seen=cumulative_tokens_seen,
                )
                if ewok_mntp_word_due:
                    next_ewok_mntp_words = advance_word_interval(
                        next_words=next_ewok_mntp_words,
                        every_words=ewok_mntp_every_words,
                        words_seen=current_words_seen or 0.0,
                    )

            mntp_word_due = (
                current_words_seen is not None
                and word_interval_due(
                    words_seen=current_words_seen,
                    next_words=next_mntp_likelihood_words,
                )
            )
            should_mntp_likelihood_eval = (
                mntp_likelihood_items is not None
                and accelerator.sync_gradients
                and (
                    (
                        mntp_likelihood_every_steps > 0
                        and global_update_step % int(mntp_likelihood_every_steps) == 0
                    )
                    or mntp_word_due
                )
            )
            if should_mntp_likelihood_eval:
                run_mntp_likelihood_eval(
                    epoch=epoch,
                    step=global_update_step,
                    tokens_seen=cumulative_tokens_seen,
                )
                if mntp_word_due:
                    next_mntp_likelihood_words = advance_word_interval(
                        next_words=next_mntp_likelihood_words,
                        every_words=mntp_likelihood_every_words,
                        words_seen=current_words_seen or 0.0,
                    )

            should_spatial_auxiliary_eval = (
                spatial_auxiliary_eval_items is not None
                and spatial_auxiliary_eval_every_steps > 0
                and accelerator.sync_gradients
                and global_update_step % int(spatial_auxiliary_eval_every_steps) == 0
            )
            if should_spatial_auxiliary_eval:
                run_spatial_auxiliary_eval(
                    epoch=epoch,
                    step=global_update_step,
                    tokens_seen=cumulative_tokens_seen,
                )

        epoch_record = {
            "type": "epoch",
            "epoch": epoch,
            "step": global_update_step,
            "global_micro_step": global_micro_step,
            "tokens_seen": int(cumulative_tokens_seen),
            "train_loss": epoch_loss_sum / max(1.0, epoch_tokens),
            "train_ce_loss": epoch_ce_sum / max(1.0, epoch_tokens),
        }
        add_objective_loss_metrics(
            epoch_record,
            prefix="train_causal",
            loss_sum=epoch_causal_loss_sum,
            ce_sum=epoch_causal_ce_sum,
            tokens=epoch_causal_tokens,
        )
        add_objective_loss_metrics(
            epoch_record,
            prefix="train_mntp",
            loss_sum=epoch_mntp_loss_sum,
            ce_sum=epoch_mntp_ce_sum,
            tokens=epoch_mntp_tokens,
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
        if epoch_nextlat_persistence_transition_count:
            epoch_nextlat_persistence_loss = (
                epoch_nextlat_persistence_loss_sum
                / epoch_nextlat_persistence_transition_count
            )
            epoch_record["nextlat_persistence_loss"] = epoch_nextlat_persistence_loss
            if "nextlat_next_h_loss" in epoch_record:
                epoch_record["nextlat_improvement_over_persistence"] = (
                    epoch_nextlat_persistence_loss
                    - epoch_record["nextlat_next_h_loss"]
                )
        if epoch_nextlat_kl_weight_count:
            epoch_record["nextlat_kl_weight"] = (
                epoch_nextlat_kl_weight_sum / epoch_nextlat_kl_weight_count
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
        if words_per_token is not None:
            epoch_record["words_seen"] = cumulative_tokens_seen * words_per_token
        if accelerator.is_main_process:
            history.append(epoch_record)
            epoch_message = (
                f"Hybrid epoch {epoch} "
                f"train_loss={epoch_record['train_loss']:.4f} "
                f"train_ce_loss={epoch_record['train_ce_loss']:.4f}"
            )
            if "nextlat_next_h_loss" in epoch_record:
                epoch_message += (
                    " nextlat_next_h_loss="
                    f"{epoch_record['nextlat_next_h_loss']:.4f}"
                )
            if "nextlat_kl_loss" in epoch_record:
                epoch_message += (
                    f" nextlat_kl_loss={epoch_record['nextlat_kl_loss']:.4f}"
                )
            print(epoch_message, flush=True)
            save_json_atomic(history, metrics_path)

        if save_every_epochs and epoch % int(save_every_epochs) == 0:
            save_gpt_bert_checkpoint(
                model,
                tokenizer,
                accelerator,
                output_dir,
                name=f"checkpoint-epoch-{epoch:04d}",
                repo=model_repo,
            )

    if save_final:
        save_gpt_bert_checkpoint(
            model,
            tokenizer,
            accelerator,
            output_dir,
            name="final",
            repo=model_repo,
        )

    if tail_average_fraction > 0.0:
        averaged_state = None
        if tail_averager is not None:
            averaged_state = tail_averager.averaged_state_dict(
                accelerator.unwrap_model(model)
            )
        save_gpt_bert_checkpoint(
            model,
            tokenizer,
            accelerator,
            output_dir,
            name=str(tail_average_name),
            repo=model_repo,
            state_dict=averaged_state,
        )
        if accelerator.is_main_process:
            if tail_averager is None:
                raise RuntimeError("Main process did not initialize tail averaging.")
            print(
                "Saved tail-averaged checkpoint: "
                f"snapshots={tail_averager.num_snapshots} "
                f"fraction={tail_average_fraction:g} "
                f"name={tail_average_name}",
                flush=True,
            )

    accelerator.wait_for_everyone()
    return history


__all__ = [
    "build_special_mask",
    "add_objective_loss_metrics",
    "benchmark_group_scopes",
    "compute_gpt_bert_loss",
    "compute_gpt_bert_loss_from_logits",
    "evaluate_mntp_likelihood_benchmark",
    "evaluate_hybrid_lm_loss",
    "has_training_loss_plot_rows",
    "make_causal_batch",
    "make_masked_batch",
    "mask_ratio_for_step",
    "objective_schedule_from_config",
    "objective_for",
    "plot_mntp_likelihood_summaries",
    "save_gpt_bert_checkpoint",
    "set_gpt_bert_is_causal",
    "special_token_ids",
    "train_hybrid_lm",
]
