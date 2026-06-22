"""A small reusable causal language-model training loop."""

from __future__ import annotations

import json
import random
import time
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
    eval_max_batches: int | None = 20,
    gradient_clip_norm: float | None = None,
    save_every_epochs: int | None = 1,
    save_final: bool = True,
    profile_optimizer_steps: bool = False,
    ewok_context_items: list[EwokItem] | None = None,
    ewok_context_every_steps: int = 0,
    ewok_context_batch_size: int = 64,
    ewok_context_score_reduction: ScoreReduction = "sum",
    ewok_context_run_name: str | None = None,
    ewok_context_plot: bool = False,
    ewok_context_plot_every_evals: int = 1,
    blimp_likelihood_items: dict[str, list[BlimpItem]] | None = None,
    blimp_likelihood_every_steps: int = 0,
    blimp_likelihood_batch_size: int = 64,
    blimp_likelihood_run_name: str | None = None,
    blimp_likelihood_plot: bool = False,
    blimp_likelihood_plot_every_evals: int = 1,
    entity_tracking_items: list[EntityTrackingItem] | None = None,
    entity_tracking_every_steps: int = 0,
    entity_tracking_batch_size: int = 64,
    entity_tracking_run_name: str | None = None,
    entity_tracking_plot: bool = False,
    entity_tracking_plot_every_evals: int = 1,
    comps_items: list[CompsItem] | None = None,
    comps_every_steps: int = 0,
    comps_batch_size: int = 64,
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

    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics_history.json"
    ewok_output_dir = output_dir / "benchmarks" / "ewok"
    ewok_items_path = ewok_output_dir / "context_items.jsonl"
    ewok_summary_path = ewok_output_dir / "context_summary.csv"
    ewok_plot_path = ewok_output_dir / "context_domain_dynamics.png"
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
    ewok_eval_count = 0
    blimp_eval_count = 0
    entity_tracking_eval_count = 0
    comps_eval_count = 0

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_tokens = 0.0
        window_loss_sum = 0.0
        window_tokens = 0.0
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
            epoch_loss_sum += loss_sum_value
            epoch_tokens += token_count_value
            window_loss_sum += loss_sum_value
            window_tokens += token_count_value

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
                if words_per_token is not None:
                    record["words_seen"] = cumulative_tokens_seen * words_per_token
                message = f"[step {global_step:06d}] train_loss={window_loss:.4f}"
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
                window_optimizer_step_seconds = 0.0
                window_optimizer_steps = 0

            should_eval = (
                val_loader is not None
                and eval_every_steps > 0
                and accelerator.sync_gradients
                and global_step % eval_every_steps == 0
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

            should_ewok_eval = (
                ewok_context_items is not None
                and ewok_context_every_steps > 0
                and accelerator.sync_gradients
                and global_step % ewok_context_every_steps == 0
            )
            if should_ewok_eval:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    result = evaluate_ewok_context_sensitivity(
                        model=unwrapped_model,
                        tokenizer=tokenizer,
                        items=ewok_context_items,
                        batch_size=ewok_context_batch_size,
                        score_reduction=ewok_context_score_reduction,
                        device=accelerator.device,
                    )
                    metadata = progress_metadata(
                        epoch=epoch,
                        step=global_step,
                        tokens_seen=cumulative_tokens_seen,
                        words_per_token=words_per_token,
                        run_name=ewok_context_run_name,
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
                    ewok_eval_count += 1
                    if (
                        ewok_context_plot
                        and ewok_eval_count % max(1, ewok_context_plot_every_evals) == 0
                    ):
                        plot_ewok_context_summary(
                            summary_csv=ewok_summary_path,
                            output_path=ewok_plot_path,
                            run_name=ewok_context_run_name,
                            score_reduction=ewok_context_score_reduction,
                            x_axis=benchmark_plot_x_axis,
                        )
                    overall = next(
                        row
                        for row in result.summary_rows
                        if row["group_scope"] == "overall" and row["group_name"] == "all"
                    )
                    record = {
                        "type": "ewok_context_sensitivity",
                        **metadata,
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
                        f"[step {global_step:06d}] "
                        f"ewok_context_score={overall['score']:.4f} "
                        f"mean_k_margin={overall['mean_k_margin']:.4f} "
                        "sum_mean_sign_disagreements="
                        f"{overall['sum_mean_sign_disagreements']}",
                        flush=True,
                    )
                    save_json_atomic(history, metrics_path)
                accelerator.wait_for_everyone()

            should_blimp_eval = (
                blimp_likelihood_items is not None
                and blimp_likelihood_every_steps > 0
                and accelerator.sync_gradients
                and global_step % blimp_likelihood_every_steps == 0
            )
            if should_blimp_eval:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    metadata = progress_metadata(
                        epoch=epoch,
                        step=global_step,
                        tokens_seen=cumulative_tokens_seen,
                        words_per_token=words_per_token,
                        run_name=blimp_likelihood_run_name,
                    )

                    summary_csvs: dict[str, Path] = {}
                    for benchmark_name, items in blimp_likelihood_items.items():
                        benchmark_dir = benchmark_output_dir / benchmark_name
                        items_path = benchmark_dir / "likelihood_items.jsonl"
                        summary_path = benchmark_dir / "likelihood_summary.csv"
                        result = evaluate_blimp_likelihood(
                            model=unwrapped_model,
                            tokenizer=tokenizer,
                            items=items,
                            batch_size=blimp_likelihood_batch_size,
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
                        summary_csvs[benchmark_name] = summary_path

                        overall = next(
                            row
                            for row in result.summary_rows
                            if row["group_scope"] == "overall" and row["group_name"] == "all"
                        )
                        record = {
                            "type": "blimp_likelihood",
                            "benchmark": benchmark_name,
                            **metadata,
                            "score": overall["score"],
                            "correct": overall["correct"],
                            "items": overall["items"],
                            "ties": overall["ties"],
                            "mean_margin": overall["mean_margin"],
                        }
                        history.append(record)
                        print(
                            f"[step {global_step:06d}] "
                            f"{benchmark_name}_score={overall['score']:.4f} "
                            f"correct={overall['correct']}/{overall['items']} "
                            f"ties={overall['ties']}",
                            flush=True,
                        )

                    blimp_eval_count += 1
                    if (
                        blimp_likelihood_plot
                        and blimp_eval_count % max(1, blimp_likelihood_plot_every_evals) == 0
                    ):
                        for benchmark_name, summary_path in summary_csvs.items():
                            benchmark_dir = benchmark_output_dir / benchmark_name
                            plot_blimp_likelihood_summary(
                                summary_csv=summary_path,
                                output_dir=benchmark_dir,
                                benchmark_label=blimp_benchmark_label(benchmark_name),
                                run_name=blimp_likelihood_run_name,
                                group_scopes=blimp_benchmark_group_scopes(benchmark_name),
                                x_axis=benchmark_plot_x_axis,
                            )
                        plot_overall_accuracy(
                            summary_csvs={
                                blimp_benchmark_label(name): path
                                for name, path in summary_csvs.items()
                            },
                            output_path=blimp_combined_plot_path,
                            run_name=blimp_likelihood_run_name,
                            x_axis=benchmark_plot_x_axis,
                        )
                    save_json_atomic(history, metrics_path)
                accelerator.wait_for_everyone()

            should_entity_tracking_eval = (
                entity_tracking_items is not None
                and entity_tracking_every_steps > 0
                and accelerator.sync_gradients
                and global_step % entity_tracking_every_steps == 0
            )
            if should_entity_tracking_eval:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    metadata = progress_metadata(
                        epoch=epoch,
                        step=global_step,
                        tokens_seen=cumulative_tokens_seen,
                        words_per_token=words_per_token,
                        run_name=entity_tracking_run_name,
                    )
                    result = evaluate_entity_tracking_likelihood(
                        model=unwrapped_model,
                        tokenizer=tokenizer,
                        items=entity_tracking_items,
                        batch_size=entity_tracking_batch_size,
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
                            x_axis=benchmark_plot_x_axis,
                        )
                    overall = next(
                        row
                        for row in result.summary_rows
                        if row["group_scope"] == "overall" and row["group_name"] == "all"
                    )
                    record = {
                        "type": "entity_tracking_likelihood",
                        **metadata,
                        "score": overall["score"],
                        "correct": overall["correct"],
                        "items": overall["items"],
                        "ties": overall["ties"],
                        "mean_margin": overall["mean_margin"],
                    }
                    history.append(record)
                    print(
                        f"[step {global_step:06d}] "
                        f"entity_tracking_score={overall['score']:.4f} "
                        f"correct={overall['correct']}/{overall['items']} "
                        f"ties={overall['ties']}",
                        flush=True,
                    )
                    save_json_atomic(history, metrics_path)
                accelerator.wait_for_everyone()

            should_comps_eval = (
                comps_items is not None
                and comps_every_steps > 0
                and accelerator.sync_gradients
                and global_step % comps_every_steps == 0
            )
            if should_comps_eval:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    metadata = progress_metadata(
                        epoch=epoch,
                        step=global_step,
                        tokens_seen=cumulative_tokens_seen,
                        words_per_token=words_per_token,
                        run_name=comps_run_name,
                    )
                    result = evaluate_comps_likelihood(
                        model=unwrapped_model,
                        tokenizer=tokenizer,
                        items=comps_items,
                        batch_size=comps_batch_size,
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
                    comps_eval_count += 1
                    if (
                        comps_plot
                        and comps_eval_count % max(1, comps_plot_every_evals) == 0
                    ):
                        plot_comps_summary(
                            summary_csv=comps_summary_path,
                            output_dir=comps_output_dir,
                            run_name=comps_run_name,
                            x_axis=benchmark_plot_x_axis,
                        )
                    overall = next(
                        row
                        for row in result.summary_rows
                        if row["group_scope"] == "overall" and row["group_name"] == "all"
                    )
                    record = {
                        "type": "comps_likelihood",
                        **metadata,
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
                        f"[step {global_step:06d}] "
                        f"comps_macro_score={overall['score']:.4f} "
                        f"micro_score={overall['micro_score']:.4f} "
                        f"correct={overall['correct']}/{overall['items']} "
                        f"ties={overall['ties']}",
                        flush=True,
                    )
                    save_json_atomic(history, metrics_path)
                accelerator.wait_for_everyone()

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
