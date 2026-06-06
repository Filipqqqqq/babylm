"""A small reusable causal language-model training loop."""

from __future__ import annotations

import json
import random
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


DEFAULT_GENERATION_PROMPTS = [
    "Sunlight filtered through the ancient oak's twisting branches.",
    "A curious cat perched on the windowsill.",
    "Alice loves spicy food, while Bob hates it. When the waiter brought the extra-hot curry,",
    "John believes his keys are in his pocket, but they are actually on the table. He reaches",
]


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

    token_losses = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)

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
        total_loss = accelerator.reduce(total_loss, reduction="sum")
        total_tokens = accelerator.reduce(total_tokens, reduction="sum")

    model.train()
    return (total_loss / total_tokens.clamp_min(1)).item()


def generate_sample(
    model: torch.nn.Module,
    tokenizer,
    accelerator,
    *,
    prompts: Sequence[str],
    max_new_tokens: int = 40,
) -> str:
    """Generate one sample from a random prompt on the main process."""

    prompt = random.choice(list(prompts))
    unwrapped_model = accelerator.unwrap_model(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)

    with torch.no_grad():
        generated = unwrapped_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(generated[0], skip_special_tokens=True)


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
    generate_every_steps: int = 500,
    generation_prompts: Sequence[str] | None = DEFAULT_GENERATION_PROMPTS,
    max_new_tokens: int = 40,
    gradient_clip_norm: float | None = None,
    save_every_epochs: int | None = 1,
    save_final: bool = True,
) -> list[dict[str, Any]]:
    """Train a causal LM and return the metric history."""

    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics_history.json"
    history: list[dict[str, Any]] = []
    global_step = 0
    if generation_prompts is None:
        generation_prompts = DEFAULT_GENERATION_PROMPTS

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_tokens = 0.0
        window_loss_sum = 0.0
        window_tokens = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for input_ids, labels, loss_mask, attention_mask in progress:
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

                optimizer.step()
                if scheduler is not None and accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_sum = (loss.detach() * token_count).float()
            token_count = token_count.float()
            if accelerator.num_processes > 1:
                loss_sum = accelerator.reduce(loss_sum, reduction="sum")
                token_count = accelerator.reduce(token_count, reduction="sum")

            if accelerator.sync_gradients:
                global_step += 1

            loss_sum_value = loss_sum.item()
            token_count_value = token_count.item()
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
                print(f"[step {global_step:06d}] train_loss={window_loss:.4f}")
                history.append(
                    {
                        "type": "train",
                        "epoch": epoch,
                        "step": global_step,
                        "loss": window_loss,
                    }
                )
                save_json_atomic(history, metrics_path)
                window_loss_sum = 0.0
                window_tokens = 0.0

            should_eval = (
                val_loader is not None
                and eval_every_steps > 0
                and accelerator.sync_gradients
                and global_step % eval_every_steps == 0
            )
            if should_eval:
                val_loss = evaluate_causal_lm_loss(
                    model,
                    val_loader,
                    accelerator,
                    max_batches=eval_max_batches,
                )
                if accelerator.is_main_process:
                    print(f"[step {global_step:06d}] val_loss={val_loss:.4f}")
                    history.append(
                        {
                            "type": "validation",
                            "epoch": epoch,
                            "step": global_step,
                            "loss": val_loss,
                            "max_batches": eval_max_batches,
                        }
                    )
                    save_json_atomic(history, metrics_path)

            should_generate = (
                generate_every_steps > 0
                and accelerator.sync_gradients
                and global_step % generate_every_steps == 0
                and accelerator.is_main_process
            )
            if should_generate:
                model.eval()
                sample = generate_sample(
                    model,
                    tokenizer,
                    accelerator,
                    prompts=generation_prompts,
                    max_new_tokens=max_new_tokens,
                )
                print(f"[step {global_step:06d}] sample: {sample}")
                history.append(
                    {
                        "type": "sample",
                        "epoch": epoch,
                        "step": global_step,
                        "text": sample,
                    }
                )
                save_json_atomic(history, metrics_path)
                model.train()

        epoch_loss = epoch_loss_sum / max(1.0, epoch_tokens)
        if accelerator.is_main_process:
            print(f"Epoch {epoch} train_loss={epoch_loss:.4f}")
            history.append(
                {
                    "type": "epoch",
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": epoch_loss,
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
