#!/usr/bin/env python3
"""Run a deterministic 10-batch GPT-2 smoke test.

This is a controlled training check, not a real training run. It verifies that:

1. The token-chunk DataLoader can produce a deterministic no-shuffle stream.
2. The new cross-entropy loss matches the old manual log_softmax + gather loss
   on every batch.
3. Two same-seed runs produce the same loss history and final parameters.

By default the script uses a tiny GPT-2-shaped model for speed. Pass
``--full-gpt2-small`` to use GPT-2 small dimensions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import yaml
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import get_cosine_schedule_with_warmup

from babysteps.data.token_chunks import build_token_chunk_loader
from babysteps.training.causal_lm_loop import compute_causal_lm_loss, set_all_seeds


PAD_ID = 50256


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(name)


def build_smoke_gpt2_config(config: dict[str, Any], *, full_gpt2_small: bool) -> GPT2Config:
    model_config = config["model"]
    context_length = int(model_config.get("context_length", 1024))

    if full_gpt2_small:
        n_embd = int(model_config.get("n_embd", 768))
        n_head = int(model_config.get("n_head", 12))
        n_layer = int(model_config.get("n_layer", 12))
    else:
        n_embd = 64
        n_head = 4
        n_layer = 2

    return GPT2Config(
        vocab_size=int(model_config.get("vocab_size", 50257)),
        bos_token_id=PAD_ID,
        eos_token_id=PAD_ID,
        unk_token_id=PAD_ID,
        pad_token_id=PAD_ID,
        n_ctx=context_length,
        n_positions=context_length,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        # Dropout is disabled so the smoke test checks optimizer/data/loss
        # determinism rather than RNG stream alignment.
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )


def manual_train_gpt_loss(model: torch.nn.Module, batch) -> torch.Tensor:
    """Mirror the original train_gpt.py log_softmax + gather loss."""

    input_ids, labels, loss_mask, attention = batch
    logits = model(input_ids=input_ids, attention_mask=attention)["logits"]
    log_probs = torch.log_softmax(logits, dim=-1)

    safe_labels = labels.clamp(min=0)
    token_log_probs = torch.gather(log_probs, -1, safe_labels.unsqueeze(-1)).squeeze(-1)

    tgt_mask = loss_mask.float()
    denom = tgt_mask.sum()
    if denom > 0:
        return -(token_log_probs * tgt_mask).sum() / denom
    return torch.tensor(0.0, device=input_ids.device, requires_grad=True)


def parameter_fingerprint(model: torch.nn.Module) -> dict[str, float]:
    """Return a lightweight deterministic summary of model parameters."""

    total_sum = 0.0
    total_abs_sum = 0.0
    total_sq_sum = 0.0
    with torch.no_grad():
        for param in model.parameters():
            data = param.detach().float().cpu()
            total_sum += data.sum().item()
            total_abs_sum += data.abs().sum().item()
            total_sq_sum += data.square().sum().item()
    return {
        "sum": total_sum,
        "abs_sum": total_abs_sum,
        "sq_sum": total_sq_sum,
    }


def max_parameter_diff(left: torch.nn.Module, right: torch.nn.Module) -> float:
    max_diff = 0.0
    with torch.no_grad():
        for left_param, right_param in zip(left.parameters(), right.parameters(), strict=True):
            diff = (left_param - right_param).abs().max().item()
            max_diff = max(max_diff, diff)
    return max_diff


def run_once(
    *,
    config: dict[str, Any],
    seed: int,
    batches: int,
    batch_size: int,
    device: torch.device,
    full_gpt2_small: bool,
    loss_atol: float,
) -> tuple[list[float], GPT2LMHeadModel, dict[str, float]]:
    set_all_seeds(seed, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    loader, dataset = build_token_chunk_loader(
        config["data"]["train_chunks"],
        batch_size=batch_size,
        pad_id=PAD_ID,
        shuffle=False,
        max_chunks=batches * batch_size,
        num_workers=0,
        seed=seed,
        pin_memory=(device.type == "cuda"),
    )
    if len(loader) != batches:
        raise RuntimeError(
            f"Expected exactly {batches} batches, got {len(loader)} "
            f"from {len(dataset)} chunks."
        )

    model = GPT2LMHeadModel(
        build_smoke_gpt2_config(config, full_gpt2_small=full_gpt2_small)
    ).to(device)
    model.train()

    training_config = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 5e-5)),
        betas=tuple(training_config.get("betas", [0.9, 0.999])),
        eps=float(training_config.get("eps", 1e-8)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    warmup_steps = training_config.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = int(float(training_config.get("warmup_ratio", 0.1)) * batches)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_steps),
        num_training_steps=batches,
    )

    losses: list[float] = []
    for step, batch in enumerate(loader, start=1):
        batch = tuple(t.to(device) for t in batch)

        reference_loss = manual_train_gpt_loss(model, batch)
        current_loss, token_count = compute_causal_lm_loss(
            model,
            input_ids=batch[0],
            labels=batch[1],
            loss_mask=batch[2],
            attention_mask=batch[3],
        )
        if token_count.item() != batch[2].sum().item():
            raise AssertionError("Token count mismatch between loss mask and loss helper.")
        if not torch.allclose(current_loss, reference_loss, atol=loss_atol, rtol=loss_atol):
            raise AssertionError(
                f"Loss formula mismatch at step {step}: "
                f"current={current_loss.item():.8f}, "
                f"reference={reference_loss.item():.8f}"
            )

        current_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(current_loss.item())

    return losses, model, parameter_fingerprint(model)


def assert_same_run(
    *,
    left_losses: list[float],
    right_losses: list[float],
    left_model: torch.nn.Module,
    right_model: torch.nn.Module,
    history_atol: float,
    parameter_atol: float,
) -> float:
    if len(left_losses) != len(right_losses):
        raise AssertionError("Loss histories have different lengths.")

    for step, (left_loss, right_loss) in enumerate(
        zip(left_losses, right_losses, strict=True),
        start=1,
    ):
        if abs(left_loss - right_loss) > history_atol:
            raise AssertionError(
                f"Loss history drift at step {step}: "
                f"{left_loss:.8f} vs {right_loss:.8f}"
            )

    max_diff = max_parameter_diff(left_model, right_model)
    if max_diff > parameter_atol:
        raise AssertionError(
            f"Final parameter drift too large: max_abs_diff={max_diff:.8g}"
        )
    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "train" / "smoke" / "gpt2_small_10m_smoke.yaml"),
        help="Training config to source data/model optimizer settings from.",
    )
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--full-gpt2-small", action="store_true")
    parser.add_argument("--loss-atol", type=float, default=2e-5)
    parser.add_argument("--history-atol", type=float, default=1e-7)
    parser.add_argument("--parameter-atol", type=float, default=1e-7)
    parser.add_argument(
        "--summary-json",
        default="/tmp/babysteps_10_batch_deterministic_summary.json",
        help="Where to write the smoke-test summary JSON.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    device = select_device(args.device)

    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Batches: {args.batches}")
    print(f"Batch size: {args.batch_size}")
    print(f"Full GPT-2 small: {args.full_gpt2_small}")

    left_losses, left_model, left_fingerprint = run_once(
        config=config,
        seed=args.seed,
        batches=args.batches,
        batch_size=args.batch_size,
        device=device,
        full_gpt2_small=args.full_gpt2_small,
        loss_atol=args.loss_atol,
    )
    right_losses, right_model, right_fingerprint = run_once(
        config=config,
        seed=args.seed,
        batches=args.batches,
        batch_size=args.batch_size,
        device=device,
        full_gpt2_small=args.full_gpt2_small,
        loss_atol=args.loss_atol,
    )

    max_diff = assert_same_run(
        left_losses=left_losses,
        right_losses=right_losses,
        left_model=left_model,
        right_model=right_model,
        history_atol=args.history_atol,
        parameter_atol=args.parameter_atol,
    )

    summary = {
        "status": "passed",
        "config": str(args.config),
        "device": str(device),
        "seed": args.seed,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "full_gpt2_small": args.full_gpt2_small,
        "losses": left_losses,
        "left_fingerprint": left_fingerprint,
        "right_fingerprint": right_fingerprint,
        "max_parameter_diff": max_diff,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("10-batch deterministic smoke test passed.")
    print("Loss history:", ", ".join(f"{loss:.4f}" for loss in left_losses))
    print(f"Max parameter diff between repeated runs: {max_diff:.8g}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
