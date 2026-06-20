"""Parity smoke tests against the original BabyLM 10M GPT-2 training script.

These tests mirror the data collation, manual loss, and one-step optimizer
update from the older ``train_gpt.py`` without importing it. Importing the old
script would trigger Hugging Face login side effects, so the reference behavior
is kept here as small local helper functions.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import get_cosine_schedule_with_warmup

from babysteps.data.token_chunks import collate_token_chunks, load_token_chunks
from babysteps.training.causal_lm_loop import compute_causal_lm_loss, set_all_seeds


DEFAULT_CACHE_DIR = Path(
    os.environ.get("BABYSTEPS_CACHE_DIR", "~/babylm-baby-steps-cache")
).expanduser()
TRAIN_CHUNKS = Path(
    os.environ.get(
        "BABYSTEPS_PARITY_TRAIN_CHUNKS",
        str(
            DEFAULT_CACHE_DIR
            / "token_chunks"
            / "babylm10m_gpt2_pretrained_block512_eos"
            / "train.pkl"
        ),
    )
).expanduser()
PAD_ID = 50256


def _require_train_chunks() -> None:
    if not TRAIN_CHUNKS.exists():
        pytest.skip(f"BabyLM 10M token chunks not found: {TRAIN_CHUNKS}")


def _reference_collate(batch: list[torch.Tensor], pad_id: int):
    """Original train_gpt.py collation logic."""

    lengths = torch.tensor([x.numel() for x in batch], dtype=torch.long)
    tokens = pad_sequence(batch, batch_first=True, padding_value=pad_id)
    _, time = tokens.shape

    positions = torch.arange(time).unsqueeze(0)
    attn_mask = (positions < lengths.unsqueeze(1)).to(tokens.device)

    input_ids = tokens[:, :-1]
    labels = tokens[:, 1:].clone()
    loss_mask = attn_mask[:, 1:]

    return input_ids, labels, loss_mask, attn_mask[:, :-1]


def _reference_manual_loss(model: torch.nn.Module, batch) -> torch.Tensor:
    """Original train_gpt.py manual log_softmax + gather loss."""

    input_ids, labels, loss_mask, attention = batch
    logits = model(input_ids=input_ids, attention_mask=attention)["logits"]
    log_probs = torch.log_softmax(logits, dim=-1)

    safe_labels = labels.clamp(min=0)
    token_log_probs = torch.gather(
        log_probs,
        -1,
        safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    tgt_mask = loss_mask.float()
    denom = tgt_mask.sum()
    if denom > 0:
        return -(token_log_probs * tgt_mask).sum() / denom
    return torch.tensor(0.0, device=input_ids.device, requires_grad=True)


def _load_small_batch(batch_size: int = 2):
    _require_train_chunks()
    chunks = load_token_chunks(TRAIN_CHUNKS, max_chunks=batch_size)
    batch = [torch.tensor(chunk, dtype=torch.long) for chunk in chunks]
    return batch


def _move_batch(batch, device: torch.device):
    return tuple(t.to(device) for t in batch)


def _device() -> torch.device:
    if os.environ.get("BABYSTEPS_PARITY_REQUIRE_CUDA") == "1":
        if not torch.cuda.is_available():
            pytest.fail("BABYSTEPS_PARITY_REQUIRE_CUDA=1 but CUDA is not available.")
        return torch.device("cuda:0")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gpt2_smoke_config() -> GPT2Config:
    """Use a tiny GPT-2 by default, or full GPT-2 small when requested."""

    if os.environ.get("BABYSTEPS_PARITY_FULL_GPT2_SMALL") == "1":
        return GPT2Config(
            vocab_size=50257,
            bos_token_id=PAD_ID,
            eos_token_id=PAD_ID,
            unk_token_id=PAD_ID,
            pad_token_id=PAD_ID,
            n_ctx=1024,
            n_positions=1024,
            n_embd=768,
            n_head=12,
            n_layer=12,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )

    return GPT2Config(
        vocab_size=50257,
        bos_token_id=PAD_ID,
        eos_token_id=PAD_ID,
        unk_token_id=PAD_ID,
        pad_token_id=PAD_ID,
        n_ctx=1024,
        n_positions=1024,
        n_embd=64,
        n_head=4,
        n_layer=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )


def test_token_chunk_collate_matches_original_train_gpt() -> None:
    batch = _load_small_batch(batch_size=4)

    reference = _reference_collate(batch, PAD_ID)
    current = collate_token_chunks(batch, pad_id=PAD_ID)

    assert len(reference) == len(current)
    for reference_tensor, current_tensor in zip(reference, current, strict=True):
        assert reference_tensor.dtype == current_tensor.dtype
        assert reference_tensor.shape == current_tensor.shape
        assert torch.equal(reference_tensor, current_tensor)


def test_causal_lm_loss_matches_original_manual_formula() -> None:
    set_all_seeds(42)
    device = _device()
    batch = _move_batch(collate_token_chunks(_load_small_batch(), pad_id=PAD_ID), device)

    model = GPT2LMHeadModel(_gpt2_smoke_config()).to(device)
    model.eval()

    with torch.no_grad():
        reference_loss = _reference_manual_loss(model, batch)
        current_loss, token_count = compute_causal_lm_loss(
            model,
            input_ids=batch[0],
            labels=batch[1],
            loss_mask=batch[2],
            attention_mask=batch[3],
        )

    assert token_count.item() == batch[2].sum().item()
    assert torch.allclose(current_loss, reference_loss, atol=2e-5, rtol=2e-5)


def test_one_optimizer_step_matches_original_manual_formula() -> None:
    set_all_seeds(123)
    device = _device()
    batch = _move_batch(collate_token_chunks(_load_small_batch(), pad_id=PAD_ID), device)

    reference_model = GPT2LMHeadModel(_gpt2_smoke_config()).to(device)
    current_model = copy.deepcopy(reference_model).to(device)
    reference_model.train()
    current_model.train()

    reference_optimizer = AdamW(
        reference_model.parameters(),
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    current_optimizer = AdamW(
        current_model.parameters(),
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    reference_scheduler = get_cosine_schedule_with_warmup(
        reference_optimizer,
        num_warmup_steps=1,
        num_training_steps=10,
    )
    current_scheduler = get_cosine_schedule_with_warmup(
        current_optimizer,
        num_warmup_steps=1,
        num_training_steps=10,
    )

    reference_loss = _reference_manual_loss(reference_model, batch)
    reference_loss.backward()
    reference_optimizer.step()
    reference_scheduler.step()
    reference_optimizer.zero_grad()

    current_loss, _ = compute_causal_lm_loss(
        current_model,
        input_ids=batch[0],
        labels=batch[1],
        loss_mask=batch[2],
        attention_mask=batch[3],
    )
    current_loss.backward()
    current_optimizer.step()
    current_scheduler.step()
    current_optimizer.zero_grad()

    assert torch.allclose(current_loss, reference_loss, atol=2e-5, rtol=2e-5)
    assert current_scheduler.get_last_lr() == pytest.approx(reference_scheduler.get_last_lr())

    max_abs_diff = 0.0
    with torch.no_grad():
        for current_param, reference_param in zip(
            current_model.parameters(),
            reference_model.parameters(),
            strict=True,
        ):
            diff = (current_param - reference_param).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)

    assert max_abs_diff < 5e-5
