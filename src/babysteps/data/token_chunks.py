"""Utilities for token-chunk datasets.

The current BabyLM 10M training data is stored as pickle files containing
``list[list[int]]`` token chunks. This module keeps that format small and
explicit while giving training scripts a reusable Dataset and collate function.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from babysteps.paths import validate_token_chunk_file


TokenChunk = Sequence[int]


class TokenChunkDataset(Dataset):
    """PyTorch Dataset for pre-tokenized integer chunks."""

    def __init__(self, chunks: Sequence[TokenChunk]) -> None:
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.chunks[idx], dtype=torch.long)


def load_token_chunks(path: str | Path, max_chunks: int | None = None) -> list[TokenChunk]:
    """Load a pickle file containing token chunks."""

    path = validate_token_chunk_file(path)
    with path.open("rb") as f:
        chunks = pickle.load(f)

    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    return chunks


def collate_token_chunks(
    batch: Sequence[torch.Tensor],
    *,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad chunks and produce causal-LM inputs, labels, and masks.

    Returns:
        input_ids: tokens without the final position.
        labels: next-token targets.
        loss_mask: true where labels should contribute to loss.
        attention_mask: true where input tokens are real, not padding.
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    lengths = torch.tensor([x.numel() for x in batch], dtype=torch.long)
    if torch.any(lengths < 2):
        raise ValueError("Every token chunk must have at least two tokens.")

    tokens = pad_sequence(batch, batch_first=True, padding_value=pad_id)
    time = tokens.size(1)
    positions = torch.arange(time).unsqueeze(0)
    full_attention_mask = positions < lengths.unsqueeze(1)

    input_ids = tokens[:, :-1].contiguous()
    labels = tokens[:, 1:].contiguous()
    loss_mask = full_attention_mask[:, 1:].contiguous()
    attention_mask = full_attention_mask[:, :-1].contiguous()

    return input_ids, labels, loss_mask, attention_mask


def build_token_chunk_loader(
    path: str | Path,
    *,
    batch_size: int,
    pad_id: int,
    shuffle: bool,
    max_chunks: int | None = None,
    num_workers: int = 0,
    seed: int | None = None,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> tuple[DataLoader, TokenChunkDataset]:
    """Build a DataLoader for a token-chunk pickle."""

    chunks = load_token_chunks(path, max_chunks=max_chunks)
    dataset = TokenChunkDataset(chunks)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        collate_fn=lambda batch: collate_token_chunks(batch, pad_id=pad_id),
    )
    return loader, dataset
