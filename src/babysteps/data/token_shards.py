"""Memory-mapped token-shard datasets for larger causal-LM runs."""

from __future__ import annotations

import glob
import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def _load_meta(data_dir: str | Path) -> dict[str, Any]:
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _list_shards(data_dir: str | Path, split: str, max_shards: int | None = None) -> list[Path]:
    pattern = str(Path(data_dir) / f"{split}_*.bin")
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No {split!r} token shards found with pattern: {pattern}")
    if max_shards is not None:
        if int(max_shards) <= 0:
            raise ValueError(f"max_shards must be positive when set, got {max_shards}.")
        paths = paths[: int(max_shards)]
    return paths


def _count_uint16_tokens(path: Path) -> int:
    nbytes = path.stat().st_size
    if nbytes % np.dtype(np.uint16).itemsize:
        raise ValueError(f"Shard byte size is not divisible by uint16 item size: {path}")
    return nbytes // np.dtype(np.uint16).itemsize


@dataclass(frozen=True)
class TokenShardDatasetConfig:
    data_dir: str | Path
    split: str
    batch_size: int
    seq_len: int
    shuffle_blocks: bool = True
    seed: int = 42
    max_blocks: int | None = None
    max_shards: int | None = None
    span_shards: bool = False


class TokenShardDataset(IterableDataset):
    """Yield already-batched causal-LM tensors from raw uint16 token shards."""

    def __init__(self, config: TokenShardDatasetConfig) -> None:
        super().__init__()
        if int(config.batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if int(config.seq_len) <= 0:
            raise ValueError("seq_len must be positive.")
        if config.max_blocks is not None and int(config.max_blocks) <= 0:
            raise ValueError("max_blocks must be positive when set.")

        self.config = config
        self.data_dir = Path(config.data_dir)
        self.split = str(config.split)
        self.batch_size = int(config.batch_size)
        self.seq_len = int(config.seq_len)
        self.shuffle_blocks = bool(config.shuffle_blocks)
        self.seed = int(config.seed)
        self.max_blocks = None if config.max_blocks is None else int(config.max_blocks)
        self.span_shards = bool(config.span_shards)
        self.meta = _load_meta(self.data_dir)

        self.shards = _list_shards(self.data_dir, self.split, max_shards=config.max_shards)
        self.shard_sizes = [_count_uint16_tokens(path) for path in self.shards]
        self.shard_offsets = [0]
        for size in self.shard_sizes:
            self.shard_offsets.append(self.shard_offsets[-1] + size)
        self.stride_tokens = self.batch_size * self.seq_len
        self.block_tokens = self.stride_tokens + 1
        if self.span_shards:
            total_tokens = self.shard_offsets[-1]
            self.blocks_per_shard = []
            self.total_blocks = max(
                0,
                1 + (total_tokens - self.block_tokens) // self.stride_tokens,
            )
        else:
            self.blocks_per_shard = [
                max(0, 1 + (size - self.block_tokens) // self.stride_tokens)
                for size in self.shard_sizes
            ]
            self.total_blocks = sum(self.blocks_per_shard)
        if self.total_blocks == 0:
            raise ValueError(
                f"No valid {self.split!r} blocks under {self.data_dir}; need at least "
                f"{self.block_tokens} tokens per shard."
            )
        self._iteration = 0

    def __len__(self) -> int:
        if self.max_blocks is None:
            return self.total_blocks
        return min(self.total_blocks, self.max_blocks)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)

        iteration = self._iteration
        self._iteration += 1
        rng = np.random.default_rng(self.seed + iteration)

        if self.span_shards:
            schedule: list[int | tuple[int, int]] = list(range(self.total_blocks))
        else:
            schedule = []
            for shard_idx, n_blocks in enumerate(self.blocks_per_shard):
                schedule.extend((shard_idx, block_idx) for block_idx in range(n_blocks))
        if self.shuffle_blocks:
            rng.shuffle(schedule)
        if self.max_blocks is not None:
            schedule = schedule[: self.max_blocks]
        schedule = schedule[worker_id::num_workers]

        memmaps: dict[int, np.memmap] = {}
        for scheduled_block in schedule:
            if self.span_shards:
                block_idx = int(scheduled_block)
                start = block_idx * self.stride_tokens
                end = start + self.block_tokens
                parts: list[np.ndarray] = []
                position = start
                while position < end:
                    shard_idx = bisect_right(self.shard_offsets, position) - 1
                    shard_start = self.shard_offsets[shard_idx]
                    local_start = position - shard_start
                    local_end = min(self.shard_sizes[shard_idx], local_start + end - position)
                    if shard_idx not in memmaps:
                        memmaps[shard_idx] = np.memmap(
                            self.shards[shard_idx],
                            dtype=np.uint16,
                            mode="r",
                        )
                    part = np.asarray(
                        memmaps[shard_idx][local_start:local_end],
                        dtype=np.int64,
                    )
                    if part.size == 0:
                        break
                    parts.append(part)
                    position += int(part.size)
                chunk = parts[0] if len(parts) == 1 else np.concatenate(parts)
            else:
                shard_idx, block_idx = scheduled_block
                if shard_idx not in memmaps:
                    memmaps[shard_idx] = np.memmap(
                        self.shards[shard_idx],
                        dtype=np.uint16,
                        mode="r",
                    )
                start = block_idx * self.stride_tokens
                end = start + self.block_tokens
                chunk = np.asarray(memmaps[shard_idx][start:end], dtype=np.int64)
            if chunk.size != self.block_tokens:
                continue

            tokens = torch.from_numpy(chunk)
            input_ids = tokens[:-1].reshape(self.batch_size, self.seq_len).contiguous()
            labels = tokens[1:].reshape(self.batch_size, self.seq_len).contiguous()
            loss_mask = torch.ones_like(labels, dtype=torch.bool)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            yield input_ids, labels, loss_mask, attention_mask


def build_token_shard_loader(
    data_dir: str | Path,
    *,
    split: str,
    batch_size: int,
    seq_len: int,
    shuffle: bool,
    max_blocks: int | None = None,
    max_shards: int | None = None,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
    span_shards: bool = False,
) -> tuple[DataLoader, TokenShardDataset]:
    """Build a DataLoader over fixed-size token-shard blocks."""

    dataset = TokenShardDataset(
        TokenShardDatasetConfig(
            data_dir=data_dir,
            split=split,
            batch_size=batch_size,
            seq_len=seq_len,
            shuffle_blocks=shuffle,
            seed=seed,
            max_blocks=max_blocks,
            max_shards=max_shards,
            span_shards=span_shards,
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return loader, dataset


__all__ = [
    "TokenShardDataset",
    "TokenShardDatasetConfig",
    "build_token_shard_loader",
]
