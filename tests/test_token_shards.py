from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from babysteps.data.token_shards import build_token_shard_loader


def _write_shard(path, tokens) -> None:
    np.asarray(tokens, dtype=np.uint16).tofile(path)


def test_token_shard_loader_yields_shifted_fixed_blocks(tmp_path) -> None:
    _write_shard(tmp_path / "train_000000.bin", range(17))
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": 32}), encoding="utf-8")

    loader, dataset = build_token_shard_loader(
        tmp_path,
        split="train",
        batch_size=2,
        seq_len=4,
        shuffle=False,
    )

    assert len(dataset) == 2
    input_ids, labels, loss_mask, attention_mask = next(iter(loader))

    assert input_ids.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert labels.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert loss_mask.dtype is torch.bool
    assert attention_mask.dtype is torch.bool
    assert loss_mask.all()
    assert attention_mask.all()


def test_token_shard_loader_respects_max_blocks(tmp_path) -> None:
    _write_shard(tmp_path / "train_000000.bin", range(33))

    _loader, dataset = build_token_shard_loader(
        tmp_path,
        split="train",
        batch_size=2,
        seq_len=4,
        shuffle=False,
        max_blocks=2,
    )

    assert dataset.total_blocks == 4
    assert len(dataset) == 2


def test_token_shard_loader_requires_matching_split(tmp_path) -> None:
    _write_shard(tmp_path / "train_000000.bin", range(17))

    with pytest.raises(FileNotFoundError, match="val"):
        build_token_shard_loader(
            tmp_path,
            split="val",
            batch_size=2,
            seq_len=4,
            shuffle=False,
        )
