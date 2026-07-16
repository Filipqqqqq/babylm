from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from babysteps.data.paired_token_repair import (
    RepairCondition,
    SyntheticComponent,
    build_paired_token_repair_datasets,
    build_weighted_synthetic_stream,
    discover_train_blocks,
    rank_train_blocks,
    replacement_block_count,
)


def write_tokens(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype=np.uint16).tofile(path)


def component(name: str, weight: float, values: list[list[int]]) -> SyntheticComponent:
    return SyntheticComponent(
        name=name,
        weight=weight,
        documents=tuple(np.asarray(item, dtype=np.uint16) for item in values),
        source_path=f"{name}.txt",
        source_sha256=name * 4,
    )


def test_replacement_block_count_rounds_to_loader_blocks() -> None:
    assert replacement_block_count(1038, 0.0) == 0
    assert replacement_block_count(1038, 0.01) == 10
    assert replacement_block_count(1038, 0.015) == 16


def test_weighted_synthetic_stream_has_exact_length_and_provenance() -> None:
    components = [
        component("major", 3, [[10, 11, 12]]),
        component("minor", 1, [[20, 21]]),
    ]

    tokens, labels, counts = build_weighted_synthetic_stream(components, total_tokens=19)

    assert tokens.dtype == np.uint16
    assert labels.dtype == np.uint16
    assert tokens.size == 19
    assert labels.size == 19
    assert sum(counts.values()) == 19
    assert counts["major"] > counts["minor"]


def test_paired_builder_preserves_validation_and_nests_repair_blocks(tmp_path) -> None:
    base = tmp_path / "base"
    # batch_size=2 and seq_len=2 gives stride=4 and block_tokens=5. Thirteen
    # training tokens therefore provide three complete loader blocks.
    write_tokens(base / "val_000000.bin", [1, 2, 3, 4, 5, 6])
    write_tokens(base / "train_000001.bin", list(range(100, 113)))
    (base / "meta.json").write_text(
        json.dumps({"tokens_written_total": 19}),
        encoding="utf-8",
    )
    conditions = [
        RepairCondition("repair_0pct", 0.0),
        RepairCondition("repair_1pct", 0.34),
        RepairCondition("repair_1p5pct", 0.67),
    ]
    components = [
        component("first", 2, [[7, 8, 9]]),
        component("second", 1, [[10, 11]]),
    ]

    manifest = build_paired_token_repair_datasets(
        base_shards_dir=base,
        output_root=tmp_path / "experiment",
        conditions=conditions,
        components=components,
        batch_size=2,
        seq_len=2,
        selection_seed=42,
        experiment_name="toy",
    )

    assert manifest["eligible_train_blocks"] == 3
    assert manifest["conditions"]["repair_0pct"]["replacement_blocks"] == 0
    assert manifest["conditions"]["repair_1pct"]["replacement_blocks"] == 1
    assert manifest["conditions"]["repair_1p5pct"]["replacement_blocks"] == 2
    one = manifest["conditions"]["repair_1pct"]["selected_blocks"]
    one_point_five = manifest["conditions"]["repair_1p5pct"]["selected_blocks"]
    assert one_point_five[: len(one)] == one

    data_root = tmp_path / "experiment" / "data" / "token_shards"
    base_val = np.fromfile(base / "val_000000.bin", dtype=np.uint16)
    for condition in conditions:
        condition_val = np.fromfile(
            data_root / condition.name / "val_000000.bin",
            dtype=np.uint16,
        )
        assert np.array_equal(base_val, condition_val)
        assert manifest["validation"][condition.name]["validated"]

    blocks = discover_train_blocks(base, batch_size=2, seq_len=2)
    ranked = rank_train_blocks(blocks, seed=42)
    control = np.fromfile(
        data_root / "repair_0pct" / "train_000001.bin",
        dtype=np.uint16,
    )
    repair = np.fromfile(
        data_root / "repair_1pct" / "train_000001.bin",
        dtype=np.uint16,
    )
    selected = ranked[0]
    changed_positions = set(np.flatnonzero(control != repair).tolist())
    allowed_positions = set(range(selected.start_token, selected.start_token + 4))
    assert changed_positions
    assert changed_positions.issubset(allowed_positions)
