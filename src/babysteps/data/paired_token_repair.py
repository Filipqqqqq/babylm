"""Build nested token-block interventions for controlled corpus experiments.

The utilities in this module start from one canonical token-shard corpus.  Each
condition copies that corpus, preserves validation exactly, and replaces a
nested, seeded subset of training-loader blocks with a deterministic synthetic
token stream.  This makes the treatment fraction the only corpus difference.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


UINT16_BYTES = np.dtype(np.uint16).itemsize


@dataclass(frozen=True)
class TrainBlock:
    """One full block consumed by ``TokenShardDataset``."""

    shard_name: str
    block_index: int
    start_token: int
    stride_tokens: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "shard_name": self.shard_name,
            "block_index": self.block_index,
            "start_token": self.start_token,
            "stride_tokens": self.stride_tokens,
        }


@dataclass(frozen=True)
class SyntheticComponent:
    """A weighted stream of already-tokenized synthetic documents."""

    name: str
    weight: float
    documents: tuple[np.ndarray, ...]
    source_path: str
    source_sha256: str


@dataclass(frozen=True)
class RepairCondition:
    """One nested intervention condition."""

    name: str
    fraction: float


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def discover_train_blocks(
    base_shards_dir: str | Path,
    *,
    batch_size: int,
    seq_len: int,
) -> list[TrainBlock]:
    """Return every complete loader block in canonical shard order."""

    base_shards_dir = Path(base_shards_dir)
    stride_tokens = int(batch_size) * int(seq_len)
    block_tokens = stride_tokens + 1
    if stride_tokens <= 0:
        raise ValueError("batch_size and seq_len must be positive.")

    shard_paths = sorted(base_shards_dir.glob("train_*.bin"))
    if not shard_paths:
        raise FileNotFoundError(f"No canonical training shards under {base_shards_dir}.")

    blocks: list[TrainBlock] = []
    for shard_path in shard_paths:
        nbytes = shard_path.stat().st_size
        if nbytes % UINT16_BYTES:
            raise ValueError(f"Shard is not uint16-aligned: {shard_path}")
        shard_tokens = nbytes // UINT16_BYTES
        blocks_in_shard = max(0, 1 + (shard_tokens - block_tokens) // stride_tokens)
        for block_index in range(blocks_in_shard):
            blocks.append(
                TrainBlock(
                    shard_name=shard_path.name,
                    block_index=block_index,
                    start_token=block_index * stride_tokens,
                    stride_tokens=stride_tokens,
                )
            )
    if not blocks:
        raise ValueError(f"Canonical shards contain no complete loader blocks: {base_shards_dir}")
    return blocks


def rank_train_blocks(blocks: Sequence[TrainBlock], *, seed: int) -> list[TrainBlock]:
    """Return a deterministic random ranking used by every nested condition."""

    ranked = list(blocks)
    random.Random(int(seed)).shuffle(ranked)
    return ranked


def replacement_block_count(total_blocks: int, fraction: float) -> int:
    if total_blocks <= 0:
        raise ValueError("total_blocks must be positive.")
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}.")
    count = int(round(total_blocks * float(fraction)))
    if fraction > 0 and count == 0:
        count = 1
    return count


def build_weighted_synthetic_stream(
    components: Sequence[SyntheticComponent],
    *,
    total_tokens: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Interleave documents with weighted-fair scheduling to an exact length.

    The returned source-label vector is used only for provenance counts.  A
    component repeats its documents in their stored order when its pool is
    exhausted, matching the repeat-oriented nature of the original repair mix.
    """

    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative.")
    if total_tokens == 0:
        return (
            np.empty((0,), dtype=np.uint16),
            np.empty((0,), dtype=np.uint16),
            {component.name: 0 for component in components},
        )
    if not components:
        raise ValueError("At least one synthetic component is required.")

    weights = np.asarray([float(component.weight) for component in components], dtype=float)
    if np.any(weights <= 0):
        raise ValueError("Synthetic component weights must be positive.")
    weights /= weights.sum()
    for component in components:
        if not component.documents:
            raise ValueError(f"Synthetic component {component.name!r} has no documents.")
        for document in component.documents:
            if document.dtype != np.uint16:
                raise TypeError(
                    f"Synthetic component {component.name!r} has a non-uint16 document."
                )
            if document.size == 0:
                raise ValueError(
                    f"Synthetic component {component.name!r} has an empty document."
                )

    produced = np.zeros((len(components),), dtype=np.int64)
    next_document = np.zeros((len(components),), dtype=np.int64)
    token_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    remaining = int(total_tokens)

    while remaining:
        normalized_progress = produced / weights
        component_index = int(np.argmin(normalized_progress))
        component = components[component_index]
        document_index = int(next_document[component_index] % len(component.documents))
        document = component.documents[document_index]
        next_document[component_index] += 1
        take = min(remaining, int(document.size))
        token_chunks.append(document[:take])
        label_chunks.append(np.full((take,), component_index, dtype=np.uint16))
        produced[component_index] += take
        remaining -= take

    tokens = np.concatenate(token_chunks).astype(np.uint16, copy=False)
    labels = np.concatenate(label_chunks).astype(np.uint16, copy=False)
    counts = {
        component.name: int(produced[index])
        for index, component in enumerate(components)
    }
    return tokens, labels, counts


def _copy_canonical_shards(base_shards_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    shard_paths = sorted(base_shards_dir.glob("*.bin"))
    if not shard_paths:
        raise FileNotFoundError(f"No canonical binary shards under {base_shards_dir}.")
    for shard_path in shard_paths:
        shutil.copy2(shard_path, destination / shard_path.name)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _shard_hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(directory.glob("*.bin"))
    }


def build_paired_token_repair_datasets(
    *,
    base_shards_dir: str | Path,
    output_root: str | Path,
    conditions: Sequence[RepairCondition],
    components: Sequence[SyntheticComponent],
    batch_size: int,
    seq_len: int,
    selection_seed: int,
    experiment_name: str,
    config_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build and validate nested 0%/repair token-shard conditions."""

    base_shards_dir = Path(base_shards_dir).resolve()
    output_root = Path(output_root).resolve()
    data_root = output_root / "data" / "token_shards"
    manifests_root = output_root / "manifests"
    if not base_shards_dir.is_dir():
        raise FileNotFoundError(f"Missing canonical shard directory: {base_shards_dir}")
    if not conditions:
        raise ValueError("At least one repair condition is required.")

    condition_names = [condition.name for condition in conditions]
    if len(condition_names) != len(set(condition_names)):
        raise ValueError("Repair condition names must be unique.")
    ordered_conditions = sorted(conditions, key=lambda condition: condition.fraction)
    if ordered_conditions[0].fraction != 0.0:
        raise ValueError("The paired experiment must include a 0% control condition.")

    blocks = discover_train_blocks(
        base_shards_dir,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    ranked_blocks = rank_train_blocks(blocks, seed=selection_seed)
    replacement_counts = {
        condition.name: replacement_block_count(len(blocks), condition.fraction)
        for condition in ordered_conditions
    }
    max_replacement_blocks = max(replacement_counts.values())
    stride_tokens = int(batch_size) * int(seq_len)
    synthetic_tokens, synthetic_labels, master_component_counts = (
        build_weighted_synthetic_stream(
            components,
            total_tokens=max_replacement_blocks * stride_tokens,
        )
    )

    base_train_paths = sorted(base_shards_dir.glob("train_*.bin"))
    base_val_paths = sorted(base_shards_dir.glob("val_*.bin"))
    if not base_val_paths:
        raise FileNotFoundError(f"No canonical validation shards under {base_shards_dir}.")
    base_hashes = _shard_hashes(base_shards_dir)

    condition_manifests: dict[str, dict[str, Any]] = {}
    for condition in ordered_conditions:
        condition_dir = data_root / condition.name
        if condition_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Condition output already exists: {condition_dir}. Use overwrite explicitly."
                )
            shutil.rmtree(condition_dir)
        _copy_canonical_shards(base_shards_dir, condition_dir)

        selected_count = replacement_counts[condition.name]
        selected_blocks = ranked_blocks[:selected_count]
        open_shards: dict[str, np.memmap] = {}
        for rank, block in enumerate(selected_blocks):
            if block.shard_name not in open_shards:
                open_shards[block.shard_name] = np.memmap(
                    condition_dir / block.shard_name,
                    dtype=np.uint16,
                    mode="r+",
                )
            shard = open_shards[block.shard_name]
            stream_start = rank * stride_tokens
            stream_end = stream_start + stride_tokens
            shard_start = block.start_token
            shard_end = shard_start + stride_tokens
            shard[shard_start:shard_end] = synthetic_tokens[stream_start:stream_end]
        for shard in open_shards.values():
            shard.flush()
        open_shards.clear()

        prefix_labels = synthetic_labels[: selected_count * stride_tokens]
        component_token_counts = {
            component.name: int(np.count_nonzero(prefix_labels == index))
            for index, component in enumerate(components)
        }
        condition_hashes = _shard_hashes(condition_dir)
        manifest = {
            "name": condition.name,
            "requested_fraction": float(condition.fraction),
            "eligible_train_blocks": len(blocks),
            "replacement_blocks": selected_count,
            "actual_block_fraction": selected_count / len(blocks),
            "stride_tokens": stride_tokens,
            "replacement_tokens": selected_count * stride_tokens,
            "selection_seed": int(selection_seed),
            "selected_blocks": [block.to_dict() for block in selected_blocks],
            "synthetic_component_tokens": component_token_counts,
            "shard_sha256": condition_hashes,
            "validation_sha256": {
                path.name: condition_hashes[path.name] for path in base_val_paths
            },
            "boundary_note": (
                "TokenShardDataset uses one-token overlap between adjacent blocks. "
                "Replacement regions cover each selected block's input span; the shared "
                "boundary target remains canonical."
            ),
        }

        base_meta_path = base_shards_dir / "meta.json"
        base_meta: dict[str, Any] = {}
        if base_meta_path.is_file():
            base_meta = json.loads(base_meta_path.read_text(encoding="utf-8"))
        paired_meta = dict(base_meta)
        paired_meta["paired_token_repair"] = {
            "experiment": experiment_name,
            "condition": condition.name,
            "requested_fraction": float(condition.fraction),
            "actual_block_fraction": manifest["actual_block_fraction"],
            "replacement_blocks": selected_count,
            "selection_seed": int(selection_seed),
            "canonical_shards": str(base_shards_dir),
        }
        _write_json(condition_dir / "meta.json", paired_meta)
        _write_json(condition_dir / "condition_manifest.json", manifest)
        condition_manifests[condition.name] = manifest

    component_manifest = [
        {
            "name": component.name,
            "weight": float(component.weight),
            "documents": len(component.documents),
            "source_path": component.source_path,
            "source_sha256": component.source_sha256,
            "master_stream_tokens": master_component_counts[component.name],
        }
        for component in components
    ]
    experiment_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment_name,
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "canonical_shards": str(base_shards_dir),
        "canonical_shard_sha256": base_hashes,
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "stride_tokens": stride_tokens,
        "eligible_train_blocks": len(blocks),
        "selection_seed": int(selection_seed),
        "ranked_blocks": [block.to_dict() for block in ranked_blocks],
        "synthetic_components": component_manifest,
        "conditions": condition_manifests,
    }
    _write_json(manifests_root / "experiment_manifest.json", experiment_manifest)
    _write_json(
        manifests_root / "validation_blocks.json",
        {
            "canonical_shards": str(base_shards_dir),
            "validation_sha256": {
                path.name: base_hashes[path.name] for path in base_val_paths
            },
        },
    )
    for condition in ordered_conditions:
        if condition.fraction <= 0:
            continue
        _write_json(
            manifests_root / f"repair_blocks_{condition.name.removeprefix('repair_')}.json",
            {
                "condition": condition.name,
                "requested_fraction": float(condition.fraction),
                "selected_blocks": condition_manifests[condition.name]["selected_blocks"],
            },
        )
    _write_json(
        manifests_root / "dataset_hashes.json",
        {
            "canonical": base_hashes,
            "conditions": {
                name: manifest["shard_sha256"]
                for name, manifest in condition_manifests.items()
            },
        },
    )

    validation = validate_paired_token_repair_datasets(
        base_shards_dir=base_shards_dir,
        output_root=output_root,
        conditions=ordered_conditions,
        ranked_blocks=ranked_blocks,
        synthetic_tokens=synthetic_tokens,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    experiment_manifest["validation"] = validation
    _write_json(manifests_root / "experiment_manifest.json", experiment_manifest)
    return experiment_manifest


def validate_paired_token_repair_datasets(
    *,
    base_shards_dir: str | Path,
    output_root: str | Path,
    conditions: Sequence[RepairCondition],
    ranked_blocks: Sequence[TrainBlock],
    synthetic_tokens: np.ndarray,
    batch_size: int,
    seq_len: int,
) -> dict[str, Any]:
    """Prove that validation and all untreated training tokens are identical."""

    base_shards_dir = Path(base_shards_dir)
    data_root = Path(output_root) / "data" / "token_shards"
    stride_tokens = int(batch_size) * int(seq_len)
    base_paths = sorted(base_shards_dir.glob("*.bin"))
    results: dict[str, Any] = {}

    previous_selected: set[tuple[str, int]] = set()
    for condition in sorted(conditions, key=lambda item: item.fraction):
        condition_dir = data_root / condition.name
        selected_count = replacement_block_count(len(ranked_blocks), condition.fraction)
        selected_blocks = list(ranked_blocks[:selected_count])
        selected_ids = {(block.shard_name, block.block_index) for block in selected_blocks}
        if not previous_selected.issubset(selected_ids):
            raise AssertionError(f"Condition {condition.name} is not nested.")
        previous_selected = selected_ids

        by_shard: dict[str, list[tuple[int, TrainBlock]]] = {}
        for rank, block in enumerate(selected_blocks):
            by_shard.setdefault(block.shard_name, []).append((rank, block))

        for base_path in base_paths:
            condition_path = condition_dir / base_path.name
            if not condition_path.is_file():
                raise FileNotFoundError(f"Missing condition shard: {condition_path}")
            if condition_path.stat().st_size != base_path.stat().st_size:
                raise AssertionError(f"Shard size changed: {condition_path}")

            base_tokens = np.memmap(base_path, dtype=np.uint16, mode="r")
            condition_tokens = np.memmap(condition_path, dtype=np.uint16, mode="r")
            expected = np.asarray(base_tokens).copy()
            for rank, block in by_shard.get(base_path.name, []):
                stream_start = rank * stride_tokens
                stream_end = stream_start + stride_tokens
                shard_start = block.start_token
                shard_end = shard_start + stride_tokens
                expected[shard_start:shard_end] = synthetic_tokens[stream_start:stream_end]
            if not np.array_equal(expected, condition_tokens):
                mismatch = int(np.flatnonzero(expected != condition_tokens)[0])
                raise AssertionError(
                    f"Unexpected token difference in {condition_path} at token {mismatch}."
                )

        results[condition.name] = {
            "validated": True,
            "replacement_blocks": selected_count,
            "actual_block_fraction": selected_count / len(ranked_blocks),
            "validation_byte_identical": True,
            "untreated_training_tokens_identical": True,
            "nested_with_previous_condition": True,
        }
    return results


__all__ = [
    "RepairCondition",
    "SyntheticComponent",
    "TrainBlock",
    "build_paired_token_repair_datasets",
    "build_weighted_synthetic_stream",
    "discover_train_blocks",
    "rank_train_blocks",
    "replacement_block_count",
    "sha256_file",
    "validate_paired_token_repair_datasets",
]
