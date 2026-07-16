#!/usr/bin/env python3
"""Build or validate the controlled GPT-BERT linguistic-repair datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.data.paired_token_repair import (  # noqa: E402
    RepairCondition,
    SyntheticComponent,
    build_paired_token_repair_datasets,
    build_weighted_synthetic_stream,
    discover_train_blocks,
    rank_train_blocks,
    replacement_block_count,
    sha256_file,
    validate_paired_token_repair_datasets,
)
from babysteps.paths import expand_env_vars, resolve_path, validate_tokenizer_dir  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "linguistic_repair_dose_response_v1"
    / "experiment.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override experiment.output_root from the YAML config.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing condition shards without rebuilding them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing condition shard directories.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = expand_env_vars(yaml.safe_load(handle))
    if not isinstance(config, dict):
        raise TypeError("Experiment config must be a mapping.")
    return config


def resolve_config_path(value: str | Path) -> Path:
    return resolve_path(value, base_dir=ROOT).resolve()


def tokenizer_boundary_ids(tokenizer) -> tuple[int, int]:
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None:
        bos_token_id = getattr(tokenizer, "cls_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "sep_token_id", None)
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("Tokenizer must define bos/eos or cls/sep token ids.")
    return int(bos_token_id), int(eos_token_id)


def tokenize_component_documents(
    path: Path,
    tokenizer,
    *,
    batch_docs: int = 256,
) -> tuple[np.ndarray, ...]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Synthetic source has no non-empty documents: {path}")
    bos_token_id, eos_token_id = tokenizer_boundary_ids(tokenizer)
    documents: list[np.ndarray] = []
    for offset in range(0, len(lines), batch_docs):
        encoded = tokenizer(
            lines[offset : offset + batch_docs],
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        for token_ids in encoded:
            document = np.empty((len(token_ids) + 2,), dtype=np.uint16)
            document[0] = bos_token_id
            if token_ids:
                document[1:-1] = np.asarray(token_ids, dtype=np.uint16)
            document[-1] = eos_token_id
            documents.append(document)
    return tuple(documents)


def prepare_inputs(config: dict[str, Any]):
    from transformers import AutoTokenizer

    experiment = config["experiment"]
    loader = config["loader"]
    base_shards_dir = resolve_config_path(experiment["base_shards_dir"])
    tokenizer_dir = resolve_config_path(experiment["tokenizer_dir"])
    validate_tokenizer_dir(tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=bool(experiment.get("trust_remote_code", True)),
    )

    conditions = [
        RepairCondition(name=str(item["name"]), fraction=float(item["fraction"]))
        for item in config["conditions"]
    ]
    components: list[SyntheticComponent] = []
    for item in config["synthetic_components"]:
        source_path = resolve_config_path(item["path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing synthetic component: {source_path}")
        documents = tokenize_component_documents(source_path, tokenizer)
        components.append(
            SyntheticComponent(
                name=str(item["name"]),
                weight=float(item["allocation_words"]),
                documents=documents,
                source_path=str(source_path),
                source_sha256=sha256_file(source_path),
            )
        )

    return {
        "experiment": experiment,
        "loader": loader,
        "base_shards_dir": base_shards_dir,
        "tokenizer_dir": tokenizer_dir,
        "conditions": conditions,
        "components": components,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    prepared = prepare_inputs(config)
    experiment = prepared["experiment"]
    loader = prepared["loader"]
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else resolve_config_path(experiment["output_root"])
    )

    batch_size = int(loader["batch_size"])
    seq_len = int(loader["seq_len"])
    selection_seed = int(experiment["selection_seed"])
    conditions = prepared["conditions"]
    components = prepared["components"]
    base_shards_dir = prepared["base_shards_dir"]

    if args.validate_only:
        blocks = discover_train_blocks(
            base_shards_dir,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        ranked_blocks = rank_train_blocks(blocks, seed=selection_seed)
        max_blocks = max(
            replacement_block_count(len(blocks), condition.fraction)
            for condition in conditions
        )
        synthetic_tokens, _, _ = build_weighted_synthetic_stream(
            components,
            total_tokens=max_blocks * batch_size * seq_len,
        )
        result = validate_paired_token_repair_datasets(
            base_shards_dir=base_shards_dir,
            output_root=output_root,
            conditions=conditions,
            ranked_blocks=ranked_blocks,
            synthetic_tokens=synthetic_tokens,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        print(f"Validated paired datasets: {output_root / 'data' / 'token_shards'}")
        return

    manifest = build_paired_token_repair_datasets(
        base_shards_dir=base_shards_dir,
        output_root=output_root,
        conditions=conditions,
        components=components,
        batch_size=batch_size,
        seq_len=seq_len,
        selection_seed=selection_seed,
        experiment_name=str(experiment["name"]),
        config_path=config_path,
        overwrite=bool(args.overwrite),
    )
    print(f"Experiment: {manifest['experiment']}")
    print(f"Eligible loader blocks: {manifest['eligible_train_blocks']}")
    for name, condition in manifest["conditions"].items():
        print(
            f"- {name}: replacement_blocks={condition['replacement_blocks']} "
            f"actual_fraction={condition['actual_block_fraction']:.6f}"
        )
    print(f"Manifest: {output_root / 'manifests' / 'experiment_manifest.json'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
