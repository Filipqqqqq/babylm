#!/usr/bin/env python3
"""Replace fixed-size BabyLM token chunks with synthetic token chunks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer

from babysteps.data.token_block_mixer import (
    SyntheticSource,
    atomic_write_json,
    atomic_write_pickle,
    build_manifest,
    load_pickle,
    mix_token_blocks,
    read_synthetic_documents,
)


ROOT = Path(__file__).resolve().parents[3]


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return expand_env_vars(yaml.safe_load(f))


def resolve_path(value: str | Path, *, base_dir: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def source_from_config(raw: dict[str, Any]) -> SyntheticSource:
    return SyntheticSource(
        path=resolve_path(raw["path"]),
        label=raw.get("label"),
        kind=raw.get("kind"),
        text_field=raw.get("text_field", "text"),
        document_mode=raw.get("document_mode", "paragraph"),
        encoding=raw.get("encoding", "utf-8"),
    )


def cli_sources(args: argparse.Namespace) -> list[SyntheticSource]:
    if not args.synthetic_file:
        return []
    return [
        SyntheticSource(
            path=resolve_path(path, base_dir=Path.cwd()),
            kind=args.synthetic_kind,
            text_field=args.text_field,
            document_mode=args.document_mode,
            encoding=args.encoding,
        )
        for path in args.synthetic_file
    ]


def config_sources(config: dict[str, Any]) -> list[SyntheticSource]:
    synthetic_config = config.get("synthetic", {})
    return [
        source_from_config(raw_source)
        for raw_source in synthetic_config.get("sources", [])
    ]


def run_from_config_and_args(config: dict[str, Any], args: argparse.Namespace) -> Path:
    base_config = config.get("base", {})
    tokenizer_config = config.get("tokenizer", {})
    synthetic_config = config.get("synthetic", {})
    output_config = config.get("output", {})
    random_config = config.get("random", {})

    base_train_chunks = resolve_path(
        args.source_train_chunks or base_config["train_chunks"],
        base_dir=Path.cwd() if args.source_train_chunks else ROOT,
    )
    tokenizer_path = resolve_path(
        args.tokenizer or tokenizer_config["path"],
        base_dir=Path.cwd() if args.tokenizer else ROOT,
    )
    output_dir = resolve_path(
        args.output_dir or output_config["dir"],
        base_dir=Path.cwd() if args.output_dir else ROOT,
    )
    output_train_chunks = output_dir / output_config.get("train_filename", "train.pkl")
    manifest_path = output_dir / output_config.get("manifest_filename", "manifest.json")

    if output_train_chunks.exists() and not (args.overwrite or output_config.get("overwrite", False)):
        raise FileExistsError(
            f"Output train chunks already exist: {output_train_chunks}. "
            "Pass --overwrite to replace them."
        )

    sources = cli_sources(args) or config_sources(config)
    if not sources:
        raise ValueError("At least one synthetic source is required.")

    requested_tokens = int(
        args.requested_tokens
        if args.requested_tokens is not None
        else synthetic_config["requested_tokens"]
    )
    block_size = int(
        args.block_size
        if args.block_size is not None
        else synthetic_config.get("block_size", 512)
    )
    seed = int(args.seed if args.seed is not None else random_config.get("seed", 42))
    add_special_tokens = bool(synthetic_config.get("add_special_tokens", False))
    separator_text = synthetic_config.get("separator_text", "\n\n")
    shuffle_documents = bool(synthetic_config.get("shuffle_documents", True))
    shuffle_synthetic_blocks = bool(synthetic_config.get("shuffle_synthetic_blocks", True))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer has no eos_token_id: {tokenizer_path}")

    print(f"Base train chunks: {base_train_chunks}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Requested synthetic tokens: {requested_tokens:,}")
    print(f"Block size: {block_size:,}")
    print(f"Seed: {seed}")
    print("Synthetic sources:")
    for source in sources:
        print(f"  - {source.path}")

    base_chunks = load_pickle(base_train_chunks)
    documents = read_synthetic_documents(sources)
    result = mix_token_blocks(
        base_chunks,
        documents,
        tokenizer,
        requested_synthetic_tokens=requested_tokens,
        block_size=block_size,
        eos_token_id=int(tokenizer.eos_token_id),
        seed=seed,
        add_special_tokens=add_special_tokens,
        separator_text=separator_text,
        shuffle_documents=shuffle_documents,
        shuffle_synthetic_blocks=shuffle_synthetic_blocks,
    )

    atomic_write_pickle(result.mixed_chunks, output_train_chunks)
    manifest = build_manifest(
        name=config.get("name"),
        result=result,
        seed=seed,
        base_train_chunks=base_train_chunks,
        output_train_chunks=output_train_chunks,
        tokenizer_path=tokenizer_path,
        sources=sources,
        add_special_tokens=add_special_tokens,
        separator_text=separator_text,
        shuffle_documents=shuffle_documents,
        shuffle_synthetic_blocks=shuffle_synthetic_blocks,
    )
    atomic_write_json(manifest, manifest_path)

    print(f"Synthetic blocks: {result.synthetic_blocks:,}")
    print(f"Effective synthetic content tokens: {result.effective_synthetic_content_tokens:,}")
    print(f"Output train chunks: {output_train_chunks}")
    print(f"Manifest: {manifest_path}")
    return output_train_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "data" / "material_world_token_mix_10m.yaml"),
        help="Path to a token-block mixing YAML config.",
    )
    parser.add_argument("--source-train-chunks", help="Override base.train_chunks.")
    parser.add_argument("--tokenizer", help="Override tokenizer.path.")
    parser.add_argument(
        "--synthetic-file",
        action="append",
        help="Synthetic .txt or .jsonl file. Can be passed more than once.",
    )
    parser.add_argument("--synthetic-kind", choices=["text", "jsonl"], help="Kind for --synthetic-file.")
    parser.add_argument("--text-field", default="text", help="JSONL text field for --synthetic-file.")
    parser.add_argument(
        "--document-mode",
        choices=["file", "line", "paragraph"],
        default="paragraph",
        help="Text document mode for --synthetic-file.",
    )
    parser.add_argument("--encoding", default="utf-8", help="Synthetic source encoding.")
    parser.add_argument("--requested-tokens", type=int, help="Synthetic token budget before flooring.")
    parser.add_argument("--block-size", type=int, help="Override synthetic.block_size.")
    parser.add_argument("--seed", type=int, help="Override random.seed.")
    parser.add_argument("--output-dir", help="Override output.dir.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output train.pkl.")
    args = parser.parse_args()

    config = load_yaml_config(args.config) if args.config else {}
    run_from_config_and_args(config, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
