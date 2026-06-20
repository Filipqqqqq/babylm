#!/usr/bin/env python3
"""Build BabyLM token chunk pickle files from cleaned text."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.paths import expand_env_vars, resolve_path, validate_tokenizer_dir


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return expand_env_vars(yaml.safe_load(f))


def load_tokenizer(tokenizer_config: dict[str, Any]):
    kind = tokenizer_config.get("kind", "pretrained_hf")
    if kind == "pretrained_hf":
        tokenizer_source = tokenizer_config["name_or_path"]
        tokenizer_path = validate_tokenizer_dir(tokenizer_source)
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path or tokenizer_source))
    elif kind == "byte_bpe":
        tokenizer_path = validate_tokenizer_dir(tokenizer_config["output_dir"])
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    else:
        raise ValueError(f"Unsupported tokenizer kind: {kind!r}")

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_split_chunks(
    *,
    clean_dir: Path,
    suffix: str,
    sources: list[str],
    tokenizer,
    chunk_size: int,
    add_special_tokens: bool,
    wrap_with_eos: bool,
    drop_remainder: bool,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    chunks: list[list[int]] = []
    records: list[dict[str, Any]] = []

    if wrap_with_eos and tokenizer.eos_token_id is None:
        raise ValueError("wrap_with_eos=True requires a tokenizer with eos_token_id.")

    for source in sources:
        path = clean_dir / f"{source}.{suffix}"
        if not path.exists():
            raise FileNotFoundError(f"Missing cleaned text file: {path}")

        text = path.read_text(encoding="utf-8")
        token_ids = tokenizer(text=text, add_special_tokens=add_special_tokens)["input_ids"]
        num_chunks = len(token_ids) // chunk_size
        if not drop_remainder and len(token_ids) % chunk_size:
            num_chunks += 1

        start_count = len(chunks)
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = start + chunk_size
            chunk = token_ids[start:end]
            if drop_remainder and len(chunk) < chunk_size:
                continue
            if wrap_with_eos:
                chunk = [tokenizer.eos_token_id] + chunk + [tokenizer.eos_token_id]
            chunks.append(chunk)

        records.append(
            {
                "source": source,
                "path": str(path),
                "chars": len(text),
                "tokens": len(token_ids),
                "chunks": len(chunks) - start_count,
            }
        )

    return chunks, records


def save_pickle(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "tokenization" / "gpt2_pretrained_10m_block512.yaml"),
        help="Path to a tokenization/chunking YAML config.",
    )
    parser.add_argument("--output-dir", help="Override chunks.output_dir.")
    parser.add_argument("--clean-train-dir", help="Override data.clean_train_dir.")
    parser.add_argument("--clean-dev-dir", help="Override data.clean_dev_dir.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "dev"],
        default=["train", "dev"],
        help="Splits to build.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    data_config = config["data"]
    chunks_config = config["chunks"]
    tokenizer = load_tokenizer(config["tokenizer"])

    sources = list(data_config["sources"])
    output_dir = resolve_path(
        args.output_dir or chunks_config["output_dir"],
        base_dir=Path.cwd() if args.output_dir else ROOT,
    )
    clean_train_dir = resolve_path(
        args.clean_train_dir or data_config["clean_train_dir"],
        base_dir=Path.cwd() if args.clean_train_dir else ROOT,
    )
    clean_dev_dir = resolve_path(
        args.clean_dev_dir or data_config["clean_dev_dir"],
        base_dir=Path.cwd() if args.clean_dev_dir else ROOT,
    )

    chunk_size = int(chunks_config.get("chunk_size", 512))
    add_special_tokens = bool(chunks_config.get("add_special_tokens", False))
    wrap_with_eos = bool(chunks_config.get("wrap_with_eos", True))
    drop_remainder = bool(chunks_config.get("drop_remainder", True))

    manifest: dict[str, Any] = {
        "name": config.get("name"),
        "config": str(Path(args.config).resolve()),
        "tokenizer": config["tokenizer"],
        "sources": sources,
        "chunk_size": chunk_size,
        "add_special_tokens": add_special_tokens,
        "wrap_with_eos": wrap_with_eos,
        "drop_remainder": drop_remainder,
        "splits": {},
    }

    split_dirs = {
        "train": (clean_train_dir, "train"),
        "dev": (clean_dev_dir, "dev"),
    }
    for split in args.splits:
        clean_dir, suffix = split_dirs[split]
        chunks, records = build_split_chunks(
            clean_dir=clean_dir,
            suffix=suffix,
            sources=sources,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            add_special_tokens=add_special_tokens,
            wrap_with_eos=wrap_with_eos,
            drop_remainder=drop_remainder,
        )
        output_path = output_dir / f"{split}.pkl"
        save_pickle(chunks, output_path)
        manifest["splits"][split] = {
            "clean_dir": str(clean_dir),
            "output_path": str(output_path),
            "num_chunks": len(chunks),
            "records": records,
        }
        print(f"{split}: wrote {len(chunks):,} chunks to {output_path}")

    save_json(manifest, output_dir / "manifest.json")
    print(f"Manifest written to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
