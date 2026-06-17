#!/usr/bin/env python3
"""Train a Hugging Face byte-level BPE tokenizer from cleaned BabyLM text."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast


ROOT = Path(__file__).resolve().parents[2]


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


def collect_train_files(clean_train_dir: Path, sources: list[str]) -> list[Path]:
    """Collect the cleaned train files used to fit the tokenizer."""

    train_files = []
    for source in sources:
        path = clean_train_dir / f"{source}.train"
        if not path.exists():
            raise FileNotFoundError(f"Missing cleaned train text file: {path}")
        train_files.append(path)
    return train_files


def build_byte_bpe_tokenizer(
    *,
    train_files: list[Path],
    vocab_size: int,
    min_frequency: int,
    special_tokens: list[str],
) -> Tokenizer:
    """Train the low-level byte-BPE tokenizer.

    Byte-level BPE gets an initial alphabet containing every byte, so ordinary
    text should remain encodable even when the vocabulary is small.
    """

    if not special_tokens:
        raise ValueError("byte_bpe tokenizers need at least one special token.")

    tokenizer = Tokenizer(BPE(unk_token=special_tokens[0]))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train([str(path) for path in train_files], trainer=trainer)
    return tokenizer


def wrap_as_hf_tokenizer(
    tokenizer: Tokenizer,
    *,
    special_tokens: list[str],
    model_max_length: int,
) -> PreTrainedTokenizerFast:
    """Wrap the raw tokenizer so AutoTokenizer can reload it later."""

    boundary_token = special_tokens[0]
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=boundary_token,
        eos_token=boundary_token,
        unk_token=boundary_token,
        pad_token=boundary_token,
        model_max_length=model_max_length,
    )

    # Additional special tokens are optional for causal GPT-2 experiments, but
    # preserving them here keeps the saved tokenizer faithful to its config.
    if len(special_tokens) > 1:
        hf_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens[1:]})

    return hf_tokenizer


def save_manifest(manifest: dict[str, Any], output_dir: Path) -> None:
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "tokenization" / "byte_bpe_8192_10m_block512.yaml"),
        help="Path to a byte-BPE tokenization/chunking YAML config.",
    )
    parser.add_argument("--clean-train-dir", help="Override data.clean_train_dir.")
    parser.add_argument("--output-dir", help="Override tokenizer.output_dir.")
    parser.add_argument("--vocab-size", type=int, help="Override tokenizer.vocab_size.")
    parser.add_argument("--min-frequency", type=int, help="Override tokenizer.min_frequency.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing tokenizer directory.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    data_config = config["data"]
    tokenizer_config = config["tokenizer"]
    chunks_config = config.get("chunks", {})

    kind = tokenizer_config.get("kind")
    if kind != "byte_bpe":
        raise ValueError(f"train_tokenizer.py only supports tokenizer.kind='byte_bpe', got {kind!r}.")

    sources = list(data_config["sources"])
    clean_train_dir = Path(args.clean_train_dir or data_config["clean_train_dir"])
    output_dir = Path(args.output_dir or tokenizer_config["output_dir"])
    vocab_size = int(args.vocab_size or tokenizer_config["vocab_size"])
    min_frequency = int(args.min_frequency or tokenizer_config.get("min_frequency", 2))
    special_tokens = list(tokenizer_config.get("special_tokens", ["<|endoftext|>"]))
    # Chunk size is a data-packing choice, not necessarily the model's maximum
    # context length. GPT-2-style BabyLM runs currently use 1024 positions.
    model_max_length = int(tokenizer_config.get("model_max_length", 1024))

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Tokenizer output directory is not empty: {output_dir}. "
            "Pass --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files = collect_train_files(clean_train_dir, sources)

    print(f"Config: {Path(args.config).resolve()}")
    print(f"Clean train dir: {clean_train_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Requested vocab size: {vocab_size:,}")
    print(f"Min frequency: {min_frequency}")
    print(f"Special tokens: {special_tokens}")
    print("Train files:")
    for path in train_files:
        print(f"  - {path}")

    tokenizer = build_byte_bpe_tokenizer(
        train_files=train_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
    hf_tokenizer = wrap_as_hf_tokenizer(
        tokenizer,
        special_tokens=special_tokens,
        model_max_length=model_max_length,
    )
    hf_tokenizer.save_pretrained(output_dir)

    # Reload once through AutoTokenizer, because that is how build_token_chunks
    # and the BabyLM eval pipeline will see this tokenizer.
    reloaded = AutoTokenizer.from_pretrained(output_dir)

    manifest = {
        "name": config.get("name"),
        "config": str(Path(args.config).resolve()),
        "kind": "byte_bpe",
        "clean_train_dir": str(clean_train_dir),
        "sources": sources,
        "train_files": [str(path) for path in train_files],
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": len(reloaded),
        "min_frequency": min_frequency,
        "special_tokens": special_tokens,
        "bos_token": reloaded.bos_token,
        "eos_token": reloaded.eos_token,
        "unk_token": reloaded.unk_token,
        "pad_token": reloaded.pad_token,
        "output_dir": str(output_dir),
    }
    save_manifest(manifest, output_dir)

    print(f"Actual tokenizer size: {len(reloaded):,}")
    print(f"EOS token id: {reloaded.eos_token_id}")
    print(f"PAD token id: {reloaded.pad_token_id}")
    print(f"Manifest written to: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
