#!/usr/bin/env python3
"""Create the BOS-by-default tokenizer used by the Chatty BERT submission."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer


PROBES = ("Hello world.", "The cat slept on the mat.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def prepare_tokenizer(source: Path, output: Path) -> None:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Source tokenizer directory does not exist: {source}")
    if source == output:
        raise SystemExit("Source and output tokenizer directories must differ.")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output tokenizer directory is not empty: {output}")

    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit("Chatty BERT preparation requires a fast tokenizer.")
    if tokenizer.bos_token is None or tokenizer.bos_token_id is None:
        raise SystemExit("Source tokenizer must define a BOS token and token ID.")

    raw_encodings = {
        text: tokenizer(text, add_special_tokens=False)["input_ids"]
        for text in PROBES
    }
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single=f"{tokenizer.bos_token}:0 $A:0",
        pair=f"{tokenizer.bos_token}:0 $A:0 $B:1",
        special_tokens=[(tokenizer.bos_token, tokenizer.bos_token_id)],
    )

    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output)

    prepared = AutoTokenizer.from_pretrained(output, use_fast=True)
    if prepared.get_vocab() != tokenizer.get_vocab():
        raise SystemExit("Prepared tokenizer vocabulary differs from its source.")
    for text, expected_raw_ids in raw_encodings.items():
        raw_ids = prepared(text, add_special_tokens=False)["input_ids"]
        default = prepared(text, return_offsets_mapping=True)
        if raw_ids != expected_raw_ids:
            raise SystemExit(f"Prepared tokenizer changed the raw encoding for {text!r}.")
        if not default["input_ids"] or default["input_ids"][0] != prepared.bos_token_id:
            raise SystemExit(f"Prepared tokenizer does not prepend BOS for {text!r}.")
        if tuple(default["offset_mapping"][0]) != (0, 0):
            raise SystemExit(f"Prepared tokenizer BOS offset is not (0, 0) for {text!r}.")

    print(f"Prepared BOS-by-default tokenizer: {output}")


def main() -> None:
    args = parse_args()
    prepare_tokenizer(args.source, args.output)


if __name__ == "__main__":
    main()
