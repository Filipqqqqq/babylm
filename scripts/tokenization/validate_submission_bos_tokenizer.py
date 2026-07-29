#!/usr/bin/env python3
"""Validate BOS behavior used by BabyLM zero-shot and finetuning evaluators."""

from __future__ import annotations

import argparse
from transformers import AutoTokenizer


PROBES = ("Hello world.", "The cat slept on the mat.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        nargs="+",
        required=True,
        help="One or more tokenizer/checkpoint directories to validate.",
    )
    parser.add_argument(
        "--reference",
        help="Optional tokenizer whose vocabulary and raw encodings must match.",
    )
    return parser.parse_args()


def validate(path: str, reference) -> None:
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tokenizer.bos_token_id is None:
        raise ValueError(f"Tokenizer has no BOS token ID: {path}")

    for text in PROBES:
        default_ids = tokenizer(text)["input_ids"]
        if not default_ids or default_ids[0] != tokenizer.bos_token_id:
            raise ValueError(f"Default encoding does not start with BOS: {path}")

        encoded = tokenizer(text, return_offsets_mapping=True)
        if encoded["offset_mapping"][0] != (0, 0):
            raise ValueError(f"BOS offset is not (0, 0): {path}")

        if reference is not None:
            reference_ids = reference(text, add_special_tokens=False)["input_ids"]
            raw_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if raw_ids != reference_ids:
                raise ValueError(f"Raw encoding differs from reference: {path}")

    if reference is not None and tokenizer.get_vocab() != reference.get_vocab():
        raise ValueError(f"Vocabulary differs from reference: {path}")

    # The official finetuning pipeline uses left padding and default special
    # tokens. Check the first attended token in every row, not column zero.
    batch = tokenizer(
        list(PROBES),
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )
    for input_ids, attention_mask in zip(batch.input_ids, batch.attention_mask):
        attended = input_ids[attention_mask.bool()]
        if not len(attended) or attended[0].item() != tokenizer.bos_token_id:
            raise ValueError(f"Finetuning-style batch does not start with BOS: {path}")

    print(f"BOS tokenizer OK: {path} (bos_token_id={tokenizer.bos_token_id})")


def main() -> None:
    args = parse_args()
    reference = (
        AutoTokenizer.from_pretrained(args.reference)
        if args.reference is not None
        else None
    )
    for path in args.tokenizer:
        validate(path, reference)


if __name__ == "__main__":
    main()
