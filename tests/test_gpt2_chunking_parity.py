"""Parity checks for plain GPT-2 BabyLM 10M token chunks.

This test rebuilds the original ``gpt_chunked.pkl`` recipe from the cleaned
CustomLlama text files and compares it chunk-by-chunk with the stored pickle.
It is gated because tokenizing the full BabyLM 10M cleaned corpus is slower than
the ordinary smoke tests.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pytest
from transformers import AutoTokenizer


CUSTOM_LLAMA_CLEAN_DIR = Path(
    "/SSD/tokenPred/babylm_10m/train_files/CustomLlama/data/babylm_10M_clean"
)
CUSTOM_LLAMA_GPT_CHUNKS = CUSTOM_LLAMA_CLEAN_DIR / "gpt_chunked.pkl"
CHUNK_SIZE = 512

# This is the order printed by the original CustomLlama tokenization notebook.
CUSTOM_LLAMA_SOURCE_ORDER = [
    "childes",
    "switchboard",
    "bnc_spoken",
    "gutenberg",
    "open_subtitles",
    "simple_wiki",
]


def _require_full_chunk_parity() -> None:
    if os.environ.get("BABYSTEPS_FULL_GPT2_CHUNK_PARITY") != "1":
        pytest.skip("Set BABYSTEPS_FULL_GPT2_CHUNK_PARITY=1 to run full chunk parity.")


def _first_difference(left: list[int], right: list[int]) -> str:
    if len(left) != len(right):
        return f"length differs: rebuilt={len(left)} stored={len(right)}"
    for idx, (left_id, right_id) in enumerate(zip(left, right, strict=True)):
        if left_id != right_id:
            return f"token {idx} differs: rebuilt={left_id} stored={right_id}"
    return "chunks are equal"


def test_plain_gpt2_babylm10m_chunks_match_customllama_gpt_chunked() -> None:
    _require_full_chunk_parity()
    if not CUSTOM_LLAMA_CLEAN_DIR.exists():
        pytest.skip(f"CustomLlama cleaned text not found: {CUSTOM_LLAMA_CLEAN_DIR}")
    if not CUSTOM_LLAMA_GPT_CHUNKS.exists():
        pytest.skip(f"CustomLlama gpt_chunked pickle not found: {CUSTOM_LLAMA_GPT_CHUNKS}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eos_token_id = tokenizer.eos_token_id

    with CUSTOM_LLAMA_GPT_CHUNKS.open("rb") as f:
        stored_chunks = pickle.load(f)

    stored_index = 0
    per_source_counts = {}
    for source in CUSTOM_LLAMA_SOURCE_ORDER:
        text_path = CUSTOM_LLAMA_CLEAN_DIR / f"{source}.train"
        assert text_path.exists(), f"Missing cleaned source file: {text_path}"

        text = text_path.read_text(encoding="utf-8")
        token_ids = tokenizer(text=text)["input_ids"]
        num_chunks = len(token_ids) // CHUNK_SIZE
        per_source_counts[source] = num_chunks

        for chunk_idx in range(num_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = start + CHUNK_SIZE
            rebuilt_chunk = [eos_token_id] + token_ids[start:end] + [eos_token_id]

            assert stored_index < len(stored_chunks), (
                f"Rebuilt more chunks than stored pickle. "
                f"source={source} chunk_idx={chunk_idx} stored_len={len(stored_chunks)}"
            )
            stored_chunk = stored_chunks[stored_index]
            assert rebuilt_chunk == stored_chunk, (
                f"Mismatch at global chunk {stored_index}, "
                f"source={source}, source_chunk={chunk_idx}: "
                f"{_first_difference(rebuilt_chunk, stored_chunk)}"
            )
            stored_index += 1

    assert stored_index == len(stored_chunks), (
        f"Stored pickle has extra chunks: rebuilt={stored_index} stored={len(stored_chunks)} "
        f"per_source_counts={per_source_counts}"
    )
