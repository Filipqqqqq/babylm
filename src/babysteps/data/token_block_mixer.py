"""Utilities for fixed-budget token-block data mixing.

This module replaces whole pre-tokenized training chunks with synthetic chunks.
The public knob is a requested synthetic token budget, but replacement happens
in full ``block_size`` content blocks so the output chunk count stays fixed.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import random
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


TokenChunk = Sequence[int]


@dataclass(frozen=True)
class SyntheticSource:
    """One text source to mix into an existing token-chunk dataset."""

    path: Path
    label: str | None = None
    kind: str | None = None
    text_field: str = "text"
    document_mode: str = "paragraph"
    encoding: str = "utf-8"


@dataclass(frozen=True)
class SyntheticDocument:
    """A single synthetic document read from a source file."""

    text: str
    source_label: str
    source_path: str
    source_kind: str
    document_index: int


@dataclass(frozen=True)
class TokenBlockMixResult:
    """Result of replacing baseline chunks with synthetic chunks."""

    mixed_chunks: list[list[int]]
    synthetic_chunks: list[list[int]]
    removed_indices: list[int]
    requested_synthetic_tokens: int
    block_size: int
    synthetic_blocks: int
    effective_synthetic_content_tokens: int
    synthetic_input_tokens: int
    available_synthetic_blocks: int
    documents_loaded: int


def detect_source_kind(path: Path, configured_kind: str | None = None) -> str:
    """Resolve a source kind from config or file extension."""

    if configured_kind:
        kind = configured_kind.strip().lower()
    else:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            kind = "jsonl"
        elif suffix == ".txt":
            kind = "text"
        else:
            raise ValueError(
                f"Could not infer synthetic source kind from {path}. "
                "Set kind to 'text' or 'jsonl'."
            )

    if kind in {"txt", "local_text"}:
        return "text"
    if kind in {"jsonl", "local_jsonl"}:
        return "jsonl"
    if kind == "text":
        return "text"
    raise ValueError(f"Unsupported synthetic source kind: {configured_kind!r}")


def _source_label(path: Path, label: str | None) -> str:
    return label or path.stem


def _read_text_documents(source: SyntheticSource, kind: str) -> list[SyntheticDocument]:
    mode = source.document_mode.strip().lower()
    if mode not in {"file", "line", "paragraph"}:
        raise ValueError("Synthetic text document_mode must be file, line, or paragraph.")

    text = source.path.read_text(encoding=source.encoding)
    if mode == "file":
        pieces = [text]
    elif mode == "line":
        pieces = [line.rstrip("\n") for line in text.splitlines()]
    else:
        pieces = re.split(r"\n\s*\n+", text)

    label = _source_label(source.path, source.label)
    documents = []
    for index, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        documents.append(
            SyntheticDocument(
                text=piece,
                source_label=label,
                source_path=str(source.path),
                source_kind=kind,
                document_index=index,
            )
        )
    return documents


def _read_jsonl_documents(source: SyntheticSource, kind: str) -> list[SyntheticDocument]:
    label = _source_label(source.path, source.label)
    documents = []
    with source.path.open("r", encoding=source.encoding) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected JSON object at {source.path}:{line_number}.")
            text = str(row.get(source.text_field, "")).strip()
            if not text:
                continue
            documents.append(
                SyntheticDocument(
                    text=text,
                    source_label=label,
                    source_path=str(source.path),
                    source_kind=kind,
                    document_index=line_number,
                )
            )
    return documents


def read_synthetic_documents(sources: Iterable[SyntheticSource]) -> list[SyntheticDocument]:
    """Read all configured synthetic sources into document records."""

    documents: list[SyntheticDocument] = []
    for source in sources:
        path = source.path
        if not path.exists():
            raise FileNotFoundError(f"Missing synthetic source: {path}")
        if not path.is_file():
            raise ValueError(f"Synthetic source path is not a file: {path}")

        kind = detect_source_kind(path, source.kind)
        if kind == "jsonl":
            documents.extend(_read_jsonl_documents(source, kind))
        else:
            documents.extend(_read_text_documents(source, kind))

    if not documents:
        raise ValueError("No non-empty synthetic documents were read.")
    return documents


def requested_tokens_to_blocks(requested_tokens: int, block_size: int) -> int:
    """Floor a requested token budget to a number of full content blocks."""

    if requested_tokens < 0:
        raise ValueError("requested_tokens must be non-negative.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    return requested_tokens // block_size


def _tokenize_text(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    try:
        encoded = tokenizer(text=text, add_special_tokens=add_special_tokens)
    except TypeError:
        encoded = tokenizer(text, add_special_tokens=add_special_tokens)

    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("Expected tokenizer output for one text, got a batch.")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def tokenize_documents(
    documents: Sequence[SyntheticDocument],
    tokenizer: Any,
    *,
    add_special_tokens: bool = False,
    separator_text: str = "\n\n",
) -> list[int]:
    """Tokenize documents and concatenate them with a small text separator."""

    separator_ids = (
        _tokenize_text(tokenizer, separator_text, add_special_tokens=False)
        if separator_text
        else []
    )
    token_ids: list[int] = []
    for document in documents:
        document_ids = _tokenize_text(
            tokenizer,
            document.text,
            add_special_tokens=add_special_tokens,
        )
        if not document_ids:
            continue
        if token_ids and separator_ids:
            token_ids.extend(separator_ids)
        token_ids.extend(document_ids)
    return token_ids


def pack_full_blocks(token_ids: Sequence[int], *, block_size: int) -> list[list[int]]:
    """Pack token ids into full content blocks, dropping the final remainder."""

    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    num_blocks = len(token_ids) // block_size
    return [
        list(token_ids[index * block_size : (index + 1) * block_size])
        for index in range(num_blocks)
    ]


def wrap_content_blocks(
    content_blocks: Sequence[Sequence[int]],
    *,
    eos_token_id: int,
) -> list[list[int]]:
    """Wrap content blocks with EOS markers to match BabyLM chunk pickles."""

    return [
        [int(eos_token_id), *[int(token_id) for token_id in block], int(eos_token_id)]
        for block in content_blocks
    ]


def mix_token_blocks(
    base_chunks: Sequence[TokenChunk],
    documents: Sequence[SyntheticDocument],
    tokenizer: Any,
    *,
    requested_synthetic_tokens: int,
    block_size: int,
    eos_token_id: int,
    seed: int,
    add_special_tokens: bool = False,
    separator_text: str = "\n\n",
    shuffle_documents: bool = True,
    shuffle_synthetic_blocks: bool = True,
) -> TokenBlockMixResult:
    """Replace floor(requested tokens / block size) chunks with synthetic chunks."""

    synthetic_blocks = requested_tokens_to_blocks(requested_synthetic_tokens, block_size)
    if synthetic_blocks <= 0:
        raise ValueError(
            "requested_synthetic_tokens is smaller than one full block; "
            "increase it or lower block_size."
        )
    if synthetic_blocks > len(base_chunks):
        raise ValueError(
            f"Cannot replace {synthetic_blocks} blocks in a dataset with "
            f"{len(base_chunks)} chunks."
        )
    expected_chunk_length = block_size + 2
    bad_chunk_indices = [
        index
        for index, chunk in enumerate(base_chunks)
        if len(chunk) != expected_chunk_length
    ][:5]
    if bad_chunk_indices:
        raise ValueError(
            f"Base chunks must be EOS-wrapped blocks of length {expected_chunk_length}; "
            f"found mismatches at indices {bad_chunk_indices}."
        )

    ordered_documents = list(documents)
    if shuffle_documents:
        random.Random(seed + 1).shuffle(ordered_documents)

    synthetic_token_ids = tokenize_documents(
        ordered_documents,
        tokenizer,
        add_special_tokens=add_special_tokens,
        separator_text=separator_text,
    )
    content_blocks = pack_full_blocks(synthetic_token_ids, block_size=block_size)
    if len(content_blocks) < synthetic_blocks:
        raise ValueError(
            f"Requested {synthetic_blocks} synthetic blocks, but sources only "
            f"produced {len(content_blocks)} full blocks."
        )

    selected_blocks = content_blocks[:synthetic_blocks]
    replacement_chunks = wrap_content_blocks(selected_blocks, eos_token_id=eos_token_id)
    if shuffle_synthetic_blocks:
        random.Random(seed + 2).shuffle(replacement_chunks)

    removed_indices = sorted(
        random.Random(seed + 3).sample(range(len(base_chunks)), synthetic_blocks)
    )

    mixed_chunks = [list(chunk) for chunk in base_chunks]
    for index, synthetic_chunk in zip(removed_indices, replacement_chunks):
        mixed_chunks[index] = synthetic_chunk

    return TokenBlockMixResult(
        mixed_chunks=mixed_chunks,
        synthetic_chunks=replacement_chunks,
        removed_indices=removed_indices,
        requested_synthetic_tokens=requested_synthetic_tokens,
        block_size=block_size,
        synthetic_blocks=synthetic_blocks,
        effective_synthetic_content_tokens=synthetic_blocks * block_size,
        synthetic_input_tokens=len(synthetic_token_ids),
        available_synthetic_blocks=len(content_blocks),
        documents_loaded=len(documents),
    )


def load_pickle(path: str | Path) -> Any:
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)


def atomic_write_pickle(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with open(fd, "wb", closefd=True) as f:
        pickle.dump(data, f)
    Path(tmp).replace(path)


def atomic_write_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with open(fd, "w", encoding="utf-8", closefd=True) as f:
        json.dump(data, f, indent=2)
    Path(tmp).replace(path)


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_manifest_entries(sources: Sequence[SyntheticSource]) -> list[dict[str, Any]]:
    entries = []
    for source in sources:
        entry = asdict(source)
        entry["path"] = str(source.path)
        entry["resolved_kind"] = detect_source_kind(source.path, source.kind)
        entry["sha256"] = sha256_file(source.path)
        entries.append(entry)
    return entries


def build_manifest(
    *,
    name: str | None,
    result: TokenBlockMixResult,
    seed: int,
    base_train_chunks: Path,
    output_train_chunks: Path,
    tokenizer_path: Path,
    sources: Sequence[SyntheticSource],
    add_special_tokens: bool,
    separator_text: str,
    shuffle_documents: bool,
    shuffle_synthetic_blocks: bool,
) -> dict[str, Any]:
    """Build a provenance manifest for a completed token-block mix."""

    return {
        "version": "token_block_mix_v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "requested_synthetic_tokens": result.requested_synthetic_tokens,
        "block_size": result.block_size,
        "synthetic_blocks": result.synthetic_blocks,
        "effective_synthetic_content_tokens": result.effective_synthetic_content_tokens,
        "seed": seed,
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256_tokenizer_json": (
                sha256_file(tokenizer_path / "tokenizer.json")
                if (tokenizer_path / "tokenizer.json").exists()
                else None
            ),
        },
        "base": {
            "train_chunks": str(base_train_chunks),
            "sha256": sha256_file(base_train_chunks),
            "num_chunks": len(result.mixed_chunks),
        },
        "synthetic": {
            "sources": source_manifest_entries(sources),
            "documents_loaded": result.documents_loaded,
            "input_tokens": result.synthetic_input_tokens,
            "available_blocks": result.available_synthetic_blocks,
            "used_blocks": result.synthetic_blocks,
            "add_special_tokens": add_special_tokens,
            "separator_text": separator_text,
            "shuffle_documents": shuffle_documents,
            "shuffle_synthetic_blocks": shuffle_synthetic_blocks,
        },
        "replacement": {
            "removed_indices": result.removed_indices,
        },
        "output": {
            "train_chunks": str(output_train_chunks),
            "num_chunks": len(result.mixed_chunks),
            "sha256": sha256_file(output_train_chunks),
        },
    }
