#!/usr/bin/env python3
"""Train a byte-BPE tokenizer and build packed token shards from text datasets.

The default source is a Hugging Face dataset, matching the original BabyLM
2026 workflow. Local text and JSONL files are also supported for custom
datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import yaml
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast


ROOT = Path(__file__).resolve().parents[3]
UINT16_LIMIT = 2**16

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _find_shell_expression_end(text: str, start: int) -> int:
    depth = 0
    index = start + 2
    while index < len(text):
        if text.startswith("${", index):
            depth += 1
            index += 2
            continue
        if text[index] == "}":
            if depth == 0:
                return index
            depth -= 1
        index += 1
    return -1


def _split_shell_default(inner: str) -> tuple[str, str] | None:
    depth = 0
    index = 0
    while index < len(inner):
        if inner.startswith("${", index):
            depth += 1
            index += 2
            continue
        if inner[index] == "}" and depth > 0:
            depth -= 1
            index += 1
            continue
        if inner.startswith(":-", index) and depth == 0:
            return inner[:index], inner[index + 2 :]
        index += 1
    return None


def _expand_shell_default_expressions(text: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(text):
        if text.startswith("${", index):
            end = _find_shell_expression_end(text, index)
            if end >= 0:
                inner = text[index + 2 : end]
                split = _split_shell_default(inner)
                if split is not None:
                    name, default = split
                    if _ENV_NAME_RE.match(name):
                        env_value = os.environ.get(name)
                        replacement = env_value if env_value else expand_env_string(default)
                        result.append(replacement)
                        index = end + 1
                        continue

        result.append(text[index])
        index += 1

    return "".join(result)


def expand_env_string(value: str) -> str:
    """Expand ${VAR:-default}, then ordinary $VAR/${VAR} references."""

    expanded = value
    for _ in range(20):
        next_value = _expand_shell_default_expressions(expanded)
        if next_value == expanded:
            break
        expanded = next_value
    return os.path.expandvars(expanded)


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return expand_env_string(value)
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return expand_env_vars(yaml.safe_load(f))


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def atomic_write_bin(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        np.asarray(arr, dtype=np.uint16).tofile(f)
    os.replace(tmp, path)


def dataset_kind(dataset_config: dict[str, Any]) -> str:
    """Return the configured dataset source kind.

    Existing configs did not specify a kind, so Hugging Face remains the
    default for backward compatibility.
    """

    return str(dataset_config.get("kind", "hf")).strip().lower()


def load_hf_dataset_split(dataset_config: dict[str, Any]):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset sources require the 'datasets' package. "
            "Install project dependencies after pulling the updated pyproject."
        ) from exc

    dataset_name = dataset_config["name"]
    dataset_kwargs: dict[str, Any] = {
        "split": dataset_config.get("split", "train"),
        "streaming": bool(dataset_config.get("streaming", True)),
    }
    dataset_config_name = dataset_config.get("config")
    if dataset_config_name:
        dataset_kwargs["name"] = dataset_config_name
    return load_dataset(dataset_name, **dataset_kwargs)


def _as_path_list(value: Any) -> list[Path]:
    if isinstance(value, (str, os.PathLike)):
        return [Path(value)]
    if isinstance(value, list):
        return [Path(item) for item in value]
    raise TypeError(
        f"Expected a path string or list of paths, got {type(value).__name__}."
    )


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError(f"Expected a string or list of strings, got {type(value).__name__}.")


def _split_value(value: Any, dataset_config: dict[str, Any], field_name: str) -> Any:
    if not isinstance(value, dict):
        return value

    split = str(dataset_config.get("split", "train"))
    if split not in value:
        available = ", ".join(sorted(str(key) for key in value))
        raise KeyError(
            f"{field_name} has no entry for split {split!r}; available: {available}"
        )
    return value[split]


def _default_local_glob(dataset_config: dict[str, Any]) -> str:
    kind = dataset_kind(dataset_config)
    if kind in {"local_jsonl", "jsonl"}:
        return "*.jsonl"
    if kind in {"local_text", "text", "txt"}:
        return "*.txt"
    return "*"


def _expand_directory_files(dataset_config: dict[str, Any]) -> list[Path]:
    raw_dirs = dataset_config.get(
        "dirs",
        dataset_config.get("dir", dataset_config.get("directory")),
    )
    if raw_dirs is None:
        return []

    raw_dirs = _split_value(raw_dirs, dataset_config, "dataset.dir")
    raw_globs = _split_value(
        dataset_config.get("glob", _default_local_glob(dataset_config)),
        dataset_config,
        "dataset.glob",
    )
    patterns = _as_str_list(raw_globs)
    if bool(dataset_config.get("recursive", False)):
        patterns = [
            pattern if "**" in pattern else f"**/{pattern}"
            for pattern in patterns
        ]

    files: list[Path] = []
    for directory in _as_path_list(raw_dirs):
        if not directory.exists():
            raise FileNotFoundError(f"Missing local dataset directory: {directory}")
        if not directory.is_dir():
            raise ValueError(
                f"Local dataset directory path is not a directory: {directory}"
            )

        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(path for path in directory.glob(pattern) if path.is_file())
        if not matches:
            joined = ", ".join(patterns)
            raise FileNotFoundError(f"No local dataset files matched {joined} under {directory}")
        files.extend(sorted(set(matches)))

    return files


def local_dataset_files(dataset_config: dict[str, Any]) -> list[Path]:
    """Resolve local files from explicit file paths or directory globs."""

    files: list[Path] = []
    raw_files = dataset_config.get("files", dataset_config.get("path"))
    if raw_files is not None:
        raw_files = _split_value(raw_files, dataset_config, "dataset.files")
        for path in _as_path_list(raw_files):
            if not path.exists():
                raise FileNotFoundError(f"Missing local dataset file: {path}")
            if not path.is_file():
                raise ValueError(f"Local dataset path is not a file: {path}")
            files.append(path)

    files.extend(_expand_directory_files(dataset_config))
    if not files:
        raise KeyError("Local dataset sources require dataset.files/path or dataset.dir/dirs.")

    return list(dict.fromkeys(files))


def iter_local_jsonl_rows(dataset_config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    encoding = str(dataset_config.get("encoding", "utf-8"))
    for path in local_dataset_files(dataset_config):
        with path.open("r", encoding=encoding) as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"Expected JSON object at {path}:{line_number}.")
                yield row


def iter_local_text_rows(dataset_config: dict[str, Any]) -> Iterator[dict[str, str]]:
    text_field = str(dataset_config.get("text_field", "text"))
    encoding = str(dataset_config.get("encoding", "utf-8"))
    document_mode = str(dataset_config.get("document_mode", "file")).strip().lower()
    skip_empty = bool(dataset_config.get("skip_empty", True))

    if document_mode not in {"file", "line"}:
        raise ValueError("local_text dataset.document_mode must be 'file' or 'line'.")

    for path in local_dataset_files(dataset_config):
        if document_mode == "file":
            text = path.read_text(encoding=encoding)
            if text or not skip_empty:
                yield {text_field: text}
            continue

        with path.open("r", encoding=encoding) as f:
            for line in f:
                text = line.rstrip("\n")
                if skip_empty and not text.strip():
                    continue
                yield {text_field: text}


def iter_dataset_rows(dataset_config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    kind = dataset_kind(dataset_config)
    if kind in {"hf", "huggingface", "hugging_face"}:
        return load_hf_dataset_split(dataset_config)
    if kind in {"local_jsonl", "jsonl"}:
        return iter_local_jsonl_rows(dataset_config)
    if kind in {"local_text", "text", "txt"}:
        return iter_local_text_rows(dataset_config)
    raise ValueError(
        f"Unsupported dataset.kind={kind!r}; expected hf, local_jsonl, or local_text."
    )


def iter_text_batches(
    dataset: Iterable[dict[str, Any]],
    *,
    text_field: str,
    batch_docs: int,
    max_docs: int | None = None,
) -> Iterator[list[str]]:
    batch: list[str] = []
    docs_seen = 0
    for example in dataset:
        if max_docs is not None and docs_seen >= max_docs:
            break
        if text_field not in example:
            raise KeyError(f"Dataset row is missing text field {text_field!r}: {list(example.keys())}")
        text = example[text_field]
        if text is None:
            continue
        batch.append(str(text))
        docs_seen += 1
        if len(batch) >= batch_docs:
            yield batch
            batch = []
    if batch:
        yield batch


def build_byte_bpe_tokenizer(
    *,
    text_batches: Iterator[list[str]],
    vocab_size: int,
    min_frequency: int,
    special_tokens: Sequence[str],
) -> Tokenizer:
    if not special_tokens:
        raise ValueError("At least one special token is required for byte-BPE training.")
    tokenizer = Tokenizer(BPE(unk_token=str(special_tokens[0])))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=list(special_tokens),
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(text_batches, trainer=trainer)
    return tokenizer


def wrap_as_hf_tokenizer(tokenizer: Tokenizer, tokenizer_config: dict[str, Any]) -> PreTrainedTokenizerFast:
    special_tokens = list(tokenizer_config.get("special_tokens", ["<|endoftext|>"]))
    unk_token = tokenizer_config.get("unk_token", special_tokens[0])
    bos_token = tokenizer_config.get("bos_token", special_tokens[0])
    eos_token = tokenizer_config.get("eos_token", special_tokens[0])
    pad_token = tokenizer_config.get("pad_token", eos_token)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        model_max_length=int(tokenizer_config.get("model_max_length", 1024)),
    )
    additional_tokens = [
        token
        for token in special_tokens
        if token not in {unk_token, bos_token, eos_token, pad_token}
    ]
    if additional_tokens:
        hf_tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens})
    return hf_tokenizer


def train_or_load_tokenizer(
    config: dict[str, Any],
    *,
    overwrite: bool,
    max_docs: int | None,
) -> PreTrainedTokenizerFast:
    dataset_config = config["dataset"]
    tokenizer_config = config["tokenizer"]
    output_dir = Path(tokenizer_config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        print(f"Tokenizer exists, reusing: {output_dir}")
        return AutoTokenizer.from_pretrained(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    text_batches = iter_text_batches(
        iter_dataset_rows(dataset_config),
        text_field=str(dataset_config.get("text_field", "text")),
        batch_docs=int(tokenizer_config.get("train_batch_docs", 1000)),
        max_docs=max_docs,
    )
    tokenizer = build_byte_bpe_tokenizer(
        text_batches=text_batches,
        vocab_size=int(tokenizer_config["vocab_size"]),
        min_frequency=int(tokenizer_config.get("min_frequency", 2)),
        special_tokens=list(tokenizer_config.get("special_tokens", ["<|endoftext|>"])),
    )
    hf_tokenizer = wrap_as_hf_tokenizer(tokenizer, tokenizer_config)
    hf_tokenizer.save_pretrained(output_dir)
    hf_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "name": config.get("name"),
        "dataset": dataset_config,
        "tokenizer": tokenizer_config,
        "actual_vocab_size": len(hf_tokenizer),
        "unk_token_id": hf_tokenizer.unk_token_id,
        "bos_token_id": hf_tokenizer.bos_token_id,
        "eos_token_id": hf_tokenizer.eos_token_id,
        "pad_token_id": hf_tokenizer.pad_token_id,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(f"Tokenizer written to: {output_dir}")
    print(f"Actual vocab size: {len(hf_tokenizer):,}")
    print(
        "Special ids: "
        f"unk={hf_tokenizer.unk_token_id} "
        f"bos={hf_tokenizer.bos_token_id} "
        f"eos={hf_tokenizer.eos_token_id} "
        f"pad={hf_tokenizer.pad_token_id}"
    )
    return hf_tokenizer


class TokenShardSink:
    """Pack document token ids into fixed-size uint16 shards."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        shard_tokens: int,
        val_shards: int,
        max_shards: int | None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.shard_tokens = int(shard_tokens)
        self.val_shards = int(val_shards)
        self.max_shards = max_shards
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer = np.empty((self.shard_tokens,), dtype=np.uint16)
        self.buffer_len = 0
        self.next_shard_idx = 0
        self.total_tokens_written = 0
        self.shards_written = {"val": 0, "train": 0}

    @property
    def at_capacity(self) -> bool:
        return self.max_shards is not None and self.next_shard_idx >= self.max_shards

    def append(self, tokens: np.ndarray) -> None:
        if self.at_capacity:
            return
        if tokens.dtype != np.uint16:
            raise TypeError(f"Expected uint16 tokens, got {tokens.dtype!r}.")
        offset = 0
        total = int(tokens.size)
        while offset < total:
            if self.at_capacity:
                return
            take = min(self.shard_tokens - self.buffer_len, total - offset)
            self.buffer[self.buffer_len : self.buffer_len + take] = tokens[offset : offset + take]
            self.buffer_len += take
            offset += take
            if self.buffer_len == self.shard_tokens:
                self._flush(self.buffer)
                self.buffer = np.empty((self.shard_tokens,), dtype=np.uint16)
                self.buffer_len = 0

    def finalize(self) -> None:
        if self.buffer_len > 0 and not self.at_capacity:
            self._flush(self.buffer[: self.buffer_len].copy())
            self.buffer_len = 0

    def _split_for_next_shard(self) -> str:
        return "val" if self.next_shard_idx < self.val_shards else "train"

    def _path_for_next_shard(self) -> Path:
        split = self._split_for_next_shard()
        return self.output_dir / f"{split}_{self.next_shard_idx:06d}.bin"

    def _flush(self, tokens: np.ndarray) -> None:
        split = self._split_for_next_shard()
        path = self._path_for_next_shard()
        atomic_write_bin(path, tokens)
        self.shards_written[split] += 1
        self.next_shard_idx += 1
        self.total_tokens_written += int(tokens.size)
        print(f"wrote {path} ({tokens.size:,} tokens)", flush=True)


def encode_document_uint16(token_ids: Sequence[int], *, bos_token_id: int, eos_token_id: int) -> np.ndarray:
    ids = np.asarray(token_ids, dtype=np.int64)
    max_id = max(int(bos_token_id), int(eos_token_id))
    min_id = min(int(bos_token_id), int(eos_token_id))
    if ids.size:
        max_id = max(max_id, int(ids.max()))
        min_id = min(min_id, int(ids.min()))
    if min_id < 0 or max_id >= UINT16_LIMIT:
        raise ValueError(f"Token id out of uint16 range: min={min_id}, max={max_id}.")

    out = np.empty((ids.size + 2,), dtype=np.uint16)
    out[0] = int(bos_token_id)
    if ids.size:
        out[1:-1] = ids.astype(np.uint16, copy=False)
    out[-1] = int(eos_token_id)
    return out


def build_token_shards(
    config: dict[str, Any],
    tokenizer,
    *,
    overwrite: bool,
    max_docs: int | None,
    max_shards: int | None,
) -> dict[str, Any]:
    dataset_config = config["dataset"]
    shard_config = config["shards"]
    output_dir = Path(shard_config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Shard output directory is not empty: {output_dir}. "
            "Pass --overwrite-shards to replace it."
        )
    if overwrite and output_dir.exists():
        for path in output_dir.glob("*.bin"):
            path.unlink()
        meta_path = output_dir / "meta.json"
        if meta_path.exists():
            meta_path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("Tokenizer must define bos_token_id and eos_token_id.")
    if len(tokenizer) > UINT16_LIMIT:
        raise ValueError(f"Tokenizer vocab size {len(tokenizer)} exceeds uint16 capacity.")

    resolved_max_shards = max_shards
    if resolved_max_shards is None and shard_config.get("max_shards") is not None:
        resolved_max_shards = int(shard_config["max_shards"])

    sink = TokenShardSink(
        output_dir=output_dir,
        shard_tokens=int(shard_config.get("shard_tokens", 1_000_000)),
        val_shards=int(shard_config.get("val_shards", 1)),
        max_shards=resolved_max_shards,
    )

    started = time.perf_counter()
    docs_processed = 0
    text_batches = iter_text_batches(
        iter_dataset_rows(dataset_config),
        text_field=str(dataset_config.get("text_field", "text")),
        batch_docs=int(shard_config.get("batch_docs", 512)),
        max_docs=max_docs,
    )
    progress = tqdm(desc="documents", unit="doc")
    for batch in text_batches:
        if sink.at_capacity:
            break
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        for token_ids in encoded:
            if sink.at_capacity:
                break
            sink.append(
                encode_document_uint16(
                    token_ids,
                    bos_token_id=int(bos_token_id),
                    eos_token_id=int(eos_token_id),
                )
            )
            docs_processed += 1
            progress.update(1)
    progress.close()
    sink.finalize()

    elapsed = time.perf_counter() - started
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "name": config.get("name"),
        "dataset": dataset_config.get(
            "name",
            dataset_config.get("files", dataset_kind(dataset_config)),
        ),
        "dataset_kind": dataset_kind(dataset_config),
        "dataset_config": dataset_config,
        "config": dataset_config.get("config"),
        "split": dataset_config.get("split", "train"),
        "text_field": dataset_config.get("text_field", "text"),
        "tokenizer": str(config["tokenizer"]["output_dir"]),
        "vocab_size": len(tokenizer),
        "dtype": "uint16",
        "bos_token_id": int(bos_token_id),
        "eos_token_id": int(eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None,
        "doc_format": "[BOS] + byte_bpe(text) + [EOS]",
        "shard_tokens": int(shard_config.get("shard_tokens", 1_000_000)),
        "val_shards": int(shard_config.get("val_shards", 1)),
        "num_shards_total": int(sink.next_shard_idx),
        "num_val_shards": int(sink.shards_written["val"]),
        "num_train_shards": int(sink.shards_written["train"]),
        "tokens_written_total": int(sink.total_tokens_written),
        "docs_processed": int(docs_processed),
        "elapsed_seconds": float(elapsed),
        "throughput_tokens_per_sec": float(sink.total_tokens_written / max(elapsed, 1e-9)),
    }
    atomic_write_json(output_dir / "meta.json", meta)
    print(f"Shard meta written to: {output_dir / 'meta.json'}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            ROOT
            / "configs"
            / "tokenization"
            / "babylm2026_strict_byte_bpe_16384_shards.yaml"
        ),
        help="Path to a token-shard build config.",
    )
    parser.add_argument("--overwrite-tokenizer", action="store_true")
    parser.add_argument("--overwrite-shards", action="store_true")
    parser.add_argument("--tokenizer-output-dir", help="Override tokenizer.output_dir.")
    parser.add_argument("--shards-output-dir", help="Override shards.output_dir.")
    parser.add_argument("--shard-tokens", type=int, default=0, help="Override shards.shard_tokens.")
    parser.add_argument("--val-shards", type=int, default=-1, help="Override shards.val_shards.")
    parser.add_argument("--max-docs", type=int, default=0, help="Optional document cap for smoke builds.")
    parser.add_argument("--max-shards", type=int, default=0, help="Optional shard cap for smoke builds.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.tokenizer_output_dir:
        config["tokenizer"]["output_dir"] = args.tokenizer_output_dir
    if args.shards_output_dir:
        config["shards"]["output_dir"] = args.shards_output_dir
    if args.shard_tokens > 0:
        config["shards"]["shard_tokens"] = int(args.shard_tokens)
    if args.val_shards >= 0:
        config["shards"]["val_shards"] = int(args.val_shards)
    max_docs = int(args.max_docs) if args.max_docs > 0 else None
    max_shards = int(args.max_shards) if args.max_shards > 0 else None

    tokenizer = train_or_load_tokenizer(
        config,
        overwrite=bool(args.overwrite_tokenizer),
        max_docs=max_docs,
    )
    build_token_shards(
        config,
        tokenizer,
        overwrite=bool(args.overwrite_shards),
        max_docs=max_docs,
        max_shards=max_shards,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
