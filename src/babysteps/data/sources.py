"""Dataset source helpers shared by data-building and analysis workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


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
    source_field = str(dataset_config.get("source_field", "source"))
    path_field = dataset_config.get("path_field")
    encoding = str(dataset_config.get("encoding", "utf-8"))
    document_mode = str(dataset_config.get("document_mode", "file")).strip().lower()
    skip_empty = bool(dataset_config.get("skip_empty", True))
    source_from_path = bool(dataset_config.get("source_from_path", False))

    if document_mode not in {"file", "line"}:
        raise ValueError("local_text dataset.document_mode must be 'file' or 'line'.")

    def make_row(path: Path, text: str) -> dict[str, str]:
        row = {text_field: text}
        if source_from_path:
            row[source_field] = path.stem
        if path_field:
            row[str(path_field)] = str(path)
        return row

    for path in local_dataset_files(dataset_config):
        if document_mode == "file":
            text = path.read_text(encoding=encoding)
            if text or not skip_empty:
                yield make_row(path, text)
            continue

        with path.open("r", encoding=encoding) as f:
            for line in f:
                text = line.rstrip("\n")
                if skip_empty and not text.strip():
                    continue
                yield make_row(path, text)


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


__all__ = [
    "dataset_kind",
    "iter_dataset_rows",
    "iter_local_jsonl_rows",
    "iter_local_text_rows",
    "iter_text_batches",
    "load_hf_dataset_split",
    "local_dataset_files",
]
