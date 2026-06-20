"""Path and environment expansion helpers for portable BabySteps configs."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
)
_INVALID_FAST_TOKENIZER_CLASSES = {"TokenizersBackend"}


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
    """Expand ${VAR:-default}, $VAR/${VAR}, and a leading ``~``."""

    expanded = value
    for _ in range(20):
        next_value = _expand_shell_default_expressions(expanded)
        if next_value == expanded:
            break
        expanded = next_value
    return os.path.expanduser(os.path.expandvars(expanded))


def expand_env_vars(value: Any) -> Any:
    """Recursively expand environment references in loaded config values."""

    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return expand_env_string(value)
    return value


def resolve_path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    """Resolve a path-like config value after env/default expansion."""

    path = Path(expand_env_string(str(value)))
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return path


def looks_like_local_path(value: str | Path) -> bool:
    """Return true for values intended to be filesystem paths, not HF repo IDs."""

    text = str(value).strip()
    if not text:
        return False
    expanded = expand_env_string(text)
    return (
        expanded.startswith(("/", "./", "../", "~"))
        or expanded in {".", ".."}
        or Path(expanded).is_absolute()
    )


def repair_local_tokenizer_config(path: str | Path) -> bool:
    """Repair known-invalid tokenizer metadata written by raw tokenizers backends."""

    path = resolve_path(path)
    config_path = path / "tokenizer_config.json"
    if not config_path.is_file():
        return False

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        return False

    if config.get("tokenizer_class") not in _INVALID_FAST_TOKENIZER_CLASSES:
        return False

    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    if config.get("backend") == "tokenizers":
        config.pop("backend")

    tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, config_path)
    return True


def validate_tokenizer_dir(
    value: str | Path,
    *,
    hint: str = "Run scripts/data/train_tokenizer.py first.",
) -> Path | None:
    """Raise a clear error when a local tokenizer directory is absent or incomplete.

    Hugging Face's ``from_pretrained`` treats missing paths as repo IDs, which
    can produce confusing validation errors. This guard only intervenes for
    obvious local paths and leaves model IDs like ``gpt2`` untouched.
    """

    if not looks_like_local_path(value):
        return None

    path = resolve_path(value)
    present = [name for name in _TOKENIZER_FILES if (path / name).exists()]
    if path.is_dir() and present:
        repair_local_tokenizer_config(path)
        return path

    expected = ", ".join(_TOKENIZER_FILES[:2])
    raise FileNotFoundError(
        "Tokenizer directory is missing or incomplete.\n"
        f"Path: {path}\n"
        f"Expected at least one tokenizer file, e.g. {expected}.\n"
        f"{hint}"
    )


def validate_token_chunk_file(
    value: str | Path,
    *,
    hint: str = "Run scripts/data/build_token_chunks.py first.",
) -> Path:
    """Return an expanded token chunk path, or raise with a build hint."""

    path = resolve_path(value)
    if path.is_file():
        return path

    raise FileNotFoundError(
        "Token chunk file is missing.\n"
        f"Path: {path}\n"
        f"{hint}"
    )
