from __future__ import annotations

import json
from pathlib import Path

import pytest

from babysteps.paths import (
    expand_env_string,
    repair_local_tokenizer_config,
    validate_token_chunk_file,
    validate_tokenizer_dir,
)


def test_env_default_expands_tilde(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BABYSTEPS_CACHE_DIR", raising=False)

    value = "${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}/tokenizers/demo"

    assert (
        expand_env_string(value)
        == str(Path.home() / "babylm-baby-steps-cache" / "tokenizers" / "demo")
    )


def test_validate_tokenizer_dir_reports_actionable_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "tokenizers" / "missing"

    with pytest.raises(FileNotFoundError) as excinfo:
        validate_tokenizer_dir(missing)

    message = str(excinfo.value)
    assert "Tokenizer directory is missing or incomplete." in message
    assert str(missing) in message
    assert "Run scripts/data/train_tokenizer.py first." in message


def test_validate_tokenizer_dir_leaves_hf_model_ids_alone() -> None:
    assert validate_tokenizer_dir("gpt2") is None


def test_validate_tokenizer_dir_repairs_raw_tokenizers_backend_metadata(
    tmp_path: Path,
) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    config_path = tokenizer_dir / "tokenizer_config.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "tokenizers",
                "tokenizer_class": "TokenizersBackend",
                "bos_token": "<s>",
                "eos_token": "</s>",
            }
        ),
        encoding="utf-8",
    )

    assert validate_tokenizer_dir(tokenizer_dir) == tokenizer_dir
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
    }
    assert not repair_local_tokenizer_config(tokenizer_dir)


def test_validate_token_chunk_file_reports_actionable_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "chunks" / "train.pkl"

    with pytest.raises(FileNotFoundError) as excinfo:
        validate_token_chunk_file(missing)

    message = str(excinfo.value)
    assert "Token chunk file is missing." in message
    assert str(missing) in message
    assert "Run scripts/data/build_token_chunks.py first." in message
