from __future__ import annotations

import json

import numpy as np

from babysteps.cli import build_token_shards


def test_dataset_kind_defaults_to_hf() -> None:
    assert build_token_shards.dataset_kind({}) == "hf"


def test_iter_dataset_rows_supports_local_jsonl(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [{"text": "hello"}, {"text": "world"}]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    dataset = {
        "kind": "local_jsonl",
        "files": {"train": str(path)},
        "split": "train",
        "text_field": "text",
    }

    assert list(build_token_shards.iter_dataset_rows(dataset)) == rows


def test_iter_dataset_rows_supports_local_text_file_mode(tmp_path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("first document\nsecond line\n", encoding="utf-8")

    dataset = {
        "kind": "local_text",
        "files": [str(path)],
        "text_field": "text",
        "document_mode": "file",
    }

    assert list(build_token_shards.iter_dataset_rows(dataset)) == [
        {"text": "first document\nsecond line\n"}
    ]


def test_iter_dataset_rows_supports_local_text_line_mode(tmp_path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("first\n\nsecond\n", encoding="utf-8")

    dataset = {
        "kind": "local_text",
        "files": str(path),
        "text_field": "text",
        "document_mode": "line",
    }

    assert list(build_token_shards.iter_dataset_rows(dataset)) == [
        {"text": "first"},
        {"text": "second"},
    ]


def test_iter_dataset_rows_supports_local_text_directory(tmp_path) -> None:
    data_dir = tmp_path / "texts"
    data_dir.mkdir()
    (data_dir / "b.txt").write_text("second\n", encoding="utf-8")
    (data_dir / "a.txt").write_text("first\n", encoding="utf-8")
    (data_dir / "skip.md").write_text("skip\n", encoding="utf-8")

    dataset = {
        "kind": "local_text",
        "dir": str(data_dir),
        "glob": "*.txt",
        "text_field": "text",
        "document_mode": "file",
    }

    assert list(build_token_shards.iter_dataset_rows(dataset)) == [
        {"text": "first\n"},
        {"text": "second\n"},
    ]


def test_iter_dataset_rows_supports_recursive_local_text_directory(tmp_path) -> None:
    data_dir = tmp_path / "texts"
    nested = data_dir / "nested"
    nested.mkdir(parents=True)
    (nested / "a.txt").write_text("nested\n", encoding="utf-8")

    dataset = {
        "kind": "local_text",
        "dir": str(data_dir),
        "glob": "*.txt",
        "recursive": True,
        "text_field": "text",
        "document_mode": "line",
    }

    assert list(build_token_shards.iter_dataset_rows(dataset)) == [
        {"text": "nested"},
    ]


def test_build_token_shards_from_local_jsonl(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [{"text": "one two"}, {"text": "three four"}, {"text": "five six"}]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    class ToyTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        def __len__(self) -> int:
            return 32

        def __call__(self, batch, **_kwargs):
            return {
                "input_ids": [
                    [index + 3 for index, _ in enumerate(text.split())]
                    for text in batch
                ]
            }

    output_dir = tmp_path / "shards"
    meta = build_token_shards.build_token_shards(
        {
            "name": "toy",
            "dataset": {
                "kind": "local_jsonl",
                "files": str(path),
                "text_field": "text",
            },
            "tokenizer": {"output_dir": str(tmp_path / "tokenizer")},
            "shards": {
                "output_dir": str(output_dir),
                "shard_tokens": 6,
                "val_shards": 1,
                "batch_docs": 2,
            },
        },
        ToyTokenizer(),
        overwrite=False,
        max_docs=None,
        max_shards=None,
    )

    assert meta["dataset_kind"] == "local_jsonl"
    assert meta["docs_processed"] == 3
    assert meta["num_val_shards"] == 1
    assert meta["num_train_shards"] == 1
    assert (output_dir / "val_000000.bin").exists()
    assert (output_dir / "train_000001.bin").exists()
    assert np.fromfile(output_dir / "val_000000.bin", dtype=np.uint16).tolist() == [
        1,
        3,
        4,
        2,
        1,
        3,
    ]
