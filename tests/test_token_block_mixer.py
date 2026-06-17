from __future__ import annotations

import json
import pickle

import pytest

from babysteps.data.token_block_mixer import (
    SyntheticSource,
    atomic_write_json,
    atomic_write_pickle,
    build_manifest,
    mix_token_blocks,
    read_synthetic_documents,
    requested_tokens_to_blocks,
)


class ToyTokenizer:
    eos_token_id = 0

    def __call__(self, text=None, add_special_tokens=False):
        del add_special_tokens
        tokens = [] if text is None else str(text).split()
        return {"input_ids": [int(token) for token in tokens]}


def test_read_synthetic_documents_supports_paragraph_text_and_jsonl(tmp_path) -> None:
    text_path = tmp_path / "synthetic.txt"
    text_path.write_text("first paragraph\n\nsecond paragraph\n", encoding="utf-8")
    jsonl_path = tmp_path / "synthetic.jsonl"
    jsonl_path.write_text(
        json.dumps({"text": "third paragraph"}) + "\n",
        encoding="utf-8",
    )

    documents = read_synthetic_documents(
        [
            SyntheticSource(path=text_path, kind="text", label="txt"),
            SyntheticSource(path=jsonl_path, kind="jsonl", label="jsonl"),
        ]
    )

    assert [document.text for document in documents] == [
        "first paragraph",
        "second paragraph",
        "third paragraph",
    ]
    assert [document.source_label for document in documents] == ["txt", "txt", "jsonl"]


def test_requested_tokens_floor_to_full_blocks() -> None:
    assert requested_tokens_to_blocks(0, 512) == 0
    assert requested_tokens_to_blocks(511, 512) == 0
    assert requested_tokens_to_blocks(512, 512) == 1
    assert requested_tokens_to_blocks(100000, 512) == 195


def test_mix_token_blocks_preserves_count_and_wraps_synthetic_chunks(tmp_path) -> None:
    source_path = tmp_path / "synthetic.txt"
    source_path.write_text("1 2 3 4 5 6 7 8 9 10", encoding="utf-8")
    documents = read_synthetic_documents([SyntheticSource(path=source_path)])
    base_chunks = [[99, index, index, index, index, 99] for index in range(6)]

    result = mix_token_blocks(
        base_chunks,
        documents,
        ToyTokenizer(),
        requested_synthetic_tokens=9,
        block_size=4,
        eos_token_id=ToyTokenizer.eos_token_id,
        seed=7,
        separator_text="",
        shuffle_documents=False,
        shuffle_synthetic_blocks=False,
    )

    assert result.synthetic_blocks == 2
    assert result.effective_synthetic_content_tokens == 8
    assert len(result.mixed_chunks) == len(base_chunks)
    assert len(result.removed_indices) == 2
    assert result.synthetic_chunks == [[0, 1, 2, 3, 4, 0], [0, 5, 6, 7, 8, 0]]
    for index in result.removed_indices:
        assert result.mixed_chunks[index] in result.synthetic_chunks


def test_mix_token_blocks_fails_when_request_exceeds_available_synthetic_blocks(tmp_path) -> None:
    source_path = tmp_path / "synthetic.txt"
    source_path.write_text("1 2 3 4 5", encoding="utf-8")
    documents = read_synthetic_documents([SyntheticSource(path=source_path)])

    with pytest.raises(ValueError, match="only produced 1 full blocks"):
        mix_token_blocks(
            [[99, 1, 2, 3, 4, 99] for _ in range(4)],
            documents,
            ToyTokenizer(),
            requested_synthetic_tokens=8,
            block_size=4,
            eos_token_id=ToyTokenizer.eos_token_id,
            seed=1,
            separator_text="",
            shuffle_documents=False,
        )


def test_manifest_records_core_provenance(tmp_path) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    base_path = tmp_path / "base.pkl"
    output_path = tmp_path / "mixed.pkl"
    source_path = tmp_path / "synthetic.txt"
    source_path.write_text("1 2 3 4", encoding="utf-8")
    atomic_write_pickle([[99, 1, 2, 3, 4, 99]], base_path)
    atomic_write_pickle([[0, 1, 2, 3, 4, 0]], output_path)

    result = mix_token_blocks(
        [[99, 1, 2, 3, 4, 99]],
        read_synthetic_documents([SyntheticSource(path=source_path)]),
        ToyTokenizer(),
        requested_synthetic_tokens=4,
        block_size=4,
        eos_token_id=ToyTokenizer.eos_token_id,
        seed=3,
        separator_text="",
        shuffle_documents=False,
        shuffle_synthetic_blocks=False,
    )
    manifest = build_manifest(
        name="toy_mix",
        result=result,
        seed=3,
        base_train_chunks=base_path,
        output_train_chunks=output_path,
        tokenizer_path=tokenizer_dir,
        sources=[SyntheticSource(path=source_path)],
        add_special_tokens=False,
        separator_text="",
        shuffle_documents=False,
        shuffle_synthetic_blocks=False,
    )
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest, manifest_path)

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["name"] == "toy_mix"
    assert loaded["requested_synthetic_tokens"] == 4
    assert loaded["effective_synthetic_content_tokens"] == 4
    assert loaded["base"]["num_chunks"] == 1
    assert loaded["output"]["num_chunks"] == 1
    assert loaded["replacement"]["removed_indices"] == result.removed_indices
    with output_path.open("rb") as f:
        assert pickle.load(f) == [[0, 1, 2, 3, 4, 0]]
