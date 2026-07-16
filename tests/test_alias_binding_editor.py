from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from babysteps.concept_curriculum.alias_binding_corpus import read_jsonl
from babysteps.concept_curriculum.alias_binding_editor import (
    DETERMINISTIC,
    AliasBindingEditConfig,
    edit_alias_binding_corpus,
    validate_edited_text,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _toy_document(document_id: str = "alias_binding_00000000") -> dict[str, Any]:
    return {
        "id": document_id,
        "text": (
            "Mira acted as a turtle and Nola acted as a violin. "
            "Mira had a hard shell. Nola made music with strings."
        ),
        "word_count": 20,
        "sequence_family": "role_binding",
        "carrier_type": "ordinary_name",
        "bindings": [
            {
                "handle": "Mira",
                "concept_id": "turtle",
                "concept": "turtle",
                "fact_id": "turtle__fact0",
                "fact": "had a hard shell",
                "fact_type": "appearance",
                "template_id": "turtle__fact0__template0",
                "template_type": "direct",
                "rendered_template": "Mira had a hard shell.",
            },
            {
                "handle": "Nola",
                "concept_id": "violin",
                "concept": "violin",
                "fact_id": "violin__fact0",
                "fact": "made music with strings",
                "fact_type": "function",
                "template_id": "violin__fact0__template0",
                "template_type": "direct",
                "rendered_template": "Nola made music with strings.",
            },
        ],
    }


def test_validate_edited_text_accepts_structure_preserving_rewrite() -> None:
    row = _toy_document()
    edited = (
        "In the game, Mira played as a turtle and Nola played as a violin. "
        "Mira had a hard shell. Nola made music with strings."
    )

    assert validate_edited_text(row, edited) == []


def test_validate_edited_text_rejects_bad_rewrites() -> None:
    row = _toy_document()

    missing_handle = (
        "In the game, Mira played as a turtle and someone played as a violin. "
        "Mira had a hard shell. Music came from strings."
    )
    forbidden = (
        "A wug is a turtle. Therefore, Mira had a hard shell. "
        "Nola acted as a violin and made music with strings."
    )

    assert "missing_handle:Nola" in validate_edited_text(row, missing_handle)
    assert "forbidden_wugs_or_therefore_pattern" in validate_edited_text(row, forbidden)


def test_deterministic_editor_writes_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "draft"
    output_dir = tmp_path / "edited"
    rows = [_toy_document("doc0"), _toy_document("doc1")]
    _write_jsonl(input_dir / "documents.jsonl", rows)

    result = edit_alias_binding_corpus(
        AliasBindingEditConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            renderer=DETERMINISTIC,
            batch_size=1,
        )
    )

    for path in result.outputs.values():
        assert path.exists()
    final_rows = read_jsonl(output_dir / "documents.jsonl")
    train_lines = (output_dir / "train.txt").read_text(encoding="utf-8").splitlines()
    assert len(final_rows) == 2
    assert train_lines == [row["text"] for row in final_rows]
    assert {row["edit_status"] for row in final_rows} == {"kept_original_qwen_keep"}
    assert result.manifest["counts"]["final_documents"] == 2


class BadRewriteRenderer:
    def __call__(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        stage: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        del prompt, system_prompt, temperature, max_tokens, stage, response_format
        return json.dumps(
            {
                "items": [
                    {
                        "id": "doc0",
                        "decision": "rewrite",
                        "edited_text": "Mira acted as a turtle. Mira had a hard shell.",
                        "reason": "bad rewrite for test",
                    }
                ]
            }
        )


def test_invalid_rewrite_keeps_original(tmp_path: Path) -> None:
    input_dir = tmp_path / "draft"
    output_dir = tmp_path / "edited"
    _write_jsonl(input_dir / "documents.jsonl", [_toy_document("doc0")])

    edit_alias_binding_corpus(
        AliasBindingEditConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            renderer=DETERMINISTIC,
        ),
        renderer=BadRewriteRenderer(),
    )

    final = read_jsonl(output_dir / "documents.jsonl")[0]
    assert final["edit_status"] == "kept_original_invalid_rewrite"
    assert final["text"] == final["original_text"]
    assert "missing_handle:Nola" in final["edit_validation_reasons"]
