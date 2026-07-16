#!/usr/bin/env python3
"""Build a deliberately leaky corpus of correct BabyLM zero-shot examples.

This is an oracle/diagnostic dataset, not valid competition training data.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "external"
    / "babylm-eval"
    / "strict"
    / "evaluation_data"
    / "full_eval"
)


def _clean(*parts: Any) -> str:
    """Join fields while making every record safe for line-document corpora."""

    return " ".join(" ".join(str(part).split()) for part in parts).strip()


def _rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def _record(
    *,
    suite: str,
    source_file: Path,
    line_number: int,
    variant: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not text:
        raise ValueError(f"Empty correct example at {source_file}:{line_number}")
    record_metadata = {
        "suite": suite,
        "source_file": source_file.name,
        "source_line_number": line_number,
        "correct_variant": variant,
        "exact_eval_leakage": True,
    }
    if metadata:
        record_metadata.update(metadata)
    return {
        "id": f"{suite}_{source_file.stem}_{line_number:06d}_{variant}",
        "text": text,
        "metadata": record_metadata,
    }


def build_unique_records(input_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for suite, directory in (
        ("blimp", input_root / "blimp_filtered"),
        ("supplement", input_root / "supplement_filtered"),
    ):
        for path in sorted(directory.glob("*.jsonl")):
            for line_number, row in _rows(path):
                records.append(
                    _record(
                        suite=suite,
                        source_file=path,
                        line_number=line_number,
                        variant="sentence_good",
                        text=_clean(row["sentence_good"]),
                        metadata={"uid": row.get("UID", path.stem)},
                    )
                )

    for path in sorted((input_root / "comps").glob("*.jsonl")):
        for line_number, row in _rows(path):
            records.append(
                _record(
                    suite="comps",
                    source_file=path,
                    line_number=line_number,
                    variant="acceptable",
                    text=_clean(row["prefix_acceptable"], row["property_phrase"]),
                    metadata={
                        "acceptable_concept": row.get("acceptable_concept", ""),
                        "property": row.get("property", ""),
                    },
                )
            )

    for path in sorted((input_root / "entity_tracking").glob("*.jsonl")):
        for line_number, row in _rows(path):
            options = row["options"]
            if not options:
                raise ValueError(f"No Entity Tracking options at {path}:{line_number}")
            records.append(
                _record(
                    suite="entity_tracking",
                    source_file=path,
                    line_number=line_number,
                    variant="option0",
                    text=_clean(row["input_prefix"], options[0]),
                    metadata={"numops": row.get("numops")},
                )
            )

    for path in sorted((input_root / "ewok_filtered").glob("*.jsonl")):
        for line_number, row in _rows(path):
            for variant, context_key, target_key in (
                ("c1_t1", "Context1", "Target1"),
                ("c2_t2", "Context2", "Target2"),
            ):
                records.append(
                    _record(
                        suite="ewok",
                        source_file=path,
                        line_number=line_number,
                        variant=variant,
                        text=_clean(row[context_key], row[target_key]),
                        metadata={"domain": row.get("Domain", path.stem)},
                    )
                )

    if not records:
        raise FileNotFoundError(f"No zero-shot records found below {input_root}")
    return records


def write_outputs(
    *, input_root: Path, output_dir: Path, repeat: int, unique_records: list[dict[str, Any]]
) -> dict[str, Any]:
    if repeat <= 0:
        raise ValueError("--repeat must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for repeat_index in range(1, repeat + 1):
        for unique_record in unique_records:
            record = {
                "id": f"{unique_record['id']}_copy{repeat_index}",
                "text": unique_record["text"],
                "metadata": {**unique_record["metadata"], "copy_index": repeat_index},
            }
            records.append(record)

    train_path = output_dir / "train.txt"
    documents_path = output_dir / "documents.jsonl"
    manifest_path = output_dir / "manifest.json"
    train_path.write_text(
        "".join(f"{record['text']}\n" for record in records), encoding="utf-8"
    )
    with documents_path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")

    unique_suite_documents = Counter(
        record["metadata"]["suite"] for record in unique_records
    )
    unique_suite_words = Counter()
    for record in unique_records:
        unique_suite_words[record["metadata"]["suite"]] += len(
            re.findall(r"\S+", record["text"])
        )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "repeat": repeat,
        "unique_documents": len(unique_records),
        "documents_total": len(records),
        "unique_words_whitespace": sum(unique_suite_words.values()),
        "words_total_whitespace": repeat * sum(unique_suite_words.values()),
        "unique_documents_by_suite": dict(sorted(unique_suite_documents.items())),
        "unique_words_by_suite": dict(sorted(unique_suite_words.items())),
        "outputs": {
            "train_text": str(train_path),
            "documents": str(documents_path),
            "manifest": str(manifest_path),
        },
        "warning": (
            "Contains exact correct examples from the BabyLM full zero-shot evaluation. "
            "This is a contaminated oracle diagnostic and must not be submitted."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    manifest = write_outputs(
        input_root=input_root,
        output_dir=output_dir,
        repeat=args.repeat,
        unique_records=build_unique_records(input_root),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
