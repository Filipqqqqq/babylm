#!/usr/bin/env python3
"""Build a leaky EWoK correct-pair text probe corpus."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.paths import resolve_path


REQUIRED_FIELDS = ("Context1", "Target1", "Context2", "Target2")


def _infer_source_label(path: Path) -> str:
    if "full_eval" in path.parts:
        return "ewok_full"
    if "fast_eval" in path.parts:
        return "ewok_fast"
    return "ewok"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "ewok"


def _read_jsonl(path: Path, *, source_label: str | None = None) -> list[dict[str, Any]]:
    rows = []
    resolved_source_label = source_label or _infer_source_label(path)
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                joined = ", ".join(missing)
                raise KeyError(f"Missing {joined} at {path}:{line_number}")
            row["_line_number"] = line_number
            row["_source_file"] = path.name
            row["_source_label"] = resolved_source_label
            rows.append(row)
    return rows


def _read_input_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        jsonl_paths = sorted(path.glob("*.jsonl"))
        if not jsonl_paths:
            raise FileNotFoundError(f"No JSONL files found in directory: {path}")
        source_label = _infer_source_label(path)
        rows = []
        for jsonl_path in jsonl_paths:
            rows.extend(_read_jsonl(jsonl_path, source_label=source_label))
        return rows
    return _read_jsonl(path)


def _join_pair(context: str, target: str) -> str:
    return f"{context.strip()} {target.strip()}".strip()


def build_correct_pair_records(rows: list[dict[str, Any]], *, repeat: int) -> list[dict[str, Any]]:
    if repeat <= 0:
        raise ValueError("--repeat must be positive.")

    records = []
    for row_index, row in enumerate(rows, start=1):
        domain = str(row.get("Domain", ""))
        source_file = str(row.get("_source_file", "ewok.jsonl"))
        source_label = str(row.get("_source_label", "ewok"))
        id_prefix = _slug(domain or Path(source_file).stem)
        pairs = [
            ("c1_t1", _join_pair(str(row["Context1"]), str(row["Target1"]))),
            ("c2_t2", _join_pair(str(row["Context2"]), str(row["Target2"]))),
        ]
        for repeat_index in range(1, repeat + 1):
            for pair_id, text in pairs:
                records.append(
                    {
                        "id": f"ewok_{id_prefix}_{row_index:05d}_{pair_id}_repeat{repeat_index}",
                        "text": text,
                        "metadata": {
                            "source": source_label,
                            "source_file": source_file,
                            "leakage_probe": True,
                            "domain": domain,
                            "concept_a": row.get("ConceptA", ""),
                            "concept_b": row.get("ConceptB", ""),
                            "context_type": row.get("ContextType", ""),
                            "context_diff": row.get("ContextDiff", ""),
                            "target_diff": row.get("TargetDiff", ""),
                            "source_line_number": row.get("_line_number"),
                            "correct_pair": pair_id,
                            "repeat_index": repeat_index,
                        },
                    }
                )
    return records


def write_outputs(
    *,
    input_path: Path,
    output_dir: Path,
    repeat: int,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.txt"
    documents_path = output_dir / "documents.jsonl"
    manifest_path = output_dir / "manifest.json"

    train_path.write_text(
        "".join(f"{record['text']}\n" for record in records),
        encoding="utf-8",
    )
    with documents_path.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    pair_counts = Counter(
        (
            str(record["metadata"].get("concept_a", "")),
            str(record["metadata"].get("concept_b", "")),
        )
        for record in records
    )
    domain_counts = Counter(
        str(record["metadata"].get("domain", ""))
        for record in records
    )
    source_file_counts = Counter(
        str(record["metadata"].get("source_file", ""))
        for record in records
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "repeat": repeat,
        "items_total": len(records) // (2 * repeat),
        "unique_correct_pairs": len(records) // repeat,
        "documents_total": len(records),
        "pair_counts": {
            f"{left}/{right}": count for (left, right), count in sorted(pair_counts.items())
        },
        "domain_counts": dict(sorted(domain_counts.items())),
        "source_file_counts": dict(sorted(source_file_counts.items())),
        "outputs": {
            "train_text": str(train_path),
            "documents": str(documents_path),
            "manifest": str(manifest_path),
        },
        "warning": (
            "This corpus contains exact EWoK evaluation contexts paired with their "
            "correct targets. Use only as a leaky oracle/upper-bound diagnostic."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="external/babylm-eval/strict/evaluation_data/fast_eval/ewok_fast/physical-dynamics.jsonl",
        help=(
            "EWoK JSONL file or directory of JSONL files containing "
            "Context1/Target1 and Context2/Target2."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/analysis/ewok_leakage/archive/physical_dynamics_correct_pairs_repeat4",
        help="Directory for train.txt, documents.jsonl, and manifest.json.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=4,
        help="Number of times to write each correct context-target pair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input, base_dir=ROOT)
    output_dir = resolve_path(args.output_dir, base_dir=ROOT)
    rows = _read_input_rows(input_path)
    records = build_correct_pair_records(rows, repeat=args.repeat)
    manifest = write_outputs(
        input_path=input_path,
        output_dir=output_dir,
        repeat=args.repeat,
        records=records,
    )
    print(f"Items: {manifest['items_total']:,}")
    print(f"Unique correct pairs: {manifest['unique_correct_pairs']:,}")
    print(f"Train documents: {manifest['documents_total']:,}")
    print(f"Train text: {manifest['outputs']['train_text']}")
    print(f"Manifest: {manifest['outputs']['manifest']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
