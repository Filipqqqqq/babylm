#!/usr/bin/env python3
"""Export a compact review sheet from material-world audited JSONL."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from babysteps.material_world.audit import read_jsonl


REVIEW_COLUMNS = [
    "id",
    "audit_status",
    "material",
    "material_class",
    "object",
    "dynamics",
    "style_id",
    "flag_codes",
    "text",
]


def _metadata(row: dict[str, Any], key: str) -> Any:
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get(key, "")
    return ""


def _as_cell(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return "" if value is None else str(value)


def review_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    output = []
    for row in rows:
        flags = row.get("audit_flags", [])
        flag_codes = []
        if isinstance(flags, list):
            flag_codes = [str(flag.get("code")) for flag in flags if isinstance(flag, dict) and flag.get("code")]
        output.append(
            {
                "id": _as_cell(row.get("id")),
                "audit_status": _as_cell(row.get("audit_status", "")),
                "material": _as_cell(_metadata(row, "material")),
                "material_class": _as_cell(_metadata(row, "material_class")),
                "object": _as_cell(_metadata(row, "object")),
                "dynamics": _as_cell(_metadata(row, "dynamics")),
                "style_id": _as_cell(_metadata(row, "style_id")),
                "flag_codes": "; ".join(flag_codes),
                "text": _as_cell(row.get("text", "")).replace("\n", " "),
            }
        )
    return output


def write_csv(path: str | Path, rows: list[dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_markdown(path: str | Path, rows: list[dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(REVIEW_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in REVIEW_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(row[column]) for column in REVIEW_COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audited_jsonl", help="Path to examples_audited.jsonl or examples_reaudited.jsonl.")
    parser.add_argument("--output-csv", required=True, help="Path for the CSV review sheet.")
    parser.add_argument("--output-md", help="Optional path for a Markdown review sheet.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = review_rows(read_jsonl(args.audited_jsonl))
    write_csv(args.output_csv, rows)
    print(f"CSV review sheet: {args.output_csv}")
    if args.output_md:
        write_markdown(args.output_md, rows)
        print(f"Markdown review sheet: {args.output_md}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
