#!/usr/bin/env python3
"""Report repeated n-grams in generated concept-curriculum text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from babysteps.concept_curriculum.repetition import build_repetition_report


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected JSON object in {path}.")
            rows.append(row)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL with a text field.")
    parser.add_argument("--output", help="Output report JSON. Prints JSON to stdout if omitted.")
    parser.add_argument("--text-field", default="text", help="Text field path, default: text.")
    parser.add_argument("--min-n", type=int, default=3, help="Smallest n-gram size.")
    parser.add_argument("--max-n", type=int, default=6, help="Largest n-gram size.")
    parser.add_argument("--min-count", type=int, default=2, help="Minimum total n-gram count.")
    parser.add_argument(
        "--min-documents",
        type=int,
        default=2,
        help="Minimum number of documents containing an n-gram.",
    )
    parser.add_argument("--top-k", type=int, default=25, help="Top n-grams to keep for each n.")
    parser.add_argument(
        "--sample-documents",
        type=int,
        default=5,
        help="Document id examples to include for each n-gram.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Keep original case instead of lowercasing tokens.",
    )
    parser.add_argument(
        "--group-by",
        action="append",
        default=[],
        help=(
            "Optional top-level or metadata field for grouped reports. "
            "May be repeated, e.g. --group-by family --group-by style_id."
        ),
    )
    return parser.parse_args(argv)


def _print_summary(report: dict[str, Any], *, output: Path | None) -> None:
    print(f"Documents: {report['document_count']}")
    print(f"Tokens: {report['token_count']}")
    for n, summary in report["summary"].items():
        print(
            f"{n}-grams: repeated_unique={summary['repeated_unique_ngrams']} "
            f"repeated_occurrences={summary['repeated_occurrences']} "
            f"ratio={summary['repetition_ratio']:.3f}"
        )
    if output is not None:
        print(f"Report: {output}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = read_jsonl(args.input)
    report = build_repetition_report(
        rows,
        min_n=args.min_n,
        max_n=args.max_n,
        min_count=args.min_count,
        min_documents=args.min_documents,
        top_k=args.top_k,
        sample_documents=args.sample_documents,
        text_field=args.text_field,
        lowercase=not args.case_sensitive,
        group_by=args.group_by,
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _print_summary(report, output=output)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
