#!/usr/bin/env python3
"""Audit generated material-world text against EWoK material-domain targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from babysteps.material_world.ewok_coverage import (  # noqa: E402
    DEFAULT_MATERIAL_DOMAINS,
    audit_ewok_coverage,
)


def _candidate_ewok_dirs() -> list[Path]:
    return [
        ROOT / "external" / "babylm-eval" / "strict" / "evaluation_data" / "fast_eval" / "ewok_fast",
        Path("/SSD/tokenPred/babylm_10m/evaluation-pipeline-2025/evaluation_data/fast_eval/ewok_fast"),
    ]


def resolve_default_ewok_dir() -> Path:
    for path in _candidate_ewok_dirs():
        if path.exists():
            return path
    candidates = "\n  - ".join(str(path) for path in _candidate_ewok_dirs())
    raise FileNotFoundError(
        "Could not find an extracted EWoK fast directory. Pass --ewok-dir. "
        f"Tried:\n  - {candidates}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ewok-dir",
        help="Directory containing EWoK domain JSONL files, e.g. ewok_fast/.",
    )
    parser.add_argument(
        "--generated",
        help=(
            "Optional generated corpus to audit. Supports JSONL with a text field "
            "or plain text paragraphs. Omit for a target-only report."
        ),
    )
    parser.add_argument(
        "--report-json",
        required=True,
        help="Path for the coverage report JSON.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DEFAULT_MATERIAL_DOMAINS),
        help="EWoK domains to audit.",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Text field when --generated points to JSONL.",
    )
    parser.add_argument(
        "--min-anchor-mentions",
        type=int,
        default=1,
        help="Generated mentions required for an EWoK lexical anchor to count as covered.",
    )
    parser.add_argument(
        "--min-concept-mentions",
        type=int,
        default=1,
        help="Generated mentions required for an EWoK concept to count as covered.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ewok_dir = Path(args.ewok_dir) if args.ewok_dir else resolve_default_ewok_dir()
    report = audit_ewok_coverage(
        ewok_dir=ewok_dir,
        generated_path=args.generated,
        report_json=args.report_json,
        domains=args.domains,
        text_field=args.text_field,
        min_anchor_mentions=args.min_anchor_mentions,
        min_concept_mentions=args.min_concept_mentions,
    )

    print(f"EWoK dir: {ewok_dir}")
    if args.generated:
        print(f"Generated corpus: {args.generated}")
    print(f"Report: {args.report_json}")
    for domain, summary in report["summary"]["domains"].items():
        print(
            f"{domain}: concepts {summary['concepts_covered']}; "
            f"anchors {summary['lexical_anchors_covered']}"
        )
        missing = summary["missing_lexical_anchors"]
        if missing:
            preview = ", ".join(missing[:12])
            suffix = "..." if len(missing) > 12 else ""
            print(f"  missing anchors: {preview}{suffix}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
