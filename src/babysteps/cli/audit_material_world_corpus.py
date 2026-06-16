#!/usr/bin/env python3
"""Audit a material-world synthetic prose JSONL file for quality and diversity issues."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.material_world.audit import audit_jsonl_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", help="Generated or edited examples JSONL, e.g. prose_edited.jsonl.")
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Path for examples_audited.jsonl.",
    )
    parser.add_argument(
        "--report-json",
        required=True,
        help="Path for audit_report.json.",
    )
    parser.add_argument(
        "--drop-ids-output",
        help="Optional path for drop_ids.txt containing hard-drop example ids.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "material_world" / "audit_config.json"),
        help="Path to audit_config.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_jsonl_file(
        args.input_jsonl,
        output_jsonl=args.output_jsonl,
        report_json=args.report_json,
        drop_ids_output=args.drop_ids_output,
        config_path=args.config,
    )
    print(f"Audited {result.report['total_examples']} examples")
    print(f"Kept: {result.report['examples_kept']}")
    print(f"Warned: {result.report['examples_warned']}")
    print(f"Dropped: {result.report['examples_dropped']}")
    print(f"Report: {args.report_json}")
    print(f"Audited JSONL: {args.output_jsonl}")
    if args.drop_ids_output:
        print(f"Drop ids: {args.drop_ids_output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
