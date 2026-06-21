#!/usr/bin/env python3
"""Export EWoK ConceptA/ConceptB columns into lexical-analysis target files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.analysis.lexical.ewok_targets import export_ewok_targets


DEFAULT_EWOK_DIR = (
    ROOT
    / "external"
    / "babylm-eval"
    / "strict"
    / "evaluation_data"
    / "fast_eval"
    / "ewok_fast"
)
DEFAULT_OUTPUT_DIR = ROOT / "configs" / "analysis" / "targets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ewok-dir",
        default=str(DEFAULT_EWOK_DIR),
        help="Directory containing EWoK fast-eval JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory for default output files. Ignored for any output path "
            "passed explicitly below."
        ),
    )
    parser.add_argument(
        "--lexical-targets-yaml",
        help=(
            "Output lexical target YAML. Defaults to "
            "<output-dir>/ewok_fast_lexical_targets.yaml."
        ),
    )
    parser.add_argument(
        "--domain-concepts-yaml",
        help=(
            "Output domain concept YAML for plotting. Defaults to "
            "<output-dir>/ewok_fast_domain_concepts.yaml."
        ),
    )
    parser.add_argument(
        "--unsupported-csv",
        help=(
            "Output audit CSV for unsupported concepts. Defaults to "
            "<output-dir>/ewok_fast_unsupported_targets.csv."
        ),
    )
    parser.add_argument(
        "--source-name",
        default="ewok_fast",
        help="Source label stored in generated lexical target metadata.",
    )
    return parser.parse_args()


def resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir)
    lexical_targets_yaml = (
        Path(args.lexical_targets_yaml)
        if args.lexical_targets_yaml
        else output_dir / "ewok_fast_lexical_targets.yaml"
    )
    domain_concepts_yaml = (
        Path(args.domain_concepts_yaml)
        if args.domain_concepts_yaml
        else output_dir / "ewok_fast_domain_concepts.yaml"
    )
    unsupported_csv = (
        Path(args.unsupported_csv)
        if args.unsupported_csv
        else output_dir / "ewok_fast_unsupported_targets.csv"
    )
    return lexical_targets_yaml, domain_concepts_yaml, unsupported_csv


def main() -> None:
    args = parse_args()
    lexical_targets_yaml, domain_concepts_yaml, unsupported_csv = resolve_outputs(args)
    result = export_ewok_targets(
        ewok_dir=args.ewok_dir,
        lexical_targets_yaml=lexical_targets_yaml,
        domain_concepts_yaml=domain_concepts_yaml,
        unsupported_csv=unsupported_csv,
        source_name=args.source_name,
    )

    print(f"Concepts read: {result['concepts_total']}")
    print(f"Lexical targets: {result['lexical_targets']}")
    print(f"Unsupported concepts: {result['unsupported_targets']}")
    print(f"Domains: {result['domains']}")
    print(f"Lexical targets YAML: {result['lexical_targets_yaml']}")
    print(f"Domain concepts YAML: {result['domain_concepts_yaml']}")
    print(f"Unsupported CSV: {result['unsupported_csv']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
