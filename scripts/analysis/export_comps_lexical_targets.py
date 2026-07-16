#!/usr/bin/env python3
"""Export COMPS concepts and property phrases into lexical target files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.analysis.lexical.comps_targets import export_comps_targets


DEFAULT_COMPS_DIR = (
    ROOT
    / "external"
    / "babylm-eval"
    / "strict"
    / "evaluation_data"
    / "full_eval"
    / "comps"
)
DEFAULT_OUTPUT_DIR = ROOT / "configs" / "analysis" / "targets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comps-dir",
        default=str(DEFAULT_COMPS_DIR),
        help="Directory containing COMPS full-eval JSONL files.",
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
        "--concept-targets-yaml",
        help=(
            "Output concept lexical target YAML. Defaults to "
            "<output-dir>/comps_concept_lexical_targets.yaml."
        ),
    )
    parser.add_argument(
        "--property-targets-yaml",
        help=(
            "Output property-phrase lexical target YAML. Defaults to "
            "<output-dir>/comps_property_phrase_lexical_targets.yaml."
        ),
    )
    parser.add_argument(
        "--unsupported-csv",
        help=(
            "Output audit CSV for unsupported targets. Defaults to "
            "<output-dir>/comps_unsupported_targets.csv."
        ),
    )
    parser.add_argument(
        "--source-name",
        default="comps",
        help="Source label stored in generated lexical target metadata.",
    )
    return parser.parse_args()


def resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir)
    concept_targets_yaml = (
        Path(args.concept_targets_yaml)
        if args.concept_targets_yaml
        else output_dir / "comps_concept_lexical_targets.yaml"
    )
    property_targets_yaml = (
        Path(args.property_targets_yaml)
        if args.property_targets_yaml
        else output_dir / "comps_property_phrase_lexical_targets.yaml"
    )
    unsupported_csv = (
        Path(args.unsupported_csv)
        if args.unsupported_csv
        else output_dir / "comps_unsupported_targets.csv"
    )
    return concept_targets_yaml, property_targets_yaml, unsupported_csv


def main() -> None:
    args = parse_args()
    concept_targets_yaml, property_targets_yaml, unsupported_csv = resolve_outputs(args)
    result = export_comps_targets(
        comps_dir=args.comps_dir,
        concept_targets_yaml=concept_targets_yaml,
        property_targets_yaml=property_targets_yaml,
        unsupported_csv=unsupported_csv,
        source_name=args.source_name,
    )

    print(f"Concepts read: {result['concepts_total']}")
    print(f"Concept lexical targets: {result['concept_targets']}")
    print(f"Properties read: {result['properties_total']}")
    print(f"Property phrase targets: {result['property_targets']}")
    print(f"Unsupported targets: {result['unsupported_targets']}")
    print(f"Concept targets YAML: {result['concept_targets_yaml']}")
    print(f"Property targets YAML: {result['property_targets_yaml']}")
    print(f"Unsupported CSV: {result['unsupported_csv']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
