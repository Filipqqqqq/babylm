#!/usr/bin/env python3
"""Build BabyLM-style positive augmentations from concept-curriculum artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.concept_curriculum.augmentation import (  # noqa: E402
    DeterministicMotChiMockGenerator,
    build_document_augmentations,
    load_yaml_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an augmentation YAML config.",
    )
    parser.add_argument("--output-dir", help="Override the config output_dir.")
    parser.add_argument("--model", help="Override renderer.model_id.")
    parser.add_argument("--api-key", help="Fireworks API key. Prefer FIREWORKS_API_KEY or --api-key-file.")
    parser.add_argument("--api-key-file", help="Plain-text file containing the Fireworks API key.")
    parser.add_argument(
        "--max-source-documents",
        type=int,
        help="Pilot override for selection.max_source_documents.",
    )
    parser.add_argument(
        "--max-augmented-documents",
        type=int,
        help="Pilot override for selection.max_augmented_documents.",
    )
    parser.add_argument(
        "--repair-failed-documents",
        action="store_true",
        help="Run a repair pass over failed audited documents.",
    )
    parser.add_argument(
        "--max-repair-rounds",
        type=int,
        help="Maximum repair rounds when repair is enabled.",
    )
    parser.add_argument(
        "--mock-fireworks",
        action="store_true",
        help="Use a deterministic local generator with the Fireworks/Qwen renderer path.",
    )
    return parser.parse_args()


def _apply_overrides(config: dict, args: argparse.Namespace) -> tuple[dict, bool]:
    config = dict(config)
    changed = False
    if args.output_dir:
        config["output_dir"] = args.output_dir
        changed = True

    selection = dict(config.get("selection") or {})
    if args.max_source_documents is not None:
        selection["max_source_documents"] = args.max_source_documents
        changed = True
    if args.max_augmented_documents is not None:
        selection["max_augmented_documents"] = args.max_augmented_documents
        changed = True
    if selection:
        config["selection"] = selection

    renderer = dict(config.get("renderer") or {})
    if args.model:
        renderer["model_id"] = args.model
        changed = True
    if args.api_key:
        renderer["api_key"] = args.api_key
        changed = True
    if args.api_key_file:
        renderer["api_key_file"] = args.api_key_file
        changed = True
    if renderer:
        config["renderer"] = renderer

    repair = dict(config.get("repair") or {})
    if args.repair_failed_documents:
        repair["enabled"] = True
        changed = True
    if args.max_repair_rounds is not None:
        repair["max_rounds"] = args.max_repair_rounds
        changed = True
    if repair:
        config["repair"] = repair
    return config, changed


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config, changed = _apply_overrides(load_yaml_config(config_path), args)
    result = build_document_augmentations(
        config,
        base_dir=ROOT,
        config_path=None if changed else config_path,
        generator=DeterministicMotChiMockGenerator() if args.mock_fireworks else None,
    )
    counts = result.manifest["counts"]
    print(f"Augmentation: {result.manifest['name']}")
    print(f"Selected source scenarios: {counts['selected_source_scenarios']:,}")
    print(f"Augmented documents: {counts['augmented_documents']:,}")
    print(f"Train documents: {counts['train_documents']:,}")
    print(f"Audit counts: pass={counts['pass']:,} warn={counts['warn']:,} fail={counts['fail']:,}")
    repair_counts = result.audit_report.get("repair_counts") or {}
    if repair_counts.get("attempted"):
        print(
            "Repair counts: "
            f"attempted={repair_counts.get('attempted', 0):,} "
            f"pass={repair_counts.get('pass', 0):,} "
            f"warn={repair_counts.get('warn', 0):,} "
            f"fail={repair_counts.get('fail', 0):,}"
        )
    print(f"Train text: {result.outputs['train']}")
    print(f"Augmented JSONL: {result.outputs['augmented_documents']}")
    print(f"Manifest: {result.outputs['manifest']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
