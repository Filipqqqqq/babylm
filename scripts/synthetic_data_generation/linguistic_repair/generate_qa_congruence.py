#!/usr/bin/env python3
"""Generate QA-congruence linguistic repair documents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.linguistic_repair import (  # noqa: E402
    DeterministicQaMockGenerator,
    build_qa_congruence_repair,
    load_yaml_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a QA repair YAML config.")
    parser.add_argument("--output-dir", help="Override output_dir.")
    parser.add_argument("--document-count", type=int, help="Override document_count.")
    parser.add_argument("--model", help="Override renderer.model_id.")
    parser.add_argument("--api-key", help="Fireworks API key. Prefer FIREWORKS_API_KEY.")
    parser.add_argument("--api-key-file", help="Plain-text Fireworks API key file.")
    parser.add_argument(
        "--documents-per-request",
        type=int,
        help="Override renderer.documents_per_request.",
    )
    parser.add_argument(
        "--mock-fireworks",
        action="store_true",
        help="Use a deterministic local generator with the Fireworks renderer path.",
    )
    return parser.parse_args(argv)


def _apply_overrides(config: dict, args: argparse.Namespace) -> tuple[dict, bool]:
    config = dict(config)
    changed = False
    if args.output_dir:
        config["output_dir"] = args.output_dir
        changed = True
    if args.document_count is not None:
        config["document_count"] = args.document_count
        changed = True

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
    if args.documents_per_request is not None:
        renderer["documents_per_request"] = args.documents_per_request
        changed = True
    if renderer:
        config["renderer"] = renderer
    return config, changed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config, changed = _apply_overrides(load_yaml_config(config_path), args)
    result = build_qa_congruence_repair(
        config,
        base_dir=ROOT,
        config_path=None if changed else config_path,
        generator=DeterministicQaMockGenerator() if args.mock_fireworks else None,
    )
    counts = result.manifest["counts"]
    print(f"QA repair: {result.name}")
    print(f"Scenario plans: {counts['scenario_plans']:,}")
    print(f"Rendered documents: {counts['rendered_documents']:,}")
    print(f"Train documents: {counts['train_documents']:,}")
    print(f"Audit counts: pass={counts['pass']:,} warn={counts['warn']:,} fail={counts['fail']:,}")
    print(f"Train text: {result.outputs['train']}")
    print(f"Documents JSONL: {result.outputs['documents']}")
    print(f"Manifest: {result.outputs['manifest']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
