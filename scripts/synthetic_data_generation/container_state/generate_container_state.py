#!/usr/bin/env python3
"""Generate the deterministic naturalized container-state curriculum."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.container_state import (  # noqa: E402
    build_container_state_corpus,
    load_yaml_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Container-state YAML config.")
    parser.add_argument("--output-dir", help="Override the configured output directory.")
    parser.add_argument(
        "--pool-min-words",
        type=int,
        help="Override the minimum generated training-pool word count.",
    )
    parser.add_argument(
        "--validation-items-per-slice",
        type=int,
        help="Override the number of examples in each custom validation slice.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml_config(config_path)
    changed = False
    if args.output_dir:
        config["output_dir"] = args.output_dir
        changed = True
    if args.pool_min_words is not None:
        config["pool_min_words"] = args.pool_min_words
        changed = True
    if args.validation_items_per_slice is not None:
        validation = dict(config.get("validation") or {})
        validation["items_per_slice"] = args.validation_items_per_slice
        config["validation"] = validation
        changed = True

    result = build_container_state_corpus(
        config,
        base_dir=ROOT,
        config_path=None if changed else config_path,
    )
    counts = result.audit_report["counts"]
    print(f"Container-state corpus: {result.name}")
    print(f"Training documents: {counts['documents']:,}")
    print(f"Training words: {counts['words']:,}")
    print(f"Validation items: {counts['validation_items']:,}")
    print(f"Operation counts: {result.audit_report['operation_count_counts']}")
    print(f"Operation kinds: {result.audit_report['operation_kind_counts']}")
    print(f"Renderers: {result.audit_report['renderer_counts']}")
    print(f"Training text: {result.outputs['train']}")
    print(f"Validation data: {result.outputs['validation_dir']}")
    print(f"Audit report: {result.outputs['audit_report']}")
    print(f"Manifest: {result.outputs['manifest']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
