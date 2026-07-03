#!/usr/bin/env python3
"""Generate deterministic QA who-person repair documents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.linguistic_repair import (  # noqa: E402
    build_qa_who_person_frames,
    load_yaml_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a QA who-person YAML config.")
    parser.add_argument("--output-dir", help="Override output_dir.")
    parser.add_argument("--document-count", type=int, help="Override document_count.")
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
    return config, changed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config, changed = _apply_overrides(load_yaml_config(config_path), args)
    result = build_qa_who_person_frames(
        config,
        base_dir=ROOT,
        config_path=None if changed else config_path,
    )
    counts = result.manifest["counts"]
    print(f"QA who-person frames: {result.name}")
    print(f"Documents: {counts['documents']:,}")
    print(f"Train documents: {counts['train_documents']:,}")
    print(f"Audit counts: pass={counts['pass']:,} warn={counts['warn']:,} fail={counts['fail']:,}")
    print(f"Frame family counts: {counts['frame_family_counts']}")
    print(f"Style counts: {counts['style_counts']}")
    print(f"Train text: {result.outputs['train']}")
    print(f"Documents JSONL: {result.outputs['documents']}")
    print(f"Manifest: {result.outputs['manifest']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
