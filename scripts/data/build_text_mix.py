#!/usr/bin/env python3
"""Build a reproducible capped raw-text corpus mix from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.data.text_mix import build_text_mix, load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a text-mix YAML config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml_config(config_path)
    result = build_text_mix(
        config,
        base_dir=ROOT,
        config_path=config_path,
    )

    counts = result.counts
    print(f"Text mix: {counts['name']}")
    print(f"Seed: {counts['seed']}")
    print(f"Output words: {counts['output_words_total']:,}")
    print(f"Output documents: {counts['output_documents_total']:,}")
    print(f"Train text: {result.outputs['train']}")
    print(f"Manifest: {result.outputs['mix_manifest']}")
    print(f"Counts: {result.outputs['counts']}")
    for source in counts["sources"]:
        print(
            f"- {source['name']}: "
            f"target={source['target_words']:,} "
            f"selected={source['selected_words']:,} "
            f"output={source['output_words']:,} "
            f"docs={source['output_documents']:,}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
