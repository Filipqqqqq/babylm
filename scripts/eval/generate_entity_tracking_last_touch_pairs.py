#!/usr/bin/env python3
"""Generate controlled paired Entity Tracking last-touch probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.evaluation.entity_tracking_counterfactuals import (  # noqa: E402
    build_manifest,
    generate_counterfactual_items,
    write_counterfactual_jsonl,
)


DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "analysis"
    / "entity_tracking_last_touch"
    / "counterfactual_pairs_seed42.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-per-family", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = generate_counterfactual_items(
        pairs_per_family=args.pairs_per_family,
        seed=args.seed,
    )
    output = write_counterfactual_jsonl(args.output, items)
    manifest = build_manifest(
        path=output,
        items=items,
        seed=args.seed,
        pairs_per_family=args.pairs_per_family,
    )
    manifest_path = output.with_name("manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {manifest['pair_count']:,} pairs / {manifest['item_count']:,} items")
    print(f"Probe data: {output}")
    print(f"Manifest:   {manifest_path}")
    print(f"SHA-256:    {manifest['sha256']}")


if __name__ == "__main__":
    main()
