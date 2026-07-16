#!/usr/bin/env python3
"""Generate the matched Entity Tracking query-touch-count ladder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.evaluation.entity_tracking_touch_ladder import (  # noqa: E402
    build_manifest,
    generate_touch_ladder_items,
    write_touch_ladder_jsonl,
)


DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "analysis"
    / "entity_tracking_query_touch_ladder"
    / "touch_ladder_seed42.jsonl"
)
DEFAULT_EXTENDED_OUTPUT = (
    ROOT
    / "data"
    / "analysis"
    / "entity_tracking_query_touch_mechanism"
    / "touch_mechanism_seed42.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-transient",
        action="store_true",
        help=(
            "Add T3/T5 controls in which non-answer objects temporarily enter "
            "and leave the query box."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = generate_touch_ladder_items(
        worlds=args.worlds,
        seed=args.seed,
        include_transient=args.include_transient,
    )
    default_output = DEFAULT_EXTENDED_OUTPUT if args.include_transient else DEFAULT_OUTPUT
    output = write_touch_ladder_jsonl(args.output or default_output, items)
    manifest = build_manifest(
        path=output,
        items=items,
        seed=args.seed,
        worlds=args.worlds,
    )
    manifest_path = output.with_name("manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest['world_count']:,} worlds / {manifest['item_count']:,} items")
    print(f"Probe data: {output}")
    print(f"Manifest:   {manifest_path}")
    print(f"SHA-256:    {manifest['sha256']}")


if __name__ == "__main__":
    main()
