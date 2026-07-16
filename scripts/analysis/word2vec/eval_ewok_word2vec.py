#!/usr/bin/env python3
"""Evaluate EWoK context sensitivity with static Word2Vec vectors."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.analysis.word2vec.corpus import tokenization_config_from_mapping
from babysteps.analysis.word2vec.ewok_eval import evaluate_ewok_items, load_ewok_items
from babysteps.analysis.word2vec.reports import write_eval_report
from babysteps.analysis.word2vec.training import load_word2vec_config
from babysteps.analysis.word2vec.vectors import load_keyed_vectors
from babysteps.paths import resolve_path


DEFAULT_CONFIG = ROOT / "configs" / "analysis" / "word2vec" / "eval_google_news_ewok.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Word2Vec EWoK eval YAML config.")
    parser.add_argument("--vectors-path", help="Override vectors.path from the config.")
    parser.add_argument("--ewok-path", help="Override ewok.path from the config.")
    parser.add_argument("--output-dir", help="Override output.dir from the config.")
    parser.add_argument("--max-items", type=int, help="Optional item cap for smoke tests.")
    parser.add_argument("--no-plots", action="store_true", help="Write CSVs only.")
    return parser.parse_args()


def _resolve_output_dir(config: dict[str, Any], output_dir: str | None) -> Path:
    if output_dir:
        return resolve_path(output_dir, base_dir=Path.cwd())
    output_config = config.get("output", {})
    base = resolve_path(
        output_config.get(
            "dir",
            "${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/analysis/word2vec/ewok_eval",
        ),
        base_dir=ROOT,
    )
    if not bool(output_config.get("timestamp", True)):
        return base
    return base / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main() -> None:
    args = parse_args()
    config = load_word2vec_config(args.config)
    if args.vectors_path:
        config.setdefault("vectors", {})["path"] = args.vectors_path
    if args.ewok_path:
        config.setdefault("ewok", {})["path"] = args.ewok_path

    vectors = load_keyed_vectors(config["vectors"], root_dir=ROOT)
    ewok_path = config.get("ewok", {}).get(
        "path",
        "external/babylm-eval/strict/evaluation_data/fast_eval/ewok_fast",
    )
    items = load_ewok_items(ewok_path, root_dir=ROOT)
    if args.max_items is not None:
        items = items[: args.max_items]

    tokenization = tokenization_config_from_mapping(config.get("preprocessing", {}))
    records = evaluate_ewok_items(items, vectors, tokenization=tokenization)
    output_dir = _resolve_output_dir(config, args.output_dir)
    paths = write_eval_report(
        records,
        output_dir,
        manifest={
            "name": config.get("name", "word2vec_ewok_eval"),
            "config": config,
            "ewok_path": ewok_path,
            "vectors": config.get("vectors", {}),
            "preprocessing": config.get("preprocessing", {}),
        },
        make_plots=not args.no_plots,
    )

    counted = sum(1 for record in records if record.is_counted)
    print(f"Items: {len(records):,}")
    print(f"Counted: {counted:,}")
    print(f"Output: {output_dir}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
