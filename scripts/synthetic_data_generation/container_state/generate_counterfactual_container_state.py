#!/usr/bin/env python3
"""Generate paired counterfactual container-state data for the pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.container_state import (  # noqa: E402
    build_counterfactual_corpus,
    load_yaml_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--pool-min-words", type=int)
    parser.add_argument("--eval-pairs", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml_config(config_path)
    changed = False
    for arg_name, config_name in (
        ("output_dir", "output_dir"),
        ("pool_min_words", "pool_min_words"),
        ("eval_pairs", "eval_pairs"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            config[config_name] = value
            changed = True
    result = build_counterfactual_corpus(
        config,
        base_dir=ROOT,
        config_path=None if changed else config_path,
    )
    counts = result.audit_report["counts"]
    print(f"Counterfactual corpus: {result.name}")
    print(f"Training pairs/documents: {counts['train_pairs']:,}/{counts['train_documents']:,}")
    print(f"Training words: {counts['train_words']:,}")
    print(f"Held-out pairs: {counts['eval_pairs']:,}")
    print(f"Operation counts: {result.audit_report['operation_count_counts']}")
    print(f"Counterfactual kinds: {result.audit_report['counterfactual_kind_counts']}")
    print(f"Training text: {result.outputs['train']}")
    print(f"Auxiliary data: {result.outputs['train_counterfactual_rank'].parent}")
    print(f"Audit report: {result.outputs['audit_report']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
