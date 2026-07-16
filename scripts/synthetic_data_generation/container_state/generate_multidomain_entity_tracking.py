#!/usr/bin/env python3
"""Generate the audited multi-domain entity-tracking v3 corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.container_state import build_multidomain_corpus, load_yaml_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/container_state/container_state_multidomain_v3_seed42.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--train-pairs", type=int)
    parser.add_argument("--eval-pairs-per-split", type=int)
    parser.add_argument(
        "--no-enforce-surface-gate",
        action="store_true",
        help="Record the surface-classifier result without failing generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml_config(config_path)
    changed = False
    for argument, key in (
        (args.output_dir, "output_dir"),
        (args.train_pairs, "train_pairs"),
        (args.eval_pairs_per_split, "eval_pairs_per_split"),
    ):
        if argument is not None:
            config[key] = argument
            changed = True
    if args.no_enforce_surface_gate:
        config["enforce_surface_gate"] = False
        changed = True
    result = build_multidomain_corpus(
        config,
        base_dir=ROOT,
        config_path=None if changed else config_path,
    )
    counts = result.audit_report["counts"]
    classifier = result.audit_report["surface_classifier"]
    print(f"Corpus: {result.name}")
    print(
        f"Train: {counts['train_pairs']:,} pairs / "
        f"{counts['train_documents']:,} documents / {counts['train_words']:,} words"
    )
    for split, count in counts["eval_pairs_per_split"].items():
        print(f"{split}: {count:,} pairs / {2 * count:,} five-choice items")
    print(
        "Surface branch classifier: "
        f"AUC={classifier['roc_auc']:.4f} "
        f"(limit={classifier['max_allowed_auc']:.4f}, passed={classifier['passed']})"
    )
    print(f"Audit: {result.outputs['audit_report']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
