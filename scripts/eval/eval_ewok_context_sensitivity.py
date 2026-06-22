#!/usr/bin/env python3
"""Run moonshot-style EWoK LM context-sensitivity diagnostics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.evaluation.babylm_fast_eval import discover_checkpoints  # noqa: E402
from babysteps.evaluation.ewok_context_sensitivity import (  # noqa: E402
    append_item_records_jsonl,
    append_summary_csv,
    evaluate_ewok_context_sensitivity,
    read_ewok_items,
)
from babysteps.paths import resolve_path  # noqa: E402


DEFAULT_RUNS_DIR = Path(os.environ.get("BABYSTEPS_RUNS_DIR", "~/babylm-baby-steps-runs")).expanduser()
DEFAULT_FULL_EWOK_DIR = (
    ROOT
    / "external"
    / "babylm-eval"
    / "strict"
    / "evaluation_data"
    / "full_eval"
    / "ewok_filtered"
)
DEFAULT_EWOK_DIR = DEFAULT_FULL_EWOK_DIR
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "evaluation" / "ewok_context_sensitivity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="Training run directory containing HF checkpoints.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        dest="checkpoints",
        help="Explicit HF checkpoint directory. Can be passed more than once.",
    )
    parser.add_argument(
        "--ewok-data-dir",
        default=str(DEFAULT_EWOK_DIR),
        help="Directory or JSONL file containing EWoK rows.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where ewok_context_items.jsonl and ewok_context_summary.csv are written.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument(
        "--score-reduction",
        choices=("sum", "mean"),
        default="sum",
        help="How to reduce target-token log probabilities before k1/k2 comparisons.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for model evaluation.",
    )
    parser.add_argument("--max-checkpoints", type=int, help="Evaluate only the first N checkpoints.")
    parser.add_argument("--max-items", type=int, help="Evaluate only the first N EWoK items.")
    parser.add_argument("--append", action="store_true", help="Append to existing output files.")
    return parser.parse_args()


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    explicit = [resolve_path(path, base_dir=ROOT) for path in (args.checkpoints or [])]
    if explicit:
        checkpoints = explicit
    elif args.run_dir:
        checkpoints = discover_checkpoints(resolve_path(args.run_dir, base_dir=ROOT))
    else:
        raise SystemExit("Provide --run-dir or at least one --checkpoint.")

    if args.max_checkpoints is not None:
        checkpoints = checkpoints[: args.max_checkpoints]
    if not checkpoints:
        raise SystemExit("No checkpoints found.")
    return checkpoints


def load_model_and_tokenizer(checkpoint: Path, device: torch.device) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir, base_dir=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    item_records_path = output_dir / "ewok_context_items.jsonl"
    summary_path = output_dir / "ewok_context_summary.csv"
    if not args.append:
        for path in (item_records_path, summary_path):
            if path.exists():
                path.unlink()

    checkpoints = resolve_checkpoints(args)
    ewok_items = read_ewok_items(resolve_path(args.ewok_data_dir, base_dir=ROOT))
    if args.max_items is not None:
        ewok_items = ewok_items[: args.max_items]
    device = torch.device(args.device)

    print(f"EWoK items: {len(ewok_items):,}")
    print(f"Output: {output_dir}")
    print(f"Reduction: {args.score_reduction}")
    print("Checkpoints:")
    for checkpoint in checkpoints:
        print(f"  - {checkpoint}")

    for checkpoint in checkpoints:
        print(f"[eval] {checkpoint.name}", flush=True)
        model, tokenizer = load_model_and_tokenizer(checkpoint, device)
        result = evaluate_ewok_context_sensitivity(
            model=model,
            tokenizer=tokenizer,
            items=ewok_items,
            batch_size=args.batch_size,
            score_reduction=args.score_reduction,
            device=device,
        )
        metadata = {
            "checkpoint": checkpoint.name,
            "checkpoint_path": str(checkpoint),
        }
        append_item_records_jsonl(result.item_records, item_records_path, metadata=metadata)
        append_summary_csv(result.summary_rows, summary_path, metadata=metadata)

        overall = next(
            row
            for row in result.summary_rows
            if row["group_scope"] == "overall" and row["group_name"] == "all"
        )
        print(
            f"       score={overall['score']:.4f} "
            f"mean_k_margin={overall['mean_k_margin']:.4f} "
            f"sum_mean_sign_disagreements={overall['sum_mean_sign_disagreements']}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Wrote: {item_records_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
