#!/usr/bin/env python3
"""Evaluate a Hugging Face causal LM on Entity Tracking likelihood."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.evaluation.entity_tracking_likelihood import (  # noqa: E402
    SUMMARY_FIELDNAMES,
    aggregate_item_records,
    append_summary_csv,
    build_item_records,
    make_continuation_requests,
    read_entity_tracking_items,
    score_continuations,
)
from babysteps.paths import resolve_path  # noqa: E402


def default_data_dir() -> Path:
    eval_root = ROOT / "external" / "babylm-eval" / "strict" / "evaluation_data"
    return eval_root / "full_eval" / "entity_tracking"


def parse_reductions(raw_value: str) -> tuple[str, ...]:
    value = raw_value.strip().lower()
    if value == "both":
        return ("sum", "mean")
    if value in {"sum", "mean"}:
        return (value,)
    raise ValueError("--score-reduction must be sum, mean, or both.")


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_dtype(value: str) -> Any:
    if value == "auto":
        return "auto"
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {value}")


def overall_row(rows: list[dict[str, Any]], scope: str = "overall") -> dict[str, Any]:
    return next(row for row in rows if row["group_scope"] == scope and row["group_name"] == "all")


def write_items_jsonl(path: Path, records: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps({**metadata, **record}, sort_keys=True))
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model path or model id.")
    parser.add_argument(
        "--data-dir",
        default=str(default_data_dir()),
        help="Entity Tracking JSONL directory.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-items", type=int, help="Optional quick-test item cap.")
    parser.add_argument("--score-reduction", choices=("sum", "mean", "both"), default="both")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device. With CUDA_VISIBLE_DEVICES=1, use cuda.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--revision", help="Optional Hugging Face revision.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "runs" / "analysis" / "entity_tracking_hf"),
    )
    parser.add_argument(
        "--no-items",
        action="store_true",
        help="Only write the summary CSV, not per-item JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir, base_dir=ROOT)
    output_dir = resolve_path(args.output_dir, base_dir=ROOT) / slugify(args.model)
    summary_path = output_dir / "likelihood_summary.csv"
    items_path = output_dir / "likelihood_items.jsonl"

    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    model_kwargs: dict[str, Any] = {
        "dtype": load_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.revision:
        tokenizer_kwargs["revision"] = args.revision
        model_kwargs["revision"] = args.revision

    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()

    items = read_entity_tracking_items(data_dir)
    if args.max_items is not None:
        items = items[: args.max_items]

    reductions = parse_reductions(args.score_reduction)
    print(f"Model: {args.model}")
    print(f"Data: {data_dir}")
    print(f"Items: {len(items):,}")
    print(f"Reductions: {', '.join(reductions)}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    score_map = score_continuations(
        model=model,
        tokenizer=tokenizer,
        requests=make_continuation_requests(items),
        batch_size=args.batch_size,
        device=device,
    )

    metadata = {
        "run_name": args.model,
        "epoch": "",
        "step": "",
        "tokens_seen": "",
        "words_seen": "",
        "checkpoint": args.revision or "",
        "checkpoint_path": args.model,
    }
    if summary_path.exists():
        summary_path.unlink()
    if items_path.exists():
        items_path.unlink()

    print("\nEntity Tracking likelihood")
    print("reduction  macro_score  micro_score  correct/items  ties  mean_margin")
    for reduction in reductions:
        records = build_item_records(
            items=items,
            score_map=score_map,
            score_reduction=reduction,  # type: ignore[arg-type]
        )
        summary_rows = aggregate_item_records(records)
        append_summary_csv(summary_rows, summary_path, metadata=metadata)
        if not args.no_items:
            write_items_jsonl(items_path, records, metadata)

        overall = overall_row(summary_rows, "overall")
        overall_micro = overall_row(summary_rows, "overall_micro")
        print(
            f"{reduction:<9} "
            f"{overall['score']:.4f}       "
            f"{overall_micro['score']:.4f}       "
            f"{overall_micro['correct']}/{overall_micro['items']}       "
            f"{overall_micro['ties']}     "
            f"{overall['mean_margin']:.4f}"
        )

    print(f"\nWrote summary: {summary_path}")
    if not args.no_items:
        print(f"Wrote items: {items_path}")
    print(f"Summary columns: {', '.join(SUMMARY_FIELDNAMES)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
