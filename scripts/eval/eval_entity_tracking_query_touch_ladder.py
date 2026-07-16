#!/usr/bin/env python3
"""Evaluate an MLM on the matched Entity Tracking query-touch ladder."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.evaluation.entity_tracking_counterfactuals import file_sha256  # noqa: E402
from babysteps.evaluation.entity_tracking_touch_ladder import (  # noqa: E402
    build_scored_records,
    read_touch_ladder_jsonl,
    summarize_scored_records,
)
from babysteps.evaluation.mlm_likelihood import (  # noqa: E402
    MlmTextSpanRequest,
    score_mlm_text_spans,
)
from babysteps.paths import resolve_path  # noqa: E402


DEFAULT_DATA = (
    ROOT
    / "data"
    / "analysis"
    / "entity_tracking_query_touch_ladder"
    / "touch_ladder_seed42.jsonl"
)
DEFAULT_RUNS_DIR = Path(
    os.environ.get("BABYSTEPS_RUNS_DIR", "~/babylm-baby-steps-runs")
).expanduser()
DEFAULT_OUTPUT = DEFAULT_RUNS_DIR / "analysis" / "entity_tracking_query_touch_ladder"


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--item-chunk-size", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--bootstrap-seed", type=int, default=2718)
    parser.add_argument("--max-worlds", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_dtype(value: str) -> torch.dtype | None:
    if value == "auto":
        return None
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def cap_worlds(items: list[Any], max_worlds: int | None) -> list[Any]:
    if max_worlds is None:
        return items
    if max_worlds <= 0:
        raise ValueError("--max-worlds must be positive.")
    allowed = []
    seen = set()
    for item in items:
        if item.base_id not in seen and len(seen) >= max_worlds:
            continue
        seen.add(item.base_id)
        allowed.append(item)
    return allowed


def score_items(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[Any],
    batch_size: int,
    item_chunk_size: int,
    device: str,
    temperature: float,
) -> tuple[list[list[float]], list[list[int]]]:
    if batch_size <= 0 or item_chunk_size <= 0:
        raise ValueError("Batch and chunk sizes must be positive.")
    option_scores = []
    token_counts = []
    starts = range(0, len(items), item_chunk_size)
    total = (len(items) + item_chunk_size - 1) // item_chunk_size
    for start in tqdm(starts, total=total):
        chunk = items[start : start + item_chunk_size]
        requests = [
            MlmTextSpanRequest(
                text=item.input_prefix + option,
                target_start_char=len(item.input_prefix),
            )
            for item in chunk
            for option in item.options
        ]
        scores = score_mlm_text_spans(
            model=model,
            tokenizer=tokenizer,
            requests=requests,
            batch_size=batch_size,
            device=device,
            temperature=temperature,
        )
        cursor = 0
        for item in chunk:
            item_scores = scores[cursor : cursor + len(item.options)]
            cursor += len(item.options)
            option_scores.append([score.sum_logprob for score in item_scores])
            token_counts.append([score.scored_token_count for score in item_scores])
        if cursor != len(scores):
            raise RuntimeError("Internal score alignment failure.")
    return option_scores, token_counts


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = ["row_type", "name", "items", "accuracy", "mean_margin", "estimate", "ci95_low", "ci95_high"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in summary["condition_order"]:
            values = summary["conditions"][condition]
            writer.writerow(
                {
                    "row_type": "condition",
                    "name": condition,
                    "items": values["items"],
                    "accuracy": values["accuracy"]["estimate"],
                    "mean_margin": values["mean_margin"],
                }
            )
        for name, values in summary["contrasts"].items():
            writer.writerow(
                {
                    "row_type": "margin_contrast",
                    "name": name,
                    "items": values["pairs"],
                    "estimate": values["mean_margin_difference"],
                    "ci95_low": values["mean_margin_difference_ci95"][0],
                    "ci95_high": values["mean_margin_difference_ci95"][1],
                }
            )


def print_summary(summary: dict[str, Any]) -> None:
    print("\nEntity Tracking query-touch ladder")
    print("condition  operations/query-touches  accuracy  mean margin")
    labels = {
        "q1": "1/1",
        "q3": "3/3",
        "q5": "5/5",
        "t3": "3/3",
        "t5": "5/5",
        "c3": "3/1",
        "c5": "5/1",
    }
    for condition in summary["condition_order"]:
        values = summary["conditions"][condition]
        print(
            f"{condition:9s} {labels[condition]:>10s}              "
            f"{100 * values['accuracy']['estimate']:7.2f}%  "
            f"{values['mean_margin']:+.4f}"
        )
    contrast_order = ["q3_minus_c3", "q5_minus_c5"]
    if "t3" in summary["conditions"]:
        contrast_order = [
            "t3_minus_c3",
            "t5_minus_c5",
            "q3_minus_t3",
            "q5_minus_t5",
        ]
    print("\nPrimary matched contrasts")
    for name in contrast_order:
        values = summary["contrasts"][name]
        low, high = values["mean_margin_difference_ci95"]
        print(
            f"{name:14s}: margin={values['mean_margin_difference']:+.4f} "
            f"95% CI [{low:+.4f}, {high:+.4f}], "
            f"accuracy={100 * values['accuracy_difference']:+.2f} pp"
        )


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data, base_dir=ROOT)
    items = cap_worlds(read_touch_ladder_jsonl(data_path), args.max_worlds)
    print(f"Model:  {args.model}")
    print(f"Data:   {data_path}")
    world_count = len({item.base_id for item in items})
    print(f"Worlds: {world_count:,}; items: {len(items):,}")
    print(f"Device: {args.device}; temperature: {args.temperature}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    dtype = load_dtype(args.dtype)
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    model = AutoModelForMaskedLM.from_pretrained(args.model, **model_kwargs)
    model = model.to(torch.device(args.device))
    model.eval()

    option_scores, token_counts = score_items(
        model=model,
        tokenizer=tokenizer,
        items=items,
        batch_size=args.batch_size,
        item_chunk_size=args.item_chunk_size,
        device=args.device,
        temperature=args.temperature,
    )
    records = build_scored_records(
        items=items,
        option_scores=option_scores,
        scored_token_counts=token_counts,
    )
    summary = summarize_scored_records(records, bootstrap_seed=args.bootstrap_seed)
    summary["evaluation"] = {
        "model": args.model,
        "data": str(data_path),
        "data_sha256": file_sha256(data_path),
        "device": args.device,
        "dtype": args.dtype,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "item_chunk_size": args.item_chunk_size,
        "scoring": "official_babylm_mlm_mask_each_completion_token_sum_logprob",
    }

    output_dir = resolve_path(args.output_dir, base_dir=ROOT) / slugify(Path(args.model).name)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "item_scores.jsonl"
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    write_jsonl(records_path, records)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_csv(csv_path, summary)
    print_summary(summary)
    print(f"\nItem scores: {records_path}")
    print(f"Summary:     {summary_path}")
    print(f"CSV:         {csv_path}")


if __name__ == "__main__":
    main()
