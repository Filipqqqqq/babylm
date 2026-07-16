#!/usr/bin/env python3
"""Evaluate an MLM on controlled Entity Tracking last-touch counterfactuals."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.evaluation.entity_tracking_counterfactuals import (  # noqa: E402
    PROBE_FAMILIES,
    build_scored_item_records,
    file_sha256,
    read_counterfactual_jsonl,
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
    / "entity_tracking_last_touch"
    / "counterfactual_pairs_seed42.jsonl"
)
DEFAULT_RUNS_DIR = Path(
    os.environ.get("BABYSTEPS_RUNS_DIR", "~/babylm-baby-steps-runs")
).expanduser()
DEFAULT_OUTPUT = DEFAULT_RUNS_DIR / "analysis" / "entity_tracking_last_touch"


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_dtype(value: str) -> torch.dtype | None:
    if value == "auto":
        return None
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model path or ID.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of masked-token variants per model batch.",
    )
    parser.add_argument(
        "--item-chunk-size",
        type=int,
        default=64,
        help="Items encoded at once; controls host-memory use.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument(
        "--max-pairs-per-family",
        type=int,
        help="Optional balanced smoke-test cap.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def balanced_pair_cap(items: list[Any], max_pairs: int | None) -> list[Any]:
    if max_pairs is None:
        return items
    if max_pairs <= 0:
        raise ValueError("--max-pairs-per-family must be positive.")
    allowed: dict[str, set[str]] = {family: set() for family in PROBE_FAMILIES}
    selected = []
    for item in items:
        family_pairs = allowed[item.family]
        if item.pair_id not in family_pairs and len(family_pairs) >= max_pairs:
            continue
        family_pairs.add(item.pair_id)
        selected.append(item)
    return selected


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
    if item_chunk_size <= 0:
        raise ValueError("--item-chunk-size must be positive.")
    option_scores: list[list[float]] = []
    token_counts: list[list[int]] = []

    starts = range(0, len(items), item_chunk_size)
    for start in tqdm(starts, total=(len(items) + item_chunk_size - 1) // item_chunk_size):
        chunk = items[start : start + item_chunk_size]
        requests = [
            MlmTextSpanRequest(
                text=f"{item.input_prefix}{option}",
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
            count = len(item.options)
            item_scores = scores[cursor : cursor + count]
            cursor += count
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
    fields = [
        "family",
        "pairs",
        "accuracy",
        "prediction_changed",
        "prediction_invariant",
        "both_correct",
        "initial_state_option_selected",
        "lexical_recency_expected_accuracy",
        "model_lexical_recency_agreement",
        "near_accuracy",
        "far_accuracy",
        "near_minus_far_margin_mean",
        "near_margin_better",
        "directional_target_shift_mean",
        "directional_target_shift_positive",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for family in PROBE_FAMILIES:
            values = summary["families"][family]
            writer.writerow(
                {
                    "family": family,
                    "pairs": values["pairs"],
                    "accuracy": values["accuracy"]["estimate"],
                    "prediction_changed": values["prediction_changed"]["estimate"],
                    "prediction_invariant": values["prediction_invariant"]["estimate"],
                    "both_correct": values["both_correct"]["estimate"],
                    "initial_state_option_selected": values.get(
                        "initial_state_option_selected", {}
                    ).get("estimate"),
                    "lexical_recency_expected_accuracy": values[
                        "lexical_recency_expected_accuracy"
                    ],
                    "model_lexical_recency_agreement": values[
                        "model_lexical_recency_agreement"
                    ],
                    "near_accuracy": values.get("near_accuracy", {}).get("estimate"),
                    "far_accuracy": values.get("far_accuracy", {}).get("estimate"),
                    "near_minus_far_margin_mean": values.get(
                        "near_minus_far_margin_mean"
                    ),
                    "near_margin_better": values.get("near_margin_better", {}).get(
                        "estimate"
                    ),
                    "directional_target_shift_mean": values.get(
                        "directional_target_shift_mean"
                    ),
                    "directional_target_shift_positive": values.get(
                        "directional_target_shift_positive", {}
                    ).get("estimate"),
                }
            )


def print_summary(summary: dict[str, Any]) -> None:
    print("\nPaired Entity Tracking last-touch diagnostic")
    print("family                 pairs  accuracy  changed  invariant  both-correct")
    for family in PROBE_FAMILIES:
        values = summary["families"][family]
        print(
            f"{family:23s} {values['pairs']:5d}  "
            f"{100 * values['accuracy']['estimate']:7.2f}%  "
            f"{100 * values['prediction_changed']['estimate']:7.2f}%  "
            f"{100 * values['prediction_invariant']['estimate']:8.2f}%  "
            f"{100 * values['both_correct']['estimate']:11.2f}%"
        )
        print(
            "  lexical heuristic accuracy/agreement: "
            f"{100 * values['lexical_recency_expected_accuracy']:.2f}% / "
            f"{100 * values['model_lexical_recency_agreement']:.2f}%"
        )
        if "initial_state_option_selected" in values:
            print(
                "  initial-state option selected: "
                f"{100 * values['initial_state_option_selected']['estimate']:.2f}%"
            )
        if family == "irrelevant_reordering":
            print(
                "  near/far accuracy: "
                f"{100 * values['near_accuracy']['estimate']:.2f}% / "
                f"{100 * values['far_accuracy']['estimate']:.2f}%; "
                f"near-far margin={values['near_minus_far_margin_mean']:+.4f}"
            )
        else:
            print(
                "  condition-sensitive target shift: "
                f"{100 * values['directional_target_shift_positive']['estimate']:.2f}%; "
                f"mean shift={values['directional_target_shift_mean']:+.4f}"
            )


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data, base_dir=ROOT)
    items = balanced_pair_cap(
        read_counterfactual_jsonl(data_path), args.max_pairs_per_family
    )
    family_counts = Counter(item.family for item in items)
    print(f"Model: {args.model}")
    print(f"Data:  {data_path}")
    print(f"Items: {len(items):,} ({dict(family_counts)})")
    print(f"Device: {args.device}; temperature: {args.temperature}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
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
    records = build_scored_item_records(
        items=items,
        option_scores=option_scores,
        scored_token_counts=token_counts,
    )
    summary = summarize_scored_records(
        records,
        bootstrap_seed=args.bootstrap_seed,
    )
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

    output_dir = resolve_path(args.output_dir, base_dir=ROOT) / slugify(
        Path(args.model).name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "item_scores.jsonl"
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    write_jsonl(records_path, records)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary_csv(csv_path, summary)

    print_summary(summary)
    print(f"\nItem scores: {records_path}")
    print(f"Summary:     {summary_path}")
    print(f"CSV:         {csv_path}")


if __name__ == "__main__":
    main()
