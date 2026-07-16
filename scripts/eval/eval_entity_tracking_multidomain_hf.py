#!/usr/bin/env python3
"""Evaluate five-choice and paired 2x2 multi-domain entity tracking."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.container_state import EVAL_SPLITS  # noqa: E402
from babysteps.evaluation.entity_tracking_likelihood import (  # noqa: E402
    EntityTrackingItem,
    aggregate_item_records,
    append_item_records_jsonl,
    append_summary_csv,
    build_item_records,
    make_continuation_requests,
    read_entity_tracking_items,
    reduce_continuation_score,
    score_continuations,
)
from babysteps.paths import resolve_path  # noqa: E402


PAIR_FIELDS = (
    "split",
    "score_reduction",
    "pairs",
    "completion_correct",
    "completion_total",
    "completion_accuracy",
    "both_correct",
    "both_correct_rate",
    "symmetric_correct",
    "symmetric_accuracy",
    "mean_completion_margin",
    "mean_symmetric_margin",
)


def parse_reductions(value: str) -> tuple[str, ...]:
    if value == "both":
        return ("sum", "mean")
    return (value,)


def load_dtype(value: str) -> Any:
    return {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def read_pairs(path: Path, max_pairs_per_split: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in EVAL_SPLITS:
        split_rows = []
        with (path / f"{split}.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    split_rows.append(json.loads(line))
        if max_pairs_per_split is not None:
            split_rows = split_rows[:max_pairs_per_split]
        rows.extend(split_rows)
    return rows


def pair_items(rows: list[dict[str, Any]]) -> list[EntityTrackingItem]:
    items = []
    for row in rows:
        for side, negative in ((1, 2), (2, 1)):
            items.append(
                EntityTrackingItem(
                    item_id=f"{row['pair_id']}-side{side}",
                    source_file=f"{row['split']}.jsonl",
                    line_number=0,
                    family=str(row["split"]),
                    numops=int(row["operation_count"]),
                    uid=f"{row['split']}_{row['operation_count']}_ops",
                    input_prefix=str(row[f"Context{side}"]),
                    options=(str(row[f"Target{side}"]), str(row[f"Target{negative}"])),
                )
            )
    return items


def pair_records(
    rows: list[dict[str, Any]], score_map: dict[int, dict[int, Any]], reduction: str
) -> list[dict[str, Any]]:
    records = []
    for pair_index, row in enumerate(rows):
        margins = []
        for offset in (0, 1):
            item_scores = score_map[2 * pair_index + offset]
            correct = reduce_continuation_score(item_scores[0], reduction)  # type: ignore[arg-type]
            wrong = reduce_continuation_score(item_scores[1], reduction)  # type: ignore[arg-type]
            margins.append(correct - wrong)
        records.append(
            {
                "pair_id": row["pair_id"],
                "split": row["split"],
                "domain": row["domain"],
                "renderer": row["renderer"],
                "counterfactual_kind": row["counterfactual_kind"],
                "score_reduction": reduction,
                "margin1": margins[0],
                "margin2": margins[1],
                "completion1_correct": margins[0] > 0.0,
                "completion2_correct": margins[1] > 0.0,
                "both_correct": margins[0] > 0.0 and margins[1] > 0.0,
                "symmetric_margin": sum(margins),
                "symmetric_correct": sum(margins) > 0.0,
            }
        )
    return records


def pair_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for split in EVAL_SPLITS:
        selected = [record for record in records if record["split"] == split]
        completion = [
            bool(record[key])
            for record in selected
            for key in ("completion1_correct", "completion2_correct")
        ]
        result.append(
            {
                "split": split,
                "score_reduction": selected[0]["score_reduction"] if selected else "",
                "pairs": len(selected),
                "completion_correct": sum(completion),
                "completion_total": len(completion),
                "completion_accuracy": sum(completion) / len(completion) if completion else 0.0,
                "both_correct": sum(bool(record["both_correct"]) for record in selected),
                "both_correct_rate": (
                    sum(bool(record["both_correct"]) for record in selected) / len(selected)
                    if selected
                    else 0.0
                ),
                "symmetric_correct": sum(
                    bool(record["symmetric_correct"]) for record in selected
                ),
                "symmetric_accuracy": (
                    sum(bool(record["symmetric_correct"]) for record in selected)
                    / len(selected)
                    if selected
                    else 0.0
                ),
                "mean_completion_margin": (
                    statistics.mean(
                        margin
                        for record in selected
                        for margin in (record["margin1"], record["margin2"])
                    )
                    if selected
                    else 0.0
                ),
                "mean_symmetric_margin": (
                    statistics.mean(record["symmetric_margin"] for record in selected)
                    if selected
                    else 0.0
                ),
            }
        )
    return result


def write_pair_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--data-root",
        default="data/container_state/container_state_multidomain_v3_seed42",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-pairs-per-split", type=int)
    parser.add_argument("--score-reduction", choices=("sum", "mean", "both"), default="both")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_path(args.data_root, base_dir=ROOT)
    output_dir = resolve_path(args.output_dir, base_dir=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "five_choice_summary.csv",
        "five_choice_items.jsonl",
        "pair_summary.csv",
        "pair_items.jsonl",
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()

    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=load_dtype(args.dtype),
        **load_kwargs,
    ).to(torch.device(args.device))
    model.eval()

    five_choice = read_entity_tracking_items(data_root / "entity_tracking")
    if args.max_pairs_per_split is not None:
        limit = 2 * args.max_pairs_per_split
        five_choice = [
            item
            for split in EVAL_SPLITS
            for item in [candidate for candidate in five_choice if candidate.family == split][:limit]
        ]
    pairs = read_pairs(data_root / "pairs", args.max_pairs_per_split)
    paired_items = pair_items(pairs)
    reductions = parse_reductions(args.score_reduction)
    print(f"Model: {args.model}")
    print(f"Five-choice items: {len(five_choice):,}; paired examples: {len(pairs):,}")

    five_scores = score_continuations(
        model=model,
        tokenizer=tokenizer,
        requests=make_continuation_requests(five_choice),
        batch_size=args.batch_size,
        device=args.device,
    )
    paired_scores = score_continuations(
        model=model,
        tokenizer=tokenizer,
        requests=make_continuation_requests(paired_items),
        batch_size=args.batch_size,
        device=args.device,
    )
    metadata = {
        "run_name": args.model,
        "checkpoint_path": args.model,
        "epoch": "",
        "step": "",
        "tokens_seen": "",
        "words_seen": "",
        "checkpoint": "",
    }
    for reduction in reductions:
        five_records = build_item_records(
            items=five_choice,
            score_map=five_scores,
            score_reduction=reduction,  # type: ignore[arg-type]
        )
        append_summary_csv(
            aggregate_item_records(five_records),
            output_dir / "five_choice_summary.csv",
            metadata=metadata,
        )
        append_item_records_jsonl(
            five_records,
            output_dir / "five_choice_items.jsonl",
            metadata=metadata,
        )
        records = pair_records(pairs, paired_scores, reduction)
        with (output_dir / "pair_items.jsonl").open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        summaries = pair_summary(records)
        write_pair_summary(output_dir / "pair_summary.csv", summaries)
        for row in summaries:
            print(
                f"{reduction} {row['split']}: five-choice recorded; "
                f"completion={row['completion_accuracy']:.4f}, "
                f"symmetric={row['symmetric_accuracy']:.4f}"
            )
    print(f"Wrote multi-domain evaluation to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
