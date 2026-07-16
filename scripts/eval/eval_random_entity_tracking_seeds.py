#!/usr/bin/env python3
"""Evaluate random-initialized causal LMs on Entity Tracking across seeds."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.evaluation.entity_tracking_likelihood import (  # noqa: E402
    aggregate_item_records,
    build_item_records,
    make_continuation_requests,
    read_entity_tracking_items,
    score_continuations,
)
from babysteps.models import build_causal_lm_model  # noqa: E402
from babysteps.paths import expand_env_vars, resolve_path, validate_tokenizer_dir  # noqa: E402


SUMMARY_FIELDNAMES = [
    "seed",
    "score_reduction",
    "score",
    "correct",
    "items",
    "ties",
    "mean_margin",
    "overall_micro_score",
    "predicted_option_0",
    "predicted_option_1",
    "predicted_option_2",
    "predicted_option_3",
    "predicted_option_4",
    "predicted_option_0_frac",
    "predicted_option_1_frac",
    "predicted_option_2_frac",
    "predicted_option_3_frac",
    "predicted_option_4_frac",
    "mean_correct_option_tokens",
    "mean_distractor_option_tokens",
    "mean_correct_minus_distractor_tokens",
    "correct_shorter_pair_frac",
    "correct_equal_pair_frac",
    "correct_longer_pair_frac",
    "correct_shorter_than_all_distractors_frac",
    "correct_longer_than_all_distractors_frac",
]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return expand_env_vars(yaml.safe_load(f))


def default_entity_tracking_data_dir() -> Path:
    eval_root = ROOT / "external" / "babylm-eval" / "strict" / "evaluation_data"
    return eval_root / "full_eval" / "entity_tracking"


def parse_reductions(raw_value: str) -> tuple[str, ...]:
    value = raw_value.strip().lower()
    if value == "both":
        return ("sum", "mean")
    if value in {"sum", "mean"}:
        return (value,)
    raise ValueError("--score-reduction must be sum, mean, or both.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def overall_row(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    return next(row for row in rows if row["group_scope"] == scope and row["group_name"] == "all")


def prediction_distribution(item_records: list[dict[str, Any]], option_count: int = 5) -> dict[str, Any]:
    counts = [0 for _ in range(option_count)]
    for record in item_records:
        predicted_index = record.get("predicted_index")
        if isinstance(predicted_index, int) and 0 <= predicted_index < option_count:
            counts[predicted_index] += 1

    total = max(1, len(item_records))
    output: dict[str, Any] = {}
    for index, count in enumerate(counts):
        output[f"predicted_option_{index}"] = count
        output[f"predicted_option_{index}_frac"] = count / total
    return output


def token_count_summary(item_records: list[dict[str, Any]]) -> dict[str, float]:
    correct_counts: list[int] = []
    distractor_counts: list[int] = []
    pair_deltas: list[int] = []
    correct_shorter_pairs = 0
    correct_equal_pairs = 0
    correct_longer_pairs = 0
    correct_shorter_than_all = 0
    correct_longer_than_all = 0

    for record in item_records:
        counts = [int(value) for value in record["scored_token_counts"]]
        correct_count = counts[0]
        item_distractor_counts = counts[1:]
        correct_counts.append(correct_count)
        distractor_counts.extend(item_distractor_counts)

        if correct_count < min(item_distractor_counts):
            correct_shorter_than_all += 1
        if correct_count > max(item_distractor_counts):
            correct_longer_than_all += 1

        for distractor_count in item_distractor_counts:
            delta = correct_count - distractor_count
            pair_deltas.append(delta)
            if delta < 0:
                correct_shorter_pairs += 1
            elif delta == 0:
                correct_equal_pairs += 1
            else:
                correct_longer_pairs += 1

    pair_count = max(1, len(pair_deltas))
    item_count = max(1, len(item_records))
    return {
        "mean_correct_option_tokens": float(np.mean(correct_counts)),
        "mean_distractor_option_tokens": float(np.mean(distractor_counts)),
        "mean_correct_minus_distractor_tokens": float(np.mean(pair_deltas)),
        "correct_shorter_pair_frac": correct_shorter_pairs / pair_count,
        "correct_equal_pair_frac": correct_equal_pairs / pair_count,
        "correct_longer_pair_frac": correct_longer_pairs / pair_count,
        "correct_shorter_than_all_distractors_frac": correct_shorter_than_all / item_count,
        "correct_longer_than_all_distractors_frac": correct_longer_than_all / item_count,
    }


def evaluate_seed(
    *,
    config: dict[str, Any],
    tokenizer: Any,
    items: list[Any],
    seed: int,
    reductions: tuple[str, ...],
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    set_seed(seed)
    model = build_causal_lm_model(config, tokenizer).to(device)
    model.eval()

    try:
        score_map = score_continuations(
            model=model,
            tokenizer=tokenizer,
            requests=make_continuation_requests(items),
            batch_size=batch_size,
            device=device,
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for reduction in reductions:
        item_records = build_item_records(
            items=items,
            score_map=score_map,
            score_reduction=reduction,  # type: ignore[arg-type]
        )
        summary_rows = aggregate_item_records(item_records)
        overall = overall_row(summary_rows, "overall")
        overall_micro = overall_row(summary_rows, "overall_micro")
        rows.append(
            {
                "seed": seed,
                "score_reduction": reduction,
                "score": float(overall["score"]),
                "correct": int(overall["correct"]),
                "items": int(overall["items"]),
                "ties": int(overall["ties"]),
                "mean_margin": float(overall["mean_margin"]),
                "overall_micro_score": float(overall_micro["score"]),
                **prediction_distribution(item_records),
                **token_count_summary(item_records),
            }
        )
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    print("\nPer-seed Entity Tracking scores")
    print("seed  reduction  score    micro    correct/items  ties  mean_margin")
    for row in rows:
        print(
            f"{row['seed']:>4}  "
            f"{row['score_reduction']:<9}  "
            f"{row['score']:.4f}  "
            f"{row['overall_micro_score']:.4f}  "
            f"{row['correct']:>7}/{row['items']:<5}  "
            f"{row['ties']:>4}  "
            f"{row['mean_margin']:.4f}"
        )

    print("\nAggregate over seeds")
    for reduction in sorted({str(row["score_reduction"]) for row in rows}):
        scores = np.asarray(
            [float(row["score"]) for row in rows if row["score_reduction"] == reduction],
            dtype=np.float64,
        )
        print(
            f"{reduction:<9} "
            f"mean={scores.mean():.4f} "
            f"std={scores.std(ddof=0):.4f} "
            f"min={scores.min():.4f} "
            f"max={scores.max():.4f}"
        )
    print("\nUniform random-choice chance for 5 options is 0.2000.")

    print("\nAggregate predicted option distribution")
    for reduction in sorted({str(row["score_reduction"]) for row in rows}):
        reduction_rows = [row for row in rows if row["score_reduction"] == reduction]
        total_items = sum(int(row["items"]) for row in reduction_rows)
        parts = []
        for index in range(5):
            count = sum(int(row[f"predicted_option_{index}"]) for row in reduction_rows)
            parts.append(f"option_{index}={count / max(1, total_items):.3f}")
        print(f"{reduction:<9} " + "  ".join(parts))

    token_row = rows[0]
    print("\nCorrect option token counts vs distractor token counts")
    print(
        "mean_correct_tokens="
        f"{token_row['mean_correct_option_tokens']:.4f}  "
        "mean_distractor_tokens="
        f"{token_row['mean_distractor_option_tokens']:.4f}  "
        "mean_correct_minus_distractor="
        f"{token_row['mean_correct_minus_distractor_tokens']:.4f}"
    )
    print(
        "pairwise fractions: "
        f"correct_shorter={token_row['correct_shorter_pair_frac']:.3f}  "
        f"equal={token_row['correct_equal_pair_frac']:.3f}  "
        f"correct_longer={token_row['correct_longer_pair_frac']:.3f}"
    )
    print(
        "per-item extremes: "
        "correct_shorter_than_all_distractors="
        f"{token_row['correct_shorter_than_all_distractors_frac']:.3f}  "
        "correct_longer_than_all_distractors="
        f"{token_row['correct_longer_than_all_distractors_frac']:.3f}"
    )


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in SUMMARY_FIELDNAMES})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/train/gpt2_8l512_bpe16384_babylm2026_strict_small_muon.yaml",
        help="BabySteps training config used only to build the random model and tokenizer.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(default_entity_tracking_data_dir()),
        help="Entity Tracking JSONL directory.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random initialization seeds.",
    )
    parser.add_argument(
        "--score-reduction",
        choices=("sum", "mean", "both"),
        default="both",
        help="Evaluate sum, mean, or both.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-items", type=int, help="Optional quick-test item cap.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device. With CUDA_VISIBLE_DEVICES=2, use the default cuda.",
    )
    parser.add_argument("--output-csv", help="Optional path for per-seed summary CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(resolve_path(args.config, base_dir=ROOT))
    tokenizer_name = config["tokenizer"].get("name", "gpt2")
    tokenizer_path = validate_tokenizer_dir(tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path or tokenizer_name))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    items = read_entity_tracking_items(resolve_path(args.data_dir, base_dir=ROOT))
    if args.max_items:
        items = items[: args.max_items]
    reductions = parse_reductions(args.score_reduction)
    device = torch.device(args.device)

    print(f"Config: {resolve_path(args.config, base_dir=ROOT)}")
    print(f"Data: {resolve_path(args.data_dir, base_dir=ROOT)}")
    print(f"Items: {len(items):,}")
    print(f"Seeds: {args.seeds}")
    print(f"Reductions: {', '.join(reductions)}")
    print(f"Device: {device}")

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        rows.extend(
            evaluate_seed(
                config=config,
                tokenizer=tokenizer,
                items=items,
                seed=seed,
                reductions=reductions,
                batch_size=args.batch_size,
                device=device,
            )
        )

    print_table(rows)
    if args.output_csv:
        write_csv(rows, resolve_path(args.output_csv, base_dir=ROOT))
        print(f"Wrote CSV: {resolve_path(args.output_csv, base_dir=ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
