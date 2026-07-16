#!/usr/bin/env python3
"""Diagnose lexical/context priors in random Entity Tracking evaluation."""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from dataclasses import replace
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
    ContinuationScore,
    EntityTrackingItem,
    aggregate_item_records,
    build_item_records,
    make_continuation_requests,
    read_entity_tracking_items,
    resolve_padding_token_id,
    score_continuations,
)
from babysteps.models import build_causal_lm_model  # noqa: E402
from babysteps.paths import expand_env_vars, resolve_path, validate_tokenizer_dir  # noqa: E402


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
STOPWORDS = {"the", "and", "a", "an"}
QUERY_RE = re.compile(r"(?s)^(.*)(Box \d+ contains\s*)$")
VALID_MODES = ("full", "query_only", "scrambled", "unconditional")
VALID_REDUCTIONS = ("sum", "mean")

SUMMARY_FIELDNAMES = [
    "seed",
    "mode",
    "score_reduction",
    "subset",
    "score",
    "micro_score",
    "correct",
    "items",
    "ties",
    "mean_margin",
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
    "score_full_overlap_r",
    "score_recent_overlap_r",
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
        return VALID_REDUCTIONS
    if value in VALID_REDUCTIONS:
        return (value,)
    raise ValueError("--score-reduction must be sum, mean, or both.")


def parse_modes(raw_modes: list[str]) -> tuple[str, ...]:
    modes: list[str] = []
    for mode in raw_modes:
        value = mode.strip().lower()
        if value == "all":
            values = VALID_MODES
        elif value in VALID_MODES:
            values = (value,)
        else:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {VALID_MODES}.")
        for item in values:
            if item not in modes:
                modes.append(item)
    return tuple(modes)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def split_context_and_query(prefix: str) -> tuple[str, str]:
    match = QUERY_RE.match(prefix)
    if match:
        return match.group(1), match.group(2)
    parts = prefix.rsplit(". ", 1)
    if len(parts) == 2:
        return parts[0] + ". ", parts[1]
    return "", prefix


def query_only_item(item: EntityTrackingItem) -> EntityTrackingItem:
    _, query = split_context_and_query(item.input_prefix)
    return replace(item, input_prefix=query)


def scrambled_item(item: EntityTrackingItem, *, rng: random.Random) -> EntityTrackingItem:
    context, query = split_context_and_query(item.input_prefix)
    sentences = [sentence.strip() for sentence in context.split(".") if sentence.strip()]
    rng.shuffle(sentences)
    scrambled_context = ". ".join(sentences)
    if scrambled_context:
        scrambled_context += ". "
    return replace(item, input_prefix=f"{scrambled_context}{query}")


def mode_items(
    items: list[EntityTrackingItem],
    *,
    mode: str,
    scramble_seed: int,
) -> list[EntityTrackingItem]:
    if mode == "full":
        return items
    if mode == "query_only":
        return [query_only_item(item) for item in items]
    if mode == "scrambled":
        rng = random.Random(scramble_seed)
        return [scrambled_item(item, rng=rng) for item in items]
    if mode == "unconditional":
        return items
    raise ValueError(f"Unsupported mode: {mode!r}")


def resolve_bos_like_token_id(tokenizer: Any) -> int:
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Tokenizer has no bos_token_id, eos_token_id, or pad_token_id.")


def _batched(items: list[tuple[int, int, tuple[int, ...]]], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def score_options_unconditional(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[EntityTrackingItem],
    batch_size: int,
    device: torch.device,
) -> dict[int, dict[int, ContinuationScore]]:
    """Score ``P(option | BOS-like-token)`` for each Entity Tracking option."""

    bos_id = resolve_bos_like_token_id(tokenizer)
    padding_id = resolve_padding_token_id(tokenizer)
    encoded: list[tuple[int, int, tuple[int, ...]]] = []
    for item_index, item in enumerate(items):
        for option_index, option in enumerate(item.options):
            option_ids = tuple(tokenizer.encode(option, add_special_tokens=False))
            if not option_ids:
                raise ValueError(f"Option has no tokens: {option!r}")
            encoded.append((item_index, option_index, (bos_id, *option_ids)))

    scores: dict[int, dict[int, ContinuationScore]] = defaultdict(dict)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in _batched(encoded, batch_size):
                max_length = max(len(input_ids) for _, _, input_ids in batch)
                input_ids_tensor = torch.full(
                    (len(batch), max_length),
                    padding_id,
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.zeros_like(input_ids_tensor)
                for row_index, (_, _, input_ids) in enumerate(batch):
                    row_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
                    input_ids_tensor[row_index, : len(input_ids)] = row_ids
                    attention_mask[row_index, : len(input_ids)] = 1

                outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
                next_tokens = input_ids_tensor[:, 1:]
                token_log_probs = torch.gather(
                    log_probs,
                    dim=-1,
                    index=next_tokens.unsqueeze(-1),
                ).squeeze(-1)

                for row_index, (item_index, option_index, input_ids) in enumerate(batch):
                    count = len(input_ids) - 1
                    total = float(token_log_probs[row_index, :count].sum().item())
                    scores[item_index][option_index] = ContinuationScore(
                        sum_logprob=total,
                        scored_token_count=count,
                        input_token_count=len(input_ids),
                    )
    finally:
        if was_training:
            model.train()

    return {item_index: dict(item_scores) for item_index, item_scores in scores.items()}


def score_mode(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[EntityTrackingItem],
    mode: str,
    batch_size: int,
    device: torch.device,
) -> dict[int, dict[int, ContinuationScore]]:
    if mode == "unconditional":
        return score_options_unconditional(
            model=model,
            tokenizer=tokenizer,
            items=items,
            batch_size=batch_size,
            device=device,
        )
    return score_continuations(
        model=model,
        tokenizer=tokenizer,
        requests=make_continuation_requests(items),
        batch_size=batch_size,
        device=device,
    )


def overall_row(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    return next(row for row in rows if row["group_scope"] == scope and row["group_name"] == "all")


def equal_token_count_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if len({int(value) for value in record["scored_token_counts"]}) == 1
    ]


def prediction_distribution(item_records: list[dict[str, Any]]) -> dict[str, float]:
    counts = [0 for _ in range(5)]
    for record in item_records:
        predicted_index = record.get("predicted_index")
        if isinstance(predicted_index, int) and 0 <= predicted_index < 5:
            counts[predicted_index] += 1
    total = max(1, len(item_records))
    return {
        f"predicted_option_{index}_frac": count / total
        for index, count in enumerate(counts)
    }


def token_count_summary(item_records: list[dict[str, Any]]) -> dict[str, float]:
    if not item_records:
        return {
            "mean_correct_option_tokens": 0.0,
            "mean_distractor_option_tokens": 0.0,
            "mean_correct_minus_distractor_tokens": 0.0,
            "correct_shorter_pair_frac": 0.0,
            "correct_equal_pair_frac": 0.0,
            "correct_longer_pair_frac": 0.0,
            "correct_shorter_than_all_distractors_frac": 0.0,
            "correct_longer_than_all_distractors_frac": 0.0,
        }

    correct_counts: list[int] = []
    distractor_counts: list[int] = []
    pair_deltas: list[int] = []
    shorter_pairs = 0
    equal_pairs = 0
    longer_pairs = 0
    shorter_than_all = 0
    longer_than_all = 0
    for record in item_records:
        counts = [int(value) for value in record["scored_token_counts"]]
        correct_count = counts[0]
        item_distractor_counts = counts[1:]
        correct_counts.append(correct_count)
        distractor_counts.extend(item_distractor_counts)
        shorter_than_all += int(correct_count < min(item_distractor_counts))
        longer_than_all += int(correct_count > max(item_distractor_counts))
        for distractor_count in item_distractor_counts:
            delta = correct_count - distractor_count
            pair_deltas.append(delta)
            shorter_pairs += int(delta < 0)
            equal_pairs += int(delta == 0)
            longer_pairs += int(delta > 0)

    pair_count = max(1, len(pair_deltas))
    item_count = max(1, len(item_records))
    return {
        "mean_correct_option_tokens": float(np.mean(correct_counts)),
        "mean_distractor_option_tokens": float(np.mean(distractor_counts)),
        "mean_correct_minus_distractor_tokens": float(np.mean(pair_deltas)),
        "correct_shorter_pair_frac": shorter_pairs / pair_count,
        "correct_equal_pair_frac": equal_pairs / pair_count,
        "correct_longer_pair_frac": longer_pairs / pair_count,
        "correct_shorter_than_all_distractors_frac": shorter_than_all / item_count,
        "correct_longer_than_all_distractors_frac": longer_than_all / item_count,
    }


def content_words(text: str) -> list[str]:
    return [word.lower() for word in WORD_RE.findall(text) if word.lower() not in STOPWORDS]


def safe_corrcoef(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return 0.0
    return float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1])


def overlap_correlations(
    *,
    items: list[EntityTrackingItem],
    item_records: list[dict[str, Any]],
    recent_window_words: int,
) -> dict[str, float]:
    scores: list[float] = []
    full_overlaps: list[float] = []
    recent_overlaps: list[float] = []
    for item, record in zip(items, item_records, strict=True):
        full_context = set(content_words(item.input_prefix))
        recent_context = set(content_words(item.input_prefix)[-recent_window_words:])
        for option, score in zip(item.options, record["option_scores"], strict=True):
            words = content_words(option)
            denom = max(1, len(words))
            full_overlaps.append(sum(word in full_context for word in words) / denom)
            recent_overlaps.append(sum(word in recent_context for word in words) / denom)
            scores.append(float(score))
    return {
        "score_full_overlap_r": safe_corrcoef(scores, full_overlaps),
        "score_recent_overlap_r": safe_corrcoef(scores, recent_overlaps),
    }


def summarize_records(
    *,
    seed: int,
    mode: str,
    reduction: str,
    subset: str,
    mode_items_: list[EntityTrackingItem],
    item_records: list[dict[str, Any]],
    recent_window_words: int,
) -> dict[str, Any]:
    if not item_records:
        return {
            "seed": seed,
            "mode": mode,
            "score_reduction": reduction,
            "subset": subset,
            "score": 0.0,
            "micro_score": 0.0,
            "correct": 0,
            "items": 0,
            "ties": 0,
            "mean_margin": 0.0,
            **prediction_distribution([]),
            **token_count_summary([]),
            "score_full_overlap_r": 0.0,
            "score_recent_overlap_r": 0.0,
        }

    summary_rows = aggregate_item_records(item_records)
    overall = overall_row(summary_rows, "overall")
    overall_micro = overall_row(summary_rows, "overall_micro")
    return {
        "seed": seed,
        "mode": mode,
        "score_reduction": reduction,
        "subset": subset,
        "score": float(overall["score"]),
        "micro_score": float(overall_micro["score"]),
        "correct": int(overall["correct"]),
        "items": int(overall["items"]),
        "ties": int(overall["ties"]),
        "mean_margin": float(overall["mean_margin"]),
        **prediction_distribution(item_records),
        **token_count_summary(item_records),
        **overlap_correlations(
            items=mode_items_,
            item_records=item_records,
            recent_window_words=recent_window_words,
        ),
    }


def evaluate_seed(
    *,
    config: dict[str, Any],
    tokenizer: Any,
    base_items: list[EntityTrackingItem],
    seed: int,
    modes: tuple[str, ...],
    reductions: tuple[str, ...],
    batch_size: int,
    device: torch.device,
    scramble_seed: int,
    recent_window_words: int,
) -> list[dict[str, Any]]:
    set_seed(seed)
    model = build_causal_lm_model(config, tokenizer).to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    try:
        for mode in modes:
            items = mode_items(base_items, mode=mode, scramble_seed=scramble_seed)
            score_map = score_mode(
                model=model,
                tokenizer=tokenizer,
                items=items,
                mode=mode,
                batch_size=batch_size,
                device=device,
            )
            for reduction in reductions:
                item_records = build_item_records(
                    items=items,
                    score_map=score_map,
                    score_reduction=reduction,  # type: ignore[arg-type]
                )
                rows.append(
                    summarize_records(
                        seed=seed,
                        mode=mode,
                        reduction=reduction,
                        subset="all",
                        mode_items_=items,
                        item_records=item_records,
                        recent_window_words=recent_window_words,
                    )
                )
                equal_records = equal_token_count_records(item_records)
                equal_items = [
                    item
                    for item, record in zip(items, item_records, strict=True)
                    if len({int(value) for value in record["scored_token_counts"]}) == 1
                ]
                rows.append(
                    summarize_records(
                        seed=seed,
                        mode=mode,
                        reduction=reduction,
                        subset="equal_token_count",
                        mode_items_=equal_items,
                        item_records=equal_records,
                        recent_window_words=recent_window_words,
                    )
                )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    print("\nEntity Tracking prior diagnostics")
    print("mode           subset             red   score   micro   opt0  margin    full_r  recent_r  items")
    for row in rows:
        print(
            f"{row['mode']:<14} "
            f"{row['subset']:<18} "
            f"{row['score_reduction']:<5} "
            f"{row['score']:.4f} "
            f"{row['micro_score']:.4f} "
            f"{row['predicted_option_0_frac']:.3f} "
            f"{row['mean_margin']:.4f} "
            f"{row['score_full_overlap_r']:+.3f} "
            f"{row['score_recent_overlap_r']:+.3f} "
            f"{row['items']:>5}"
        )

    print("\nAggregate by mode/reduction/subset")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["score_reduction"], row["subset"])].append(row)
    for (mode, reduction, subset), group_rows in sorted(grouped.items()):
        scores = np.asarray([float(row["score"]) for row in group_rows])
        opt0 = np.asarray([float(row["predicted_option_0_frac"]) for row in group_rows])
        print(
            f"{mode:<14} {subset:<18} {reduction:<5} "
            f"score_mean={scores.mean():.4f} score_std={scores.std(ddof=0):.4f} "
            f"option0_mean={opt0.mean():.4f}"
        )

    first = rows[0]
    print("\nToken-count diagnostic from first row")
    print(
        f"mean_correct_tokens={first['mean_correct_option_tokens']:.4f}  "
        f"mean_distractor_tokens={first['mean_distractor_option_tokens']:.4f}  "
        f"mean_correct_minus_distractor={first['mean_correct_minus_distractor_tokens']:.4f}"
    )
    print(
        f"correct_shorter_pair_frac={first['correct_shorter_pair_frac']:.3f}  "
        f"correct_equal_pair_frac={first['correct_equal_pair_frac']:.3f}  "
        f"correct_longer_pair_frac={first['correct_longer_pair_frac']:.3f}"
    )
    print("\nUniform random-choice chance for 5 options is 0.2000.")


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
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["all"],
        help="Modes: all, full, query_only, scrambled, unconditional.",
    )
    parser.add_argument(
        "--score-reduction",
        choices=("sum", "mean", "both"),
        default="both",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-items", type=int, help="Optional quick-test item cap.")
    parser.add_argument("--scramble-seed", type=int, default=123)
    parser.add_argument("--recent-window-words", type=int, default=64)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device. With CUDA_VISIBLE_DEVICES=2, use the default cuda.",
    )
    parser.add_argument("--output-csv", help="Optional path for diagnostic summary CSV.")
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
    modes = parse_modes(args.modes)
    reductions = parse_reductions(args.score_reduction)
    device = torch.device(args.device)

    print(f"Config: {resolve_path(args.config, base_dir=ROOT)}")
    print(f"Data: {resolve_path(args.data_dir, base_dir=ROOT)}")
    print(f"Items: {len(items):,}")
    print(f"Seeds: {args.seeds}")
    print(f"Modes: {', '.join(modes)}")
    print(f"Reductions: {', '.join(reductions)}")
    print(f"Device: {device}")

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        rows.extend(
            evaluate_seed(
                config=config,
                tokenizer=tokenizer,
                base_items=items,
                seed=seed,
                modes=modes,
                reductions=reductions,
                batch_size=args.batch_size,
                device=device,
                scramble_seed=args.scramble_seed,
                recent_window_words=args.recent_window_words,
            )
        )

    print_table(rows)
    if args.output_csv:
        output_path = resolve_path(args.output_csv, base_dir=ROOT)
        write_csv(rows, output_path)
        print(f"Wrote CSV: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
