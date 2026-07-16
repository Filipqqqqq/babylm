#!/usr/bin/env python3
"""Analyze recency shortcuts in official Entity Tracking predictions.

This script consumes the ``predictions.json`` written by the BabyLM strict
zero-shot evaluator.  It does not load a model or rerun inference.

Definitions used here:

* A non-nothing example has a gold option other than exactly ``nothing.``.
* An operation "touches" the query box when its instruction literally
  mentions that box, either as a source or destination.
* The query box is touched in the last three instructions when its most recent
  touching operation has zero, one, or two later operation instructions.
* The recency heuristic gives each option the mean last-occurrence position of
  its unique content words in the context, then selects the highest-scoring
  option.  Ties receive expected random tie-break credit.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (
    ROOT
    / "external"
    / "babylm-eval"
    / "strict"
    / "evaluation_data"
    / "full_eval"
    / "entity_tracking"
)

QUERY_RE = re.compile(r"Box (?P<box>\d+) contains\s*$")
OPERATION_RE = re.compile(r"\b(?:Move|Remove|Put)\b[^.]*\.")
WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
CONTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "box",
    "contains",
    "from",
    "into",
    "move",
    "nothing",
    "put",
    "remove",
    "the",
    "to",
}
FAMILIES = ("regular", "ambiref", "move_contents")


@dataclass(frozen=True)
class ExampleResult:
    family: str
    num_ops: int
    target: str
    prediction: str
    correct: bool
    is_nothing: bool
    instruction_count: int
    last_touch_distance: int | None
    recency_heuristic_credit: float


def content_words(text: str) -> tuple[str, ...]:
    """Return lower-cased non-control words in token order."""

    return tuple(
        word.lower()
        for word in WORD_RE.findall(text)
        if word.lower() not in CONTENT_STOPWORDS
    )


def unique_content_words(text: str) -> tuple[str, ...]:
    """Return content words once each, preserving first-seen order."""

    return tuple(dict.fromkeys(content_words(text)))


def expected_correct_credit(scores: list[float]) -> float:
    """Return expected accuracy when ties are broken uniformly at random."""

    maximum = max(scores)
    winners = [index for index, score in enumerate(scores) if score == maximum]
    return 1.0 / len(winners) if 0 in winners else 0.0


def recency_heuristic_credit(context: str, options: list[str]) -> float:
    """Score options by the mean last position of their unique content words."""

    last_positions: dict[str, int] = {}
    for index, word in enumerate(content_words(context)):
        last_positions[word] = index

    scores: list[float] = []
    for option in options:
        words = unique_content_words(option)
        if not words:
            scores.append(-1.0)
            continue
        scores.append(mean(last_positions.get(word, -1) for word in words))
    return expected_correct_credit(scores)


def parse_context(prefix: str) -> tuple[int, str, list[str]]:
    """Return the queried box, context without the query, and operations."""

    query_match = QUERY_RE.search(prefix)
    if query_match is None:
        raise ValueError(f"Could not parse final query from input_prefix: {prefix!r}")
    query_box = int(query_match.group("box"))
    context = prefix[: query_match.start()].rstrip()
    operations = OPERATION_RE.findall(context)
    return query_box, context, operations


def last_touch_distance(query_box: int, operations: list[str]) -> int | None:
    """Count operation instructions after the final query-box mention."""

    box_re = re.compile(rf"\bBox {query_box}\b")
    touching_indices = [
        index for index, operation in enumerate(operations) if box_re.search(operation)
    ]
    if not touching_indices:
        return None
    return len(operations) - touching_indices[-1] - 1


def load_examples(data_dir: Path, predictions_path: Path) -> list[ExampleResult]:
    """Join official records to predictions and validate all UID alignments."""

    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    prediction_indices: dict[str, int] = defaultdict(int)
    examples: list[ExampleResult] = []

    for data_path in sorted(data_dir.glob("*.jsonl")):
        family = data_path.stem
        for raw_line in data_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(raw_line)
            num_ops = int(record["numops"])
            uid = f"{family}_{num_ops}_ops"
            if uid not in predictions:
                raise ValueError(f"Predictions are missing UID {uid!r}")

            prediction_index = prediction_indices[uid]
            uid_predictions = predictions[uid]["predictions"]
            if prediction_index >= len(uid_predictions):
                raise ValueError(f"Not enough predictions for UID {uid!r}")
            prediction_record = uid_predictions[prediction_index]
            expected_id = f"{uid}_{prediction_index}"
            if prediction_record.get("id") != expected_id:
                raise ValueError(
                    f"Prediction alignment failed: expected {expected_id!r}, "
                    f"found {prediction_record.get('id')!r}"
                )
            prediction_indices[uid] += 1

            options = [str(option).strip() for option in record["options"]]
            target = options[0]
            prediction = str(prediction_record["pred"]).strip()
            query_box, context, operations = parse_context(record["input_prefix"])
            distance = last_touch_distance(query_box, operations)
            observed_num_ops = sum(
                bool(re.search(rf"\bBox {query_box}\b", operation))
                for operation in operations
            )
            if observed_num_ops != num_ops:
                raise ValueError(
                    f"numops mismatch for {expected_id}: metadata={num_ops}, "
                    f"parsed_query_box_touches={observed_num_ops}"
                )

            examples.append(
                ExampleResult(
                    family=family,
                    num_ops=num_ops,
                    target=target,
                    prediction=prediction,
                    correct=prediction == target,
                    is_nothing=target.lower() == "nothing.",
                    instruction_count=len(operations),
                    last_touch_distance=distance,
                    recency_heuristic_credit=recency_heuristic_credit(context, options),
                )
            )

    for uid, uid_predictions in predictions.items():
        expected = len(uid_predictions["predictions"])
        observed = prediction_indices.get(uid, 0)
        if observed != expected:
            raise ValueError(
                f"Prediction count mismatch for {uid!r}: used {observed}, found {expected}"
            )
    return examples


def accuracy_summary(examples: list[ExampleResult]) -> dict[str, Any]:
    correct = sum(example.correct for example in examples)
    total = len(examples)
    return {
        "correct": correct,
        "items": total,
        "accuracy": correct / total if total else None,
    }


def official_macro_accuracy(examples: list[ExampleResult]) -> float | None:
    """Apply the official family/numops macro averaging to a filtered subset."""

    uid_scores: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for example in examples:
        uid_scores[(example.family, example.num_ops)].append(example.correct)

    family_scores: list[float] = []
    for family in FAMILIES:
        scores = [
            mean(uid_scores[(family, num_ops)])
            for num_ops in range(6)
            if uid_scores[(family, num_ops)]
        ]
        if scores:
            family_scores.append(mean(scores))
    return mean(family_scores) if family_scores else None


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def select(
    examples: Iterable[ExampleResult],
    predicate: Callable[[ExampleResult], bool],
) -> list[ExampleResult]:
    return [example for example in examples if predicate(example)]


def build_summary(examples: list[ExampleResult]) -> dict[str, Any]:
    non_nothing = select(examples, lambda example: not example.is_nothing)
    nothing = select(examples, lambda example: example.is_nothing)

    overall: dict[str, Any] = {}
    for name, subset in (
        ("all", examples),
        ("nothing", nothing),
        ("non_nothing", non_nothing),
    ):
        overall[name] = {
            **accuracy_summary(subset),
            "official_macro_accuracy": official_macro_accuracy(subset),
        }

    by_num_ops: list[dict[str, Any]] = []
    for num_ops in range(6):
        subset = select(non_nothing, lambda example, n=num_ops: example.num_ops == n)
        full_subset = select(examples, lambda example, n=num_ops: example.num_ops == n)
        by_num_ops.append(
            {
                "num_ops": num_ops,
                **accuracy_summary(subset),
                "nothing_fraction_all_examples": (
                    sum(example.is_nothing for example in full_subset) / len(full_subset)
                    if full_subset
                    else None
                ),
                "mean_instruction_count": (
                    mean(example.instruction_count for example in subset) if subset else None
                ),
                "query_touched_in_last_3_fraction": (
                    sum(
                        example.last_touch_distance is not None
                        and example.last_touch_distance <= 2
                        for example in subset
                    )
                    / len(subset)
                    if subset
                    else None
                ),
                "recency_heuristic_expected_accuracy": (
                    mean(example.recency_heuristic_credit for example in subset)
                    if subset
                    else None
                ),
            }
        )

    positive_ops = select(non_nothing, lambda example: example.num_ops > 0)
    distance_groups = (
        ("last_instruction", lambda value: value == 0),
        ("one_instruction_back", lambda value: value == 1),
        ("two_instructions_back", lambda value: value == 2),
        ("three_to_five_back", lambda value: value is not None and 3 <= value <= 5),
        ("six_or_more_back", lambda value: value is not None and value >= 6),
        ("never_touched", lambda value: value is None),
    )
    by_last_touch: list[dict[str, Any]] = []
    for name, predicate in distance_groups:
        subset = select(
            positive_ops,
            lambda example, fn=predicate: fn(example.last_touch_distance),
        )
        if subset:
            by_last_touch.append({"distance_group": name, **accuracy_summary(subset)})

    return {
        "definitions": {
            "non_nothing": "Gold option is not exactly 'nothing.'.",
            "query_box_touch": (
                "An operation instruction literally mentions the queried box as a source "
                "or destination."
            ),
            "last_3": (
                "The final query-box-touching operation has no more than two later "
                "operation instructions."
            ),
            "recency_heuristic": (
                "Select the option with the greatest mean last-occurrence position of "
                "its unique content words in the context; split credit uniformly for ties."
            ),
        },
        "validation": {
            "examples": len(examples),
            "numops_equals_parsed_query_box_touches_for_every_example": True,
        },
        "overall": overall,
        "non_nothing_by_num_ops": by_num_ops,
        "non_nothing_positive_ops_by_last_touch_distance": by_last_touch,
    }


def print_report(summary: dict[str, Any]) -> None:
    print("Entity Tracking prediction/recency analysis")
    print(f"Validated examples: {summary['validation']['examples']:,}")
    print("Chance accuracy with five options: 20.00%")

    print("\nTarget subsets")
    print("subset        correct/items  micro    official_macro")
    for name in ("all", "nothing", "non_nothing"):
        row = summary["overall"][name]
        print(
            f"{name:<13} {row['correct']:>4}/{row['items']:<5}   "
            f"{percent(row['accuracy']):>7}  {percent(row['official_macro_accuracy']):>14}"
        )

    print("\nNon-nothing examples by numops")
    print("ops  correct/items  model    instructions  touched_last3  recency_heuristic")
    for row in summary["non_nothing_by_num_ops"]:
        print(
            f"{row['num_ops']:>3}  {row['correct']:>4}/{row['items']:<5}   "
            f"{percent(row['accuracy']):>7}  "
            f"{row['mean_instruction_count']:>12.2f}  "
            f"{percent(row['query_touched_in_last_3_fraction']):>13}  "
            f"{percent(row['recency_heuristic_expected_accuracy']):>17}"
        )

    print("\nNon-nothing examples with 1-5 ops by last query-box touch")
    print("distance              correct/items  model")
    for row in summary["non_nothing_positive_ops_by_last_touch_distance"]:
        print(
            f"{row['distance_group']:<21} {row['correct']:>4}/{row['items']:<5}   "
            f"{percent(row['accuracy']):>7}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Official Entity Tracking predictions.json path.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Entity Tracking JSONL directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for the complete machine-readable analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(args.data_dir.resolve(), args.predictions.resolve())
    summary = build_summary(examples)
    print_report(summary)
    if args.output_json:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote JSON: {output_path}")


if __name__ == "__main__":
    main()
