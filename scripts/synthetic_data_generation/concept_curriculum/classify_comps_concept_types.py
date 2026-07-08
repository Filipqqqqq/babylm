#!/usr/bin/env python3
"""Classify COMPS concept names into broad prompt-routing types."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.material_world.fireworks_client import chat_json_or_text, load_api_key


DEFAULT_BATCH_SIZE = 40
DEFAULT_MAX_TOKENS = 4096
DEFAULT_RETRY_MAX_TOKENS = 6144
DEFAULT_OUTPUT_DIR = (
    "data/concept_curriculum/comps_concept_type_classification/qwen_batch40"
)
DEFAULT_ANNOTATION_OUTPUT = (
    "configs/concept_curriculum/comps_concept_properties/"
    "comps_concept_types_qwen_batch40.json"
)
DEFAULT_COMPS_PATH = (
    "external/babylm-eval/strict/evaluation_data/full_eval/comps/comps_base.jsonl"
)


def _load_comps_module() -> Any:
    module_name = "comps_concept_properties_for_type_classification"
    module_path = ROOT / "src/babysteps/concept_curriculum/comps_concept_properties.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMPS = _load_comps_module()


class ClassificationBatchError(ValueError):
    """A validation or parsing failure for one classifier batch."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        raw_text: str = "",
    ) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.raw_text = raw_text


@dataclass(frozen=True)
class BatchClassificationResult:
    batch_index: int
    requested_concepts: tuple[str, ...]
    concept_types: dict[str, tuple[str, ...]]
    uncertain: tuple[dict[str, str], ...]
    empty_label_concepts: tuple[str, ...]
    max_tokens: int
    attempt_index: int


def split_batches(items: tuple[str, ...] | list[str], *, batch_size: int) -> tuple[tuple[str, ...], ...]:
    """Split items into deterministic fixed-size batches."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    values = tuple(items)
    return tuple(
        values[start : start + int(batch_size)]
        for start in range(0, len(values), int(batch_size))
    )


def build_classifier_prompt(concepts: tuple[str, ...], *, allowed_labels: tuple[str, ...]) -> str:
    """Build a compact concept-name-only classifier prompt."""

    return f"""You are assigning broad semantic categories to concept names for ordinary educational text generation.

Choose zero or more labels from this allowed list:
{json.dumps(list(allowed_labels), ensure_ascii=False)}

Use only labels from the allowed list. Do not invent new labels.
If the best everyday category is not in the allowed list, choose the closest broader allowed label instead.

Use multiple labels when a concept naturally has multiple common roles.
Examples:
salmon -> ["fish", "food"]
grape -> ["plant", "food"]
kiwi -> ["fruit"]
carrot -> ["vegetable"]
parsley -> ["plant", "ingredient"]
ice_cream -> ["dessert"]
sugar -> ["food", "substance"]
cotton -> ["plant", "material"]
ball -> ["toy", "sports_equipment"]
clarinet -> ["musical_instrument"]
bikini -> ["clothing"]
bomb -> ["weapon"]
ladle -> ["kitchen_utensil"]
strainer -> ["kitchen_utensil"]
unknown_word -> []

If uncertain, use [] for that concept and add a short uncertain entry.

Concepts:
{json.dumps(list(concepts), ensure_ascii=False, indent=2)}

Return only compact JSON with this exact shape:
{{"concept_types": {{"concept_id": ["label"]}}, "uncertain": [{{"concept": "concept_id", "reason": "short reason"}}]}}

Include every requested concept exactly once in concept_types. Use only allowed labels. Do not include explanations outside JSON. /no_think"""


def _duplicate_tracking_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    duplicate_keys: list[str] = []
    for key, value in pairs:
        if key in result:
            duplicate_keys.append(str(key))
        result[key] = value
    if duplicate_keys:
        result["__duplicate_keys__"] = duplicate_keys
    return result


def _contains_duplicate_key_marker(value: Any) -> bool:
    if isinstance(value, dict):
        if "__duplicate_keys__" in value:
            return True
        return any(_contains_duplicate_key_marker(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_duplicate_key_marker(item) for item in value)
    return False


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a model response into a JSON object, allowing fenced JSON."""

    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        stripped = stripped.rsplit("```", 1)[0].strip()
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and start < end:
            stripped = stripped[start : end + 1]
    try:
        payload = json.loads(stripped, object_pairs_hook=_duplicate_tracking_hook)
    except json.JSONDecodeError as exc:
        raise ClassificationBatchError(f"malformed JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ClassificationBatchError("classifier response must be a JSON object")
    if _contains_duplicate_key_marker(payload):
        raise ClassificationBatchError("classifier response contains duplicate JSON keys")
    return payload


def validate_classification_payload(
    payload: dict[str, Any],
    *,
    requested_concepts: tuple[str, ...],
    allowed_labels: tuple[str, ...],
) -> BatchClassificationResult:
    """Validate one parsed classifier response."""

    raw_mapping = payload.get("concept_types")
    if not isinstance(raw_mapping, dict):
        raise ClassificationBatchError("classifier response is missing object field concept_types")

    expected = set(requested_concepts)
    observed = set(str(key) for key in raw_mapping)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ClassificationBatchError(
            "classifier response concept set mismatch "
            f"missing={missing} extra={extra}"
        )

    allowed = set(allowed_labels)
    normalized: dict[str, tuple[str, ...]] = {}
    invalid_labels: dict[str, list[str]] = {}
    empty_label_concepts: list[str] = []
    for concept in requested_concepts:
        raw_labels = raw_mapping[concept]
        if not isinstance(raw_labels, list):
            raise ClassificationBatchError(
                f"labels for {concept!r} must be a list"
            )
        labels: list[str] = []
        seen: set[str] = set()
        for raw_label in raw_labels:
            label = COMPS.normalize_concept_type(raw_label)
            if not label:
                continue
            if label not in allowed:
                invalid_labels.setdefault(concept, []).append(str(raw_label))
                continue
            if label not in seen:
                labels.append(label)
                seen.add(label)
        if invalid_labels.get(concept):
            continue
        if not labels:
            empty_label_concepts.append(concept)
        normalized[concept] = tuple(labels)

    if invalid_labels:
        raise ClassificationBatchError(f"classifier response has invalid labels: {invalid_labels}")

    uncertain = []
    raw_uncertain = payload.get("uncertain", [])
    if raw_uncertain is None:
        raw_uncertain = []
    if not isinstance(raw_uncertain, list):
        raise ClassificationBatchError("uncertain must be a list")
    for item in raw_uncertain:
        if not isinstance(item, dict):
            raise ClassificationBatchError("uncertain entries must be objects")
        concept = str(item.get("concept", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if concept not in expected:
            raise ClassificationBatchError(f"uncertain concept {concept!r} was not requested")
        uncertain.append({"concept": concept, "reason": reason[:160]})

    return BatchClassificationResult(
        batch_index=-1,
        requested_concepts=requested_concepts,
        concept_types=normalized,
        uncertain=tuple(uncertain),
        empty_label_concepts=tuple(empty_label_concepts),
        max_tokens=0,
        attempt_index=0,
    )


def should_retry_classification_failure(exc: BaseException) -> bool:
    return isinstance(exc, ClassificationBatchError) and exc.retryable


def result_to_json(result: BatchClassificationResult) -> dict[str, Any]:
    return {
        "batch_index": result.batch_index,
        "requested_concepts": list(result.requested_concepts),
        "concept_types": {
            concept: list(labels)
            for concept, labels in result.concept_types.items()
        },
        "uncertain": list(result.uncertain),
        "empty_label_concepts": list(result.empty_label_concepts),
        "max_tokens": result.max_tokens,
        "attempt_index": result.attempt_index,
    }


def classify_batch(
    *,
    batch_index: int,
    concepts: tuple[str, ...],
    api_key: str,
    model_id: str,
    max_tokens: int,
    attempt_index: int,
) -> tuple[BatchClassificationResult, str]:
    """Call Qwen for one batch and return validated labels plus raw text."""

    raw = chat_json_or_text(
        model=model_id,
        api_key=api_key,
        messages=[
            {
                "role": "system",
                "content": "Return compact valid JSON only. Do not explain.",
            },
            {
                "role": "user",
                "content": build_classifier_prompt(
                    concepts,
                    allowed_labels=COMPS.ALLOWED_CONCEPT_TYPES,
                ),
            },
        ],
        temperature=0.0,
        max_tokens=int(max_tokens),
        reasoning_effort="none",
        reasoning_history="disabled",
        timeout_seconds=600,
        max_request_retries=4,
    )
    try:
        result = validate_classification_payload(
            parse_json_object(raw),
            requested_concepts=concepts,
            allowed_labels=COMPS.ALLOWED_CONCEPT_TYPES,
        )
    except ClassificationBatchError as exc:
        raise ClassificationBatchError(
            str(exc),
            retryable=exc.retryable,
            raw_text=raw,
        ) from exc
    return (
        BatchClassificationResult(
            batch_index=batch_index,
            requested_concepts=result.requested_concepts,
            concept_types=result.concept_types,
            uncertain=result.uncertain,
            empty_label_concepts=result.empty_label_concepts,
            max_tokens=int(max_tokens),
            attempt_index=int(attempt_index),
        ),
        raw,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def summarize_values(values: list[str] | tuple[str, ...], *, limit: int = 12) -> str:
    visible = list(values[:limit])
    suffix = "" if len(values) <= limit else f", ... (+{len(values) - limit} more)"
    return "[" + ", ".join(repr(value) for value in visible) + suffix + "]"


def merge_batch_results(
    *,
    batch_paths: tuple[Path, ...],
    expected_concepts: tuple[str, ...],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    concept_types: dict[str, list[str]] = {}
    uncertain: list[dict[str, str]] = []
    empty_label_concepts: list[str] = []
    for batch_path in batch_paths:
        if not batch_path.exists():
            continue
        payload = json.loads(batch_path.read_text(encoding="utf-8"))
        for concept, labels in payload.get("concept_types", {}).items():
            if concept in concept_types:
                raise ValueError(f"duplicate classified concept {concept!r}")
            concept_types[concept] = list(labels)
        uncertain.extend(payload.get("uncertain", []))
        empty_label_concepts.extend(payload.get("empty_label_concepts", []))

    expected = set(expected_concepts)
    observed = set(concept_types)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            "merged concept set mismatch "
            f"missing_count={len(missing)} missing={summarize_values(missing)} "
            f"extra_count={len(extra)} extra={summarize_values(extra)}"
        )

    type_counts = Counter(label for labels in concept_types.values() for label in labels)
    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "qwen_concept_name_classification",
            "batch_size": int(batch_size),
            "concept_count": len(expected_concepts),
            "allowed_labels": list(COMPS.ALLOWED_CONCEPT_TYPES),
            "type_counts": dict(sorted(type_counts.items())),
            "empty_label_count": len(empty_label_concepts),
            "uncertain_count": len(uncertain),
        },
        "concept_types": {concept: concept_types[concept] for concept in expected_concepts},
        "uncertain": uncertain,
        "empty_label_concepts": sorted(set(empty_label_concepts)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comps-path", default=DEFAULT_COMPS_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--annotation-output", default=DEFAULT_ANNOTATION_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--batch-index", type=int, help="Run only one batch index.")
    parser.add_argument("--force", action="store_true", help="Re-run batches with parsed outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Write prompts without Fireworks calls.")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-file", default="fireworks_api.txt")
    parser.add_argument("--model-id", default=COMPS.DEFAULT_MODEL_ID)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--retry-max-tokens", type=int, default=DEFAULT_RETRY_MAX_TOKENS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    comps_path = (ROOT / args.comps_path).resolve() if not Path(args.comps_path).is_absolute() else Path(args.comps_path)
    output_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    annotation_output = (
        ROOT / args.annotation_output
        if not Path(args.annotation_output).is_absolute()
        else Path(args.annotation_output)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    concepts = COMPS.load_comps_acceptable_concepts(comps_path)
    batches = split_batches(concepts, batch_size=int(args.batch_size))
    selected_indices = (
        [int(args.batch_index)]
        if args.batch_index is not None
        else list(range(len(batches)))
    )
    if any(index < 0 or index >= len(batches) for index in selected_indices):
        raise SystemExit(f"batch-index must be between 0 and {len(batches) - 1}")

    if not args.dry_run:
        api_key = load_api_key(api_key=args.api_key, api_key_file=str(ROOT / args.api_key_file))
    else:
        api_key = ""

    parsed_paths: list[Path] = []
    for batch_index, batch in enumerate(batches):
        parsed_paths.append(output_dir / f"batch_{batch_index:03d}_parsed.json")
        if batch_index not in selected_indices:
            continue
        prompt_path = output_dir / f"batch_{batch_index:03d}_prompt.txt"
        prompt_path.write_text(
            build_classifier_prompt(batch, allowed_labels=COMPS.ALLOWED_CONCEPT_TYPES),
            encoding="utf-8",
        )
        parsed_path = output_dir / f"batch_{batch_index:03d}_parsed.json"
        if parsed_path.exists() and not args.force:
            print(f"batch {batch_index:03d}: parsed output exists; skipping")
            continue
        if args.dry_run:
            print(f"batch {batch_index:03d}: wrote prompt only concepts={len(batch)}")
            continue

        last_error: BaseException | None = None
        for attempt_index, max_tokens in enumerate((args.max_tokens, args.retry_max_tokens)):
            raw_path = output_dir / f"batch_{batch_index:03d}_raw_attempt{attempt_index}.txt"
            try:
                result, raw = classify_batch(
                    batch_index=batch_index,
                    concepts=batch,
                    api_key=api_key,
                    model_id=str(args.model_id),
                    max_tokens=int(max_tokens),
                    attempt_index=attempt_index,
                )
                raw_path.write_text(raw, encoding="utf-8")
                write_json(parsed_path, result_to_json(result))
                error_path = output_dir / f"batch_{batch_index:03d}_error.json"
                if error_path.exists():
                    error_path.unlink()
                print(
                    f"batch {batch_index:03d}: ok concepts={len(batch)} "
                    f"max_tokens={max_tokens} attempt={attempt_index}"
                )
                last_error = None
                break
            except ClassificationBatchError as exc:
                last_error = exc
                if exc.raw_text:
                    raw_path.write_text(exc.raw_text, encoding="utf-8")
                if not should_retry_classification_failure(exc):
                    break
                print(f"batch {batch_index:03d}: retryable failure: {exc}")
        if last_error is not None:
            write_json(
                output_dir / f"batch_{batch_index:03d}_error.json",
                {
                    "batch_index": batch_index,
                    "error": str(last_error),
                    "retryable": should_retry_classification_failure(last_error),
                },
            )
            raise SystemExit(f"batch {batch_index:03d} failed: {last_error}")

    if args.dry_run:
        print(f"dry run wrote prompts for {len(selected_indices)} batch(es)")
        return 0

    missing_parsed_batches = [
        index
        for index, parsed_path in enumerate(parsed_paths)
        if not parsed_path.exists()
    ]
    if args.batch_index is not None and missing_parsed_batches:
        print(
            f"batch {int(args.batch_index):03d}: ok; merged annotations deferred until "
            f"all {len(batches)} parsed batches exist "
            f"(missing_batch_count={len(missing_parsed_batches)} "
            f"missing_batches={missing_parsed_batches[:12]}"
            f"{' ...' if len(missing_parsed_batches) > 12 else ''})"
        )
        return 0

    try:
        merged = merge_batch_results(
            batch_paths=tuple(parsed_paths),
            expected_concepts=concepts,
            batch_size=int(args.batch_size),
        )
    except ValueError as exc:
        print(f"not writing merged annotations yet: {exc}")
        return 0

    write_json(annotation_output, merged)
    print(
        f"wrote {annotation_output} concepts={len(merged['concept_types'])} "
        f"uncertain={len(merged['uncertain'])} empty={len(merged['empty_label_concepts'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
