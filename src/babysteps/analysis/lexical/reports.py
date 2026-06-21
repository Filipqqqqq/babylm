"""Report builders and writers for lexical analysis."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from babysteps.analysis.lexical.context import filter_counter_keys
from babysteps.analysis.lexical.stats import EXPOSURE_BUCKETS, TargetStats, exposure_bucket
from babysteps.analysis.lexical.targets import LexicalTarget
from babysteps.analysis.lexical.tokenization import TokenizerAnalyzer


def counter_summary(counter: Counter[str], *, limit: int = 10) -> str:
    return "; ".join(f"{value}:{count}" for value, count in counter.most_common(limit))


def mean(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return float(ordered[index])


def write_csv(rows: list[dict[str, Any]], path: str | Path, *, fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_target_rows(
    targets: list[LexicalTarget],
    stats_by_target: dict[str, TargetStats],
) -> list[dict[str, Any]]:
    rows = []
    for target in targets:
        stats = stats_by_target[target.id]
        mean_subwords_per_occurrence = (
            stats.subword_occurrence_total / stats.total_mentions
            if stats.total_mentions
            else 0.0
        )
        observed_type_fertilities = [
            len(tokenization.split())
            for tokenization in stats.tokenization_counts
            if tokenization
        ]
        rows.append(
            {
                "target_id": target.id,
                "label": target.label,
                "surfaces": ", ".join(target.surfaces),
                "lemmas": ", ".join(target.lemmas),
                "total_mentions": stats.total_mentions,
                "surface_match_mentions": stats.surface_match_mentions,
                "lemma_match_mentions": stats.lemma_match_mentions,
                "document_count": len(stats.doc_ids),
                "exposure_bucket": exposure_bucket(stats.total_mentions),
                "mean_subwords_per_occurrence": f"{mean_subwords_per_occurrence:.4f}",
                "mean_subwords_per_observed_type": f"{mean(observed_type_fertilities):.4f}",
                "top_surfaces": counter_summary(stats.surface_counts),
                "top_lemmas": counter_summary(stats.lemma_counts),
                "top_tokenizations": counter_summary(stats.tokenization_counts, limit=5),
                "top_sources": counter_summary(stats.source_counts),
            }
        )
    return rows


def build_threshold_rows(target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row["exposure_bucket"]) for row in target_rows)
    total = len(target_rows)
    return [
        {
            "bucket": bucket,
            "target_count": counts[bucket],
            "target_rate": f"{(counts[bucket] / total if total else 0.0):.6f}",
        }
        for bucket in EXPOSURE_BUCKETS
    ]


def build_context_summary_rows(
    targets: list[LexicalTarget],
    stats_by_target: dict[str, TargetStats],
    *,
    context_neighbor_stopwords: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    stopwords = context_neighbor_stopwords or frozenset()
    for target in targets:
        stats = stats_by_target[target.id]
        left_content_neighbors = filter_counter_keys(
            stats.left_neighbor_counts,
            stopwords,
        )
        right_content_neighbors = filter_counter_keys(
            stats.right_neighbor_counts,
            stopwords,
        )
        rows.append(
            {
                "target_id": target.id,
                "label": target.label,
                "total_mentions": stats.total_mentions,
                "top_surfaces": counter_summary(stats.surface_counts),
                "top_sources": counter_summary(stats.source_counts),
                "top_left_neighbors": counter_summary(stats.left_neighbor_counts),
                "top_right_neighbors": counter_summary(stats.right_neighbor_counts),
                "top_left_content_neighbors": counter_summary(left_content_neighbors),
                "top_right_content_neighbors": counter_summary(right_content_neighbors),
            }
        )
    return rows


def build_global_word_rows(
    word_counts: Counter[str],
    tokenizer_analyzer: TokenizerAnalyzer,
) -> list[dict[str, Any]]:
    rows = []
    for word, count in sorted(word_counts.items(), key=lambda item: (-item[1], item[0])):
        record = tokenizer_analyzer.analyze(word)
        rows.append(
            {
                "word": word,
                "count": count,
                "fertility": record.fertility,
                "pieces": record.pieces_text,
            }
        )
    return rows


def build_global_summary(
    word_counts: Counter[str],
    tokenizer_analyzer: TokenizerAnalyzer,
    *,
    min_frequency_for_fragmented: int = 1,
    top_n: int = 50,
) -> dict[str, Any]:
    type_fertilities = []
    occurrence_fertility_total = 0
    fragmented_candidates = []
    for word, count in word_counts.items():
        record = tokenizer_analyzer.analyze(word)
        type_fertilities.append(record.fertility)
        occurrence_fertility_total += record.fertility * count
        if count >= min_frequency_for_fragmented and record.fertility > 1:
            fragmented_candidates.append(
                {
                    "word": word,
                    "count": count,
                    "fertility": record.fertility,
                    "pieces": record.pieces_text,
                }
            )

    total_word_tokens = sum(word_counts.values())
    fragmented_candidates.sort(
        key=lambda row: (-int(row["fertility"]), -int(row["count"]), str(row["word"]))
    )
    return {
        "total_word_tokens": total_word_tokens,
        "unique_word_types": len(word_counts),
        "mean_subwords_per_type": mean(type_fertilities),
        "median_subwords_per_type": float(statistics.median(type_fertilities)) if type_fertilities else 0.0,
        "p95_subwords_per_type": percentile(type_fertilities, 0.95),
        "mean_subwords_per_occurrence": (
            occurrence_fertility_total / total_word_tokens if total_word_tokens else 0.0
        ),
        "min_frequency_for_fragmented": int(min_frequency_for_fragmented),
        "top_fragmented_frequent_words": fragmented_candidates[:top_n],
    }


TARGET_COVERAGE_FIELDS = [
    "target_id",
    "label",
    "surfaces",
    "lemmas",
    "total_mentions",
    "surface_match_mentions",
    "lemma_match_mentions",
    "document_count",
    "exposure_bucket",
    "mean_subwords_per_occurrence",
    "mean_subwords_per_observed_type",
    "top_surfaces",
    "top_lemmas",
    "top_tokenizations",
    "top_sources",
]

THRESHOLD_FIELDS = ["bucket", "target_count", "target_rate"]

GLOBAL_WORD_FIELDS = ["word", "count", "fertility", "pieces"]

CONTEXT_SUMMARY_FIELDS = [
    "target_id",
    "label",
    "total_mentions",
    "top_surfaces",
    "top_sources",
    "top_left_neighbors",
    "top_right_neighbors",
    "top_left_content_neighbors",
    "top_right_content_neighbors",
]


__all__ = [
    "CONTEXT_SUMMARY_FIELDS",
    "GLOBAL_WORD_FIELDS",
    "TARGET_COVERAGE_FIELDS",
    "THRESHOLD_FIELDS",
    "build_context_summary_rows",
    "build_global_summary",
    "build_global_word_rows",
    "build_target_rows",
    "build_threshold_rows",
    "write_csv",
    "write_json",
    "write_jsonl",
]
