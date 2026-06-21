"""In-memory aggregate state for lexical analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from babysteps.analysis.lexical.targets import LexicalTarget


@dataclass
class TargetStats:
    """Aggregates collected for one lexical target."""

    target: LexicalTarget
    total_mentions: int = 0
    surface_match_mentions: int = 0
    lemma_match_mentions: int = 0
    subword_occurrence_total: int = 0
    doc_ids: set[int] = field(default_factory=set)
    surface_counts: Counter[str] = field(default_factory=Counter)
    lemma_counts: Counter[str] = field(default_factory=Counter)
    tokenization_counts: Counter[str] = field(default_factory=Counter)
    source_counts: Counter[str] = field(default_factory=Counter)
    left_neighbor_counts: Counter[str] = field(default_factory=Counter)
    right_neighbor_counts: Counter[str] = field(default_factory=Counter)

    def observe(
        self,
        *,
        doc_index: int,
        source: str,
        surface: str,
        lemma: str,
        match_kinds: set[str],
        tokenization: str,
        subword_count: int,
        left_neighbors: list[str],
        right_neighbors: list[str],
    ) -> None:
        self.total_mentions += 1
        if "surface" in match_kinds:
            self.surface_match_mentions += 1
        if "lemma" in match_kinds:
            self.lemma_match_mentions += 1
        self.subword_occurrence_total += int(subword_count)
        self.doc_ids.add(int(doc_index))
        self.surface_counts[surface] += 1
        self.lemma_counts[lemma] += 1
        self.tokenization_counts[tokenization] += 1
        if source:
            self.source_counts[source] += 1
        self.left_neighbor_counts.update(left_neighbors)
        self.right_neighbor_counts.update(right_neighbors)


def exposure_bucket(count: int) -> str:
    """Bucket target mention counts for compact exposure reports."""

    if count == 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 10:
        return "6-10"
    if count <= 50:
        return "11-50"
    if count <= 100:
        return "51-100"
    if count <= 500:
        return "101-500"
    if count <= 1000:
        return "501-1000"
    return "1001+"


EXPOSURE_BUCKETS = ("0", "1-5", "6-10", "11-50", "51-100", "101-500", "501-1000", "1001+")


__all__ = ["EXPOSURE_BUCKETS", "TargetStats", "exposure_bucket"]
